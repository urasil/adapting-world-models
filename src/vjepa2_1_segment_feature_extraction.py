# V-JEPA 2.1 expects tensor TxCxHxW (num_frames, channels, height, width).
# Extracts ONE feature per HoloAssist fine-grained action segment: 18432
# tokens (32 temporal x 576 spatial), kept un-pooled. Pool them down to
# (32, 1024) afterwards with data_prep/convert_features_spatialpool.py if
# a run needs the smaller/faster representation.
#
# Writes two files per video into --output_dir:
#   <video_name>.npy       - features only, (num_segments, 32, 576, 1024)
#                             float16. Plain (uncompressed) .npy so training
#                             can mmap straight into it instead of
#                             decompressing a multi-GB archive per read.
#   <video_name>_meta.npz  - small per-segment metadata (segment_ids, labels,
#                             verb, noun, start_sec, end_sec), read by
#                             data_prep/build_mistake_detection_dataset.py and
#                             data_prep/enrich_index_with_actions.py. Kept
#                             separate from the features so it stays tiny and
#                             fast to load even though the features array is
#                             huge.
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import av

LOCAL_CKPT = "vjepa2_1_vitl_dist_vitG_384.pt"
HUB_NAME   = "vjepa2_1_vit_large_384"
RESOLUTION = 384
NUM_FRAMES = 64

# 32 temporal positions (64 frames / tubelet 2) * 24*24 spatial patches (384/16)^2 = 18,432 tokens
T_TOKENS = NUM_FRAMES // 2                 # 32
S_TOKENS = (RESOLUTION // 16) ** 2         # 576
EXPECTED_TOKENS = T_TOKENS * S_TOKENS      # 18432
EXPECTED_HIDDEN = 1024

# Map HoloAssist correctness strings to a binary label
LABEL_MAP = {
    "Correct Action": 0,
    "Wrong Action, corrected by instructor verbally": 1,
    "Wrong Action, corrected by student": 1,
    "Wrong Action, not corrected": 1,
    # "otherwise": filtered out
}


def load_model(device: str):
    print(f"Loading {HUB_NAME} from local checkpoint on {device}")

    # Build architecture via PyTorch Hub with pretrained=False, then load local ckpt
    result = torch.hub.load("facebookresearch/vjepa2", HUB_NAME, pretrained=False)
    model = result[0] if isinstance(result, tuple) else result  # encoder = result[0]

    ckpt = torch.load(LOCAL_CKPT, map_location="cpu", weights_only=True)
    print(f"  ckpt top-level keys: {list(ckpt.keys())}")

    # For the distilled V-JEPA 2.1 models, the EMA copy of the student encoder is
    # the intended evaluation model (per the paper). Fall back to "encoder" for
    # non-distilled variants.
    state = ckpt.get("ema_encoder", ckpt.get("encoder", ckpt))
    state = {k.replace("module.", "").replace("backbone.", ""): v
             for k, v in state.items()}
    src_key = "ema_encoder" if "ema_encoder" in ckpt else (
              "encoder" if "encoder" in ckpt else "<root>")
    print(f"  using checkpoint sub-dict: '{src_key}'")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    processor = torch.hub.load(
        "facebookresearch/vjepa2", "vjepa2_preprocessor", crop_size=RESOLUTION
    )
    model = model.to(device=device).eval()
    autocast_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"  model on {device} (weights fp32, autocast={autocast_dtype})")
    return processor, model, autocast_dtype


def spatial_pool(feats_18432x1024: np.ndarray) -> np.ndarray:
    # collapses spatial tokens, keeps temporal structure: (T*S, hidden) -> (T, S, hidden) -> mean over S -> (T, hidden) i.e. (32, 1024)
    n, h = feats_18432x1024.shape
    assert n == EXPECTED_TOKENS, f"expected {EXPECTED_TOKENS} tokens, got {n}"
    assert h == EXPECTED_HIDDEN, f"expected hidden {EXPECTED_HIDDEN}, got {h}"
    # ViT-style flatten order in V-JEPA 2 is (T_tok, S_tok), so reshape on the left
    return feats_18432x1024.reshape(T_TOKENS, S_TOKENS, h).mean(axis=1)


def extract_one_clip(processor, model, dtype, frames_tchw: torch.Tensor) -> np.ndarray:
    # runs one (NUM_FRAMES, 3, H, W) uint8 clip through the V-JEPA 2.1 encoder, returns spatially-pooled (32, 1024) float16
    # Preprocessor expects (T, C, H, W) uint8; returns a tuple/list whose [0]
    # is the resized/normalised tensor.
    out = processor(frames_tchw)[0]

    # Add batch dim if needed, move to device + dtype
    if out.ndim == 4:           # (C, T, H, W) or (T, C, H, W) — needs batch dim
        video = out.unsqueeze(0)
    elif out.ndim == 5:         # already (B, ...) — preprocessor handled it
        video = out
    else:
        raise RuntimeError(f"unexpected processor output dims: {out.ndim}")
    
    device = next(model.parameters()).device
    video = video.to(device=device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                feats = model(video)  # (1, 18432, 1024) # autocast picks fp16/bf16 ops where safe
        else:
            feats = model(video)

    feats_np = feats[0].cpu().to(torch.float16).numpy()  # (18432, 1024)
    return feats_np.reshape(T_TOKENS, S_TOKENS, EXPECTED_HIDDEN)  # (32, 576, 1024)


def sample_frame_indices(start_frame: int, end_frame: int, num_frames: int) -> List[int]:
    # Uniform sampling: the same recipe the V-JEPA 2 paper / HF processor docs use
    # for variable length clips.
    if end_frame <= start_frame:
        end_frame = start_frame + 1  # avoid degenerate zero-length spans
    idxs = np.linspace(start_frame, end_frame, num=num_frames).round().astype(int)
    return idxs.tolist()


def _decode_frames(video_path: str, indices: List[int]) -> np.ndarray:
    # Decode specific frame indices via PyAV. Returns (N, H, W, C) uint8
    unique = sorted(set(indices))
    needed = set(unique)
    frames_dict: dict = {}

    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    tb = float(stream.time_base)

    seek_ts = max(0, int(unique[0] / fps / tb) - 1)
    container.seek(seek_ts, stream=stream, backward=True)

    for frame in container.decode(stream):
        fi = round(float(frame.pts) * tb * fps)
        if fi in needed:
            frames_dict[fi] = frame.to_ndarray(format='rgb24')
            needed.discard(fi)
        if not needed or fi > unique[-1] + 5:
            break

    container.close()

    result = []
    for i in indices:
        if i in frames_dict:
            result.append(frames_dict[i])
        elif frames_dict:
            nearest = min(frames_dict, key=lambda x: abs(x - i))
            result.append(frames_dict[nearest])
        else:
            raise RuntimeError(f"No frames decoded near index {i} in {video_path}")
    return np.stack(result)


def extract_segments_for_video(video_path: str, annotation_entry: dict, processor, model, dtype, test: bool = False) -> Dict[int, dict]:
    # Extract one V-JEPA 2.1 feature per fine-grained action segment in this video
    with av.open(str(video_path)) as _c:
        _s = _c.streams.video[0]
        fps = float(_s.average_rate)
        total_frames = _s.frames or int(_s.duration * float(_s.time_base) * fps)

    results = {}
    segments = [ev for ev in annotation_entry["events"] if ev["label"] == "Fine grained action"]

    for i, ev in enumerate(segments):
        attrs = ev.get("attributes", {})
        correctness = attrs.get("Action Correctness")
        if correctness not in LABEL_MAP:
            continue

        start_frame = int(round(ev["start"] * fps))
        end_frame = int(round(ev["end"] * fps))
        # clamp to video bounds
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame, min(end_frame, total_frames - 1))

        idxs = sample_frame_indices(start_frame, end_frame, NUM_FRAMES)
        try:
            frames = torch.from_numpy(_decode_frames(str(video_path), idxs)).permute(0, 3, 1, 2)
        except Exception as e:
            print(f"  segment {ev['id']} frame fetch failed: {e}")
            continue

        features = extract_one_clip(processor, model, dtype, frames)  # (32, 1024)
        results[ev["id"]] = {
            "features": features,  # (32, 1024) float16, spatially pooled
            "label": LABEL_MAP[correctness],
            "start": float(ev["start"]),
            "end": float(ev["end"]),
            "verb": attrs.get("Verb", ""),
            "noun": attrs.get("Noun", ""),
            "correctness_raw": correctness,
            "sampled_frame_indices": np.array(idxs, dtype=np.int32),
            "fps": float(fps)
        }

        if test and len(results) >= 3:
            print(f" --test: stopping after {len(results)} segments")
            break

        if (i + 1) % 50 == 0:
            print(f"  segment {i + 1}/{len(segments)}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract V-JEPA 2.1 representations per HoloAssist fine-grained action segment, "
                    "spatially pooled to (32, 1024) per segment."
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        required=True,
        help="Path to data-annotation-trainval-v1_1.json",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing HoloAssist videos. Each video is found at "
             "<video_dir>/<video_name>/Export_py/Video_compress.mp4.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save per-video feature files (one .npy + one "
             "_meta.npz per video).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Stop after 3 segments per video (for quick CPU smoke-testing).",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Optional path to a split file. If not present, all videos in the annotation file are processed."
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.annotation_path) as f:
        annotations = json.load(f)

    if args.split_file is not None:
        with open(args.split_file) as f:
            split_names = {line.strip() for line in f if line.strip()}
        before = len(annotations)
        annotations = [e for e in annotations if e["video_name"] in split_names]
        print(f"Filtered annotations from {before} -> {len(annotations)} videos "
              f"using split file {args.split_file} ({len(split_names)}) names listed")
    print(f"Processing {len(annotations)} videos")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model, dtype = load_model(device)

    for vi, entry in enumerate(annotations):
        video_name = entry["video_name"]
        out_path = Path(args.output_dir) / f"{video_name}.npy"
        meta_path = Path(args.output_dir) / f"{video_name}_meta.npz"
        if out_path.exists() and meta_path.exists():
            print(f"[{vi+1}/{len(annotations)}] {video_name}: already done, skipping")
            continue
        video_path = Path(args.video_dir) / video_name / "Export_py" / "Video_compress.mp4"
        video_path = video_path if video_path.exists() else None
        if video_path is None:
            print(f"[{vi+1}/{len(annotations)}] {video_name}: video file not found, skipping")
            continue

        print(f"[{vi+1}/{len(annotations)}] {video_name}")
        try:
            seg_results = extract_segments_for_video(
                video_path, entry, processor, model, dtype, test=args.test
            )
        except Exception as e:
            print(f"  failed: {e}")
            continue

        if not seg_results:
            print(f"  no usable segments")
            continue

        # Pack into one array for compact storage. All segments share the same
        # (T_tokens, hidden_dim) = (32, 1024) shape, so we can stack.
        seg_ids = sorted(seg_results.keys())
        feats = np.stack([seg_results[s]["features"] for s in seg_ids], axis=0).astype(np.float16)
        labels = np.array([seg_results[s]["label"] for s in seg_ids], dtype=np.int8)
        starts = np.array([seg_results[s]["start"] for s in seg_ids], dtype=np.float64)
        ends = np.array([seg_results[s]["end"] for s in seg_ids], dtype=np.float64)
        verbs = np.array([seg_results[s]["verb"] for s in seg_ids], dtype=object)
        nouns = np.array([seg_results[s]["noun"] for s in seg_ids], dtype=object)

        np.save(out_path, feats)
        np.savez(
            meta_path,
            segment_ids=np.array(seg_ids, dtype=np.int64),
            labels=labels,
            start_sec=starts,
            end_sec=ends,
            verb=verbs,
            noun=nouns,
        )
        print(f"saved {len(seg_ids)} segments -> {out_path} "
              f"(features {feats.shape}, mistakes {labels.sum()})")


if __name__ == "__main__":
    main()
