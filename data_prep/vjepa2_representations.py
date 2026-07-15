"""
# v-jepa-2 expects tensor txcxhxw (num_frames, channels, height, width).
# this version extracts one feature per holoassist fine-grained action segment,
# rather than per fixed-stride window, to match the holoassist mistake-detection
# benchmark (per-fine-grained-action binary classification).

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import torch
# this is not installed as I have switched to av later. this was just for trial before
# the switch to V-JEPA 2.1
# from torchcodec.decoders import VideoDecoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.holoassist_labels import LABEL_MAP
from src.utils.vjepa2_clip import MODEL_ID, NUM_FRAMES, load_model, extract_one_clip, sample_frame_indices

def extract_segments_for_video(video_path: str, annotation_entry: dict, processor, model, test: bool = False) -> Dict[int, dict]:
    # one V-JEPA 2 feature per fine-grained action segment -> {segment_id: {features, label, start, end, verb, noun, correctness_raw, sampled_frame_indices, fps}}
    decoder = VideoDecoder(video_path)
    fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames

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
            frames = decoder.get_frames_at(indices=idxs).data
        except Exception as e:
            print(f"  segment {ev['id']} frame fetch failed: {e}")
            continue

        features = extract_one_clip(processor, model, frames)
        results[ev["id"]] = {
            "features": features,
            "label": LABEL_MAP[correctness],
            "start": float(ev["start"]),
            "end": float(ev["end"]),
            "verb": attrs.get("Verb", ""),
            "noun": attrs.get("Noun", ""),
            "correctness_raw": correctness,
            "sampled_frame_indices": np.array(idxs, dtype=np.int32),
            "fps": float(fps),
        }

        if test and len(results) >= 3: 
            print(f" --test: stopping after {len(results)} segments")
            break

        if (i + 1) % 50 == 0:
            print(f"  segment {i + 1}/{len(segments)}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Extract V-JEPA-2 representations per HoloAssist fine-grained action segment.")
    parser.add_argument("--annotation_path", type=str, required=True, help="Path to data-annotation-trainval-v1_1.json")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing HoloAssist videos. "
        "Each video is found at <video_dir>/<video_name>/<some>.mp4 - adjust find_video_path() to your layout.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save per-video feature files (one .npz per video).")
    parser.add_argument("--test", action="store_true", help="Stop after 5 segments (for quick CPU smoke-testing).")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.annotation_path) as f:
        annotations = json.load(f)
    print(f"Processing {len(annotations)} videos")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_model(device)

    for vi, entry in enumerate(annotations):
        video_name = entry["video_name"]
        out_path = Path(args.output_dir) / f"{video_name}.npz"
        if out_path.exists():
            print(f"[{vi+1}/{len(annotations)}] {video_name}: already done, skipping")
            continue

        # ---- adapt this to wherever your videos live ----
        # holoassist's release uses a per-session folder containing several files.
        # common layouts: <video_dir>/<video_name>/export_py/video_compress.mp4
        # or                <video_dir>/<video_name>.mp4
        # pick whichever matches your download.
        video_path = Path(args.video_dir) / video_name / "Export_py" / "Video_compress.mp4"
        video_path = video_path if video_path.exists() else None
        if video_path is None:
            print(f"[{vi+1}/{len(annotations)}] {video_name}: video file not found, skipping")
            continue

        print(f"[{vi+1}/{len(annotations)}] {video_name}")
        try:
            seg_results = extract_segments_for_video(video_path, entry, processor, model, test=args.test)
        except Exception as e:
            print(f"  failed: {e}")
            continue

        if not seg_results:
            print(f"  no usable segments")
            continue

        # pack into arrays for compact storage. all segments share the same
        # (num_tokens, hidden_dim) shape, so we can stack.
        seg_ids = sorted(seg_results.keys())
        feats = np.stack([seg_results[s]["features"] for s in seg_ids], axis=0).astype(np.float16)
        labels = np.array([seg_results[s]["label"] for s in seg_ids], dtype=np.int8)
        starts = np.array([seg_results[s]["start"] for s in seg_ids], dtype=np.float32)
        ends = np.array([seg_results[s]["end"] for s in seg_ids], dtype=np.float32)
        verbs = np.array([seg_results[s]["verb"] for s in seg_ids])
        nouns = np.array([seg_results[s]["noun"] for s in seg_ids])
        raw = np.array([seg_results[s]["correctness_raw"] for s in seg_ids])
        frame_idxs = np.stack([seg_results[s]["sampled_frame_indices"] for s in seg_ids], axis=0)
        # fps is constant per video, store as scalar
        fps_val = np.float32(next(iter(seg_results.values()))["fps"])

        np.savez_compressed(
            out_path,
            segment_ids=np.array(seg_ids, dtype=np.int32),
            features=feats,          # (n, 2048, 1024) float16 - all spacetime tokens
            labels=labels,
            start_sec=starts,
            end_sec=ends,
            verb=verbs,
            noun=nouns,
            correctness_raw=raw,
            sampled_frame_indices=frame_idxs,  # (n, 64) int32 - exact frames used
            fps=fps_val,
            video_name=np.array(video_name),
            model_id=np.array(MODEL_ID),
        )
        print(f"  saved {len(seg_ids)} segments -> {out_path}  (features {feats.shape}, mistakes {labels.sum()})")

if __name__ == "__main__":
    main()
"""