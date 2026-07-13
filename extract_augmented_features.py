# re-extracts v-jepa 2.1 features for every segment (correct + mistake) in a video after
# applying one augmentation to the decoded frames.
# augments ALL segments, not just mistakes, so the model can't learn "augmented style = mistake"
# from an un-augmented prefix history.
# writes output_dir/<video>_aug_<aug_name>.npz - same keys/shapes as the originals, drop-in
# compatible; HoloMistakeDataset resolves this virtual video name across feature dirs.
# usage (one gpu job per aug): python extract_augmented_features.py --features_dir ... --video_dir ...
#   --output_dir ... --aug_name warm_tint --seed 42
# run once per aug: warm_tint, cold_tint, crop_90, warm_crop (see scripts/extract_augmented_features.sh)

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.vjepa2_1_segment_feature_extraction import (
    load_model,
    _decode_frames,
    T_TOKENS,        # 32  (= num_frames // tubelet_size)
    S_TOKENS,        # 576 (= (384/16)^2 spatial patches)
    EXPECTED_HIDDEN, # 1024
)
from src.augmentations import AUG_REGISTRY, TRAIN_AUG_NAMES


# feature extraction helper

def extract_and_pool(
    processor,
    model,
    dtype,
    frames_tchw: torch.Tensor,
) -> np.ndarray:
    """
    Run V-JEPA 2.1 on a (T, C, H, W) uint8 clip and return spatially-pooled
    features shaped (T_TOKENS=32, EXPECTED_HIDDEN=1024) as float16.

    This matches the format stored in the original .npz feature files
    (spatial mean-pool of the 576 patch tokens → kept only temporal dimension).
    """
    out = processor(frames_tchw)[0]
    video = out.unsqueeze(0) if out.ndim == 4 else out

    device = next(model.parameters()).device
    video = video.to(device=device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                feats = model(video)   # (1, t_tokens*s_tokens, hidden)
        else:
            feats = model(video)

    raw = feats[0].cpu().to(torch.float16).numpy()  # (18432, 1024)
    # spatial mean-pool:  (t*s, d) → reshape → (t, s, d) → mean(s) → (t, d)
    return (
        raw.reshape(T_TOKENS, S_TOKENS, EXPECTED_HIDDEN)
           .mean(axis=1)                                  # (32, 1024)
           .astype(np.float16)
    )


# per-video processing

def process_video(
    npz_path: Path,
    video_path: Path,
    aug_fn,
    aug_kwargs: dict,
    processor,
    model,
    dtype,
    base_rng_seed: int,
) -> dict | None:
    """
    For one video:
      1. Load stored frame indices and metadata from the original .npz.
      2. For each segment: decode frames → apply augmentation → run encoder
         → spatial-pool.
      3. Return a dict with the stacked augmented features + all metadata
         fields needed to write a new .npz.

    Returns None if a fatal error occurs (e.g. corrupt video).
    The function uses per-segment deterministic seeds derived from
    base_rng_seed so repeated runs produce identical augmentations.
    """
    import random as _random

    try:
        npz = np.load(str(npz_path), allow_pickle=True)
    except Exception as e:
        print(f"  could not load npz {npz_path}: {e}")
        return None

    frame_idxs_all = npz["sampled_frame_indices"]   # (n_segs, 64) int32
    n_segs = len(frame_idxs_all)

    augmented_features: list[np.ndarray] = []

    for si in range(n_segs):
        idxs = frame_idxs_all[si].tolist()

        try:
            raw_np = _decode_frames(str(video_path), idxs)   # (64, h, w, 3) uint8
        except Exception as e:
            print(f"\n  seg {si}/{n_segs}: frame decode failed: {e}")
            return None

        frames_tchw = torch.from_numpy(raw_np).permute(0, 3, 1, 2)  # (t, c, h, w) uint8

        # deterministic per-segment seed for stochastic spatial crops.
        kwargs = dict(aug_kwargs)
        import inspect
        if "rng" in inspect.signature(aug_fn).parameters:
            kwargs["rng"] = _random.Random(base_rng_seed + si)

        aug_frames = aug_fn(frames_tchw, **kwargs)
        pooled = extract_and_pool(processor, model, dtype, aug_frames)  # (32, 1024)
        augmented_features.append(pooled)

    return {
        "features":              np.stack(augmented_features, axis=0),  # (n, 32, 1024) f16
        "segment_ids":           npz["segment_ids"],
        "labels":                npz["labels"],
        "start_sec":             npz["start_sec"],
        "end_sec":               npz["end_sec"],
        "verb":                  npz["verb"],
        "noun":                  npz["noun"],
        "correctness_raw":       npz["correctness_raw"],
        "sampled_frame_indices": npz["sampled_frame_indices"],
        "fps":                   npz["fps"],
    }


# cli

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-extract V-JEPA 2.1 features (spatially pooled) after augmentation."
    )
    p.add_argument(
        "--features_dir", required=True,
        help="Directory containing the original per-video .npz feature files "
             "(e.g. features/vjepa2_train).",
    )
    p.add_argument(
        "--video_dir", required=True,
        help="Root directory for HoloAssist videos.  Each video is at "
             "<video_dir>/<video_name>/Export_py/Video_compress.mp4.",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Directory for augmented .npz output files "
             "(e.g. features/vjepa2_train_aug).  Created if absent.",
    )
    p.add_argument(
        "--aug_name", required=True,
        choices=list(AUG_REGISTRY.keys()),
        help=f"Augmentation to apply.  Training variants: {TRAIN_AUG_NAMES}",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for stochastic augmentations (e.g. crop jitter). "
             "Each segment gets a deterministic child seed: seed*10000 + video_idx*100 + seg_idx.",
    )
    p.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip videos whose output .npz already exists (default: on).",
    )
    p.add_argument(
        "--no_resume", dest="resume", action="store_false",
        help="Overwrite existing output files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    features_dir = Path(args.features_dir)
    video_dir    = Path(args.video_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aug_info   = AUG_REGISTRY[args.aug_name]
    aug_fn     = aug_info["fn"]
    aug_kwargs = dict(aug_info["kwargs"])

    print(f"\n{'='*60}")
    print("Augmented V-JEPA 2.1 Feature Extraction")
    print(f"{'='*60}")
    print(f"  aug_name     : {args.aug_name}")
    print(f"  description  : {aug_info['description']}")
    print(f"  kwargs       : {aug_kwargs}")
    print(f"  features_dir : {features_dir}")
    print(f"  video_dir    : {video_dir}")
    print(f"  output_dir   : {output_dir}")
    print(f"  seed         : {args.seed}")
    print(f"  resume       : {args.resume}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU detected.  Inference will be very slow on CPU.\n"
              "Consider submitting via scripts/extract_augmented_features.sh\n")
    processor, model, dtype = load_model(device)

    npz_files = sorted(features_dir.glob("*.npz"))
    n_total   = len(npz_files)
    print(f"Found {n_total} .npz files in {features_dir}\n")

    n_done = n_skipped = n_failed = 0
    t0 = time.time()

    for vi, npz_path in enumerate(npz_files):
        video_name = npz_path.stem
        aug_video_name = f"{video_name}_aug_{args.aug_name}"
        out_path = output_dir / f"{aug_video_name}.npz"

        if args.resume and out_path.exists():
            n_skipped += 1
            continue

        video_path = video_dir / video_name / "Export_py" / "Video_compress.mp4"
        if not video_path.exists():
            print(f"[{vi+1}/{n_total}] {video_name}: video file not found, skipping")
            n_failed += 1
            continue

        # quick peek at segment count without loading features array
        with np.load(str(npz_path), allow_pickle=True) as _peek:
            n_segs = int(_peek["labels"].shape[0])

        print(f"[{vi+1}/{n_total}] {video_name} ({n_segs} segs) ...",
              end=" ", flush=True)
        t_v = time.time()

        # deterministic base seed that varies per video
        base_rng_seed = args.seed * 10_000 + vi * 100

        result = process_video(
            npz_path    = npz_path,
            video_path  = video_path,
            aug_fn      = aug_fn,
            aug_kwargs  = aug_kwargs,
            processor   = processor,
            model       = model,
            dtype       = dtype,
            base_rng_seed = base_rng_seed,
        )

        if result is None:
            print("FAILED")
            n_failed += 1
            continue

        np.savez_compressed(
            out_path,
            segment_ids           = result["segment_ids"],
            features              = result["features"],       # (n, 32, 1024) float16
            labels                = result["labels"],
            start_sec             = result["start_sec"],
            end_sec               = result["end_sec"],
            verb                  = result["verb"],
            noun                  = result["noun"],
            correctness_raw       = result["correctness_raw"],
            sampled_frame_indices = result["sampled_frame_indices"],
            fps                   = result["fps"],
            video_name            = np.array(aug_video_name),
            model_id              = np.array("vjepa2_1_vit_large_384"),
            checkpoint            = np.array("vjepa2_1_vitl_dist_vitG_384.pt"),
            pooling               = np.array("spatial_mean"),
            augmentation          = np.array(args.aug_name),
        )

        elapsed = time.time() - t_v
        n_processed_so_far = n_done + 1
        total_elapsed = time.time() - t0
        rate = n_processed_so_far / total_elapsed           # videos/sec
        remaining = n_total - vi - 1 - n_skipped           # rough
        eta_min = (remaining / max(rate, 1e-6)) / 60.0

        mistakes = int(result["labels"].sum())
        print(f"done  ({elapsed:.1f}s, {n_segs} segs, {mistakes} mistakes, "
              f"ETA ~{eta_min:.0f} min)")

        n_done += 1

    print(f"\n{'='*60}")
    print(f"Extraction complete")
    print(f"  processed  : {n_done}")
    print(f"  skipped    : {n_skipped}  (already existed)")
    print(f"  failed     : {n_failed}")
    print(f"  output_dir : {output_dir}")
    total_min = (time.time() - t0) / 60.0
    print(f"  wall time  : {total_min:.1f} min")


if __name__ == "__main__":
    main()
