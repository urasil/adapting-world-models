# checks whether v-jepa 2.1 features shift a lot under augmentation, vs. how much they
# naturally vary between different segments.
# for n mistake segments: re-decode the stored frames, encode original + each augmented
# version, compare full-token (32,576,1024, split into centre/edge since crops mostly hit
# the edge) and pooled (32,1024, what the classifier actually trains on) cosine similarity.
# baselines: natural variation between different mistake segments, and mistake-vs-correct
# separation, both from stored pooled features.
# saves results/augmentation_sensitivity.json.
# usage (gpu job, see scripts/test_augmentation_sensitivity.sh): python test_augmentation_sensitivity.py
#   --features_dir ... --video_dir ... --output ... --n_segments 30 --n_baseline 20 --seed 42 [--smoke]

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# local imports
sys.path.insert(0, str(Path(__file__).parent))
from src.vjepa2_1_segment_feature_extraction import (
    load_model,
    _decode_frames,
    T_TOKENS,   # 32
    S_TOKENS,   # 576
    EXPECTED_HIDDEN,  # 1024
)
from src.augmentations import AUG_REGISTRY, slight_crop, warm_tint, combined_aug


# spatial centre/edge mask
# v-jepa 2.1 uses 24×24 spatial patches (384 / 16 = 24).
#
# "centre" = inner 22×22 block (rows 1-22, cols 1-22) = 484 tokens (84 %).
# "edge"   = 1-token-wide border all around             =  92 tokens (16 %).
#
# motivation: a crop_frac=0.90 excludes 5 % of each dimension per side,
# which is 0.05 × 24 = 1.2 tokens/side.  a symmetric 1-token border is the
# smallest integer approximation of that exclusion zone, giving the correct
# directionality: centre tokens >> edge tokens for a mild crop.
# (exact 90/10 is not reachable on a 24×24 grid with an integer border;
#  border=1 gives the closest symmetric split: 84 % centre, 16 % edge.)

_GRID = 24                                              # patches per side
_row  = np.arange(S_TOKENS) // _GRID                   # (576,) row index
_col  = np.arange(S_TOKENS) %  _GRID                   # (576,) col index
CENTER_MASK = ((_row >= 1) & (_row < 23) &
               (_col >= 1) & (_col < 23))              # (576,) bool — inner 22×22
EDGE_MASK   = ~CENTER_MASK


# feature extraction helpers

def run_encoder(processor, model, dtype, frames_tchw: torch.Tensor) -> np.ndarray:
    """
    Run V-JEPA 2.1 on a (T, C, H, W) uint8 clip.
    Returns FULL token features: (T_TOKENS=32, S_TOKENS=576, HIDDEN=1024) float32.
    No spatial pooling — caller decides what to do with spatial dimension.
    """
    out = processor(frames_tchw)[0]
    video = out.unsqueeze(0) if out.ndim == 4 else out

    device = next(model.parameters()).device
    video = video.to(device=device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                feats = model(video)  # (1, t*s, hidden)
        else:
            feats = model(video)

    raw = feats[0].cpu().float().numpy()               # (18432, 1024)
    return raw.reshape(T_TOKENS, S_TOKENS, EXPECTED_HIDDEN).astype(np.float32)
    #                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                  (32, 576, 1024) — full spatial structure preserved


def _pool(feats: np.ndarray) -> np.ndarray:
    """(32, 576, 1024) → (32, 1024) spatial mean-pool."""
    return feats.mean(axis=1)


def _global_vec(feats: np.ndarray) -> np.ndarray:
    """(32, 576, 1024) → (1024,) mean over T and S."""
    return feats.mean(axis=(0, 1))


def _norm_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalise last axis."""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


# full-token metrics

def full_token_cosine_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Compare every spatial token independently.
    a, b: (32, 576, 1024) float32

    Returns:
      mean/std/min — distribution of per-token cosine similarities (18432 values)
      per_t_mean   — (32,) mean cos-sim across 576 spatial tokens per time step
      centre_mean  — mean cos-sim for the inner 22×22 patch block (484 tokens, 84 %)
      edge_mean    — mean cos-sim for the 1-token border patches (92 tokens, 16 %)
    """
    # (32, 576) cosine similarity at every token
    a_n = _norm_rows(a)   # (32, 576, 1024)
    b_n = _norm_rows(b)
    per_token = (a_n * b_n).sum(axis=-1)              # (32, 576)

    flat = per_token.ravel()                           # 18432 values
    return {
        "mean":        float(flat.mean()),
        "std":         float(flat.std()),
        "min":         float(flat.min()),
        "per_t_mean":  per_token.mean(axis=1).tolist(), # (32,) — temporal profile
        "centre_mean": float(per_token[:, CENTER_MASK].mean()),
        "edge_mean":   float(per_token[:, EDGE_MASK].mean()),
    }


def full_global_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine sim between the (1024,) global mean vectors (pool over T and S)."""
    a_g = _norm_rows(_global_vec(a).reshape(1, -1)).ravel()
    b_g = _norm_rows(_global_vec(b).reshape(1, -1)).ravel()
    return float(np.dot(a_g, b_g))


# pooled metrics (matching downstream classifier input)

def pooled_cosine_sim_per_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Spatially pool first, then compare per temporal position.
    a, b: (32, 576, 1024) → pool → (32, 1024) → cosine sim → (32,)
    """
    ap = _norm_rows(_pool(a))   # (32, 1024)
    bp = _norm_rows(_pool(b))
    return (ap * bp).sum(axis=1)


def pooled_global_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine sim between spatially-pooled global vectors. Matches stored .npz format."""
    a_g = _norm_rows(_pool(a).mean(axis=0, keepdims=True)).ravel()
    b_g = _norm_rows(_pool(b).mean(axis=0, keepdims=True)).ravel()
    return float(np.dot(a_g, b_g))


def pooled_l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    """L2 between spatially-pooled global vectors."""
    return float(np.linalg.norm(_pool(a).mean(axis=0) - _pool(b).mean(axis=0)))


# helpers for stored (already-pooled) baseline features

def global_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """For stored (32, 1024) pooled features: mean-pool over T then cosine sim."""
    a_g = a.mean(axis=0); a_g /= np.linalg.norm(a_g) + 1e-8
    b_g = b.mean(axis=0); b_g /= np.linalg.norm(b_g) + 1e-8
    return float(np.dot(a_g, b_g))


def l2_dist_global(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


# data helpers

def find_processable_segments(
    features_dir: Path,
    video_dir: Path,
    label: int,
    max_segments: int,
    rng: random.Random,
) -> List[Dict]:
    """
    Return up to `max_segments` segments with the given label (0=correct, 1=mistake)
    from videos where both a .npz feature file AND the source video exist.

    Each returned dict has:
      video_name, seg_idx (index into npz arrays), label,
      frame_indices (64,), start_sec, end_sec, verb, noun,
      stored_feats (32, 1024) float32  — the already-extracted features
      video_path (Path)
    """
    results = []
    npz_files = sorted(features_dir.glob("*.npz"))
    rng.shuffle(npz_files)

    for npz_path in npz_files:
        video_name = npz_path.stem
        video_path = video_dir / video_name / "Export_py" / "Video_compress.mp4"
        if not video_path.exists():
            continue

        data = np.load(str(npz_path))
        labels = data["labels"]  # (n,) int8

        seg_indices = list(np.where(labels == label)[0])
        rng.shuffle(seg_indices)

        for si in seg_indices:
            results.append({
                "video_name": video_name,
                "seg_idx": int(si),
                "label": int(label),
                "frame_indices": data["sampled_frame_indices"][si].tolist(),
                "start_sec": float(data["start_sec"][si]),
                "end_sec": float(data["end_sec"][si]),
                "verb": str(data["verb"][si]),
                "noun": str(data["noun"][si]),
                "stored_feats": data["features"][si].astype(np.float32),  # (32, 1024)
                "video_path": video_path,
            })
            if len(results) >= max_segments:
                return results

    return results


# main logic

def process_segment(seg: Dict, processor, model, dtype, aug_names: List[str],
                    rng: random.Random) -> Dict:
    """
    For one segment:
      - Re-decode exact original frames
      - Run encoder on originals → full (32, 576, 1024) features
      - For each augmentation, apply → run encoder → compute both families of metrics:
          FULL: compare every spatial token independently
          POOLED: spatial mean-pool first (matches downstream classifier format)
    """
    t0 = time.time()
    frame_idxs = seg["frame_indices"]
    video_path  = str(seg["video_path"])

    # decode frames once
    raw_np = _decode_frames(video_path, frame_idxs)  # (64, h, w, 3) uint8
    frames_orig = torch.from_numpy(raw_np).permute(0, 3, 1, 2)  # (t, c, h, w) uint8

    # --- original pass: full (32, 576, 1024) ---
    orig_feats = run_encoder(processor, model, dtype, frames_orig)  # (32, 576, 1024)

    # sanity check against stored (already-pooled) features
    stored_cos = global_cosine_sim(_pool(orig_feats), seg["stored_feats"])

    out = {
        "video_name": seg["video_name"],
        "seg_idx": seg["seg_idx"],
        "label": seg["label"],
        "start_sec": seg["start_sec"],
        "end_sec": seg["end_sec"],
        "verb": seg["verb"],
        "noun": seg["noun"],
        "duration_sec": seg["end_sec"] - seg["start_sec"],
        "sanity_cos_sim_vs_stored": stored_cos,
        "augmentations": {},
    }

    # --- augmentation passes ---
    for aug_name in aug_names:
        aug_info = AUG_REGISTRY[aug_name]
        kwargs   = dict(aug_info["kwargs"])
        if "rng" in aug_info["fn"].__code__.co_varnames:
            kwargs["rng"] = random.Random(rng.randint(0, 2**31))

        aug_frames = aug_info["fn"](frames_orig, **kwargs)
        aug_feats  = run_encoder(processor, model, dtype, aug_frames)  # (32, 576, 1024)

        # full-token comparison (most honest: every spatial token compared independently)
        ft = full_token_cosine_stats(orig_feats, aug_feats)

        # pooled comparison (matches downstream classifier: spatial pool first)
        pooled_per_t = pooled_cosine_sim_per_t(orig_feats, aug_feats).tolist()

        out["augmentations"][aug_name] = {
            # full-token metrics
            # these compare all 18,432 tokens (32 temporal × 576 spatial) individually
            "full_cos_sim_mean":         ft["mean"],
            "full_cos_sim_std":          ft["std"],
            "full_cos_sim_min":          ft["min"],
            "full_cos_sim_per_t_mean":   ft["per_t_mean"],   # (32,) temporal profile
            "full_cos_sim_centre":       ft["centre_mean"],  # inner 22×22 spatial block (84 %)
            "full_cos_sim_edge":         ft["edge_mean"],    # 1-token border (16 %)
            "full_cos_sim_centre_minus_edge": ft["centre_mean"] - ft["edge_mean"],

            # pooled metrics (matches .npz stored format & downstream model)
            # spatial mean-pool → (32, 1024) first, then compare
            "pooled_cos_sim_global":     pooled_global_cosine_sim(orig_feats, aug_feats),
            "pooled_cos_sim_per_t_mean": float(np.mean(pooled_per_t)),
            "pooled_cos_sim_per_t_std":  float(np.std(pooled_per_t)),
            "pooled_cos_sim_per_t_min":  float(np.min(pooled_per_t)),
            "pooled_l2_dist":            pooled_l2_dist(orig_feats, aug_feats),
            "pooled_per_t":              pooled_per_t,       # (32,) for plotting
        }

    out["elapsed_sec"] = round(time.time() - t0, 2)
    return out


def compute_pairwise_baselines(
    segments: List[Dict], processor, model, dtype,
    n_pairs: int, rng: random.Random, label: int
) -> List[Dict]:
    """
    Compute cosine similarity between pairs of DIFFERENT segments (same label).
    Uses stored features (no re-encoding needed — fast).

    Returns list of {"seg_a", "seg_b", "cos_sim_global", "l2_dist_global"} dicts.
    """
    results = []
    idxs = list(range(len(segments)))

    attempts = 0
    while len(results) < n_pairs and attempts < n_pairs * 5:
        attempts += 1
        i, j = rng.sample(idxs, 2)
        if segments[i]["video_name"] == segments[j]["video_name"] and \
                segments[i]["seg_idx"] == segments[j]["seg_idx"]:
            continue

        a = segments[i]["stored_feats"]
        b = segments[j]["stored_feats"]
        results.append({
            "seg_a": f"{segments[i]['video_name']}[{segments[i]['seg_idx']}]",
            "seg_b": f"{segments[j]['video_name']}[{segments[j]['seg_idx']}]",
            "same_video": segments[i]["video_name"] == segments[j]["video_name"],
            "cos_sim_global": global_cosine_sim(a, b),
            "l2_dist_global": l2_dist_global(a, b),
        })
    return results


def compute_cross_class_baselines(
    mistakes: List[Dict], corrects: List[Dict],
    n_pairs: int, rng: random.Random,
) -> List[Dict]:
    """Cosine similarity between random mistake ↔ correct pairs."""
    results = []
    for _ in range(n_pairs):
        m = rng.choice(mistakes)
        c = rng.choice(corrects)
        a = m["stored_feats"]
        b = c["stored_feats"]
        results.append({
            "mistake_seg": f"{m['video_name']}[{m['seg_idx']}]",
            "correct_seg": f"{c['video_name']}[{c['seg_idx']}]",
            "cos_sim_global": global_cosine_sim(a, b),
            "l2_dist_global": l2_dist_global(a, b),
        })
    return results


# cli

def parse_args():
    p = argparse.ArgumentParser(
        description="Test V-JEPA 2.1 feature sensitivity to mild augmentations."
    )
    p.add_argument("--features_dir", default="features/vjepa2_train",
                   help="Directory with .npz feature files")
    p.add_argument("--video_dir", default="holoassist/videos",
                   help="Root directory for HoloAssist videos")
    p.add_argument("--output", default="results/augmentation_sensitivity.json",
                   help="Path to write JSON results")
    p.add_argument("--n_segments", type=int, default=30,
                   help="Number of mistake segments to augment+encode (default 30)")
    p.add_argument("--n_baseline", type=int, default=20,
                   help="Number of pairwise baseline comparisons per condition (default 20)")
    p.add_argument("--n_correct", type=int, default=30,
                   help="Number of correct segments to load for cross-class baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augmentations", nargs="+",
                   default=list(AUG_REGISTRY.keys()),
                   choices=list(AUG_REGISTRY.keys()),
                   help="Which augmentations to test (default: all three)")
    p.add_argument("--crop_frac", type=float, default=None,
                   help="Override crop_frac for slight_crop and combined augmentations "
                        "(default: use registry value, currently 0.70).  "
                        "E.g. --crop_frac 0.9 for a milder 90%% crop.")
    p.add_argument("--smoke", action="store_true",
                   help="Quick test: 3 segments, 3 baseline pairs, skips combined aug")
    return p.parse_args()


def main():
    args = parse_args()
    rng  = random.Random(args.seed)

    # ---- override crop_frac in registry if requested ----
    if args.crop_frac is not None:
        for aug_name in ("slight_crop", "combined"):
            if aug_name in AUG_REGISTRY:
                AUG_REGISTRY[aug_name]["kwargs"]["crop_frac"] = args.crop_frac
                AUG_REGISTRY[aug_name]["description"] = (
                    AUG_REGISTRY[aug_name]["description"]
                    .replace("70 %", f"{int(args.crop_frac * 100)} %")
                    .replace("(70 %)", f"({int(args.crop_frac * 100)} %)")
                )
        print(f"[crop_frac override] slight_crop and combined now use "
              f"crop_frac={args.crop_frac}")

    if args.smoke:
        args.n_segments  = 3
        args.n_baseline  = 3
        args.n_correct   = 3
        args.augmentations = ["slight_crop", "warm_tint"]
        print("[smoke] Running with 3 segments, 3 baseline pairs, 2 augmentations.")

    features_dir = Path(args.features_dir)
    video_dir    = Path(args.video_dir)
    out_path     = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("V-JEPA 2.1 Augmentation Sensitivity Test")
    print(f"{'='*60}")
    print(f"  features_dir : {features_dir}")
    print(f"  video_dir    : {video_dir}")
    print(f"  n_segments   : {args.n_segments}  (mistakes)")
    print(f"  augmentations: {args.augmentations}")
    print(f"  n_baseline   : {args.n_baseline} pairs per condition")
    print(f"  seed         : {args.seed}")
    print()

    # ---- load model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU detected. Inference will be very slow on CPU.\n"
              "Consider submitting via scripts/test_augmentation_sensitivity.sh")
    processor, model, dtype = load_model(device)

    # ---- gather segments ----
    print(f"Searching for {args.n_segments} mistake segments...")
    mistakes = find_processable_segments(
        features_dir, video_dir, label=1, max_segments=args.n_segments, rng=rng
    )
    print(f"  Found {len(mistakes)} mistake segments across "
          f"{len(set(s['video_name'] for s in mistakes))} videos")

    if len(mistakes) == 0:
        print("ERROR: No processable segments found. Check --features_dir and --video_dir.")
        sys.exit(1)

    print(f"Searching for {args.n_correct} correct segments (for baseline)...")
    corrects = find_processable_segments(
        features_dir, video_dir, label=0, max_segments=args.n_correct, rng=rng
    )
    print(f"  Found {len(corrects)} correct segments")
    print()

    # ---- process each mistake segment ----
    segment_results = []
    for i, seg in enumerate(mistakes):
        dur  = seg['end_sec'] - seg['start_sec']
        desc = f"{seg['video_name']} seg[{seg['seg_idx']}] " \
               f"({seg['verb']} {seg['noun']}, {dur:.1f}s)"
        print(f"[{i+1}/{len(mistakes)}] {desc}", flush=True)

        try:
            result = process_segment(seg, processor, model, dtype, args.augmentations, rng)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        # print quick summary line
        aug_summaries = "  ".join(
            f"{k}: full={v['full_cos_sim_mean']:.4f} pooled={v['pooled_cos_sim_global']:.4f} "
            f"(ctr={v['full_cos_sim_centre']:.4f} edge={v['full_cos_sim_edge']:.4f})"
            for k, v in result["augmentations"].items()
        )
        print(f"  sanity={result['sanity_cos_sim_vs_stored']:.4f}  "
              f"{aug_summaries}  [{result['elapsed_sec']:.1f}s]")

        segment_results.append(result)

    # ---- baselines (cpu, uses stored features) ----
    print(f"\nComputing pairwise baselines ({args.n_baseline} pairs each)...")
    mistake_pairs = compute_pairwise_baselines(
        mistakes, processor, model, dtype, args.n_baseline, rng, label=1
    )
    correct_pairs = compute_pairwise_baselines(
        corrects, processor, model, dtype, args.n_baseline, rng, label=0
    )
    cross_pairs = compute_cross_class_baselines(mistakes, corrects, args.n_baseline, rng)

    # ---- aggregate summary ----
    summary = {}
    for aug_name in args.augmentations:
        segs_with_aug = [s for s in segment_results if aug_name in s.get("augmentations", {})]
        if not segs_with_aug:
            continue

        def _agg(key):
            vals = [s["augmentations"][aug_name][key] for s in segs_with_aug]
            return {"n": len(vals), "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)), "min": float(np.min(vals)),
                    "max": float(np.max(vals))}

        summary[aug_name] = {
            "full":   _agg("full_cos_sim_mean"),
            "centre": _agg("full_cos_sim_centre"),
            "edge":   _agg("full_cos_sim_edge"),
            "centre_minus_edge": _agg("full_cos_sim_centre_minus_edge"),
            "pooled": _agg("pooled_cos_sim_global"),
        }

    def _pair_summary(pairs, key="cos_sim_global"):
        vals = [p[key] for p in pairs]
        return {
            "n": len(vals),
            "mean": float(np.mean(vals)) if vals else None,
            "std":  float(np.std(vals))  if vals else None,
            "min":  float(np.min(vals))  if vals else None,
            "max":  float(np.max(vals))  if vals else None,
        } if vals else {}

    summary["_baseline_mistake_pairs"] = _pair_summary(mistake_pairs)
    summary["_baseline_correct_pairs"] = _pair_summary(correct_pairs)
    summary["_baseline_cross_class"]   = _pair_summary(cross_pairs)

    # ---- save ----
    output = {
        "config": {
            "n_segments":   len(segment_results),
            "n_baseline":   args.n_baseline,
            "augmentations": args.augmentations,
            "aug_descriptions": {k: AUG_REGISTRY[k]["description"] for k in args.augmentations},
            "aug_kwargs": {k: {
                kk: str(vv) for kk, vv in AUG_REGISTRY[k]["kwargs"].items()
            } for k in args.augmentations},
            "features_dir": str(features_dir),
            "video_dir":    str(video_dir),
            "seed":         args.seed,
            "device":       device,
        },
        "segments":       segment_results,
        "baselines": {
            "mistake_pairs": mistake_pairs,
            "correct_pairs": correct_pairs,
            "cross_class":   cross_pairs,
        },
        "summary": summary,
    }

    # make json serialisable (remove stored_feats numpy arrays)
    for seg in output["segments"]:
        seg.pop("stored_feats", None)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {out_path}")
    print("\nRun  python analyse_augmentation_sensitivity.py --results", out_path,
          " to see the full report.")


if __name__ == "__main__":
    main()
