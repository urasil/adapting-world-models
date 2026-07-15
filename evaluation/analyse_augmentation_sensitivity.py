# cpu-only analysis of the json produced by evaluation/test_augmentation_sensitivity.py.
# prints mean cos-sim between original/augmented features vs natural in/between-class

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# formatting helpers

def fmt_stat(d: dict, key: str = "cos_sim_global") -> str:
    # formats mean +/- std (min-max) from a summary dict
    if not d or "mean" not in d or d["mean"] is None:
        return "n/a"
    return f"{d['mean']:.4f} ± {d['std']:.4f}  (min={d['min']:.4f}, max={d['max']:.4f}, n={d['n']})"

VERDICT_THRESHOLDS = {
    # (lower_bound_relative_to_interclass, verdict, colour_hint)
    0.95: ("VERY SAFE",
           "Augmented features are nearly identical to originals. "
           "Much higher similarity than natural inter-segment variation. "
           "Augmentation is valid for data expansion."),
    0.90: ("SAFE",
           "Augmented features are very close to originals. "
           "Higher similarity than natural variation → safe to augment."),
    0.80: ("MODERATE",
           "Augmented features are noticeably shifted but still in the same "
           "neighbourhood. Augmentation may be valid; monitor classification accuracy."),
    0.0:  ("RISKY",
           "Augmented features deviate substantially from originals, "
           "approaching natural inter-segment variation. "
           "Augmentation may distort the data distribution."),
}

def interpret(aug_mean: float, natural_mean: float) -> str:
    # chooses a verdict based on aug_mean cos-sim
    # we compare aug_mean directly (cos_sim), not relative to natural_mean,
    # because aug should be > natural variation to be safe.
    for threshold, (verdict, explanation) in sorted(VERDICT_THRESHOLDS.items(), reverse=True):
        if aug_mean >= threshold:
            return f"{verdict}\n    {explanation}"
    return "RISKY\n    Features are drastically different."

def _collect_aug_sims(segments: List[Dict], aug_name: str) -> List[float]:
    return [s["augmentations"][aug_name]["cos_sim_global"] for s in segments if aug_name in s.get("augmentations", {})]

def _collect_aug_per_t_sims(segments: List[Dict], aug_name: str) -> List[float]:
    # all per-temporal-position cosine similarities, flattened
    out = []
    for s in segments:
        if aug_name in s.get("augmentations", {}):
            out.extend(s["augmentations"][aug_name]["per_t"])
    return out

# main report

def print_report(data: dict) -> None:
    cfg      = data["config"]
    segs     = data["segments"]
    baselines = data["baselines"]
    summary  = data["summary"]

    aug_names = cfg["augmentations"]

    print()
    print("=" * 70)
    print("  V-JEPA 2.1 AUGMENTATION SENSITIVITY REPORT")
    print("=" * 70)
    print(f"  Segments tested  : {cfg['n_segments']} mistake segments")
    print(f"  Features dir     : {cfg['features_dir']}")
    print(f"  Device           : {cfg['device']}")
    print(f"  Augmentations    : {', '.join(aug_names)}")
    print()

    # ---- sanity check ----
    sanity_sims = [s.get("sanity_cos_sim_vs_stored") for s in segs
                   if s.get("sanity_cos_sim_vs_stored") is not None]
    if sanity_sims:
        print(f"Sanity check (re-encoded vs stored features):")
        print(f"  mean cos-sim = {np.mean(sanity_sims):.4f} ± {np.std(sanity_sims):.4f}")
        if np.mean(sanity_sims) < 0.95:
            print("  Low sanity similarity - possible pipeline mismatch or "
                  "frame-decoding inconsistency; interpret results with caution.")
        else:
            print("  Good - re-encoded features match stored features closely.")
        print()

    # ---- baselines ----
    print("-" * 70)
    print("BASELINE: Natural variation (using stored pre-computed features)")
    print("-" * 70)

    bp = summary.get("_baseline_mistake_pairs", {})
    bc = summary.get("_baseline_correct_pairs", {})
    bx = summary.get("_baseline_cross_class",   {})

    print(f"  Mistake ↔ Mistake  (different segs) : {fmt_stat(bp)}")
    print(f"  Correct ↔ Correct  (different segs) : {fmt_stat(bc)}")
    print(f"  Mistake ↔ Correct  (cross-class)    : {fmt_stat(bx)}")
    print()

    natural_mean = bp.get("mean") or 0.80  # fallback if no pairs

    # ---- per-augmentation results ----
    print("-" * 70)
    print("AUGMENTATION RESULTS")
    print("-" * 70)

    for aug_name in aug_names:
        aug_desc = cfg["aug_descriptions"].get(aug_name, aug_name)
        aug_kw   = cfg["aug_kwargs"].get(aug_name, {})
        sum_d    = summary.get(aug_name, {})

        sims   = _collect_aug_sims(segs, aug_name)
        per_t  = _collect_aug_per_t_sims(segs, aug_name)

        print(f"\n  [{aug_name}]")
        print(f"  Description : {aug_desc}")
        print(f"  Parameters  : {aug_kw}")
        print()

        if not sims:
            print("  No results.")
            continue

        print(f"  Global mean-pooled cosine similarity (orig ↔ aug):")
        print(f"    {fmt_stat(sum_d)}")

        if per_t:
            print(f"  Per-temporal-position cosine similarity (all T positions):")
            print(f"    mean={np.mean(per_t):.4f}  std={np.std(per_t):.4f}  "
                  f"min={np.min(per_t):.4f}  max={np.max(per_t):.4f}  "
                  f"n={len(per_t)}")

        print(f"\n  Verdict: {interpret(sum_d.get('mean', 0.0), natural_mean)}")

    # ---- per-segment table ----
    print()
    print("-" * 70)
    print("PER-SEGMENT BREAKDOWN")
    print("-" * 70)
    header = f"  {'Video':<30} {'Verb':<12} {'Dur':>5}s  "
    header += "  ".join(f"{a[:8]:>8}" for a in aug_names)
    header += "  sanity"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for s in segs:
        sanity = s.get("sanity_cos_sim_vs_stored")
        sanity_str = f"{sanity:.3f}" if sanity is not None else " n/a "
        row = f"  {s['video_name']:<30} {s['verb']:<12} {s['duration_sec']:>5.1f}  "
        for aug_name in aug_names:
            aug_d = s.get("augmentations", {}).get(aug_name, {})
            sim   = aug_d.get("cos_sim_global")
            row  += f"  {sim:.4f}" if sim is not None else "     n/a"
        row += f"  {sanity_str}"
        print(row)

    # ---- summary recommendation ----
    print()
    print("=" * 70)
    print("  SUMMARY RECOMMENDATION")
    print("=" * 70)

    all_verdicts = []
    for aug_name in aug_names:
        sum_d = summary.get(aug_name, {})
        m = sum_d.get("mean")
        if m is None:
            continue
        if m >= 0.95:
            label = "VERY SAFE"
        elif m >= 0.90:
            label = "SAFE"
        elif m >= 0.80:
            label = "MODERATE"
        else:
            label = "RISKY"
        all_verdicts.append((aug_name, m, label))
        print(f"  {aug_name:<20} cos_sim={m:.4f}  → {label}")

    print()
    if natural_mean > 0:
        print(f"  Natural mistake-to-mistake variation: mean cos_sim = {natural_mean:.4f}")
        print()
        safe_augs = [name for name, m, label in all_verdicts
                     if label in ("SAFE", "VERY SAFE")]
        risky_augs = [name for name, m, label in all_verdicts
                      if label == "RISKY"]

        if safe_augs:
            print(f"  Safe augmentations for data expansion: {', '.join(safe_augs)}")
            print(f"     These produce features that are much closer to the original")
            print(f"     than the natural variation between different mistake segments.")
        if risky_augs:
            print(f"  Risky augmentations (not recommended): {', '.join(risky_augs)}")

    print("=" * 70)
    print()

# optional plot

def save_plot(data: dict, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping plot.")
        return

    cfg     = data["config"]
    summary = data["summary"]
    aug_names = cfg["augmentations"]

    labels = aug_names + ["mistake↔mistake\n(natural)", "correct↔correct\n(natural)", "mistake↔correct\n(cross-class)"]
    means  = [summary.get(a, {}).get("mean", 0) for a in aug_names]
    stds   = [summary.get(a, {}).get("std",  0) for a in aug_names]

    for key in ["_baseline_mistake_pairs", "_baseline_correct_pairs", "_baseline_cross_class"]:
        b = summary.get(key, {})
        means.append(b.get("mean", 0) or 0)
        stds.append( b.get("std",  0) or 0)

    colours = ["steelblue"] * len(aug_names) + ["coral", "coral", "purple"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, means, yerr=stds, color=colours, capsize=5, alpha=0.8)
    ax.axhline(0.95, ls="--", lw=1, color="green",  label="0.95 (very safe)")
    ax.axhline(0.90, ls="--", lw=1, color="orange", label="0.90 (safe)")
    ax.axhline(0.80, ls="--", lw=1, color="red",    label="0.80 (moderate)")
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Cosine similarity (global mean-pooled feature)")
    ax.set_title("V-JEPA 2.1 feature sensitivity to augmentation vs natural variation")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")

# cli

def main():
    p = argparse.ArgumentParser(description="Analyse augmentation sensitivity results.")
    p.add_argument("--results", required=True, help="Path to results JSON from evaluation/test_augmentation_sensitivity.py")
    p.add_argument("--plot", action="store_true", help="Save a bar chart PNG alongside the results JSON")
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run evaluation/test_augmentation_sensitivity.py first.")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    print_report(data)

    if args.plot:
        plot_path = str(results_path).replace(".json", "_plot.png")
        save_plot(data, plot_path)

if __name__ == "__main__":
    main()
