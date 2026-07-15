"""
Builds a summary figure from augmentation_sensitivity.json.

This script is CPU-only and is intended to be run on the login node, since it
only reads a precomputed results JSON and renders a matplotlib figure.

Usage:
    python plot_augmentation_results.py --results results/augmentation_sensitivity.json \
        --output results/augmentation_summary_figure.pdf
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# colour palette
C_CROP   = "#4C72B0"   # blue
C_WARM   = "#DD8452"   # orange
C_COMB   = "#55A868"   # green
C_MM     = "#C44E52"   # red  – mistake vs mistake baseline
C_CC     = "#8172B2"   # purple – correct vs correct baseline
C_XC     = "#937860"   # brown – cross-class baseline
C_LIGHT  = "#f7f7f7"

AUG_COLOURS   = {"slight_crop": C_CROP, "warm_tint": C_WARM, "combined": C_COMB}
AUG_LABELS    = {
    "slight_crop": "Slight crop\n(70 % of frame)",
    "warm_tint":   "Warm tint\n(R×1.18, B×0.90)",
    "combined":    "Combined\n(crop + tint)",
}
BASE_COLOURS  = {"mistake↔mistake": C_MM, "correct↔correct": C_CC, "mistake↔correct": C_XC}
BASE_LABELS   = {
    "mistake↔mistake":  "Mistake ↔ mistake\n(natural variation)",
    "correct↔correct":  "Correct ↔ correct\n(natural variation)",
    "mistake↔correct":  "Mistake ↔ correct\n(class separation)",
}

def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def seg_sims(data: dict, aug: str) -> list[float]:
    return [
        s["augmentations"][aug]["cos_sim_global"]
        for s in data["segments"] if aug in s.get("augmentations", {})
    ]


def per_t_all(data: dict, aug: str) -> list[float]:
    out = []
    for s in data["segments"]:
        if aug in s.get("augmentations", {}):
            out.extend(s["augmentations"][aug]["per_t"])
    return out


def l2_sims(data: dict, aug: str) -> list[float]:
    return [
        s["augmentations"][aug]["l2_dist_global"]
        for s in data["segments"] if aug in s.get("augmentations", {})
    ]


# main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/augmentation_sensitivity.json")
    p.add_argument("--output",  default="results/augmentation_summary_figure.pdf")
    args = p.parse_args()

    data = load(args.results)
    summary = data["summary"]
    n_segs  = data["config"]["n_segments"]

    augs   = ["slight_crop", "warm_tint", "combined"]
    bases  = ["mistake↔mistake", "correct↔correct", "mistake↔correct"]
    base_keys = {
        "mistake↔mistake":  "_baseline_mistake_pairs",
        "correct↔correct":  "_baseline_correct_pairs",
        "mistake↔correct":  "_baseline_cross_class",
    }

    # figure layout
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")

    gs = GridSpec(2, 3, figure=fig,
                  left=0.07, right=0.97, top=0.88, bottom=0.10,
                  hspace=0.52, wspace=0.38)

    ax_main  = fig.add_subplot(gs[:, 0])   # tall left panel - headline comparison
    ax_box   = fig.add_subplot(gs[0, 1])   # top-middle - per-segment box plots
    ax_pert  = fig.add_subplot(gs[0, 2])   # top-right  - per-t stability
    ax_l2    = fig.add_subplot(gs[1, 1])   # bottom-middle - l2 distance
    ax_gap   = fig.add_subplot(gs[1, 2])   # bottom-right - gap / safety margin

    # panel a: headline bar chart
    ax = ax_main
    all_labels, all_means, all_stds, all_cols = [], [], [], []

    for aug in augs:
        s = summary[aug]
        all_labels.append(AUG_LABELS[aug])
        all_means.append(s["mean"])
        all_stds.append(s["std"])
        all_cols.append(AUG_COLOURS[aug])

    for bk in bases:
        s = summary[base_keys[bk]]
        all_labels.append(BASE_LABELS[bk])
        all_means.append(s["mean"])
        all_stds.append(s["std"])
        all_cols.append(BASE_COLOURS[bk])

    y = np.arange(len(all_labels))
    bars = ax.barh(y, all_means, xerr=all_stds, color=all_cols, alpha=0.85,
                   height=0.6, capsize=4, error_kw={"elinewidth": 1.5, "ecolor": "dimgrey"})

    # annotate with value
    for i, (m, s) in enumerate(zip(all_means, all_stds)):
        ax.text(m + 0.002, i, f"{m:.3f}", va="center", ha="left", fontsize=8.5,
                color="black", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(all_labels, fontsize=9)
    ax.set_xlabel("Cosine similarity (global mean-pooled features)", fontsize=9)
    ax.set_xlim(0.60, 1.04)
    ax.axvline(1.0, lw=0.7, ls="--", color="grey", alpha=0.5)
    ax.set_title("A  Augmented vs natural variation", fontsize=10, fontweight="bold", loc="left")

    # divider between augmentations and baselines
    ax.axhline(2.5, lw=1, ls=":", color="grey", alpha=0.6)
    ax.text(0.605, 2.56, "── augmentations ──", fontsize=7.5, color="grey", va="bottom")
    ax.text(0.605, 2.44, "── baselines ──────", fontsize=7.5, color="grey", va="top")

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", ls="--", alpha=0.3)

    # panel b: per-segment box plots
    ax = ax_box
    per_seg_data = [seg_sims(data, aug) for aug in augs]
    bp = ax.boxplot(per_seg_data, vert=True, patch_artist=True,
                    medianprops={"color": "white", "linewidth": 2},
                    whiskerprops={"linewidth": 1.2},
                    capprops={"linewidth": 1.2},
                    flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})

    for patch, aug in zip(bp["boxes"], augs):
        patch.set_facecolor(AUG_COLOURS[aug])
        patch.set_alpha(0.75)

    # overlay individual points (jittered)
    rng = np.random.default_rng(0)
    for i, (vals, aug) in enumerate(zip(per_seg_data, augs), start=1):
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=AUG_COLOURS[aug], s=18, alpha=0.6, zorder=3)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Crop", "Warm tint", "Combined"], fontsize=9)
    ax.set_ylabel("Cosine similarity (per segment)", fontsize=9)
    ax.set_ylim(0.92, 1.005)
    ax.axhline(summary["_baseline_mistake_pairs"]["mean"], lw=1.2, ls="--",
               color=C_MM, alpha=0.7, label=f"Mistake baseline ({summary['_baseline_mistake_pairs']['mean']:.3f})")
    ax.axhline(summary["_baseline_correct_pairs"]["mean"], lw=1.2, ls="--",
               color=C_CC, alpha=0.7, label=f"Correct baseline ({summary['_baseline_correct_pairs']['mean']:.3f})")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("B  Per-segment distribution\n(n=30 mistake segments)", fontsize=10, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.3)

    # panel c: per-temporal-position stability
    ax = ax_pert
    for aug in augs:
        per_t_by_seg = []
        for s in data["segments"]:
            if aug in s.get("augmentations", {}):
                per_t_by_seg.append(s["augmentations"][aug]["per_t"])  # (32,)

        arr = np.array(per_t_by_seg)   # (n_segs, 32)
        mean_t = arr.mean(axis=0)
        std_t  = arr.std(axis=0)
        t = np.arange(1, 33)
        ax.plot(t, mean_t, color=AUG_COLOURS[aug], lw=1.8, label=AUG_LABELS[aug].split("\n")[0])
        ax.fill_between(t, mean_t - std_t, mean_t + std_t,
                        color=AUG_COLOURS[aug], alpha=0.15)

    ax.set_xlabel("Temporal position (1 = start → 32 = end of clip)", fontsize=9)
    ax.set_ylabel("Cosine similarity", fontsize=9)
    ax.set_xlim(1, 32)
    ax.set_ylim(0.92, 1.01)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("C  Per-temporal-position similarity\n(mean ± std across segments)", fontsize=10, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(ls="--", alpha=0.3)

    # panel d: l2 distance
    ax = ax_l2
    l2_data = [l2_sims(data, aug) for aug in augs]
    bp2 = ax.boxplot(l2_data, vert=True, patch_artist=True,
                     medianprops={"color": "white", "linewidth": 2},
                     whiskerprops={"linewidth": 1.2},
                     capprops={"linewidth": 1.2},
                     flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})
    for patch, aug in zip(bp2["boxes"], augs):
        patch.set_facecolor(AUG_COLOURS[aug])
        patch.set_alpha(0.75)
    for i, (vals, aug) in enumerate(zip(l2_data, augs), start=1):
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=AUG_COLOURS[aug], s=18, alpha=0.6, zorder=3)

    # add baseline l2s from cross-class pairs
    xc_l2 = np.mean([p["l2_dist_global"] for p in data["baselines"]["cross_class"]])
    mm_l2 = np.mean([p["l2_dist_global"] for p in data["baselines"]["mistake_pairs"]])
    ax.axhline(mm_l2, lw=1.2, ls="--", color=C_MM, alpha=0.7,
               label=f"Mistake baseline ({mm_l2:.1f})")
    ax.axhline(xc_l2, lw=1.2, ls="--", color=C_XC, alpha=0.7,
               label=f"Cross-class baseline ({xc_l2:.1f})")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Crop", "Warm tint", "Combined"], fontsize=9)
    ax.set_ylabel("L2 distance (global mean-pooled features)", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("D  L2 distance in feature space\n(lower = more similar)", fontsize=10, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.3)

    # panel e: safety margin gap
    ax = ax_gap
    # for each segment, compute gap between its aug cos-sim and the mean mistake baseline
    base_mean = summary["_baseline_mistake_pairs"]["mean"]
    for aug in augs:
        gaps = [s["augmentations"][aug]["cos_sim_global"] - base_mean
                for s in data["segments"] if aug in s.get("augmentations", {})]
        verb_labels = [f"{s['verb']} {s['noun']}"
                       for s in data["segments"] if aug in s.get("augmentations", {})]
        sorted_idx = np.argsort(gaps)
        ax.barh(np.arange(len(gaps)) + augs.index(aug) * 0.22,
                np.array(gaps)[sorted_idx],
                height=0.20, color=AUG_COLOURS[aug], alpha=0.75,
                label=AUG_LABELS[aug].split("\n")[0])

    ax.axvline(0, lw=1.0, color="black", alpha=0.5)
    ax.set_xlabel("Safety margin\n(aug cos-sim − mistake baseline mean)", fontsize=9)
    ax.set_title("E  Safety margin per segment\n(positive = augmented stays within class)", fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", ls="--", alpha=0.3)

    # suptitle
    fig.suptitle(
        "V-JEPA 2.1 Feature Sensitivity to Data Augmentation - HoloAssist Mistake Segments\n"
        f"n = {n_segs} mistake segments · augmentation: 70 % Gaussian crop & warm tint (R×1.18, B×0.90)",
        fontsize=11, fontweight="bold", y=0.96
    )

    # save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for ext in [str(out), str(out).replace(".pdf", ".png")]:
        plt.savefig(ext, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved → {ext}")


if __name__ == "__main__":
    main()
