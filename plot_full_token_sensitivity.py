# builds a figure from augmentation_sensitivity_full_tokens.json: full/centre/edge per-token
# cosine similarity, for slight_crop, warm_tint, combined.
# usage: python plot_full_token_sensitivity.py --results results/augmentation_sensitivity_full_tokens.json
#   --output results/full_token_sensitivity.pdf

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# colour palette
C_CROP  = "#4C72B0"   # blue
C_WARM  = "#DD8452"   # orange
C_COMB  = "#55A868"   # green

C_MEAN   = "#333333"  # dark
C_CENTRE = "#E15759"  # red  (centre tokens)
C_EDGE   = "#76B7B2"  # teal (edge tokens)

C_MM  = "#C44E52"   # mistake↔mistake baseline
C_CC  = "#8172B2"   # correct↔correct baseline
C_XC  = "#937860"   # cross-class baseline

AUG_COLOURS = {"slight_crop": C_CROP, "warm_tint": C_WARM, "combined": C_COMB}
AUG_LABELS  = {
    "slight_crop": "Slight Crop\n(Gaussian-centred, 70 % of frame)",
    "warm_tint":   "Warm Tint\n(R×1.18, G×1.04, B×0.90)",
    "combined":    "Combined\n(Crop 70 % + Warm Tint)",
}
AUG_SHORT = {"slight_crop": "Crop", "warm_tint": "Warm tint", "combined": "Combined"}

METRIC_LABELS = {
    "full_cos_sim_mean":   "All tokens\n(mean)",
    "full_cos_sim_centre": "Centre\ntokens",
    "full_cos_sim_edge":   "Edge\ntokens",
}
METRIC_COLOURS = {
    "full_cos_sim_mean":   C_MEAN,
    "full_cos_sim_centre": C_CENTRE,
    "full_cos_sim_edge":   C_EDGE,
}
METRICS = ["full_cos_sim_mean", "full_cos_sim_centre", "full_cos_sim_edge"]
AUGS    = ["slight_crop", "warm_tint", "combined"]


# helpers
def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_vals(data: dict, aug: str, metric: str) -> np.ndarray:
    return np.array([
        s["augmentations"][aug][metric]
        for s in data["segments"]
        if aug in s.get("augmentations", {})
    ])


def get_delta(data: dict, aug: str) -> np.ndarray:
    """Centre minus edge per segment."""
    return np.array([
        s["augmentations"][aug]["full_cos_sim_centre_minus_edge"]
        for s in data["segments"]
        if aug in s.get("augmentations", {})
    ])


def styled_boxplot(ax, data_list, positions, colours, rng):
    """Box plots with individual jittered points."""
    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=0.35,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2.2},
        whiskerprops={"linewidth": 1.3},
        capprops={"linewidth": 1.3},
        flierprops={"marker": "o", "markersize": 3.5, "alpha": 0.4},
        manage_ticks=False,
    )
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.80)
    for vals, pos, col in zip(data_list, positions, colours):
        jitter = rng.uniform(-0.10, 0.10, len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals,
                   color=col, s=14, alpha=0.55, zorder=4)
    return bp


# main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results",
                   default="results/augmentation_sensitivity_full_tokens.json")
    p.add_argument("--output",
                   default="results/full_token_sensitivity.png")
    args = p.parse_args()

    data    = load(args.results)
    summary = data["summary"]
    n_segs  = data["config"]["n_segments"]
    rng     = np.random.default_rng(42)

    # read crop_frac from config so labels are always accurate
    crop_frac = float(data["config"]["aug_kwargs"].get("slight_crop", {}).get("crop_frac", 0.70))
    crop_pct  = int(round(crop_frac * 100))
    AUG_LABELS["slight_crop"] = f"Slight Crop\n(Gaussian-centred, {crop_pct} % of frame)"
    AUG_LABELS["combined"]    = f"Combined\n(Crop {crop_pct} % + Warm Tint)"

    # baseline values
    bl_mm = summary["_baseline_mistake_pairs"]
    bl_cc = summary["_baseline_correct_pairs"]

    # figure layout
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor("white")

    gs = GridSpec(
        2, 4, figure=fig,
        left=0.06, right=0.97, top=0.88, bottom=0.13,
        hspace=0.55, wspace=0.38,
    )

    ax_summary = fig.add_subplot(gs[0, :])   # full top row — summary grouped bars
    ax_crop    = fig.add_subplot(gs[1, 0])   # bottom col 0 — slight_crop detail
    ax_warm    = fig.add_subplot(gs[1, 1])   # bottom col 1 — warm_tint detail
    ax_comb    = fig.add_subplot(gs[1, 2])   # bottom col 2 — combined detail
    ax_delta   = fig.add_subplot(gs[1, 3])   # bottom col 3 — centre−edge delta

    # panel a: grouped summary bar chart (3 aug × 3 metrics)
    ax = ax_summary

    n_aug    = len(AUGS)
    n_metric = len(METRICS)
    group_w  = 0.8
    bar_w    = group_w / n_metric
    group_centres = np.arange(n_aug)

    for mi, metric in enumerate(METRICS):
        offsets = (mi - (n_metric - 1) / 2) * bar_w
        means = [summary[aug]["full" if metric == "full_cos_sim_mean"
                               else "centre" if metric == "full_cos_sim_centre"
                               else "edge"]["mean"]
                 for aug in AUGS]
        stds  = [summary[aug]["full" if metric == "full_cos_sim_mean"
                               else "centre" if metric == "full_cos_sim_centre"
                               else "edge"]["std"]
                 for aug in AUGS]
        ax.bar(
            group_centres + offsets, means,
            width=bar_w * 0.92,
            yerr=stds,
            color=METRIC_COLOURS[metric],
            alpha=0.82,
            capsize=4,
            error_kw={"elinewidth": 1.5, "ecolor": "dimgrey"},
            zorder=3,
        )
        # annotate bars
        for x, m, s in zip(group_centres + offsets, means, stds):
            ax.text(x, m + s + 0.005, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=7.5, color="black",
                    fontweight="bold")

    # baseline reference lines (no label= — legend handled by fig.legend below)
    ax.axhline(bl_mm["mean"], lw=1.4, ls="--", color=C_MM, alpha=0.75)
    ax.axhline(bl_cc["mean"], lw=1.4, ls="--", color=C_CC, alpha=0.75)

    ax.set_xticks(group_centres)
    ax.set_xticklabels([AUG_LABELS[a] for a in AUGS], fontsize=10)
    ax.set_ylabel("Cosine similarity (original ↔ augmented)", fontsize=10)
    ax.set_ylim(0.45, 1.055)
    ax.axhline(1.0, lw=0.7, ls=":", color="grey", alpha=0.4)
    ax.set_title(
        "A   Full-token cosine similarity by augmentation and spatial region",
        fontsize=11, fontweight="bold", loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.28, zorder=0)

    # shade the two "crop-affected" augmentation groups subtly
    for gi in [0, 2]:
        ax.axvspan(gi - 0.45, gi + 0.45, alpha=0.04, color=AUG_COLOURS[AUGS[gi]], zorder=0)

    # panel b: slight crop — per-segment distributions (box + scatter)
    ax = ax_crop
    positions = [1, 2, 3]
    vals_crop = [get_vals(data, "slight_crop", m) for m in METRICS]
    cols_crop = [METRIC_COLOURS[m] for m in METRICS]
    styled_boxplot(ax, vals_crop, positions, cols_crop, rng)

    ax.axhline(bl_mm["mean"], lw=1.3, ls="--", color=C_MM, alpha=0.7)
    ax.axhline(bl_cc["mean"], lw=1.3, ls="--", color=C_CC, alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(["All tokens", "Centre", "Edge"], fontsize=9.5)
    ax.set_ylabel("Cosine similarity (per segment)", fontsize=9.5)
    ax.set_ylim(0.38, 0.74)
    ax.set_title(
        f"B   {AUG_SHORT['slight_crop']}  (n={n_segs})",
        fontsize=10.5, fontweight="bold", loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.28)

    # panel c: warm tint — per-segment distributions
    ax = ax_warm
    vals_warm = [get_vals(data, "warm_tint", m) for m in METRICS]
    styled_boxplot(ax, vals_warm, positions, cols_crop, rng)  # same metric colours

    ax.axhline(bl_mm["mean"], lw=1.3, ls="--", color=C_MM, alpha=0.7)
    ax.axhline(bl_cc["mean"], lw=1.3, ls="--", color=C_CC, alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(["All tokens", "Centre", "Edge"], fontsize=9.5)
    ax.set_ylabel("Cosine similarity (per segment)", fontsize=9.5)
    ax.set_ylim(0.92, 1.004)
    ax.set_title(
        f"C   {AUG_SHORT['warm_tint']}  (n={n_segs})",
        fontsize=10.5, fontweight="bold", loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.28)

    # panel d: combined — per-segment distributions
    ax = ax_comb
    vals_comb = [get_vals(data, "combined", m) for m in METRICS]
    styled_boxplot(ax, vals_comb, positions, cols_crop, rng)

    ax.axhline(bl_mm["mean"], lw=1.3, ls="--", color=C_MM, alpha=0.7)
    ax.axhline(bl_cc["mean"], lw=1.3, ls="--", color=C_CC, alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(["All tokens", "Centre", "Edge"], fontsize=9.5)
    ax.set_ylabel("Cosine similarity (per segment)", fontsize=9.5)
    ax.set_ylim(0.38, 0.74)
    ax.set_title(
        f"D   {AUG_SHORT['combined']}  (n={n_segs})",
        fontsize=10.5, fontweight="bold", loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.28)

    # panel e: centre − edge spatial asymmetry across all augmentations
    ax = ax_delta

    delta_data  = [get_delta(data, aug) for aug in AUGS]
    delta_cols  = [AUG_COLOURS[aug] for aug in AUGS]
    delta_pos   = [1, 2, 3]

    styled_boxplot(ax, delta_data, delta_pos, delta_cols, rng)
    ax.axhline(0, lw=1.2, ls="-", color="black", alpha=0.45)

    ax.set_xticks(delta_pos)
    ax.set_xticklabels([AUG_SHORT[a] for a in AUGS], fontsize=9.5)
    ax.set_ylabel("Centre − Edge cosine similarity", fontsize=9.5)
    ax.set_title(
        "E   Spatial asymmetry\n(centre − edge tokens)",
        fontsize=10.5, fontweight="bold", loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.28)

    # annotate mean delta for each
    for pos, vals, aug in zip(delta_pos, delta_data, AUGS):
        m = vals.mean()
        pval_sign = "↑" if m > 0.001 else ("↓" if m < -0.001 else "≈0")
        ax.text(pos, vals.max() + 0.003, f"Δ={m:+.4f}", ha="center", va="bottom",
                fontsize=8, color=AUG_COLOURS[aug], fontweight="bold")

    # single shared legend for the whole figure
    import matplotlib.lines as mlines
    legend_handles = [
        # spatial region boxes (panels b–d)
        mpatches.Patch(color=METRIC_COLOURS["full_cos_sim_mean"],   alpha=0.82,
                       label="All tokens (mean)"),
        mpatches.Patch(color=METRIC_COLOURS["full_cos_sim_centre"], alpha=0.82,
                       label="Centre tokens"),
        mpatches.Patch(color=METRIC_COLOURS["full_cos_sim_edge"],   alpha=0.82,
                       label="Edge tokens"),
        # baselines (dashed lines in panels a–d)
        mlines.Line2D([], [], color=C_MM, lw=1.4, ls="--",
                      label=f"Mistake↔Mistake baseline ({bl_mm['mean']:.3f})"),
        mlines.Line2D([], [], color=C_CC, lw=1.4, ls="--",
                      label=f"Correct↔Correct baseline ({bl_cc['mean']:.3f})"),
    ]
    fig.legend(
        handles=legend_handles,
        fontsize=8.5,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.0),
        framealpha=0.95,
        edgecolor="lightgrey",
        columnspacing=1.5,
    )

    # suptitle
    fig.suptitle(
        "V-JEPA 2  Full-token Feature Sensitivity to Data Augmentation\n"
        f"HoloAssist mistake segments · n = {n_segs} · "
        "metrics: mean-pooled, centre-token, edge-token cosine similarity",
        fontsize=12, fontweight="bold", y=0.97,
    )

    # save (png only)
    out = Path(args.output).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
