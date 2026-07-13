# same as plot_full_token_sensitivity.py but for spatially-pooled features - the format the
# downstream classifier actually trains on. layout mirrors that script's panels exactly.
# usage: python plot_pooled_sensitivity.py --results results/augmentation_sensitivity_full_tokens_crop90.json
#   --output results/pooled_sensitivity_crop90.png

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

# colour palette
C_CROP  = "#4C72B0"   # blue
C_WARM  = "#DD8452"   # orange
C_COMB  = "#55A868"   # green

# metric colours — mirror of all/centre/edge but for global/per-t mean/per-t min
C_GLOBAL = "#333333"  # dark  (analogous to "all tokens")
C_MEAN   = "#E15759"  # red   (analogous to "centre tokens")
C_MIN    = "#76B7B2"  # teal  (analogous to "edge tokens")

C_MM  = "#C44E52"   # mistake↔mistake baseline
C_CC  = "#8172B2"   # correct↔correct baseline

AUG_COLOURS = {"slight_crop": C_CROP, "warm_tint": C_WARM, "combined": C_COMB}
AUG_LABELS  = {
    "slight_crop": "Slight Crop\n(Gaussian-centred, 90 % of frame)",
    "warm_tint":   "Warm Tint\n(R×1.18, G×1.04, B×0.90)",
    "combined":    "Combined\n(Crop 90 % + Warm Tint)",
}
AUG_SHORT = {"slight_crop": "Crop", "warm_tint": "Warm tint", "combined": "Combined"}

METRICS = ["pooled_cos_sim_global", "pooled_cos_sim_per_t_mean", "pooled_cos_sim_per_t_min"]
METRIC_LABELS = {
    "pooled_cos_sim_global":     "Global\n(mean-pool)",
    "pooled_cos_sim_per_t_mean": "Per-T mean\n(avg frame)",
    "pooled_cos_sim_per_t_min":  "Per-T min\n(worst frame)",
}
METRIC_COLOURS = {
    "pooled_cos_sim_global":     C_GLOBAL,
    "pooled_cos_sim_per_t_mean": C_MEAN,
    "pooled_cos_sim_per_t_min":  C_MIN,
}
AUGS = ["slight_crop", "warm_tint", "combined"]


# helpers
def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_vals(data: dict, aug: str, metric: str) -> np.ndarray:
    """Per-segment scalar values for a given pooled metric."""
    return np.array([
        s["augmentations"][aug][metric]
        for s in data["segments"]
        if aug in s.get("augmentations", {})
    ])


def get_per_t_std(data: dict, aug: str) -> np.ndarray:
    """Per-segment temporal variability (std of per-T cosine sims)."""
    return np.array([
        s["augmentations"][aug]["pooled_cos_sim_per_t_std"]
        for s in data["segments"]
        if aug in s.get("augmentations", {})
    ])


def styled_boxplot(ax, data_list, positions, colours, rng):
    """Box plots with individual jittered points — mirrors plot_full_token_sensitivity."""
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
                   default="results/augmentation_sensitivity_full_tokens_crop90.json")
    p.add_argument("--output",
                   default="results/pooled_sensitivity_crop90.png")
    args = p.parse_args()

    data    = load(args.results)
    summary = data["summary"]
    n_segs  = data["config"]["n_segments"]
    rng     = np.random.default_rng(42)

    # update labels from actual crop_frac in config
    crop_frac = float(data["config"]["aug_kwargs"].get("slight_crop", {}).get("crop_frac", 0.90))
    crop_pct  = int(round(crop_frac * 100))
    AUG_LABELS["slight_crop"] = f"Slight Crop\n(Gaussian-centred, {crop_pct} % of frame)"
    AUG_LABELS["combined"]    = f"Combined\n(Crop {crop_pct} % + Warm Tint)"

    # baseline reference lines
    bl_mm = summary["_baseline_mistake_pairs"]
    bl_cc = summary["_baseline_correct_pairs"]

    # pre-collect per-segment values (all, per-t mean, per-t min) for each aug
    seg_vals = {
        aug: {m: get_vals(data, aug, m) for m in METRICS}
        for aug in AUGS
    }
    seg_std = {aug: get_per_t_std(data, aug) for aug in AUGS}

    # figure layout (mirrors full_token_sensitivity plot exactly)
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
    ax_var     = fig.add_subplot(gs[1, 3])   # bottom col 3 — temporal variability

    # panel a: grouped summary bar chart (3 augs × 3 pooled metrics)
    ax = ax_summary

    n_aug    = len(AUGS)
    n_metric = len(METRICS)
    group_w  = 0.8
    bar_w    = group_w / n_metric
    group_centres = np.arange(n_aug)

    for mi, metric in enumerate(METRICS):
        offsets = (mi - (n_metric - 1) / 2) * bar_w
        means   = [seg_vals[aug][metric].mean() for aug in AUGS]
        stds    = [seg_vals[aug][metric].std()  for aug in AUGS]
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
        for x, m, s in zip(group_centres + offsets, means, stds):
            ax.text(x, m + s + 0.0005, f"{m:.4f}",
                    ha="center", va="bottom", fontsize=7.5, color="black",
                    fontweight="bold")

    # baseline reference lines
    ax.axhline(bl_mm["mean"], lw=1.4, ls="--", color=C_MM, alpha=0.75)
    ax.axhline(bl_cc["mean"], lw=1.4, ls="--", color=C_CC, alpha=0.75)

    ax.set_xticks(group_centres)
    ax.set_xticklabels([AUG_LABELS[a] for a in AUGS], fontsize=10)
    ax.set_ylabel("Cosine similarity (original ↔ augmented)", fontsize=10)

    # y-axis: tight around the pooled data range with some headroom
    all_pooled = np.concatenate([seg_vals[a][m] for a in AUGS for m in METRICS])
    y_lo = max(0.40, all_pooled.min() - 0.02)
    y_hi = min(1.02, all_pooled.max() + 0.015)
    ax.set_ylim(y_lo, y_hi)

    ax.axhline(1.0, lw=0.7, ls=":", color="grey", alpha=0.4)
    ax.set_title(
        "A   Pooled cosine similarity by augmentation (spatial mean-pool → downstream format)",
        fontsize=11, fontweight="bold", loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.28, zorder=0)

    # shade crop-affected augmentation groups
    for gi in [0, 2]:
        ax.axvspan(gi - 0.45, gi + 0.45, alpha=0.04, color=AUG_COLOURS[AUGS[gi]], zorder=0)

    # shared y-limits for panels b–d (so they're visually comparable)
    bd_lo = max(0.40, all_pooled.min() - 0.015)
    bd_hi = min(1.005, all_pooled.max() + 0.010)

    positions  = [1, 2, 3]
    met_cols   = [METRIC_COLOURS[m] for m in METRICS]

    # panel b: slight crop — per-segment distributions
    ax = ax_crop
    vals = [seg_vals["slight_crop"][m] for m in METRICS]
    styled_boxplot(ax, vals, positions, met_cols, rng)

    ax.axhline(bl_mm["mean"], lw=1.3, ls="--", color=C_MM, alpha=0.7)
    ax.axhline(bl_cc["mean"], lw=1.3, ls="--", color=C_CC, alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Global", "Per-T mean", "Per-T min"], fontsize=9.5)
    ax.set_ylabel("Cosine similarity (per segment)", fontsize=9.5)
    ax.set_ylim(bd_lo, bd_hi)
    ax.set_title(
        f"B   {AUG_SHORT['slight_crop']}  (n={n_segs})",
        fontsize=10.5, fontweight="bold", loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.28)

    # panel c: warm tint — per-segment distributions
    ax = ax_warm
    vals = [seg_vals["warm_tint"][m] for m in METRICS]
    styled_boxplot(ax, vals, positions, met_cols, rng)

    ax.axhline(bl_mm["mean"], lw=1.3, ls="--", color=C_MM, alpha=0.7)
    ax.axhline(bl_cc["mean"], lw=1.3, ls="--", color=C_CC, alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Global", "Per-T mean", "Per-T min"], fontsize=9.5)
    ax.set_ylabel("Cosine similarity (per segment)", fontsize=9.5)
    ax.set_ylim(bd_lo, bd_hi)
    ax.set_title(
        f"C   {AUG_SHORT['warm_tint']}  (n={n_segs})",
        fontsize=10.5, fontweight="bold", loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.28)

    # panel d: combined — per-segment distributions
    ax = ax_comb
    vals = [seg_vals["combined"][m] for m in METRICS]
    styled_boxplot(ax, vals, positions, met_cols, rng)

    ax.axhline(bl_mm["mean"], lw=1.3, ls="--", color=C_MM, alpha=0.7)
    ax.axhline(bl_cc["mean"], lw=1.3, ls="--", color=C_CC, alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Global", "Per-T mean", "Per-T min"], fontsize=9.5)
    ax.set_ylabel("Cosine similarity (per segment)", fontsize=9.5)
    ax.set_ylim(bd_lo, bd_hi)
    ax.set_title(
        f"D   {AUG_SHORT['combined']}  (n={n_segs})",
        fontsize=10.5, fontweight="bold", loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.28)

    # panel e: temporal variability (per-t std) across augmentations
    # analogue of "spatial asymmetry (centre − edge)" from full-token plot
    ax = ax_var

    var_data = [seg_std[aug] for aug in AUGS]
    var_cols = [AUG_COLOURS[aug] for aug in AUGS]
    styled_boxplot(ax, var_data, [1, 2, 3], var_cols, rng)

    ax.axhline(0, lw=1.2, ls="-", color="black", alpha=0.45)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([AUG_SHORT[a] for a in AUGS], fontsize=9.5)
    ax.set_ylabel("Std of per-T cosine similarity", fontsize=9.5)
    ax.set_title(
        "E   Temporal variability\n(per-T cos-sim std)",
        fontsize=10.5, fontweight="bold", loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.28)

    # annotate mean std for each augmentation
    for pos, vals, aug in zip([1, 2, 3], var_data, AUGS):
        m = vals.mean()
        ax.text(pos, vals.max() + vals.max() * 0.05, f"σ={m:.5f}",
                ha="center", va="bottom",
                fontsize=8, color=AUG_COLOURS[aug], fontweight="bold")

    # single shared legend
    legend_handles = [
        mpatches.Patch(color=METRIC_COLOURS["pooled_cos_sim_global"],     alpha=0.82,
                       label="Global (spatial+temporal mean-pool)"),
        mpatches.Patch(color=METRIC_COLOURS["pooled_cos_sim_per_t_mean"], alpha=0.82,
                       label="Per-T mean (avg over 32 frames)"),
        mpatches.Patch(color=METRIC_COLOURS["pooled_cos_sim_per_t_min"],  alpha=0.82,
                       label="Per-T min (worst single frame)"),
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
        "V-JEPA 2  Pooled Feature Sensitivity to Data Augmentation\n"
        f"HoloAssist mistake segments · n = {n_segs} · "
        "metrics: global, per-T mean, per-T min cosine similarity",
        fontsize=12, fontweight="bold", y=0.97,
    )

    # save
    out = Path(args.output).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
