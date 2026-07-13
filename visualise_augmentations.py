# decodes one real holoassist frame and saves original vs each augmentation side by side.
# cpu-only, runs on the login node in seconds.
# usage: python visualise_augmentations.py [--video NAME --frame N] [--output results/aug_comparison.png]

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.augmentations import slight_crop, warm_tint, combined_aug


# helpers

def decode_one_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Return (H, W, 3) uint8 RGB array for the closest decoded frame."""
    container = av.open(video_path)
    stream    = container.streams.video[0]
    fps       = float(stream.average_rate)
    tb        = float(stream.time_base)

    seek_ts = max(0, int(frame_idx / fps / tb) - 1)
    container.seek(seek_ts, stream=stream, backward=True)

    best = None
    for frame in container.decode(stream):
        fi = round(float(frame.pts) * tb * fps)
        if best is None or abs(fi - frame_idx) < abs(best[0] - frame_idx):
            best = (fi, frame.to_ndarray(format="rgb24"))
        if fi >= frame_idx:
            break
    container.close()
    return best[1]


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) uint8 tensor → (H, W, C) uint8 numpy."""
    return t[0].permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)


def show_img(ax, img_np: np.ndarray, title: str, note: str = ""):
    ax.imshow(img_np)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    if note:
        ax.set_xlabel(note, fontsize=8, color="dimgrey")
    ax.axis("off")


def draw_crop_box(ax, H: int, W: int, crop_frac: float, top: int, left: int):
    """Overlay the actual crop window on an axis."""
    crop_h = int(H * crop_frac)
    crop_w = int(W * crop_frac)
    rect = mpatches.Rectangle(
        (left, top), crop_w, crop_h,
        linewidth=2.5, edgecolor="#00FF7F", facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)
    ax.text(left + 6, top + 20, f"crop window\n({int(crop_frac*100)}% of frame)",
            color="#00FF7F", fontsize=7.5, va="top",
            bbox=dict(facecolor="black", alpha=0.45, pad=2, edgecolor="none"))


def diff_map(orig: np.ndarray, aug: np.ndarray, scale: float = 4.0) -> np.ndarray:
    """Absolute pixel-difference map, brightened for visibility."""
    diff = np.abs(orig.astype(np.int16) - aug.astype(np.int16)).astype(np.float32)
    return (diff * scale).clip(0, 255).astype(np.uint8)


# main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir",  default="holoassist/videos")
    p.add_argument("--video",      default="R014-7July-DSLR",
                   help="Video name (folder under video_dir)")
    p.add_argument("--frame",      type=int, default=437,
                   help="Frame index to decode")
    p.add_argument("--output",     default="results/augmentation_comparison.png")
    p.add_argument("--seed",       type=int, default=7)
    p.add_argument("--crop_frac",  type=float, default=0.90)
    p.add_argument("--n_crop_examples", type=int, default=3,
                   help="How many different random crop offsets to show")
    args = p.parse_args()

    video_path = Path(args.video_dir) / args.video / "Export_py" / "Video_compress.mp4"
    if not video_path.exists():
        print(f"ERROR: video not found at {video_path}")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Decoding frame {args.frame} from {args.video}…")
    frame_np = decode_one_frame(str(video_path), args.frame)   # (h, w, 3) uint8
    H, W, _  = frame_np.shape
    print(f"  Frame: {W}×{H}  dtype={frame_np.dtype}")

    # wrap in (1, c, h, w) so augmentation functions work
    frame_t = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0)

    rng = random.Random(args.seed)

    # produce augmented versions
    warm_np = to_numpy(warm_tint(frame_t))

    # produce n_crop_examples crop variants with known offsets for visualisation
    crop_frac     = args.crop_frac
    crop_h_px     = int(H * crop_frac)
    crop_w_px     = int(W * crop_frac)
    max_top_px    = H - crop_h_px
    max_left_px   = W - crop_w_px

    crops = []
    for i in range(args.n_crop_examples):
        seed_i = args.seed + i * 31
        _rng   = random.Random(seed_i)
        sigma  = 0.4 * (max_top_px / 2.0)
        top_i  = int(max(0, min(max_top_px,  round(_rng.gauss(max_top_px  / 2, sigma)))))
        sigma  = 0.4 * (max_left_px / 2.0)
        left_i = int(max(0, min(max_left_px, round(_rng.gauss(max_left_px / 2, sigma)))))
        crop_t = slight_crop(frame_t, crop_frac=crop_frac, crop_frac_jitter=0.0,
                             center_bias_sigma=0.0,
                             rng=random.Random(seed_i))
        # recompute actual offset (zero jitter, so crop_frac is exact)
        crops.append((crop_t, top_i, left_i))

    comb_np = to_numpy(combined_aug(
        frame_t, crop_frac=crop_frac, center_bias_sigma=0.4,
        rng=random.Random(args.seed)
    ))

    # layout
    # row 0 : original │ warm_tint │ combined
    # row 1 : slight_crop × n_crop_examples (with crop box overlay)
    # row 2 : δ diff maps (brightened ×4) for warm / combined / crop[0]

    n_cols = max(3, args.n_crop_examples)
    fig = plt.figure(figsize=(5 * n_cols, 12))
    fig.suptitle(
        f"Augmentation comparison — {args.video}  frame {args.frame}  "
        f"(task: rotate lens_cover)",
        fontsize=12, fontweight="bold", y=0.995
    )
    gs = fig.add_gridspec(3, n_cols, hspace=0.32, wspace=0.04)

    # -- row 0 --
    ax0 = fig.add_subplot(gs[0, 0])
    show_img(ax0, frame_np, "Original")

    ax1 = fig.add_subplot(gs[0, 1])
    show_img(ax1, warm_np, "warm_tint",
             note="R×1.18  G×1.04  B×0.90  (warm light)")

    ax2 = fig.add_subplot(gs[0, 2])
    show_img(ax2, comb_np, "combined  (crop + warm)",
             note=f"Gaussian crop {int(crop_frac*100)}%  +  warm tint")

    for c in range(3, n_cols):
        fig.add_subplot(gs[0, c]).axis("off")

    # -- row 1: crop variants --
    for i, (crop_t, top_i, left_i) in enumerate(crops):
        ax = fig.add_subplot(gs[1, i])
        crop_np = to_numpy(crop_t)
        show_img(ax, frame_np,   # show original as background for crop box
                 f"slight_crop #{i+1}",
                 note=f"offset  top={top_i}px  left={left_i}px  "
                      f"(centre of {H}×{W})")
        draw_crop_box(ax, H, W, crop_frac, top_i, left_i)

    for c in range(len(crops), n_cols):
        fig.add_subplot(gs[1, c]).axis("off")

    # -- row 2: diff maps --
    diff_cases = [
        (warm_np,       "Δ warm_tint  (×4)",    "channel shift is uniform across frame"),
        (comb_np,       "Δ combined   (×4)",    "boundary artefact from resize + colour shift"),
        (to_numpy(crops[0][0]), "Δ slight_crop #1  (×4)", "edge region replaced by stretched content"),
    ]
    for c, (aug_np, title, note) in enumerate(diff_cases):
        ax = fig.add_subplot(gs[2, c])
        dm = diff_map(frame_np, aug_np, scale=4.0)
        ax.imshow(dm)
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_xlabel(f"{note}\nmean |Δ| = {dm.mean()/4:.1f} px",
                      fontsize=7.5, color="dimgrey")
        ax.axis("off")

    for c in range(3, n_cols):
        fig.add_subplot(gs[2, c]).axis("off")

    plt.savefig(args.output, dpi=130, bbox_inches="tight")
    print(f"\nSaved → {args.output}")
    print("Frame dimensions:", f"{W}×{H}")
    crop_offset_pct_h = (max_top_px  / 2) / H * 100
    crop_offset_pct_w = (max_left_px / 2) / W * 100
    print(f"Crop slack: ±{max_top_px//2}px vertical ({crop_offset_pct_h:.1f}% of height), "
          f"±{max_left_px//2}px horizontal ({crop_offset_pct_w:.1f}% of width)")
    print("With center_bias_sigma=0.4 the 1-sigma offset is "
          f"±{int(0.4 * max_top_px/2)}px vertical / "
          f"±{int(0.4 * max_left_px/2)}px horizontal — task stays well in frame.")


if __name__ == "__main__":
    main()
