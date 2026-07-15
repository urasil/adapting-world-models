# frame-level augmentations for V-JEPA augmentation sensitivity testing and augmented-data feature extraction.
# all functions operate on uint8 (T, C, H, W) tensors and return the same dtype/shape, so they slot
# straight into the existing feature-extraction pipeline.
# augmentations are intentionally mild -- they mimic plausible real-world variation (slightly different
# framing, indoor warm/cold lighting shift) rather than aggressive photometric distortions.
# AUG_REGISTRY_SENSITIVITY: original 3-entry registry used by the augmentation sensitivity test script.
# AUG_REGISTRY: full registry including the 4 training-data augmentation variants (warm_tint, cold_tint,
# crop_90, warm_crop), which produce ~4x more mistake examples, taking total mistakes to ~1/4 of correct.

from __future__ import annotations

import random
import torch
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Spatial: slight random crop (default: crop to 85–95 % of spatial extent)
# ---------------------------------------------------------------------------

def slight_crop(
    frames: torch.Tensor,
    crop_frac: float = 0.75,
    crop_frac_jitter: float = 0.05,
    center_bias_sigma: float = 0.4,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    # randomly crops each frame to crop_frac +/- crop_frac_jitter of the original size (same window for every frame, then resized back to (H, W)); crop offset is Gaussian-centred (sigma=center_bias_sigma x max offset) so the task area near frame centre stays visible
    if rng is None:
        rng = random.Random()

    T, C, H, W = frames.shape

    # Determine actual crop fraction with jitter
    actual_frac = crop_frac + rng.uniform(-crop_frac_jitter, crop_frac_jitter)
    actual_frac = max(0.70, min(1.0, actual_frac))  # safety clamp

    crop_h = int(H * actual_frac)
    crop_w = int(W * actual_frac)

    # Maximum distance the crop top-left corner can move from its centred position
    max_top  = H - crop_h   # total slack in height
    max_left = W - crop_w   # total slack in width

    # Centre-biased offset using a Gaussian.
    # Centre crop would place top = max_top // 2, left = max_left // 2.
    # We perturb by N(0, sigma * max/2) and clamp to [0, max].
    def _gaussian_offset(max_val: int) -> int:
        if max_val == 0:
            return 0
        sigma = center_bias_sigma * (max_val / 2.0)
        centre = max_val / 2.0
        offset = rng.gauss(centre, sigma)
        return int(max(0, min(max_val, round(offset))))

    top  = _gaussian_offset(max_top)
    left = _gaussian_offset(max_left)

    # Crop — same window for every frame in the clip
    cropped = frames[:, :, top:top + crop_h, left:left + crop_w]  # (T, C, crop_h, crop_w)

    # Resize back to (H, W) using bilinear interpolation
    # F.interpolate needs float; convert, resize, convert back
    cropped_f = cropped.float()
    resized_f = F.interpolate(
        cropped_f, size=(H, W), mode="bilinear", align_corners=False
    )
    return resized_f.to(torch.uint8)


# ---------------------------------------------------------------------------
# Colour: warm tint (boost red, slight green boost, reduce blue)
# ---------------------------------------------------------------------------

def warm_tint(
    frames: torch.Tensor,
    r_factor: float = 1.18,
    g_factor: float = 1.04,
    b_factor: float = 0.90,
) -> torch.Tensor:
    # warm-light colour shift: scales RGB channels independently and clamps to [0, 255]; defaults mimic warm (~3200K) tungsten lighting (R+18%, G+4%, B-10%)
    frames_f = frames.float()  # avoid uint8 overflow during scaling
    factors = torch.tensor([r_factor, g_factor, b_factor],
                           dtype=torch.float32)  # (3,)

    # Broadcast (T, C, H, W) × (C,) → expand factors to (1, C, 1, 1)
    factors = factors.view(1, 3, 1, 1)
    tinted  = (frames_f * factors).clamp(0, 255)
    return tinted.to(torch.uint8)


# ---------------------------------------------------------------------------
# Colour: cold tint (reduce red, slight green dip, boost blue)
# ---------------------------------------------------------------------------

def cold_tint(
    frames: torch.Tensor,
    r_factor: float = 0.85,
    g_factor: float = 0.97,
    b_factor: float = 1.15,
) -> torch.Tensor:
    # cool-light colour shift, the complement of warm_tint; mimics cool (~6500K) daylight (R-15%, G-3%, B+15%), matched in magnitude so neither augmentation dominates in L2 feature distance
    frames_f = frames.float()
    factors = torch.tensor([r_factor, g_factor, b_factor],
                           dtype=torch.float32).view(1, 3, 1, 1)
    tinted = (frames_f * factors).clamp(0, 255)
    return tinted.to(torch.uint8)


# ---------------------------------------------------------------------------
# Combined augmentations (crop + tint)
# ---------------------------------------------------------------------------

def combined_aug(
    frames: torch.Tensor,
    crop_frac: float = 0.70,
    crop_frac_jitter: float = 0.05,
    center_bias_sigma: float = 0.4,
    r_factor: float = 1.18,
    g_factor: float = 1.04,
    b_factor: float = 0.90,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    # applies slight_crop followed by warm_tint in one call
    frames = slight_crop(frames, crop_frac=crop_frac,
                         crop_frac_jitter=crop_frac_jitter,
                         center_bias_sigma=center_bias_sigma, rng=rng)
    frames = warm_tint(frames, r_factor=r_factor,
                       g_factor=g_factor, b_factor=b_factor)
    return frames


def warm_crop(
    frames: torch.Tensor,
    crop_frac: float = 0.90,
    crop_frac_jitter: float = 0.03,
    center_bias_sigma: float = 0.4,
    r_factor: float = 1.18,
    g_factor: float = 1.04,
    b_factor: float = 0.90,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    # 90% centre-biased crop followed by warm tint; mild enough that V-JEPA 2.1 features stay highly similar to originals (pooled cosine sim ~0.997, per augmentation sensitivity analysis) -- one of the 4 training-data variants used to upsample the minority mistake class
    frames = slight_crop(frames, crop_frac=crop_frac,
                         crop_frac_jitter=crop_frac_jitter,
                         center_bias_sigma=center_bias_sigma, rng=rng)
    frames = warm_tint(frames, r_factor=r_factor,
                       g_factor=g_factor, b_factor=b_factor)
    return frames


def cold_crop(
    frames: torch.Tensor,
    crop_frac: float = 0.90,
    crop_frac_jitter: float = 0.03,
    center_bias_sigma: float = 0.4,
    r_factor: float = 0.85,
    g_factor: float = 0.97,
    b_factor: float = 1.15,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    # 90% centre-biased crop followed by cold tint
    frames = slight_crop(frames, crop_frac=crop_frac,
                         crop_frac_jitter=crop_frac_jitter,
                         center_bias_sigma=center_bias_sigma, rng=rng)
    frames = cold_tint(frames, r_factor=r_factor,
                       g_factor=g_factor, b_factor=b_factor)
    return frames


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

# Original registry — used by the augmentation sensitivity test script.
AUG_REGISTRY_SENSITIVITY = {
    "slight_crop": {
        "fn": slight_crop,
        "kwargs": {"crop_frac": 0.70, "crop_frac_jitter": 0.05, "center_bias_sigma": 0.4},
        "description": "Gaussian-centred crop (≈70 % of frame), resized back to original",
    },
    "warm_tint": {
        "fn": warm_tint,
        "kwargs": {"r_factor": 1.18, "g_factor": 1.04, "b_factor": 0.90},
        "description": "Warm tint (R×1.18, G×1.04, B×0.90) — mimics warm indoor lighting",
    },
    "combined": {
        "fn": combined_aug,
        "kwargs": {
            "crop_frac": 0.70, "crop_frac_jitter": 0.05, "center_bias_sigma": 0.4,
            "r_factor": 1.18, "g_factor": 1.04, "b_factor": 0.90,
        },
        "description": "Gaussian-centred crop (70 %) + warm tint combined",
    },
}

# Full registry — includes the 4 training-data augmentation variants.
# These 4 variants are applied to ALL segments (correct + mistake) per video,
# producing augmented .npz files used to upsample the minority mistake class
# to ~1/4 of the correct class during training.
#
#   n_correct  = 118,058   n_mistake_orig = 5,996
#   target     = 118,058 / 4 = 29,515
#   with 4 aug variants: 5,996 × 5 = 29,980 ≈ target  ✓
AUG_REGISTRY = {
    # Sensitivity-test entries (kept for backward compatibility)
    "slight_crop": AUG_REGISTRY_SENSITIVITY["slight_crop"],
    "combined":    AUG_REGISTRY_SENSITIVITY["combined"],

    # Training-data augmentation variants
    "warm_tint": {
        "fn": warm_tint,
        "kwargs": {"r_factor": 1.18, "g_factor": 1.04, "b_factor": 0.90},
        "description": "Warm tint (R×1.18, G×1.04, B×0.90) — mimics warm indoor lighting",
    },
    "cold_tint": {
        "fn": cold_tint,
        "kwargs": {"r_factor": 0.85, "g_factor": 0.97, "b_factor": 1.15},
        "description": "Cold tint (R×0.85, G×0.97, B×1.15) — mimics cool daylight / overcast",
    },
    "crop_90": {
        "fn": slight_crop,
        "kwargs": {"crop_frac": 0.90, "crop_frac_jitter": 0.03, "center_bias_sigma": 0.4},
        "description": "Gaussian-centred 90 % crop with ±3 % jitter, resized back to original",
    },
    "warm_crop": {
        "fn": warm_crop,
        "kwargs": {
            "crop_frac": 0.90, "crop_frac_jitter": 0.03, "center_bias_sigma": 0.4,
            "r_factor": 1.18, "g_factor": 1.04, "b_factor": 0.90,
        },
        "description": "90 % crop + warm tint — combines spatial and colour augmentation",
    },
    "cold_crop": {
        "fn": cold_crop,
        "kwargs": {
            "crop_frac": 0.90, "crop_frac_jitter": 0.03, "center_bias_sigma": 0.4,
            "r_factor": 0.85, "g_factor": 0.97, "b_factor": 1.15,
        },
        "description": "90 % crop + cold tint — combines spatial and colour augmentation",
    },
}

# The 4 canonical training augmentation names (in order).
TRAIN_AUG_NAMES = ["warm_tint", "cold_tint", "crop_90", "warm_crop"]
