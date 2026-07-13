"""Shared between vjepa2_representations.py and
extract_representations_action_segment_based.py: loading facebook/vjepa2-vitl-fpc64-256
and running it on a single 64-frame clip. Video decoding is deliberately NOT
shared here since the two scripts use different decoder backends (torchcodec
vs PyAV)."""
from typing import List

import numpy as np
import torch
from transformers import AutoVideoProcessor, AutoModel

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
NUM_FRAMES = 64


def load_model(device: str):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"Loading model {MODEL_ID} on {device} with dtype {dtype}")
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = (AutoModel.from_pretrained(MODEL_ID, dtype=dtype, attn_implementation="sdpa").to(device).eval())
    return processor, model


def extract_one_clip(processor, model, frames_tchw: torch.Tensor) -> np.ndarray:
    """Run V-JEPA 2 on one 64-frame clip. Returns (2048, 1024) float16.

    All spacetime tokens are kept: 8 temporal positions (64 frames / tubelet-size-8)
    x 16x16 spatial patches = 2048 tokens, each of dimension 1024 (ViT-L).
    """
    inputs = processor(frames_tchw, return_tensors="pt").to(model.device)
    with torch.no_grad():
        features = model.get_vision_features(**inputs)  # (1, 2048, 1024)
    return features[0].cpu().to(torch.float16).numpy()


def sample_frame_indices(start_frame: int, end_frame: int, num_frames: int) -> List[int]:
    """
    Pick exactly num_frames indices spanning [start_frame, end_frame] inclusive.
    Uniform sampling — the same recipe the V-JEPA 2 paper / HF processor docs use
    for variable-length clips. If the span is shorter than num_frames, indices
    will repeat (linspace + round handles this naturally), which is the standard
    pad-by-repeat behaviour for short clips.
    """
    if end_frame <= start_frame:
        end_frame = start_frame + 1  # avoid degenerate zero-length spans
    idxs = np.linspace(start_frame, end_frame, num=num_frames).round().astype(int)
    return idxs.tolist()
