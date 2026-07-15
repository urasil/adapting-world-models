# shared model loading + single-clip inference for the two representation-extraction scripts
# video decoding stays separate since they use different decoder backends (torchcodec vs pyav)
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
    # runs v-jepa 2 on one 64-frame clip, keeping all spacetime tokens -> (2048, 1024) float16
    inputs = processor(frames_tchw, return_tensors="pt").to(model.device)
    with torch.no_grad():
        features = model.get_vision_features(**inputs)  # (1, 2048, 1024)
    return features[0].cpu().to(torch.float16).numpy()

def sample_frame_indices(start_frame: int, end_frame: int, num_frames: int) -> List[int]:
    # picks num_frames indices evenly spanning [start_frame, end_frame], repeating if the span is shorter
    if end_frame <= start_frame:
        end_frame = start_frame + 1  # avoid degenerate zero-length spans
    idxs = np.linspace(start_frame, end_frame, num=num_frames).round().astype(int)
    return idxs.tolist()
