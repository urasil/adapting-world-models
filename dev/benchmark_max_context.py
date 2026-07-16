# finds the max number of context segments that fit on one gpu.
# attention is segment-by-segment sdpa (flashattention v2), so memory scales o(s*p*d) not o((s*p)^2).
# prints: s  n_segs  n_tokens  peak_gb  headroom_gb

from __future__ import annotations

import gc
import sys
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.models.segment_predictor import SegmentMaskPredictorAC

CFG = dict(
    img_size=(384, 384),
    patch_size=16,
    num_frames=64,
    tubelet_size=2,
    embed_dim=1024,
    predictor_embed_dim=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    verb_vocab=[f"verb{i}" for i in range(47)],
    noun_vocab=[f"noun{i}" for i in range(158)],
    normalize_targets=True,
    use_activation_checkpointing=True,
)

T_SEG = 32
H = W = 24
P = T_SEG * H * W   # 18 432

def mem_gb(device):
    torch.cuda.synchronize(device)
    return torch.cuda.memory_reserved(device) / 1e9

def one_pass(model: nn.Module, S: int, device: torch.device) -> float:
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    ctx_f = torch.randn(1, S, P, 1024, device=device, dtype=torch.float32)
    ctx_v = torch.zeros(1, S, device=device, dtype=torch.long)
    ctx_n = torch.zeros(1, S, device=device, dtype=torch.long)
    tgt_v = torch.zeros(1,    device=device, dtype=torch.long)
    tgt_n = torch.zeros(1,    device=device, dtype=torch.long)
    tgt_f = torch.randn(1, P, 1024, device=device, dtype=torch.float32)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(ctx_f, ctx_v, ctx_n, tgt_v, tgt_n)

    loss = F.l1_loss(model.normalise_targets(pred.float()), model.normalise_targets(tgt_f.view(1, T_SEG, H * W, 1024)))
    loss.backward()
    model.zero_grad(set_to_none=True)

    del ctx_f, ctx_v, ctx_n, tgt_v, tgt_n, tgt_f, pred, loss
    gc.collect()
    torch.cuda.empty_cache()

    return torch.cuda.max_memory_reserved(device) / 1e9

def main():
    if not torch.cuda.is_available():
        print("No CUDA GPU found.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"GPU: {gpu_name}  ({total_gb:.0f} GB)")

    model = SegmentMaskPredictorAC(**CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    param_fp32 = sum(p.numel() * 4 for p in model.parameters()) / 1e9
    print(f"Model: {n_params:.0f}M params  ({param_fp32:.2f} GB fp32)")
    print(f"  reserved after load: {mem_gb(device):.2f} GB")
    print()

    print(f"{'S':>4}  {'n_segs':>6}  {'N_tokens':>9}  {'peak_GB':>8}  {'headroom_GB':>12}")
    print("-" * 52)

    for S in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]:
        n_segs = S + 1
        N = n_segs * (2 + P)
        try:
            peak_gb = one_pass(model, S, device)
            headroom = total_gb - peak_gb
            ok = "✓" if headroom > 2 else "~"
            print(f"{S:>4}  {n_segs:>6}  {N:>9}  {peak_gb:>7.2f}G  {headroom:>11.1f}G  {ok}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{S:>4}  {n_segs:>6}  {N:>9}  {'OOM':>8}", flush=True)
            break
        except Exception as e:
            print(f"{S:>4}  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
            break

    print("\nDone.")

if __name__ == "__main__":
    main()
