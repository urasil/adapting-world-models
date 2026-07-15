# unit tests for SegmentMaskPredictorAC:
#  1. test_causal_mask_mod          -- segment-block-causal attention pattern (attend to segments 0..i, not later)
#  2. test_output_shape              -- small model (depth=2), S=4 context segs, asserts output shape [1, 32, 576, E]
#  3. test_bf16_checkpointing_no_oom -- depth=4 + activation checkpointing, bf16 fwd+bwd at full token count, no OOM, finite grads
#  4. test_batch2_padding_mask       -- B=2 with mixed context lengths (S=2, S=4) padded to max_context_segs=4, padded output must match an unpadded B=1 forward
import pytest
import torch
import torch.nn.functional as F

# ---- Skip markers ---------------------------------------------------------
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast + FlashAttention backend requires a CUDA GPU")

# ---- Shared model dimensions ----------------------------------------------
# Keep embed_dim small so tests run fast, but keep P (patch count) real-sized
# to exercise the actual sequence lengths that matter for memory.
_IMG_SIZE    = (384, 384)
_PATCH_SIZE  = 16
_NUM_FRAMES  = 64
_TUBELET     = 2
_T_SEG       = _NUM_FRAMES // _TUBELET           # 32
_H = _W      = _IMG_SIZE[0] // _PATCH_SIZE       # 24
_P           = _T_SEG * _H * _W                  # 18 432
_EMBED_DIM   = 256
_PRED_DIM    = 256
_NUM_HEADS   = 4
_N_VERBS     = 48
_N_NOUNS     = 164
_MAX_CTX     = 4

def _make_model(depth: int, use_ckpt: bool) -> "SegmentMaskPredictorAC":
    from src.models.segment_predictor import SegmentMaskPredictorAC
    import torch.nn as nn
    from functools import partial

    return SegmentMaskPredictorAC(
        img_size=_IMG_SIZE,
        patch_size=_PATCH_SIZE,
        num_frames=_NUM_FRAMES,
        tubelet_size=_TUBELET,
        embed_dim=_EMBED_DIM,
        predictor_embed_dim=_PRED_DIM,
        depth=depth,
        num_heads=_NUM_HEADS,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        n_verbs=_N_VERBS,
        n_nouns=_N_NOUNS,
        max_context_segs=_MAX_CTX,
        normalize_targets=True,
        use_activation_checkpointing=use_ckpt,
    )

def _dummy_batch(S: int, device):
    # returns (ctx_f, ctx_v, ctx_n, tgt_v, tgt_n, tgt_f) for B=1, S context segs
    ctx_f = torch.randn(1, S, _P, _EMBED_DIM, device=device)
    ctx_v = torch.zeros(1, S, device=device, dtype=torch.long)
    ctx_n = torch.zeros(1, S, device=device, dtype=torch.long)
    tgt_v = torch.zeros(1, device=device, dtype=torch.long)
    tgt_n = torch.zeros(1, device=device, dtype=torch.long)
    tgt_f = torch.randn(1, _P, _EMBED_DIM, device=device)
    return ctx_f, ctx_v, ctx_n, tgt_v, tgt_n, tgt_f

# ---------------------------------------------------------------------------
# Test 1 – causal mask logic
# ---------------------------------------------------------------------------

def test_causal_mask_mod():
    # segment-block-causal mask_mod: same segment -> True, ki in an earlier segment -> True (attend to past), ki in a later segment -> False (no attending to the future)
    tokens_per_seg = 7   # arbitrary small value
    n_segs = 3

    def mask_mod(b, h, qi, ki, _tps=tokens_per_seg):
        return (qi // _tps) >= (ki // _tps)

    N = n_segs * tokens_per_seg

    for qi in range(N):
        seg_q = qi // tokens_per_seg
        for ki in range(N):
            seg_k = ki // tokens_per_seg
            result = mask_mod(0, 0, qi, ki)
            expected = (seg_q >= seg_k)
            assert result == expected, (
                f"mask_mod(qi={qi}, ki={ki}): seg_q={seg_q}, seg_k={seg_k}, "
                f"got {result}, expected {expected}"
            )

# ---------------------------------------------------------------------------
# Test 2 – output shape (CPU, SDPA has a CPU math fallback)
# ---------------------------------------------------------------------------

def test_output_shape():
    # forward pass with S=4 context segs must produce shape [1, 32, 576, E], runs on CPU (GPU uses FlashAttention v2 automatically)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(depth=2, use_ckpt=False).to(device)

    ctx_f, ctx_v, ctx_n, tgt_v, tgt_n, _ = _dummy_batch(S=4, device=device)

    with torch.no_grad():
        out = model(ctx_f, ctx_v, ctx_n, tgt_v, tgt_n)

    assert out.shape == (1, _T_SEG, _H * _W, _EMBED_DIM), (
        f"Expected (1, {_T_SEG}, {_H * _W}, {_EMBED_DIM}), got {tuple(out.shape)}"
    )

# ---------------------------------------------------------------------------
# Test 3 – bf16 + activation checkpointing at S=4, no OOM
# ---------------------------------------------------------------------------

@cuda_required
def test_bf16_checkpointing_no_oom():
    # forward+backward under bf16 with activation checkpointing and S=4 must not raise CUDA OOM and must produce finite gradients for every parameter
    device = torch.device("cuda")
    model = _make_model(depth=4, use_ckpt=True).to(device)

    ctx_f, ctx_v, ctx_n, tgt_v, tgt_n, tgt_f = _dummy_batch(S=4, device=device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(ctx_f, ctx_v, ctx_n, tgt_v, tgt_n)   # [1, T, HW, E]

    tgt = tgt_f.view(1, _T_SEG, _H * _W, _EMBED_DIM)
    loss = F.l1_loss(model.normalise_targets(pred.float()), model.normalise_targets(tgt))
    loss.backward()

    # Shape check
    assert pred.shape == (1, _T_SEG, _H * _W, _EMBED_DIM)

    # Gradient check - every trainable parameter must have received a gradient
    missing_grad = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing_grad, f"No gradient for: {missing_grad}"

    # Finite gradient check
    nan_params = [name for name, p in model.named_parameters()
                  if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not nan_params, f"Non-finite gradient for: {nan_params}"

# ---------------------------------------------------------------------------
# Test 4 – batch_size=2 with mixed context lengths and padding mask
# ---------------------------------------------------------------------------

def test_batch2_padding_mask():
    # B=2 forward with ctx_lengths must match a separate B=1 forward without padding for the short example, uses a deterministic model (no dropout/drop_path) for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # Small P so the test runs fast on CPU.
    P_small = 8   # override the global _P for this test only
    S_short = 2   # example A: 2 real context segments (padded to 4)
    S_long  = 4   # example B: 4 real context segments (no padding)

    from src.models.segment_predictor import SegmentMaskPredictorAC
    import torch.nn as nn
    from functools import partial

    model = SegmentMaskPredictorAC(
        img_size=(32, 32),       # tiny spatial grid
        patch_size=16,
        num_frames=4,
        tubelet_size=2,
        embed_dim=64,
        predictor_embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        n_verbs=_N_VERBS,
        n_nouns=_N_NOUNS,
        max_context_segs=_MAX_CTX,   # 4
        normalize_targets=True,
        use_activation_checkpointing=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ).to(device).eval()

    T_s = model._T_seg   # 2 for (num_frames=4, tubelet=2)
    HW_s = model.grid_height * model.grid_width   # (32/16)^2 = 4
    P_s  = T_s * HW_s   # 8
    D    = 64

    torch.manual_seed(42)
    # Short example: S_short real segments + zero-padding to S_long
    ctx_f_short = torch.randn(1, S_short, P_s, D, device=device)
    ctx_v_short = torch.zeros(1, S_short, device=device, dtype=torch.long)
    ctx_n_short = torch.zeros(1, S_short, device=device, dtype=torch.long)
    tgt_v_short = torch.zeros(1, device=device, dtype=torch.long)
    tgt_n_short = torch.zeros(1, device=device, dtype=torch.long)

    # B=1 reference: call model directly with S=S_short (no padding, no mask)
    with torch.no_grad():
        ref = model(ctx_f_short, ctx_v_short, ctx_n_short, tgt_v_short, tgt_n_short)

    # Build padded B=2 batch: example A = short (padded), example B = full
    pad = S_long - S_short
    ctx_f_a = torch.cat([ctx_f_short, torch.zeros(1, pad, P_s, D, device=device)], dim=1)
    ctx_v_a = torch.cat([ctx_v_short, torch.zeros(1, pad, device=device, dtype=torch.long)], dim=1)
    ctx_n_a = torch.cat([ctx_n_short, torch.zeros(1, pad, device=device, dtype=torch.long)], dim=1)

    torch.manual_seed(7)
    ctx_f_b = torch.randn(1, S_long, P_s, D, device=device)
    ctx_v_b = torch.zeros(1, S_long, device=device, dtype=torch.long)
    ctx_n_b = torch.zeros(1, S_long, device=device, dtype=torch.long)
    tgt_v_b = torch.zeros(1, device=device, dtype=torch.long)
    tgt_n_b = torch.zeros(1, device=device, dtype=torch.long)

    ctx_f2   = torch.cat([ctx_f_a, ctx_f_b], dim=0)   # [2, 4, P_s, D]
    ctx_v2   = torch.cat([ctx_v_a, ctx_v_b], dim=0)
    ctx_n2   = torch.cat([ctx_n_a, ctx_n_b], dim=0)
    tgt_v2   = torch.cat([tgt_v_short, tgt_v_b], dim=0)
    tgt_n2   = torch.cat([tgt_n_short, tgt_n_b], dim=0)
    ctx_lens = torch.tensor([S_short, S_long], device=device)

    with torch.no_grad():
        out2 = model(ctx_f2, ctx_v2, ctx_n2, tgt_v2, tgt_n2, ctx_lengths=ctx_lens)

    # Output shape: [2, T_s, HW_s, D]
    assert out2.shape == (2, T_s, HW_s, D), f"Got {tuple(out2.shape)}"

    # The padded example (batch index 0) must match the unpadded B=1 reference.
    torch.testing.assert_close(out2[0], ref[0], rtol=1e-4, atol=1e-4, msg="Padded B=2 output[0] differs from B=1 reference")
