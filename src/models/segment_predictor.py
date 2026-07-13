# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extensions copyright (c) University of Cambridge.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.ac_predictor import VisionTransformerPredictorAC
from src.utils.modules import rotate_queries_or_keys
from src.utils.tensors import trunc_normal_


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _separate_positions(pos_ids: torch.Tensor, H: int, W: int):
    """Decompose flat patch-space IDs into (temporal, height, width) coordinates."""
    HW = H * W
    frame_ids = pos_ids // HW
    rem = pos_ids - HW * frame_ids
    height_ids = rem // W
    width_ids = rem - W * height_ids
    return frame_ids.float(), height_ids.float(), width_ids.float()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SegmentMaskPredictorAC(VisionTransformerPredictorAC):
    """Single-pass masked-prediction head for next-segment world-model pre-training.

    Architecture
    ============
    Context segments 0..S-1 and a query segment S are assembled into a tensor
    of shape [B, n_segs, tokens_per_seg, D].  Each segment contributes
    COND_TOKENS=2 conditioning tokens (verb + noun embedding) followed by
    T_seg * H * W patch tokens.  For context segments the patch tokens are
    projected encoder features; for the query segment they are mask tokens.

    Attention is segment-block-causal: tokens in segment i attend to all
    tokens in segments 0..i and nothing later.  This is implemented as n_segs
    separate SDPA calls per transformer layer, each with a growing key-value
    context window.  SDPA dispatches to FlashAttention v2 (O(N_q) memory).

    Positional encoding
    ===================
    3D RoPE (time × height × width) with a global monotonic temporal axis:
    segment s, frame f → temporal index s * T_seg + f.
    Conditioning tokens for segment s receive temporal index s * T_seg and
    spatial index (0, 0), making their spatial RoPE component the identity —
    matching the ACRoPEAttention convention for action tokens.

    Key differences from VisionTransformerPredictorAC
    ==================================================
    * action_encoder / state_encoder / extrinsics_encoder removed.
    * verb_embedding + noun_embedding replace them (two tokens per segment).
    * Learnable mask_token for the query block.
    * Segment-by-segment SDPA (FlashAttention v2) instead of a dense or
      FlexAttention mask.  No N×N materialisation at any S.
    * forward() is fully overridden; parent forward() is never called.

    checkpoint_min_segs:
        Activation checkpointing trades ~30-40% extra compute (recomputing
        each block's forward during backward) for lower peak memory.  Shorter
        examples (small n_segs = ctx_len + 1) need much less memory and may
        not need checkpointing at all.  If set, checkpointing is only applied
        when n_segs >= checkpoint_min_segs for that example; shorter examples
        skip it and pay no recompute overhead.  None (default) checkpoints
        every example whenever use_activation_checkpointing=True, matching
        the previous behaviour.
    """

    COND_TOKENS = 2  # verb token + noun token per segment

    def __init__(
        self,
        img_size=(384, 384),
        patch_size=16,
        num_frames=64,
        tubelet_size=2,
        embed_dim=1024,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        use_rope=True,
        n_verbs: int = 47,
        n_nouns: int = 158,
        max_context_segs: int = 8,
        normalize_targets: bool = True,
        checkpoint_min_segs: int | None = None,
        **kwargs,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_std=init_std,
            uniform_power=uniform_power,
            use_silu=use_silu,
            wide_silu=wide_silu,
            is_frame_causal=False,
            use_activation_checkpointing=use_activation_checkpointing,
            use_rope=True,
            action_embed_dim=7,   # placeholder so parent doesn't error
            use_extrinsics=False,
            **kwargs,
        )

        del self.action_encoder
        del self.state_encoder
        del self.extrinsics_encoder

        self.verb_embedding = nn.Embedding(n_verbs, predictor_embed_dim)
        self.noun_embedding = nn.Embedding(n_nouns, predictor_embed_dim)
        trunc_normal_(self.verb_embedding.weight, std=init_std)
        trunc_normal_(self.noun_embedding.weight, std=init_std)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        self.normalize_targets = normalize_targets
        if normalize_targets:
            self.target_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

        self._predictor_embed_dim = predictor_embed_dim
        self.max_context_segs = max_context_segs
        self.checkpoint_min_segs = checkpoint_min_segs
        self._T_seg = num_frames // tubelet_size   # 32

        # RoPE per-axis head-dim split — must match ACRoPEAttention exactly.
        head_dim = predictor_embed_dim // num_heads
        self._d_dim = int(2 * ((head_dim // 3) // 2))
        self._h_dim = int(2 * ((head_dim // 3) // 2))
        self._w_dim = int(2 * ((head_dim // 3) // 2))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pos_ids(self, n_segs: int, device: torch.device) -> torch.Tensor:
        """Build global position IDs for all tokens in an n_segs-segment sequence.

        Layout per segment s (0-indexed):
            [verb_cond, noun_cond, frame_0_patch_0, ..., frame_{T-1}_patch_{P-1}]

        Conditioning tokens for segment s get pos_id = s * T * H * W, which
        decodes to temporal = s*T, h = 0, w = 0 (spatial RoPE is identity).
        """
        T = self._T_seg
        H = self.grid_height
        W = self.grid_width
        P = T * H * W   # 18 432

        seg_idx   = torch.arange(n_segs, device=device)
        seg_offset = seg_idx * P                                       # [n_segs]

        cond_ids  = seg_offset.unsqueeze(1).expand(-1, self.COND_TOKENS)  # [n_segs, 2]
        local_patch = torch.arange(P, device=device)
        patch_ids = seg_offset.unsqueeze(1) + local_patch.unsqueeze(0)    # [n_segs, P]

        ids = torch.cat([cond_ids, patch_ids], dim=1)   # [n_segs, C+P]
        return ids.flatten()                             # [n_segs * (C+P)]

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_ids: torch.Tensor,
        grid_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply 3D RoPE to Q and K, replicating ACRoPEAttention's convention."""
        B, num_heads, N, head_dim = q.shape
        H, W = self.grid_height, self.grid_width

        d_pos, h_pos, w_pos = _separate_positions(pos_ids, H, W)
        h_pos = h_pos * (grid_size / H)
        w_pos = w_pos * (grid_size / W)

        def _expand(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0).unsqueeze(0).expand(B, num_heads, N).to(q.dtype)

        d_pos, h_pos, w_pos = _expand(d_pos), _expand(h_pos), _expand(w_pos)

        s = 0
        qd = rotate_queries_or_keys(q[..., s:s + self._d_dim], pos=d_pos)
        kd = rotate_queries_or_keys(k[..., s:s + self._d_dim], pos=d_pos)
        s += self._d_dim
        qh = rotate_queries_or_keys(q[..., s:s + self._h_dim], pos=h_pos)
        kh = rotate_queries_or_keys(k[..., s:s + self._h_dim], pos=h_pos)
        s += self._h_dim
        qw = rotate_queries_or_keys(q[..., s:s + self._w_dim], pos=w_pos)
        kw = rotate_queries_or_keys(k[..., s:s + self._w_dim], pos=w_pos)
        s += self._w_dim

        if s < head_dim:
            q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
            k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)
        return q, k

    def _run_block(
        self,
        blk,
        x: torch.Tensor,                           # [B, n_segs, tps, D]
        pos_ids: torch.Tensor,
        seg_valid_mask: torch.Tensor | None = None,  # [B, n_segs] bool; None = all valid
    ) -> torch.Tensor:
        """Run one ACBlock with segment-block-causal SDPA (FlashAttention v2).

        Attention pattern: tokens in segment i attend to ALL tokens from
        segments 0..i (past and present), but not from segments i+1..n_segs-1.
        This is implemented by calling SDPA once per segment with an
        incrementally growing key-value context — no N×N materialisation.

        SDPA dispatches to FlashAttention v2 when dtype=bf16/fp16, head_dim∈
        {32,64,128,256}, and no additive mask is passed, giving O(N_q) memory.

        When seg_valid_mask is provided (batch contains padded segments), an
        additive key mask is built per SDPA call so padded key positions receive
        -inf before softmax, zeroing their contribution.  SDPA then falls back
        to the memory-efficient attention backend (still O(N_q), not O(N²)).
        """
        B, n_segs, tps, D = x.shape
        N = n_segs * tps
        num_heads = blk.attn.num_heads
        head_dim  = D // num_heads

        # ---- Norm + QKV (flat sequence for one GEMM) --------------------
        y = blk.norm1(x.reshape(B, N, D))                        # [B, N, D]
        qkv = (
            blk.attn.qkv(y)
            .unflatten(-1, (3, num_heads, head_dim))
            .permute(2, 0, 3, 1, 4)
        )                                                          # [3, B, H, N, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]                        # each [B, H, N, hd]
        q, k = self._apply_rope(q, k, pos_ids, blk.attn.grid_size)

        #Segment causal SDPA: [B, H, n_segs, tps, hd]
        q_s = q.unflatten(2, (n_segs, tps))
        k_s = k.unflatten(2, (n_segs, tps))
        v_s = v.unflatten(2, (n_segs, tps))

        # For segment i: Q_i attends to K_{0..i}, V_{0..i}.
        # Each SDPA call uses the FlashAttention v2 backend (O(N_q) memory).
        seg_outs = []
        for i in range(n_segs):
            q_i  = q_s[:, :, i].contiguous()                # [B, H, tps, hd]
            k_kv = k_s[:, :, :i + 1].flatten(2, 3).contiguous()  # [B, H, (i+1)*tps, hd]
            v_kv = v_s[:, :, :i + 1].flatten(2, 3).contiguous()
            if seg_valid_mask is not None:
                # Build additive key mask: 0 where valid, -inf where padded.
                # [B, i+1] → expand to token level → [B, 1, 1, (i+1)*tps]
                seg_v = seg_valid_mask[:, :i + 1]            # [B, i+1]
                kv_valid = (
                    seg_v.unsqueeze(-1)
                    .expand(-1, -1, tps)
                    .reshape(B, 1, 1, (i + 1) * tps)
                )
                attn_mask = kv_valid.to(q_i.dtype).masked_fill(~kv_valid, float("-inf")).masked_fill(kv_valid, 0.0)
                seg_outs.append(F.scaled_dot_product_attention(q_i, k_kv, v_kv, attn_mask=attn_mask))
            else:
                seg_outs.append(F.scaled_dot_product_attention(q_i, k_kv, v_kv))

        # Reassemble: [B, H, n_segs, tps, hd] → [B, N, D]
        attn_out = (
            torch.stack(seg_outs, dim=2)
            .flatten(2, 3)
            .transpose(1, 2)
            .reshape(B, N, D)
        )
        attn_out = blk.attn.proj(attn_out)
        attn_out = blk.attn.proj_drop(attn_out)

        x_flat = x.reshape(B, N, D) + blk.drop_path(attn_out)

        # ---- MLP --------------------------------------------------------
        x_flat = x_flat + blk.drop_path(blk.mlp(blk.norm2(x_flat)))

        return x_flat.unflatten(1, (n_segs, tps))             # [B, n_segs, tps, D]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        context_feats: torch.Tensor,              # [B, S, P, E]
        context_verb_ids: torch.Tensor,           # [B, S]
        context_noun_ids: torch.Tensor,           # [B, S]
        target_verb_id: torch.Tensor,             # [B]
        target_noun_id: torch.Tensor,             # [B]
        ctx_lengths: torch.Tensor | None = None,  # [B] int: true S per example (< S if padded)
    ) -> torch.Tensor:                            # [B, T_seg, H*W, E]
        B, S, P, _ = context_feats.shape
        D = self._predictor_embed_dim
        C = self.COND_TOKENS
        tps    = C + P #tokens per segment
        n_segs = S + 1

        # ---- Project context features to predictor dimension ------------
        x_ctx = self.predictor_embed(context_feats)#[B, S, P, D]

        v_ctx = self.verb_embedding(context_verb_ids)#[B, S, D]
        n_ctx = self.noun_embedding(context_noun_ids)#[B, S, D]
        v_qry = self.verb_embedding(target_verb_id)#[B, D]
        n_qry = self.noun_embedding(target_noun_id)#[B, D]

        #[B, n_segs, tps, D]
        cond_ctx = torch.stack([v_ctx, n_ctx], dim=2)#[B, S, 2, D]
        ctx_segs = torch.cat([cond_ctx, x_ctx], dim=2)#[B, S, tps, D]

        cond_qry = torch.stack([v_qry, n_qry], dim=1)#[B, 2, D]
        mask_toks = self.mask_token.expand(B, P, D)#[B, P, D]
        qry_seg  = torch.cat([cond_qry, mask_toks], dim=1)#[B, tps, D]

        x = torch.cat([ctx_segs, qry_seg.unsqueeze(1)], dim=1)#[B, n_segs, tps, D]

        # ---- Position IDs (computed once, shared across blocks) ---------
        pos_ids = self._build_pos_ids(n_segs, device=x.device)

        # ---- Segment validity mask (only needed when batch has padding) --
        # seg_valid_mask[b, j] = True if segment j is real for example b.
        # Context segments at positions >= ctx_lengths[b] are padding.
        seg_valid_mask = None
        if ctx_lengths is not None and int(ctx_lengths.min()) < S:
            seg_valid_mask = torch.zeros(B, n_segs, dtype=torch.bool, device=x.device)
            for b in range(B):
                seg_valid_mask[b, :ctx_lengths[b]] = True
            seg_valid_mask[:, -1] = True   #query segment always valid

        # ---- Predictor blocks -------------------------------------------
        do_checkpoint = self.use_activation_checkpointing and (
            self.checkpoint_min_segs is None or n_segs >= self.checkpoint_min_segs
        )
        for blk in self.predictor_blocks:
            if do_checkpoint:
                #checkpoint only needs to track the gradient-carrying tensor x!
                x = torch.utils.checkpoint.checkpoint(
                    lambda _x, _b=blk, _m=seg_valid_mask: self._run_block(_b, _x, pos_ids, _m), x, use_reentrant=False)
            else:
                x = self._run_block(blk, x, pos_ids, seg_valid_mask)

        # ---- Read out query patch tokens --------------------------------
        x_query = x[:, -1, C:, :]#[B, P, D]
        x_query = self.predictor_norm(x_query)
        x_query = self.predictor_proj(x_query)#[B, P, E]

        return x_query.view(B, self._T_seg, self.grid_height * self.grid_width, -1)

    def normalise_targets(self, feats: torch.Tensor) -> torch.Tensor:
        if self.normalize_targets:
            return self.target_norm(feats)
        return feats


def segment_mask_predictor_ac(**kwargs):
    model = SegmentMaskPredictorAC(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
