from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.models.ac_predictor import VisionTransformerPredictorAC
from src.utils.modules import rotate_queries_or_keys
from src.utils.tensors import trunc_normal_


# Helpers

def _separate_positions(pos_ids: torch.Tensor, H: int, W: int):
    """Decompose flat patch-space IDs into (temporal, height, width) coordinates."""
    HW = H * W
    frame_ids = pos_ids // HW
    rem = pos_ids - HW * frame_ids
    height_ids = rem // W
    width_ids = rem - W * height_ids
    return frame_ids.float(), height_ids.float(), width_ids.float()


def _text_vocab_embeddings(words: list[str], model_name: str) -> torch.Tensor:
    """Encodes each vocab word/phrase with a frozen pretrained text encoder, taking the CLS token's final hidden state. Underscore-joined labels (e.g. "joy_con_controller")
    are turned into spaces first since the encoder's tokenizer is trained on natural text, not identifiers. Returns [len(words), hidden_dim] on CPU, the caller registers
    it as a buffer, so it moves with the model's .to(device)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).eval()

    texts = [w.replace("_", " ") for w in words]
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        hidden = encoder(**batch).last_hidden_state  # [n, L, H]

    return hidden[:, 0, :]  # [n, H] CLS token


# Main class

class SegmentMaskPredictorAC(VisionTransformerPredictorAC):
    """Single-pass masked-prediction head for next-segment world-model pre-training.

    Context segments 0..S-1 and a query segment S are packed into one tensor
    of shape [B, n_segs, tokens_per_seg, D]. Each segment is COND_TOKENS=2
    conditioning tokens (verb + noun embedding) followed by T_seg*H*W patch
    tokens — projected encoder features for context segments, mask tokens for
    the query segment.

    Attention is segment-causal: segment i attends to all tokens in segments
    0..i and nothing after. We run this as one SDPA call per segment per
    layer with a growing key/value window, so FlashAttention v2 handles it in
    O(N_q) memory instead of materialising an NxN mask.

    Positional encoding is 3D RoPE (time, height, width) on a single global
    temporal axis: segment s, frame f -> temporal index s*T_seg + f.
    Conditioning tokens use temporal index s*T_seg and spatial index (0, 0),
    so their spatial RoPE component is the identity — matching how
    ACRoPEAttention treats action tokens.

    Differences from VisionTransformerPredictorAC:
    - action_encoder / state_encoder / extrinsics_encoder are gone; verb/noun
      conditioning takes their place, via a frozen pretrained text encoder
      (CLS-token embedding per vocab word, cached as a buffer at
      construction time) plus a small trainable verb_proj/noun_proj Linear
      into predictor_embed_dim. Verb and noun stay as two separate
      conditioning tokens, each independently encoded — no separator needed.
    - Adds a learnable mask_token for the query segment.
    - forward() is fully overridden and never calls the parent's forward().

    checkpoint_min_segs: activation checkpointing saves memory but costs
    ~30-40% more compute (recomputing each block on the backward pass). Short
    examples don't need it. If set, only examples with
    n_segs >= checkpoint_min_segs get checkpointed; shorter ones skip it.
    Leave as None to checkpoint everything whenever
    use_activation_checkpointing=True.
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
        verb_vocab: list[str] = None,
        noun_vocab: list[str] = None,
        text_encoder_name: str = "bert-base-uncased",
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

        if verb_vocab is None or noun_vocab is None:
            raise ValueError("verb_vocab and noun_vocab are required (lists of vocab words, ordered by id)")

        verb_text_embed = _text_vocab_embeddings(verb_vocab, text_encoder_name)
        noun_text_embed = _text_vocab_embeddings(noun_vocab, text_encoder_name)
        self.register_buffer("verb_text_embed", verb_text_embed)  # [n_verbs, H], frozen
        self.register_buffer("noun_text_embed", noun_text_embed)  # [n_nouns, H], frozen

        text_dim = verb_text_embed.shape[-1]
        self.verb_proj = nn.Linear(text_dim, predictor_embed_dim)
        self.noun_proj = nn.Linear(text_dim, predictor_embed_dim)
        for proj in (self.verb_proj, self.noun_proj):
            trunc_normal_(proj.weight, std=init_std)
            nn.init.constant_(proj.bias, 0)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        self.normalize_targets = normalize_targets
        if normalize_targets:
            self.target_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

        self._predictor_embed_dim = predictor_embed_dim
        self.max_context_segs = max_context_segs
        self.checkpoint_min_segs = checkpoint_min_segs
        self._T_seg = num_frames // tubelet_size   # 32

        # RoPE per-axis head-dim split, must match ACRoPEAttention exactly.
        head_dim = predictor_embed_dim // num_heads
        self._d_dim = int(2 * ((head_dim // 3) // 2))
        self._h_dim = int(2 * ((head_dim // 3) // 2))
        self._w_dim = int(2 * ((head_dim // 3) // 2))

    # Internal helpers

    def _build_pos_ids(self, n_segs: int, device: torch.device) -> torch.Tensor:
        """Global position IDs for every token in an n_segs-segment sequence.

        Each segment s is laid out as [verb, noun, patch_0, ..., patch_{P-1}].
        The conditioning tokens get pos_id = s*T*H*W, which decodes to
        temporal = s*T, h = 0, w = 0 — spatial RoPE is the identity for them.
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
        """Run one block with segment-causal attention (FlashAttention v2).

        Segment i attends to all tokens in segments 0..i, never anything
        later. We call SDPA once per segment with an incrementally growing
        key/value range, so there's no NxN attention matrix.

        If seg_valid_mask is given (batch has padded segments), we add a
        -inf mask over the padded key positions before softmax. That bumps
        SDPA onto the memory-efficient backend instead of FlashAttention, but
        it's still O(N_q), not O(N²).
        """
        B, n_segs, tps, D = x.shape
        N = n_segs * tps
        num_heads = blk.attn.num_heads
        head_dim  = D // num_heads

        # Norm + QKV, computed for the flat sequence in one GEMM
        y = blk.norm1(x.reshape(B, N, D))                        # [B, N, D]
        qkv = (
            blk.attn.qkv(y)
            .unflatten(-1, (3, num_heads, head_dim))
            .permute(2, 0, 3, 1, 4)
        )                                                          # [3, B, H, N, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]                        # each [B, H, N, hd]
        q, k = self._apply_rope(q, k, pos_ids, blk.attn.grid_size)

        # Split into segments: [B, H, n_segs, tps, hd]
        q_s = q.unflatten(2, (n_segs, tps))
        k_s = k.unflatten(2, (n_segs, tps))
        v_s = v.unflatten(2, (n_segs, tps))

        # Segment i's queries attend to keys/values from segments 0..i.
        seg_outs = []
        for i in range(n_segs):
            q_i  = q_s[:, :, i].contiguous()                # [B, H, tps, hd]
            k_kv = k_s[:, :, :i + 1].flatten(2, 3).contiguous()  # [B, H, (i+1)*tps, hd]
            v_kv = v_s[:, :, :i + 1].flatten(2, 3).contiguous()
            if seg_valid_mask is not None:
                # Additive key mask: 0 where valid, -inf where padded.
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

        # Reassemble segments back into the flat sequence: [B, N, D]
        attn_out = (
            torch.stack(seg_outs, dim=2)
            .flatten(2, 3)
            .transpose(1, 2)
            .reshape(B, N, D)
        )
        attn_out = blk.attn.proj(attn_out)
        attn_out = blk.attn.proj_drop(attn_out)

        x_flat = x.reshape(B, N, D) + blk.drop_path(attn_out)

        x_flat = x_flat + blk.drop_path(blk.mlp(blk.norm2(x_flat)))

        return x_flat.unflatten(1, (n_segs, tps))             # [B, n_segs, tps, D]

    # Forward

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
        tps    = C + P  # tokens per segment
        n_segs = S + 1

        # Project context features to the predictor's embedding dim
        x_ctx = self.predictor_embed(context_feats)  # [B, S, P, D]

        v_ctx = self.verb_proj(self.verb_text_embed[context_verb_ids])  # [B, S, D]
        n_ctx = self.noun_proj(self.noun_text_embed[context_noun_ids])  # [B, S, D]
        v_qry = self.verb_proj(self.verb_text_embed[target_verb_id])  # [B, D]
        n_qry = self.noun_proj(self.noun_text_embed[target_noun_id])  # [B, D]

        cond_ctx = torch.stack([v_ctx, n_ctx], dim=2)  # [B, S, 2, D]
        ctx_segs = torch.cat([cond_ctx, x_ctx], dim=2)  # [B, S, tps, D]

        cond_qry = torch.stack([v_qry, n_qry], dim=1)  # [B, 2, D]
        mask_toks = self.mask_token.expand(B, P, D)  # [B, P, D]
        qry_seg  = torch.cat([cond_qry, mask_toks], dim=1)  # [B, tps, D]

        x = torch.cat([ctx_segs, qry_seg.unsqueeze(1)], dim=1)  # [B, n_segs, tps, D]

        # Position IDs are the same for every block, so build them once
        pos_ids = self._build_pos_ids(n_segs, device=x.device)

        # Only needed when the batch has padded context segments.
        # seg_valid_mask[b, j] = True if segment j is real for example b;
        # context segments at positions >= ctx_lengths[b] are padding.
        seg_valid_mask = None
        if ctx_lengths is not None and int(ctx_lengths.min()) < S:
            seg_valid_mask = torch.zeros(B, n_segs, dtype=torch.bool, device=x.device)
            for b in range(B):
                seg_valid_mask[b, :ctx_lengths[b]] = True
            seg_valid_mask[:, -1] = True  # query segment is always valid

        do_checkpoint = self.use_activation_checkpointing and (
            self.checkpoint_min_segs is None or n_segs >= self.checkpoint_min_segs
        )
        for blk in self.predictor_blocks:
            if do_checkpoint:
                # Only x carries gradients, so that's all checkpoint needs to track
                x = torch.utils.checkpoint.checkpoint(
                    lambda _x, _b=blk, _m=seg_valid_mask: self._run_block(_b, _x, pos_ids, _m), x, use_reentrant=False)
            else:
                x = self._run_block(blk, x, pos_ids, seg_valid_mask)

        # Read out the query segment's patch tokens
        x_query = x[:, -1, C:, :]  # [B, P, D]
        x_query = self.predictor_norm(x_query)
        x_query = self.predictor_proj(x_query)  # [B, P, E]

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
