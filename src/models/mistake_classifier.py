import torch
import torch.nn as nn
from src.models.attentive_probe import AttentivePooler


class MistakeClassifier(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=16, depth=4, max_segments=64, tokens_per_segment=32):
        super().__init__()
        self.tokens_per_segment = tokens_per_segment
        self.max_segments = max_segments

        # Learnable per-segment embedding, added to every token in segment i
        self.segment_embed = nn.Parameter(torch.zeros(1, max_segments, 1, embed_dim))
        nn.init.trunc_normal_(self.segment_embed, std=0.02)

        self.input_norm = nn.LayerNorm(embed_dim) # can remove this
        self.pooler = AttentivePooler(num_queries=1, embed_dim=embed_dim, num_heads=num_heads, depth=depth, complete_block=True)
        self.head = nn.Linear(embed_dim, 1)  # single logit for BCEWithLogitsLoss

    def forward(self, features, lengths):
        """
        features: (B, max_k, 32, D) float32, zero-padded along max_k
        lengths:  (B,) int64, real k per example
        returns:  (B,) float logits
        """
        B, max_k, T, D = features.shape
        assert T == self.tokens_per_segment

        # Add segment embeddings (broadcast over the 32 tokens of each segment)
        x = features + self.segment_embed[:, :max_k]      # (B, max_k, T, D)
        x = self.input_norm(x)

        # Flatten segments and intra-segment positions into one token sequence
        x = x.reshape(B, max_k * T, D)                    # (B, max_k*T, D)

        # Build key-padding mask: True = padded position
        positions_per_example = lengths * T               # (B,)
        arange = torch.arange(max_k * T, device=x.device)
        key_padding_mask = arange[None, :] >= positions_per_example[:, None]

        pooled = self.pooler(x, key_padding_mask=key_padding_mask)  # (B, 1, D)
        return self.head(pooled.squeeze(1)).squeeze(-1)             # (B,)
