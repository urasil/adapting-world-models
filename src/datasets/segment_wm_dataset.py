"""Dataset and collate utilities for next-segment world-model pre-training."""

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from mistake_detection_dataset import HoloMistakeDataset


class _ExamplesProxy:
    """Lightweight read-only view mapping SegmentWMDataset positions to base examples.

    Allows VideoGroupedSampler and other train_mistake_detection utilities to
    call dataset.examples[i] without copying the full examples list.
    """
    __slots__ = ("_base", "_idx")

    def __init__(self, base_examples: list, indices: list[int]):
        self._base = base_examples
        self._idx  = indices

    def __getitem__(self, i: int):
        return self._base[self._idx[i]]

    def __len__(self) -> int:
        return len(self._idx)


class SegmentWMDataset(Dataset):
    """Wraps HoloMistakeDataset for world-model pre-training.

    Each item corresponds to one decision point: a variable-length context of
    S = (prefix_len - 1) segments and one target segment.

    Returns
    -------
    context_feats   : [max_context_segs, P, E]  float16  padded with zeros
    context_verb_ids: [max_context_segs]         long     padded with 0
    context_noun_ids: [max_context_segs]         long     padded with 0
    target_feats    : [P, E]     float16
    target_verb_id  : scalar     long
    target_noun_id  : scalar     long
    label           : scalar     float32  (0 = correct, 1 = mistake)
    ctx_len         : scalar     int      actual number of context segments

    Filtering
    ---------
    train_mode=True  : examples with prefix_len >= 2. (Class-balance subsampling
                        of label == 0 vs label == 1, if desired, is done by the
                        caller post-split via subsample_negatives from
                        train_mistake_detection.py — see train_segment_wm.py.)
    train_mode=False : examples with prefix_len >= 2.

    max_gap_to_target_secs (default 2.0):
        Drop examples where the last context segment ends more than this many
        seconds before the target segment starts.  Uses prefix_start_secs /
        prefix_end_secs written by enrich_index_with_actions.py.
        Set to None or inf to disable.

    target_duration_range_secs (default (0.5, 4.0)):
        Drop examples whose target segment duration falls outside this range.
        V-JEPA 2.1 feature extraction samples a fixed number of frames per
        segment regardless of duration, so segment duration controls the
        effective sampled frame rate; bounding it keeps that rate comparable
        across examples. Only the target segment is checked, not context:
        applying the same bound to every context segment compounds across
        up to max_context_segs segments and was measured to discard the
        majority of examples (>50%) for a confound that only affects ~18%
        of individual segments, vs ~20% discarded when checking the target
        alone. Context-segment duration is left unconstrained.
        Set to None to disable.

    The HoloMistakeDataset truncates long prefixes to max_segments.  We pass
    max_context_segs + 1 so that after dropping the target we have at most
    max_context_segs context segments.
    """

    def __init__(
        self,
        index_path: str | Path,
        features_dirs=None,
        train_mode: bool = True,
        max_context_segs: int = 8,
        max_gap_to_target_secs: float | None = 2.0,
        target_duration_range_secs: tuple[float, float] | None = (0.5, 4.0),
    ):
        self._base = HoloMistakeDataset(
            index_path=index_path,
            features_dir=features_dirs,
            max_segments=max_context_segs + 1,
        )

        _max_gap = float("inf") if max_gap_to_target_secs is None else max_gap_to_target_secs

        self._indices: list[int] = []
        n_gap_filtered = 0
        n_duration_filtered = 0
        for i, ex in enumerate(self._base.examples):
            if len(ex["prefix_npz_indices"]) < 2:
                continue

            # Temporal-continuity filter on the last context → target transition.
            # prefix_start_secs / prefix_end_secs are populated by the enrichment
            # script; skip the check if they're absent (e.g., legacy index files).
            if "prefix_start_secs" in ex and "prefix_end_secs" in ex:
                ends   = ex["prefix_end_secs"]
                starts = ex["prefix_start_secs"]
                if len(ends) >= 2:
                    gap = starts[-1] - ends[-2]
                    if gap > _max_gap:
                        n_gap_filtered += 1
                        continue

                if target_duration_range_secs is not None:
                    lo, hi = target_duration_range_secs
                    target_duration = ends[-1] - starts[-1]
                    if not (lo <= target_duration <= hi):
                        n_duration_filtered += 1
                        continue

            self._indices.append(i)

        if n_gap_filtered:
            print(f"SegmentWMDataset: dropped {n_gap_filtered} examples with "
                  f"last-context→target gap > {max_gap_to_target_secs}s")
        if n_duration_filtered:
            print(f"SegmentWMDataset: dropped {n_duration_filtered} examples with "
                  f"target duration outside {target_duration_range_secs}s")

        self.train_mode = train_mode
        self.max_context_segs = max_context_segs

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def examples(self):
        """Lazy indexed view of base examples in filtered order.

        VideoGroupedSampler (and other utilities from train_mistake_detection)
        access dataset.examples[i] expecting the example dict at dataset
        position i.  This property maps those positions through self._indices
        to the underlying HoloMistakeDataset.
        """
        return _ExamplesProxy(self._base.examples, self._indices)

    def __getitem__(self, i: int):
        orig = self._indices[i]
        ex = self._base.examples[orig]

        # Load features (possibly truncated to max_context_segs+1 by the base).
        feats, label = self._base[orig]   # (k, P, E) float16

        # Align action-ID lists with the same truncation the base applied.
        k = feats.shape[0]
        verb_ids = ex["prefix_verb_ids"][-k:]
        noun_ids = ex["prefix_noun_ids"][-k:]

        # Split the last segment off as the prediction target.
        context_feats = feats[:-1]                                  # [S, P, E]
        target_feats  = feats[-1]                                   # [P, E]
        ctx_verb = torch.tensor(verb_ids[:-1], dtype=torch.long)   # [S]
        ctx_noun = torch.tensor(noun_ids[:-1], dtype=torch.long)   # [S]
        tgt_verb = torch.tensor(verb_ids[-1],  dtype=torch.long)   # scalar
        tgt_noun = torch.tensor(noun_ids[-1],  dtype=torch.long)   # scalar

        ctx_len = context_feats.shape[0]

        return (context_feats, ctx_verb, ctx_noun, target_feats, tgt_verb, tgt_noun,
                label, ctx_len)


def wm_collate_fn(batch):
    """Collate variable-length context batches.

    Padding is done to the batch-local maximum context length, not the global
    max_context_segs.  At B=1 (batch_size=1) this is always a no-op: every
    example's true ctx_len equals the batch max, so ctx_lengths.min() == S and
    the model never builds a seg_valid_mask, ensuring FA2 fires unconditionally.
    """
    ctx_f, ctx_v, ctx_n, tgt_f, tgt_v, tgt_n, labels, ctx_lens = zip(*batch)
    max_len = max(t.shape[0] for t in ctx_f)

    def pad_feats(t):
        p = max_len - t.shape[0]
        if p == 0:
            return t
        return torch.cat([t, t.new_zeros(p, t.shape[1], t.shape[2])], dim=0)

    def pad_ids(t):
        p = max_len - t.shape[0]
        if p == 0:
            return t
        return torch.cat([t, t.new_zeros(p, dtype=t.dtype)], dim=0)

    return (
        torch.stack([pad_feats(t) for t in ctx_f]),           # [B, max_len, P, E]
        torch.stack([pad_ids(t)   for t in ctx_v]),           # [B, max_len]
        torch.stack([pad_ids(t)   for t in ctx_n]),           # [B, max_len]
        torch.stack(list(tgt_f)),                              # [B, P, E]
        torch.stack(list(tgt_v)),                              # [B]
        torch.stack(list(tgt_n)),                              # [B]
        torch.tensor([float(l) for l in labels]),              # [B]
        torch.tensor(list(ctx_lens), dtype=torch.long),        # [B]
    )


def compute_ctx_lens(subset, max_context_segs: int) -> list[int]:
    """ctx_len (context length, excluding target) for each position in a Subset of SegmentWMDataset."""
    full = subset.dataset
    return [
        min(len(full.examples[idx]["prefix_npz_indices"]) - 1, max_context_segs)
        for idx in subset.indices
    ]


class LengthGroupedBatchSampler(Sampler):
    """Batches examples of identical ctx_len together so wm_collate_fn never pads.

    Wraps a base per-example sampler (e.g. VideoGroupedSampler), preserving its
    iteration order as closely as possible: examples are held in small
    per-length buffers and a batch is emitted as soon as `batch_size` examples
    of the same length have accumulated. Every batch is therefore uniform-
    length, so wm_collate_fn's padding is always a no-op and the model never
    builds a seg_valid_mask — FlashAttention v2 keeps firing, same as at
    batch_size=1 (see wm_collate_fn docstring).

    With drop_last=True (default, matching the rest of this codebase), at
    most (batch_size - 1) examples per distinct ctx_len value are left in an
    incomplete buffer at epoch end and dropped — negligible relative to
    dataset size, and which specific examples are dropped varies by epoch
    since the underlying order is reshuffled by the base sampler.
    """

    def __init__(self, base_sampler, ctx_lens: list[int], batch_size: int, drop_last: bool = True):
        self.base_sampler = base_sampler
        self.ctx_lens = ctx_lens
        self.batch_size = batch_size
        self.drop_last = drop_last

    def set_epoch(self, epoch: int):
        if hasattr(self.base_sampler, "set_epoch"):
            self.base_sampler.set_epoch(epoch)

    def __iter__(self):
        buffers: dict[int, list[int]] = {}
        for idx in self.base_sampler:
            length = self.ctx_lens[idx]
            buf = buffers.setdefault(length, [])
            buf.append(idx)
            if len(buf) == self.batch_size:
                yield buf
                buffers[length] = []
        if not self.drop_last:
            for buf in buffers.values():
                if buf:
                    yield buf

    def __len__(self) -> int:
        counts = Counter(self.ctx_lens)
        n_full = sum(c // self.batch_size for c in counts.values())
        if self.drop_last:
            return n_full
        n_partial = sum(1 for c in counts.values() if c % self.batch_size)
        return n_full + n_partial
