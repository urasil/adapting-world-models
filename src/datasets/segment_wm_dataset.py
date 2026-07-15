# dataset and collate utilities for next-segment world-model pre-training

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from mistake_detection_dataset import HoloMistakeDataset

class _ExamplesProxy:
    # read-only view mapping our positions to base examples, so callers can index without copying the list
    __slots__ = ("_base", "_idx")

    def __init__(self, base_examples: list, indices: list[int]):
        self._base = base_examples
        self._idx  = indices

    def __getitem__(self, i: int):
        return self._base[self._idx[i]]

    def __len__(self) -> int:
        return len(self._idx)

class SegmentWMDataset(Dataset):
    # wraps HoloMistakeDataset for next-segment prediction: context = S prior segments, target = the last one
    # keeps examples with prefix_len >= 2, drops ones with a big context-to-target gap or an odd-length target
    # (only the target's duration is checked, since bounding every context segment discards way more examples)

    def __init__(self, index_path: str | Path, features_dirs=None, train_mode: bool = True,
                 max_context_segs: int = 8, max_gap_to_target_secs: float | None = 2.0,
                 target_duration_range_secs: tuple[float, float] | None = (0.5, 4.0)):
        self._base = HoloMistakeDataset(index_path=index_path, features_dir=features_dirs, max_segments=max_context_segs + 1)

        _max_gap = float("inf") if max_gap_to_target_secs is None else max_gap_to_target_secs

        self._indices: list[int] = []
        n_gap_filtered = 0
        n_duration_filtered = 0
        for i, ex in enumerate(self._base.examples):
            if len(ex["prefix_npz_indices"]) < 2:
                continue

            # skip the gap/duration check if prefix_start_secs / prefix_end_secs are missing (legacy index files)
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
        # lazy view of base examples in our filtered order, for callers that expect dataset.examples[i]
        return _ExamplesProxy(self._base.examples, self._indices)

    def __getitem__(self, i: int):
        orig = self._indices[i]
        ex = self._base.examples[orig]

        # base may have truncated the prefix to max_context_segs+1
        feats, label = self._base[orig]   # (k, P, E) float16

        # match the same truncation on the verb/noun id lists
        k = feats.shape[0]
        verb_ids = ex["prefix_verb_ids"][-k:]
        noun_ids = ex["prefix_noun_ids"][-k:]

        # last segment is the prediction target, the rest is context
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
    # pads context to the batch's max length, not the global max, so a uniform-length batch is always a no-op
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
    # context length (excluding target) for each position in a Subset of SegmentWMDataset
    full = subset.dataset
    return [min(len(full.examples[idx]["prefix_npz_indices"]) - 1, max_context_segs) for idx in subset.indices]

class LengthGroupedBatchSampler(Sampler):
    # batches examples of the same ctx_len together so wm_collate_fn never has to pad
    # wraps a base sampler, buffering examples per length until batch_size accumulates
    # with drop_last, up to (batch_size - 1) examples per length are left over each epoch and dropped

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
