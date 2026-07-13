import json
from functools import lru_cache
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


def _clean_path_value(value) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return text


@lru_cache(maxsize=512)
def _load_npz_features(path: str, feature_key: str) -> np.ndarray:
    """Load and cache per-video feature array.

    Two formats are supported:

    .npy (uncompressed, preferred for full-token features)
        Loaded with mmap_mode='r': the OS memory-maps the file and only
        reads the pages that are actually accessed.  For a prefix of k
        segments from a video with N total segments, __getitem__ will
        touch k × (32×576×1024 × 2 bytes) ≈ 528 MB on average instead of
        the full ~3 GB file.  Sequential prefix indices (the common case)
        trigger contiguous disk reads at full RDS bandwidth (~3 GB/s),
        reducing per-example I/O from ~15 s to ~0.18 s.
        The returned object is a numpy memmap; the lru_cache keeps the file
        handle alive so the OS page cache is not prematurely evicted.
        maxsize=512: all 1466 train videos fit (512 ≥ 1466/num_workers×2).

    .npz / .npz_compressed (fallback)
        Decompresses and loads the entire video array into RAM.  For full-
        token features this takes ~15 s per file.  Suitable only for the
        pre-pooled features (11 MB/file).
    """
    if path.endswith(".npy"):
        # memory-mapped: lazy row reads; no decompression overhead.
        return np.load(path, mmap_mode="r")    # (n, t, s, d) or (n, t, d), float16

    # compressed .npz fallback — loads entire file into ram.
    with np.load(path) as data:
        if feature_key not in data:
            raise KeyError(
                f"Feature key {feature_key!r} not found in {path}. "
                f"Available keys: {list(data.keys())}"
            )
        return data[feature_key].copy()        # keep original dtype (float16)


class HoloMistakeDataset(Dataset):
    def __init__(self, index_path: str | Path, features_dir: str | Path | None = None, max_segments=None):
        """
        Args:
            index_path:   path to the .json index built by build_mistake_detection_dataset.py
            features_dir: override the features directory stored in the index
                          (useful when moving files between machines)
                          Can be a single path or a list of paths to search.
        """
        with open(index_path) as f:
            idx = json.load(f)

        # support multiple feature directories for train/val splits
        if features_dir:
            if isinstance(features_dir, (list, tuple)):
                self.features_dirs = [Path(_clean_path_value(d)) for d in features_dir]
            else:
                self.features_dirs = [Path(_clean_path_value(features_dir))]
        else:
            # try to infer from index
            if "features_dirs" in idx:
                self.features_dirs = [Path(_clean_path_value(d)) for d in idx["features_dirs"]]
            else:
                self.features_dirs = [Path(_clean_path_value(idx["features_dir"]))]
        
        self.feature_key = idx["feature_key"]
        self.examples = idx["examples"]
        self.max_segments = max_segments
        self._path_cache: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def _find_feature_file(self, video_name: str) -> str:
        """Find the feature file for a video, preferring .npy over .npz.

        Result is cached in self._path_cache so that the NFS stat syscall
        (one network round-trip per call) is paid once at first access, not
        once per example per epoch.
        """
        if video_name in self._path_cache:
            return self._path_cache[video_name]
        for fdir in self.features_dirs:
            for ext in (".npy", ".npz"):
                path = fdir / f"{video_name}{ext}"
                if path.exists():
                    self._path_cache[video_name] = str(path)
                    return str(path)
        searched = ", ".join(str(d) for d in self.features_dirs)
        raise FileNotFoundError(
            f"Missing feature file for {video_name!r}. Searched: {searched}."
        )

    def __getitem__(self, i: int):
        ex = self.examples[i]
        path = self._find_feature_file(ex['video_name'])
        # feats: (n, t, d) pre-pooled  or  (n, t, s, d) full-token — float16
        feats = _load_npz_features(path, self.feature_key)
        indices = ex["prefix_npz_indices"]
        if self.max_segments is not None and len(indices) > self.max_segments:
            indices = indices[-self.max_segments:]
        prefix = feats[indices]   # (k, t, d) or (k, t, s, d)  float16

        # full-token features: flatten spatial tokens into token dim.
        # (k, t, s, d) → (k, t*s, d); e.g. (k, 32, 576, 1024) → (k, 18432, 1024)
        if prefix.ndim == 4:
            k, T, S, D = prefix.shape
            prefix = prefix.reshape(k, T * S, D)

        # keep float16 to halve pinned-memory and h2d transfer volume.
        # the model runs under bf16 autocast so fp16 input is fine.
        return (
            torch.from_numpy(np.array(prefix, dtype=np.float16)),  # (k, tokens_per_seg, d)
            torch.tensor(ex["label"], dtype=torch.float),
        )


def collate_fn(batch):
    """
    Pad variable-length sequences to the longest in the batch.

    Returns:
        features: (B, max_k, tokens_per_seg, D) float32, zero-padded along max_k.
                  tokens_per_seg = T     (32)  for pre-pooled features.
                  tokens_per_seg = T*S (18432) for full-token features.
                  D = 1024 in both cases.
        labels:   (B,) float32  — for BCEWithLogitsLoss
        lengths:  (B,) int64   — actual sequence lengths (k per example), for
                  attention masking
    """
    features, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    _, T, D = features[0].shape
    padded = torch.zeros(len(features), lengths.max().item(), T, D, dtype=torch.float16)
    for i, f in enumerate(features):
        padded[i, : f.shape[0]] = f
    return padded, torch.stack(labels), lengths


def build_dataloader(index_path: str | Path, batch_size: int = 32, shuffle: bool = True, features_dir: str | Path | None = None, max_segments=None) -> DataLoader:
    dataset = HoloMistakeDataset(index_path, features_dir=features_dir, max_segments=max_segments)
    pin = torch.cuda.is_available()  #unsupported on mps
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, pin_memory=pin)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the dataset loader")
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--features_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    loader = build_dataloader(args.index_path, batch_size=args.batch_size,
                              shuffle=False, features_dir=args.features_dir)

    print(f"Dataset: {len(loader.dataset)} examples, {len(loader)} batches")
    for features, labels, lengths in loader:
        print(f"  features: {tuple(features.shape)}, dtype={features.dtype}")
        print(f"  labels:   {labels.tolist()}, dtype={labels.dtype}")
        print(f"  lengths:  {lengths.tolist()}")
        break

