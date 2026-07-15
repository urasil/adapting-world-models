# pools full-token features (t,32,576,1024) down to (t,32,1024) via mean over spatial tokens
# ~576x smaller on disk, much cheaper i/o and gpu compute during training
# usage: python convert_features_spatialpool.py [--split train|val|both]

import argparse
import os
import time
from pathlib import Path

import numpy as np

SPLITS = {
    "train": (
        "/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features/vjepa2_train_npy",
        "/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features/vjepa2_train_npy_spatpool",
    ),
    "val": (
        "/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features/vjepa2_val_npy",
        "/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features/vjepa2_val_npy_spatpool",
    ),
}

def convert_split(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.npy"))
    print(f"  {len(files)} files: {in_dir} -> {out_dir}")

    t_start = time.time()
    for i, src in enumerate(files):
        dst = out_dir / src.name
        if dst.exists():
            continue  # resume-safe: skip already converted

        x = np.load(src, mmap_mode="r")          # (t, 32, 576, 1024)
        assert x.ndim == 4 and x.shape[2] == 576, f"unexpected shape {x.shape} in {src.name}"
        pooled = x.mean(axis=2).astype(np.float16)  # (t, 32, 1024)
        np.save(dst, pooled)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (len(files) - i - 1) / rate
            print(f"    [{i+1}/{len(files)}]  {rate:.1f} files/s  "
                  f"ETA {remaining/60:.1f} min", flush=True)

    print(f"  Done in {(time.time()-t_start)/60:.1f} min")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    args = parser.parse_args()

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        in_dir, out_dir = SPLITS[split]
        print(f"\n=== {split} ===")
        convert_split(Path(in_dir), Path(out_dir))

if __name__ == "__main__":
    main()
