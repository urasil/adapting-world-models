# converts compressed .npz feature files to plain .npy files.
# fp16 activations barely compress (~1.08x), so decompressing a whole 3gb file just to
# mmap-read a small slice wastes ~15s/example, uncompressed .npy lets np.load mmap straight
# to the rows we need, ~80x faster per-example io during training!!!!!!!!!!!!!!
# only keeps the 'features' array - everything else already lives in the index json.
# usage: python convert_features_to_npy.py --src_dir ... --dst_dir ... --workers 4
# workers limited by HPC

import argparse
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

def convert_one(args):
    src_path, dst_path, feature_key = args
    if dst_path.exists():
        return str(src_path.name), "skipped", 0.0

    try:
        t0 = time.perf_counter()
        with np.load(src_path) as data:
            if feature_key not in data:
                raise KeyError(f"{feature_key!r} not found in {src_path}")
            arr = data[feature_key]          # still compressed in memory
            np.save(dst_path, arr)           # write uncompressed .npy
        elapsed = time.perf_counter() - t0
        size_gb = dst_path.stat().st_size / 1e9
        return str(src_path.name), f"{size_gb:.2f} GB  {elapsed:.1f}s", elapsed
    except zipfile.BadZipFile:
        return str(src_path.name), "SKIPPED (bad zip)", 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir",     required=True)
    parser.add_argument("--dst_dir",     required=True)
    parser.add_argument("--feature_key", default="features")
    parser.add_argument("--workers",     type=int, default=8)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(src_dir.glob("*.npz"))
    if not src_files:
        raise FileNotFoundError(f"No .npz files found in {src_dir}")
    print(f"Found {len(src_files)} .npz files in {src_dir}")
    print(f"Output dir: {dst_dir}")
    print(f"Workers: {args.workers}")
    print()

    tasks = [(f, dst_dir / (f.stem + ".npy"), args.feature_key) for f in src_files]

    t_start = time.perf_counter()
    done = 0
    total_s = 0.0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_one, t): t[0].name for t in tasks}
        for fut in as_completed(futures):
            name, status, elapsed = fut.result()
            done += 1
            total_s += elapsed
            elapsed_wall = time.perf_counter() - t_start
            rate = done / elapsed_wall
            eta  = (len(tasks) - done) / max(rate, 1e-6)
            eta_str = f"{eta/3600:.1f}h" if eta >= 3600 else f"{eta/60:.0f}min"
            print(f"[{done:4d}/{len(tasks)}] {name}  {status}  "
                  f"(wall {elapsed_wall/60:.1f}min, ETA {eta_str})",
                  flush=True)

    wall = time.perf_counter() - t_start
    print(f"\nDone. {done} files in {wall/60:.1f} min  "
          f"(avg {total_s/max(done,1):.1f}s CPU/file)")

if __name__ == "__main__":
    main()
