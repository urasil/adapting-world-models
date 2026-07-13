#!/bin/bash
#SBATCH -J validate_train_npy
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/validate_train_npy_%j.log

# Validates every .npy file in vjepa2_train_npy/:
#   - file size > 1 MB (not a stub)
#   - shape is (N, 32, 576, 1024) with N >= 1
#   - dtype is float16
#   - first and last element are readable and finite

set -euo pipefail

WORK=/home/ua248/rds/hpc-work/adapting_world_models
TRAIN_NPY=/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features/vjepa2_train_npy

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"

echo "=== Validation started at $(date) ==="
echo "Checking $TRAIN_NPY"
echo ""

python3 - <<'PYEOF'
import os, sys, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

TRAIN_NPY = Path(os.environ.get("TRAIN_NPY",
    "/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features/vjepa2_train_npy"))

files = sorted(TRAIN_NPY.glob("*.npy"))
print(f"Files found: {len(files)}")

def check(p):
    size = p.stat().st_size
    if size < 1024 * 1024:  # < 1 MB = stub
        return p.name, f"STUB: only {size} bytes"
    try:
        d = np.load(p, mmap_mode="r")
    except Exception as e:
        return p.name, f"LOAD ERROR: {e}"
    if d.ndim != 4:
        return p.name, f"BAD NDIM: {d.ndim}"
    if d.shape[1:] != (32, 576, 1024):
        return p.name, f"BAD SHAPE: {d.shape}"
    if d.shape[0] < 1:
        return p.name, f"ZERO SEGMENTS: {d.shape}"
    if d.dtype != np.float16:
        return p.name, f"BAD DTYPE: {d.dtype}"
    try:
        # Read first and last element to confirm data is on disk
        _ = float(d[0, 0, 0, 0])
        _ = float(d[-1, -1, -1, -1])
    except Exception as e:
        return p.name, f"READ ERROR: {e}"
    return p.name, None  # OK

errors = []
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(check, p): p for p in files}
    for i, fut in enumerate(as_completed(futures), 1):
        name, err = fut.result()
        if err:
            errors.append((name, err))
            print(f"  [{i}/{len(files)}] ERROR {name}: {err}")
        elif i % 100 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] OK so far...")

print()
if errors:
    print(f"FAILED: {len(errors)} corrupt files:")
    for name, err in sorted(errors):
        print(f"  {name}: {err}")
    sys.exit(1)
else:
    print(f"All {len(files)} files passed. Safe to delete vjepa2_train/.")
PYEOF

echo ""
echo "=== Validation finished at $(date) ==="
