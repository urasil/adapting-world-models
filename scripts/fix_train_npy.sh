#!/bin/bash
#SBATCH -J fix_train_npy
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/fix_train_npy_%j.log

# Fixes 972 truncated .npy files in vjepa2_train_npy/ by converting from their
# valid .npz counterparts in vjepa2_train/.
#
# For each file the order is:
#   1. Convert .npz -> .npy
#   2. Verify the .npy (size, shape, dtype, first+last element readable)
#   3. ONLY if verification passes: delete the .npz
#   If verification fails the .npz is kept and the error is reported.
#
# Space strategy (disk is at 100%):
#   First, delete the .npz files whose .npy is already confirmed good (~1.3T freed).
#   Then process bad files one at a time, deleting each corrupt .npy before writing
#   its replacement so the truncated data doesn't consume space during conversion.
#
# Do NOT delete vjepa2_train/ before running this.

set -euo pipefail

export WORK=/home/ua248/rds/hpc-work/adapting_world_models
export NPZ_DIR=/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features/vjepa2_train
export NPY_DIR=/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features/vjepa2_train_npy

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo ""

python3 -u - <<'PYEOF'
import os, sys, numpy as np, zipfile, functools, operator
from pathlib import Path

NPZ_DIR = Path(os.environ["NPZ_DIR"])
NPY_DIR = Path(os.environ["NPY_DIR"])


def npy_is_valid(p: Path) -> tuple[bool, str]:
    """
    Returns (True, "") if the .npy file is fully written and readable,
    or (False, reason) if it is a stub, truncated, or otherwise broken.
    """
    try:
        with open(p, "rb") as f:
            version = np.lib.format.read_magic(f)
            read_hdr = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                        else np.lib.format.read_array_header_2_0)
            shape, _, dtype = read_hdr(f)
            header_offset = f.tell()
    except Exception as e:
        return False, f"header unreadable: {e}"

    expected_size = header_offset + functools.reduce(operator.mul, shape, 1) * dtype.itemsize
    actual_size = p.stat().st_size
    if actual_size < expected_size:
        return False, f"truncated: {actual_size/1e9:.2f} GB of expected {expected_size/1e9:.2f} GB"

    if shape[1:] != (32, 576, 1024):
        return False, f"wrong shape: {shape}"
    if dtype != np.dtype("float16"):
        return False, f"wrong dtype: {dtype}"

    # Actually read the first and last element to confirm data is on disk
    try:
        d = np.load(p, mmap_mode="r")
        _ = float(d[0, 0, 0, 0])
        _ = float(d[-1, -1, -1, -1])
        if not (np.isfinite(d[0, 0, 0, 0]) or True):  # NaN/inf are still "readable"
            pass
    except Exception as e:
        return False, f"data unreadable: {e}"

    return True, ""


# ── Classify existing .npy files ──────────────────────────────────────────────
print("Classifying .npy files (header + size check)...", flush=True)
good, bad = [], []
for p in sorted(NPY_DIR.glob("*.npy")):
    ok, _ = npy_is_valid(p)
    if ok:
        good.append(p.stem)
    else:
        bad.append(p.stem)

print(f"  Already good: {len(good)}")
print(f"  Need fixing:  {len(bad)}")


# ── Step 1: Free space by deleting .npz files for already-good .npy ──────────
print(f"\n--- Step 1: Removing {len(good)} unneeded .npz files to free space ---",
      flush=True)
freed = 0
for name in good:
    npz = NPZ_DIR / f"{name}.npz"
    if npz.exists():
        freed += npz.stat().st_size
        npz.unlink()
print(f"  Freed {freed / 1e12:.2f} TB", flush=True)


# ── Step 2: Convert each bad .npy, verify, then delete .npz ──────────────────
print(f"\n--- Step 2: Converting {len(bad)} truncated .npy files ---", flush=True)
converted = 0
errors = []

for i, name in enumerate(bad, 1):
    npz_path = NPZ_DIR / f"{name}.npz"
    npy_path = NPY_DIR / f"{name}.npy"
    prefix = f"  [{i}/{len(bad)}] {name}"

    # Check the .npz is usable before doing anything
    if not npz_path.exists():
        errors.append((name, "no .npz — needs GPU extraction"))
        print(f"{prefix}: SKIP (no .npz)", flush=True)
        continue
    try:
        with zipfile.ZipFile(npz_path):
            pass
    except Exception as e:
        errors.append((name, f"corrupt .npz: {e}"))
        print(f"{prefix}: SKIP (corrupt .npz: {e})", flush=True)
        continue

    # Delete the truncated .npy now so its space is available for the new file
    if npy_path.exists():
        npy_path.unlink()

    # Convert
    try:
        with np.load(npz_path) as d:
            feats = d["features"]          # (N, 32, 576, 1024) float16
            np.save(npy_path, feats)
    except Exception as e:
        errors.append((name, f"conversion failed: {e}"))
        print(f"{prefix}: ERROR during conversion: {e}", flush=True)
        continue

    # Verify the new .npy before touching the .npz
    ok, reason = npy_is_valid(npy_path)
    if not ok:
        errors.append((name, f"verification failed: {reason}"))
        print(f"{prefix}: ERROR verification failed ({reason}) — .npz kept", flush=True)
        continue

    # Verification passed — safe to delete the .npz
    npz_path.unlink()
    converted += 1
    print(f"{prefix}: OK  shape={feats.shape}  "
          f"{npy_path.stat().st_size/1e9:.2f} GB", flush=True)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print(f"Converted and verified: {converted}/{len(bad)}", flush=True)
if errors:
    print(f"Errors ({len(errors)}) — .npz files kept for these:")
    for name, reason in errors:
        print(f"  {name}: {reason}")
    sys.exit(1)
else:
    remaining = list(NPZ_DIR.glob("*.npz"))
    print(f"All conversions successful.")
    print(f"Remaining .npz files in vjepa2_train/: {len(remaining)}")
    if remaining:
        print("  (these have corrupt .npz and need GPU extraction):")
        for p in remaining:
            print(f"  {p.name}")
    else:
        print("  vjepa2_train/ is now empty and safe to delete.")
PYEOF

echo ""
echo "=== Job finished at $(date) ==="
