#!/bin/bash
#SBATCH -J fix_missing_npy
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/fix_missing_npy_%j.log

# Run this AFTER deleting vjepa2_train/ to free disk space.
#
# This script:
#   1. Removes corrupt files (3 corrupt .npz, 140 corrupt 128-byte .npy stubs)
#   2. GPU-extracts 3 missing videos (z073, z195 train; z133 val) -> .npy
#   3. Converts the 140 valid val .npz -> .npy

set -euo pipefail

export WORK=/home/ua248/rds/hpc-work/adapting_world_models
export HOLOASSIST=$WORK/holoassist
export DEST=/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features
export TRAIN_NPY=$DEST/vjepa2_train_npy
export VAL_NPZ=$DEST/vjepa2_val
export VAL_NPY=$DEST/vjepa2_val_npy
export TMP=$DEST/tmp_extract_$$

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"

echo "=== Job started at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Step 1: Clean up corrupt files ────────────────────────────────────────────
echo ""
echo "--- Step 1: Removing corrupt files ---"
# Corrupt 15M val .npz (disk ran out mid-write)
rm -f "$VAL_NPZ/z133-aug-12-22-switch.npz"
# Recreate val_npy dir in case it was deleted, then remove any stub .npy files
mkdir -p "$VAL_NPY"
find "$VAL_NPY" -name "*.npy" -size -1k -delete
echo "Done"

# ── Step 2: GPU-extract 3 missing train videos ────────────────────────────────
echo ""
echo "--- Step 2: Extracting 3 missing train videos ---"
mkdir -p "$TMP"
printf "z073-july-01-22-gopro\nz195-sep-17-22-knarrevik_assemble\nz205-sep-25-22-rashult_disassemble\n" > "$TMP/missing_train.txt"

# Remove the truncated z205.npy left behind by fix_train_npy.sh
rm -f "$TRAIN_NPY/z205-sep-25-22-rashult_disassemble.npy"

cd "$WORK"
python3 -u src/vjepa2_1_segment_feature_extraction.py \
    --annotation_path "$HOLOASSIST/data-annotation-trainval-v1_1.json" \
    --video_dir       "$HOLOASSIST/videos" \
    --output_dir      "$TMP" \
    --split_file      "$TMP/missing_train.txt"

echo "--- Converting 3 train .npz -> .npy ---"
python3 - <<'PYEOF'
import os, numpy as np
from pathlib import Path

tmp       = Path(os.environ["TMP"])
train_npy = Path(os.environ["TRAIN_NPY"])

for name in ["z073-july-01-22-gopro", "z195-sep-17-22-knarrevik_assemble",
             "z205-sep-25-22-rashult_disassemble"]:
    npz = tmp / f"{name}.npz"
    npy = train_npy / f"{name}.npy"
    with np.load(npz) as d:
        feats = d["features"]
        print(f"  {name}: shape={feats.shape} dtype={feats.dtype}")
        np.save(npy, feats)
    os.remove(npz)
    print(f"  -> {npy.stat().st_size / 1e9:.2f} GB saved to train_npy/")
PYEOF

# ── Step 3: GPU-extract missing val video ─────────────────────────────────────
echo ""
echo "--- Step 3: Extracting 1 missing val video ---"
printf "z133-aug-12-22-switch\n" > "$TMP/missing_val.txt"

python3 -u src/vjepa2_1_segment_feature_extraction.py \
    --annotation_path "$HOLOASSIST/data-annotation-trainval-v1_1.json" \
    --video_dir       "$HOLOASSIST/videos" \
    --output_dir      "$VAL_NPZ" \
    --split_file      "$TMP/missing_val.txt"

echo "--- Converting z133 .npz -> .npy, verifying, then deleting .npz ---"
python3 - <<'PYEOF'
import os, sys, numpy as np, functools, operator
from pathlib import Path

def npy_is_valid(p):
    try:
        with open(p, "rb") as f:
            version = np.lib.format.read_magic(f)
            read_hdr = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                        else np.lib.format.read_array_header_2_0)
            shape, _, dtype = read_hdr(f)
            header_offset = f.tell()
    except Exception as e:
        return False, f"header unreadable: {e}"
    expected = header_offset + functools.reduce(operator.mul, shape, 1) * dtype.itemsize
    if p.stat().st_size < expected:
        return False, f"truncated: {p.stat().st_size/1e9:.2f} GB of {expected/1e9:.2f} GB"
    if shape[1:] != (32, 576, 1024):
        return False, f"wrong shape: {shape}"
    try:
        d = np.load(p, mmap_mode="r")
        _ = float(d[0, 0, 0, 0])
        _ = float(d[-1, -1, -1, -1])
    except Exception as e:
        return False, f"data unreadable: {e}"
    return True, ""

npz = Path(os.environ["VAL_NPZ"]) / "z133-aug-12-22-switch.npz"
npy = Path(os.environ["VAL_NPY"]) / "z133-aug-12-22-switch.npy"
with np.load(npz) as d:
    feats = d["features"]
    print(f"  z133: shape={feats.shape} dtype={feats.dtype}")
    np.save(npy, feats)
ok, reason = npy_is_valid(npy)
if not ok:
    print(f"  ERROR verification failed: {reason} — .npz kept")
    sys.exit(1)
npz.unlink()
print(f"  -> {npy.stat().st_size / 1e9:.2f} GB saved, .npz deleted")
PYEOF

# ── Step 4: Convert 140 remaining val .npz -> .npy, verify, delete .npz ───────
echo ""
echo "--- Step 4: Converting remaining val .npz -> .npy ---"
python3 - <<'PYEOF'
import os, sys, numpy as np, functools, operator
from pathlib import Path

def npy_is_valid(p):
    try:
        with open(p, "rb") as f:
            version = np.lib.format.read_magic(f)
            read_hdr = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                        else np.lib.format.read_array_header_2_0)
            shape, _, dtype = read_hdr(f)
            header_offset = f.tell()
    except Exception as e:
        return False, f"header unreadable: {e}"
    expected = header_offset + functools.reduce(operator.mul, shape, 1) * dtype.itemsize
    if p.stat().st_size < expected:
        return False, f"truncated: {p.stat().st_size/1e9:.2f} GB of {expected/1e9:.2f} GB"
    if shape[1:] != (32, 576, 1024):
        return False, f"wrong shape: {shape}"
    try:
        d = np.load(p, mmap_mode="r")
        _ = float(d[0, 0, 0, 0])
        _ = float(d[-1, -1, -1, -1])
    except Exception as e:
        return False, f"data unreadable: {e}"
    return True, ""

val_npz = Path(os.environ["VAL_NPZ"])
val_npy = Path(os.environ["VAL_NPY"])

errors = []
skipped = converted = 0
for npz_path in sorted(val_npz.glob("*.npz")):
    npy_path = val_npy / (npz_path.stem + ".npy")
    # Skip if already a valid .npy exists
    ok, _ = npy_is_valid(npy_path) if npy_path.exists() else (False, "")
    if ok:
        skipped += 1
        continue
    print(f"  converting {npz_path.stem}...", flush=True)
    with np.load(npz_path) as d:
        np.save(npy_path, d["features"])
    ok, reason = npy_is_valid(npy_path)
    if not ok:
        errors.append((npz_path.stem, reason))
        print(f"    ERROR: {reason} — .npz kept", flush=True)
        continue
    npz_path.unlink()
    print(f"    -> {npy_path.stat().st_size / 1e9:.2f} GB, .npz deleted", flush=True)
    converted += 1

print(f"\nDone: {converted} converted, {skipped} skipped (already valid)")
if errors:
    print(f"Errors ({len(errors)}) — .npz files kept:")
    for name, reason in errors:
        print(f"  {name}: {reason}")
    sys.exit(1)
PYEOF

# ── Cleanup ────────────────────────────────────────────────────────────────────
rm -rf "$TMP"

# ── Step 5: Validation ─────────────────────────────────────────────────────────
echo ""
echo "--- Step 5: Validation ---"
python3 - <<'PYEOF'
import os, numpy as np
from pathlib import Path

train_npy = Path(os.environ["TRAIN_NPY"])
val_npy   = Path(os.environ["VAL_NPY"])

train_files = sorted(train_npy.glob("*.npy"))
val_files   = sorted(val_npy.glob("*.npy"))

print(f"train_npy: {len(train_files)} files")
print(f"val_npy:   {len(val_files)} files  (expected 141)")

errors = 0
for label, paths in [("train", train_files[:3] + train_files[-3:]),
                     ("val",   val_files[:3] + val_files[-3:])]:
    for p in paths:
        if p.stat().st_size < 1024:
            print(f"  ERROR stub: {p}")
            errors += 1
            continue
        try:
            d = np.load(p, mmap_mode="r")
            assert d.ndim == 4 and d.shape[1] == 32 and d.shape[2] == 576 and d.shape[3] == 1024, \
                f"unexpected shape {d.shape}"
            print(f"  OK {label} {p.name}: shape={d.shape} dtype={d.dtype}")
        except Exception as e:
            print(f"  ERROR {p.name}: {e}")
            errors += 1

if errors:
    raise SystemExit(f"\n{errors} validation errors — do not run training until fixed")
print("\nAll checks passed. Ready for training.")
PYEOF

echo ""
echo "=== Job finished at $(date) ==="
