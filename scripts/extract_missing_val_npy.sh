#!/bin/bash
#SBATCH -J extract_missing_val_npy
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/extract_missing_val_npy_%j.log

set -euo pipefail

WORK=/home/ua248/rds/hpc-work/adapting_world_models
HOLOASSIST=$WORK/holoassist
VAL_NPY=/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features/vjepa2_val_npy
VAL_SPLIT=$HOLOASSIST/splits/val-v1_2.txt
MISSING_LIST=$WORK/tmp_missing_val_$$.txt

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"

echo "=== Job started at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Step 1: Compute missing list ───────────────────────────────────────────────
echo ""
echo "--- Step 1: Computing missing val videos ---"
python3 - <<PYEOF
from pathlib import Path

expected = [l.strip() for l in open("$VAL_SPLIT") if l.strip()]
have     = {p.stem for p in Path("$VAL_NPY").glob("*.npy")}
missing  = [v for v in expected if v not in have]

print(f"Val split:    {len(expected)} videos")
print(f"Already have: {len(have)} .npy files")
print(f"Missing:      {len(missing)} videos")

with open("$MISSING_LIST", "w") as f:
    f.write("\n".join(missing) + "\n")
PYEOF

n_missing=$(wc -l < "$MISSING_LIST")
if [ "$n_missing" -eq 0 ]; then
    echo "Nothing to do — all val videos already present."
    rm -f "$MISSING_LIST"
    exit 0
fi
echo "Extracting $n_missing videos..."

# ── Step 2: Extract directly to .npy ──────────────────────────────────────────
echo ""
echo "--- Step 2: Extracting features ---"
cd "$WORK"
python3 -u src/vjepa2_1_segment_feature_extraction.py \
    --annotation_path "$HOLOASSIST/data-annotation-trainval-v1_1.json" \
    --video_dir       "$HOLOASSIST/videos" \
    --output_dir      "$VAL_NPY" \
    --split_file      "$MISSING_LIST"

rm -f "$MISSING_LIST"

# ── Step 3: Validate all 207 val videos present and loadable ───────────────────
echo ""
echo "--- Step 3: Validation ---"
python3 - <<PYEOF
import sys, numpy as np
from pathlib import Path

expected = [l.strip() for l in open("$VAL_SPLIT") if l.strip()]
val_npy  = Path("$VAL_NPY")
have     = {p.stem for p in val_npy.glob("*.npy")}
missing  = [v for v in expected if v not in have]

print(f"val_npy: {len(have)} files (expected {len(expected)})")

errors = 0
if missing:
    print(f"Still missing ({len(missing)}):")
    for v in missing:
        print(f"  {v}")
    errors += len(missing)

files = sorted(val_npy.glob("*.npy"))
for p in files[:5] + files[-5:]:
    try:
        d = np.load(p, mmap_mode="r")
        assert d.ndim == 4 and d.shape[1:] == (32, 576, 1024), f"bad shape {d.shape}"
        print(f"  OK {p.name}: shape={d.shape} dtype={d.dtype}")
    except Exception as e:
        print(f"  ERROR {p.name}: {e}")
        errors += 1

if errors:
    sys.exit(f"{errors} validation error(s) — do not run training.")
print(f"\nAll {len(expected)} val videos present and valid. Ready for training.")
PYEOF

echo ""
echo "GPU peak memory: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1/1024 " GB"}')"
echo "=== Job finished at $(date) ==="
