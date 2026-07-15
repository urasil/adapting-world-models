#!/bin/bash
#SBATCH -J build_mistake_dataset_val
#SBATCH -A MLMI-ua248-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=build_mistake_dataset_val_%j.log
set -euo pipefail

WORK=/home/ua248/rds/hpc-work/adapting_world_models
HOLOASSIST=$WORK/holoassist
FEATURES=$WORK/features/vjepa2_val
OUT_DIR=$WORK/datasets

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"

mkdir -p "$OUT_DIR"

echo "=== Job started at $(date) ==="
echo "Features dir: $FEATURES"

n_meta=$(find "$FEATURES" -name "*_meta.npz" 2>/dev/null | wc -l)
echo "Metadata files in features dir: $n_meta"

python3 -u "$WORK/data_prep/build_mistake_detection_dataset.py" \
    --annotation_path "$HOLOASSIST/data-annotation-trainval-v1_1.json" \
    --features_dir    "$FEATURES" \
    --output_path     "$OUT_DIR/mistake_detection_index_val.json"

echo ""
echo "=== Build finished at $(date) ==="

python3 -u - <<'EOF'
import json, sys
path = "/home/ua248/rds/hpc-work/adapting_world_models/datasets/mistake_detection_index_val.json"
idx = json.load(open(path))
n = idx["num_examples"]
if n == 0:
    print("ERROR: 0 examples produced")
    sys.exit(1)
labels = [e["label"] for e in idx["examples"]]
correct = labels.count(0)
mistake = labels.count(1)
print(f"Total examples : {n}")
print(f"  Correct (0)  : {correct}  ({100*correct/n:.1f}%)")
print(f"  Mistake (1)  : {mistake}  ({100*mistake/n:.1f}%)")
EOF
