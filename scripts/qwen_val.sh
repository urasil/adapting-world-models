#!/bin/bash
#SBATCH -J qwen_val
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=logs/qwen_val_%j.log
set -euo pipefail

# Usage:
#   sbatch qwen_val.sh                          # defaults to actions_with_labels
#   sbatch qwen_val.sh none                     # no prior context
#   sbatch qwen_val.sh actions                  # Flavour A
#   sbatch qwen_val.sh actions_with_labels      # Flavour B (oracle priors)
PRIOR_MODE="${1:-actions_with_labels}"

WORK=/home/ua248/rds/hpc-work/adapting_world_models
HOLOASSIST="$WORK/holoassist"
VAL_LIST="$HOLOASSIST/splits/val-v1_2.txt"
OUT_DIR="$WORK/results/qwen_val/${PRIOR_MODE}"
OUT_FILE="$OUT_DIR/predictions.jsonl"
mkdir -p "$OUT_DIR"

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"
export HF_HOME="$WORK/hf_cache"

echo "=== Job started at $(date) ==="
echo "Prior mode:  $PRIOR_MODE"
echo "Output file: $OUT_FILE"

python3 -u "$WORK/qwen_val.py" \
    --annotations "$HOLOASSIST/data-annotation-trainval-v1_1.json" \
    --video-dir   "$HOLOASSIST/videos" \
    --val-list    "$VAL_LIST" \
    --out         "$OUT_FILE" \
    --prior-mode  "$PRIOR_MODE"

echo "=== Job finished at $(date) ==="
