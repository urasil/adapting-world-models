#!/bin/bash
#SBATCH -J qwen-full-v2-direct
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=runs/qwen_reasoning_full_v2/slurm-%j.log

set -euo pipefail

REPO_ROOT="/home/ua248/rds/hpc-work/adapting_world_models"
ANN="${REPO_ROOT}/holoassist/data-annotation-trainval-v1_1.json"
VIDEO_DIR="${REPO_ROOT}/holoassist/videos"
VAL_LIST="${REPO_ROOT}/holoassist/splits/val-v1_2.txt"
OUT_ROOT="${REPO_ROOT}/runs/qwen_reasoning_full_v2/qwen_val"
PRIOR_MODE="actions_with_labels"

cd "$REPO_ROOT"

module purge
module load python/3.11.0-icl
source "$REPO_ROOT/venv_ampere/bin/activate"
export HF_HOME="$REPO_ROOT/hf_cache"

mkdir -p "$OUT_ROOT/direct"

echo "=== Job started at $(date) ==="

python3 -u "$REPO_ROOT/qwen_val.py" \
    --annotations "$ANN" \
    --video-dir "$VIDEO_DIR" \
    --val-list "$VAL_LIST" \
    --out "$OUT_ROOT/direct/results.jsonl" \
    --prior-mode "$PRIOR_MODE" \
    --reasoning-mode direct

echo "=== Job finished at $(date) ==="
