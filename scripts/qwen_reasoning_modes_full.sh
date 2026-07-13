#!/bin/bash
#SBATCH -J qwen-reasoning-full
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=22:00:00
#SBATCH --output=runs/qwen_reasoning_full/slurm-%j.log

set -euo pipefail

REPO_ROOT="/home/ua248/rds/hpc-work/adapting_world_models"
ANN="${REPO_ROOT}/holoassist/data-annotation-trainval-v1_1.json"
VIDEO_DIR="${REPO_ROOT}/holoassist/videos"
VAL_LIST="${REPO_ROOT}/holoassist/splits/val-v1_2.txt"
OUT_ROOT="${REPO_ROOT}/runs/qwen_reasoning_full/qwen_val"
PRIOR_MODE="actions_with_labels"

cd "$REPO_ROOT"

module purge
module load python/3.11.0-icl
source "$REPO_ROOT/venv_ampere/bin/activate"
export HF_HOME="$REPO_ROOT/hf_cache"

mkdir -p "$REPO_ROOT/runs/qwen_reasoning_full"
mkdir -p "$OUT_ROOT/direct" "$OUT_ROOT/cot"

echo "=== Job started at $(date) ==="
echo "Annotations:  $ANN"
echo "Video dir:    $VIDEO_DIR"
echo "Val list:     $VAL_LIST"
echo "Prior mode:   $PRIOR_MODE"
echo "Output root:  $OUT_ROOT"

for reasoning_mode in direct cot; do
    out_file="$OUT_ROOT/$reasoning_mode/results.jsonl"
    echo ""
    echo "=== Running $reasoning_mode mode ==="
    echo "Output file: $out_file"
    python3 -u "$REPO_ROOT/qwen_val.py" \
        --annotations "$ANN" \
        --video-dir "$VIDEO_DIR" \
        --val-list "$VAL_LIST" \
        --out "$out_file" \
        --prior-mode "$PRIOR_MODE" \
        --reasoning-mode "$reasoning_mode"
done

echo "=== Job finished at $(date) ==="