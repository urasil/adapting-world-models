#!/bin/bash
#SBATCH -J qwen-reasoning-smoke
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=runs/qwen_reasoning_smoke/slurm-%j.log

set -euo pipefail

REPO_ROOT="/home/ua248/rds/hpc-work/adapting_world_models"

cd "$REPO_ROOT"

module purge
module load python/3.11.0-icl
source "$REPO_ROOT/venv_ampere/bin/activate"
export HF_HOME="$REPO_ROOT/hf_cache"

echo "=== Job started at $(date) ==="
echo "Using Python: $(python3 -V)"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Run smoke test
python3 -u "$REPO_ROOT/smoke_qwen_reasoning_modes.py" \
  --annotations "$REPO_ROOT/holoassist/data-annotation-trainval-v1_1.json" \
  --video-dir "$REPO_ROOT/holoassist/videos" \
  --output-root "$REPO_ROOT/runs/qwen_reasoning_smoke" \
  --prior-mode "actions_with_labels"

echo "=== Job finished at $(date) ==="
