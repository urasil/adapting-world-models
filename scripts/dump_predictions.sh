#!/bin/bash
#SBATCH -J dump_preds
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/dump_preds_%j.log

set -euo pipefail

WORK=/home/ua248/rds/hpc-work/adapting_world_models
RUN_DIR="${1:-$WORK/runs/full_bs128_lr1e-3_wd1e-2_29411291}"

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"

export HF_HOME="$WORK/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== dump_predictions started at $(date) ==="
echo "Run dir: $RUN_DIR"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

python3 -u "$WORK/evaluation/dump_val_predictions.py" \
    --run_dir "$RUN_DIR" \
    --num_workers 0 \
    --batch_size 256

echo "=== Done at $(date) ==="
