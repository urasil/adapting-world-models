#!/bin/bash
#SBATCH -J smoke_wm_confirm
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:15:00
#SBATCH --output=logs/smoke_wm_confirm_%j.log
set -euo pipefail

# Confirmatory smoke test for the config scripts/train_segment_wm.sh is about
# to run for ~150+ hours: S=2, depth=6, D=1024, batch_size=2, act_ckpt off,
# exclude_tasks marius/knarrevik/rashult. Prior evidence for this exact combo
# is only 18 micro-steps (~72s) from an interactive session that got cut
# short — this run gives it a full 60-minute budget so it reaches a real
# PASS/FAIL verdict, exercises every ctx_len up to the full S=2 context many
# times, and reports a steady-state s/step to sanity-check the ~15.3h/epoch
# estimate before commiting a multi-day job to it.

WORK=/home/ua248/rds/hpc-work/adapting_world_models
cd "$WORK"

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Job started at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python3 -u smoke_test_wm.py \
    --train_index  datasets/wm_train_index.json \
    --train_split  holoassist/splits/train-v1_2.txt \
    --val_split    holoassist/splits/val-v1_2.txt \
    --max_context_segs 2 --depth 6 --embed_dim 1024 --num_heads 16 \
    --batch_size 2 --grad_accum 64 \
    --exclude_tasks marius,knarrevik,rashult \
    --minutes 60

echo "=== Job finished at $(date) ==="
