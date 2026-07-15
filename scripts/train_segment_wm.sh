#!/bin/bash
#SBATCH -J train_wm
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=35:00:00
#SBATCH --output=logs/train_wm_%j.log
set -euo pipefail

# Self-chaining launcher for SegmentMaskPredictorAC world-model pre-training.
#
# Each SLURM allocation is capped at 36h, but the job needs ~150h+ total for
# 10 epochs, so this script combines two independent safety nets:
#
#   1. Job-level: it only asks train_segment_wm.py for EPOCHS_PER_JOB more
#      epochs (not all 10), so the process finishes cleanly and exits on its
#      own well inside the 35h budget, then re-submits itself via `sbatch`
#      for the next segment. Normal operation never relies on SLURM
#      SIGKILLing the job.
#   2. Process-level: train_segment_wm.py now checkpoints last.pt every
#      --ckpt_every_steps optimizer steps (~20, ~1h at the confirmed
#      throughput), not just at epoch boundaries. If the node dies for any
#      reason (hardware fault, pre-emption, OOM, a bad batch) at any point,
#      at most ~1h of work is lost, not up to a full ~15h epoch. Checkpoint
#      writes are atomic (temp file + rename), so a crash mid-save can't
#      corrupt last.pt either.
#
# runs/wm_v1/last.pt        = most recent checkpoint (mid-epoch or epoch-end)
# runs/wm_v1/best_val_l1_ep*.pt = best validation checkpoint seen so far
#
# Config (S=2, depth=6, D=1024, batch_size=4, act_ckpt off, exclude_tasks
# marius/knarrevik/rashult) is the one validated safe by dev/smoke_test_wm.py:
# ~40 GB peak (vs 80 GB card) and ~1.54s/example.
#
# CAVEAT: if this job dies uncleanly (hard crash, not a normal exit), the
# self-resubmission line at the bottom never runs and the chain stops. Check
# `squeue -u $USER` / `sacct` periodically; if the chain has stopped short of
# 10 epochs, just run `sbatch scripts/train_segment_wm.sh` again by hand —
# it will pick up from last.pt (losing at most ~1h) and keep going.

WORK=/home/ua248/rds/hpc-work/adapting_world_models
cd "$WORK"

OUT_DIR="$WORK/runs/wm_v1"
TOTAL_EPOCHS=10
EPOCHS_PER_JOB=2   # ~2 x 15.3h = 30.6h train + ~1h val, inside the 35h cap.
                   # Re-derive this once the confirmatory smoke test (see
                   # scripts/smoke_wm_confirm.sh) reports real s/example.

mkdir -p "$OUT_DIR" "$WORK/logs"

module purge
module load rhel8/slurm python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Job started at $(date) ==="
echo "Job ID: $SLURM_JOB_ID   Node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# last.pt's 'epoch' field is the epoch currently in progress if resume_iter>0
# (mid-epoch checkpoint), or the last *completed* epoch if resume_iter==0.
read_progress() {
    python3 -c "
import torch
c = torch.load('$OUT_DIR/last.pt', map_location='cpu')
ep, ri = c['epoch'], c.get('resume_iter', 0) or 0
print((ep - 1) if ri else ep)
"
}

DONE_EPOCHS=0
RESUME_FLAG=""
if [ -f "$OUT_DIR/last.pt" ]; then
    DONE_EPOCHS=$(read_progress)
    RESUME_FLAG="--resume"
fi

if [ "$DONE_EPOCHS" -ge "$TOTAL_EPOCHS" ]; then
    echo "Already at $DONE_EPOCHS/$TOTAL_EPOCHS epochs — nothing to do."
    exit 0
fi

TARGET_EPOCHS=$(( DONE_EPOCHS + EPOCHS_PER_JOB ))
if [ "$TARGET_EPOCHS" -gt "$TOTAL_EPOCHS" ]; then
    TARGET_EPOCHS=$TOTAL_EPOCHS
fi

echo "done_epochs=$DONE_EPOCHS  target_epochs=$TARGET_EPOCHS/$TOTAL_EPOCHS  resume=${RESUME_FLAG:-no}"

python3 -u training/train_segment_wm.py \
    --train_index   datasets/wm_train_index.json \
    --val_index     datasets/wm_val_index.json \
    --train_split   holoassist/splits/train-v1_2.txt \
    --val_split     holoassist/splits/val-v1_2.txt \
    --output_dir    "$OUT_DIR" \
    --max_context_segs 2 \
    --embed_dim 1024 --predictor_embed_dim 1024 --depth 6 --num_heads 16 \
    --batch_size 4 --grad_accum_steps 64 \
    --exclude_tasks marius,knarrevik,rashult \
    --epochs "$TARGET_EPOCHS" \
    --ckpt_every_steps 20 \
    --num_workers 8 \
    $RESUME_FLAG

echo "=== Job finished at $(date) ==="

NEW_DONE=$(read_progress)
if [ "$NEW_DONE" -lt "$TOTAL_EPOCHS" ]; then
    echo "Epoch $NEW_DONE/$TOTAL_EPOCHS done — submitting next segment."
    sbatch "$WORK/scripts/train_segment_wm.sh"
else
    echo "All $TOTAL_EPOCHS epochs complete. best last.pt / best_val_l1_ep*.pt are in $OUT_DIR"
fi
