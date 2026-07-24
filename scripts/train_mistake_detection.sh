#!/bin/bash
#SBATCH -J train_probe_d1
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=29:00:00
#SBATCH --output=logs/probe_d1_%j.log
set -euo pipefail
# Attentive probe, depth=1, full token grid (18432 tokens/segment).
# Self-chaining: full training (~39h) exceeds the 36h SLURM cap, so each job
# only runs EPOCHS_PER_JOB epochs, exits on its own, and resubmits itself
# until last.pt reaches TOTAL_EPOCHS. --total_epochs keeps the LR schedule
# consistent across jobs even though --epochs (this job's target) is smaller.
#
# Usage: sbatch train_mistake_detection.sh [TOTAL_EPOCHS LR WD DEPTH MICRO_BS ACCUM NEG_RATIO MAX_SEG EPOCHS_PER_JOB]

TOTAL_EPOCHS="${1:-10}"
LR="${2:-1e-3}"
WD="${3:-1e-2}"
DEPTH="${4:-1}"
MICRO_BS="${5:-2}"
ACCUM="${6:-64}"
NEG_RATIO="${7:-10.0}"       # neg:pos ratio for downsampling; set to "" to disable
MAX_SEG="${8:-16}"           # max segments per prefix (was 64; halved for I/O speed)
EPOCHS_PER_JOB="${9:-7}"     # ~3.9h/epoch historically -> ~27h/job, safe under 36h

WORK=/home/ua248/rds/hpc-work/adapting_world_models
TRAIN_FEATURES=/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features/vjepa2_train_npy
VAL_FEATURES=/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features/vjepa2_val_npy

INDEX_PATH="$WORK/datasets/mistake_detection_index_full.json"
TRAIN_SPLIT_PATH="$WORK/holoassist/splits/train-v1_2.txt"
VAL_SPLIT_PATH="$WORK/holoassist/splits/val-v1_2.txt"
TAG="probe_d${DEPTH}_bs$((MICRO_BS * ACCUM))_lr${LR}_wd${WD}"
# fixed path (no job-ID suffix) so every resubmission finds the same last.pt
OUT_DIR="$WORK/runs/${TAG}"
mkdir -p "$OUT_DIR" "$WORK/logs"

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"

export HF_HOME="$WORK/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

read_done_epochs() {
    if [ ! -f "$OUT_DIR/last.pt" ]; then
        echo 0
        return
    fi
    python3 -c "
import torch
c = torch.load('$OUT_DIR/last.pt', map_location='cpu', weights_only=False)
print(c['epoch'])
"
}

DONE_EPOCHS=$(read_done_epochs)
RESUME_FLAG=""
if [ "$DONE_EPOCHS" -gt 0 ]; then
    RESUME_FLAG="--resume"
fi

echo "=== Job started at $(date) ==="
echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $SLURMD_NODENAME"
echo "GPU:             $CUDA_VISIBLE_DEVICES"
echo "Total epochs:    $TOTAL_EPOCHS"
echo "Done so far:     $DONE_EPOCHS"
echo "LR:              $LR"
echo "WD:              $WD"
echo "Depth:           $DEPTH"
echo "Micro batch:     $MICRO_BS"
echo "Accum steps:     $ACCUM"
echo "Effective bs:    $((MICRO_BS * ACCUM))"
echo "Tokens/seg:      18432  (32 temporal × 576 spatial, no pooling)"
echo "Max segments:    $MAX_SEG"
echo "Neg ratio:       $NEG_RATIO"
echo "Precision:       bf16"
echo "Train feats:     $TRAIN_FEATURES"
echo "Val feats:       $VAL_FEATURES"
echo "Output dir:      $OUT_DIR"

if [ "$DONE_EPOCHS" -ge "$TOTAL_EPOCHS" ]; then
    echo "Already at $DONE_EPOCHS/$TOTAL_EPOCHS epochs, nothing to do."
    exit 0
fi

TARGET_EPOCHS=$(( DONE_EPOCHS + EPOCHS_PER_JOB ))
if [ "$TARGET_EPOCHS" -gt "$TOTAL_EPOCHS" ]; then
    TARGET_EPOCHS=$TOTAL_EPOCHS
fi
echo "This job's target epoch: $TARGET_EPOCHS/$TOTAL_EPOCHS  resume=${RESUME_FLAG:-no}"

# ---- Environment sanity check --------------------------------------------
echo ""
echo "--- Environment check ---"
python3 -c "import torch; print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ---- Training run for this job's segment ----------------------------------
echo ""
echo "--- Starting training ---"
python3 -u "$WORK/training/train_mistake_detection.py" \
    --index_path         "$INDEX_PATH" \
    --train_split_file   "$TRAIN_SPLIT_PATH" \
    --val_split_file     "$VAL_SPLIT_PATH" \
    --train_features_dir "$TRAIN_FEATURES" \
    --val_features_dir   "$VAL_FEATURES" \
    --output_dir         "$OUT_DIR" \
    --epochs             "$TARGET_EPOCHS" \
    --total_epochs       "$TOTAL_EPOCHS" \
    --batch_size         "$MICRO_BS" \
    --grad_accum_steps   "$ACCUM" \
    --depth              "$DEPTH" \
    --tokens_per_segment 18432 \
    --lr                 "$LR" \
    --weight_decay       "$WD" \
    --num_workers        8 \
    --log_every          50 \
    --max_segments       "$MAX_SEG" \
    --precision          bf16 \
    ${NEG_RATIO:+--neg_subsample_ratio "$NEG_RATIO"} \
    ${RESUME_FLAG}

# ---- Post-run summary ------------------------------------------------------
echo ""
echo "--- Post-run summary ---"
ls -lh "$OUT_DIR/"
echo ""
echo "Best epoch & AP so far:"
python3 -c "
import json
h = json.load(open('$OUT_DIR/history.json'))
best = max(h, key=lambda e: e['val_ap'])
e = best
print(f'  epoch {e[\"epoch\"]}: AP={e[\"val_ap\"]:.4f}  AUROC={e[\"val_auroc\"]:.4f}  F1_inv_freq={e[\"val_f1_inv_freq\"]:.4f}')
print(f'  neg(no-mistake):  P={e[\"val_prec_neg\"]:.3f}  R={e[\"val_rec_neg\"]:.3f}  F1={e[\"val_f1_neg\"]:.3f}')
print(f'  pos(mistake):     P={e[\"val_prec_pos\"]:.3f}  R={e[\"val_rec_pos\"]:.3f}  F1={e[\"val_f1_pos\"]:.3f}')
print(f'  macro={e[\"val_f1_macro\"]:.3f}  bAcc={e[\"val_balanced_acc\"]:.3f}')
"

# ---- Resubmit if the true total hasn't been reached yet --------------------
NEW_DONE=$(read_done_epochs)
if [ "$NEW_DONE" -lt "$TOTAL_EPOCHS" ]; then
    echo ""
    echo "Epoch $NEW_DONE/$TOTAL_EPOCHS done, submitting next segment."
    sbatch "$WORK/scripts/train_mistake_detection.sh" \
        "$TOTAL_EPOCHS" "$LR" "$WD" "$DEPTH" "$MICRO_BS" "$ACCUM" "$NEG_RATIO" "$MAX_SEG" "$EPOCHS_PER_JOB"
else
    echo ""
    echo "All $TOTAL_EPOCHS epochs complete. best_*_ep*.pt in $OUT_DIR"
fi

echo "=== Job finished at $(date) ==="
