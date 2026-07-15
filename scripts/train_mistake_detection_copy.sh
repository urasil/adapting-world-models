#!/bin/bash
#SBATCH -J train_probe_d1
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/probe_d1_%j.log
set -euo pipefail
# Attentive probe, depth=1 (one cross-attention layer, NO spatial pooling).
# Tokens per segment = 32 temporal × 576 spatial = 18 432 full tokens.
# Effective batch size 128 via gradient accumulation: micro_bs=2, accum=64.
#
# Why micro_bs=2 / accum=64 and not larger micro batches:
#   At max_segments=64 the KV sequence is 64×18 432=1 179 648 tokens.
#   With micro_bs=2 peak GPU memory is ~34 GB (well within A100 80 GB).
#   micro_bs=4 would push to ~68 GB — technically fits but leaves little
#   headroom; micro_bs=2 is the safe choice.
#   With depth=1 there is NO self-attention (no Block layers), so memory
#   scales as O(B×N) not O(B×N²).  The 1-query cross-attention produces a
#   (B, H, 1, N) attention score tensor (~75 MB at max sequence) not N².
#
# Wall time: ~18–24 h for 20 epochs (compute-bound after .npy conversion).
#
#   GPU compute: K/V projections are COMPUTE-bound (AI≈512 FLOPs/byte >>
#   A100 roofline 156 FLOPs/byte).  3.09 TFLOPs K+V fwd per micro-batch
#   @ 70% eff ≈ 55 ms total (fwd+bwd).  57,500 micro-batches/epoch ≈
#   53 min/epoch → ~18 h GPU compute for 20 epochs.
#
#   I/O with .npy mmap: reads only the needed prefix rows (~530 MB avg).
#   At 3 GB/s RDS bandwidth → ~180 ms per example.  With 8 workers this
#   is hidden behind the 55 ms GPU step (8 workers × 180 ms = 1.4 s
#   pipeline depth >> one 55 ms step), so I/O does NOT add to wall time.
#
#   WHY .npy is required: .npz_compressed decompresses at ~15 s/file
#   (1.08x compression ratio — features are incompressible).  With random
#   shuffling that would be 60+ h/epoch.  The .npy mmap path cuts this
#   to ~180 ms/example by avoiding full-file decompression.
#
#   The ETA printed after epoch 1 is the reliable number.
#
# Usage:
#   sbatch train_mistake_detection.sh              # defaults: 20 epochs, lr=1e-3, wd=1e-2
#   sbatch train_mistake_detection.sh 20 3e-3      # epochs, lr override (for sweeps)

EPOCHS="${1:-20}"
LR="${2:-1e-3}"
WD="${3:-1e-2}"
DEPTH="${4:-1}"
MICRO_BS="${5:-2}"
ACCUM="${6:-64}"

WORK=/home/ua248/rds/hpc-work/adapting_world_models
# Full-token features — .npy (uncompressed, mmap-friendly) preferred.
# Run convert_features_to_npy.py first if these dirs don't exist yet.
# Fallback: dataset._find_feature_file tries .npy then .npz automatically.
TRAIN_FEATURES=/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features/vjepa2_train_npy
VAL_FEATURES=/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features/vjepa2_val_npy

INDEX_PATH="$WORK/datasets/mistake_detection_index_full.json"
TRAIN_SPLIT_PATH="$WORK/holoassist/splits/train-v1_2.txt"
VAL_SPLIT_PATH="$WORK/holoassist/splits/val-v1_2.txt"
TAG="probe_d${DEPTH}_bs$((MICRO_BS * ACCUM))_lr${LR}_wd${WD}"
OUT_DIR="$WORK/runs/${TAG}_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR" "$WORK/logs"

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"

export HF_HOME="$WORK/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Job started at $(date) ==="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Epochs:        $EPOCHS"
echo "LR:            $LR"
echo "WD:            $WD"
echo "Depth:         $DEPTH"
echo "Micro batch:   $MICRO_BS"
echo "Accum steps:   $ACCUM"
echo "Effective bs:  $((MICRO_BS * ACCUM))"
echo "Tokens/seg:    18432  (32 temporal × 576 spatial, no pooling)"
echo "Max segments:  64"
echo "Precision:     fp32"
echo "Train feats:   $TRAIN_FEATURES"
echo "Val feats:     $VAL_FEATURES"
echo "Output dir:    $OUT_DIR"

# ---- Environment sanity check --------------------------------------------
echo ""
echo "--- Environment check ---"
python3 -c "import torch; print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ---- Full training run ---------------------------------------------------
echo ""
echo "--- Starting training ---"
python3 -u "$WORK/training/train_mistake_detection.py" \
    --index_path         "$INDEX_PATH" \
    --train_split_file   "$TRAIN_SPLIT_PATH" \
    --val_split_file     "$VAL_SPLIT_PATH" \
    --train_features_dir "$TRAIN_FEATURES" \
    --val_features_dir   "$VAL_FEATURES" \
    --output_dir         "$OUT_DIR" \
    --epochs             "$EPOCHS" \
    --batch_size         "$MICRO_BS" \
    --grad_accum_steps   "$ACCUM" \
    --depth              "$DEPTH" \
    --tokens_per_segment 18432 \
    --lr                 "$LR" \
    --weight_decay       "$WD" \
    --num_workers        8 \
    --log_every          50 \
    --max_segments       64 \
    --precision          fp32

# ---- Post-run summary ----------------------------------------------------
echo ""
echo "--- Post-run summary ---"
ls -lh "$OUT_DIR/"
echo ""
echo "Best epoch & AP:"
python3 -c "
import json
h = json.load(open('$OUT_DIR/history.json'))
best = max(h, key=lambda e: e['val_ap'])
e = best
print(f'  epoch {e[\"epoch\"]}: AP={e[\"val_ap\"]:.4f}  AUROC={e[\"val_auroc\"]:.4f}  F1_inv_freq={e[\"val_f1_inv_freq\"]:.4f}')
print(f'  neg(no-mistake):  P={e[\"val_prec_neg\"]:.3f}  R={e[\"val_rec_neg\"]:.3f}  F1={e[\"val_f1_neg\"]:.3f}')
print(f'  pos(mistake):     P={e[\"val_prec_pos\"]:.3f}  R={e[\"val_rec_pos\"]:.3f}  F1={e[\"val_f1_pos\"]:.3f}')
print(f'  macro={e[\"val_f1_macro\"]:.3f}  bAcc={e[\"val_balanced_acc\"]:.3f}  thr={e[\"val_best_thr\"]:.2f}')
"

echo "=== Job finished at $(date) ==="
