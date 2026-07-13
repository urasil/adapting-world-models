#!/bin/bash
#SBATCH -J aug_extract
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/aug_extract_%a_%j.log

# =============================================================================
# Augmented V-JEPA 2.1 Feature Extraction — SLURM array job
#
# Runs four jobs in parallel (one per augmentation variant):
#   array 0 → warm_tint
#   array 1 → cold_tint
#   array 2 → crop_90
#   array 3 → warm_crop
#
# Each job processes all training videos (~1466), re-extracting V-JEPA 2.1
# spatially-pooled features (32 × 1024) after applying the augmentation.
# Output files are written to features/vjepa2_train_aug/ as:
#   {video_name}_aug_{aug_name}.npz
#
# Estimated wall time: ~30–36 h per job (matches original extraction time).
# All four jobs run simultaneously as an array.
#
# Submit:
#   sbatch scripts/extract_augmented_features.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/aug_extract_0_<jobid>.log
#
# After all four jobs finish, build the augmented index:
#   python build_augmented_mistake_index.py \
#       --original_index   datasets/mistake_detection_index.json \
#       --aug_features_dir features/vjepa2_train_aug \
#       --aug_names        warm_tint cold_tint crop_90 warm_crop \
#       --output_path      datasets/mistake_detection_index_aug.json \
#       --target_ratio     4
#
# Then verify the new index before training:
#   python build_augmented_mistake_index.py ... --dry_run
# =============================================================================

set -euo pipefail

WORK=/home/ua248/rds/hpc-work/adapting_world_models
HOLOASSIST=$WORK/holoassist

# Map SLURM array index → augmentation name
AUG_NAMES=(warm_tint cold_tint crop_90 warm_crop)
AUG_NAME="${AUG_NAMES[$SLURM_ARRAY_TASK_ID]}"

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"

mkdir -p "$WORK/logs"
mkdir -p "$WORK/features/vjepa2_train_aug"

echo "================================================================"
echo "Augmented Feature Extraction"
echo "  Array task   : $SLURM_ARRAY_TASK_ID / 3"
echo "  Augmentation : $AUG_NAME"
echo "  Node         : $(hostname)"
echo "  GPU          : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  Started      : $(date)"
echo "================================================================"
echo ""

cd "$WORK"

python3 -u extract_augmented_features.py \
    --features_dir  "$WORK/features/vjepa2_train" \
    --video_dir     "$HOLOASSIST/videos" \
    --output_dir    "$WORK/features/vjepa2_train_aug" \
    --aug_name      "$AUG_NAME" \
    --seed          42 \
    --resume

echo ""
echo "================================================================"
echo "Extraction finished: $AUG_NAME  at $(date)"
echo "Files written to: $WORK/features/vjepa2_train_aug/"
ls -lh "$WORK/features/vjepa2_train_aug/"*"_aug_${AUG_NAME}.npz" 2>/dev/null | wc -l \
    | xargs -I{} echo "  {} .npz files for $AUG_NAME"
echo "================================================================"
