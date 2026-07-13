#!/bin/bash
#SBATCH -J aug_sensitivity
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/aug_sensitivity_%j.log

# -----------------------------------------------------------------------
# V-JEPA 2.1 augmentation sensitivity test
#
# What this does:
#   - Loads 30 mistake segments from features/vjepa2_train/
#   - Re-decodes their source frames and applies slight_crop / warm_tint /
#     combined augmentation
#   - Runs each through V-JEPA 2.1 and computes cosine similarity
#     to the non-augmented version
#   - Also computes pairwise baseline cosine similarities (mistake-vs-mistake,
#     correct-vs-correct, mistake-vs-correct) from stored features
#   - Writes results/augmentation_sensitivity.json
#   - After the job: run  python analyse_augmentation_sensitivity.py
#                          --results results/augmentation_sensitivity.json
#
# Estimated time: ~45–60 min for 30 segments × 3 augmentations on 1 A100
# (3 encoder passes per segment × ~45 s/pass on GPU)
#
# To run just a quick 3-segment smoke test (useful if debugging), set
#   --n_segments 3 --smoke
# That takes ~5–10 min.
# -----------------------------------------------------------------------

set -euo pipefail

WORK=/home/ua248/rds/hpc-work/adapting_world_models
HOLOASSIST=$WORK/holoassist

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"

mkdir -p "$WORK/logs"
mkdir -p "$WORK/results"

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

cd "$WORK"

python3 -u test_augmentation_sensitivity.py \
    --features_dir  "$WORK/features/vjepa2_train" \
    --video_dir     "$HOLOASSIST/videos" \
    --output        "$WORK/results/augmentation_sensitivity_full_tokens_crop90.json" \
    --n_segments    30 \
    --n_baseline    20 \
    --n_correct     30 \
    --seed          42 \
    --crop_frac     0.9 \
    --augmentations slight_crop warm_tint combined

echo ""
echo "=== Extraction finished at $(date) ==="
echo ""
echo "=== Running offline analysis ==="
python3 -u analyse_augmentation_sensitivity.py \
    --results "$WORK/results/augmentation_sensitivity.json"

echo ""
echo "=== Job finished at $(date) ==="
