#!/bin/bash
#SBATCH -J vjepa2_nopooling_train
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/extract_nopooling_train_%j.log

set -euo pipefail

WORK=/home/ua248/rds/hpc-work/adapting_world_models
HOLOASSIST=$WORK/holoassist
OUT=/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features/vjepa2_train

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"

echo "=== Job started at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p "$OUT"

cd "$WORK"
python3 -u src/vjepa2_1_segment_feature_extraction.py \
    --annotation_path "$HOLOASSIST/data-annotation-trainval-v1_1.json" \
    --video_dir       "$HOLOASSIST/videos" \
    --output_dir      "$OUT" \
    --split_file      "$HOLOASSIST/splits/train-v1_2.txt"

echo "=== Job finished at $(date) ==="
