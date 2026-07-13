#!/bin/bash
#SBATCH -J convert_npy
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/convert_npy_%j.log
set -euo pipefail

WORK=/home/ua248/rds/hpc-work/adapting_world_models
BASE=/home/ua248/rds/rds-altaslp-8YSp2LXTlkY/data/holoassist_features

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"

mkdir -p "$WORK/logs"

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK"
echo ""

for SPLIT in train val; do
    SRC="$BASE/vjepa2_${SPLIT}"
    DST="$BASE/vjepa2_${SPLIT}_npy"
    N=$(ls "$SRC"/*.npz 2>/dev/null | wc -l)
    echo "--- Converting $SPLIT: $N files ---"
    echo "  src: $SRC"
    echo "  dst: $DST"
    python3 -u "$WORK/convert_features_to_npy.py" \
        --src_dir "$SRC" \
        --dst_dir "$DST" \
        --workers 4
    echo ""
done

echo "=== Job finished at $(date) ==="
