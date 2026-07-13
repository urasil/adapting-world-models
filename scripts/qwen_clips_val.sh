#!/bin/bash
#SBATCH -J qwen_clips_val
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --output=logs/qwen_clips_val_%j.log
set -euo pipefail

# Usage:
#   sbatch scripts/qwen_clips_val.sh                          # clips, direct (default)
#   sbatch scripts/qwen_clips_val.sh clips                    # clips, direct
#   sbatch scripts/qwen_clips_val.sh clips_labelled           # oracle clips with labels
#   sbatch scripts/qwen_clips_val.sh none                     # target clip only (no priors)
#   sbatch scripts/qwen_clips_val.sh clips cot                # clips + chain-of-thought
#   sbatch scripts/qwen_clips_val.sh clips direct 4           # clips, direct, max 4 prior clips
# 12 clips used for validation
# Arguments:
#   $1 : prior_mode      (default: clips)
#   $2 : reasoning_mode  (default: direct)
#   $3 : max_prior_clips (default: unset → all prior clips shown)
PRIOR_MODE="${1:-clips}"
REASONING_MODE="${2:-direct}"
MAX_PRIOR_CLIPS="${3:-}"

WORK=/home/ua248/rds/hpc-work/adapting_world_models
HOLOASSIST="$WORK/holoassist"
VAL_LIST="$HOLOASSIST/splits/val-v1_2.txt"

# Output directory encodes the run configuration
if [ -n "$MAX_PRIOR_CLIPS" ]; then
    RUN_TAG="${PRIOR_MODE}_${REASONING_MODE}_maxprior${MAX_PRIOR_CLIPS}"
else
    RUN_TAG="${PRIOR_MODE}_${REASONING_MODE}"
fi
OUT_DIR="$WORK/results/qwen_clips_val/${RUN_TAG}"
OUT_FILE="$OUT_DIR/predictions.jsonl"
mkdir -p "$OUT_DIR"

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"
export HF_HOME="$WORK/hf_cache"

echo "=== Job started at $(date) ==="
echo "Prior mode:       $PRIOR_MODE"
echo "Reasoning mode:   $REASONING_MODE"
echo "Max prior clips:  ${MAX_PRIOR_CLIPS:-all}"
echo "Output file:      $OUT_FILE"

# Build optional --max-prior-clips argument
MAX_CLIPS_ARG=""
if [ -n "$MAX_PRIOR_CLIPS" ]; then
    MAX_CLIPS_ARG="--max-prior-clips $MAX_PRIOR_CLIPS"
fi

python3 -u "$WORK/qwen_clips_val.py" \
    --annotations "$HOLOASSIST/data-annotation-trainval-v1_1.json" \
    --video-dir   "$HOLOASSIST/videos" \
    --val-list    "$VAL_LIST" \
    --out         "$OUT_FILE" \
    --prior-mode  "$PRIOR_MODE" \
    --reasoning-mode "$REASONING_MODE" \
    $MAX_CLIPS_ARG

echo "=== Job finished at $(date) ==="
