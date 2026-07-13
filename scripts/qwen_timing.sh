#!/bin/bash
#SBATCH -J qwen_timing
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=qwen_timing_%j.log
set -euo pipefail

HOLOASSIST=/home/ua248/rds/hpc-work/adapting_world_models/holoassist
WORK=/home/ua248/rds/hpc-work/adapting_world_models

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"

export HF_HOME="$WORK/hf_cache"

echo "=== Job started at $(date) ==="

python3 -u - <<'EOF'
import json, os, sys, time
import torch
sys.path.insert(0, '/home/ua248/rds/hpc-work/adapting_world_models')
from qwen_baseline import load_model, process_video

HOLOASSIST = '/home/ua248/rds/hpc-work/adapting_world_models/holoassist'
ANN_PATH   = f'{HOLOASSIST}/data-annotation-trainval-v1_1.json'
VIDEO_DIR  = f'{HOLOASSIST}/videos'

LABEL_MAP = {
    'Correct Action',
    'Wrong Action, corrected by instructor verbally',
    'Wrong Action, corrected by student',
    'Wrong Action, not corrected',
}

with open(ANN_PATH) as f:
    ann = json.load(f)

entry = None
for e in ann:
    vpath = f'{VIDEO_DIR}/{e["video_name"]}/Export_py/Video_compress.mp4'
    if os.path.exists(vpath):
        entry = e
        video_path = vpath
        break
if entry is None:
    sys.exit("No video found on disk")

n_segs = sum(1 for ev in entry['events']
             if ev['label'] == 'Fine grained action'
             and ev.get('attributes', {}).get('Action Correctness') in LABEL_MAP)

print(f"Video:             {entry['video_name']}")
print(f"Labeled segments:  {n_segs}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

t_load_start = time.perf_counter()
processor, model = load_model(device)
if device == 'cuda':
    torch.cuda.synchronize()
t_load_end = time.perf_counter()
print(f"Model loaded on {device} in {t_load_end - t_load_start:.1f}s")

if device == 'cuda':
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after load: {mem:.2f} GB")

t0 = time.perf_counter()
results = process_video(video_path, entry, processor, model, test=True)
if device == 'cuda':
    torch.cuda.synchronize()
t1 = time.perf_counter()

elapsed = t1 - t0
n_done  = len(results)
per_seg = elapsed / n_done if n_done else float('nan')

if device == 'cuda':
    mem_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"GPU peak memory:       {mem_peak:.2f} GB")

print(f"\n=== Timing results ===")
print(f"Segments processed : {n_done}  (--test mode)")
print(f"Total time         : {elapsed:.1f}s")
print(f"Per segment        : {per_seg:.2f}s")
print(f"\nExtrapolation (fill in your val segment count):")
for n_val in [500, 1000, 2000, 5000]:
    h = per_seg * n_val / 3600
    print(f"  {n_val:>5} val segments -> {h:.1f} GPU-hours")
EOF

echo "=== Job finished at $(date) ==="

