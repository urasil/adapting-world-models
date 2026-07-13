#!/bin/bash
#SBATCH -J vjepa2_timing
#SBATCH -A MLMI-ua248-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=timing_%j.log
set -euo pipefail

HOLOASSIST=/home/ua248/rds/hpc-work/adapting_world_models/holoassist
WORK=/home/ua248/rds/hpc-work/adapting_world_models

module purge
module load python/3.11.0-icl
source "$WORK/venv_ampere/bin/activate"

echo "=== Job started at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python3 -u - <<'EOF'
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, '/home/ua248/rds/hpc-work/adapting_world_models')
from extract_representations_action_segment_based import load_model, extract_segments_for_video

HOLOASSIST = '/home/ua248/rds/hpc-work/adapting_world_models/holoassist'
ANN_PATH   = f'{HOLOASSIST}/data-annotation-trainval-v1_1.json'
VIDEO_DIR  = f'{HOLOASSIST}/videos'
OUT_PATH   = '/tmp/timing_test.npz'

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
             and ev.get('attributes', {}).get('Action Correctness') in
                 {'Correct Action',
                  'Wrong Action, corrected by instructor verbally',
                  'Wrong Action, corrected by student',
                  'Wrong Action, not corrected'})

print(f"Video:    {entry['video_name']}")
print(f"Segments: {n_segs}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor, model = load_model(device)
if device == 'cuda':
    torch.cuda.synchronize()
print(f"Model loaded on {device}")

t0 = time.perf_counter()
results = extract_segments_for_video(video_path, entry, processor, model)
if device == 'cuda':
    torch.cuda.synchronize()
t1 = time.perf_counter()

elapsed = t1 - t0
n_done  = len(results)
per_seg = elapsed / n_done if n_done else float('nan')

seg_ids = sorted(results.keys())
feats   = np.stack([results[s]['features'] for s in seg_ids]).astype(np.float16)
labels  = np.array([results[s]['label'] for s in seg_ids], dtype=np.int8)
np.savez_compressed(OUT_PATH, features=feats, labels=labels)
size_mb = os.path.getsize(OUT_PATH) / 1e6

print(f"\n=== Results ===")
print(f"Segments extracted : {n_done}")
print(f"Total time         : {elapsed:.1f}s")
print(f"Per segment        : {per_seg:.2f}s")
print(f"Feature shape      : {feats.shape}  dtype={feats.dtype}")
print(f"File size          : {size_mb:.1f} MB  ({size_mb/n_done:.2f} MB/segment)")
print(f"Extrapolated 159k  : {per_seg * 159000 / 3600:.1f} GPU-hours")
EOF

echo "=== Job finished at $(date) ==="
