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

# Add src/ to path so we can import the feature extractor module
sys.path.insert(0, '/home/ua248/rds/hpc-work/adapting_world_models/src')
from vjepa2_1_segment_feature_extraction import (
    load_model,
    extract_segments_for_video,
    LABEL_MAP,
)

HOLOASSIST = '/home/ua248/rds/hpc-work/adapting_world_models/holoassist'
ANN_PATH   = f'{HOLOASSIST}/data-annotation-trainval-v1_1.json'
VIDEO_DIR  = f'{HOLOASSIST}/videos'
SPLIT_FILE = f'{HOLOASSIST}/splits/train-v1_2.txt'
OUT_PATH   = '/tmp/timing_test.npz'

# Restrict to train split, then find the first video that's actually on disk
with open(SPLIT_FILE) as f:
    train_names = {line.strip() for line in f if line.strip()}
print(f"Train split: {len(train_names)} videos listed")

with open(ANN_PATH) as f:
    ann = json.load(f)
ann = [e for e in ann if e['video_name'] in train_names]
print(f"Annotations matching train split: {len(ann)}")

entry = None
video_path = None
for e in ann:
    vpath = f'{VIDEO_DIR}/{e["video_name"]}/Export_py/Video_compress.mp4'
    if os.path.exists(vpath):
        entry = e
        video_path = vpath
        break
if entry is None:
    sys.exit("No train video found on disk")

n_segs = sum(
    1 for ev in entry['events']
    if ev['label'] == 'Fine grained action'
    and ev.get('attributes', {}).get('Action Correctness') in LABEL_MAP
)
print(f"Video:    {entry['video_name']}")
print(f"Segments: {n_segs}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load_model now returns (processor, model, dtype)
processor, model, dtype = load_model(device)
if device == 'cuda':
    torch.cuda.synchronize()
print(f"Model loaded on {device}, dtype={dtype}")

# --- warm-up: first forward pass triggers cuDNN autotuning, kernel compilation,
# memory allocator behaviour, etc. Excluding it gives a much more honest per-seg time.
print("Warm-up pass (excluded from timing)...")
_ = extract_segments_for_video(video_path, entry, processor, model, dtype, test=True)
if device == 'cuda':
    torch.cuda.synchronize()

# --- timed run on the full video
t0 = time.perf_counter()
results = extract_segments_for_video(video_path, entry, processor, model, dtype)
if device == 'cuda':
    torch.cuda.synchronize()
t1 = time.perf_counter()

elapsed = t1 - t0
n_done  = len(results)
per_seg = elapsed / n_done if n_done else float('nan')

# Write a real .npz to confirm I/O works and to measure file size
seg_ids = sorted(results.keys())
feats   = np.stack([results[s]['features'] for s in seg_ids]).astype(np.float16)
labels  = np.array([results[s]['label'] for s in seg_ids], dtype=np.int8)
np.savez_compressed(OUT_PATH, features=feats, labels=labels)
size_mb = os.path.getsize(OUT_PATH) / 1e6

# Sanity checks on the output
assert feats.shape[1:] == (32, 1024), f"unexpected feature shape: {feats.shape}"
assert feats.dtype == np.float16
assert set(np.unique(labels).tolist()).issubset({0, 1})

# Peak GPU memory
if device == 'cuda':
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
else:
    peak_gb = float('nan')

# Estimate total work from the train split annotations
total_train_segs = 0
for e in ann:
    for ev in e['events']:
        if (ev['label'] == 'Fine grained action'
                and ev.get('attributes', {}).get('Action Correctness') in LABEL_MAP):
            total_train_segs += 1

print(f"\n=== Results ===")
print(f"Segments extracted     : {n_done}")
print(f"Total time             : {elapsed:.1f}s")
print(f"Per segment            : {per_seg:.2f}s")
print(f"Feature shape          : {feats.shape}  dtype={feats.dtype}")
print(f"Mistake fraction       : {labels.mean():.3f}")
print(f"File size              : {size_mb:.1f} MB  ({size_mb/n_done:.2f} MB/segment)")
print(f"Peak GPU memory        : {peak_gb:.2f} GB")
print(f"\n=== Train-split projection ===")
print(f"Total train segments   : {total_train_segs}")
print(f"Estimated GPU-hours    : {per_seg * total_train_segs / 3600:.1f}")
print(f"Estimated total size   : {(size_mb / n_done) * total_train_segs / 1024:.1f} GB")
EOF

echo "=== Job finished at $(date) ==="