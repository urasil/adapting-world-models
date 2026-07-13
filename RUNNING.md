# Running the Pipeline
Very simple end-to-end guide:
1. pull HoloAssist  
2. extract V-JEPA 2.1 features 
3. build dataset indices 
4. run/evaluate the three model families (attentive probe, world model, VLM)

## 0. Environment
```bash
module load python/3.11.0-icl
python3 -m venv venv_ampere
source venv_ampere/bin/activate
pip install -r requirements.txt
export HF_HOME="$WORK/hf_cache"
```

The V-JEPA 2.1 encoder architecture is pulled from `facebookresearch/vjepa2` via `torch.hub`, weights are loaded from the local checkpoint `vjepa2_1_vitl_dist_vitG_384.pt` in the repo root.

## 1. Pull the HoloAssist dataset
```bash
cd holoassist
sbatch download_holoassist.sh
```

This downloads into `holoassist/`:
- `video_compress.tar` — compressed videos (extract so each clip lives at
  `holoassist/videos/<video_name>/Export_py/Video_compress.mp4`)
- `data-annotation-trainval-v1_1.json` — action/mistake annotations
- `data-splits-v1_2.zip` — train/val/test split lists -> unzip into
  `holoassist/splits/{train,val,test}-v1_2.txt`

## 2. Extract V-JEPA 2.1 features

`src/vjepa2_1_segment_feature_extraction.py` extracts one feature per
HoloAssist fine-grained action segment (64 sampled frames -> spatially
pooled to a `(32, 1024)` tensor per segment) and writes one `.npz` per
video.

```bash
sbatch scripts/extraction.sh       # train split -> features/vjepa2_train/*.npz
sbatch scripts/extraction_val.sh   # val split   -> features/vjepa2_val/*.npz
```

Manual invocation (e.g. for a quick smoke test with `--test`, which stops
after 3 segments/video):

```bash
python3 src/vjepa2_1_segment_feature_extraction.py \
    --annotation_path holoassist/data-annotation-trainval-v1_1.json \
    --video_dir       holoassist/videos \
    --output_dir      features/vjepa2_train \
    --split_file      holoassist/splits/train-v1_2.txt \
    --test
```

### 2a. Convert to `.npy` (faster loading for training)

```bash
sbatch scripts/convert_features_to_npy.sh
```

Converts every `.npz` in the source dirs to a features-only `.npy` at
`.../vjepa2_{train,val}_npy/`, which is what the attentive-probe/world-model
training scripts read from.

## 3. Build dataset indices

**Mistake-detection index** (used by the attentive probe): for every coarse
action, emit one example per prefix of fine-grained sub-actions, with a
binary mistake label on the last sub-action.

```bash
sbatch scripts/build_mistake_dataset.sh      # -> datasets/mistake_detection_index.json      (train)
sbatch scripts/build_mistake_dataset_val.sh  # -> datasets/mistake_detection_index_val.json   (val)
python3 merge_mistake_datasets.py            # -> datasets/mistake_detection_index_full.json  (merged, split by video-name list at train time)
```

**World-model index**: enrich the mistake-detection indices with
per-segment verb/noun IDs and timestamps.

```bash
python3 enrich_index_with_actions.py \
    --train_index  datasets/mistake_detection_index.json \
    --val_index    datasets/mistake_detection_index_val.json \
    --train_npz_dir features/vjepa2_train \
    --val_npz_dir   features/vjepa2_val \
    --train_npy_dir /path/to/vjepa2_train_npy \
    --val_npy_dir   /path/to/vjepa2_val_npy \
    --output_dir    datasets
# -> datasets/wm_train_index.json, datasets/wm_val_index.json
```

## 4. Evaluating the attentive probe

Train (also periodically evaluates on val and writes checkpoints +
`history.json` to `runs/<tag>_<jobid>/`):

```bash
sbatch scripts/train_mistake_detection.sh 10 1e-3 1e-2   # EPOCHS LR WD (see script header for full arg list)
```

Given a trained run directory (e.g. `runs/full_bs128_lr1e-3_wd1e-2_29411291/`,
containing `best_ap_ep08.pt`, `best_macro_f1_ep08.pt`, `last.pt`, ...),
compute benchmark metrics (macro-F1, AP, AUROC, plus random baselines):

```bash
sbatch scripts/eval_benchmark_metrics.sh runs/full_bs128_lr1e-3_wd1e-2_29411291
# equivalent direct call, note --checkpoint must match an actual file in run_dir:
python3 eval_benchmark_metrics.py \
    --run_dir   runs/full_bs128_lr1e-3_wd1e-2_29411291 \
    --checkpoint best_ap_ep08.pt \
    --batch_size 256
```

Dump per-example val predictions (for joining against Qwen/VLM predictions
by `(video_name, coarse_id, target_segment_id)`):

```bash
sbatch scripts/dump_predictions.sh runs/full_bs128_lr1e-3_wd1e-2_29411291
# equivalent direct call:
python3 dump_val_predictions.py \
    --run_dir    runs/full_bs128_lr1e-3_wd1e-2_29411291 \
    --checkpoint best_ap_ep08.pt
```

## 5. Evaluating the world model

Train (self-chaining SLURM job — resubmits itself until `TOTAL_EPOCHS` is
reached, resuming from `last.pt` each time):

```bash
sbatch scripts/train_segment_wm.sh   # -> runs/wm_v1/{last.pt, best_val_l1_ep*.pt}
```

Evaluate a checkpoint on val (anomaly score = per-token L1 between
predicted and actual next-segment features; reports AP, AUROC,
inverse-frequency F1, per-class P/R/F1 — same metric family as the probe):

```bash
python3 eval_segment_wm.py \
    --val_index datasets/wm_val_index.json \
    --ckpt      runs/wm_v1/best_val_l1_ep05.pt \
    --output    runs/wm_v1/eval.json
```

## 6. Evaluating the VLM (Qwen3-VL-8B-Instruct, zero-shot)

Two eval scripts, differing in how much visual context is given per
prediction. Both write one JSON line per example (with prediction +
ground-truth label) to `--out`, and are resumable (skip
already-completed videos on restart).

**Frame-based** (`qwen_val.py`): one frame per sub-action, with prior
sub-actions given as text/labels only.

```bash
sbatch scripts/qwen_val.sh actions_with_labels   # prior-mode: none | actions | actions_with_labels (default)
sbatch scripts/qwen_val.sh none                  # no prior context, target frame only

# equivalent direct call:
python3 qwen_val.py \
    --annotations holoassist/data-annotation-trainval-v1_1.json \
    --video-dir   holoassist/videos \
    --val-list    holoassist/splits/val-v1_2.txt \
    --out         results/qwen_val/actions_with_labels/predictions.jsonl \
    --prior-mode  actions_with_labels \
    --reasoning-mode direct   # or: cot
```

**Clip-based** (`qwen_clips_val.py`): short video clips instead of single
frames, both for prior sub-actions and the target.

```bash
sbatch scripts/qwen_clips_val.sh clips direct              # prior-mode=clips, reasoning=direct, all prior clips
sbatch scripts/qwen_clips_val.sh clips_labelled direct     # oracle: prior clips shown with their ground-truth labels
sbatch scripts/qwen_clips_val.sh none direct                # target clip only, no prior context
sbatch scripts/qwen_clips_val.sh clips cot 4                # chain-of-thought, capped at 4 prior clips

# equivalent direct call:
python3 qwen_clips_val.py \
    --annotations holoassist/data-annotation-trainval-v1_1.json \
    --video-dir   holoassist/videos \
    --val-list    holoassist/splits/val-v1_2.txt \
    --out         results/qwen_clips_val/clips_direct/predictions.jsonl \
    --prior-mode  clips \
    --reasoning-mode direct \
    --max-prior-clips 4   # optional
```

`--prior-mode` choices: `qwen_val.py` -> `none | actions | actions_with_labels`;
`qwen_clips_val.py` -> `none | clips | clips_labelled`.
`--reasoning-mode` choices (both scripts): `direct | cot`.
