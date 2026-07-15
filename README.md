# Running the Pipeline

1. Download the HoloAssist dataset.
2. Extract V-JEPA 2.1 features from the videos.
3. Build the dataset indices used for training.
4. Train and evaluate the attentive probe.
5. Train and evaluate the world model.
6. Evaluate the VLM baseline (Qwen), no training needed.

Quick note on the commands, since this is run on the Cambridge cluster, most steps are launched with `sbatch some_script.sh`, which submits the job to that cluster's SLURM scheduler. Every `sbatch` command is wrapped by a simple `python3` command that actually does the work. 

## 0. Set up the environment

```bash
python3 -m venv venv_ampere
source venv_ampere/bin/activate
pip install -r requirements.txt
export HF_HOME="$WORK/hf_cache"
```

The V-JEPA 2.1 encoder architecture is pulled from `facebookresearch/vjepa2` through `torch.hub`. The weights themselves are loaded separately from a local checkpoint file, `vjepa2_1_vitl_dist_vitG_384.pt`, which is in the GitHub repo. 

## 1. Get the HoloAssist dataset

```bash
cd holoassist
sbatch download_holoassist.sh
```

This downloads three things into `holoassist/`:

- `video_compress.tar`, the compressed videos. Extract it so each clip ends up at `holoassist/videos/<video_name>/Export_py/Video_compress.mp4`.
- `data-annotation-trainval-v1_1.json`, the action and mistake annotations.
- `data-splits-v1_2.zip`, the train, val, and test split lists. Unzip it into `holoassist/splits/{train,val,test}-v1_2.txt`.

## 2. Extract V-JEPA 2.1 features

The script that does this is `src/vjepa2_1_segment_feature_extraction.py`. For every fine-grained action segment in HoloAssist, it samples 64 frames and runs them through the V-JEPA 2.1 encoder. What comes out for each segment is the full set of 18,432 tokens (32 temporal positions by 576 spatial positions), each 1024-dimensional.

For every video, this writes two files into the output directory:

- `<video_name>.npy`, the features themselves, shaped `(num_segments, 32, 576, 1024)`. This is a plain, uncompressed `.npy` file, as compression doesn't provide the storage benefit I previously thought it would.
- `<video_name>_meta.npz`, a small file holding everything else about each segment: its label, verb, noun, start and end times, and segment ID. This stays tiny even though the features file next to it can be huge, and it's what the dataset-building scripts in step 3 read from.

```bash
sbatch scripts/extraction.sh       #train split, writes to features/vjepa2_train/
sbatch scripts/extraction_val.sh   #val split, writes to features/vjepa2_val/
```

The plain call behind those (test flag is useful for a quick 3 segment CPU smoke test).

```bash
python3 src/vjepa2_1_segment_feature_extraction.py \
    --annotation_path holoassist/data-annotation-trainval-v1_1.json \
    --video_dir       holoassist/videos \
    --output_dir      features/vjepa2_train \
    --split_file      holoassist/splits/train-v1_2.txt \
    --test
```

### Optional: pooling the features down

Roughly 576 times smaller on disk.

```bash
python3 data_prep/convert_features_spatialpool.py --split both
```

Note that this script currently has its input and output directories
hardcoded at the top of the file rather than taken as arguments.

## 3. Build the dataset indices

Two indices are built here, the second layered on top of the first.

The mistake-detection index is what the attentive probe trains on. For every coarse action, it creates one training example per prefix of
fine-grained sub-actions within that action, with a binary mistake label on the final sub-action in each prefix.

```bash
sbatch scripts/build_mistake_dataset.sh      #writes datasets/mistake_detection_index.json (train)
sbatch scripts/build_mistake_dataset_val.sh  #writes datasets/mistake_detection_index_val.json (val)
python3 data_prep/merge_mistake_datasets.py  #merges both into datasets/mistake_detection_index_full.json
```

(The merged file is what training actually loads. It gets split back into train and val at training time using the video-name split lists from step 1.)

The world-model index takes those same indices and adds the extra fields the world model needs but the probe doesn't -> per-segment verb and noun IDs,
plus timestamps. This information comes from the `_meta.npz` files written in step 2.

```bash
python3 data_prep/enrich_index_with_actions.py \
    --train_index        datasets/mistake_detection_index.json \
    --val_index           datasets/mistake_detection_index_val.json \
    --train_features_dir  features/vjepa2_train \
    --val_features_dir    features/vjepa2_val \
    --output_dir           datasets
```

This writes `datasets/wm_train_index.json` and `datasets/wm_val_index.json`.

## 4. Train and evaluate the attentive probe

```bash
sbatch scripts/train_mistake_detection.sh 10 1e-3 1e-2 #arguments are EPOCHS LR WD, see the script header for the full list
```

Once you have a trained run directory, something like `runs/full_bs128_lr1e-3_wd1e-2_29411291/`, containing files such as `best_ap_ep08.pt`, `best_macro_f1_ep08.pt`, and `last.pt`.

Computing eval metrics:

```bash
sbatch scripts/eval_benchmark_metrics.sh runs/full_bs128_lr1e-3_wd1e-2_29411291

#the plain call, --checkpoint has to match a real file in run_dir
python3 evaluation/eval_benchmark_metrics.py \
    --run_dir   runs/full_bs128_lr1e-3_wd1e-2_29411291 \
    --checkpoint best_ap_ep08.pt \
    --batch_size 256
```

You can also dump per-example predictions on val, which is useful for comparing against the Qwen predictions by matching on `(video_name, coarse_id, target_segment_id)`.

```bash
sbatch scripts/dump_predictions.sh runs/full_bs128_lr1e-3_wd1e-2_29411291

python3 evaluation/dump_val_predictions.py \
    --run_dir    runs/full_bs128_lr1e-3_wd1e-2_29411291 \
    --checkpoint best_ap_ep08.pt
```

## 5. Train and evaluate the world model

```bash
sbatch scripts/train_segment_wm.sh #writes runs/wm_v1/last.pt and runs/wm_v1/best_val_l1_ep*.pt
```

To evaluate a checkpoint on val, the per-token L1 distance between the predicted and actual next-segment features is used. This reports AP, AUROC, inverse-frequency F1, and per-class precision, recall, and F1, the same metrics used for the probe, so the two are directly comparable.

```bash
python3 evaluation/eval_segment_wm.py \
    --val_index datasets/wm_val_index.json \
    --ckpt      runs/wm_v1/best_val_l1_ep05.pt \
    --output    runs/wm_v1/eval.json
```

## 6. Evaluate the VLM baseline (Qwen3-VL-8B-Instruct, zero-shot)

There's no training in this stage, only zero-shot evaluation. Here, we write one JSON line per example, with the prediction and the ground-truth label, to whatever path you pass as `--out`.

`qwen/qwen_clips_val.py` shows the model short video clips for the target sub-action, with prior sub-actions optionally shown as clips too.

```bash
sbatch scripts/qwen_clips_val.sh clips_labelled direct #oracle mode: prior clips shown together with their ground-truth labels
sbatch scripts/qwen_clips_val.sh clips cot 12 #chain-of-thought reasoning, capped at 12 prior clips

python3 qwen/qwen_clips_val.py \
    --annotations holoassist/data-annotation-trainval-v1_1.json \
    --video-dir   holoassist/videos \
    --val-list    holoassist/splits/val-v1_2.txt \
    --out         results/qwen_clips_val/clips_direct/predictions.jsonl \
    --prior-mode  clips \
    --reasoning-mode direct \
    --max-prior-clips 4   # optional
```

The `--reasoning-mode` choices are `direct` and `cot`.

