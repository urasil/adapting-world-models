# trains MistakeClassifier (attentive probe on frozen v-jepa 2.1 features) on holoassist
# mistake detection.
# follows the v-jepa2 paper's action-anticipation recipe (sec 6, table 19: focal loss + lr/wd
# sweep) rather than its classification recipe (sec 5, table 15) - this task is anticipation-
# shaped (variable context, single label, heavy imbalance), not fixed-clip classification.
# one run = one point in that lr/wd grid; sweep 20x via slurm to match the paper's protocol.
# focal_alpha defaults to 1 - positive_rate, since mistakes are the rare positive class here
# (~4%) and we want to up-weight them, the opposite of the usual coco convention.
# deviations from a strict replication: features are pre-extracted once (fine for a frozen,
# deterministic encoder), spatially mean-pooled to 32x1024 instead of the full 18432 tokens
# (loses spatial detail - worth revisiting if this underperforms), variable-length prefixes
# handled via key_padding_mask, val split by video not example (avoids leaking near-duplicate
# prefixes into val).
# usage: python train_mistake_classifier.py --index_path ... --output_dir ... --epochs 20
#   --batch_size 128 --num_workers 4
# sweep (matches table 19): lr in [5e-3,3e-3,1e-3,3e-4,1e-4] x wd in [1e-4,1e-3,1e-2,1e-1]

from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Sampler
from sklearn.metrics import (average_precision_score, roc_auc_score, balanced_accuracy_score,
                              f1_score, precision_recall_fscore_support)
from mistake_detection_dataset import HoloMistakeDataset, collate_fn
from src.models.mistake_classifier import MistakeClassifier

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_split_file(path: str) -> set[str]:
    """Load a HoloAssist split file: one video name per line, possibly with
    comments (#...) and blank lines. Returns a set of video names."""
    videos = set()
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                videos.add(line)
    if not videos:
        raise ValueError(f"Split file {path} contained no video names.")
    return videos


def official_video_split(dataset: HoloMistakeDataset, train_split_file: str, val_split_file: str) -> tuple[List[int], List[int]]:
    """Partition dataset examples by the HoloAssist official train/val
    split files (one video name per line). Examples whose video_name
    appears in neither file are SILENTLY DROPPED with a warning -- this
    happens when the index is built from a superset of the official
    splits, or when split files use a different naming convention.
    A hard error is more honest if you expect full coverage."""
    train_videos = load_split_file(train_split_file)
    val_videos = load_split_file(val_split_file)

    overlap = train_videos & val_videos
    if overlap:
        raise ValueError(
            f"Train and val split files overlap on {len(overlap)} videos, "
            f"e.g. {sorted(overlap)[:3]}. Refusing to proceed -- this would "
            f"leak training data into validation."
        )

    train_idx, val_idx, dropped = [], [], []
    for i, ex in enumerate(dataset.examples):
        name = ex["video_name"]
        if name in train_videos:
            train_idx.append(i)
        elif name in val_videos:
            val_idx.append(i)
        else:
            dropped.append(name)

    if dropped:
        unique_dropped = sorted(set(dropped))
        print(f"WARNING: dropped {len(dropped)} examples from "
              f"{len(unique_dropped)} videos not listed in either split "
              f"file. First 5: {unique_dropped[:5]}", flush=True)

    if not train_idx:
        raise ValueError(
            f"No training examples after applying split. Either the train "
            f"split file lists videos not in the index, or the index lists "
            f"videos under different names. Sample index video name: "
            f"{dataset.examples[0]['video_name']!r}. Sample split-file "
            f"video name: {next(iter(train_videos))!r}."
        )
    if not val_idx:
        raise ValueError("No validation examples after applying split.")

    return train_idx, val_idx

def video_split(dataset: HoloMistakeDataset, val_frac: float = 0.1,
                seed: int = 0) -> tuple[List[int], List[int]]:
    """Split by video name, not by example. Adjacent sliding-window prefixes
    share most content; a per-example split would leak near-duplicates."""
    rng = random.Random(seed)
    videos = sorted({ex["video_name"] for ex in dataset.examples})
    rng.shuffle(videos)
    n_val = max(1, int(round(len(videos) * val_frac)))
    val_videos = set(videos[:n_val])

    train_idx, val_idx = [], []
    for i, ex in enumerate(dataset.examples):
        (val_idx if ex["video_name"] in val_videos else train_idx).append(i)
    return train_idx, val_idx

def subsample_negatives(dataset: HoloMistakeDataset, indices: List[int], neg_to_pos_ratio: float, seed: int = 0) -> List[int]:
    """Per-video stratified negative subsampling.

    For each video: keep all positives + up to neg_to_pos_ratio × n_pos negatives.
    If the video has fewer negatives than the quota, all its negatives are kept.
    Videos with no positives contribute no examples (no ratio to anchor to).
    """
    rng = random.Random(seed)

    video_pos: dict[str, list[int]] = {}
    video_neg: dict[str, list[int]] = {}
    for i in indices:
        vname = dataset.examples[i]["video_name"]
        if dataset.examples[i]["label"] == 1:
            video_pos.setdefault(vname, []).append(i)
        else:
            video_neg.setdefault(vname, []).append(i)

    kept, total_neg_avail, total_neg_kept, n_capped, n_no_pos = [], 0, 0, 0, 0
    for vname in sorted(set(video_pos) | set(video_neg)):
        pos = video_pos.get(vname, [])
        neg = video_neg.get(vname, [])
        kept.extend(pos)
        quota = round(len(pos) * neg_to_pos_ratio)
        if len(neg) <= quota:
            kept.extend(neg)
            total_neg_kept += len(neg)
        else:
            kept.extend(rng.sample(neg, quota))
            total_neg_kept += quota
            n_capped += 1
        total_neg_avail += len(neg)
        if not pos:
            n_no_pos += 1

    result = sorted(kept)
    print(
        f"Neg subsample (per-video {neg_to_pos_ratio}:1): "
        f"kept {total_neg_kept}/{total_neg_avail} negatives "
        f"({n_capped} videos capped, {n_no_pos} videos with no positives skipped) "
        f"→ {len(result)} total",
        flush=True,
    )
    return result


def task_of_video(video_name: str) -> str:
    """Map a HoloAssist video_name to its task keyword (substring after the
    last '-', e.g. 'gladom_assemble', 'gopro'). IKEA furniture tasks use
    '_assemble'/'_disassemble' suffixes on the furniture name; strip that so
    assemble/disassemble count as the same task."""
    last = video_name.lower().split("-")[-1]
    for suffix in ("_assemble", "_disassemble"):
        if last.endswith(suffix):
            return last[: -len(suffix)]
    return last


def filter_excluded_tasks(dataset: HoloMistakeDataset, indices: List[int],
                           exclude_tasks: List[str]) -> List[int]:
    """Drop every example whose video belongs to one of exclude_tasks.

    Removing whole tasks (rather than randomly subsampling negatives) keeps
    every remaining task's data fully intact, so the model still sees
    complete videos for each task it trains on.
    """
    exclude = {t.lower() for t in exclude_tasks}
    kept = [i for i in indices if task_of_video(dataset.examples[i]["video_name"]) not in exclude]
    print(
        f"Task exclusion ({sorted(exclude)}): kept {len(kept)}/{len(indices)} examples",
        flush=True,
    )
    return kept


def compute_class_stats(dataset: HoloMistakeDataset, indices: List[int]) -> dict:
    labels = np.fromiter((dataset.examples[i]["label"] for i in indices),
                         dtype=np.int64, count=len(indices))
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    return {
        "n": len(labels), "n_pos": n_pos, "n_neg": n_neg,
        "pos_rate": float(labels.mean()),
        "pos_weight": (n_neg / n_pos) if n_pos > 0 else 0.0,
    }

def check_max_segments(dataset: HoloMistakeDataset,
                       indices: List[int], max_segments: int) -> int:
    longest = 0
    for i in indices:
        k = len(dataset.examples[i]["prefix_npz_indices"])
        if k > longest:
            longest = k
    if longest > max_segments:
        print(
            f"WARNING: longest prefix has {longest} segments, but max_segments={max_segments}. "
            f"Prefixes longer than {max_segments} will be truncated to the most recent {max_segments} segments.",
            flush=True,
        )
    return longest


def validate_feature_coverage(dataset: HoloMistakeDataset, indices: List[int]) -> None:
    """Fail fast if any selected video lacks a corresponding feature file."""
    missing_videos = []
    seen_videos = set()
    for i in indices:
        video_name = dataset.examples[i]["video_name"]
        if video_name in seen_videos:
            continue
        seen_videos.add(video_name)
        try:
            dataset._find_feature_file(video_name)
        except FileNotFoundError:
            missing_videos.append(video_name)

    if missing_videos:
        searched = ", ".join(str(d) for d in dataset.features_dirs)
        sample = ", ".join(repr(v) for v in missing_videos[:10])
        raise FileNotFoundError(
            f"Missing feature files for {len(missing_videos)} videos before training. "
            f"First missing: {sample}. Searched: {searched}."
        )

class FocalLossBinary(nn.Module):
    """Binary focal loss on logits (Lin et al., 2017).
        L = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Convention here: alpha is the weight applied to the POSITIVE class
    (target=1). To up-weight the minority class when positives are rare
    (mistake detection on HoloAssist, ~4% positives), set alpha > 0.5.
    The original Lin et al. value of alpha=0.25 was for COCO object
    detection where target=1 was the dense object class -- the semantic
    flip relative to our setup means we want alpha=0.75 by default.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        # p_t = exp(-bce): bce = -log(p_t) by definition. numerically stable.
        # for target=1: p_t = sigmoid(logits) = p.
        # for target=0: p_t = 1 - sigmoid(logits) = 1 - p.
        p_t = torch.exp(-bce)
        # alpha_t: alpha for positives, (1 - alpha) for negatives.
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def cosine_lr(step: int, total_steps: int, base_lr: float,
              final_lr: float = 0.0) -> float:
    """Half-cosine from base_lr to final_lr over total_steps. The paper's
    warmup-constant-cooldown schedule is equivalent in the warmup=0,
    final_lr=0 limit, which is the probe setting."""
    if step >= total_steps:
        return final_lr
    progress = step / max(1, total_steps)
    return final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def evaluate(model, loader, criterion, device, use_bf16: bool) -> dict:
    """Compute val loss + ranking and threshold-based metrics.

    Threshold selection: threshold sweep that maximises positive-class (mistake)
    F1. All per-class precision/recall/F1 are reported at this threshold.

    Reported metrics
    ----------------
    Ranking (threshold-free):
      ap        – average precision (primary; robust to imbalance)
      auroc     – area under ROC curve

    Per-class at best threshold (maximises F1 of positive class):
      prec_neg / rec_neg / f1_neg  – class 0: "no mistake"
      prec_pos / rec_pos / f1_pos  – class 1: "mistake" (rare, ~4 %)

    Aggregates at best threshold:
      f1_macro      – unweighted mean of f1_neg and f1_pos
      f1_inv_freq   – inverse-frequency weighted F1; weight_i = N_total / N_i
                      so the rare mistake class (~4 %) dominates this score
      balanced_acc  – mean of per-class recall (= balanced accuracy)
      best_thr      – threshold chosen by the sweep
    """
    model.eval()
    losses, logits_all, labels_all = [], [], []

    for features, labels, lengths in loader:
        features = features.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(features, lengths)
            loss = criterion(logits, labels)

        losses.append(loss.item() * labels.size(0))
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(labels.cpu().numpy())

    logits_all = np.concatenate(logits_all)
    labels_all = np.concatenate(labels_all)
    probs = 1.0 / (1.0 + np.exp(-logits_all))

    # threshold sweep: maximise positive-class (mistake) f1.
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s_pos = [f1_score(labels_all, probs >= t, zero_division=0) for t in thresholds]
    best_idx  = int(np.argmax(f1s_pos))
    best_thr  = float(thresholds[best_idx])
    preds     = (probs >= best_thr).astype(int)

    # per-class p / r / f1 at the chosen threshold.
    # labels=[0,1] forces both rows even if all predictions fall in one class.
    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        labels_all, preds, labels=[0, 1], zero_division=0
    )

    # inverse-frequency weighted f1.
    # weight_i = n_total / n_i  (proportional to 1/class_freq).
    # normalised so weights sum to 1.  with ~4 % positives the mistake class
    # receives weight ≈ 0.96 and no-mistake ≈ 0.04.
    n_total = len(labels_all)
    n_neg   = int(sup_arr[0])   # true label counts from precision_recall_fscore_support
    n_pos   = int(sup_arr[1])
    w_neg   = n_total / max(n_neg, 1)
    w_pos   = n_total / max(n_pos, 1)
    w_sum   = w_neg + w_pos
    f1_inv_freq = float((f1_arr[0] * w_neg + f1_arr[1] * w_pos) / w_sum)

    return {
        "loss":         float(np.sum(losses) / n_total),
        "ap":           float(average_precision_score(labels_all, probs)),
        "auroc":        float(roc_auc_score(labels_all, probs)),
        "best_thr":     best_thr,
        # class 0 – no mistake
        "prec_neg":     float(prec_arr[0]),
        "rec_neg":      float(rec_arr[0]),
        "f1_neg":       float(f1_arr[0]),
        # class 1 – mistake
        "prec_pos":     float(prec_arr[1]),
        "rec_pos":      float(rec_arr[1]),
        "f1_pos":       float(f1_arr[1]),
        # aggregates
        "f1_macro":     float(f1_score(labels_all, preds, average="macro",    zero_division=0)),
        "f1_inv_freq":  f1_inv_freq,
        "balanced_acc": float(balanced_accuracy_score(labels_all, preds)),
        # dataset info
        "n":            n_total,
        "n_pos":        n_pos,
        "pos_rate":     float(labels_all.mean()),
    }

def train_one_epoch(model, loader, optimizer, criterion, device, *, base_lr: float, total_steps: int, start_step: int, grad_clip: float, use_bf16: bool, log_every: int, accum_steps: int = 1) -> dict:
    model.train()
    running, n_seen, step, n_skipped = 0.0, 0, start_step, 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    n_micro = len(loader)
    for it, (features, labels, lengths) in enumerate(loader):
        if (lengths == 0).any():
            print(f"WARNING: batch at iter {it+1} has example(s) with zero-length prefix, skipping", flush=True)
            n_skipped += 1
            continue

        features = features.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        # set lr on micro-batches that will trigger an optimizer step. we do
        # this on every micro-batch for simplicity; param_groups update is
        # cheap and the value is constant within an accumulation window
        # anyway (step only increments after .step()).
        lr = cosine_lr(step, total_steps, base_lr=base_lr, final_lr=0.0)
        for g in optimizer.param_groups:
            g["lr"] = lr

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(features, lengths)
            loss = criterion(logits, labels)

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            # don't call backward — nan gradients crash flash attention's backward kernel.
            # don't zero_grad either: let the rest of the accumulation window proceed normally
            # so the optimizer step at the next boundary still fires with a full (minus one)
            # window of valid gradients.
            print(f"WARNING: non-finite loss ({loss_val}) at iter {it+1}, skipping batch", flush=True)
            n_skipped += 1
            continue

        (loss / accum_steps).backward()

        running += loss_val * labels.size(0)
        n_seen += labels.size(0)

        is_accum_boundary = ((it + 1) % accum_steps == 0)
        is_last_micro = (it + 1 == n_micro)
        if is_accum_boundary or is_last_micro:
            if grad_clip is not None and grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

        if (it + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"    iter {it+1}/{n_micro}  loss={running/n_seen:.4f}  "
                  f"lr={lr:.2e}  ips={n_seen/elapsed:.1f}", flush=True)

    return {"loss": running / max(1, n_seen), "end_step": step, "n_skipped": n_skipped}


class VideoGroupedSampler(Sampler):
    """Shuffle at video level, preserve within-video order.

    Each epoch: randomise the order of videos, then emit all examples from
    each video consecutively. This keeps NFS page-cache hot across examples
    from the same file while still providing inter-video diversity in every
    optimizer step.
    """
    def __init__(self, dataset: Subset, seed: int = 0):
        self.epoch = 0
        self.seed = seed
        full = dataset.dataset
        # map each subset index → video_name
        video_to_indices: dict[str, list[int]] = {}
        for subset_pos, full_idx in enumerate(dataset.indices):
            vname = full.examples[full_idx]["video_name"]
            video_to_indices.setdefault(vname, []).append(subset_pos)
        # sort within each video by full_idx so prefix lengths are ascending
        for vname in video_to_indices:
            video_to_indices[vname].sort(
                key=lambda pos: dataset.indices[pos]
            )
        self.groups = list(video_to_indices.values())

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.groups), generator=rng).tolist()
        for g in order:
            yield from self.groups[g]

    def __len__(self):
        return sum(len(g) for g in self.groups)


def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--index_path", required=True, type=str)
    parser.add_argument("--features_dir", default=None, type=str,
                        help="Override features_dir stored in the index.")
    parser.add_argument("--val_frac", type=float, default=0.1)
    # model (must match feature dims -- v-jepa 2.1 vit-l gives 1024)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--max_segments", type=int, default=64)
    parser.add_argument("--tokens_per_segment", type=int, default=32)
    # loss -- defaults set for binary imbalanced anticipation-like setting
    parser.add_argument("--loss", choices=["focal", "bce_posw"], default="focal",
                        help="focal: focal loss (paper-faithful for EK100-style "
                             "imbalanced label space). bce_posw: BCEWithLogitsLoss "
                             "with pos_weight = N_neg/N_pos.")
    parser.add_argument("--focal_alpha", type=float, default=None,
                        help="Weight on the POSITIVE class. If omitted, use "
                             "1 - positive_rate from the training split, which "
                             "directly matches the observed class imbalance.")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focusing parameter from focal loss. 2.0 is the "
                             "standard default and mildly down-weights easy examples.")
    # optimisation -- one point in the paper's anticipation-probe grid
    # (table 19): lr in {5e-3, 3e-3, 1e-3, 3e-4, 1e-4}, wd in {1e-4, 1e-3,
    # 1e-2, 1e-1}, batch_size 128, 20 epochs. sweep these externally.
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                        help="bf16 uses torch.autocast on cuda; fp32 disables it.")
    # runtime
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--neg_subsample_ratio", type=float, default=None,
                        help="Keep all positives + this many × n_pos negatives. "
                             "E.g. 4.0 keeps 4:1 neg:pos. Speeds up training by "
                             "reducing dataset size; pos_weight is recomputed.")
    parser.add_argument("--exclude_tasks", type=str, default=None,
                        help="Comma-separated task keywords to drop entirely "
                             "from both train and val (e.g. "
                             "'gladom,marius,knarrevik,rashult' to drop the "
                             "IKEA furniture tasks). Matched against the "
                             "task keyword parsed from video_name. Unlike "
                             "--neg_subsample_ratio, this removes whole "
                             "videos rather than randomly dropping negatives, "
                             "so every remaining task stays fully intact.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from output_dir/last.pt if it exists.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                    help="Accumulate gradients over this many micro-batches "
                         "before stepping. Effective batch size = "
                         "batch_size * grad_accum_steps. Use to recover the "
                         "paper's bs=128 setting when GPU memory only fits "
                         "smaller micro-batches.")
    parser.add_argument("--train_split_file", default=None, type=str,
                    help="Path to HoloAssist train split (one video name "
                         "per line). If set together with --val_split_file, "
                         "uses the official split and --val_frac is ignored.")
    parser.add_argument("--val_split_file", default=None, type=str,
                    help="Path to HoloAssist val split. See --train_split_file.")
    parser.add_argument("--train_features_dir", default=None, type=str,
                    help="Override features directory for training split. "
                         "Useful when train and val features are in separate dirs.")
    parser.add_argument("--val_features_dir", default=None, type=str,
                    help="Override features directory for validation split. "
                         "Useful when train and val features are in separate dirs.")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(args.precision == "bf16" and device.type == "cuda")
    print(f"Device: {device}  precision: {args.precision} "
          f"(autocast={use_bf16})", flush=True)

    # ---- dataset & split ---------------------------------------------------
    # support separate train/val feature directories. holomistakedataset
    # already sanitizes accidental quote-wrapped values.
    if args.train_features_dir and args.val_features_dir:
        features_dir_override = [args.train_features_dir, args.val_features_dir]
    elif args.train_features_dir:
        features_dir_override = args.train_features_dir
    elif args.val_features_dir:
        features_dir_override = args.val_features_dir
    elif args.features_dir:
        features_dir_override = args.features_dir
    else:
        features_dir_override = None

    full = HoloMistakeDataset(
        args.index_path,
        features_dir=features_dir_override,
        max_segments=args.max_segments,
    )
    #train_idx, val_idx = video_split(full, val_frac=args.val_frac, seed=args.seed)
    train_idx, val_idx = official_video_split(full, args.train_split_file, args.val_split_file)
    if args.exclude_tasks is not None:
        exclude_tasks = [t.strip() for t in args.exclude_tasks.split(",") if t.strip()]
        train_idx = filter_excluded_tasks(full, train_idx, exclude_tasks)
        val_idx = filter_excluded_tasks(full, val_idx, exclude_tasks)
    if args.neg_subsample_ratio is not None:
        train_idx = subsample_negatives(full, train_idx, args.neg_subsample_ratio, seed=args.seed)
    longest = check_max_segments(full, train_idx + val_idx, args.max_segments)
    validate_feature_coverage(full, train_idx + val_idx)
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val "
          f"(out of {len(full)} total)", flush=True)
    print(f"Longest prefix: {longest} segments (max_segments={args.max_segments})",
          flush=True)

    stats = compute_class_stats(full, train_idx)
    if stats["n_pos"] == 0:
        raise ValueError("Training split contains no positive examples; class weighting is undefined.")
    print(f"Train class stats: n={stats['n']}  pos={stats['n_pos']} "
          f"({100*stats['pos_rate']:.2f}%)  pos_weight={stats['pos_weight']:.2f}",
          flush=True)

    train_ds = Subset(full, train_idx)
    val_ds = Subset(full, val_idx)

    pin = device.type == "cuda"
    train_sampler = VideoGroupedSampler(train_ds, seed=args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=pin, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=pin, persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # ---- model ------------------------------------------------------------
    model = MistakeClassifier(
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        depth=args.depth, max_segments=args.max_segments,
        tokens_per_segment=args.tokens_per_segment,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params/1e6:.2f}M", flush=True)

    # ---- optimiser, loss --------------------------------------------------
    # adamw; single param group (probe is small, no ln/bias group split).
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )

    if args.loss == "focal":
        focal_alpha = args.focal_alpha if args.focal_alpha is not None else (1.0 - stats["pos_rate"])
        criterion = FocalLossBinary(alpha=focal_alpha, gamma=args.focal_gamma)
        print(
            f"Loss: focal(alpha={focal_alpha:.4f}, gamma={args.focal_gamma}) "
            f"[alpha={focal_alpha:.4f} = 1 - train_pos_rate={1.0 - stats['pos_rate']:.4f}]",
            flush=True,
        )
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(stats["pos_weight"], device=device,
                                    dtype=torch.float)
        )
        print(f"Loss: BCEWithLogitsLoss(pos_weight={stats['pos_weight']:.2f})",
              flush=True)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps) 
    total_steps = steps_per_epoch * args.epochs
    effective_bs = args.batch_size * args.grad_accum_steps
    print(f"Total optimisation steps: {total_steps} "
      f"({steps_per_epoch}/epoch x {args.epochs} epochs)", flush=True)

    # ---- resume -----------------------------------------------------------
    history = []
    best = {
        "epoch": -1, "ap": -1.0,
        "macro_f1": -1.0, "epoch_macro_f1": -1,
        "inv_freq": -1.0, "epoch_inv_freq": -1,
        "mistake_f1": -1.0, "epoch_mistake_f1": -1,
    }
    start_epoch = 1
    step = 0

    resume_path = output_dir / "last.pt"
    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        step = ckpt["epoch"] * steps_per_epoch
        raw_best = ckpt.get("best", {})
        for k in best:
            best[k] = raw_best.get(k, best[k])
        if (output_dir / "history.json").exists():
            with open(output_dir / "history.json") as f:
                history = json.load(f)
        print(f"Resumed from epoch {ckpt['epoch']}  "
              f"(step={step}, best_ap={best['ap']:.4f})", flush=True)
    elif args.resume:
        print("--resume set but no checkpoint found, starting from scratch.", flush=True)

    # ---- train ------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        t_epoch = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            base_lr=args.lr, total_steps=total_steps, start_step=step,
            grad_clip=args.grad_clip, use_bf16=use_bf16,
            log_every=args.log_every, accum_steps=args.grad_accum_steps,
        )
        step = train_stats["end_step"]
        val = evaluate(model, val_loader, criterion, device, use_bf16)

        elapsed = time.time() - t_epoch
        epoch_log = {
            "epoch": epoch, "train_loss": train_stats["loss"],
            "skipped_batches": train_stats["n_skipped"],
            **{f"val_{k}": v for k, v in val.items()},
            "time_s": elapsed,
        }
        history.append(epoch_log)
        eta_s = elapsed * (args.epochs - epoch)
        eta_str = f"{eta_s/3600:.1f}h" if eta_s >= 3600 else f"{eta_s/60:.0f}min"
        print(
            f"[epoch {epoch:02d}/{args.epochs}]  "
            f"tr_loss={train_stats['loss']:.4f}  v_loss={val['loss']:.4f}  "
            f"AP={val['ap']:.4f}  AUROC={val['auroc']:.4f}  "
            f"thr={val['best_thr']:.2f}  skipped={train_stats['n_skipped']}  "
            f"({elapsed:.0f}s, ETA {eta_str})",
            flush=True,
        )
        print(
            f"           "
            f"neg(no-mistake):  P={val['prec_neg']:.3f}  R={val['rec_neg']:.3f}  F1={val['f1_neg']:.3f}"
            f"  |  "
            f"pos(mistake):     P={val['prec_pos']:.3f}  R={val['rec_pos']:.3f}  F1={val['f1_pos']:.3f}"
            f"  |  "
            f"macro={val['f1_macro']:.3f}  inv-freq-wt={val['f1_inv_freq']:.3f}  bAcc={val['balanced_acc']:.3f}"
            f"  (n_pos={val['n_pos']})",
            flush=True,
        )

        # ckpt holds a reference to `best`; update best before torch.save calls
        # so every file reflects the current best state.
        ckpt = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args), "metrics": epoch_log,
            "best": best,
        }

        improved = []
        for metric_key, metric_val, epoch_key, file_tag in [
            ("ap",         val["ap"],         "epoch",           "best_ap"),
            ("macro_f1",   val["f1_macro"],   "epoch_macro_f1",  "best_macro_f1"),
            ("inv_freq",   val["f1_inv_freq"],"epoch_inv_freq",  "best_inv_freq"),
            ("mistake_f1", val["f1_pos"],     "epoch_mistake_f1","best_mistake_f1"),
        ]:
            if metric_val > best[metric_key]:
                best[metric_key] = metric_val
                best[epoch_key] = epoch
                improved.append((file_tag, metric_val))

        torch.save(ckpt, output_dir / "last.pt")

        # best checkpoints — epoch in filename, previous file deleted on update.
        for file_tag, metric_val in improved:
            for old in output_dir.glob(f"{file_tag}_ep*.pt"):
                old.unlink()
            torch.save(ckpt, output_dir / f"{file_tag}_ep{epoch:02d}.pt")
        if improved:
            msgs = [f"{tag}={v:.4f}(ep{epoch:02d})" for tag, v in improved]
            print(f"  -> new best: {', '.join(msgs)}", flush=True)

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(
        f"\nDone."
        f"  best_ap={best['ap']:.4f}(ep{best['epoch']:02d})"
        f"  best_macro_f1={best['macro_f1']:.4f}(ep{best['epoch_macro_f1']:02d})"
        f"  best_inv_freq={best['inv_freq']:.4f}(ep{best['epoch_inv_freq']:02d})"
        f"  best_mistake_f1={best['mistake_f1']:.4f}(ep{best['epoch_mistake_f1']:02d})",
        flush=True,
    )

if __name__ == "__main__":
    main()
