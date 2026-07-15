# evaluates a trained SegmentMaskPredictorAC on the holoassist val split.
# anomaly score = per-token mean l1 between normalised predicted and actual next-segment
# features (higher = more likely mistake). reports inverse-frequency-weighted f1 (threshold
# swept on eval set), ap, plus per-class precision/recall/f1 - same metrics as training/train_mistake_detection.py.
# usage: python eval_segment_wm.py --val_index ... --ckpt ... --output ...

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datasets.segment_wm_dataset import SegmentWMDataset, wm_collate_fn
from src.models.segment_predictor import segment_mask_predictor_ac

@torch.no_grad()
def run_eval(model, loader, device, use_bf16: bool) -> dict:
    model.eval()
    scores_all, labels_all = [], []

    for batch in loader:
        ctx_f, ctx_v, ctx_n, tgt_f, tgt_v, tgt_n, labels, ctx_lens = batch
        ctx_f    = ctx_f.to(device, dtype=torch.float32, non_blocking=True)
        tgt_f    = tgt_f.to(device, dtype=torch.float32, non_blocking=True)
        ctx_v    = ctx_v.to(device, non_blocking=True)
        ctx_n    = ctx_n.to(device, non_blocking=True)
        tgt_v    = tgt_v.to(device, non_blocking=True)
        tgt_n    = tgt_n.to(device, non_blocking=True)
        ctx_lens = ctx_lens.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            pred = model(ctx_f, ctx_v, ctx_n, tgt_v, tgt_n, ctx_lengths=ctx_lens)  # [b, t, h*w, e]

        B, T, HW, E = pred.shape
        target = tgt_f.view(B, T, HW, E)

        pred_norm   = model.normalise_targets(pred.float())
        target_norm = model.normalise_targets(target)

        # per-example mean l1 across all token positions and channels.
        l1 = F.l1_loss(pred_norm, target_norm, reduction="none")  # [b, t, hw, e]
        l1_score = l1.mean(dim=(1, 2, 3)).cpu().numpy()           # [b]

        scores_all.append(l1_score)
        labels_all.append(labels.numpy())

    scores = np.concatenate(scores_all)   # [n]
    labels = np.concatenate(labels_all)   # [n]

    # ---- threshold sweep: maximise inverse-frequency-weighted f1 ----------
    n_total = len(labels)
    n_pos   = int(labels.sum())
    n_neg   = n_total - n_pos
    w_pos   = n_total / max(n_pos, 1)
    w_neg   = n_total / max(n_neg, 1)
    w_sum   = w_pos + w_neg

    thresholds = np.linspace(scores.min(), scores.max(), 199)
    best_f1_inv = -1.0
    best_thr    = float(thresholds[len(thresholds) // 2])

    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        _, _, f1_arr, sup_arr = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
        f1_inv = float((f1_arr[0] * w_neg + f1_arr[1] * w_pos) / w_sum)
        if f1_inv > best_f1_inv:
            best_f1_inv = f1_inv
            best_thr    = float(thr)

    preds = (scores >= best_thr).astype(int)
    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    n_neg_true = int(sup_arr[0])
    n_pos_true = int(sup_arr[1])
    w_pos   = n_total / max(n_pos_true, 1)
    w_neg   = n_total / max(n_neg_true, 1)
    w_sum   = w_pos + w_neg

    return {
        "n":           n_total,
        "n_pos":       n_pos,
        "pos_rate":    float(labels.mean()),
        "ap":          float(average_precision_score(labels, scores)),
        "auroc":       float(roc_auc_score(labels, scores)) if n_pos > 0 and n_neg > 0 else float("nan"),
        "best_thr":    best_thr,
        "f1_inv_freq": float((f1_arr[0] * w_neg + f1_arr[1] * w_pos) / w_sum),
        "f1_macro":    float(f1_score(labels, preds, average="macro", zero_division=0)),
        "balanced_acc":float(balanced_accuracy_score(labels, preds)),
        # class 0 – correct
        "prec_neg": float(prec_arr[0]), "rec_neg": float(rec_arr[0]), "f1_neg": float(f1_arr[0]),
        # class 1 – mistake
        "prec_pos": float(prec_arr[1]), "rec_pos": float(rec_arr[1]), "f1_pos": float(f1_arr[1]),
        "mean_l1_correct": float(scores[labels == 0].mean()) if n_neg > 0 else float("nan"),
        "mean_l1_mistake": float(scores[labels == 1].mean()) if n_pos > 0 else float("nan"),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_index",       required=True)
    parser.add_argument("--ckpt",            required=True)
    parser.add_argument("--output",          default=None)
    parser.add_argument("--max_context_segs",        type=int,   default=16)
    parser.add_argument("--max_gap_to_target_secs",  type=float, default=2.0)
    parser.add_argument("--precision",               choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--num_workers",             type=int,   default=4)
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = (args.precision == "bf16" and device.type == "cuda")
    print(f"Device: {device}  precision: {args.precision}", flush=True)

    # ---- load checkpoint metadata to get vocab sizes ----------------------
    raw = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = raw.get("args", {})
    n_verbs = ckpt_args.get("n_verbs", None)
    n_nouns = ckpt_args.get("n_nouns", None)

    # fall back to index metadata if not in ckpt
    if n_verbs is None or n_nouns is None:
        with open(args.val_index) as f:
            idx_meta = json.load(f)
        n_verbs = idx_meta["n_verbs"]
        n_nouns = idx_meta["n_nouns"]

    # ---- model -------------------------------------------------------------
    model = segment_mask_predictor_ac(
        img_size=(384, 384),
        patch_size=16,
        num_frames=64,
        tubelet_size=2,
        embed_dim=ckpt_args.get("embed_dim", 1024),
        predictor_embed_dim=ckpt_args.get("predictor_embed_dim", 1024),
        depth=ckpt_args.get("depth", 24),
        num_heads=ckpt_args.get("num_heads", 16),
        n_verbs=n_verbs,
        n_nouns=n_nouns,
        max_context_segs=args.max_context_segs,
        normalize_targets=ckpt_args.get("normalize_targets", True),
        use_activation_checkpointing=False,  # not needed at eval
    ).to(device)

    model.load_state_dict(raw["model"])
    print(f"Loaded model from {args.ckpt}", flush=True)

    # ---- dataset -----------------------------------------------------------
    val_ds = SegmentWMDataset(args.val_index, train_mode=False,
        max_context_segs=args.max_context_segs, max_gap_to_target_secs=args.max_gap_to_target_secs)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=wm_collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    print(f"Val examples: {len(val_ds)}", flush=True)

    # ---- eval --------------------------------------------------------------
    metrics = run_eval(model, val_loader, device, use_bf16)

    print(
        f"\nResults:"
        f"\n  AP={metrics['ap']:.4f}  AUROC={metrics['auroc']:.4f}"
        f"\n  f1_inv_freq={metrics['f1_inv_freq']:.4f}  f1_macro={metrics['f1_macro']:.4f}"
        f"  bAcc={metrics['balanced_acc']:.4f}  thr={metrics['best_thr']:.4f}"
        f"\n  correct (class 0): P={metrics['prec_neg']:.3f}  R={metrics['rec_neg']:.3f}"
        f"  F1={metrics['f1_neg']:.3f}"
        f"\n  mistake (class 1): P={metrics['prec_pos']:.3f}  R={metrics['rec_pos']:.3f}"
        f"  F1={metrics['f1_pos']:.3f}"
        f"\n  mean_l1_correct={metrics['mean_l1_correct']:.4f}"
        f"  mean_l1_mistake={metrics['mean_l1_mistake']:.4f}"
        f"\n  n={metrics['n']}  n_pos={metrics['n_pos']}"
        f"  pos_rate={metrics['pos_rate']:.3%}",
        flush=True,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nWrote {args.output}", flush=True)

if __name__ == "__main__":
    main()
