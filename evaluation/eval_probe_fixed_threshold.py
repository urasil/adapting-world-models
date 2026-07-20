# evaluates a trained probe (pooled or full-grid) at a threshold fixed in advance,
# default 0.5, instead of sweeping the validation set to maximise the reported metric.
# checkpoint should be one selected by a threshold-free criterion (best_ap.pt),
# so neither the checkpoint nor the threshold has been chosen by looking at
# validation performance under any decision rule.
# usage: python evaluation/eval_probe_fixed_threshold.py --run_dir runs/<run_name> --checkpoint best_ap.pt

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_recall_fscore_support

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.datasets.mistake_detection_dataset import HoloMistakeDataset, collate_fn
from src.models.mistake_classifier import MistakeClassifier
from training.train_mistake_detection import official_video_split


def inv_freq_f1(labels, preds):
    n_total = len(labels)
    prec, rec, f1, sup = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    n_neg, n_pos = int(sup[0]), int(sup[1])
    w_neg, w_pos = n_total / max(n_neg, 1), n_total / max(n_pos, 1)
    w_sum = w_neg + w_pos
    return {
        "f1_inv_freq": float((f1[0] * w_neg + f1[1] * w_pos) / w_sum),
        "f1_correct": float(f1[0]), "f1_mistake": float(f1[1]),
        "prec_mistake": float(prec[1]), "rec_mistake": float(rec[1]),
        "n": n_total, "n_pos": n_pos,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default="best_ap.pt",
                    help="Use a checkpoint chosen by a threshold-free criterion (AP), not best_inv_freq/best_mistake_f1.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Fixed decision threshold, chosen in advance. Default 0.5, the natural cutoff for a probability.")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("args", json.loads((run_dir / "config.json").read_text()))

    features_dir_override = None
    if cfg.get("train_features_dir") and cfg.get("val_features_dir"):
        features_dir_override = [cfg["train_features_dir"], cfg["val_features_dir"]]
    elif cfg.get("features_dir"):
        features_dir_override = cfg["features_dir"]

    full = HoloMistakeDataset(cfg["index_path"], features_dir=features_dir_override, max_segments=cfg["max_segments"])
    _, val_idx = official_video_split(full, cfg["train_split_file"], cfg["val_split_file"])
    val_loader = DataLoader(Subset(full, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = MistakeClassifier(embed_dim=cfg["embed_dim"], num_heads=cfg["num_heads"], depth=cfg["depth"],
                               max_segments=cfg["max_segments"], tokens_per_segment=cfg["tokens_per_segment"]).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    logits_all, labels_all = [], []
    with torch.no_grad():
        for features, labels, lengths in val_loader:
            features = features.to(device, dtype=torch.float32)
            lengths = lengths.to(device)
            logits_all.append(model(features, lengths).float().cpu().numpy())
            labels_all.append(labels.numpy())

    labels = np.concatenate(labels_all)
    probs = 1.0 / (1.0 + np.exp(-np.concatenate(logits_all)))
    preds = (probs >= args.threshold).astype(int)
    m = inv_freq_f1(labels, preds)

    print(f"Run: {run_dir.name}  checkpoint: {args.checkpoint}  threshold: {args.threshold} (fixed, not tuned on val)")
    print(f"n={m['n']}  n_mistake={m['n_pos']}  pos_rate={100*labels.mean():.2f}%")
    print(f"f1_inv_freq={m['f1_inv_freq']:.4f}  f1_correct={m['f1_correct']:.4f}  f1_mistake={m['f1_mistake']:.4f}  "
          f"prec_mistake={m['prec_mistake']:.4f}  rec_mistake={m['rec_mistake']:.4f}")


if __name__ == "__main__":
    main()
