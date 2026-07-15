# evaluates the best checkpoint: macro-f1 (primary, at threshold 0.5), ap (secondary),
# plus f1* from a threshold sweep (diagnostic upper bound). also runs two random baselines
# for sanity - ours score ~50% macro-f1, not ~28% like holoassist table 7, 
# usage: python eval_benchmark_metrics.py --run_dir runs/<run_name>

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.datasets.mistake_detection_dataset import HoloMistakeDataset, collate_fn
from src.models.mistake_classifier import MistakeClassifier
from training.train_mistake_detection import official_video_split

# metric helpers

def _eval(labels, preds):
    # per-class and macro metrics for binary predictions
    preds = np.asarray(preds, dtype=int)
    return {
        "f1_score":     float(f1_score(labels, preds, average="macro",  zero_division=0)),
        "prec_correct": float(precision_score(labels, preds, pos_label=0, zero_division=0)),
        "rec_correct":  float(recall_score(labels,    preds, pos_label=0, zero_division=0)),
        "f1_correct":   float(f1_score(labels,        preds, pos_label=0, zero_division=0)),
        "prec_mistake": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "rec_mistake":  float(recall_score(labels,    preds, pos_label=1, zero_division=0)),
        "f1_mistake":   float(f1_score(labels,        preds, pos_label=1, zero_division=0)),
    }

def _eval_at_threshold(probs, labels, threshold):
    return _eval(labels, (probs >= threshold).astype(int))

def random_baselines(labels, n_trials=500, seed=42):
    # constant-0 and Monte Carlo stratified-random baselines
    pos_rate = float(labels.mean())
    rng = np.random.default_rng(seed)

    m_const0 = _eval(labels, np.zeros(len(labels), dtype=int))

    trials = [_eval(labels, rng.random(len(labels)) < pos_rate) for _ in range(n_trials)]
    m_strat = {k: float(np.mean([t[k] for t in trials])) for k in trials[0]}

    return m_const0, m_strat, pos_rate

# inference

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    n_batches = len(loader)
    for i, (features, labels, lengths) in enumerate(loader):
        features = features.to(device, non_blocking=True)
        lengths  = lengths.to(device, non_blocking=True)
        logits = model(features, lengths)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(labels.numpy())
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_batches}]", flush=True)
    return np.concatenate(logits_all), np.concatenate(labels_all)

# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",     required=True)
    parser.add_argument("--checkpoint",  default="best_ap.pt")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size",  type=int, default=256)
    args = parser.parse_args()

    run_dir   = Path(args.run_dir)
    cfg       = json.loads((run_dir / "config.json").read_text())
    ckpt_path = run_dir / args.checkpoint

    print("=" * 70)
    print(f"Run:        {run_dir.name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config:     lr={cfg['lr']}  wd={cfg['weight_decay']}  "
          f"depth={cfg['depth']}  max_seg={cfg['max_segments']}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ---- dataset -----------------------------------------------------------
    features_dir_override = None
    if cfg.get("train_features_dir") and cfg.get("val_features_dir"):
        features_dir_override = [cfg["train_features_dir"], cfg["val_features_dir"]]
    elif cfg.get("features_dir"):
        features_dir_override = cfg["features_dir"]

    full = HoloMistakeDataset(cfg["index_path"], features_dir=features_dir_override, max_segments=cfg["max_segments"])
    _, val_idx = official_video_split(full, cfg["train_split_file"], cfg["val_split_file"])
    val_ds = Subset(full, val_idx)

    n_pos    = sum(int(full.examples[i]["label"]) for i in val_idx)
    pos_rate = n_pos / len(val_idx)
    print(f"Val set: {len(val_idx)} examples  ({100*pos_rate:.2f}% positive)", flush=True)

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    # ---- model + checkpoint ------------------------------------------------
    model = MistakeClassifier(
        embed_dim=cfg["embed_dim"], num_heads=cfg["num_heads"],
        depth=cfg["depth"], max_segments=cfg["max_segments"],
        tokens_per_segment=cfg["tokens_per_segment"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    epoch_str = str(ckpt.get("epoch", "?")) if isinstance(ckpt, dict) else "?"
    print(f"Loaded checkpoint (epoch {epoch_str})", flush=True)

    # ---- inference ---------------------------------------------------------
    print("\nRunning inference on val set ...", flush=True)
    logits, labels = run_inference(model, val_loader, device)
    probs = 1.0 / (1.0 + np.exp(-logits))

    # ---- threshold sweep (for f1* upper bound) -----------------------------
    thresholds   = np.linspace(0.01, 0.99, 99)
    macro_f1s    = [_eval_at_threshold(probs, labels, t)["f1_score"] for t in thresholds]
    best_idx     = int(np.argmax(macro_f1s))
    best_thr     = float(thresholds[best_idx])

    m_05         = _eval_at_threshold(probs, labels, 0.5)
    m_best       = _eval_at_threshold(probs, labels, best_thr)
    ap           = float(average_precision_score(labels, probs))
    auroc        = float(roc_auc_score(labels, probs))

    # ---- random baselines --------------------------------------------------
    m_const0, m_strat, _ = random_baselines(labels)

    # ---- print results -----------------------------------------------------
    W = 72
    print("\n" + "=" * W)
    print("RESULTS  (our val split - not strictly comparable to Table 7)")
    print("=" * W)
    print(f"\n  Val: n={len(labels)}  pos_rate={100*labels.mean():.2f}%  epoch={epoch_str}\n")

    hdr = f"  {'Model':<28}  {'F-score':>8}  {'C.Prec':>7}  {'C.Rec':>7}  {'M.Prec':>7}  {'M.Rec':>7}"
    sep = "  " + "-" * (W - 2)
    print(hdr)
    print(sep)

    def row(name, m, note=""):
        s = (f"  {name:<28}  {100*m['f1_score']:8.2f}  "
             f"{100*m['prec_correct']:7.2f}  {100*m['rec_correct']:7.2f}  "
             f"{100*m['prec_mistake']:7.2f}  {100*m['rec_mistake']:7.2f}")
        return s + (f"  {note}" if note else "")

    # holoassist table 7 reference (different metric space - see note below)
    ha = [
        ("HoloAssist Random †",  27.71, 60.87, 10.22, 15.00, 46.15),
        ("HoloAssist RGB †",     35.11, 82.56, 40.51, 12.96, 26.92),
        ("HoloAssist Hands †",   40.19, 92.68, 52.41, 12.50, 31.25),
        ("HoloAssist R+H †",     36.18, 85.51, 43.07,  9.68, 11.54),
        ("HoloAssist R+H+E †",   32.08, 88.57, 42.76, 11.43, 50.00),
    ]
    for name, f, cp, cr, mp, mr in ha:
        print(f"  {name:<28}  {f:8.2f}  {cp:7.2f}  {cr:7.2f}  {mp:7.2f}  {mr:7.2f}")
    print(sep)

    print(row("Ours thr=0.50  [primary]", m_05))
    print(row(f"Ours F1* thr={best_thr:.2f} [upper bnd]", m_best))
    print(sep)

    print(row("Baseline: const-0",        m_const0))
    print(row("Baseline: stratified rand", m_strat))

    print(f"""
  † Table 7 metric appears to differ from global macro-F1.
    Our random baselines score ~{100*m_strat['f1_score']:.0f}% vs Table 7 Random = 27.71.
    Numbers are indicative; the leaderboard test set is held out.
""")

    # ---- secondary metrics -------------------------------------------------
    print(f"  Secondary (threshold-free):")
    print(f"    AP (avg precision): {ap:.4f}")
    print(f"    AUROC:              {auroc:.4f}")

    # ---- classification report at thr=0.5 ----------------------------------
    preds_05 = (probs >= 0.5).astype(int)
    print(f"\n  Full classification report at thr=0.50:")
    print(classification_report(labels, preds_05, target_names=["correct", "mistake"], zero_division=0, digits=4))

    # ---- save --------------------------------------------------------------
    out = {
        "run":        run_dir.name,
        "checkpoint": args.checkpoint,
        "epoch":      epoch_str,
        "primary":    {"threshold": 0.5, **m_05},
        "upper_bound": {"threshold": best_thr, **m_best},
        "baselines":  {"const_0": m_const0, "stratified_random": m_strat},
        "secondary":  {"ap": ap, "auroc": auroc},
    }
    out_path = run_dir / "benchmark_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")
    print("=" * W)

if __name__ == "__main__":
    main()
