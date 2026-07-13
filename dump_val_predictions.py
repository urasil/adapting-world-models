# dumps per-example val predictions to jsonl for comparison with qwen.
# each line: {video_name, coarse_id, target_segment_id, label, prob, pred}
# join key vs qwen results: (video_name, coarse_id, target_segment_id)
# usage: python dump_val_predictions.py --run_dir runs/<run_name>

import argparse
import json
import sys
from pathlib import Path

print("imports: stdlib done", flush=True)
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
print("imports: torch done", flush=True)

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from mistake_detection_dataset import HoloMistakeDataset, collate_fn
print("imports: dataset done", flush=True)
from src.models.mistake_classifier import MistakeClassifier
print("imports: model done", flush=True)
from train_mistake_detection import official_video_split
print("imports: all done", flush=True)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",     required=True)
    parser.add_argument("--checkpoint",  default="best_ap.pt")
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size",  type=int, default=256)
    args = parser.parse_args()

    run_dir   = Path(args.run_dir)
    cfg       = json.loads((run_dir / "config.json").read_text())
    ckpt_path = run_dir / args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run: {run_dir.name}  checkpoint: {args.checkpoint}  device: {device}", flush=True)

    # ---- dataset ----
    features_dir_override = None
    if cfg.get("train_features_dir") and cfg.get("val_features_dir"):
        features_dir_override = [cfg["train_features_dir"], cfg["val_features_dir"]]
    elif cfg.get("features_dir"):
        features_dir_override = cfg["features_dir"]

    full = HoloMistakeDataset(
        cfg["index_path"],
        features_dir=features_dir_override,
        max_segments=cfg["max_segments"],
    )
    _, val_idx = official_video_split(full, cfg["train_split_file"], cfg["val_split_file"])

    # collect metadata in the same order as the dataloader (shuffle=false)
    meta = [
        {
            "video_name":        full.examples[i]["video_name"],
            "coarse_id":         full.examples[i]["coarse_id"],
            "target_segment_id": full.examples[i]["target_segment_id"],
            "label":             int(full.examples[i]["label"]),
        }
        for i in val_idx
    ]

    val_ds = Subset(full, val_idx)
    print(f"Val set: {len(val_idx)} examples", flush=True)

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    # ---- model + checkpoint ----
    model = MistakeClassifier(
        embed_dim=cfg["embed_dim"], num_heads=cfg["num_heads"],
        depth=cfg["depth"], max_segments=cfg["max_segments"],
        tokens_per_segment=cfg["tokens_per_segment"],
    ).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    print(f"Loaded checkpoint (epoch {epoch})", flush=True)

    # ---- inference ----
    print("Running inference ...", flush=True)
    logits, labels = run_inference(model, val_loader, device)
    probs = 1.0 / (1.0 + np.exp(-logits))

    assert len(probs) == len(meta), f"length mismatch: {len(probs)} vs {len(meta)}"
    assert all(int(labels[j]) == meta[j]["label"] for j in range(len(labels))), \
        "label mismatch between dataset meta and DataLoader output"

    # ---- write jsonl ----
    out_path = run_dir / "val_predictions.jsonl"
    with open(out_path, "w") as f:
        for m, p in zip(meta, probs):
            row = {**m, "prob": float(p), "pred": int(p >= args.threshold)}
            f.write(json.dumps(row) + "\n")

    n_pred_pos = int((probs >= args.threshold).sum())
    print(f"\nWrote {len(meta)} lines to {out_path}")
    print(f"Predictions at thr={args.threshold}: {n_pred_pos} positive ({100*n_pred_pos/len(meta):.2f}%)")
    print(f"Ground truth:                        {int(labels.sum())} positive ({100*labels.mean():.2f}%)")


if __name__ == "__main__":
    main()
