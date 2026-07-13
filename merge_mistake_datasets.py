#!/usr/bin/env python3
# merges train and val mistake-detection indices into one dataset index

import json
import sys
from pathlib import Path

def main():
    # load both indices
    train_path = Path("datasets/mistake_detection_index.json")
    val_path = Path("datasets/mistake_detection_index_val.json")
    
    if not train_path.exists():
        print(f"ERROR: {train_path} not found", file=sys.stderr)
        sys.exit(1)
    if not val_path.exists():
        print(f"ERROR: {val_path} not found", file=sys.stderr)
        sys.exit(1)
    
    with open(train_path) as f:
        train_idx = json.load(f)
    
    with open(val_path) as f:
        val_idx = json.load(f)
    
    # merge examples
    all_examples = train_idx["examples"] + val_idx["examples"]

    # features_dir should differ (train/val loaded separately); fall back to train's either way
    if train_idx["features_dir"] == val_idx["features_dir"]:
        features_dir = train_idx["features_dir"]
    else:
        features_dir = train_idx["features_dir"]

    # build merged index
    merged = {
        "features_dir": features_dir,
        "feature_key": "features",
        "num_examples": len(all_examples),
        "examples": all_examples,
        "note": "Merged train and val sets; features are in separate directories"
    }
    
    # save merged index
    out_path = Path("datasets/mistake_detection_index_full.json")
    with open(out_path, "w") as f:
        json.dump(merged, f)
    
    print(f"Merged {len(train_idx['examples'])} train + {len(val_idx['examples'])} val examples")
    print(f"Total: {len(all_examples)} examples")
    
    # stats
    labels = [e["label"] for e in all_examples]
    correct = labels.count(0)
    mistake = labels.count(1)
    print(f"\nClass distribution:")
    print(f"  Correct (0): {correct:7d} ({100*correct/len(all_examples):5.2f}%)")
    print(f"  Mistake (1): {mistake:7d} ({100*mistake/len(all_examples):5.2f}%)")
    print(f"\nWrote: {out_path}")
    
    # check train/val don't overlap
    train_videos = {e["video_name"] for e in train_idx["examples"]}
    val_videos = {e["video_name"] for e in val_idx["examples"]}
    
    if train_videos & val_videos:
        print(f"WARNING: {len(train_videos & val_videos)} videos appear in both train and val!")
    else:
        print(f"✓ No video overlap between train ({len(train_videos)}) and val ({len(val_videos)} videos)")

if __name__ == "__main__":
    main()
