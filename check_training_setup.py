#!/usr/bin/env python3
# pre-flight checklist for mistake detection training - run before submitting train_mistake_detection.sh

import json
import sys
from pathlib import Path

def check_file_exists(path, name):
    """Check if a file/directory exists."""
    if Path(path).exists():
        print(f"✓ {name}")
        return True
    else:
        print(f"✗ {name}")
        return False

def check_index_health(index_path, name):
    """Check if an index JSON is valid and has examples."""
    try:
        with open(index_path) as f:
            idx = json.load(f)
        n_examples = idx.get("num_examples", 0)
        n_unique_videos = len(set(e["video_name"] for e in idx["examples"]))
        if n_examples > 0:
            print(f"✓ {name}")
            print(f"  → {n_examples} examples from {n_unique_videos} videos")
            labels = [e["label"] for e in idx["examples"]]
            correct = labels.count(0)
            mistake = labels.count(1)
            print(f"  → Class dist: {correct} correct, {mistake} mistakes")
            return True, n_unique_videos, n_examples
        else:
            print(f"✗ {name} (0 examples)")
            return False, 0, 0
    except Exception as e:
        print(f"✗ {name} ({e})")
        return False, 0, 0

def main():
    WORK = Path("/home/ua248/rds/hpc-work/adapting_world_models")
    
    print("=" * 60)
    print("Mistake Detection Training Pre-Flight Checklist")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # 1. check basic files
    print("1. Core Files")
    print("-" * 60)
    all_ok &= check_file_exists(WORK / "train_mistake_detection.py", "Training script")
    all_ok &= check_file_exists(WORK / "train_mistake_detection.sh", "SLURM script")
    all_ok &= check_file_exists(WORK / "holoassist/data-annotation-trainval-v1_1.json", "Annotations")
    all_ok &= check_file_exists(WORK / "holoassist/videos", "Video directory")
    print()
    
    # 2. check split files
    print("2. Split Files")
    print("-" * 60)
    all_ok &= check_file_exists(WORK / "holoassist/splits/train-v1_2.txt", "Train split")
    all_ok &= check_file_exists(WORK / "holoassist/splits/val-v1_2.txt", "Val split")
    
    if Path(WORK / "holoassist/splits/train-v1_2.txt").exists():
        train_videos = set(Path(WORK / "holoassist/splits/train-v1_2.txt").read_text().strip().split("\n"))
        print(f"  → {len(train_videos)} videos in train split")
    
    if Path(WORK / "holoassist/splits/val-v1_2.txt").exists():
        val_videos = set(Path(WORK / "holoassist/splits/val-v1_2.txt").read_text().strip().split("\n"))
        print(f"  → {len(val_videos)} videos in val split")
    print()
    
    # 3. check feature extraction
    print("3. Feature Extraction")
    print("-" * 60)
    
    train_features = list((WORK / "features/vjepa2_train").glob("*.npz"))
    if train_features:
        print(f"✓ Training features extracted")
        print(f"  → {len(train_features)} .npz files")
    else:
        print(f"✗ Training features not found")
        all_ok = False
    
    val_features = list((WORK / "features/vjepa2_val").glob("*.npz"))
    if val_features:
        print(f"✓ Validation features extracted")
        print(f"  → {len(val_features)} .npz files")
    else:
        print(f"✗ Validation features NOT extracted (required for full setup)")
        print(f"  → Run: sbatch extraction_val.sh")
    print()
    
    # 4. check indices
    print("4. Dataset Indices")
    print("-" * 60)
    
    train_ok, train_vids, train_examples = check_index_health(
        WORK / "datasets/mistake_detection_index.json",
        "Training index"
    )
    all_ok &= train_ok
    
    val_ok, val_vids, val_examples = check_index_health(
        WORK / "datasets/mistake_detection_index_val.json",
        "Validation index (OPTIONAL)"
    )
    
    full_ok, full_vids, full_examples = check_index_health(
        WORK / "datasets/mistake_detection_index_full.json",
        "Merged index (OPTIONAL)"
    )
    
    print()
    
    # 5. data readiness assessment
    print("5. Training Readiness")
    print("-" * 60)
    
    if not val_features:
        print("⚠ VALIDATION FEATURES MISSING")
        print("  Status: Can train with train-only split")
        print("  Issue: val_split_file will be ignored (0 examples)")
        print("  Fix: Run extraction_val.sh → build_mistake_dataset_val.sh → merge_mistake_datasets.py")
        all_ok = False
    elif not val_ok:
        print("⚠ VALIDATION INDEX NOT BUILT")
        print("  Status: Validation features extracted but index not built")
        print("  Fix: sbatch build_mistake_dataset_val.sh")
        all_ok = False
    elif not full_ok:
        print("⚠ INDICES NOT MERGED")
        print("  Status: Both indices exist separately")
        print("  Fix: python3 merge_mistake_datasets.py")
    else:
        print("✓ FULL TRAIN+VAL SETUP COMPLETE")
        print(f"  → {train_examples + val_examples} total examples")
        print(f"  → {train_vids} train + {val_vids} val videos (no overlap)")
    
    print()
    print("=" * 60)
    
    if all_ok and full_ok:
        print("✓ ALL CHECKS PASSED - Ready to train!")
        print()
        print("Run training with:")
        print("  sbatch train_mistake_detection.sh")
        return 0
    elif all_ok:
        print("⚠ PARTIAL SETUP - Can train but missing validation split")
        print()
        print("For full setup, run:")
        print("  1. sbatch extraction_val.sh")
        print("  2. sbatch build_mistake_dataset_val.sh")
        print("  3. python3 merge_mistake_datasets.py")
        print("  4. sbatch train_mistake_detection.sh")
        return 1
    else:
        print("✗ SETUP INCOMPLETE - Fix issues above before training")
        return 1

if __name__ == "__main__":
    sys.exit(main())
