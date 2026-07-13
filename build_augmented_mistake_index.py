# builds a combined index that adds augmented mistake examples to the original training index,
# bumping the mistake class from ~1:20 up to ~1:4 vs correct (n_correct~118058, n_mistake~5996).
# augments the whole npz per video, not just mistakes, so the model can't learn "augmented style = mistake"
# from an un-augmented prefix history.
# usage: python build_augmented_mistake_index.py --original_index ... --aug_features_dir ...
#   --aug_names warm_tint cold_tint crop_90 warm_crop --output_path ... --target_ratio 4 [--dry_run]

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


# helpers

def load_index(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def class_counts(examples: list[dict]) -> tuple[int, int]:
    """Return (n_correct, n_mistake) for a list of example dicts."""
    labels = [e["label"] for e in examples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    return n_neg, n_pos   # (correct, mistake)


def make_augmented_example(orig: dict, aug_name: str) -> dict:
    """
    Return a copy of *orig* with the video_name suffixed by _aug_{aug_name}.
    All index fields (prefix_npz_indices, label, coarse_id, …) are preserved
    verbatim; only the video_name changes so that the feature loader picks up
    the corresponding augmented .npz.
    """
    return {
        **orig,
        "video_name": f"{orig['video_name']}_aug_{aug_name}",
    }


def check_aug_coverage(
    mistake_examples: list[dict],
    aug_features_dir: Path,
    aug_name: str,
) -> tuple[list[dict], list[str]]:
    """
    For a given aug_name, return (coverable_examples, missing_video_names).
    An example is coverable if its augmented .npz file exists on disk.
    """
    coverable = []
    missing_videos: set[str] = set()
    for ex in mistake_examples:
        aug_video = f"{ex['video_name']}_aug_{aug_name}"
        npz = aug_features_dir / f"{aug_video}.npz"
        if npz.exists():
            coverable.append(ex)
        else:
            missing_videos.add(ex["video_name"])
    return coverable, sorted(missing_videos)


# main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine original + augmented mistake examples to reach "
                    "target class ratio."
    )
    p.add_argument(
        "--original_index", required=True,
        help="Path to the original dataset index JSON "
             "(e.g. datasets/mistake_detection_index.json).",
    )
    p.add_argument(
        "--aug_features_dir", required=True,
        help="Directory that contains the augmented .npz files produced by "
             "extract_augmented_features.py "
             "(e.g. features/vjepa2_train_aug).",
    )
    p.add_argument(
        "--aug_names", nargs="+",
        default=["warm_tint", "cold_tint", "crop_90", "warm_crop"],
        help="Augmentation variant names to consider (in priority order). "
             "Default: warm_tint cold_tint crop_90 warm_crop",
    )
    p.add_argument(
        "--output_path", required=True,
        help="Path to write the combined index JSON.",
    )
    p.add_argument(
        "--target_ratio", type=float, default=4.0,
        help="Desired  n_correct / n_mistake  ratio in the output index. "
             "Default 4 → mistake class is 1/4 the correct class. "
             "Use 0 to include all available augmented examples.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print statistics and exit without writing any file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    original_index  = Path(args.original_index)
    aug_features_dir = Path(args.aug_features_dir)
    output_path     = Path(args.output_path)

    # load original index
    idx = load_index(str(original_index))
    original_examples = idx["examples"]
    n_correct_orig, n_mistake_orig = class_counts(original_examples)
    n_total_orig = len(original_examples)

    # features directory from original index
    orig_features_dir = idx.get("features_dir") or (
        idx.get("features_dirs", [None])[0]
    )

    print(f"\n{'='*60}")
    print("Build Augmented Mistake Index")
    print(f"{'='*60}")
    print(f"  original index   : {original_index}")
    print(f"  aug_features_dir : {aug_features_dir}")
    print(f"  aug_names        : {args.aug_names}")
    print(f"  target_ratio     : {args.target_ratio}")
    print()
    print(f"Original index stats:")
    print(f"  total examples : {n_total_orig:,}")
    print(f"  correct (0)    : {n_correct_orig:,}")
    print(f"  mistake (1)    : {n_mistake_orig:,}")
    print(f"  pos rate       : {100*n_mistake_orig/n_total_orig:.2f}%")

    # target mistake count
    if args.target_ratio > 0:
        n_mistake_target = int(math.ceil(n_correct_orig / args.target_ratio))
    else:
        n_mistake_target = None   # include everything available

    n_aug_needed = (
        max(0, n_mistake_target - n_mistake_orig)
        if n_mistake_target is not None else None
    )

    if n_mistake_target is not None:
        print(f"\nTarget:")
        print(f"  n_mistake_target : {n_mistake_target:,}  "
              f"(= {n_correct_orig:,} / {args.target_ratio})")
        print(f"  n_aug_needed     : {n_aug_needed:,}  "
              f"(= {n_mistake_target:,} − {n_mistake_orig:,})")
    else:
        print(f"\nTarget: include ALL available augmented examples.")

    # separate mistake examples from original
    mistake_examples = [e for e in original_examples if e["label"] == 1]
    print(f"\nUnique mistake examples : {len(mistake_examples):,}")

    # check coverage per augmentation
    print(f"\nAugmented .npz coverage in {aug_features_dir}:")
    aug_pools: dict[str, list[dict]] = {}
    for aug_name in args.aug_names:
        coverable, missing = check_aug_coverage(
            mistake_examples, aug_features_dir, aug_name
        )
        aug_pools[aug_name] = coverable
        status = "✓" if not missing else f"✗ {len(missing)} missing videos"
        print(f"  {aug_name:<14}  {len(coverable):>6,} / {len(mistake_examples):,}  {status}")
        if missing and len(missing) <= 5:
            print(f"               missing: {missing}")

    # assemble augmented examples
    # add augmented variants one full aug_name at a time, in priority order.
    augmented_examples: list[dict] = []

    for aug_name in args.aug_names:
        pool = aug_pools[aug_name]
        if not pool:
            continue

        if n_aug_needed is not None:
            still_needed = n_aug_needed - len(augmented_examples)
            if still_needed <= 0:
                break
            take = min(len(pool), still_needed)
        else:
            take = len(pool)

        batch = [make_augmented_example(ex, aug_name) for ex in pool[:take]]
        augmented_examples.extend(batch)
        print(f"\n  +{take:,} augmented examples from '{aug_name}'  "
              f"(pool size {len(pool):,})")

    n_aug_added = len(augmented_examples)
    total_mistakes = n_mistake_orig + n_aug_added
    total_examples = n_total_orig + n_aug_added
    achieved_ratio = (n_correct_orig / total_mistakes) if total_mistakes > 0 else float("inf")

    print(f"\n{'─'*60}")
    print(f"Summary:")
    print(f"  original examples : {n_total_orig:,}")
    print(f"  augmented added   : {n_aug_added:,}")
    print(f"  total examples    : {total_examples:,}")
    print(f"  correct (0)       : {n_correct_orig:,}")
    print(f"  mistake (1)       : {total_mistakes:,}  "
          f"({100*total_mistakes/total_examples:.2f}%)")
    print(f"  achieved ratio    : 1 : {achieved_ratio:.2f}  "
          f"(target 1:{args.target_ratio})")

    if args.dry_run:
        print("\n[dry_run] No file written.")
        return

    # write combined index
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        # both feature directories so holomistakedataset._find_feature_file
        # can locate original and augmented npz files transparently.
        "features_dirs": [
            str(orig_features_dir),
            str(aug_features_dir.resolve()),
        ],
        "feature_key": idx["feature_key"],
        "num_examples": total_examples,
        "num_correct": n_correct_orig,
        "num_mistake": total_mistakes,
        "num_aug_added": n_aug_added,
        "aug_names_used": args.aug_names,
        "target_ratio": args.target_ratio,
        "achieved_ratio": round(achieved_ratio, 4),
        "examples": original_examples + augmented_examples,
    }

    with open(output_path, "w") as f:
        json.dump(combined, f)

    print(f"\nWrote combined index → {output_path}")
    print(f"  ({total_examples:,} examples, "
          f"{total_mistakes:,} mistakes, "
          f"{n_correct_orig:,} correct)")


if __name__ == "__main__":
    main()
