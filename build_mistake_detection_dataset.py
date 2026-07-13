# for each coarse action c = [f_1..f_k] we emit k examples: input = [phi(f_1)..phi(f_i)], label = correctness(f_i).
# doesn't materialise the prefix feature tensors (~50gb) - just records indices into the existing .npz files.
# see the bottom for a minimal torch dataset that consumes this index.
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np

def build_examples_for_video(entry: dict, npz_path: Path):
    """Return (examples, label_counts) for one video, or ([], Counter()) if unusable."""
    npz = np.load(npz_path, allow_pickle=True)
    seg_ids = npz["segment_ids"].astype(int)
    starts = npz["start_sec"].astype(float)
    ends = npz["end_sec"].astype(float)
    labels = npz["labels"].astype(int)
    npz.close()

    # midpoint -> coarse action assignment is robust to small boundary jitter
    midpoints = 0.5 * (starts + ends)

    coarse_actions = [ev for ev in entry["events"] if ev["label"] == "Coarse grained action"]
    if not coarse_actions: return [], Counter()

    examples = []
    counts = Counter()
    video_name = entry["video_name"]

    for coarse in coarse_actions:
        c_start, c_end = float(coarse["start"]), float(coarse["end"])

        # collect fine-grained segments whose midpoint lies in this coarse action
        inside = []  # (start_sec, npz_idx, segment_id, label)
        for npz_idx in range(len(seg_ids)):
            if c_start <= midpoints[npz_idx] <= c_end: inside.append((starts[npz_idx], npz_idx, int(seg_ids[npz_idx]), int(labels[npz_idx])))

        if not inside: continue
        inside.sort(key=lambda t: t[0])

        ordered_npz_idx = [t[1] for t in inside]
        ordered_seg_ids = [t[2] for t in inside]
        ordered_labels = [t[3] for t in inside]

        # progressive prefixes: example k uses positions 0..k of this coarse action
        for k in range(len(inside)):
            examples.append({
                "video_name": video_name,
                "coarse_id": int(coarse["id"]),
                "prefix_npz_indices": ordered_npz_idx[: k + 1],
                "prefix_segment_ids": ordered_seg_ids[: k + 1],
                "target_segment_id": ordered_seg_ids[k],
                "label": ordered_labels[k],
            })
            counts[ordered_labels[k]] += 1
    return examples, counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", required=True,
                        help="data-annotation-trainval-v1_1.json")
    parser.add_argument("--features_dir", required=True,
                        help="directory of per-video .npz files from vjepa2_1_segment_feature_extraction.py")
    parser.add_argument("--output_path", required=True,
                        help="output .json index file")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="stop after processing this many annotation entries (smoke test)")
    args = parser.parse_args()

    with open(args.annotation_path) as f:
        annotations = json.load(f)

    features_dir = Path(args.features_dir)

    all_examples = []
    total_counts = Counter()
    used_videos = 0
    missing_videos = 0

    if args.max_videos is not None:
        annotations = annotations[: args.max_videos]
        print(f"Smoke-test mode: capped at {args.max_videos} annotation entries")

    for vi, entry in enumerate(annotations):
        npz_path = features_dir / f"{entry['video_name']}.npz"
        if not npz_path.exists():
            missing_videos += 1
            continue

        examples, counts = build_examples_for_video(entry, npz_path)
        if examples:
            all_examples.extend(examples)
            total_counts.update(counts)
            used_videos += 1

        if (vi + 1) % 100 == 0:
            print(f"  processed {vi + 1}/{len(annotations)} videos, "
                  f"{len(all_examples)} examples so far")

    correct = total_counts.get(0, 0)
    mistake = total_counts.get(1, 0)
    total = correct + mistake
    print(f"\nVideos used: {used_videos} (missing .npz: {missing_videos})")
    print(f"Examples:    {total}")
    if total:
        print(f"  correct: {correct} ({100 * correct / total:.2f}%)")
        print(f"  mistake: {mistake} ({100 * mistake / total:.2f}%)")

    out = {
        "features_dir": str(features_dir.resolve()),
        "feature_key": "features",   # array inside each .npz; shape (num_segments, t, d)
        "num_examples": len(all_examples),
        "examples": all_examples,
    }
    with open(args.output_path, "w") as f:
        json.dump(out, f)
    print(f"Wrote {args.output_path}")

# sketch of how to consume the index in pytorch (not run by this script):
#
#   import json, numpy as np, torch
#   from pathlib import path
#   from torch.utils.data import dataset
#
#   class holomistakedataset(dataset):
#       def __init__(self, index_path):
#           idx = json.load(open(index_path))
#           self.features_dir = path(idx["features_dir"])
#           self.examples = idx["examples"]
#           self._cache = {}  # video_name -> features array
#
#       def _load(self, video_name):
#           if video_name not in self._cache:
#               self._cache[video_name] = np.load(
#                   self.features_dir / f"{video_name}.npz")["features"]
#           return self._cache[video_name]
#
#       def __len__(self):
#           return len(self.examples)
#
#       def __getitem__(self, i):
#           ex = self.examples[i]
#           feats = self._load(ex["video_name"])[ex["prefix_npz_indices"]]  # (k, t, d)
#           return torch.from_numpy(feats.astype(np.float32)), ex["label"]
#
#   # collate_fn must pad variable-length sequences (length 1..k).

if __name__ == "__main__":
    main()

