# val runner for qwen_clips.py (clips-as-context), analogous to qwen_val.py (text-based context).
# loads the model once, writes one jsonl line per example, resumes partial runs on restart.

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_clips import (          # noqa: E402
    load_model, process_video,
    PRIOR_MODES, REASONING_MODES,
)
from src.utils.qwen_common import (  # noqa: E402
    expected_example_count, load_completed_videos, rewrite_without_video,
)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run Qwen3-VL clips-based mistake detection on HoloAssist val. "
            "Prior sub-actions are fed as video clips (not just text), with "
            "the target clip clearly marked for evaluation."
        )
    )
    ap.add_argument("--annotations",    required=True, help="Path to annotation JSON.")
    ap.add_argument("--video-dir",      required=True, help="Root directory containing per-video folders.")
    ap.add_argument("--val-list",       required=True, help="Text file listing val video names.")
    ap.add_argument("--out",            required=True, help="Output JSONL file path.")
    ap.add_argument(
        "--prior-mode", choices=PRIOR_MODES, default="clips",
        help=(
            "'none': only the target clip. "
            "'clips': prior clips as video (no labels). "
            "'clips_labelled': prior clips with correct/mistake labels (oracle)."
        ),
    )
    ap.add_argument(
        "--reasoning-mode", choices=REASONING_MODES, default="direct",
        help="'direct': immediate prediction. 'cot': chain-of-thought.",
    )
    ap.add_argument(
        "--max-prior-clips", type=int, default=None,
        help="Cap on the number of most-recent prior clips to include (default: all).",
    )
    ap.add_argument(
        "--fps", type=float, default=2.0,
        help="Frames per second to sample from each clip (default 2.0).",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Ignore existing output file and rerun everything.",
    )
    ap.add_argument(
        "--max-videos", type=int, default=None,
        help="Only process this many videos (debug / smoke test).",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.force and out_path.exists():
        out_path.unlink()
        print(f"--force: removed existing {out_path}")

    with open(args.val_list) as f:
        val_videos = [ln.strip() for ln in f if ln.strip()]
    if args.max_videos is not None:
        val_videos = val_videos[: args.max_videos]
    val_set = set(val_videos)

    print(f"Validation videos requested: {len(val_videos)}")
    print(f"Prior mode:                  {args.prior_mode}")
    print(f"Reasoning mode:              {args.reasoning_mode}")
    print(f"Max prior clips:             {args.max_prior_clips}")
    print(f"FPS:                         {args.fps}")

    with open(args.annotations) as f:
        ann = json.load(f)
    by_name = {e["video_name"]: e for e in ann if e["video_name"] in val_set}
    missing_in_ann = [v for v in val_videos if v not in by_name]
    if missing_in_ann:
        print(f"WARNING: {len(missing_in_ann)} val video(s) not in annotation:")
        for v in missing_in_ann:
            print(f"  - {v}")

    written_counts = load_completed_videos(out_path)
    total_expected = 0
    todo = []
    for v in val_videos:
        if v not in by_name:
            continue
        entry = by_name[v]
        n_expected = expected_example_count(entry)
        total_expected += n_expected
        n_written = written_counts.get(v, 0)
        if n_written == n_expected and n_expected > 0:
            print(f"[skip] {v}: {n_written}/{n_expected} examples already written")
            continue
        if 0 < n_written < n_expected:
            print(f"[partial] {v}: {n_written}/{n_expected} written -- will redo")
            rewrite_without_video(out_path, v)
        todo.append((v, entry, n_expected))

    print(f"\nTotal expected examples across val: {total_expected}")
    print(f"Videos to process this run:          {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    processor, model = load_model(device)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"Model loaded on {device} in {time.perf_counter() - t0:.1f}s")
    if device == "cuda":
        print(f"GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    run_t0 = time.perf_counter()
    egs_done = 0

    for i, (video_name, entry, n_expected) in enumerate(todo, 1):
        video_path = f"{args.video_dir}/{video_name}/Export_py/Video_compress.mp4"
        if not os.path.exists(video_path):
            print(f"[{i}/{len(todo)}] MISSING VIDEO: {video_path} -- skipping")
            continue

        print(f"\n[{i}/{len(todo)}] {video_name}  ({n_expected} examples expected)")
        v_t0 = time.perf_counter()
        try:
            results = process_video(
                video_path, entry, processor, model,
                prior_mode=args.prior_mode,
                reasoning_mode=args.reasoning_mode,
                fps=args.fps,
                max_prior_clips=args.max_prior_clips,
                test=False,
            )
        except Exception as e:
            print(f"  ERROR processing {video_name}: {e}")
            continue
        v_dt = time.perf_counter() - v_t0

        # atomic append: write to temp then append to output
        tmp = out_path.with_suffix(".jsonl.partial")
        with tmp.open("w") as f:
            for r in results:
                rec = dict(r)
                rec["video_name"] = video_name
                f.write(json.dumps(rec) + "\n")
        with tmp.open() as src, out_path.open("a") as dst:
            for line in src:
                dst.write(line)
        tmp.unlink()

        egs_done += len(results)
        elapsed = time.perf_counter() - run_t0
        rate = egs_done / elapsed if egs_done else float("nan")
        print(
            f"  done in {v_dt:.1f}s  ({len(results)} examples, "
            f"{v_dt / max(len(results), 1):.2f}s/example)"
        )
        print(
            f"  cumulative: {egs_done} examples in {elapsed / 60:.1f} min "
            f"({rate:.2f} ex/s)"
        )

    if device == "cuda":
        print(f"\nGPU peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"\nAll done. Total time: {(time.perf_counter() - run_t0) / 60:.1f} min")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
