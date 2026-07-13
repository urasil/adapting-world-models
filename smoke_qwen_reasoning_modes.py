#!/usr/bin/env python3
# previews and smoke-tests qwen reasoning modes on one video example: prints the direct and
# cot prompts, then runs qwen_baseline.py and qwen_val.py twice each on a single-video subset,
# to exercise the shared prompt/inference path without the full dataset.

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from qwen_baseline import build_prompt, group_fine_by_coarse


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ANN = REPO_ROOT / "holoassist" / "data-annotation-trainval-v1_1.json"
DEFAULT_VIDEO_DIR = REPO_ROOT / "holoassist" / "videos"


def pick_example(annotations: list[dict], video_dir: Path, video_name: str | None) -> tuple[dict, dict, list[dict]]:
    """Return (entry, coarse, members) for the first usable example."""
    if video_name:
        candidates = [entry for entry in annotations if entry.get("video_name") == video_name]
        if not candidates:
            raise ValueError(f"video_name {video_name!r} not found in annotations")
    else:
        candidates = annotations

    for entry in candidates:
        video_path = video_dir / entry["video_name"] / "Export_py" / "Video_compress.mp4"
        if not video_path.exists():
            continue
        groups = group_fine_by_coarse(entry)
        if not groups:
            continue
        coarse, members = groups[0]
        return entry, coarse, members

    raise RuntimeError("No usable video example found")


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview and smoke-test both Qwen reasoning modes.")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANN)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs" / "qwen_reasoning_smoke")
    parser.add_argument("--video-name", type=str, default=None, help="Optional specific video to use.")
    parser.add_argument("--prior-mode", choices=("none", "actions", "actions_with_labels"), default="actions_with_labels")
    parser.add_argument("--skip-run", action="store_true", help="Only print prompts, do not launch smoke runs.")
    args = parser.parse_args()

    with args.annotations.open() as f:
        annotations = json.load(f)

    entry, coarse, members = pick_example(annotations, args.video_dir, args.video_name)
    k = 1
    prompt_direct = build_prompt(coarse, members, k, args.prior_mode, reasoning_mode="direct")
    prompt_cot = build_prompt(coarse, members, k, args.prior_mode, reasoning_mode="cot")

    video_path = args.video_dir / entry["video_name"] / "Export_py" / "Video_compress.mp4"
    target_attrs = members[k - 1].get("attributes", {})
    target_desc = " ".join(
        part for part in (
            str(target_attrs.get("Verb", "")).strip(),
            str(target_attrs.get("Adjective", "")).strip(),
            str(target_attrs.get("Noun", "")).strip(),
        )
        if part
    ) or "(unspecified action)"

    print(f"Selected video: {entry['video_name']}")
    print(f"Example clip:   {video_path}")
    print(f"Coarse action:  {coarse.get('attributes', {}).get('Action sentence', '(unspecified coarse action)')}")
    print(f"Target action:  {target_desc}")

    print("\n=== Direct prompt ===")
    print(prompt_direct)
    print("\n=== CoT prompt ===")
    print(prompt_cot)

    if args.skip_run:
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen_reasoning_smoke_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        single_ann = tmpdir_path / "single_annotation.json"
        single_val = tmpdir_path / "single_val.txt"
        with single_ann.open("w") as f:
            json.dump([entry], f, indent=2)
        with single_val.open("w") as f:
            f.write(entry["video_name"] + "\n")

        for script_name in ("qwen_baseline.py", "qwen_val.py"):
            for reasoning_mode in ("direct", "cot"):
                out_dir = args.output_root / script_name.replace(".py", "") / reasoning_mode
                out_dir.mkdir(parents=True, exist_ok=True)
                if script_name == "qwen_baseline.py":
                    command = [
                        sys.executable,
                        str(REPO_ROOT / script_name),
                        "--annotation_path", str(single_ann),
                        "--video_dir", str(args.video_dir),
                        "--output_dir", str(out_dir),
                        "--prior-mode", args.prior_mode,
                        "--reasoning-mode", reasoning_mode,
                        "--max-videos", "1",
                        "--test",
                    ]
                else:
                    command = [
                        sys.executable,
                        str(REPO_ROOT / script_name),
                        "--annotations", str(single_ann),
                        "--video-dir", str(args.video_dir),
                        "--val-list", str(single_val),
                        "--out", str(out_dir / "results.jsonl"),
                        "--prior-mode", args.prior_mode,
                        "--reasoning-mode", reasoning_mode,
                    ]
                run_command(command)


if __name__ == "__main__":
    main()