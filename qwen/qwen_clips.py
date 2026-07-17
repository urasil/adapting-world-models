# qwen3-vl mistake detection using clips (not just text) as visual context for prior
# sub-actions, then the target clip.
# prior_mode: none | clips | clips_labelled (clips + correct/mistake oracle label)
# reasoning_mode: direct | cot

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from qwen_vl_utils import process_vision_info

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.qwen_common import (
    MODEL_ID, LABEL_MAP, load_model, group_fine_by_coarse,
    describe_coarse_action, correctness_label_text,
    parse_mistake_response,
)

PRIOR_MODES = ("none", "clips", "clips_labelled")
REASONING_MODES = ("direct", "cot")

DIRECT_RESPONSE_BLOCK = (
    "Decide, based only on what you can see in the clip and on the prior "
    "sub-action clips shown above, whether this sub-action is a correct "
    "progression towards the overall task or a mistake, given what the "
    "performer has already done.\n\n"
    "Respond in EXACTLY this format and nothing else:\n"
    "MISTAKE: <no|yes>\n"
    "EXPLANATION: <one or two sentences>"
)

COT_RESPONSE_BLOCK = (
    "Reason step by step before deciding whether this sub-action is a "
    "correct progression towards the overall task or a mistake, given what "
    "the performer has already done. Respond in EXACTLY this format and "
    "nothing else:\n\n"
    "OBSERVATION: <describe concretely what the person does in the clip: "
    "which object(s) they handle, what action they perform, and where they "
    "direct it>\n"
    "EXPECTED:    <given the overall task and the sub-actions completed so "
    "far, what would be the correct progression at this point in the task? "
    "Note that preparatory actions are necessary even if they don't directly "
    "complete the task goal.>\n"
    "COMPARISON:  <does what you observed on the target clip broadly match "
    "that expected progression? If it matches, there is no mistake. If not, "
    "there might be a mistake.>\n"
    "MISTAKE:     <no|yes>\n"
    "EXPLANATION: <one sentence summary>"
)

def build_content(video_path: str, coarse: dict, members: List[dict], k: int, prior_mode: str,
                   reasoning_mode: str = "direct", fps: float = 8.0, min_frames_prior: int = 2,
                   max_frames_prior: int = 16, min_frames_target: int = 2, max_frames_target: int = 16,
                   max_prior_clips: Optional[int] = None) -> Tuple[List[dict], str]:
    # builds the interleaved content list for the Qwen3-VL message, returns (content_blocks, prompt_text) where prompt_text is a human-readable reconstruction for logging
    if prior_mode not in PRIOR_MODES:
        raise ValueError(f"prior_mode must be one of {PRIOR_MODES}, got {prior_mode!r}")
    if reasoning_mode not in REASONING_MODES:
        raise ValueError(f"reasoning_mode must be one of {REASONING_MODES}, got {reasoning_mode!r}")

    K = len(members)
    coarse_desc = describe_coarse_action(coarse)
    target = members[k - 1]

    content: List[dict] = []

    # task header
    header = (
        "You are watching a first-person egocentric procedural task video.\n"
        f"Overall task (the instructor's coarse-grained instruction): {coarse_desc}\n"
        f"This task consists of {K} consecutive sub-actions performed by the person (the performer).\n"
        "Your job is to judge, given what the performer has already done "
        "towards the overall task, whether the CURRENT sub-action shown "
        "below is a correct progression or a mistake in that progression.\n"
        "You will be shown video clips of the prior sub-actions, each "
        "preceded by a text label, followed by the target clip to evaluate.\n"
    )

    # prior clips
    if k == 1:
        header += "This is the first sub-action of the task. The performer has not done anything yet.\n"
        content.append({"type": "text", "text": header})
    elif prior_mode == "none":
        header += (
            f"The performer has already completed {k - 1} prior sub-action(s) "
            "in this task, but their details are not shown to you here.\n"
        )
        content.append({"type": "text", "text": header})
    else:
        prior_indices = list(range(k - 1))  # 0-indexed positions of prior sub-actions
        n_prior_total = len(prior_indices)

        if max_prior_clips is not None and n_prior_total > max_prior_clips:
            prior_indices = prior_indices[-max_prior_clips:]

        header += (
            f"Review the {len(prior_indices)} prior sub-action(s) below "
            "before evaluating the target clip.\n"
        )
        content.append({"type": "text", "text": header})

        for i in prior_indices:
            m = members[i]

            if prior_mode == "clips_labelled":
                lbl = correctness_label_text(m["attributes"]["Action Correctness"])
                clip_label = f"\nSub-action {i + 1} of {K}, this sub-action was {lbl}\n"
            else:
                clip_label = f"\nSub-action {i + 1} of {K}\n"

            content.append({"type": "text", "text": clip_label})
            content.append({
                "type": "video",
                "video": str(video_path),
                "video_start": float(m["start"]),
                "video_end": float(m["end"]),
                "fps": fps,
                "min_frames": min_frames_prior,
                "max_frames": max_frames_prior,
            })

    # target clip (clearly marked)
    separator = "=" * 60
    target_intro = (
        f"\n{separator}\n"
        f"TARGET CLIP - Sub-action {k} of {K} (the current sub-action to judge)\n"
        f"This is the clip you need to evaluate as correct or containing a mistake.\n"
        f"{separator}\n"
    )
    content.append({"type": "text", "text": target_intro})
    content.append({
        "type": "video",
        "video": str(video_path),
        "video_start": float(target["start"]),
        "video_end": float(target["end"]),
        "fps": fps,
        "min_frames": min_frames_target,
        "max_frames": max_frames_target,
    })

    # response format
    response_block = DIRECT_RESPONSE_BLOCK if reasoning_mode == "direct" else COT_RESPONSE_BLOCK
    content.append({"type": "text", "text": "\n" + response_block})

    # human-readable reconstruction for logging
    prompt_text = "\n".join(
        item["text"] if item["type"] == "text"
        else f"[VIDEO {item.get('video_start', '?'):.1f}s to {item.get('video_end', '?'):.1f}s]"
        for item in content
    )

    return content, prompt_text

# inference

def run_inference_clips(processor, model, content_blocks: List[dict], max_new_tokens: int = 512) -> str:
    # runs Qwen3-VL inference with interleaved text / video content blocks
    messages = [{"role": "user", "content": content_blocks}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # return_video_metadata=true is required for qwen3vl: without it, process_vision_info
    # uses the qwen2.5vl legacy path and puts fps as a list in video_kwargs, which the
    # qwen3vl processor rejects (it validates fps as int|float|none, not list).
    # with it, video_inputs contains (tensor, metadata_dict) tuples. we must unpack those
    # and pass the metadata separately via video_metadata= so that the basevideoprocessor
    # can construct videometadata objects with the native fps and frame indices needed for
    # correct temporal position encoding.
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if video_inputs:
        video_tensors = [v for v, _ in video_inputs]
        video_metadatas = [m for _, m in video_inputs]
    else:
        video_tensors = video_metadatas = None
    inputs = processor(
        text=[text], images=image_inputs, videos=video_tensors,
        video_metadata=video_metadatas,
        padding=True, return_tensors="pt", **video_kwargs,
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

def process_video(
    video_path: str,
    annotation_entry: dict,
    processor,
    model,
    prior_mode: str = "clips",
    reasoning_mode: str = "direct",
    fps: float = 8.0,
    max_prior_clips: Optional[int] = None,
    test: bool = False,
) -> List[dict]:
    # predicts correctness for every fine-grained sub-action in the video, for each target k, prior sub-actions in the same coarse task are provided as video clips (when prior_mode != 'none') interleaved with text labels, with the target clip clearly marked
    if prior_mode not in PRIOR_MODES:
        raise ValueError(f"prior_mode must be one of {PRIOR_MODES}, got {prior_mode!r}")
    if reasoning_mode not in REASONING_MODES:
        raise ValueError(f"reasoning_mode must be one of {REASONING_MODES}, got {reasoning_mode!r}")

    groups = group_fine_by_coarse(annotation_entry)
    total_examples = sum(len(members) for _, members in groups)
    print(
        f"  {len(groups)} coarse actions, {total_examples} examples "
        f"(prior_mode={prior_mode}, reasoning_mode={reasoning_mode})"
    )

    results: List[dict] = []
    n_done = 0

    for coarse, members in groups:
        K = len(members)
        for k_idx, target in enumerate(members):
            k = k_idx + 1  # 1-indexed
            attrs = target.get("attributes", {})
            correctness = attrs["Action Correctness"]

            content_blocks, prompt_text = build_content(
                video_path=video_path,
                coarse=coarse,
                members=members,
                k=k,
                prior_mode=prior_mode,
                reasoning_mode=reasoning_mode,
                fps=fps,
                max_prior_clips=max_prior_clips,
            )

            try:
                response = run_inference_clips(processor, model, content_blocks)
            except Exception as e:
                print(f"  segment {target['id']} inference failed: {e}")
                continue

            parsed = parse_mistake_response(response)

            results.append({
                "coarse_id": int(coarse["id"]),
                "target_segment_id": int(target["id"]),
                "k_in_coarse": k,
                "K_total": K,
                "prior_segment_ids": [int(m["id"]) for m in members[:k_idx]],
                "target_start_sec": float(target["start"]),
                "target_end_sec": float(target["end"]),
                "verb": attrs.get("Verb", ""),
                "noun": attrs.get("Noun", ""),
                "correctness_raw": correctness,
                "label": LABEL_MAP[correctness],
                "prompt": prompt_text,
                "mistake_detection": parsed,
            })

            n_done += 1
            if test and n_done >= 3:
                print(f"  --test: stopping after {n_done} examples")
                return results
            if n_done % 10 == 0:
                print(f"  example {n_done}/{total_examples}")

    return results

# entry point

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-VL zero-shot mistake-detection on HoloAssist using fine-grained "
            "action clips as visual context. Prior sub-actions within the same coarse "
            "task are fed as video clips (not just text), interleaved with text labels. "
            "The target clip to evaluate is clearly marked in the prompt."
        )
    )
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--video_dir",        type=str, required=True)
    parser.add_argument("--output_dir",       type=str, required=True)
    parser.add_argument(
        "--prior-mode", choices=PRIOR_MODES, default="clips",
        help=(
            "How to present prior sub-actions. "
            "'none': only the target clip (no prior context). "
            "'clips': prior clips as video, verb/noun labels only. "
            "'clips_labelled': prior clips as video with correct/mistake labels (oracle)."
        ),
    )
    parser.add_argument("--reasoning-mode", choices=REASONING_MODES, default="direct",
        help="'direct': immediate prediction. 'cot': chain-of-thought reasoning.")
    parser.add_argument("--max-prior-clips", type=int, default=None,
        help="Maximum number of most-recent prior clips to include. Default None means all prior clips are shown.")
    parser.add_argument("--fps", type=float, default=8.0, help="Frames per second to sample from each clip (default 2.0).")
    parser.add_argument("--test", action="store_true", help="Stop after 3 examples per video (smoke test).")
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.annotation_path) as f:
        annotations = json.load(f)
    print(f"Processing {len(annotations)} videos (prior_mode={args.prior_mode}, reasoning_mode={args.reasoning_mode})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_model(device)

    entries = annotations[: args.max_videos] if args.max_videos else annotations
    for vi, entry in enumerate(entries):
        video_name = entry["video_name"]
        out_path = out_dir / f"{video_name}.json"
        if out_path.exists():
            print(f"[{vi+1}/{len(entries)}] {video_name}: already done, skipping")
            continue

        video_path = Path(args.video_dir) / video_name / "Export_py" / "Video_compress.mp4"
        if not video_path.exists():
            print(f"[{vi+1}/{len(entries)}] {video_name}: video not found, skipping")
            continue

        print(f"[{vi+1}/{len(entries)}] {video_name}")
        t0 = time.time()
        try:
            seg_results = process_video(
                str(video_path), entry, processor, model,
                prior_mode=args.prior_mode,
                reasoning_mode=args.reasoning_mode,
                fps=args.fps,
                max_prior_clips=args.max_prior_clips,
                test=args.test,
            )
        except Exception as e:
            print(f"  failed: {e}")
            continue

        result = {
            "video_name": video_name,
            "model": MODEL_ID,
            "framing": "clips_prior_context",
            "prior_mode": args.prior_mode,
            "reasoning_mode": args.reasoning_mode,
            "fps": args.fps,
            "max_prior_clips": args.max_prior_clips,
            "examples": seg_results,
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"  saved {len(seg_results)} examples -> {out_path} "
            f"({time.time() - t0:.1f}s)"
        )

if __name__ == "__main__":
    main()
