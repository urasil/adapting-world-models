import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from qwen_vl_utils import process_vision_info

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.qwen_common import (
    MODEL_ID, LABEL_MAP, load_model, group_fine_by_coarse,
    describe_fine_action, describe_coarse_action, correctness_label_text,
    parse_mistake_response,
)

# three prior-context flavours.
#   none                : no prior context at all (per-segment baseline)
#   actions             : list prior sub-actions by name, no labels (flavour a)
#   actions_with_labels : list prior sub-actions with gold correct/mistake labels (flavour b)
PRIOR_MODES = ("none", "actions", "actions_with_labels")

# two reasoning modes for evaluation.
#   direct : immediate prediction (single response block)
#   cot    : chain-of-thought with observations, expectations, comparisons
REASONING_MODES = ("direct", "cot")

NEUTRAL_DEFINITION = (
    "A sub-action is correct if the person performs the intended action, "
    "with the right object, in a reasonable order. A sub-action "
    "contains a mistake if the person performs the wrong action, "
    "skips a required part or does it in the wrong order. "
    )

DIRECT_RESPONSE_BLOCK = (
    "Decide whether this sub-action is performed correctly or contains a "
    "mistake, based only on what you can see in the clip and on the prior "
    "sub-actions listed above.\n\n"
    "Respond in EXACTLY this format and nothing else:\n"
    "MISTAKE: <no|yes>\n"
    "EXPLANATION: <one or two sentences>"
)

COT_RESPONSE_BLOCK_TEMPLATE = (
    "Reason step by step before deciding. Respond in EXACTLY this format "
    "and nothing else:\n\n"
    "OBSERVATION: <describe concretely what the person does in the clip: "
    "which object(s) they handle, what action they perform, and where they "
    "direct it>\n"
    "EXPECTED:    <given the overall task and prior sub-actions, what would "
    "a correct execution of \"{target_desc}\" look like? Note that preparatory actions are necessary "
    "even if they don't directly complete the task goal.>\n"
    "COMPARISON:  <does what you observed broadly match what was expected? If it matches, "
    "there is no mistake. If not, there might be a mistake.>\n"
    "MISTAKE:     <no|yes>\n"
    "EXPLANATION: <one sentence summary>"
)

def build_prompt(coarse: dict, members: List[dict], k: int,
                 prior_mode: str, reasoning_mode: str = "direct") -> str:
    if reasoning_mode not in REASONING_MODES:
        raise ValueError(f"reasoning_mode must be one of {REASONING_MODES}, got {reasoning_mode!r}")

    K = len(members)
    coarse_desc = describe_coarse_action(coarse)
    target_desc = describe_fine_action(members[k - 1])

    header = (
        "You are watching a first-person egocentric procedural task video.\n"
        f"Overall task: {coarse_desc}\n"
        f"This task consists of {K} consecutive sub-actions.\n"
    )

    # build prior block (coherent wording regardless of context presence)
    if prior_mode == "none" or k == 1:
        prior_block = "This is the first sub-action of the task, there are no prior sub-actions.\n"
    elif prior_mode == "actions":
        lines = [f"  {i+1}. {describe_fine_action(members[i])}" for i in range(k - 1)]
        prior_block = "Previous sub-actions in this task, in order:\n" + "\n".join(lines) + "\n"
    elif prior_mode == "actions_with_labels":
        lines = []
        for i in range(k - 1):
            lbl = correctness_label_text(members[i]["attributes"]["Action Correctness"])
            lines.append(f"  {i+1}. {describe_fine_action(members[i])} [{lbl}]")
        prior_block = (
            "Previous sub-actions in this task, in order "
            "(each annotated correct or mistake):\n" + "\n".join(lines) + "\n"
        )
    else:
        raise ValueError(f"unknown prior_mode: {prior_mode}")

    target_intro = (
        f"\n{NEUTRAL_DEFINITION}\n\n"
        f"The video clip below shows sub-action {k} of {K}: {target_desc}.\n"
    )

    if reasoning_mode == "direct":
        response_block = DIRECT_RESPONSE_BLOCK
    else:  # cot
        response_block = COT_RESPONSE_BLOCK_TEMPLATE.format(target_desc=target_desc)

    return header + prior_block + target_intro + response_block

# frame-decoding helpers removed: `process_vision_info` now handles video file i/o.

def run_inference(
    processor, model, video_path: str, prompt: str,
    video_start: float | None = None, video_end: float | None = None,
    fps: float = 8.0, min_frames: int = 4, max_frames: int = 32, max_new_tokens: int = 512
) -> str:
    # pass the video by path and let process_vision_info handle decoding.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "video_start": video_start,
                    "video_end": video_end,
                    "fps": fps,
                    "min_frames": min_frames,
                    "max_frames": max_frames,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, image_patch_size=processor.image_processor.patch_size, return_video_kwargs=True
    )
    if isinstance(video_kwargs.get("fps"), list):
        video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else None
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt", **video_kwargs,
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

def process_video(video_path: str, annotation_entry: dict, processor, model, prior_mode: str = "actions_with_labels", reasoning_mode: str = "direct", test: bool = False) -> List[dict]:
    # for every coarse action, predicts each sub-action's correctness from only the target segment's frames plus an optional text description of prior sub-actions
    if prior_mode not in PRIOR_MODES:
        raise ValueError(f"prior_mode must be one of {PRIOR_MODES}, got {prior_mode!r}")
    if reasoning_mode not in REASONING_MODES:
        raise ValueError(f"reasoning_mode must be one of {REASONING_MODES}, got {reasoning_mode!r}")

    groups = group_fine_by_coarse(annotation_entry)
    total_examples = sum(len(members) for _, members in groups)
    print(f"  {len(groups)} coarse actions, {total_examples} examples (prior_mode={prior_mode}, reasoning_mode={reasoning_mode})")

    results: List[dict] = []
    n_done = 0
    for coarse, members in groups:
        K = len(members)
        for k_idx, target in enumerate(members):
            k = k_idx + 1  # 1-indexed
            attrs = target.get("attributes", {})
            correctness = attrs["Action Correctness"]  # guaranteed by group_fine_by_coarse

            # build prompt and let process_vision_info handle video decoding from file
            prompt = build_prompt(coarse, members, k, prior_mode, reasoning_mode)
            # construct a minimal video kwargs dict telling the vision utils how
            # to sample the target clip. process_vision_info will clamp frames.
            # we pass the container path as the video identifier, the helper will open and decode.
            video_kwargs = {
                "video": str(video_path),
                "video_start": float(target["start"]),
                "video_end": float(target["end"]),
                "fps": 8.0,
                "min_frames": 2,
                "max_frames": 32,
            }

            try:
                response = run_inference(processor, model, video_path, prompt, video_start=video_kwargs["video_start"],
                    video_end=video_kwargs["video_end"], fps=video_kwargs["fps"],
                    min_frames=video_kwargs["min_frames"], max_frames=video_kwargs["max_frames"])
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
                "prompt": prompt,
                "mistake_detection": parsed,
            })

            n_done += 1
            if test and n_done >= 3:
                print(f"  --test: stopping after {n_done} examples")
                return results
            if n_done % 10 == 0:
                print(f"  example {n_done}/{total_examples}")

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL zero-shot mistake-detection baseline on HoloAssist. "
                    "Visual input is the target sub-action only; prior sub-actions "
                    "are conveyed via text in the prompt."
    )
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prior-mode", choices=PRIOR_MODES, default="actions_with_labels",
                        help="How to describe prior sub-actions in the prompt. "
                             "'none': no prior context. "
                             "'actions': verb/noun list, no labels (Flavour A). "
                             "'actions_with_labels': verb/noun + correct/mistake (Flavour B).")
    parser.add_argument("--reasoning-mode", choices=REASONING_MODES, default="direct",
                        help="Prompting strategy for the model. "
                             "'direct': immediate prediction. "
                             "'cot': chain-of-thought reasoning.")
    parser.add_argument("--test", action="store_true",
                        help="Stop after 3 examples per video (smoke test).")
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
            seg_results = process_video(str(video_path), entry, processor, model,
                prior_mode=args.prior_mode, reasoning_mode=args.reasoning_mode, test=args.test)
        except Exception as e:
            print(f"  failed: {e}")
            continue

        result = {
            "video_name": video_name,
            "model": MODEL_ID,
            "framing": "text_prior_context",
            "prior_mode": args.prior_mode,
            "reasoning_mode": args.reasoning_mode,
            "num_frames_per_segment": None,
            "examples": seg_results,
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved {len(seg_results)} examples -> {out_path} ({time.time() - t0:.1f}s)")

if __name__ == "__main__":
    main()
