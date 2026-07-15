# shared between qwen_baseline.py and qwen_clips.py: model loading, annotation grouping, response parsing
import json
import re
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.utils.holoassist_labels import LABEL_MAP

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

def load_model(device: str):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=dtype, device_map="auto" if device.startswith("cuda") else None, attn_implementation="sdpa",
    ).eval()
    if not device.startswith("cuda"):
        model = model.to(device)
    return processor, model

def group_fine_by_coarse(annotation_entry: dict) -> List[Tuple[dict, List[dict]]]:
    # groups fine actions under their coarse action, in time order, dropping unlabeled or unmatched ones
    coarse = sorted([ev for ev in annotation_entry["events"] if ev["label"] == "Coarse grained action"], key=lambda x: x["start"])
    fine = [ev for ev in annotation_entry["events"] if ev["label"] == "Fine grained action"]

    groups: List[Tuple[dict, List[dict]]] = [(c, []) for c in coarse]
    for f in fine:
        attrs = f.get("attributes", {})
        if attrs.get("Action Correctness") not in LABEL_MAP:
            continue
        mid = 0.5 * (f["start"] + f["end"])
        for c, members in groups:
            if c["start"] <= mid <= c["end"]:
                members.append(f)
                break

    out = []
    for c, members in groups:
        if members:
            members.sort(key=lambda x: x["start"])
            out.append((c, members))
    return out

def describe_fine_action(ev: dict) -> str:
    # compact verb/adjective/noun description, with a fallback for missing attrs
    a = ev.get("attributes", {})
    parts = [str(a.get("Verb", "")).strip(), str(a.get("Adjective", "")).strip(), str(a.get("Noun", "")).strip()]
    desc = " ".join(p for p in parts if p and p.lower() != "none")
    return desc if desc else "(unspecified action)"

def describe_coarse_action(coarse: dict) -> str:
    a = coarse.get("attributes", {})
    sent = str(a.get("Action sentence", "")).strip()
    if sent:
        return sent
    parts = [str(a.get("Verb", "")).strip(), str(a.get("Adjective", "")).strip(), str(a.get("Noun", "")).strip()]
    return " ".join(p for p in parts if p and p.lower() != "none") or "(unspecified coarse action)"

def correctness_label_text(correctness: str) -> str:
    return "mistake" if LABEL_MAP[correctness] == 1 else "correct"

def parse_mistake_response(response: str) -> dict:
    out = {"mistake": None, "explanation": None, "raw": response}
    m = re.search(r"MISTAKE\s*:\s*(yes|no)", response, re.IGNORECASE)
    if m:
        out["mistake"] = m.group(1).lower() == "yes"
    e = re.search(r"EXPLANATION\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    if e:
        out["explanation"] = e.group(1).strip()
    return out

# resume support for qwen_val.py / qwen_clips_val.py: both write one jsonl line per example

def expected_example_count(entry: dict) -> int:
    # how many examples process_video will emit for this annotation entry
    return sum(len(members) for _, members in group_fine_by_coarse(entry))

def load_completed_videos(out_path: Path) -> dict:
    counts: dict = {}
    if not out_path.exists():
        return counts
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = rec.get("video_name")
            if v:
                counts[v] = counts.get(v, 0) + 1
    return counts

def rewrite_without_video(out_path: Path, video_name: str) -> None:
    if not out_path.exists():
        return
    tmp = out_path.with_suffix(".jsonl.tmp")
    kept = dropped = 0
    with out_path.open() as fin, tmp.open("w") as fout:
        for line in fin:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line)
                kept += 1
                continue
            if rec.get("video_name") == video_name:
                dropped += 1
            else:
                fout.write(line)
                kept += 1
    tmp.replace(out_path)
    print(f"  [resume] dropped {dropped} partial line(s) for {video_name}, kept {kept}")
