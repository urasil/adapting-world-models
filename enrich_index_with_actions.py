# adds per-segment verb/noun ids + timestamps to the mistake-detection index files.
# vocab is built from train data only; val-only verbs/nouns map to __unknown__ so eval stays unbiased.
# also writes prefix start/end secs so segmentwmdataset can filter for temporal continuity
# without re-reading npz files.
# usage: python enrich_index_with_actions.py --train_index ... --val_index ... --train_npz_dir ...
#   --val_npz_dir ... --train_npy_dir ... --val_npy_dir ... --output_dir datasets
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def _load_meta(video_name: str, npz_dirs: list[str]):
    """Return (verb, noun, start_sec, end_sec) arrays from the NPZ for a video."""
    for d in npz_dirs:
        path = os.path.join(d, f"{video_name}.npz")
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            verb      = data["verb"].astype(str)
            noun      = data["noun"].astype(str)
            start_sec = data["start_sec"].tolist()
            end_sec   = data["end_sec"].tolist()
            data.close()
            return verb, noun, start_sec, end_sec
    return None, None, None, None


def _build_vocabs(examples: list[dict], npz_dirs: list[str]):
    """Build verb/noun vocabs from the given examples (train only for an unbiased eval)."""
    verb_counts: Counter = Counter()
    noun_counts: Counter = Counter()
    cache: dict = {}

    for ex in examples:
        vname = ex["video_name"]
        if vname not in cache:
            cache[vname] = _load_meta(vname, npz_dirs)
        verbs, nouns, _, _ = cache[vname]
        if verbs is None:
            continue
        for idx in ex["prefix_npz_indices"]:
            verb_counts[str(verbs[idx])] += 1
            noun_counts[str(nouns[idx])] += 1

    def _make_vocab(counts: Counter) -> dict[str, int]:
        # sort by frequency descending; 0 reserved for unknown
        pairs = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        vocab = {word: i + 1 for i, (word, _) in enumerate(pairs)}
        vocab["__unknown__"] = 0
        return vocab

    return _make_vocab(verb_counts), _make_vocab(noun_counts)


def _enrich(examples: list[dict], npz_dirs: list[str],
            verb_vocab: dict, noun_vocab: dict) -> tuple[list[dict], int]:
    """Add prefix_verb_ids, prefix_noun_ids, prefix_start_secs, prefix_end_secs."""
    cache: dict = {}
    n_missing = 0
    out = []

    for ex in examples:
        vname = ex["video_name"]
        if vname not in cache:
            cache[vname] = _load_meta(vname, npz_dirs)
        verbs, nouns, start_secs, end_secs = cache[vname]

        n = len(ex["prefix_npz_indices"])
        ex2 = dict(ex)

        if verbs is None:
            n_missing += 1
            ex2["prefix_verb_ids"]    = [0] * n
            ex2["prefix_noun_ids"]    = [0] * n
            ex2["prefix_start_secs"]  = [0.0] * n
            ex2["prefix_end_secs"]    = [0.0] * n
            out.append(ex2)
            continue

        vids, nids, tstarts, tends = [], [], [], []
        for idx in ex["prefix_npz_indices"]:
            vids.append(verb_vocab.get(str(verbs[idx]), 0))
            nids.append(noun_vocab.get(str(nouns[idx]), 0))
            tstarts.append(float(start_secs[idx]))
            tends.append(float(end_secs[idx]))

        ex2["prefix_verb_ids"]   = vids
        ex2["prefix_noun_ids"]   = nids
        ex2["prefix_start_secs"] = tstarts
        ex2["prefix_end_secs"]   = tends
        out.append(ex2)

    return out, n_missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_index",   required=True)
    parser.add_argument("--val_index",     required=True)
    parser.add_argument("--train_npz_dir", required=True)
    parser.add_argument("--val_npz_dir",   required=True)
    parser.add_argument("--train_npy_dir", required=True)
    parser.add_argument("--val_npy_dir",   required=True)
    parser.add_argument("--output_dir",    required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.train_index) as f:
        train_idx = json.load(f)
    with open(args.val_index) as f:
        val_idx = json.load(f)

    npz_dirs = [args.train_npz_dir, args.val_npz_dir]

    # build vocab from train examples only.
    print(f"Building vocabs from {len(train_idx['examples'])} train examples …")
    verb_vocab, noun_vocab = _build_vocabs(train_idx["examples"], npz_dirs)
    n_verbs = max(verb_vocab.values()) + 1
    n_nouns = max(noun_vocab.values()) + 1
    print(f"  verbs: {n_verbs - 1} types  →  n_verbs={n_verbs}")
    print(f"  nouns: {n_nouns - 1} types  →  n_nouns={n_nouns}")

    print("Enriching train…")
    train_enriched, train_miss = _enrich(train_idx["examples"], npz_dirs, verb_vocab, noun_vocab)
    print(f"  {len(train_enriched)} examples, {train_miss} missing npz")

    print("Enriching val…  (val-only verbs/nouns → id=0/__unknown__)")
    val_enriched, val_miss = _enrich(val_idx["examples"], npz_dirs, verb_vocab, noun_vocab)
    print(f"  {len(val_enriched)} examples, {val_miss} missing npz")

    # count how many val tokens fall back to __unknown__
    n_unk_v = sum(1 for ex in val_enriched for v in ex["prefix_verb_ids"] if v == 0)
    n_unk_n = sum(1 for ex in val_enriched for n in ex["prefix_noun_ids"] if n == 0)
    n_tok_v = sum(len(ex["prefix_verb_ids"]) for ex in val_enriched)
    n_tok_n = sum(len(ex["prefix_noun_ids"]) for ex in val_enriched)
    print(f"  val unknown-verb tokens: {n_unk_v}/{n_tok_v} ({100*n_unk_v/max(1,n_tok_v):.1f}%)")
    print(f"  val unknown-noun tokens: {n_unk_n}/{n_tok_n} ({100*n_unk_n/max(1,n_tok_n):.1f}%)")

    def _vocab_list(v: dict) -> list:
        return sorted(((w, i) for w, i in v.items()), key=lambda x: x[1])

    def _save(examples, orig, npy_dir, npz_dir, fname):
        out = {
            "features_dirs": [str(Path(npy_dir).resolve()), str(Path(npz_dir).resolve())],
            "feature_key": orig["feature_key"],
            "num_examples": len(examples),
            "n_verbs": n_verbs,
            "n_nouns": n_nouns,
            "verb_vocab": _vocab_list(verb_vocab),
            "noun_vocab": _vocab_list(noun_vocab),
            "examples": examples,
        }
        p = output_dir / fname
        with open(p, "w") as f:
            json.dump(out, f)
        print(f"Wrote {p}  ({len(examples)} examples)")

    _save(train_enriched, train_idx, args.train_npy_dir, args.train_npz_dir, "wm_train_index.json")
    _save(val_enriched,   val_idx,   args.val_npy_dir,   args.val_npz_dir,   "wm_val_index.json")
    print("Done.")


if __name__ == "__main__":
    main()
