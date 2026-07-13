# smoke test for SegmentMaskPredictorAC training - runs the real pipeline (data/fwd/bwd/optim)
# with production hyperparams (grad_accum=128, s=16, bf16, activation checkpointing), logging to
# a timestamped file so results survive after the gpu session closes.
# passes if: no oom, every loss/grad-norm is finite (grad norm < 1000), at least one optimizer
# step completes, and the loss right after that step is finite.
# usage: python smoke_test_wm.py --train_index ... --train_split ... --val_split ... --minutes 25

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.datasets.segment_wm_dataset import (
    LengthGroupedBatchSampler,
    SegmentWMDataset,
    compute_ctx_lens,
    wm_collate_fn,
)
from src.models.segment_predictor import segment_mask_predictor_ac
from train_mistake_detection import (
    VideoGroupedSampler,
    filter_excluded_tasks,
    load_split_file,
    set_seed,
    subsample_negatives,
)


# helpers

class Tee:
    """Write to both stdout and a file, flushing after every line."""
    def __init__(self, path: Path):
        self._f = open(path, "w", buffering=1)   # line-buffered
        self._stdout = sys.stdout

    def write(self, msg: str):
        self._stdout.write(msg)
        self._f.write(msg)

    def flush(self):
        self._stdout.flush()
        self._f.flush()

    def close(self):
        self._f.close()


def mem_str() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    peak = torch.cuda.max_memory_reserved() / 1e9
    cur  = torch.cuda.memory_reserved()     / 1e9
    return f"peak={peak:.1f}G cur={cur:.1f}G"


def grad_norm(model) -> tuple[float, bool]:
    """Return (total_norm, has_nan) for all parameter gradients."""
    total, has_nan = 0.0, False
    for p in model.parameters():
        if p.grad is not None:
            n = float(p.grad.data.norm(2))
            if not math.isfinite(n):
                has_nan = True
            total += n * n
    return math.sqrt(total), has_nan


def split_indices(dataset, train_file):
    train_videos = load_split_file(train_file)
    return [
        pos for pos, orig in enumerate(dataset._indices)
        if dataset._base.examples[orig]["video_name"] in train_videos
    ]


def next_batch(loader_iter, loader, sampler):
    try:
        return next(loader_iter)
    except StopIteration:
        epoch = getattr(sampler, "_epoch", 0) + 1
        sampler._epoch = epoch
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        return next(iter(loader))


# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_index",            required=True)
    parser.add_argument("--train_split",            required=True)
    parser.add_argument("--val_split",              required=True)
    parser.add_argument("--max_context_segs",       type=int,   default=16)
    parser.add_argument("--batch_size",             type=int,   default=1,
                        help="Examples of identical ctx_len are grouped into each batch "
                             "(LengthGroupedBatchSampler) so wm_collate_fn never pads and "
                             "FlashAttention v2 keeps firing, same as at batch_size=1.")
    parser.add_argument("--max_gap_to_target_secs", type=float, default=2.0)
    parser.add_argument("--neg_subsample_ratio",    type=float, default=None,
                        help="Cap correct (label==0) examples per video at this many × "
                             "the video's mistake count, then drop the mistakes "
                             "(training remains correct-only). Mirrors the attentive probe.")
    parser.add_argument("--exclude_tasks", type=str, default=None,
                        help="Comma-separated task keywords to drop entirely "
                             "(e.g. 'marius,knarrevik,rashult' to keep only gladom). "
                             "Applied before --neg_subsample_ratio.")
    parser.add_argument("--grad_accum",             type=int,   default=128,
                        help="Micro-steps per optimizer step (match real training).")
    parser.add_argument("--minutes",                type=float, default=25,
                        help="Wall-clock budget. A partial optimizer step is forced "
                             "at the deadline so training is always proven end-to-end.")
    parser.add_argument("--lr",                     type=float, default=1e-4)
    parser.add_argument("--seed",                   type=int,   default=0)
    parser.add_argument("--depth",                  type=int,   default=24)
    parser.add_argument("--embed_dim",              type=int,   default=1024,
                        help="Predictor internal width (encoder output dim is always 1024). "
                             "Must satisfy embed_dim %% num_heads == 0 and head_dim in {32,64,128}.")
    parser.add_argument("--num_heads",              type=int,   default=16)
    parser.add_argument("--act_ckpt", action="store_true", default=False,
                        help="Enable activation checkpointing (~4/3× compute, lower peak memory). "
                             "Required at S=16; wasteful at S≤8 where peak fits in 80 GB without it.")
    parser.add_argument("--checkpoint_min_segs", type=int, default=None,
                        help="Only apply checkpointing when an example's n_segs "
                             "(ctx_len+1) >= this. Shorter examples skip the recompute "
                             "overhead. None = checkpoint always (when --act_ckpt set).")
    args = parser.parse_args()

    # log file
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    logpath = Path(f"smoke_test_wm_{stamp}.log")
    tee     = Tee(logpath)
    sys.stdout = tee   # all print() goes to both terminal and file

    def p(*a, **kw):
        kw.setdefault("flush", True)
        print(*a, **kw)

    # header
    p(f"{'='*68}")
    p(f"WM Smoke Test  {stamp}")
    p(f"Log: {logpath.resolve()}")
    p(f"{'='*68}")

    set_seed(args.seed)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"
    if device.type == "cuda":
        p(f"GPU : {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")
    p(f"Config: S={args.max_context_segs}  depth={args.depth}  "
      f"predictor_dim={args.embed_dim}  heads={args.num_heads}  "
      f"grad_accum={args.grad_accum}  lr={args.lr}  "
      f"budget={args.minutes:.0f} min  bf16={use_bf16}  act_ckpt={args.act_ckpt}")

    t_start  = time.time()
    deadline = t_start + args.minutes * 60

    # dataset
    with open(args.train_index) as f:
        idx_meta = json.load(f)
    n_verbs, n_nouns = idx_meta["n_verbs"], idx_meta["n_nouns"]
    p(f"\nVocab: {n_verbs} verbs  {n_nouns} nouns")
    p("Loading dataset …")

    ds = SegmentWMDataset(
        args.train_index,
        train_mode=True,
        max_context_segs=args.max_context_segs,
        max_gap_to_target_secs=args.max_gap_to_target_secs,
    )
    train_idx = split_indices(ds, args.train_split)
    if args.exclude_tasks is not None:
        exclude = [t.strip() for t in args.exclude_tasks.split(",") if t.strip()]
        train_idx = filter_excluded_tasks(ds, train_idx, exclude)
    if args.neg_subsample_ratio is not None:
        train_idx = subsample_negatives(ds, train_idx, args.neg_subsample_ratio, seed=args.seed)
    train_idx = [i for i in train_idx if ds.examples[i]["label"] == 0]
    train_ds  = Subset(ds, train_idx)
    p(f"Train examples (after official split + filters"
      f"{f', excluding tasks {args.exclude_tasks}' if args.exclude_tasks is not None else ''}"
      f"{f', capped at {args.neg_subsample_ratio}:1 per video' if args.neg_subsample_ratio is not None else ''}"
      f", correct-only): {len(train_ds)}")

    # count examples that reach full s=max_context_segs
    n_full = sum(
        1 for i in train_idx
        if min(len(ds._base.examples[ds._indices[i]]["prefix_npz_indices"]) - 1,
               args.max_context_segs) == args.max_context_segs
    )
    p(f"  with full S={args.max_context_segs} context : "
      f"{n_full} ({100*n_full/max(len(train_idx),1):.1f}%)")

    sampler = VideoGroupedSampler(train_ds, seed=args.seed)
    batch_sampler = LengthGroupedBatchSampler(
        sampler, compute_ctx_lens(train_ds, args.max_context_segs),
        batch_size=args.batch_size, drop_last=True,
    )
    loader  = DataLoader(
        train_ds, batch_sampler=batch_sampler,
        num_workers=2, collate_fn=wm_collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True, prefetch_factor=2,
    )
    p(f"  micro-steps per epoch: {len(loader)}  (batch_size={args.batch_size})")

    # model
    # encoder output dim is fixed by v-jepa2 features on disk (always 1024).
    # --embed_dim controls the predictor's internal width only.
    encoder_dim   = 1024
    predictor_dim = args.embed_dim
    head_dim      = predictor_dim // args.num_heads
    if predictor_dim % args.num_heads != 0:
        raise ValueError(f"embed_dim={predictor_dim} must be divisible by num_heads={args.num_heads}")
    if head_dim % 8 != 0:
        p(f"WARNING: head_dim={head_dim} is not a multiple of 8 — SDPA will use the "
          f"math backend (correct but slow). Use --num_heads {predictor_dim // 64} for head_dim=64.")
    if not args.act_ckpt:
        tps = 18432
        peak_gb = args.depth * (args.max_context_segs + 1) ** 2 * tps * predictor_dim * 2 / 1e9
        p(f"WARNING: act_ckpt=False — estimated peak ≈ {peak_gb:.0f} GB at "
          f"S={args.max_context_segs}, depth={args.depth}, D={predictor_dim}. "
          f"Pass --act_ckpt if this exceeds 75 GB.")
    p("\nBuilding model …")
    model = segment_mask_predictor_ac(
        img_size=(384, 384), patch_size=16,
        num_frames=64, tubelet_size=2,
        embed_dim=encoder_dim,
        predictor_embed_dim=predictor_dim,
        depth=args.depth, num_heads=args.num_heads,
        n_verbs=n_verbs, n_nouns=n_nouns,
        max_context_segs=args.max_context_segs,
        normalize_targets=True,
        use_activation_checkpointing=args.act_ckpt,
        checkpoint_min_segs=args.checkpoint_min_segs,
    ).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    p(f"Trainable params: {n_params/1e6:.1f}M")
    p(f"Memory after model load: {mem_str()}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=1e-2, betas=(0.9, 0.999),
    )

    # training loop
    p(f"\nRunning for up to {args.minutes:.0f} minutes "
      f"(grad_accum={args.grad_accum}; optimizer step forced at deadline) …")
    hdr = (f"\n{'micro':>6}  {'elapsed':>8}  {'ctx_len':>7}  {'loss':>8}  "
           f"{'grad_norm':>10}  {'mem_peak':>10}  {'s/step':>7}  note")
    p(hdr)
    p("-" * 75)

    losses_all:  list[float] = []
    losses_window: list[float] = []   # current accum window
    nan_events:  list[str]   = []
    oom_caught   = False
    opt_steps    = 0
    micro_total  = 0
    step_losses_after_opt: list[float] = []   # first few losses after first opt step

    torch.cuda.reset_peak_memory_stats(device)
    t_micro = time.time()
    loader_iter = iter(loader)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    def do_optimizer_step(note: str) -> tuple[float, bool]:
        """Fire the optimizer. Returns (grad_norm_val, has_nan_grad)."""
        gn, has_nan = grad_norm(model)
        if not has_nan:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return gn, has_nan

    try:
        while True:
            past_deadline = time.time() >= deadline
            accum_full    = len(losses_window) >= args.grad_accum

            # fire optimizer if accum is full or deadline arrived with pending grads
            if (accum_full or (past_deadline and losses_window)):
                gn, has_nan = do_optimizer_step(
                    "scheduled" if accum_full else "forced-at-deadline"
                )
                opt_steps += 1
                note = "OPT" if accum_full else "OPT(forced)"
                if has_nan:
                    note += " NaN-grad!"
                    nan_events.append(f"opt step {opt_steps}: NaN gradient (gn={gn:.3f})")
                avg_w = sum(losses_window) / max(len(losses_window), 1)
                peak  = torch.cuda.max_memory_reserved(device) / 1e9
                elapsed = time.time() - t_start
                p(f"{'':>6}  {elapsed:>7.1f}s  {'':>7}  {avg_w:>8.4f}  "
                  f"{gn:>10.3f}  {peak:>9.2f}G  {'':>7}  {note}")
                torch.cuda.reset_peak_memory_stats(device)
                losses_window.clear()

            if past_deadline:
                break

            # micro-step
            batch = next_batch(loader_iter, loader, sampler)
            ctx_f, ctx_v, ctx_n, tgt_f, tgt_v, tgt_n, labels, ctx_lens = batch
            ctx_len_val = int(ctx_lens[0])

            ctx_f    = ctx_f.to(device, dtype=torch.bfloat16, non_blocking=True)
            tgt_f    = tgt_f.to(device, dtype=torch.bfloat16, non_blocking=True)
            ctx_v    = ctx_v.to(device, non_blocking=True)
            ctx_n    = ctx_n.to(device, non_blocking=True)
            tgt_v    = tgt_v.to(device, non_blocking=True)
            tgt_n    = tgt_n.to(device, non_blocking=True)
            ctx_ld   = ctx_lens.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                 enabled=use_bf16):
                pred = model(ctx_f, ctx_v, ctx_n, tgt_v, tgt_n, ctx_lengths=ctx_ld)

            B, T, HW, E = pred.shape
            target   = tgt_f.view(B, T, HW, E)
            pred_n   = model.normalise_targets(pred.float())
            tgt_n_   = model.normalise_targets(target.detach().float())
            loss     = F.l1_loss(pred_n, tgt_n_)
            loss_val = float(loss)

            micro_total += 1
            step_s = time.time() - t_micro
            t_micro = time.time()
            peak_gb = torch.cuda.max_memory_reserved(device) / 1e9
            elapsed = time.time() - t_start

            if not math.isfinite(loss_val):
                msg = f"micro {micro_total}: loss={loss_val} ctx_len={ctx_len_val}"
                nan_events.append(msg)
                note = "*** NaN/Inf loss"
                p(f"{micro_total:>6}  {elapsed:>7.1f}s  {ctx_len_val:>7}  "
                  f"{loss_val:>8}  {'':>10}  {peak_gb:>9.2f}G  {step_s:>6.1f}s  {note}")
                optimizer.zero_grad(set_to_none=True)
                losses_window.clear()
                torch.cuda.reset_peak_memory_stats(device)
                continue

            (loss / args.grad_accum).backward()
            losses_all.append(loss_val)
            losses_window.append(loss_val)

            # track first few losses after the first optimizer step
            if opt_steps >= 1 and len(step_losses_after_opt) < 5:
                step_losses_after_opt.append(loss_val)

            # partial gradient norm (accumulated so far — proxy, not final)
            gn_partial, _ = grad_norm(model)

            note = f"{len(losses_window)}/{args.grad_accum}"
            p(f"{micro_total:>6}  {elapsed:>7.1f}s  {ctx_len_val:>7}  "
              f"{loss_val:>8.4f}  {gn_partial:>10.3f}  {peak_gb:>9.2f}G  "
              f"{step_s:>6.1f}s  {note}")
            torch.cuda.reset_peak_memory_stats(device)

    except torch.cuda.OutOfMemoryError as e:
        oom_caught = True
        p(f"\n*** CUDA OOM: {e}")

    # final verdict
    total_s = time.time() - t_start
    p(f"\n{'='*68}")
    p("SMOKE TEST RESULTS")
    p(f"{'='*68}")
    p(f"  Wall time             : {total_s/60:.1f} min")
    p(f"  Micro-steps           : {micro_total}")
    p(f"  Optimizer steps       : {opt_steps}")
    p(f"  NaN/Inf events        : {len(nan_events)}")
    for ev in nan_events:
        p(f"    - {ev}")

    if losses_all:
        n4 = max(1, len(losses_all) // 4)
        p(f"  Loss first-quarter avg: {sum(losses_all[:n4])/n4:.4f}")
        p(f"  Loss last-quarter  avg: {sum(losses_all[-n4:])/n4:.4f}")
        p(f"  Loss range            : [{min(losses_all):.4f}, {max(losses_all):.4f}]")
    if step_losses_after_opt:
        p(f"  First losses after opt: {[f'{x:.4f}' for x in step_losses_after_opt]}")

    passed = (
        not oom_caught
        and len(nan_events) == 0
        and opt_steps >= 1
        and micro_total >= 1
        and (not losses_all or max(losses_all) < 10.0)
    )
    verdict = "PASS" if passed else "FAIL"
    p(f"\n  Verdict : {verdict}")
    if not passed:
        reasons = []
        if oom_caught:                          reasons.append("CUDA OOM")
        if nan_events:                          reasons.append("NaN/Inf events")
        if opt_steps == 0:                      reasons.append("no optimizer step completed")
        if losses_all and max(losses_all) >= 10: reasons.append("loss >= 10")
        for r in reasons:
            p(f"    reason: {r}")
    p(f"{'='*68}")
    p(f"Full log: {logpath.resolve()}")

    sys.stdout = tee._stdout   # restore
    tee.close()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
