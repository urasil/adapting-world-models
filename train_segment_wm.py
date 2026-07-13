# trains SegmentMaskPredictorAC for next-segment world-model pretraining.
# only correct segments (label==0) are trained on; --neg_subsample_ratio caps correct examples
# per video at n x that video's mistake count (mirrors the attentive-probe subsampling).
# loss: l1 between channel-layernorm-normalised prediction and target.
# usage: python train_segment_wm.py --train_index ... --val_index ... --train_split ...
#   --val_split ... --output_dir ... --predictor_ckpt ... --epochs 20 --grad_accum_steps 8

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.datasets.segment_wm_dataset import (
    LengthGroupedBatchSampler,
    SegmentWMDataset,
    compute_ctx_lens,
    wm_collate_fn,
)
from src.models.segment_predictor import segment_mask_predictor_ac

# reuse split/sampler utilities from the existing training script.
from train_mistake_detection import (
    VideoGroupedSampler,
    filter_excluded_tasks,
    load_split_file,
    set_seed,
    subsample_negatives,
)


def warmup_constant_decay_lr(step: int, total_steps: int, *, warmup_lr: float, peak_lr: float,
                              final_lr: float = 0.0,
                              warmup_frac: float = 4500 / 94500,
                              decay_frac: float = 4500 / 94500) -> float:
    """Warmup-constant-decay schedule (V-JEPA 2 Appendix B.1): linear warmup
    from warmup_lr to peak_lr, hold constant, then linear decay to final_lr.
    Phase lengths are the fixed fractions of total_steps matching the paper's
    4500/85500/4500-iteration split, since our total step count differs."""
    warmup_steps = round(total_steps * warmup_frac)
    decay_steps  = round(total_steps * decay_frac)
    decay_start  = total_steps - decay_steps

    if step < warmup_steps:
        return warmup_lr + (peak_lr - warmup_lr) * step / max(1, warmup_steps)
    if step < decay_start:
        return peak_lr
    progress = (step - decay_start) / max(1, decay_steps)
    return peak_lr + (final_lr - peak_lr) * min(1.0, progress)


# data split

def wm_train_split(dataset: SegmentWMDataset, train_file: str, val_file: str) -> list[int]:
    """Positions in `dataset` (built from --train_index) whose video is in the
    official train split. `dataset` only contains train videos by construction
    (--train_index is pre-split), so val examples are never expected here --
    val comes from the separate --val_index file. The val split file is only
    used to check for train/val overlap and to flag stray videos."""
    train_videos = load_split_file(train_file)
    val_videos   = load_split_file(val_file)

    overlap = train_videos & val_videos
    if overlap:
        raise ValueError(f"Split files overlap on {len(overlap)} videos.")

    train_idx, dropped = [], []
    for pos, orig in enumerate(dataset._indices):
        name = dataset._base.examples[orig]["video_name"]
        if name in train_videos:
            train_idx.append(pos)
        elif name not in val_videos:
            dropped.append(name)

    if dropped:
        print(f"WARNING: dropped {len(set(dropped))} videos not in either split.")
    if not train_idx:
        raise ValueError("No training examples after split.")
    return train_idx


# checkpoint loading

def load_predictor_ckpt(model: nn.Module, ckpt_path: str, device) -> None:
    """Load predictor weights from a V-JEPA checkpoint (strict=False).

    Common checkpoint formats are handled: bare state-dict, or a dict with
    a 'target_encoder', 'predictor', or 'model' key.
    """
    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict):
        for key in ("predictor", "target_predictor", "model"):
            if key in raw:
                raw = raw[key]
                break
    missing, unexpected = model.load_state_dict(raw, strict=False)
    print(f"Predictor ckpt loaded from {ckpt_path}")
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")


# checkpoint saving

def save_checkpoint(path: Path, *, epoch, step, resume_iter, model, optimizer, args, best_val_l1):
    """Save to a temp file then rename, so a crash mid-write can never leave
    a truncated checkpoint at `path` (os.replace is atomic on the same
    filesystem)."""
    ckpt = {
        "epoch": epoch, "step": step, "resume_iter": resume_iter,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "args": vars(args), "best_val_l1": best_val_l1,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


# training loop

def train_one_epoch(model, loader, optimizer, device, *, warmup_lr, peak_lr, total_steps,
                    start_step, grad_clip, use_bf16, log_every, accum_steps,
                    epoch, start_iter, ckpt_every_steps, ckpt_path, best_val_l1, args):
    model.train()
    running = 0.0
    n_seen = 0
    n_skipped = 0
    step = start_step
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    n_micro = len(loader)

    for it, batch in enumerate(loader):
        if it < start_iter:
            continue  # fast-forward past micro-batches already done before a mid-epoch resume
        ctx_f, ctx_v, ctx_n, tgt_f, tgt_v, tgt_n, labels, ctx_lens = batch

        ctx_f    = ctx_f.to(device, dtype=torch.float32, non_blocking=True)
        tgt_f    = tgt_f.to(device, dtype=torch.float32, non_blocking=True)
        ctx_v    = ctx_v.to(device, non_blocking=True)
        ctx_n    = ctx_n.to(device, non_blocking=True)
        tgt_v    = tgt_v.to(device, non_blocking=True)
        tgt_n    = tgt_n.to(device, non_blocking=True)
        ctx_lens = ctx_lens.to(device, non_blocking=True)

        lr = warmup_constant_decay_lr(step, total_steps, warmup_lr=warmup_lr, peak_lr=peak_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            pred = model(ctx_f, ctx_v, ctx_n, tgt_v, tgt_n, ctx_lengths=ctx_lens)   # [b, t, h*w, e]

        # reshape target to match output shape; compute l1 in float32.
        B, T, HW, E = pred.shape
        target = tgt_f.view(B, T, HW, E)

        pred_norm   = model.normalise_targets(pred.float())
        target_norm = model.normalise_targets(target.detach())
        loss = F.l1_loss(pred_norm, target_norm)

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            print(f"WARNING: non-finite loss at iter {it+1}, skipping.")
            n_skipped += 1
            continue

        (loss / accum_steps).backward()
        running += loss_val * B
        n_seen  += B

        is_boundary = ((it + 1) % accum_steps == 0) or (it + 1 == n_micro)
        if is_boundary:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            is_epoch_end = (it + 1 == n_micro)
            if ckpt_every_steps and not is_epoch_end and step % ckpt_every_steps == 0:
                save_checkpoint(ckpt_path, epoch=epoch, step=step, resume_iter=it + 1,
                                model=model, optimizer=optimizer, args=args,
                                best_val_l1=best_val_l1)
                print(f"    [mid-epoch checkpoint] epoch {epoch} iter {it+1}/{n_micro} "
                      f"step {step}", flush=True)

        if (it + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"    iter {it+1}/{n_micro}  l1={running/n_seen:.4f}  "
                  f"lr={lr:.2e}  ips={n_seen/elapsed:.1f}", flush=True)

    return {"loss": running / max(1, n_seen), "end_step": step, "n_skipped": n_skipped}


@torch.no_grad()
def validate(model, loader, device, use_bf16):
    model.eval()
    running = 0.0
    n_seen = 0
    for batch in loader:
        ctx_f, ctx_v, ctx_n, tgt_f, tgt_v, tgt_n, labels, ctx_lens = batch
        ctx_f    = ctx_f.to(device, dtype=torch.float32, non_blocking=True)
        tgt_f    = tgt_f.to(device, dtype=torch.float32, non_blocking=True)
        ctx_v    = ctx_v.to(device, non_blocking=True)
        ctx_n    = ctx_n.to(device, non_blocking=True)
        tgt_v    = tgt_v.to(device, non_blocking=True)
        tgt_n    = tgt_n.to(device, non_blocking=True)
        ctx_lens = ctx_lens.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            pred = model(ctx_f, ctx_v, ctx_n, tgt_v, tgt_n, ctx_lengths=ctx_lens)

        B, T, HW, E = pred.shape
        target = tgt_f.view(B, T, HW, E)
        pred_norm   = model.normalise_targets(pred.float())
        target_norm = model.normalise_targets(target)
        running += F.l1_loss(pred_norm, target_norm).item() * B
        n_seen  += B

    return running / max(1, n_seen)


# main

def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--train_index",  required=True)
    parser.add_argument("--val_index",    required=True)
    parser.add_argument("--train_split",  required=True, help="holoassist/splits/train-v1_2.txt")
    parser.add_argument("--val_split",    required=True, help="holoassist/splits/val-v1_2.txt")
    # model
    parser.add_argument("--predictor_ckpt", default=None,
                        help="V-JEPA 2.1 predictor checkpoint for weight initialisation.")
    parser.add_argument("--embed_dim",           type=int, default=1024)
    parser.add_argument("--predictor_embed_dim", type=int, default=1024)
    parser.add_argument("--depth",               type=int, default=24)
    parser.add_argument("--num_heads",           type=int, default=16)
    parser.add_argument("--act_ckpt", action="store_true", default=False,
                        help="Enable activation checkpointing (~4/3x compute, lower peak memory). "
                             "Required at large S (e.g. S=16); wasteful at S<=8 where peak fits "
                             "in 80 GB without it.")
    parser.add_argument("--checkpoint_min_segs", type=int, default=None,
                        help="Only apply activation checkpointing when an example's "
                             "n_segs (ctx_len+1) >= this. Shorter examples skip the "
                             "~30-40%% recompute overhead. None = checkpoint always "
                             "(when --act_ckpt set).")
    parser.add_argument("--max_context_segs",        type=int,   default=16)
    parser.add_argument("--max_gap_to_target_secs",  type=float, default=2.0,
                        help="Drop examples where last-context→target gap > this (seconds).")
    parser.add_argument("--neg_subsample_ratio", type=float, default=None,
                        help="Cap correct (label==0) training examples per video at this "
                             "many × the video's mistake count, then drop the mistakes "
                             "themselves (training remains correct-only). Mirrors the "
                             "subsampling used for the attentive probe; shrinks dataset size.")
    parser.add_argument("--exclude_tasks", type=str, default=None,
                        help="Comma-separated task keywords to drop entirely from training "
                             "(e.g. 'marius,knarrevik,rashult' to keep only the gladom IKEA "
                             "task). Matched against the task keyword parsed from video_name "
                             "via task_of_video. Removes whole videos, applied before "
                             "--neg_subsample_ratio.")
    parser.add_argument("--normalize_targets",       action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Examples of identical ctx_len are grouped into each batch "
                             "(LengthGroupedBatchSampler) so wm_collate_fn never pads and "
                             "FlashAttention v2 keeps firing, same as at batch_size=1.")
    # optimisation
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--lr",              type=float, default=4.25e-4,
                        help="Peak LR (V-JEPA 2-AC Appendix B.1 value).")
    parser.add_argument("--warmup_lr",       type=float, default=7.5e-5,
                        help="Initial LR before warmup (V-JEPA 2-AC Appendix B.1 value).")
    parser.add_argument("--weight_decay",    type=float, default=0.04,
                        help="Constant weight decay (V-JEPA 2-AC Appendix B.1 value).")
    parser.add_argument("--grad_clip",       type=float, default=1.0)
    parser.add_argument("--grad_accum_steps",type=int,   default=64,
                        help="With batch_size=4 this gives an effective batch of 256, "
                             "matching V-JEPA 2-AC Appendix B.1.")
    parser.add_argument("--precision",       choices=["bf16", "fp32"], default="bf16")
    # runtime
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--log_every",    type=int, default=100)
    parser.add_argument("--ckpt_every_steps", type=int, default=20,
                        help="Save last.pt every this many optimizer steps, in addition "
                             "to the save at every epoch boundary, so a crash mid-epoch "
                             "loses at most this many steps of progress. 0 disables "
                             "mid-epoch checkpointing (checkpoint only at epoch end).")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = (args.precision == "bf16" and device.type == "cuda")
    print(f"Device: {device}  precision: {args.precision}", flush=True)

    # ---- dataset -----------------------------------------------------------
    with open(args.train_index) as f:
        idx_meta = json.load(f)
    n_verbs = idx_meta["n_verbs"]
    n_nouns = idx_meta["n_nouns"]

    full_ds = SegmentWMDataset(
        args.train_index,
        train_mode=True,
        max_context_segs=args.max_context_segs,
        max_gap_to_target_secs=args.max_gap_to_target_secs,
    )
    val_ds_full = SegmentWMDataset(
        args.val_index,
        train_mode=False,
        max_context_segs=args.max_context_segs,
        max_gap_to_target_secs=args.max_gap_to_target_secs,
    )

    train_pos = wm_train_split(full_ds, args.train_split, args.val_split)

    if args.exclude_tasks is not None:
        exclude = [t.strip() for t in args.exclude_tasks.split(",") if t.strip()]
        train_pos = filter_excluded_tasks(full_ds, train_pos, exclude)
    if args.neg_subsample_ratio is not None:
        train_pos = subsample_negatives(full_ds, train_pos, args.neg_subsample_ratio, seed=args.seed)
    train_pos = [i for i in train_pos if full_ds.examples[i]["label"] == 0]
    print(f"Training examples (correct-only"
          f"{f', excluding tasks {args.exclude_tasks}' if args.exclude_tasks is not None else ''}"
          f"{f', capped at {args.neg_subsample_ratio}:1 per video' if args.neg_subsample_ratio is not None else ''}"
          f"): {len(train_pos)}", flush=True)

    train_ds = Subset(full_ds, train_pos)

    # predictor is trained on correct segments only; keep validation l1 on the
    # same distribution so it's a fair comparison to the training loss. the
    # mistake-detection signal (predicted vs. encoded features on mistake
    # segments) is computed downstream, not as part of this val loss.
    val_pos = list(range(len(val_ds_full)))
    if args.exclude_tasks is not None:
        val_pos = filter_excluded_tasks(val_ds_full, val_pos, exclude)
    val_pos = [i for i in val_pos if val_ds_full.examples[i]["label"] == 0]
    val_ds  = Subset(val_ds_full, val_pos)

    print(f"Train examples: {len(train_ds)}  Val examples (correct-only): {len(val_ds)}", flush=True)

    pin = device.type == "cuda"
    train_sampler = VideoGroupedSampler(train_ds, seed=args.seed)
    train_batch_sampler = LengthGroupedBatchSampler(
        train_sampler, compute_ctx_lens(train_ds, args.max_context_segs),
        batch_size=args.batch_size, drop_last=True,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_batch_sampler,
        num_workers=args.num_workers, collate_fn=wm_collate_fn,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_batch_sampler = LengthGroupedBatchSampler(
        range(len(val_ds)), compute_ctx_lens(val_ds, args.max_context_segs),
        batch_size=args.batch_size, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_batch_sampler,
        num_workers=args.num_workers, collate_fn=wm_collate_fn,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # ---- model -------------------------------------------------------------
    model = segment_mask_predictor_ac(
        img_size=(384, 384),
        patch_size=16,
        num_frames=64,
        tubelet_size=2,
        embed_dim=args.embed_dim,
        predictor_embed_dim=args.predictor_embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        n_verbs=n_verbs,
        n_nouns=n_nouns,
        max_context_segs=args.max_context_segs,
        normalize_targets=args.normalize_targets,
        use_activation_checkpointing=args.act_ckpt,
        checkpoint_min_segs=args.checkpoint_min_segs,
    ).to(device)

    if args.predictor_ckpt:
        load_predictor_ckpt(model, args.predictor_ckpt, device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params/1e6:.2f}M", flush=True)

    # ---- optimiser ---------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps     = steps_per_epoch * args.epochs
    print(f"Total steps: {total_steps} ({steps_per_epoch}/epoch × {args.epochs} epochs)",
          flush=True)

    # ---- resume ------------------------------------------------------------
    history     = []
    best_val_l1 = float("inf")
    start_epoch = 1
    start_iter  = 0
    step        = 0

    resume_path = output_dir / "last.pt"
    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step        = ckpt.get("step", 0)
        best_val_l1 = ckpt.get("best_val_l1", float("inf"))
        resume_iter = ckpt.get("resume_iter") or 0
        if resume_iter:
            # crashed mid-epoch: redo the rest of this same epoch from where
            # the last mid-epoch checkpoint left off.
            start_epoch = ckpt["epoch"]
            start_iter  = resume_iter
            print(f"Resumed mid-epoch: epoch {start_epoch}, iter {start_iter}, "
                  f"step {step}", flush=True)
        else:
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from epoch {ckpt['epoch']}", flush=True)
    elif args.resume:
        print("--resume set but no checkpoint found, starting fresh.", flush=True)

    # ---- training loop -----------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        t0 = time.time()

        tr = train_one_epoch(
            model, train_loader, optimizer, device,
            warmup_lr=args.warmup_lr, peak_lr=args.lr, total_steps=total_steps, start_step=step,
            grad_clip=args.grad_clip, use_bf16=use_bf16,
            log_every=args.log_every, accum_steps=args.grad_accum_steps,
            epoch=epoch, start_iter=start_iter, ckpt_every_steps=args.ckpt_every_steps,
            ckpt_path=resume_path, best_val_l1=best_val_l1, args=args,
        )
        start_iter = 0  # only the first (possibly resumed) epoch skips ahead
        step = tr["end_step"]
        val_l1 = validate(model, val_loader, device, use_bf16)

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:02d}/{args.epochs}]  "
            f"tr_l1={tr['loss']:.4f}  val_l1={val_l1:.4f}  "
            f"skipped={tr['n_skipped']}  ({elapsed:.0f}s)",
            flush=True,
        )

        history.append({"epoch": epoch, "train_l1": tr["loss"],
                        "val_l1": val_l1, "time_s": elapsed})

        save_checkpoint(resume_path, epoch=epoch, step=step, resume_iter=0,
                        model=model, optimizer=optimizer, args=args, best_val_l1=best_val_l1)

        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            for old in output_dir.glob("best_val_l1_ep*.pt"):
                old.unlink()
            save_checkpoint(output_dir / f"best_val_l1_ep{epoch:02d}.pt", epoch=epoch, step=step,
                            resume_iter=0, model=model, optimizer=optimizer, args=args,
                            best_val_l1=best_val_l1)
            print(f"  → new best val_l1={val_l1:.4f}", flush=True)

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. best_val_l1={best_val_l1:.4f}", flush=True)


if __name__ == "__main__":
    main()
