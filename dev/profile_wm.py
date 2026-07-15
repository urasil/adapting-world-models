from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, schedule

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datasets.segment_wm_dataset import (
    LengthGroupedBatchSampler,
    SegmentWMDataset,
    compute_ctx_lens,
    wm_collate_fn,
)
from src.models.segment_predictor import segment_mask_predictor_ac
from training.train_mistake_detection import (
    VideoGroupedSampler,
    filter_excluded_tasks,
    load_split_file,
    set_seed,
)

def split_indices(dataset, train_file):
    train_videos = load_split_file(train_file)
    return [pos for pos, orig in enumerate(dataset._indices) if dataset._base.examples[orig]["video_name"] in train_videos]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_index", required=True)
    ap.add_argument("--train_split", required=True)
    ap.add_argument("--max_context_segs", type=int, default=4)
    ap.add_argument("--max_gap_to_target_secs", type=float, default=2.0)
    ap.add_argument("--exclude_tasks", type=str, default=None,
                    help="Comma-separated task keywords to drop entirely "
                         "(e.g. 'marius,knarrevik,rashult' to keep only gladom).")
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--embed_dim", type=int, default=1024)
    ap.add_argument("--num_heads", type=int, default=16)
    ap.add_argument("--act_ckpt", action="store_true", default=False)
    ap.add_argument("--checkpoint_min_segs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup_steps", type=int, default=3)
    ap.add_argument("--active_steps", type=int, default=5)
    ap.add_argument("--trace_out", default="wm_trace.json")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "profiler script requires a GPU"

    with open(args.train_index) as f:
        idx_meta = json.load(f)
    n_verbs, n_nouns = idx_meta["n_verbs"], idx_meta["n_nouns"]

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
    train_idx = [i for i in train_idx if ds.examples[i]["label"] == 0]
    from torch.utils.data import Subset, DataLoader
    train_ds = Subset(ds, train_idx)
    sampler = VideoGroupedSampler(train_ds, seed=args.seed)
    batch_sampler = LengthGroupedBatchSampler(
        sampler, compute_ctx_lens(train_ds, args.max_context_segs),
        batch_size=args.batch_size, drop_last=True,
    )
    loader = DataLoader(
        train_ds, batch_sampler=batch_sampler,
        num_workers=2, collate_fn=wm_collate_fn,
        pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    model = segment_mask_predictor_ac(
        img_size=(384, 384), patch_size=16, num_frames=64, tubelet_size=2,
        embed_dim=1024, predictor_embed_dim=args.embed_dim,
        depth=args.depth, num_heads=args.num_heads,
        n_verbs=n_verbs, n_nouns=n_nouns,
        max_context_segs=args.max_context_segs,
        normalize_targets=True,
        use_activation_checkpointing=args.act_ckpt,
        checkpoint_min_segs=args.checkpoint_min_segs,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    loader_iter = iter(loader)

    def step():
        batch = next(loader_iter)
        ctx_f, ctx_v, ctx_n, tgt_f, tgt_v, tgt_n, labels, ctx_lens = batch
        ctx_f = ctx_f.to(device, dtype=torch.bfloat16, non_blocking=True)
        tgt_f = tgt_f.to(device, dtype=torch.bfloat16, non_blocking=True)
        ctx_v = ctx_v.to(device, non_blocking=True)
        ctx_n = ctx_n.to(device, non_blocking=True)
        tgt_v = tgt_v.to(device, non_blocking=True)
        tgt_n = tgt_n.to(device, non_blocking=True)
        ctx_ld = ctx_lens.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(ctx_f, ctx_v, ctx_n, tgt_v, tgt_n, ctx_lengths=ctx_ld)
        B, T, HW, E = pred.shape
        target = tgt_f.view(B, T, HW, E)
        pred_n = model.normalise_targets(pred.float())
        tgt_n_ = model.normalise_targets(target.detach().float())
        loss = F.l1_loss(pred_n, tgt_n_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    print(f"Warmup: {args.warmup_steps} steps …")
    for _ in range(args.warmup_steps):
        step()
    torch.cuda.synchronize()

    print(f"Profiling: {args.active_steps} steps …")
    prof_schedule = schedule(wait=0, warmup=0, active=args.active_steps, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.active_steps):
            step()
            prof.step()
    torch.cuda.synchronize()

    print("\n=== Top ops by self CUDA time ===")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))

    print("\n=== Top ops by self CPU time (launch overhead, Python loop cost) ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

    prof.export_chrome_trace(args.trace_out)
    print(f"\nChrome trace written to {args.trace_out} (open at chrome://tracing)")

if __name__ == "__main__":
    main()
