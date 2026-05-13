#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""End-to-end Llama-1B SM120 single-token decode benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from machete.kernels.llama1b import (
    LLAMA1B_HEAD_DIM,
    LLAMA1B_HIDDEN,
    LLAMA1B_INTERMEDIATE,
    LLAMA1B_KV_DIM,
    LLAMA1B_Q_DIM,
    LLAMA1B_SM120_FINAL_MATVEC_BLOCK,
    LLAMA1B_SM120_MATVEC_BLOCK,
    LLAMA1B_SM120_THREADS_PER_BLOCK,
    LLAMA1B_VOCAB,
    schedule_decode_model_sm120,
)
from machete.megakernel import Megakernel, MegakernelConfig, OverlapTileScheduler


def _randn(shape, *, dtype=torch.bfloat16, scale=0.02):
    return torch.randn(*shape, dtype=dtype, device="cuda") * scale


def make_llama1b_decode_state(
    *,
    batch: int,
    cache_len: int,
    num_layers: int,
    include_final: bool,
    dtype=torch.bfloat16,
):
    weights = {
        "cos": torch.randn(cache_len, LLAMA1B_HEAD_DIM, dtype=dtype, device="cuda"),
        "sin": torch.randn(cache_len, LLAMA1B_HEAD_DIM, dtype=dtype, device="cuda"),
    }
    for layer_idx in range(num_layers):
        pfx = f"layer.{layer_idx}"
        weights[f"{pfx}.attn_norm"] = torch.ones(LLAMA1B_HIDDEN, dtype=dtype, device="cuda")
        weights[f"{pfx}.mlp_norm"] = torch.ones(LLAMA1B_HIDDEN, dtype=dtype, device="cuda")
        weights[f"{pfx}.W_q"] = _randn((LLAMA1B_Q_DIM, LLAMA1B_HIDDEN), dtype=dtype)
        weights[f"{pfx}.W_k"] = _randn((LLAMA1B_KV_DIM, LLAMA1B_HIDDEN), dtype=dtype)
        weights[f"{pfx}.W_v"] = _randn((LLAMA1B_KV_DIM, LLAMA1B_HIDDEN), dtype=dtype)
        weights[f"{pfx}.W_o"] = _randn((LLAMA1B_HIDDEN, LLAMA1B_Q_DIM), dtype=dtype)
        weights[f"{pfx}.W_gate"] = _randn((LLAMA1B_INTERMEDIATE, LLAMA1B_HIDDEN), dtype=dtype)
        weights[f"{pfx}.W_up"] = _randn((LLAMA1B_INTERMEDIATE, LLAMA1B_HIDDEN), dtype=dtype)
        weights[f"{pfx}.W_down"] = _randn((LLAMA1B_HIDDEN, LLAMA1B_INTERMEDIATE), dtype=dtype)

    num_q_heads = LLAMA1B_Q_DIM // LLAMA1B_HEAD_DIM
    num_kv_heads = LLAMA1B_KV_DIM // LLAMA1B_HEAD_DIM
    x_buffers = [
        torch.randn(batch, 1, LLAMA1B_HIDDEN, dtype=dtype, device="cuda"),
        torch.empty(batch, 1, LLAMA1B_HIDDEN, dtype=dtype, device="cuda"),
    ]
    residual_buffers = [
        torch.zeros(batch, 1, LLAMA1B_HIDDEN, dtype=dtype, device="cuda"),
        torch.empty(batch, 1, LLAMA1B_HIDDEN, dtype=dtype, device="cuda"),
    ]
    k_cache = torch.randn(
        num_layers,
        batch,
        cache_len,
        num_kv_heads,
        LLAMA1B_HEAD_DIM,
        dtype=dtype,
        device="cuda",
    )
    v_cache = torch.randn_like(k_cache)
    q_buf = torch.empty(batch, 1, num_q_heads * LLAMA1B_HEAD_DIM, dtype=dtype, device="cuda")
    attn_out_buf = torch.empty_like(q_buf)
    mlp_h_buf = torch.empty(batch, 1, LLAMA1B_INTERMEDIATE, dtype=dtype, device="cuda")

    final_norm = lm_head = logits = None
    if include_final:
        final_norm = torch.ones(LLAMA1B_HIDDEN, dtype=dtype, device="cuda")
        lm_head = _randn((LLAMA1B_VOCAB, LLAMA1B_HIDDEN), dtype=dtype)
        logits = torch.empty(batch, 1, LLAMA1B_VOCAB, dtype=dtype, device="cuda")

    return {
        "weights": weights,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "x_buffers": x_buffers,
        "residual_buffers": residual_buffers,
        "q_buf": q_buf,
        "attn_out_buf": attn_out_buf,
        "mlp_h_buf": mlp_h_buf,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "logits": logits,
    }


def build_kernel(
    *,
    batch: int,
    cache_len: int,
    num_layers: int,
    include_final: bool,
    matvec_block: int,
    final_matvec_block: int,
    fa_num_splits: int = 0,
    split_upgate: bool = True,
    threads_per_block: int = LLAMA1B_SM120_THREADS_PER_BLOCK,
    num_sms: int | None = None,
    num_pages: int | None = None,
    mma_reg_count: int = 96,
    page_size: int | None = None,
):
    if cache_len < 1:
        raise ValueError("cache_len must be at least 1")
    if threads_per_block < LLAMA1B_SM120_THREADS_PER_BLOCK:
        raise ValueError(
            f"threads_per_block={threads_per_block} is unsafe for Llama-1B staged matvecs; "
            f"use at least {LLAMA1B_SM120_THREADS_PER_BLOCK}"
        )
    if num_sms is not None:
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        if num_sms < 1 or num_sms > sm_count:
            raise ValueError(f"num_sms must be in [1, {sm_count}], got {num_sms}")
    state = make_llama1b_decode_state(
        batch=batch,
        cache_len=cache_len,
        num_layers=num_layers,
        include_final=include_final,
    )
    if fa_num_splits < 0:
        fa_num_splits = 8 if cache_len < 384 else min(64, max(16, cache_len // 8))
    schedule = schedule_decode_model_sm120(
        batch=batch,
        cache_pos=cache_len - 1,
        num_layers=num_layers,
        weights=state["weights"],
        k_cache=state["k_cache"],
        v_cache=state["v_cache"],
        x_buffers=state["x_buffers"],
        residual_buffers=state["residual_buffers"],
        q_buf=state["q_buf"],
        attn_out_buf=state["attn_out_buf"],
        mlp_h_buf=state["mlp_h_buf"],
        final_norm=state["final_norm"],
        lm_head=state["lm_head"],
        logits=state["logits"],
        matvec_block=matvec_block,
        final_matvec_block=final_matvec_block,
        fa_num_splits=fa_num_splits,
        split_upgate=split_upgate,
        page_size=page_size or 49152,
    )
    page_size = max(op.static_dims.get("page_size", page_size or 49152) for op in schedule.ops)
    kernel = Megakernel(
        schedule.ops,
        config=MegakernelConfig(
            threads_per_block=threads_per_block,
            page_size=page_size,
            loader_idle_sleep_ns=0,
            mma_reg_count=mma_reg_count,
            num_sms=num_sms,
            num_pages=num_pages,
        ),
        scheduler=OverlapTileScheduler(),
    )
    keep_alive = [state, schedule.keep_alive]
    return kernel, keep_alive


def time_kernel(kernel: Megakernel, *, warmup: int, rep: int):
    kernel.compile()
    for _ in range(warmup):
        kernel.run(validate=False)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        kernel.run(sync=False, validate=False)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--cache-len", type=int, default=128, help="Total KV length attended by the decode token")
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--matvec-block", type=int, default=LLAMA1B_SM120_MATVEC_BLOCK)
    parser.add_argument("--final-matvec-block", type=int, default=LLAMA1B_SM120_FINAL_MATVEC_BLOCK)
    parser.add_argument("--fa-num-splits", type=int, default=-1, help="Attention KV splits; -1 uses an automatic cache-length heuristic")
    parser.add_argument("--split-upgate", dest="split_upgate", action="store_true", help="Use per-down-chunk up/gate slices")
    parser.add_argument("--no-split-upgate", dest="split_upgate", action="store_false", help="Use one full-width up/gate op")
    parser.set_defaults(split_upgate=True)
    parser.add_argument("--no-final", action="store_true")
    parser.add_argument("--threads-per-block", type=int, default=LLAMA1B_SM120_THREADS_PER_BLOCK)
    parser.add_argument("--num-sms", type=int, default=0, help="Persistent CTAs to launch; 0 uses the device/default cap")
    parser.add_argument("--num-pages", type=int, default=0, help="Instruction page-ring pages; 0 uses auto")
    parser.add_argument("--mma-reg-count", type=int, default=96)
    parser.add_argument("--page-size", type=int, default=49152)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    include_final = not args.no_final
    resolved_fa_num_splits = args.fa_num_splits
    if resolved_fa_num_splits < 0:
        resolved_fa_num_splits = 8 if args.cache_len < 384 else min(64, max(16, args.cache_len // 8))
    kernel, keep_alive = build_kernel(
        batch=args.batch,
        cache_len=args.cache_len,
        num_layers=args.layers,
        include_final=include_final,
        matvec_block=args.matvec_block,
        final_matvec_block=args.final_matvec_block,
        fa_num_splits=resolved_fa_num_splits,
        split_upgate=args.split_upgate,
        threads_per_block=args.threads_per_block,
        num_sms=args.num_sms or None,
        num_pages=args.num_pages or None,
        mma_reg_count=args.mma_reg_count,
        page_size=args.page_size,
    )
    keep_alive.append(kernel)
    ms = time_kernel(kernel, warmup=args.warmup, rep=args.rep)
    print(
        "llama1b_sm120_decode "
        f"batch={args.batch} cache_len={args.cache_len} layers={args.layers} "
        f"final={include_final} matvec_block={args.matvec_block} final_matvec_block={args.final_matvec_block} "
        f"range=auto "
        f"fa_num_splits={resolved_fa_num_splits} split_upgate={args.split_upgate} scheduler=overlap "
        f"threads_per_block={args.threads_per_block} num_sms={args.num_sms or 'auto'} num_pages={kernel._layout.num_pages} "
        f"num_slots={kernel._layout.num_slots} mma_reg_count={args.mma_reg_count} "
        f"ops={len(kernel.ops)} tiles={kernel.total_tiles} smem_kb={kernel.smem_size // 1024} "
        f"ms={ms:.4f} tok_s={args.batch * 1000.0 / ms:.2f}"
    )


if __name__ == "__main__":
    main()
