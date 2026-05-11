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
    schedule_llama1b_decode_attention_sm120,
)
from machete.kernels.decode_matvec.sm120 import (
    MatvecResidualSm120Op,
    MatvecResidualParallelSm120Op,
    MatvecParallelSm120Op,
    MatvecSm120Op,
    FinalRmsLmHeadParallelSm120Op,
    RmsGateUpSiluSm120Op,
    RmsKMatvecRopeCacheSm120Op,
    RmsQMatvecRopeSm120Op,
    RmsVMatvecCacheSm120Op,
    schedule_final_sm120 as schedule_direct_final_sm120,
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
    kernel_path: str,
    matvec_block: int,
    final_matvec_block: int,
    fa_num_splits: int = 0,
    split_upgate: bool = False,
    threads_per_block: int = LLAMA1B_SM120_THREADS_PER_BLOCK,
    num_sms: int | None = None,
    mma_reg_count: int = 96,
):
    if cache_len < 1:
        raise ValueError("cache_len must be at least 1")
    if threads_per_block < LLAMA1B_SM120_THREADS_PER_BLOCK:
        raise ValueError(
            f"threads_per_block={threads_per_block} is unsafe for Llama-1B staged matvecs; "
            f"use at least {LLAMA1B_SM120_THREADS_PER_BLOCK}"
        )
    state = make_llama1b_decode_state(
        batch=batch,
        cache_len=cache_len,
        num_layers=num_layers,
        include_final=include_final,
    )
    if kernel_path == "staged":
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
        )
    elif kernel_path == "direct":
        ops = []
        keep_alive = []
        cur = 0
        for layer_idx in range(num_layers):
            nxt = 1 - cur
            pfx = f"layer.{layer_idx}"
            cos = state["weights"]["cos"][cache_len - 1 : cache_len]
            sin = state["weights"]["sin"][cache_len - 1 : cache_len]
            num_q_heads = LLAMA1B_Q_DIM // LLAMA1B_HEAD_DIM
            num_kv_heads = LLAMA1B_KV_DIM // LLAMA1B_HEAD_DIM
            kv_group_size = num_q_heads // num_kv_heads
            q_4d = state["q_buf"].view(batch, 1, num_q_heads, LLAMA1B_HEAD_DIM)
            k_window = state["k_cache"][layer_idx][:, :cache_len]
            v_window = state["v_cache"][layer_idx][:, :cache_len]
            o_4d = state["attn_out_buf"].view(batch, 1, num_q_heads, LLAMA1B_HEAD_DIM)

            ops += RmsQMatvecRopeSm120Op.schedule(
                x=state["x_buffers"][cur],
                residual_in=state["residual_buffers"][cur],
                norm_weight=state["weights"][f"{pfx}.attn_norm"],
                weight=state["weights"][f"{pfx}.W_q"],
                cos=cos,
                sin=sin,
                residual_out=state["residual_buffers"][nxt],
                q=state["q_buf"],
                tile_sizes={"S": 1, "O": matvec_block},
            )
            ops += RmsKMatvecRopeCacheSm120Op.schedule(
                x=state["x_buffers"][cur],
                residual_in=state["residual_buffers"][cur],
                norm_weight=state["weights"][f"{pfx}.attn_norm"],
                weight=state["weights"][f"{pfx}.W_k"],
                cos=cos,
                sin=sin,
                dst_cache=k_window,
                cache_pos=cache_len - 1,
                tile_sizes={"S": 1, "O": matvec_block},
            )
            ops += RmsVMatvecCacheSm120Op.schedule(
                x=state["x_buffers"][cur],
                residual_in=state["residual_buffers"][cur],
                norm_weight=state["weights"][f"{pfx}.attn_norm"],
                weight=state["weights"][f"{pfx}.W_v"],
                cos=cos,
                sin=sin,
                dst_cache=v_window,
                cache_pos=cache_len - 1,
                tile_sizes={"S": 1, "O": matvec_block},
            )

            attn_ops, attn_keep = schedule_llama1b_decode_attention_sm120(
                q=q_4d,
                k=k_window,
                v=v_window,
                o=o_4d,
                kv_group_size=kv_group_size,
                num_splits=fa_num_splits,
            )
            ops += attn_ops
            ops += MatvecResidualParallelSm120Op.schedule(
                a=state["attn_out_buf"],
                weight=state["weights"][f"{pfx}.W_o"],
                residual_in=state["residual_buffers"][nxt],
                residual_out=state["residual_buffers"][nxt],
                tile_sizes={"S": 1, "O": matvec_block},
            )
            ops += RmsGateUpSiluSm120Op.schedule(
                x=state["residual_buffers"][nxt],
                norm_weight=state["weights"][f"{pfx}.mlp_norm"],
                gate_weight=state["weights"][f"{pfx}.W_gate"],
                up_weight=state["weights"][f"{pfx}.W_up"],
                y=state["mlp_h_buf"],
                tile_sizes={"S": 1, "D": matvec_block},
            )
            down_keep = []
            for reduction_block in range(LLAMA1B_INTERMEDIATE // LLAMA1B_HIDDEN):
                start = reduction_block * LLAMA1B_HIDDEN
                stop = start + LLAMA1B_HIDDEN
                a_block = state["mlp_h_buf"][:, :, start:stop]
                w_block = state["weights"][f"{pfx}.W_down"][:, start:stop]
                down_keep += [a_block, w_block]
                if reduction_block == 0:
                    ops += MatvecParallelSm120Op.schedule(
                        a=a_block,
                        weight=w_block,
                        y=state["x_buffers"][nxt],
                        tile_sizes={"S": 1, "O": matvec_block},
                    )
                else:
                    ops += MatvecResidualParallelSm120Op.schedule(
                        a=a_block,
                        weight=w_block,
                        residual_in=state["x_buffers"][nxt],
                        residual_out=state["x_buffers"][nxt],
                        tile_sizes={"S": 1, "O": matvec_block},
                    )
            keep_alive += [cos, sin, q_4d, k_window, v_window, o_4d, *attn_keep, *down_keep]
            cur = nxt

        if include_final:
            ops += schedule_direct_final_sm120(
                x=state["x_buffers"][cur],
                residual_in=state["residual_buffers"][cur],
                residual_out=state["residual_buffers"][cur],
                final_norm=state["final_norm"],
                lm_head=None,
                logits=None,
                seq_len=1,
            )
            ops += FinalRmsLmHeadParallelSm120Op.schedule(
                x=state["residual_buffers"][cur],
                norm_weight=state["final_norm"],
                weight=state["lm_head"],
                logits=state["logits"],
                tile_sizes={"S": 1, "V": 16},
            )

        class _Schedule:
            pass

        schedule = _Schedule()
        schedule.ops = ops
        schedule.keep_alive = keep_alive
    else:
        raise ValueError(f"unknown kernel_path={kernel_path!r}")
    page_size = max(op.static_dims.get("page_size", 49152) for op in schedule.ops)
    kernel = Megakernel(
        schedule.ops,
        config=MegakernelConfig(
            threads_per_block=threads_per_block,
            page_size=page_size,
            loader_idle_sleep_ns=0,
            mma_reg_count=mma_reg_count,
            num_sms=num_sms,
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
    parser.add_argument(
        "--kernel-path",
        choices=("staged", "direct"),
        default="staged",
        help="staged uses Llama-1B specialized staged kernels; direct uses generic SM120 global-memory matvec kernels",
    )
    parser.add_argument("--matvec-block", type=int, default=LLAMA1B_SM120_MATVEC_BLOCK)
    parser.add_argument("--final-matvec-block", type=int, default=LLAMA1B_SM120_FINAL_MATVEC_BLOCK)
    parser.add_argument("--fa-num-splits", type=int, default=0, help="Attention KV splits; use -1 for an automatic cache-length heuristic")
    parser.add_argument("--split-upgate", dest="split_upgate", action="store_true", help="Use per-down-chunk up/gate slices")
    parser.add_argument("--no-split-upgate", dest="split_upgate", action="store_false", help="Use one full-width up/gate op")
    parser.set_defaults(split_upgate=False)
    parser.add_argument("--no-final", action="store_true")
    parser.add_argument("--threads-per-block", type=int, default=LLAMA1B_SM120_THREADS_PER_BLOCK)
    parser.add_argument("--num-sms", type=int, default=0, help="Persistent CTAs to launch; 0 uses the device/default cap")
    parser.add_argument("--mma-reg-count", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    include_final = not args.no_final
    kernel, keep_alive = build_kernel(
        batch=args.batch,
        cache_len=args.cache_len,
        num_layers=args.layers,
        include_final=include_final,
        kernel_path=args.kernel_path,
        matvec_block=args.matvec_block,
        final_matvec_block=args.final_matvec_block,
        fa_num_splits=args.fa_num_splits,
        split_upgate=args.split_upgate,
        threads_per_block=args.threads_per_block,
        num_sms=args.num_sms or None,
        mma_reg_count=args.mma_reg_count,
    )
    keep_alive.append(kernel)
    ms = time_kernel(kernel, warmup=args.warmup, rep=args.rep)
    print(
        "llama1b_sm120_decode "
        f"batch={args.batch} cache_len={args.cache_len} layers={args.layers} "
        f"final={include_final} kernel_path={args.kernel_path} "
        f"matvec_block={args.matvec_block} final_matvec_block={args.final_matvec_block} "
        f"fa_num_splits={args.fa_num_splits} split_upgate={args.split_upgate} scheduler=overlap "
        f"threads_per_block={args.threads_per_block} num_sms={args.num_sms or 'auto'} mma_reg_count={args.mma_reg_count} "
        f"ops={len(kernel.ops)} tiles={kernel.total_tiles} smem_kb={kernel.smem_size // 1024} "
        f"ms={ms:.4f} tok_s={args.batch * 1000.0 / ms:.2f}"
    )


if __name__ == "__main__":
    main()
