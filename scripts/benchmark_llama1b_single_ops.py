#!/usr/bin/env python
"""Benchmark isolated Llama-1B SM120 decode matvec ops."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from machete.kernels.llama1b import (  # noqa: E402
    LLAMA1B_HIDDEN,
    LLAMA1B_INTERMEDIATE,
    LLAMA1B_HEAD_DIM,
    LLAMA1B_KV_DIM,
    LLAMA1B_Q_DIM,
    LLAMA1B_SM120_QKV_HEAD_BLOCK,
    LLAMA1B_SM120_MATVEC_BLOCK,
    LLAMA1B_SM120_THREADS_PER_BLOCK,
    LLAMA1B_VOCAB,
    Llama1BDownMatvecSm120Op,
    Llama1BFinalRmsLmHeadSingleStageSm120Op,
    Llama1BFinalRmsLmHeadSm120Op,
    Llama1BMatvecResidualSm120Op,
    Llama1BRmsQKVCacheSm120Op,
    Llama1BRmsUpGateSiluSm120Op,
    schedule_llama1b_decode_attention_sm120,
)
from machete.megakernel import Megakernel, MegakernelConfig, OverlapTileScheduler  # noqa: E402


def _randn(shape, *, dtype=torch.bfloat16, scale=0.02):
    return torch.randn(*shape, dtype=dtype, device="cuda") * scale


def _run(name: str, ops: list, *, warmup: int, rep: int):
    page_size = max(op.static_dims.get("page_size", 49152) for op in ops)
    kernel = Megakernel(
        ops,
        config=MegakernelConfig(
            threads_per_block=LLAMA1B_SM120_THREADS_PER_BLOCK,
            page_size=page_size,
            loader_idle_sleep_ns=0,
            mma_reg_count=96,
        ),
        scheduler=OverlapTileScheduler(),
    )
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
    ms = start.elapsed_time(end) / rep
    print(
        f"{name:36s} ops={len(ops):2d} tiles={kernel.total_tiles:6d} "
        f"smem_kb={kernel.smem_size // 1024:3d} ms={ms:.4f} us={ms * 1000.0:.1f}"
    )
    return kernel


def bench_o_proj(*, warmup: int, rep: int):
    a = _randn((1, 1, LLAMA1B_HIDDEN))
    weight = _randn((LLAMA1B_HIDDEN, LLAMA1B_HIDDEN))
    residual = _randn((1, 1, LLAMA1B_HIDDEN))
    out = torch.empty_like(residual)
    _run(
        f"staged o_proj residual O={LLAMA1B_SM120_MATVEC_BLOCK}",
        Llama1BMatvecResidualSm120Op.schedule(
            a=a,
            weight=weight,
            residual_in=residual,
            residual_out=out,
            tile_sizes={"S": 1, "O": LLAMA1B_SM120_MATVEC_BLOCK},
        ),
        warmup=warmup,
        rep=rep,
    )


def bench_upgate(*, warmup: int, rep: int):
    x = _randn((1, 1, LLAMA1B_HIDDEN))
    norm = torch.ones(LLAMA1B_HIDDEN, dtype=torch.bfloat16, device="cuda")
    up = _randn((LLAMA1B_INTERMEDIATE, LLAMA1B_HIDDEN))
    gate = _randn((LLAMA1B_INTERMEDIATE, LLAMA1B_HIDDEN))
    y = torch.empty((1, 1, LLAMA1B_INTERMEDIATE), dtype=torch.bfloat16, device="cuda")
    _run(
        f"staged rms upgate O={LLAMA1B_SM120_MATVEC_BLOCK}",
        Llama1BRmsUpGateSiluSm120Op.schedule(
            x=x,
            norm_weight=norm,
            up_weight=up,
            gate_weight=gate,
            y=y,
            tile_sizes={"S": 1, "O": LLAMA1B_SM120_MATVEC_BLOCK},
        ),
        warmup=warmup,
        rep=rep,
    )


def bench_down(*, warmup: int, rep: int):
    a = _randn((1, 1, LLAMA1B_HIDDEN))
    weight = _randn((LLAMA1B_HIDDEN, LLAMA1B_HIDDEN))
    y = torch.empty_like(a)
    residual = _randn((1, 1, LLAMA1B_HIDDEN))
    out = torch.empty_like(a)
    _run(
        f"staged down first O={LLAMA1B_SM120_MATVEC_BLOCK}",
        Llama1BDownMatvecSm120Op.schedule(
            a=a,
            weight=weight,
            y=y,
            tile_sizes={"S": 1, "O": LLAMA1B_SM120_MATVEC_BLOCK},
        ),
        warmup=warmup,
        rep=rep,
    )
    _run(
        f"staged down residual O={LLAMA1B_SM120_MATVEC_BLOCK}",
        Llama1BMatvecResidualSm120Op.schedule(
            a=a,
            weight=weight,
            residual_in=residual,
            residual_out=out,
            tile_sizes={"S": 1, "O": LLAMA1B_SM120_MATVEC_BLOCK},
        ),
        warmup=warmup,
        rep=rep,
    )


def bench_qkv(*, warmup: int, rep: int):
    num_q_heads = LLAMA1B_Q_DIM // LLAMA1B_HEAD_DIM
    num_kv_heads = LLAMA1B_KV_DIM // LLAMA1B_HEAD_DIM
    kv_group_size = num_q_heads // num_kv_heads
    x = _randn((1, 1, LLAMA1B_HIDDEN))
    residual = _randn((1, 1, LLAMA1B_HIDDEN))
    residual_out = torch.empty_like(x)
    norm = torch.ones(LLAMA1B_HIDDEN, dtype=torch.bfloat16, device="cuda")
    q = torch.empty((1, 1, LLAMA1B_Q_DIM), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.empty((1, 128, num_kv_heads, LLAMA1B_HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    v_cache = torch.empty_like(k_cache)
    q_w = _randn((LLAMA1B_Q_DIM, LLAMA1B_HIDDEN))
    k_w = _randn((LLAMA1B_KV_DIM, LLAMA1B_HIDDEN))
    v_w = _randn((LLAMA1B_KV_DIM, LLAMA1B_HIDDEN))
    qkv_w = torch.cat(
        [
            part
            for kv_head in range(num_kv_heads)
            for part in (
                q_w[kv_head * kv_group_size * LLAMA1B_HEAD_DIM : (kv_head + 1) * kv_group_size * LLAMA1B_HEAD_DIM],
                k_w[kv_head * LLAMA1B_HEAD_DIM : (kv_head + 1) * LLAMA1B_HEAD_DIM],
                v_w[kv_head * LLAMA1B_HEAD_DIM : (kv_head + 1) * LLAMA1B_HEAD_DIM],
            )
        ],
        dim=0,
    )
    cos = _randn((1, LLAMA1B_HEAD_DIM))
    sin = _randn((1, LLAMA1B_HEAD_DIM))
    _run(
        f"staged rms qkv O={LLAMA1B_SM120_QKV_HEAD_BLOCK}",
        Llama1BRmsQKVCacheSm120Op.schedule(
            x=x,
            residual_in=residual,
            norm_weight=norm,
            weight=qkv_w,
            cos=cos,
            sin=sin,
            residual_out=residual_out,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_pos=127,
            tile_sizes={"S": 1, "O": LLAMA1B_SM120_QKV_HEAD_BLOCK},
            kv_group_size=kv_group_size,
        ),
        warmup=warmup,
        rep=rep,
    )


def bench_attention(*, warmup: int, rep: int):
    num_q_heads = LLAMA1B_Q_DIM // LLAMA1B_HEAD_DIM
    num_kv_heads = LLAMA1B_KV_DIM // LLAMA1B_HEAD_DIM
    kv_group_size = num_q_heads // num_kv_heads
    q = _randn((1, 1, num_q_heads, LLAMA1B_HEAD_DIM))
    k = _randn((1, 128, num_kv_heads, LLAMA1B_HEAD_DIM))
    v = _randn((1, 128, num_kv_heads, LLAMA1B_HEAD_DIM))
    o = torch.empty_like(q)
    ops, keep_alive = schedule_llama1b_decode_attention_sm120(
        q=q,
        k=k,
        v=v,
        o=o,
        kv_group_size=kv_group_size,
        num_splits=12,
    )
    keep_alive.append(o)
    _run("decode attention splits=12", ops, warmup=warmup, rep=rep)


def bench_lm_head(*, warmup: int, rep: int):
    x = _randn((1, 1, LLAMA1B_HIDDEN))
    norm = torch.ones(LLAMA1B_HIDDEN, dtype=torch.bfloat16, device="cuda")
    weight = _randn((LLAMA1B_VOCAB, LLAMA1B_HIDDEN))
    logits = torch.empty((1, 1, LLAMA1B_VOCAB), dtype=torch.bfloat16, device="cuda")
    for block in (12, 16, 24):
        cls = Llama1BFinalRmsLmHeadSingleStageSm120Op if block > 12 else Llama1BFinalRmsLmHeadSm120Op
        _run(
            f"staged final lm_head O={block}",
            cls.schedule(
                x=x,
                norm_weight=norm,
                weight=weight,
                logits=logits,
                tile_sizes={"S": 1, "O": block},
            ),
            warmup=warmup,
            rep=rep,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--op",
        choices=("o_proj", "upgate", "down", "qkv", "attention", "lm_head", "all"),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.op in ("o_proj", "all"):
        bench_o_proj(warmup=args.warmup, rep=args.rep)
    if args.op in ("upgate", "all"):
        bench_upgate(warmup=args.warmup, rep=args.rep)
    if args.op in ("down", "all"):
        bench_down(warmup=args.warmup, rep=args.rep)
    if args.op in ("qkv", "all"):
        bench_qkv(warmup=args.warmup, rep=args.rep)
    if args.op in ("attention", "all"):
        bench_attention(warmup=args.warmup, rep=args.rep)
    if args.op in ("lm_head", "all"):
        bench_lm_head(warmup=args.warmup, rep=max(5, args.rep // 2))


if __name__ == "__main__":
    main()
