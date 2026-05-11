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

from machete.kernels.decode_matvec.sm120 import (  # noqa: E402
    FinalRmsLmHeadParallelSm120Op,
    FinalRmsLmHeadSm120Op,
    MatvecResidualParallelSm120Op,
    MatvecResidualSm120Op,
    RmsGateUpSiluSm120Op,
)
from machete.kernels.llama1b import (  # noqa: E402
    LLAMA1B_HIDDEN,
    LLAMA1B_INTERMEDIATE,
    LLAMA1B_SM120_THREADS_PER_BLOCK,
    LLAMA1B_VOCAB,
    Llama1BFinalRmsLmHeadSingleStageSm120Op,
    Llama1BFinalRmsLmHeadSm120Op,
    Llama1BMatvecResidualSm120Op,
    Llama1BMatvecResidualVec4Sm120Op,
    Llama1BRmsUpGateSiluSm120Op,
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
        "staged o_proj residual O=10",
        Llama1BMatvecResidualSm120Op.schedule(
            a=a,
            weight=weight,
            residual_in=residual,
            residual_out=out,
            tile_sizes={"S": 1, "O": 10},
        ),
        warmup=warmup,
        rep=rep,
    )
    _run(
        "staged vec4 o_proj O=10",
        Llama1BMatvecResidualVec4Sm120Op.schedule(
            a=a,
            weight=weight,
            residual_in=residual,
            residual_out=out,
            tile_sizes={"S": 1, "O": 10},
        ),
        warmup=warmup,
        rep=rep,
    )
    _run(
        "direct o_proj residual O=16",
        MatvecResidualSm120Op.schedule(
            a=a,
            weight=weight,
            residual_in=residual,
            residual_out=out,
            tile_sizes={"S": 1, "O": 16},
        ),
        warmup=warmup,
        rep=rep,
    )
    _run(
        "direct parallel o_proj O=16",
        MatvecResidualParallelSm120Op.schedule(
            a=a,
            weight=weight,
            residual_in=residual,
            residual_out=out,
            tile_sizes={"S": 1, "O": 16},
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
        "staged rms upgate O=10",
        Llama1BRmsUpGateSiluSm120Op.schedule(
            x=x,
            norm_weight=norm,
            up_weight=up,
            gate_weight=gate,
            y=y,
            tile_sizes={"S": 1, "O": 10},
        ),
        warmup=warmup,
        rep=rep,
    )
    _run(
        "direct rms upgate D=16",
        RmsGateUpSiluSm120Op.schedule(
            x=x,
            norm_weight=norm,
            up_weight=up,
            gate_weight=gate,
            y=y,
            tile_sizes={"S": 1, "D": 16},
        ),
        warmup=warmup,
        rep=rep,
    )


def bench_lm_head(*, warmup: int, rep: int):
    x = _randn((1, 1, LLAMA1B_HIDDEN))
    norm = torch.ones(LLAMA1B_HIDDEN, dtype=torch.bfloat16, device="cuda")
    weight = _randn((LLAMA1B_VOCAB, LLAMA1B_HIDDEN))
    logits = torch.empty((1, 1, LLAMA1B_VOCAB), dtype=torch.bfloat16, device="cuda")
    for block in (10, 16, 22, 24):
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
    _run(
        "direct final lm_head V=16",
        FinalRmsLmHeadSm120Op.schedule(
            x=x,
            norm_weight=norm,
            weight=weight,
            logits=logits,
            tile_sizes={"S": 1, "V": 16},
        ),
        warmup=warmup,
        rep=rep,
    )
    _run(
        "direct parallel lm_head V=16",
        FinalRmsLmHeadParallelSm120Op.schedule(
            x=x,
            norm_weight=norm,
            weight=weight,
            logits=logits,
            tile_sizes={"S": 1, "V": 16},
        ),
        warmup=warmup,
        rep=rep,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=("o_proj", "upgate", "lm_head", "all"), default="all")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.op in ("o_proj", "all"):
        bench_o_proj(warmup=args.warmup, rep=args.rep)
    if args.op in ("upgate", "all"):
        bench_upgate(warmup=args.warmup, rep=args.rep)
    if args.op in ("lm_head", "all"):
        bench_lm_head(warmup=args.warmup, rep=max(5, args.rep // 2))


if __name__ == "__main__":
    main()
