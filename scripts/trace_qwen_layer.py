#!/usr/bin/env python
"""Write cutedsl-trace files for Qwen 3.5 layer forward/backward."""

import argparse
import contextlib
import io
import os

import torch

from benchmarks.kernels.benchmark_qwen3_5_layer import (
    D2,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_DIM,
    Q_DIM,
    megakernel_forward_build,
    megakernel_layer_bwd_build,
)
from machete.megakernel import OverlapTileScheduler


def _alloc_qwen_layer(batch: int, seq_len: int):
    torch.manual_seed(2026)
    dtype = torch.bfloat16
    device = "cuda"
    return (
        batch,
        seq_len,
        torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device),
        torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device),
        torch.randn(HIDDEN, dtype=dtype, device=device),
        torch.randn(Q_DIM, HIDDEN, dtype=dtype, device=device) * 0.02,
        torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02,
        torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02,
        torch.ones(HEAD_DIM, dtype=dtype, device=device),
        torch.ones(HEAD_DIM, dtype=dtype, device=device),
        torch.randn(seq_len, D2, dtype=dtype, device=device),
        torch.randn(seq_len, D2, dtype=dtype, device=device),
        torch.randn(HIDDEN, Q_DIM, dtype=dtype, device=device) * 0.02,
        torch.randn(HIDDEN, dtype=dtype, device=device),
        torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=dtype, device=device) * 0.02,
        torch.randn(HIDDEN, INTERMEDIATE, dtype=dtype, device=device) * 0.02,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fwd", "bwd"], default="fwd")
    parser.add_argument("--scheduler", choices=["default", "overlap"], default="overlap")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=32768)
    parser.add_argument("--output", default="traces/qwen_layer.nanotrace")
    parser.add_argument("--perfetto-output", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    scheduler = OverlapTileScheduler() if args.scheduler == "overlap" else None
    qwen_args = _alloc_qwen_layer(args.batch, args.seq_len)
    build = megakernel_forward_build if args.mode == "fwd" else megakernel_layer_bwd_build

    with contextlib.redirect_stdout(io.StringIO()):
        result = build(
            *qwen_args,
            page_size=args.page_size,
            scheduler=scheduler,
            tracing=True,
        )
    spec = result[0]
    kernel = spec._keep_alive[0]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    kernel.write_trace(args.output)
    print(f"wrote {args.output}")

    if args.perfetto_output:
        os.makedirs(os.path.dirname(args.perfetto_output) or ".", exist_ok=True)
        kernel.write_trace_perfetto(args.perfetto_output)
        print(f"wrote {args.perfetto_output}")


if __name__ == "__main__":
    main()
