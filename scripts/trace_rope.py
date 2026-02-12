#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Run RoPE via the megakernel with tracing enabled.

Produces a .nanotrace file that can be opened in a trace visualizer to
inspect per-SM tile timelines.

Usage:
    python scripts/trace_rope.py
    python scripts/trace_rope.py --batch 4 --seq-len 512 --n-heads 32 --head-dim 128
    python scripts/trace_rope.py -o my_trace.nanotrace
"""

import argparse

import torch

from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.rope import RopeOp
from machete.kernels.rope.ref import rope_pytorch


def main():
    parser = argparse.ArgumentParser(description="Trace RoPE megakernel execution")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("-o", "--output", type=str, default="traces/rope.nanotrace")
    args = parser.parse_args()

    b, s, h, d = args.batch, args.seq_len, args.n_heads, args.head_dim
    print(f"Shape: batch={b}, seq_len={s}, n_heads={h}, head_dim={d}")

    # Create inputs
    torch.manual_seed(42)
    q = torch.randn(b, s, h, d, dtype=torch.float32, device="cuda")
    cos = torch.randn(s, d // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(s, d // 2, dtype=torch.float32, device="cuda")

    # Reference output for correctness check
    q_ref = rope_pytorch(q, cos, sin)

    # Schedule and run with tracing
    q_flat = q.view(b * s, h, d)
    ops = [RopeOp.schedule(q=q_flat, cos=cos, sin=sin, tile_sizes={"M": 1})]
    config = MegakernelConfig(tracing=True)
    kernel = Megakernel(ops, config=config)

    print(kernel)
    print()

    kernel.run()

    # Correctness check
    max_diff = (q.view(b, s, h, d) - q_ref).abs().max().item()
    print(f"Max diff vs PyTorch reference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Correctness check failed: max_diff={max_diff}"

    # Write trace
    kernel.write_trace(args.output)
    print(f"Trace written to: {args.output}")


if __name__ == "__main__":
    main()
