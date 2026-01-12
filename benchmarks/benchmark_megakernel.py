# Copyright (c) 2025, Machete Authors
import torch
from typing import Dict, Tuple, Callable

from machete.megakernel.core import Megakernel
from machete.kernels.gated_linear.sm80 import GatedLinearSM80, GatedLinear as GatedLinearOp
from machete.kernels.rope.sm80 import RopeSM80
from quack.cute_dsl_utils import torch2cute_dtype_map


def do_bench(fn, warmup=25, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(rep):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / rep


def main():
    device = "cuda"
    dtype = torch.float16
    cute_dtype = torch2cute_dtype_map[dtype]

    # --- Config: RoPE + GatedLinear ---
    configs = {}
    for b_sz, s_sz, h_sz, d_sz in [(1, 4096, 32, 128), (2, 8192, 32, 128)]:
        q_tensor = torch.randn(b_sz, s_sz, h_sz, d_sz, device=device, dtype=dtype)
        cos_tensor = torch.randn(s_sz, d_sz, device=device, dtype=dtype)
        sin_tensor = torch.randn(s_sz, d_sz, device=device, dtype=dtype)
        gate_tensor = torch.randn(b_sz * s_sz, h_sz * d_sz, device=device, dtype=dtype)
        out_tensor = torch.empty(b_sz * s_sz, h_sz * d_sz, device=device, dtype=dtype)
        barrier_tensor = torch.zeros(1, device=device, dtype=torch.int32)
        configs[f"B={b_sz} S={s_sz} H={h_sz} D={d_sz}"] = (
            q_tensor,
            cos_tensor,
            sin_tensor,
            gate_tensor,
            out_tensor,
            barrier_tensor,
            s_sz,
            h_sz,
            d_sz,
        )

    rope_impl = RopeSM80(dtype, 128)
    gl_impl = GatedLinearSM80(dtype, "silu")

    print(f"\n{'=' * 20} RoPE + GatedLinear Fusion {'=' * 20}")
    print(f"{'Config':<25} | {'Provider':<15} | {'Time (ms)':<10}")
    print("-" * 60)

    for config_name, args in configs.items():
        q, cos, sin, gate, out, barrier, s, h, d = args

        # Sequential
        rope_op = RopeSM80(dtype, d)

        def run_seq(q=q, cos=cos, sin=sin, gate=gate, out=out):
            rope_op(q, cos, sin)
            GatedLinearOp.apply(q.view(-1, h * d), gate, "silu")

        ms_seq = do_bench(run_seq)

        # Megakernel
        mk = Megakernel(f"fuse_rope_gl_{config_name.replace('=', '_').replace(' ', '_')}", mode="forward")
        mk.add(rope_impl, q.view(-1, h, d), cos, sin, s)
        mk.add(gl_impl, q.view(-1, h * d), gate, out, h * d)
        grid = [q.shape[0] * q.shape[1], 1, 1]
        block = [256, 1, 1]

        def run_mk(mk=mk, barrier=barrier, grid=grid, block=block):
            mk.launch(barrier, grid[0], grid, block)

        ms_mk = do_bench(run_mk)

        print(f"{config_name:<25} | {'Sequential':<15} | {ms_seq:<10.4f}")
        print(f"{'':<25} | {'Megakernel':<15} | {ms_mk:<10.4f}")
        print("-" * 60)


if __name__ == "__main__":
    main()
