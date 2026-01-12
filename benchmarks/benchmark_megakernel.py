# Copyright (c) 2025, Machete Authors
import torch

from machete.megakernel.core import Megakernel
from machete.kernels.gated_linear.sm80 import GatedLinearSM80
from machete.kernels.gated_linear import GatedLinear as GatedLinearOp
from machete.kernels.rope.sm80 import RopeSM80


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
        gl_op = GatedLinearOp(dtype, "silu")

        def run_seq(q=q, cos=cos, sin=sin, gate=gate, out=out):
            rope_op(q, cos, sin)
            gl_op(q.view(-1, h * d), gate)

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

        # Backward Benchmark (Megakernel only for now as Seq backward requires autograd plumbing setup or manual calls)
        # We can implement Sequential Backward manually easily.

        d_out = torch.randn_like(out)
        d_a = torch.empty_like(q)
        d_b = torch.empty_like(gate)
        d_q = torch.empty_like(q)  # In-place update of d_a in reality

        # Sequential Backward
        # GL Backward -> d_a, d_b
        # RoPE Backward -> d_q (from d_a)

        # Need to cast to right shapes
        # GL Bwd: d_c, a, b, d_a, d_b, n_cols
        # RoPE Bwd: smem, mq, cos, sin, s

        mk_bwd = Megakernel(f"fuse_rope_gl_{config_name.replace('=', '_').replace(' ', '_')}", mode="backward")
        mk_bwd.add(
            gl_impl,
            d_out.view(-1, h * d),
            q.view(-1, h * d),
            gate.view(-1, h * d),
            d_a.view(-1, h * d),
            d_b.view(-1, h * d),
            h * d,
        )
        mk_bwd.add(rope_impl, d_a.view(-1, h, d), cos, sin, s)

        def run_mk_bwd(mk=mk_bwd, barrier=barrier, grid=grid, block=block):
            mk.launch(barrier, grid[0], grid, block)

        ms_mk_bwd = do_bench(run_mk_bwd)

        # Sequential Backward (Kernel Launch-based)
        mk_gl_bwd = Megakernel(f"gl_bwd_only_{config_name.replace('=', '_').replace(' ', '_')}", mode="backward")
        mk_gl_bwd.add(
            gl_impl,
            d_out.view(-1, h * d),
            q.view(-1, h * d),
            gate.view(-1, h * d),
            d_a.view(-1, h * d),
            d_b.view(-1, h * d),
            h * d,
        )

        mk_rope_bwd = Megakernel(f"rope_bwd_only_{config_name.replace('=', '_').replace(' ', '_')}", mode="backward")
        mk_rope_bwd.add(rope_impl, d_a.view(-1, h, d), cos, sin, s)

        def run_seq_bwd(mk1=mk_gl_bwd, mk2=mk_rope_bwd, barrier=barrier, grid=grid, block=block):
            mk1.launch(barrier, grid[0], grid, block)
            mk2.launch(barrier, grid[0], grid, block)

        ms_seq_bwd = do_bench(run_seq_bwd)

        print(f"{'':<25} | {'Sequential Bwd':<15} | {ms_seq_bwd:<10.4f}")
        print(f"{'':<25} | {'Megakernel Bwd':<15} | {ms_mk_bwd:<10.4f}")

        print("-" * 60)


if __name__ == "__main__":
    main()
