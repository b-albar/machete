# Copyright (c) 2025, Machete Authors
import argparse
import os
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
    parser = argparse.ArgumentParser(description="Benchmark Machete Megakernels")
    parser.add_argument("--trace", action="store_true", help="Enable tracing for the first config")
    parser.add_argument("--warmup", type=int, default=25, help="Number of warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="Number of benchmark iterations")
    args_cli = parser.parse_args()

    device = "cuda"
    dtype = torch.float16

    print(f"\n{'=' * 20} RoPE + GatedLinear Fusion {'=' * 20}")
    print(f"{'Config':<25} | {'Provider':<15} | {'Time (ms)':<10}")
    print("-" * 60)

    config_list = []
    for b_sz in [1, 2, 4, 8, 16]:
        for s_sz in [1024, 2048, 4096, 8192]:
            config_list.append((b_sz, s_sz, 32, 128))

    rope_impl = RopeSM80(dtype, 128)
    gl_impl = GatedLinearSM80(dtype, "silu")

    for i, config in enumerate(config_list):
        b_sz, s_sz, h_sz, d_sz = config
        config_name = f"B={b_sz} S={s_sz} H={h_sz} D={d_sz}"

        # Enable tracing only for the first configuration if requested
        trace_fwd = f"megakernel_fwd_{b_sz}_{s_sz}.nanotrace" if args_cli.trace and i == 0 else None
        trace_bwd = f"megakernel_bwd_{b_sz}_{s_sz}.nanotrace" if args_cli.trace and i == 0 else None

        if trace_fwd and os.path.exists(trace_fwd):
            os.remove(trace_fwd)
        if trace_bwd and os.path.exists(trace_bwd):
            os.remove(trace_bwd)

        try:
            # Create tensors inside the loop to save memory
            q = torch.randn(b_sz, s_sz, h_sz, d_sz, device=device, dtype=dtype)
            cos = torch.randn(s_sz, d_sz, device=device, dtype=dtype)
            sin = torch.randn(s_sz, d_sz, device=device, dtype=dtype)
            gate = torch.randn(b_sz * s_sz, h_sz * d_sz, device=device, dtype=dtype)
            out = torch.empty(b_sz * s_sz, h_sz * d_sz, device=device, dtype=dtype)

            # Sequential
            rope_op = RopeSM80(dtype, d_sz)
            gl_op = GatedLinearOp(dtype, "silu")

            def run_seq(q=q, cos=cos, sin=sin, gate=gate, out=out):
                rope_op(q, cos, sin)
                gl_op(q.view(-1, h_sz * d_sz), gate)

            ms_seq = do_bench(run_seq, warmup=args_cli.warmup, rep=args_cli.rep)

            # Megakernel (No Bubbles mode with paged pool)
            mk = Megakernel(
                f"fuse_rope_gl_{config_name.replace('=', '_').replace(' ', '_')}",
                mode="forward",
                num_stages=0,  # Enable No Bubbles pipelining
            )
            mk.add(rope_impl, q.view(-1, h_sz, d_sz), cos, sin, s_sz)
            mk.add(gl_impl, q.view(-1, h_sz * d_sz), gate, out, h_sz * d_sz)
            grid = [q.shape[0] * q.shape[1], 1, 1]
            block = [256, 1, 1]

            # One launch for tracing if requested
            if trace_fwd:
                mk.launch(grid[0], grid, block, trace_file=trace_fwd)

            def run_mk(mk=mk, grid=grid, block=block):
                mk.launch(grid[0], grid, block)

            ms_mk = do_bench(run_mk, warmup=args_cli.warmup, rep=args_cli.rep)

            print(f"{config_name:<25} | {'Sequential':<15} | {ms_seq:<10.4f}")
            print(f"{'':<25} | {'Megakernel':<15} | {ms_mk:<10.4f}")

            # Backward Benchmark
            d_out = torch.randn_like(out)
            d_a = torch.empty_like(q)
            d_b = torch.empty_like(gate)

            mk_bwd = Megakernel(f"fuse_rope_gl_{config_name.replace('=', '_').replace(' ', '_')}", mode="backward")
            mk_bwd.add(
                gl_impl,
                d_out.view(-1, h_sz * d_sz),
                q.view(-1, h_sz * d_sz),
                gate.view(-1, h_sz * d_sz),
                d_a.view(-1, h_sz * d_sz),
                d_b.view(-1, h_sz * d_sz),
                h_sz * d_sz,
            )
            mk_bwd.add(rope_impl, d_a.view(-1, h_sz, d_sz), cos, sin, s_sz)

            # One launch for tracing if requested
            if trace_bwd:
                mk_bwd.launch(grid[0], grid, block, trace_file=trace_bwd)

            def run_mk_bwd(mk=mk_bwd, grid=grid, block=block):
                mk.launch(grid[0], grid, block)

            ms_mk_bwd = do_bench(run_mk_bwd, warmup=args_cli.warmup, rep=args_cli.rep)

            # Sequential Backward (Kernel Launch-based)
            mk_gl_bwd = Megakernel(f"gl_bwd_only_{config_name.replace('=', '_').replace(' ', '_')}", mode="backward")
            mk_gl_bwd.add(
                gl_impl,
                d_out.view(-1, h_sz * d_sz),
                q.view(-1, h_sz * d_sz),
                gate.view(-1, h_sz * d_sz),
                d_a.view(-1, h_sz * d_sz),
                d_b.view(-1, h_sz * d_sz),
                h_sz * d_sz,
            )

            mk_rope_bwd = Megakernel(
                f"rope_bwd_only_{config_name.replace('=', '_').replace(' ', '_')}", mode="backward"
            )
            mk_rope_bwd.add(rope_impl, d_a.view(-1, h_sz, d_sz), cos, sin, s_sz)

            def run_seq_bwd(mk1=mk_gl_bwd, mk2=mk_rope_bwd, grid=grid, block=block):
                mk1.launch(grid[0], grid, block)
                mk2.launch(grid[0], grid, block)

            ms_seq_bwd = do_bench(run_seq_bwd, warmup=args_cli.warmup, rep=args_cli.rep)

            print(f"{'':<25} | {'Sequential Bwd':<15} | {ms_seq_bwd:<10.4f}")
            print(f"{'':<25} | {'Megakernel Bwd':<15} | {ms_mk_bwd:<10.4f}")

            if args_cli.trace and i == 0:
                print(f"\n[TRACE] Forward trace written to: {trace_fwd}")
                print(f"[TRACE] Backward trace written to: {trace_bwd}")
                print("[TRACE] Visualize them with local visualizer")
                print("-" * 60)
                break

        except torch.OutOfMemoryError:
            print(f"{config_name:<25} | {'OOM':<15} | {'N/A'}")
            torch.cuda.empty_cache()
            import gc

            gc.collect()

        print("-" * 60)


if __name__ == "__main__":
    main()
