# Copyright (c) 2025, Machete Authors
"""
Benchmark demonstrating page-aware scheduling with tracing.

This benchmark showcases the dependency-aware scheduling and generates
trace files that show page acquire/release events alongside L/C/S phases.
"""

import torch
from machete.megakernel.core import Megakernel
from machete.megakernel.scheduler import (
    NoBubblesConfig,
    PageAwareScheduler,
    OpDescriptor,
)
from machete.kernels.gated_linear.sm80 import GatedLinearSM80


def benchmark_page_aware_scheduling():
    """Benchmark page-aware scheduling and generate trace."""
    device = "cuda"
    dtype = torch.float16

    # Setup dimensions
    batch_size = 4
    seq_len = 256
    hidden_dim = 4096
    n_elements = batch_size * seq_len

    print("=" * 70)
    print("Page-Aware Scheduling Benchmark")
    print("=" * 70)
    print(f"Batch: {batch_size}, SeqLen: {seq_len}, Hidden: {hidden_dim}")
    print(f"Elements: {n_elements}")

    # Create test tensors
    x = torch.randn(n_elements, hidden_dim, device=device, dtype=dtype)
    gate = torch.randn(n_elements, hidden_dim, device=device, dtype=dtype)
    out = torch.empty(n_elements, hidden_dim, device=device, dtype=dtype)

    x2 = torch.randn(n_elements, hidden_dim, device=device, dtype=dtype)
    gate2 = torch.randn(n_elements, hidden_dim, device=device, dtype=dtype)
    out2 = torch.empty(n_elements, hidden_dim, device=device, dtype=dtype)

    x3 = torch.randn(n_elements, hidden_dim, device=device, dtype=dtype)
    gate3 = torch.randn(n_elements, hidden_dim, device=device, dtype=dtype)
    out3 = torch.empty(n_elements, hidden_dim, device=device, dtype=dtype)

    # Create megakernel with page-aware scheduling (4 pages)
    gl1 = GatedLinearSM80(dtype, "silu")
    gl2 = GatedLinearSM80(dtype, "silu")
    gl3 = GatedLinearSM80(dtype, "silu")

    # ---------------------------------------------------------------------
    # Show the scheduling visualization
    # ---------------------------------------------------------------------
    print("\n--- Scheduler Preview ---")
    config = NoBubblesConfig(num_pages=4)
    scheduler = PageAwareScheduler(config)

    ops = [
        OpDescriptor("GatedLinear1", 0, reads={"x"}, writes={"out"}, needs_block_sync=False),
        OpDescriptor("GatedLinear2", 1, reads={"x2"}, writes={"out2"}, needs_block_sync=False),
        OpDescriptor("GatedLinear3", 2, reads={"x3"}, writes={"out3"}, needs_block_sync=False),
    ]
    scheduler.generate_page_aware_schedule(ops, pages_per_op=2)
    print(scheduler.visualize_page_schedule())

    # ---------------------------------------------------------------------
    # Benchmark without tracing
    # ---------------------------------------------------------------------
    print("\n--- Benchmark (No Tracing) ---")

    mk = Megakernel("page_aware_bench", mode="forward", num_stages=4, page_size=16384)
    mk.add(gl1, x, gate, out, hidden_dim)
    mk.add(gl2, x2, gate2, out2, hidden_dim)
    mk.add(gl3, x3, gate3, out3, hidden_dim)

    grid = [n_elements, 1, 1]
    block = [256, 1, 1]

    # Warmup
    for _ in range(3):
        mk.launch(n_elements, grid, block)
    torch.cuda.synchronize()

    # Benchmark
    import time

    n_iters = 10
    start = time.perf_counter()
    for _ in range(n_iters):
        mk.launch(n_elements, grid, block)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Average time: {elapsed / n_iters * 1000:.3f} ms")
    print(f"Throughput: {n_elements * hidden_dim * 3 * 2 / (elapsed / n_iters) / 1e9:.2f} GB/s (approx)")

    # ---------------------------------------------------------------------
    # Generate trace with page events
    # ---------------------------------------------------------------------
    print("\n--- Generating Trace ---")

    mk_traced = Megakernel("page_aware_traced", mode="forward", num_stages=4, page_size=16384)
    mk_traced.add(gl1, x, gate, out, hidden_dim)
    mk_traced.add(gl2, x2, gate2, out2, hidden_dim)
    mk_traced.add(gl3, x3, gate3, out3, hidden_dim)

    trace_file = "page_aware_scheduling.nanotrace"
    mk_traced.launch(n_elements, grid, block, trace_file=trace_file)
    print(f"Trace written to: {trace_file}")
    print("Open with cutedsl-trace visualizer to see page acquire/release events!")

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_page_aware_scheduling()
