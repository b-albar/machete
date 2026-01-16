# Copyright (c) 2025, Machete Authors
"""
Tests demonstrating the optimization of the "No Bubbles" mechanism.
Focuses on overlap between Load, Compute, and Store phases across multiple stages.
"""

import pytest
import torch
import cutlass.cute as cute
from machete.megakernel.interface import (
    WarpSpecializedKernel,
    WarpConfig,
    WarpRole,
    reads,
    writes,
    warp_role,
)
from machete.megakernel.core import Megakernel


class HeavyComputeKernel(WarpSpecializedKernel):
    """Kernel with tunable compute intensity to show pipeline overlap."""

    TILE_SIZE = 512

    def __init__(self, iterations: int = 200, dtype=cute.Float16):
        self.iterations = iterations
        self.cute_dtype = dtype

    @property
    def warp_config(self) -> WarpConfig:
        # 12 consumers + Loader + Storer + others = 16-20 warps
        return WarpConfig(num_consumer_warps=12)

    @property
    def smem_size(self) -> int:
        return self.TILE_SIZE * 2  # 2 bytes per element (fp16)

    def get_logical_grid_size(self, input_t, output_t, n) -> int:
        return (n + self.TILE_SIZE - 1) // self.TILE_SIZE

    @warp_role(WarpRole.LOADER)
    @reads("input")
    @cute.jit
    def load_forward(self, paged_pool, page_idx, logical_idx, smem, input_t, output_t, n):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        base = logical_idx * self.TILE_SIZE

        # Loader warp loads elements
        for i in range(self.TILE_SIZE // 32):
            idx = base + lane + i * 32
            if idx < n:
                smem[lane + i * 32] = input_t[idx]

    @warp_role(WarpRole.CONSUMER)
    @writes("smem")
    @cute.jit
    def compute_forward(self, logical_idx, smem, input_t, output_t, n):
        tidx, _, _ = cute.arch.thread_idx()
        warp_id = tidx // 32
        lane = tidx % 32

        # We have 12 consumer warps
        elems_per_warp = self.TILE_SIZE // 12
        start = warp_id * elems_per_warp

        # Artificial high arithmetic intensity to show overlap
        for i in range((elems_per_warp + 31) // 32):
            local = start + lane + i * 32
            if local < start + elems_per_warp and local < self.TILE_SIZE:
                val = smem[local]
                # Heavy math loop
                for _ in range(self.iterations):
                    val = val * val + val
                smem[local] = val

    @warp_role(WarpRole.STORER)
    @writes("output")
    @cute.jit
    def store_forward(self, paged_pool, page_idx, logical_idx, smem, input_t, output_t, n):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        base = logical_idx * self.TILE_SIZE

        for i in range(self.TILE_SIZE // 32):
            idx = base + lane + i * 32
            if idx < n:
                output_t[idx] = smem[lane + i * 32]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sequential_vs_no_bubbles(cuda_device, trace_file):
    """Generate trace for no_bubbles kernel and verify it works.

    Note: Warp-specialized kernels ALWAYS use warp-specialized scheduling,
    so there's no "sequential" mode for comparison. This test verifies that
    the No Bubbles optimization works correctly.
    """
    n = 1024 * 16
    x = torch.ones(n, dtype=torch.float16, device=cuda_device)
    y_nb = torch.zeros(n, dtype=torch.float16, device=cuda_device)

    # No Bubbles Version (num_stages=4)
    kernel_nb = HeavyComputeKernel(iterations=100)
    mk_nb = Megakernel(
        name="no_bubbles_optimized", num_stages=4, page_size=kernel_nb.TILE_SIZE * 2
    )
    mk_nb.add(kernel_nb, x, y_nb, n)

    nb_trace = trace_file.replace(".nanotrace", "_nobubbles.nanotrace")
    mk_nb.launch_logical(
        block=(kernel_nb.warp_config.total_threads, 1, 1),
        trace_file=nb_trace,
    )

    # Verify the output is non-zero (computation happened)
    assert (y_nb != 0).any(), "No bubbles output should be non-zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--trace-kernels"])
