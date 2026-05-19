# Copyright (c) 2025, Machete Authors
"""Synthetic TMA ops for framework tests and scaling benchmarks."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from machete.megakernel.interpreter import (
    mbarrier_arrive_expect_tx,
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_wait,
)
from machete.megakernel.ops import Op


SYNTHETIC_TMA_TILE_M = 64
SYNTHETIC_TMA_N = 64
SYNTHETIC_TMA_ELEM_BYTES = 2  # fp16


class SyntheticTMAAddOneOp(Op):
    """Minimal TMA op for integration, scaling, and dispatch tests.

    The op intentionally keeps the compute body trivial. The goal is to stress:
    - TMA descriptor plumbing
    - runtime transport selection
    - handler backend scaling
    rather than math throughput.
    """

    reads = {"x": (None, ("M", "N"))}
    writes = {"y": (None, ("M", "N"))}
    tile = ("M",)

    tma_loads = {"x"}
    tma_stores = {"y"}

    @cute.jit
    def load(self, page_ptr, tile_M, x_tma, x_tma_gmem, work_mbar):
        sA = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_M)),
        )
        gA = cute.local_tile(
            x_tma_gmem,
            (self.N, self.tile_size_M),
            (None, None),
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            x_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )

        nbytes = Int32(self.tile_size_M * self.N * SYNTHETIC_TMA_ELEM_BYTES)
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(x_tma, tAgA[(None, 0, tile_M)], tAsA, tma_bar_ptr=mbar_ptr)

    @cute.jit
    def compute(self, page_ptr, tile_M, x, y):
        tidx = cute.arch.thread_idx()[0]
        total_elems = self.tile_size_M * self.N
        s = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((total_elems,)),
        )
        one = self.x_dtype(1.0)
        for i in range(tidx, total_elems, self.threads_per_row):
            s[i] = s[i] + one

    @cute.jit
    def store(self, page_ptr, tile_M, y_tma, y_tma_gmem):
        sA = cute.make_tensor(
            cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_M)),
        )
        gA = cute.local_tile(
            y_tma_gmem,
            (self.N, self.tile_size_M),
            (None, None),
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            y_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        with cute.arch.elect_one():
            cute.copy(y_tma, tAsA, tAgA[(None, 0, tile_M)])
