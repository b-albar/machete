# Copyright (c) 2025, Machete Authors
"""
TMA megakernel integration test: TMA load (G2S), compute in smem, TMA store (S2G).

Tests that the megakernel framework correctly:
1. Creates TMA descriptors for both G2S (load) and S2G (store)
2. Threads TMA params (atom + gmem) through kernel -> dispatch -> op
3. Op's load/store methods use TMA copy atoms for async DMA
"""

import pytest
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from machete.megakernel.megakernel import Megakernel
from machete.megakernel.ops import Op
from machete.megakernel.interpreter import mbarrier_arrive_expect_tx
from machete.utils.testing import is_hopper_available

requires_hopper = pytest.mark.skipif(
    not is_hopper_available(), reason="Requires Hopper (SM90+) GPU",
)

TILE_M = 64
N_STATIC = 64
ELEM_BYTES = 2  # fp16


# -- Ops -----------------------------------------------------------------------

class TMAAddOneOp(Op):
    """Add 1.0: TMA load x (G2S), add 1.0 in smem, TMA store y (S2G).

    Config via self (set by framework at compile time):
        self.tile_size_M, self.N, self.x_dtype, self.y_dtype, self.threads_per_row

    Method params:
        page_ptr: shared memory page pointer
        tile_M: tile index for M dimension
        x_tma, x_tma_gmem: TMA G2S copy atom and gmem tensor (load)
        y_tma, y_tma_gmem: TMA S2G copy atom and gmem tensor (store)
        work_mbar: mbarrier pointer (load only, async)
    """

    reads = {"x": (None, ("M", "N"))}
    writes = {"y": (None, ("M", "N"))}
    tile = ("M",)

    tma_loads = {"x"}
    tma_stores = {"y"}

    @cute.jit
    def load(self, page_ptr, tile_M, x_tma, x_tma_gmem, work_mbar):
        # Smem tile in page: (N, tile_M) col-major to match TMA descriptor.
        # TMA requires mode 0 contiguous, so gmem tensor is transposed and
        # tile shape is reversed from (tile_M, N) to (N, tile_M).
        sA = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_M)),
        )

        # Standard CuTe TMA pattern: local_tile → group_modes → tma_partition
        # Then index tAgA to select the specific tile for this tile_M.
        gA = cute.local_tile(
            x_tma_gmem, (self.N, self.tile_size_M), (None, None),
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            x_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )

        # Signal mbarrier with expected tx bytes (single thread via elect_one)
        # + issue async TMA copy (ALL threads — required for warp convergence)
        # Index: (None=TMA modes, 0=N tile (only 1), tile_M=M tile index)
        nbytes = Int32(self.tile_size_M * self.N * ELEM_BYTES)
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(x_tma, tAgA[(None, 0, tile_M)], tAsA, tma_bar_ptr=mbar_ptr)

    @cute.jit
    def compute(self, page_ptr, tile_M, x, y):
        # Add 1.0 to every element in smem (in-place).
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
        # Smem tile: (N, tile_M) col-major to match TMA descriptor.
        sA = cute.make_tensor(
            cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_M)),
        )

        # Standard CuTe TMA pattern: local_tile → group_modes → tma_partition
        gA = cute.local_tile(
            y_tma_gmem, (self.N, self.tile_size_M), (None, None),
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            y_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )

        # S2G: source=smem, dest=gmem (reversed from G2S).
        # No mbarrier needed. DMA warp loop handles commit_group + wait_group.
        with cute.arch.elect_one():
            cute.copy(y_tma, tAsA, tAgA[(None, 0, tile_M)])


# -- Tests ---------------------------------------------------------------------

@requires_hopper
class TestTMAMegakernel:

    def test_tma_add_one_single_tile(self):
        """Single tile: TMA load (G2S), add 1.0, TMA store (S2G)."""
        torch.manual_seed(42)
        x = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.full((TILE_M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")
        op = TMAAddOneOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        Megakernel([op]).run()
        torch.testing.assert_close(y, x + 1.0, atol=1e-3, rtol=1e-3)

    def test_tma_add_one_multi_tile(self):
        """Multiple tiles (M=256, tile_M=64): verifies tile indexing works."""
        M = 256
        torch.manual_seed(42)
        x = torch.randn(M, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.full((M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")
        op = TMAAddOneOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        Megakernel([op]).run()
        torch.testing.assert_close(y, x + 1.0, atol=1e-3, rtol=1e-3)
