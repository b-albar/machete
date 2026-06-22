# Copyright (c) 2025, Machete Authors
"""
TMA megakernel integration test: TMA load (G2S), compute in smem, TMA store (S2G).

Tests that the megakernel framework correctly:
1. Creates TMA descriptors for both G2S (load) and S2G (store)
2. Threads TMA params (atom + gmem) through kernel -> dispatch -> op
3. Op's load/store methods use TMA copy atoms for async DMA
"""

import importlib.util

import pytest
import torch

if importlib.util.find_spec("cutlass") is None:
    pytest.skip("Requires CUTLASS", allow_module_level=True)

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from machete.megakernel.backend import HandlerBackend
from machete.megakernel.megakernel import Megakernel, MegakernelConfig
from machete.megakernel.ops import Op
from machete.megakernel.interpreter import (
    mbarrier_arrive_expect_tx,
    mbarrier_init,
    mbarrier_init_fence_async_proxy,
    mbarrier_wait,
    named_barrier_sync,
)
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


class ComputeTmaBNHDDChunkCopyOp(Op):
    """Compute-issued TMA copy from a BNHD tensor for one D tile."""

    reads = {"x": (None, ("B", "N", "H", "D"))}
    writes = {"y": (None, ("B", "N", "H", "D"))}
    tile = ("B", "D")
    tma_compute_loads = {"x"}
    tma_loads = set()
    tma_stores = set()

    D_BLOCK = 64
    H_STATIC = 2
    N_BLOCK = 16

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name == "x":
            return (1, cls.N_BLOCK, 1, cls.D_BLOCK)
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name != "x":
            return None
        d, h, n, b = tma_tile_shape
        return (
            f"cute.make_layout(({d}, {h}, {n}, {b}), "
            f"stride=(1, {d * n}, {d}, {d * n}))"
        )

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_D, x, y, x_tma, x_tma_gmem, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        ready = page_ptr + Int32(self.D_BLOCK * self.N_BLOCK * ELEM_BYTES)
        ready_ptr = cute.make_ptr(cutlass.Int64, ready, cute.AddressSpace.smem)
        if warp_idx == Int32(0):
            if tidx == Int32(0):
                mbarrier_init(ready, Int32(1))
            mbarrier_init_fence_async_proxy()
        named_barrier_sync(Int32(1), Int32(self.threads_per_row))

        sX_tma = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout(
                (self.D_BLOCK, 1, self.N_BLOCK, 1),
                stride=(1, self.D_BLOCK * self.N_BLOCK, self.D_BLOCK, self.D_BLOCK * self.N_BLOCK),
            ),
        )
        gX_tma = cute.local_tile(
            x_tma_gmem,
            (self.D_BLOCK, 1, self.N_BLOCK, 1),
            (None, None, None, None),
        )
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            x_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sX_tma, 0, 4),
            cute.group_modes(gX_tma, 0, 4),
        )
        if warp_idx == Int32(0):
            with cute.arch.elect_one():
                mbarrier_arrive_expect_tx(ready, Int32(self.D_BLOCK * self.N_BLOCK * ELEM_BYTES))
        cute.copy(
            x_tma,
            tXgX[(None, Int32(tile_D), Int32(0), Int32(0), Int32(tile_B))],
            tXsX,
            tma_bar_ptr=ready_ptr,
        )
        mbarrier_wait(ready, Int32(0))
        named_barrier_sync(Int32(1), Int32(self.threads_per_row))

        sX = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout((self.N_BLOCK, self.D_BLOCK), stride=(self.D_BLOCK, 1)),
        )
        idx = tidx
        while idx < Int32(self.N_BLOCK * self.D_BLOCK):
            n = idx // Int32(self.D_BLOCK)
            d = idx - n * Int32(self.D_BLOCK)
            y[tile_B, n, Int32(0), tile_D * Int32(self.D_BLOCK) + d] = sX[n, d]
            idx = idx + Int32(self.threads_per_row)


class LoadTmaBNHDDChunkCopyOp(ComputeTmaBNHDDChunkCopyOp):
    """Load-phase TMA version of the same BNHD D-tile copy."""

    tma_compute_loads = set()
    tma_loads = {"x"}

    @cute.jit
    def load(self, page_ptr, tile_B, tile_D, x_tma, x_tma_gmem, work_mbar):
        sX_tma = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout(
                (self.D_BLOCK, 1, self.N_BLOCK, 1),
                stride=(1, self.D_BLOCK * self.N_BLOCK, self.D_BLOCK, self.D_BLOCK * self.N_BLOCK),
            ),
        )
        gX_tma = cute.local_tile(
            x_tma_gmem,
            (self.D_BLOCK, 1, self.N_BLOCK, 1),
            (None, None, None, None),
        )
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            x_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sX_tma, 0, 4),
            cute.group_modes(gX_tma, 0, 4),
        )
        nbytes = Int32(self.D_BLOCK * self.N_BLOCK * ELEM_BYTES)
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(
            x_tma,
            tXgX[(None, Int32(tile_D), Int32(0), Int32(0), Int32(tile_B))],
            tXsX,
            tma_bar_ptr=mbar_ptr,
        )

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_D, x, y):
        tidx = cute.arch.thread_idx()[0]
        sX = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout((self.N_BLOCK, self.D_BLOCK), stride=(self.D_BLOCK, 1)),
        )
        idx = tidx
        while idx < Int32(self.N_BLOCK * self.D_BLOCK):
            n = idx // Int32(self.D_BLOCK)
            d = idx - n * Int32(self.D_BLOCK)
            y[tile_B, n, Int32(0), tile_D * Int32(self.D_BLOCK) + d] = sX[n, d]
            idx = idx + Int32(self.threads_per_row)


class LoadTmaBNDHeadSliceDChunkCopyOp(Op):
    """Load-phase TMA copy from a per-head BND tensor view for one D tile."""

    reads = {"x": (None, ("B", "N", "D"))}
    writes = {"y": (None, ("B", "N", "D"))}
    tile = ("B", "D")
    tma_loads = {"x"}

    D_BLOCK = 64
    N_BLOCK = 16

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name == "x":
            return (1, cls.N_BLOCK, cls.D_BLOCK)
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name != "x":
            return None
        d, n, b = tma_tile_shape
        return f"cute.make_layout(({d}, {n}, {b}), stride=(1, {d}, {d * n}))"

    @cute.jit
    def load(self, page_ptr, tile_B, tile_D, x_tma, x_tma_gmem, work_mbar):
        sX_tma = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout(
                (self.D_BLOCK, self.N_BLOCK, 1),
                stride=(1, self.D_BLOCK, self.D_BLOCK * self.N_BLOCK),
            ),
        )
        gX_tma = cute.local_tile(
            x_tma_gmem,
            (self.D_BLOCK, self.N_BLOCK, 1),
            (None, None, None),
        )
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            x_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sX_tma, 0, 3),
            cute.group_modes(gX_tma, 0, 3),
        )
        nbytes = Int32(self.D_BLOCK * self.N_BLOCK * ELEM_BYTES)
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(
            x_tma,
            tXgX[(None, Int32(tile_D), Int32(0), Int32(tile_B))],
            tXsX,
            tma_bar_ptr=mbar_ptr,
        )

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_D, x, y):
        tidx = cute.arch.thread_idx()[0]
        sX = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout((self.N_BLOCK, self.D_BLOCK), stride=(self.D_BLOCK, 1)),
        )
        idx = tidx
        while idx < Int32(self.N_BLOCK * self.D_BLOCK):
            n = idx // Int32(self.D_BLOCK)
            d = idx - n * Int32(self.D_BLOCK)
            y[tile_B, n, tile_D * Int32(self.D_BLOCK) + d] = sX[n, d]
            idx = idx + Int32(self.threads_per_row)


class LoadTmaNDLeadingDChunkCopyOp(Op):
    """Load-phase 2D TMA copy with D as the contiguous leading TMA mode."""

    reads = {"x": (None, ("N", "D"))}
    writes = {"y": (None, ("N", "D"))}
    tile = ("D",)
    tma_loads = {"x"}

    D_BLOCK = 64
    N_BLOCK = 16

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name == "x":
            return (cls.N_BLOCK, cls.D_BLOCK)
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name != "x":
            return None
        d, n = tma_tile_shape
        return f"cute.make_layout(({d}, {n}), stride=(1, {d}))"

    @cute.jit
    def load(self, page_ptr, tile_D, x_tma, x_tma_gmem, work_mbar):
        sX_tma = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout((self.D_BLOCK, self.N_BLOCK), stride=(1, self.D_BLOCK)),
        )
        gX_tma = cute.local_tile(x_tma_gmem, (self.D_BLOCK, self.N_BLOCK), (Int32(tile_D), None))
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            x_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sX_tma, 0, 2),
            cute.group_modes(gX_tma, 0, 2),
        )
        nbytes = Int32(self.D_BLOCK * self.N_BLOCK * ELEM_BYTES)
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(
            x_tma,
            tXgX[(None, Int32(0))],
            tXsX,
            tma_bar_ptr=mbar_ptr,
        )

    @cute.jit
    def compute(self, page_ptr, tile_D, x, y):
        tidx = cute.arch.thread_idx()[0]
        sX = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout((self.N_BLOCK, self.D_BLOCK), stride=(self.D_BLOCK, 1)),
        )
        idx = tidx
        while idx < Int32(self.N_BLOCK * self.D_BLOCK):
            n = idx // Int32(self.D_BLOCK)
            d = idx - n * Int32(self.D_BLOCK)
            y[n, tile_D * Int32(self.D_BLOCK) + d] = sX[n, d]
            idx = idx + Int32(self.threads_per_row)


# -- Tests ---------------------------------------------------------------------

@requires_hopper
class TestTMAMegakernel:

    def test_tma_add_one_single_tile(self):
        """Single tile: TMA load (G2S), add 1.0, TMA store (S2G)."""
        torch.manual_seed(42)
        x = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.full((TILE_M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")
        ops = TMAAddOneOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        Megakernel(ops).run()
        torch.testing.assert_close(y, x + 1.0, atol=1e-3, rtol=1e-3)

    def test_tma_add_one_multi_tile(self):
        """Multiple tiles (M=256, tile_M=64): verifies tile indexing works."""
        M = 256
        torch.manual_seed(42)
        x = torch.randn(M, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.full((M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")
        ops = TMAAddOneOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        Megakernel(ops).run()
        torch.testing.assert_close(y, x + 1.0, atol=1e-3, rtol=1e-3)

    def test_tma_runtime_transport_records_preserve_permuted_views(self):
        """Compact TMA dispatch must reconstruct the descriptor's tensor view."""
        old_value = getattr(HandlerBackend, "runtime_transport_records", False)
        HandlerBackend.runtime_transport_records = True
        try:
            M = 256
            torch.manual_seed(42)
            x = torch.randn(M, N_STATIC, dtype=torch.float16, device="cuda")
            y = torch.full((M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")
            ops = TMAAddOneOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
            Megakernel(ops).run()
            torch.testing.assert_close(y, x + 1.0, atol=1e-3, rtol=1e-3)
        finally:
            HandlerBackend.runtime_transport_records = old_value

    def test_compute_tma_bnhd_d_chunk_coordinates(self):
        """Compute-issued TMA must honor nonzero D tiles for contiguous BNHD."""
        B, N, H, D = 1, 16, 2, 256
        x = torch.empty(B, N, H, D, dtype=torch.float16, device="cuda")
        for d in range(D):
            x[:, :, 0, d] = float(d)
            x[:, :, 1, d] = float(1000 + d)
        y = torch.full_like(x, -1.0)
        ops = ComputeTmaBNHDDChunkCopyOp.schedule(
            x=x,
            y=y,
            tile_sizes={"D": 64},
        )
        Megakernel(
            ops,
            config=MegakernelConfig(num_sms=1, threads_per_block=128, page_size=32768),
        ).run()
        torch.testing.assert_close(y[:, :, 0, :], x[:, :, 0, :], atol=0, rtol=0)

    def test_load_tma_bnhd_d_chunk_coordinates(self):
        """Load-phase TMA must honor nonzero D tiles for contiguous BNHD."""
        B, N, H, D = 1, 16, 2, 256
        x = torch.empty(B, N, H, D, dtype=torch.float16, device="cuda")
        for d in range(D):
            x[:, :, 0, d] = float(d)
            x[:, :, 1, d] = float(1000 + d)
        y = torch.full_like(x, -1.0)
        ops = LoadTmaBNHDDChunkCopyOp.schedule(
            x=x,
            y=y,
            tile_sizes={"D": 64},
        )
        Megakernel(
            ops,
            config=MegakernelConfig(num_sms=1, threads_per_block=128, page_size=32768),
        ).run()
        torch.testing.assert_close(y[:, :, 0, :], x[:, :, 0, :], atol=0, rtol=0)

    def test_load_tma_bnd_head_slice_d_chunk_coordinates(self):
        """Per-head BND view TMA must honor nonzero D tiles."""
        B, N, H, D = 1, 16, 2, 256
        x4 = torch.empty(B, N, H, D, dtype=torch.float16, device="cuda")
        for d in range(D):
            x4[:, :, 0, d] = float(d)
            x4[:, :, 1, d] = float(1000 + d)
        x = x4[:, :, 0, :]
        y = torch.full_like(x, -1.0)
        ops = LoadTmaBNDHeadSliceDChunkCopyOp.schedule(
            x=x,
            y=y,
            tile_sizes={"D": 64},
        )
        Megakernel(
            ops,
            config=MegakernelConfig(num_sms=1, threads_per_block=128, page_size=32768),
        ).run()
        torch.testing.assert_close(y, x, atol=0, rtol=0)

    def test_load_tma_nd_leading_d_chunk_coordinates(self):
        """2D TMA must honor nonzero tiles in the contiguous leading mode."""
        N, D = 16, 256
        x = torch.empty(N, D, dtype=torch.float16, device="cuda")
        for d in range(D):
            x[:, d] = float(d)
        y = torch.full_like(x, -1.0)
        ops = LoadTmaNDLeadingDChunkCopyOp.schedule(
            x=x,
            y=y,
            tile_sizes={"D": 64},
        )
        Megakernel(
            ops,
            config=MegakernelConfig(num_sms=1, threads_per_block=128, page_size=32768),
        ).run()
        torch.testing.assert_close(y, x, atol=0, rtol=0)

    def test_tma_add_one_multi_wave(self):
        """A resident CTA can process more than one TMA tile without hanging."""
        M = TILE_M * 4
        torch.manual_seed(42)
        x = torch.randn(M, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.full((M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")
        ops = TMAAddOneOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        Megakernel(
            ops,
            config=MegakernelConfig(num_sms=2, threads_per_block=128),
        ).run()
        torch.testing.assert_close(y, x + 1.0, atol=1e-3, rtol=1e-3)

    def test_tma_replay_rejects_dma_only_thread_geometry(self):
        """TMA replay must not silently launch with no compute warps."""
        torch.manual_seed(42)
        x = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.full((TILE_M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")
        ops = TMAAddOneOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        with pytest.raises(RuntimeError, match="at least one compute warp"):
            Megakernel(ops, config=MegakernelConfig(threads_per_block=96)).compile()

    def test_tma_add_one_single_tile_noinline(self):
        """Noinline path should rebuild exec TMA from runtime desc pointers."""
        torch.manual_seed(42)
        x = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.full((TILE_M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")
        ops = TMAAddOneOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        Megakernel(ops, config=MegakernelConfig()).run()
        torch.testing.assert_close(y, x + 1.0, atol=1e-3, rtol=1e-3)

    def test_repeated_identical_ops_share_handler_but_keep_distinct_bindings(self):
        """Shared handlers must still dispatch per-op tensor bindings correctly."""
        torch.manual_seed(42)
        x0 = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda")
        y0 = torch.full((TILE_M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")
        x1 = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda")
        y1 = torch.full((TILE_M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")

        ops = (
            TMAAddOneOp.schedule(x=x0, y=y0, tile_sizes={"M": TILE_M})
            + TMAAddOneOp.schedule(x=x1, y=y1, tile_sizes={"M": TILE_M})
        )
        kernel = Megakernel(ops)

        assert len(kernel._backend_ir.handler_specs) == 1

        kernel.run()

        torch.testing.assert_close(y0, x0 + 1.0, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(y1, x1 + 1.0, atol=1e-3, rtol=1e-3)

    def test_repeated_tma_ops_keep_distinct_bindings(self):
        """Repeated TMA ops must preserve distinct desc slots."""
        torch.manual_seed(42)
        x0 = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda")
        y0 = torch.full((TILE_M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")
        x1 = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda")
        y1 = torch.full((TILE_M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")

        ops = (
            TMAAddOneOp.schedule(x=x0, y=y0, tile_sizes={"M": TILE_M})
            + TMAAddOneOp.schedule(x=x1, y=y1, tile_sizes={"M": TILE_M})
        )
        kernel = Megakernel(ops, config=MegakernelConfig())

        assert len(kernel._backend_ir.handler_specs) == 1

        kernel.run()

        torch.testing.assert_close(y0, x0 + 1.0, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(y1, x1 + 1.0, atol=1e-3, rtol=1e-3)

    def test_repeated_tma_ops_keep_distinct_bindings_with_runtime_transport_records(self):
        """Compact TMA dispatch must not reuse desc slots across shared handlers."""
        old_value = getattr(HandlerBackend, "runtime_transport_records", False)
        HandlerBackend.runtime_transport_records = True
        try:
            torch.manual_seed(42)
            x0 = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda")
            y0 = torch.full((TILE_M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")
            x1 = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda")
            y1 = torch.full((TILE_M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")

            ops = (
                TMAAddOneOp.schedule(x=x0, y=y0, tile_sizes={"M": TILE_M})
                + TMAAddOneOp.schedule(x=x1, y=y1, tile_sizes={"M": TILE_M})
            )
            kernel = Megakernel(ops, config=MegakernelConfig())

            assert len(kernel._backend_ir.handler_specs) == 1
            kernel.compile()
            assert len(kernel._tma_registry.descriptors) == 4
            assert kernel._phase_local_desc_slot_widths["load"] == 1
            assert kernel._phase_local_desc_slot_widths["store"] == 1
            torch.testing.assert_close(
                kernel._phase_local_desc_slot_tensors["load"].cpu(),
                torch.tensor([0, 2], dtype=torch.int32),
            )
            torch.testing.assert_close(
                kernel._phase_local_desc_slot_tensors["store"].cpu(),
                torch.tensor([1, 3], dtype=torch.int32),
            )

            kernel.run()

            torch.testing.assert_close(y0, x0 + 1.0, atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(y1, x1 + 1.0, atol=1e-3, rtol=1e-3)
        finally:
            HandlerBackend.runtime_transport_records = old_value

    def test_tma_kernel_cache_reuse_across_same_shape_allocations(self):
        """Compiled TMA kernels should be reusable across same-shape reallocations."""
        torch.manual_seed(42)

        x0 = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda")
        y0 = torch.full((TILE_M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")
        k0 = Megakernel(TMAAddOneOp.schedule(x=x0, y=y0, tile_sizes={"M": TILE_M}))
        k0.compile()

        x1 = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda")
        y1 = torch.full((TILE_M, N_STATIC), -999.0, dtype=torch.float16, device="cuda")
        k1 = Megakernel(TMAAddOneOp.schedule(x=x1, y=y1, tile_sizes={"M": TILE_M}))
        k1.compile()

        assert k0._compiled_kernel is k1._compiled_kernel

        k1.run()
        torch.testing.assert_close(y1, x1 + 1.0, atol=1e-3, rtol=1e-3)
