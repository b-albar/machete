# Copyright (c) 2025, Machete Authors
"""Validation suite for multi-page requests + per-page release mbarriers.

Exercises the decoupled physical-page allocator added to the phased ring
replay: an op declares it wants N (>1) shared-memory pages, receives a
contiguous N-page region (sub-page i at page_ptr + i*machete_aligned_page_size),
and each physical page is released independently via its own page_finished
mbarrier — either after compute (early/out-of-order) or after store.

The `DualPageOp` below routes data x -> page1 -> page0 with a "trap": it writes
x+1 into page1, clobbers page0, then copies page1 back into page0. The result is
x+1 ONLY IF page0 and page1 are genuinely distinct physical pages; if the two
addresses aliased, the clobber would destroy the value and the output would be
the trap sentinel. page1 is marked release-after-compute, so the allocator must
reclaim it out-of-order while page0 is still pending its store.
"""

import importlib.util

import pytest
import torch

if importlib.util.find_spec("cutlass") is None:
    pytest.skip("Requires CUTLASS", allow_module_level=True)

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from machete.megakernel.megakernel import Megakernel, MegakernelConfig
from machete.megakernel.ops import Op
from machete.megakernel.paged_memory import NPageLayout, ld_shared_i32, st_shared_i32
from machete.megakernel.interpreter import mbarrier_arrive_expect_tx
from machete.utils.testing import is_hopper_available

requires_hopper = pytest.mark.skipif(
    not is_hopper_available(), reason="Requires Hopper/Blackwell (SM90+) GPU with TMA",
)

TILE_M = 64
N_STATIC = 64
ELEM_BYTES = 2  # fp16
TRAP = -999.0


class DualPageOp(Op):
    """x + 1 routed through a second, early-released page (distinctness trap).

    Requests 2 pages. page0 (page_ptr) is TMA-loaded with x and TMA-stored as y.
    page1 (page_ptr + aligned_page_size) is scratch, released after compute.
    """

    reads = {"x": (None, ("M", "N"))}
    writes = {"y": (None, ("M", "N"))}
    tile = ("M",)

    tma_loads = {"x"}
    tma_stores = {"y"}

    # Two distinct smem pages; page index 1 is freed right after compute.
    requested_page_count = 2

    @classmethod
    def page_release_after_compute_mask(cls, page_size: int) -> int:
        return 0b10  # page 1 -> release after compute (early); page 0 after store

    @cute.jit
    def load(self, page_ptr, tile_M, x_tma, x_tma_gmem, work_mbar):
        sA = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_M)),
        )
        gA = cute.local_tile(x_tma_gmem, (self.N, self.tile_size_M), (None, None))
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            x_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        nbytes = Int32(self.tile_size_M * self.N * ELEM_BYTES)
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(x_tma, tAgA[(None, 0, tile_M)], tAsA, tma_bar_ptr=mbar_ptr)

    @cute.jit
    def compute(self, page_ptr, tile_M, x, y):
        tidx = cute.arch.thread_idx()[0]
        total_elems = self.tile_size_M * self.N
        # page0 holds x (from load); page1 is the contiguous next physical page.
        s0 = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((total_elems,)),
        )
        s1 = cute.make_tensor(
            cute.make_ptr(
                self.x_dtype,
                page_ptr + Int32(self.machete_aligned_page_size),
                cute.AddressSpace.smem,
            ),
            cute.make_layout((total_elems,)),
        )
        one = self.x_dtype(1.0)
        trap = self.x_dtype(TRAP)
        for i in range(tidx, total_elems, self.threads_per_row):
            s1[i] = s0[i] + one   # x+1 into page1
            s0[i] = trap          # clobber page0 (no-op for result iff distinct)
            s0[i] = s1[i]         # restore from page1 -> only correct if distinct

    @cute.jit
    def store(self, page_ptr, tile_M, y_tma, y_tma_gmem):
        sA = cute.make_tensor(
            cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_M)),
        )
        gA = cute.local_tile(y_tma_gmem, (self.N, self.tile_size_M), (None, None))
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            y_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        with cute.arch.elect_one():
            cute.copy(y_tma, tAsA, tAgA[(None, 0, tile_M)])


# Single-page op for the mixed-N fusion test (reuses the same TMA pattern).
class _AddTwoOp(Op):
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
        gA = cute.local_tile(x_tma_gmem, (self.N, self.tile_size_M), (None, None))
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            x_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        nbytes = Int32(self.tile_size_M * self.N * ELEM_BYTES)
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
        two = self.x_dtype(2.0)
        for i in range(tidx, total_elems, self.threads_per_row):
            s[i] = s[i] + two

    @cute.jit
    def store(self, page_ptr, tile_M, y_tma, y_tma_gmem):
        sA = cute.make_tensor(
            cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_M)),
        )
        gA = cute.local_tile(y_tma_gmem, (self.N, self.tile_size_M), (None, None))
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            y_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        with cute.arch.elect_one():
            cute.copy(y_tma, tAsA, tAgA[(None, 0, tile_M)])


class StoreReadsSecondPageOp(DualPageOp):
    """Keep page1 live until store and TMA-store directly from that page."""

    @classmethod
    def page_release_after_compute_mask(cls, page_size: int) -> int:
        return 0

    @cute.jit
    def compute(self, page_ptr, tile_M, x, y):
        tidx = cute.arch.thread_idx()[0]
        total_elems = self.tile_size_M * self.N
        s0 = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((total_elems,)),
        )
        s1 = cute.make_tensor(
            cute.make_ptr(
                self.x_dtype,
                page_ptr + Int32(self.machete_aligned_page_size),
                cute.AddressSpace.smem,
            ),
            cute.make_layout((total_elems,)),
        )
        three = self.x_dtype(3.0)
        trap = self.x_dtype(TRAP)
        for i in range(tidx, total_elems, self.threads_per_row):
            s1[i] = s0[i] + three
            s0[i] = trap

    @cute.jit
    def store(self, page_ptr, tile_M, y_tma, y_tma_gmem):
        sA = cute.make_tensor(
            cute.make_ptr(
                self.y_dtype,
                page_ptr + Int32(self.machete_aligned_page_size),
                cute.AddressSpace.smem,
            ),
            cute.make_layout((self.N, self.tile_size_M)),
        )
        gA = cute.local_tile(y_tma_gmem, (self.N, self.tile_size_M), (None, None))
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            y_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        with cute.arch.elect_one():
            cute.copy(y_tma, tAsA, tAgA[(None, 0, tile_M)])


class StoreStepReadsSecondPageOp(StoreReadsSecondPageOp):
    """Exercise store_step state while page1 remains live until store release."""

    enable_store_step_by_default = True

    def store_step_count(self) -> int:
        return 2

    @cute.jit
    def store_step(
        self,
        page_ptr,
        tile_M,
        y_tma,
        y_tma_gmem,
        store_state_ptr,
    ):
        state = ld_shared_i32(store_state_ptr)
        total_elems = self.tile_size_M * self.N
        s0 = cute.make_tensor(
            cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((total_elems,)),
        )
        s1 = cute.make_tensor(
            cute.make_ptr(
                self.y_dtype,
                page_ptr + Int32(self.machete_aligned_page_size),
                cute.AddressSpace.smem,
            ),
            cute.make_layout((total_elems,)),
        )
        if state == Int32(0):
            for i in range(total_elems):
                s0[i] = s1[i]
            st_shared_i32(store_state_ptr, Int32(1))
        if state == Int32(1):
            sA = cute.make_tensor(
                cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem),
                cute.make_layout((self.N, self.tile_size_M)),
            )
            gA = cute.local_tile(y_tma_gmem, (self.N, self.tile_size_M), (None, None))
            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                y_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA, 0, 2),
            )
            cute.copy(y_tma, tAsA, tAgA[(None, 0, tile_M)])
            st_shared_i32(store_state_ptr, Int32(2))


class _NoPageMultiPageOp(DualPageOp):
    uses_smem_page = False


class _BadReleaseMaskOp(DualPageOp):
    @classmethod
    def page_release_after_compute_mask(cls, page_size: int) -> int:
        return 0b100


def test_layout_smem_accounting():
    """page_finished array is gated and fits in the existing 128B scratch slack."""
    off = NPageLayout(num_pages=3, num_slots=3, page_size=32768)
    on = NPageLayout(num_pages=3, num_slots=3, page_size=32768,
                     page_release_mbarriers=True)
    # The page_finished[num_pages] block lives right after the two slot arrays.
    assert on.page_finished_mbar_offset(0) == on.mbarrier_offset + 2 * on.num_slots * 8
    assert on.page_finished_mbar_offset(0) < on.pages_start
    # Single-page (off) layout is unchanged; pages start at the same offset.
    assert off.pages_start == on.pages_start


def test_for_device_rejects_minimum_layout_that_does_not_fit():
    min_layout = NPageLayout(
        num_pages=2,
        page_size=32768,
        page_release_mbarriers=True,
    )
    with pytest.raises(ValueError, match="Cannot fit 2 pages"):
        NPageLayout.for_device(
            page_size=32768,
            max_smem=min_layout.total_size - 1,
            min_pages=2,
            page_release_mbarriers=True,
        )


@requires_hopper
class TestMultiPageAllocator:
    def _run(self, M, op_cls=DualPageOp, config=None):
        torch.manual_seed(0)
        x = torch.randn(M, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.full((M, N_STATIC), TRAP, dtype=torch.float16, device="cuda")
        ops = op_cls.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        Megakernel(ops, config=config).run() if config else Megakernel(ops).run()
        return x, y

    def test_single_tile_distinct_pages(self):
        """One tile: proves page1 is a distinct, usable address (else -> TRAP)."""
        x, y = self._run(TILE_M)
        torch.testing.assert_close(y, x + 1.0, atol=1e-3, rtol=1e-3)

    def test_many_tiles_rotation_and_early_release(self):
        """8 tiles over a 4-page ring: exercises 2 blocks, rotation, page reuse,
        and out-of-order reclaim of the early-released page1."""
        x, y = self._run(
            TILE_M * 8, config=MegakernelConfig(num_pages=4, page_size=16384)
        )
        torch.testing.assert_close(y, x + 1.0, atol=1e-3, rtol=1e-3)

    def test_skip_on_wrap_tight_ring(self):
        """3-page ring with N=2: the allocator must skip the wrap (pages {0,1}
        only) and never straddle the boundary, across many tiles."""
        x, y = self._run(
            TILE_M * 6, config=MegakernelConfig(num_pages=3, page_size=16384)
        )
        torch.testing.assert_close(y, x + 1.0, atol=1e-3, rtol=1e-3)

    def test_default_config_autopages(self):
        """Auto-detected num_pages must satisfy the >=N floor and run correctly."""
        x, y = self._run(TILE_M * 4)
        torch.testing.assert_close(y, x + 1.0, atol=1e-3, rtol=1e-3)

    def test_mixed_n_fusion(self):
        """Fuse a 2-page op and a 1-page op (independent tensors) in one kernel:
        the allocator must handle heterogeneous per-op page counts."""
        torch.manual_seed(1)
        x0 = torch.randn(TILE_M * 2, N_STATIC, dtype=torch.float16, device="cuda")
        y0 = torch.full((TILE_M * 2, N_STATIC), TRAP, dtype=torch.float16, device="cuda")
        x1 = torch.randn(TILE_M * 2, N_STATIC, dtype=torch.float16, device="cuda")
        y1 = torch.full((TILE_M * 2, N_STATIC), TRAP, dtype=torch.float16, device="cuda")
        ops = (
            DualPageOp.schedule(x=x0, y=y0, tile_sizes={"M": TILE_M})
            + _AddTwoOp.schedule(x=x1, y=y1, tile_sizes={"M": TILE_M})
        )
        Megakernel(ops, config=MegakernelConfig(num_pages=4, page_size=16384)).run()
        torch.testing.assert_close(y0, x0 + 1.0, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(y1, x1 + 2.0, atol=1e-3, rtol=1e-3)

    def test_second_page_can_remain_live_until_store(self):
        """Store reads page1 directly; releasing page1 after compute would race."""
        x, y = self._run(
            TILE_M * 6,
            op_cls=StoreReadsSecondPageOp,
            config=MegakernelConfig(num_pages=3, page_size=16384),
        )
        torch.testing.assert_close(y, x + 3.0, atol=1e-3, rtol=1e-3)

    def test_store_step_can_read_second_page(self):
        """Stepped store copies page1 to page0, then TMA-stores in a later step."""
        x, y = self._run(
            TILE_M * 4,
            op_cls=StoreStepReadsSecondPageOp,
            config=MegakernelConfig(num_pages=3, page_size=16384),
        )
        torch.testing.assert_close(y, x + 3.0, atol=1e-3, rtol=1e-3)

    def test_requires_enough_pages(self):
        """An op requesting more pages than fit must raise, not silently corrupt."""
        torch.manual_seed(2)
        x = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.full((TILE_M, N_STATIC), TRAP, dtype=torch.float16, device="cuda")
        ops = DualPageOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        with pytest.raises(ValueError, match="requests 2"):
            Megakernel(ops, config=MegakernelConfig(num_pages=1)).compile()

    def test_rejects_multipage_no_smem_page_contract(self):
        torch.manual_seed(3)
        x = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.full((TILE_M, N_STATIC), TRAP, dtype=torch.float16, device="cuda")
        ops = _NoPageMultiPageOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        with pytest.raises(ValueError, match="uses_smem_page=False"):
            Megakernel(ops, config=MegakernelConfig(num_pages=2))

    def test_rejects_release_mask_outside_requested_pages(self):
        torch.manual_seed(4)
        x = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.full((TILE_M, N_STATIC), TRAP, dtype=torch.float16, device="cuda")
        ops = _BadReleaseMaskOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        with pytest.raises(ValueError, match="outside requested_page_count"):
            Megakernel(ops, config=MegakernelConfig(num_pages=2))
