# Copyright (c) 2025, Machete Authors
"""
Mbarrier + cp.async.bulk verification tests.

Tests (incremental complexity):
1. mbarrier round-trip: all threads write → mbarrier → read
2. mbarrier producer/consumer: thread 0 writes → mbarrier → all read
3. cp.async.bulk G2S: bulk copy gmem→smem via mbarrier, all threads read
4. cp.async.bulk G2S + S2G round-trip: bulk copy in → smem → bulk copy out
5. cp.async.bulk G2S + compute + S2G: bulk load → scale ×2 → bulk store
"""

import pytest
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkG2SOp,
    CopyBulkS2GOp,
    group_bulk_copy_modes,
)
import cuda.bindings.driver as cuda

from machete.utils.testing import is_hopper_available

THREADS = 128
TILE_SIZE = 512  # elements for bulk copy tests

requires_hopper = pytest.mark.skipif(
    not is_hopper_available(),
    reason="Requires Hopper (SM90+) GPU",
)


# =============================================================================
# Test 1: Basic mbarrier round-trip
# =============================================================================


class MbarrierRoundTrip:
    """All threads write to smem, mbarrier sync, all threads read back."""

    @cute.jit
    def __call__(self, out: cute.Tensor, stream: cuda.CUstream):
        self.kernel(out).launch(
            grid=[1, 1, 1],
            block=[THREADS, 1, 1],
            smem=2048,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, out: cute.Tensor):
        tidx = cute.arch.thread_idx()[0]

        smem = cutlass.utils.SmemAllocator()
        mbar = smem.allocate(cutlass.Int64)
        sdata = smem.allocate_tensor(
            cutlass.Float16,
            cute.make_layout((THREADS,)),
        )

        if tidx == 0:
            cute.arch.mbarrier_init(mbar, THREADS)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        sdata[tidx] = cutlass.Float16(tidx + 1)
        cute.arch.mbarrier_arrive(mbar)
        cute.arch.mbarrier_wait(mbar, 0)

        out[tidx] = sdata[tidx]


# =============================================================================
# Test 2: Producer (thread 0) / consumer (all threads)
# =============================================================================


class MbarrierProducerConsumer:
    """Thread 0 doubles input → smem → mbarrier → all threads read."""

    @cute.jit
    def __call__(self, inp: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        self.kernel(inp, out).launch(
            grid=[1, 1, 1],
            block=[THREADS, 1, 1],
            smem=2048,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, inp: cute.Tensor, out: cute.Tensor):
        tidx = cute.arch.thread_idx()[0]

        smem = cutlass.utils.SmemAllocator()
        mbar = smem.allocate(cutlass.Int64)
        sdata = smem.allocate_tensor(
            cutlass.Float16,
            cute.make_layout((THREADS,)),
        )

        if tidx == 0:
            cute.arch.mbarrier_init(mbar, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if tidx == 0:
            for i in range(THREADS):
                val = inp[i]
                sdata[i] = val + val
            cute.arch.mbarrier_arrive(mbar)

        cute.arch.mbarrier_wait(mbar, 0)
        out[tidx] = sdata[tidx]


# =============================================================================
# Test 3: cp.async.bulk G2S load + mbarrier
# Thread 0 issues bulk copy gmem→smem, all threads read after mbarrier
# =============================================================================


class BulkCopyG2S:
    """cp.async.bulk G2S: thread 0 bulk-copies tile from gmem→smem."""

    def __init__(self):
        self.dtype = cutlass.Float16
        # Total bits for bulk copy: TILE_SIZE elements × 16 bits
        self.num_bits = TILE_SIZE * 16

    @cute.jit
    def __call__(self, inp: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        # Create bulk copy atom in JIT (host) context
        bulk_g2s_atom = cute.make_copy_atom(
            CopyBulkG2SOp(), self.dtype, num_bits_per_copy=self.num_bits,
        )

        self.kernel(inp, out, bulk_g2s_atom).launch(
            grid=[1, 1, 1],
            block=[THREADS, 1, 1],
            smem=TILE_SIZE * 2 + 128,  # tile bytes + mbarrier
            stream=stream,
        )

    @cute.kernel
    def kernel(self, inp: cute.Tensor, out: cute.Tensor, bulk_atom: cute.CopyAtom):
        tidx = cute.arch.thread_idx()[0]

        smem = cutlass.utils.SmemAllocator()
        mbar = smem.allocate(cutlass.Int64)
        sdata = smem.allocate_tensor(
            self.dtype,
            cute.make_layout((TILE_SIZE,)),
            byte_alignment=128,
        )

        # Init mbarrier: arrive count = 1, expect num_bytes transaction
        num_bytes = TILE_SIZE * 2  # fp16 = 2 bytes
        if tidx == 0:
            cute.arch.mbarrier_init(mbar, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, num_bytes)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # Create gmem view for the input tile
        gdata = cute.make_tensor(
            inp.iterator,
            cute.make_layout((TILE_SIZE,)),
        )

        # Thread 0: issue cp.async.bulk G2S
        if tidx == 0:
            gsrc, sdst = group_bulk_copy_modes(gdata, sdata)
            cute.copy(bulk_atom, gsrc, sdst, mbar_ptr=mbar)

        # All threads wait for bulk copy to complete
        cute.arch.mbarrier_wait(mbar, 0)

        # Each thread reads from smem → global output
        elems_per_thread = TILE_SIZE // THREADS
        for i in range(elems_per_thread):
            idx = tidx + cutlass.Int32(i * THREADS)
            out[idx] = sdata[idx]


# =============================================================================
# Test 4: cp.async.bulk G2S + S2G round-trip
# =============================================================================


class BulkCopyRoundTrip:
    """cp.async.bulk G2S → S2G: copy in, copy out."""

    def __init__(self):
        self.dtype = cutlass.Float16
        self.num_bits = TILE_SIZE * 16

    @cute.jit
    def __call__(self, inp: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        bulk_g2s_atom = cute.make_copy_atom(
            CopyBulkG2SOp(), self.dtype, num_bits_per_copy=self.num_bits,
        )
        bulk_s2g_atom = cute.make_copy_atom(
            CopyBulkS2GOp(), self.dtype, num_bits_per_copy=self.num_bits,
        )

        self.kernel(inp, out, bulk_g2s_atom, bulk_s2g_atom).launch(
            grid=[1, 1, 1],
            block=[THREADS, 1, 1],
            smem=TILE_SIZE * 2 + 128,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        inp: cute.Tensor,
        out: cute.Tensor,
        g2s_atom: cute.CopyAtom,
        s2g_atom: cute.CopyAtom,
    ):
        tidx = cute.arch.thread_idx()[0]

        smem = cutlass.utils.SmemAllocator()
        mbar = smem.allocate(cutlass.Int64)
        sdata = smem.allocate_tensor(
            self.dtype,
            cute.make_layout((TILE_SIZE,)),
            byte_alignment=128,
        )

        num_bytes = TILE_SIZE * 2
        if tidx == 0:
            cute.arch.mbarrier_init(mbar, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, num_bytes)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        gA = cute.make_tensor(inp.iterator, cute.make_layout((TILE_SIZE,)))
        gC = cute.make_tensor(out.iterator, cute.make_layout((TILE_SIZE,)))

        # G2S bulk load
        if tidx == 0:
            gsrc, sdst = group_bulk_copy_modes(gA, sdata)
            cute.copy(g2s_atom, gsrc, sdst, mbar_ptr=mbar)

        cute.arch.mbarrier_wait(mbar, 0)
        cute.arch.barrier()

        # S2G bulk store (thread 0)
        if tidx == 0:
            ssrc, gdst = group_bulk_copy_modes(sdata, gC)
            cute.copy(s2g_atom, ssrc, gdst)

        # Wait for store
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        cute.arch.barrier()


# =============================================================================
# Test 5: cp.async.bulk G2S + compute + S2G
# =============================================================================


class BulkCopyComputeStore:
    """cp.async.bulk G2S → scale×2 in smem → cp.async.bulk S2G."""

    def __init__(self):
        self.dtype = cutlass.Float16
        self.num_bits = TILE_SIZE * 16

    @cute.jit
    def __call__(self, inp: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        bulk_g2s_atom = cute.make_copy_atom(
            CopyBulkG2SOp(), self.dtype, num_bits_per_copy=self.num_bits,
        )
        bulk_s2g_atom = cute.make_copy_atom(
            CopyBulkS2GOp(), self.dtype, num_bits_per_copy=self.num_bits,
        )

        self.kernel(inp, out, bulk_g2s_atom, bulk_s2g_atom).launch(
            grid=[1, 1, 1],
            block=[THREADS, 1, 1],
            smem=TILE_SIZE * 2 + 128,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        inp: cute.Tensor,
        out: cute.Tensor,
        g2s_atom: cute.CopyAtom,
        s2g_atom: cute.CopyAtom,
    ):
        tidx = cute.arch.thread_idx()[0]

        smem = cutlass.utils.SmemAllocator()
        mbar = smem.allocate(cutlass.Int64)
        sdata = smem.allocate_tensor(
            self.dtype,
            cute.make_layout((TILE_SIZE,)),
            byte_alignment=128,
        )

        num_bytes = TILE_SIZE * 2
        if tidx == 0:
            cute.arch.mbarrier_init(mbar, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, num_bytes)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        gA = cute.make_tensor(inp.iterator, cute.make_layout((TILE_SIZE,)))
        gC = cute.make_tensor(out.iterator, cute.make_layout((TILE_SIZE,)))

        # G2S bulk load
        if tidx == 0:
            gsrc, sdst = group_bulk_copy_modes(gA, sdata)
            cute.copy(g2s_atom, gsrc, sdst, mbar_ptr=mbar)

        cute.arch.mbarrier_wait(mbar, 0)
        cute.arch.barrier()

        # Compute: all threads scale smem values by 2
        elems_per_thread = TILE_SIZE // THREADS
        for i in range(elems_per_thread):
            idx = tidx + cutlass.Int32(i * THREADS)
            val = sdata[idx]
            sdata[idx] = val + val

        cute.arch.barrier()

        # S2G bulk store (thread 0)
        if tidx == 0:
            ssrc, gdst = group_bulk_copy_modes(sdata, gC)
            cute.copy(s2g_atom, ssrc, gdst)

        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        cute.arch.barrier()


# =============================================================================
# Test Functions
# =============================================================================


def _make_stream():
    torch_stream = torch.cuda.Stream()
    cu_stream = cuda.CUstream(torch_stream.cuda_stream)
    return torch_stream, cu_stream


@requires_hopper
def test_mbarrier_round_trip():
    """All threads write sequential values, mbarrier sync, read back."""
    output = torch.zeros(THREADS, dtype=torch.float16, device="cuda")
    out_ct = from_dlpack(output, assumed_align=16)
    _, cu_stream = _make_stream()

    kernel = MbarrierRoundTrip()
    kernel(out_ct, cu_stream)
    torch.cuda.synchronize()

    expected = torch.arange(1, THREADS + 1, dtype=torch.float16, device="cuda")
    diff = (output - expected).abs().max().item()
    print(f"mbarrier round-trip: max diff = {diff}")
    assert diff < 1e-5, f"mbarrier round-trip failed: max diff {diff}"


@requires_hopper
def test_mbarrier_producer_consumer():
    """Thread 0 doubles input in smem, all threads read result via mbarrier."""
    inp = torch.randn(THREADS, dtype=torch.float16, device="cuda")
    output = torch.zeros(THREADS, dtype=torch.float16, device="cuda")

    inp_ct = from_dlpack(inp, assumed_align=16)
    out_ct = from_dlpack(output, assumed_align=16)
    _, cu_stream = _make_stream()

    kernel = MbarrierProducerConsumer()
    kernel(inp_ct, out_ct, cu_stream)
    torch.cuda.synchronize()

    expected = inp * 2.0
    diff = (output - expected).abs().max().item()
    print(f"mbarrier producer/consumer: max diff = {diff}")
    assert diff < 1e-3, f"mbarrier producer/consumer failed: max diff {diff}"


@requires_hopper
def test_bulk_copy_g2s():
    """cp.async.bulk G2S: thread 0 bulk-copies tile, all threads read."""
    inp = torch.randn(TILE_SIZE, dtype=torch.float16, device="cuda")
    output = torch.zeros(TILE_SIZE, dtype=torch.float16, device="cuda")

    inp_ct = from_dlpack(inp, assumed_align=16)
    out_ct = from_dlpack(output, assumed_align=16)
    _, cu_stream = _make_stream()

    kernel = BulkCopyG2S()
    kernel(inp_ct, out_ct, cu_stream)
    torch.cuda.synchronize()

    diff = (output - inp).abs().max().item()
    print(f"bulk G2S: max diff = {diff}")
    assert diff < 1e-5, f"bulk G2S failed: max diff {diff}"


@requires_hopper
def test_bulk_copy_round_trip():
    """cp.async.bulk G2S → S2G round-trip."""
    inp = torch.randn(TILE_SIZE, dtype=torch.float16, device="cuda")
    output = torch.zeros(TILE_SIZE, dtype=torch.float16, device="cuda")

    inp_ct = from_dlpack(inp, assumed_align=16)
    out_ct = from_dlpack(output, assumed_align=16)
    _, cu_stream = _make_stream()

    kernel = BulkCopyRoundTrip()
    kernel(inp_ct, out_ct, cu_stream)
    torch.cuda.synchronize()

    diff = (output - inp).abs().max().item()
    print(f"bulk round-trip: max diff = {diff}")
    assert diff < 1e-5, f"bulk round-trip failed: max diff {diff}"


@requires_hopper
def test_bulk_copy_compute_store():
    """cp.async.bulk G2S → scale×2 → S2G."""
    inp = torch.randn(TILE_SIZE, dtype=torch.float16, device="cuda")
    output = torch.zeros(TILE_SIZE, dtype=torch.float16, device="cuda")

    inp_ct = from_dlpack(inp, assumed_align=16)
    out_ct = from_dlpack(output, assumed_align=16)
    _, cu_stream = _make_stream()

    kernel = BulkCopyComputeStore()
    kernel(inp_ct, out_ct, cu_stream)
    torch.cuda.synchronize()

    expected = inp * 2.0
    diff = (output - expected).abs().max().item()
    print(f"bulk G2S+compute+S2G: max diff = {diff}")
    assert diff < 1e-3, f"bulk G2S+compute+S2G failed: max diff {diff}"
