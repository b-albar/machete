# Copyright (c) 2025, Machete Authors
"""
Two-op megakernel test: MulTwo (x*2) -> AddTwo (+2) with async cp.async.bulk DMA.

Each op uses three phases with CuTe copy API:
    load:    async cp.async.bulk G->S via cute.copy (signals work_mbar)
    compute: element-wise in shared memory
    store:   cp.async.bulk S->G via cute.copy

Dependency (MulTwoOp writes y, AddTwoOp reads y) is auto-detected.
"""

import pytest
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkG2SOp,
    CopyBulkS2GOp,
    group_bulk_copy_modes,
)
from machete.megakernel.megakernel import Megakernel
from machete.megakernel.ops import Op
from machete.megakernel.interpreter import mbarrier_arrive_expect_tx
from machete.utils.testing import is_hopper_available

requires_hopper = pytest.mark.skipif(
    not is_hopper_available(), reason="Requires Hopper (SM90+) GPU",
)

TILE_ELEMS = 32
ELEM_BYTES = 2  # fp16 / bf16
NUM_BITS = TILE_ELEMS * 16  # bits per tile for copy atom


# -- Ops -----------------------------------------------------------------------

class MulTwoOp(Op):
    reads = {"x": (None, ("M",))}
    writes = {"y": (None, ("M",))}
    tile = (("M", TILE_ELEMS),)

    @staticmethod
    def load(page_ptr, tile_m, tile_n, tile_l, op_config_ptr, work_mbar,
             x=None, x_dtype=None, tile_size_M=0):
        nbytes = Int32(tile_size_M * ELEM_BYTES)

        # CuTe copy atom for bulk G->S
        g2s = cute.make_copy_atom(CopyBulkG2SOp(), x_dtype, num_bits_per_copy=NUM_BITS)

        # Smem tile in page
        s_tile = cute.make_tensor(
            cute.make_ptr(x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((tile_size_M,)),
        )
        # Gmem tile at offset
        g_tile = cute.make_tensor(
            x.iterator + tile_m * Int32(tile_size_M),
            cute.make_layout((tile_size_M,)),
        )

        # Signal work_mbar with expected tx bytes + issue async copy
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        mbarrier_arrive_expect_tx(work_mbar, nbytes)
        gsrc, sdst = group_bulk_copy_modes(g_tile, s_tile)
        cute.copy(g2s, gsrc, sdst, mbar_ptr=mbar_ptr)
        # Returns immediately — MMA warps wait on work_mbar before compute

    @staticmethod
    def compute(page_ptr, tile_m, tile_n, tile_l, op_config_ptr,
                x_dtype=None, tile_size_M=0, tidx=0, num_threads=0):
        s = cute.make_tensor(
            cute.make_ptr(x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((tile_size_M,)),
        )
        for i in range(tidx, tile_size_M, num_threads):
            val = s[i]
            s[i] = val + val

    @staticmethod
    def store(page_ptr, tile_m, tile_n, tile_l, op_config_ptr,
              y=None, y_dtype=None, tile_size_M=0):
        nbytes = Int32(tile_size_M * ELEM_BYTES)
        s2g = cute.make_copy_atom(CopyBulkS2GOp(), y_dtype, num_bits_per_copy=NUM_BITS)

        s_tile = cute.make_tensor(
            cute.make_ptr(y_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((tile_size_M,)),
        )
        g_tile = cute.make_tensor(
            y.iterator + tile_m * Int32(tile_size_M),
            cute.make_layout((tile_size_M,)),
        )

        ssrc, gdst = group_bulk_copy_modes(s_tile, g_tile)
        cute.copy(s2g, ssrc, gdst)


class AddTwoOp(Op):
    reads = {"a": (None, ("M",))}
    writes = {"b": (None, ("M",))}
    tile = (("M", TILE_ELEMS),)

    @staticmethod
    def load(page_ptr, tile_m, tile_n, tile_l, op_config_ptr, work_mbar,
             a=None, a_dtype=None, tile_size_M=0):
        nbytes = Int32(tile_size_M * ELEM_BYTES)
        g2s = cute.make_copy_atom(CopyBulkG2SOp(), a_dtype, num_bits_per_copy=NUM_BITS)

        s_tile = cute.make_tensor(
            cute.make_ptr(a_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((tile_size_M,)),
        )
        g_tile = cute.make_tensor(
            a.iterator + tile_m * Int32(tile_size_M),
            cute.make_layout((tile_size_M,)),
        )

        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        mbarrier_arrive_expect_tx(work_mbar, nbytes)
        gsrc, sdst = group_bulk_copy_modes(g_tile, s_tile)
        cute.copy(g2s, gsrc, sdst, mbar_ptr=mbar_ptr)

    @staticmethod
    def compute(page_ptr, tile_m, tile_n, tile_l, op_config_ptr,
                a_dtype=None, tile_size_M=0, tidx=0, num_threads=0):
        s = cute.make_tensor(
            cute.make_ptr(a_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((tile_size_M,)),
        )
        two = a_dtype(2.0)
        for i in range(tidx, tile_size_M, num_threads):
            s[i] = s[i] + two

    @staticmethod
    def store(page_ptr, tile_m, tile_n, tile_l, op_config_ptr,
              b=None, b_dtype=None, tile_size_M=0):
        nbytes = Int32(tile_size_M * ELEM_BYTES)
        s2g = cute.make_copy_atom(CopyBulkS2GOp(), b_dtype, num_bits_per_copy=NUM_BITS)

        s_tile = cute.make_tensor(
            cute.make_ptr(b_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((tile_size_M,)),
        )
        g_tile = cute.make_tensor(
            b.iterator + tile_m * Int32(tile_size_M),
            cute.make_layout((tile_size_M,)),
        )

        ssrc, gdst = group_bulk_copy_modes(s_tile, g_tile)
        cute.copy(s2g, ssrc, gdst)


# -- Tests ---------------------------------------------------------------------

@requires_hopper
class TestSmemPipelinedOps:

    def test_mul_two(self):
        x = torch.randn(1024, dtype=torch.float16, device="cuda")
        y = torch.empty_like(x)
        Megakernel([MulTwoOp.schedule(x=x, y=y)]).run()
        torch.testing.assert_close(y, x * 2, atol=1e-3, rtol=1e-3)

    def test_add_two(self):
        a = torch.randn(1024, dtype=torch.float16, device="cuda")
        b = torch.empty_like(a)
        Megakernel([AddTwoOp.schedule(a=a, b=b)]).run()
        torch.testing.assert_close(b, a + 2, atol=1e-3, rtol=1e-3)

    def test_mul_then_add(self):
        x = torch.randn(1024, dtype=torch.float16, device="cuda")
        y = torch.empty_like(x)
        z = torch.empty_like(x)
        ops = [MulTwoOp.schedule(x=x, y=y), AddTwoOp.schedule(a=y, b=z)]
        Megakernel(ops).run()
        torch.testing.assert_close(z, x * 2 + 2, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mul_then_add_dtypes(self, dtype):
        x = torch.randn(1024, dtype=dtype, device="cuda")
        y = torch.empty_like(x)
        z = torch.empty_like(x)
        ops = [MulTwoOp.schedule(x=x, y=y), AddTwoOp.schedule(a=y, b=z)]
        Megakernel(ops).run()
        tol = dict(atol=5e-2, rtol=5e-2) if dtype == torch.bfloat16 else dict(atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(z, x * 2 + 2, **tol)
