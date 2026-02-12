# Copyright (c) 2025, Machete Authors
"""
RMSNorm Op for the Megakernel.

Applies Root Mean Square Layer Normalization:
    y = x / sqrt(mean(x²) + eps) * weight

Pipelined load/compute/store with shared memory staging for the forward pass:
    load:    async cp.async.bulk G->S (x tile)
    compute: read x from smem, weight from global->regs, write y to smem
    store:   cp.async.bulk S->G (y tile)

Weight is loaded to registers in compute (small, shared across tiles).
Backward reads from global memory (x+dout don't fit in one page for large D).

Configuration:
    - threads_per_row: from MegakernelConfig.threads_per_block (compile-time constant)
    - tile_size_M: from tile_sizes at schedule time, constrained by PAGE_SIZE

Forward:
    rstd = 1 / sqrt(mean(x²) + eps)
    y = x * rstd * weight

Usage:
    from machete.kernels.rms_norm import RMSNormOp
    from machete.megakernel import Megakernel

    x_2d = x.view(-1, D).contiguous()
    ops = [RMSNormOp.schedule(x=x_2d, weight=w, y=y, tile_sizes={"M": 4})]
    kernel = Megakernel(ops)
    kernel.run()
"""

import operator

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkG2SOp,
    CopyBulkS2GOp,
    group_bulk_copy_modes,
)

from machete.megakernel.ops import Op
from machete.megakernel.interpreter import mbarrier_arrive_expect_tx
from machete.megakernel.paged_memory import PAGE_SIZE


# =============================================================================
# Constants
# =============================================================================

RMSNORM_EPS = 1e-6


# =============================================================================
# RMSNorm Op
# =============================================================================


class RMSNormOp(Op):
    """RMSNorm operation for the megakernel framework.

    Applies Root Mean Square Layer Normalization:
        y = x / sqrt(mean(x²) + eps) * weight

    Forward uses pipelined load/compute/store with shared memory:
    - load: async G->S for x (DMA warp, elect_one)
    - compute: warp-parallel normalization, x from smem, weight from global
    - store: S->G for y (DMA warp, all threads)

    Backward reads from global memory (no pipelining).

    Shared memory layout per page (forward):
        [tile_size_M * D elements]  -- x on load, y after compute

    Tensor declarations:
        x:      (M, D)  -- input tensor (bf16/fp16/fp32)
        weight: (D,)    -- per-element scale (bf16/fp16/fp32)
        y:      (M, D)  -- output tensor (bf16/fp16/fp32)

    Tiling:
        tile_M indexes row groups (ceil(M / tile_size_M) tiles).
        Each tile processes tile_size_M rows (matching warp count).

    Requirements:
        D >= 32 (warp-parallel vectorized access)
        tile_size_M * D * elem_bytes <= PAGE_SIZE (16KB)
    """

    # dtype=None means infer from tensor at schedule time (supports bf16/fp16/fp32)
    reads = {
        "x": (None, ("M", "D")),
        "weight": (None, ("D",)),
    }
    writes = {"y": (None, ("M", "D"))}
    tile = ("M", "D")

    backward_reads = {
        "dout": (None, ("M", "D")),
        "x": (None, ("M", "D")),
        "weight": (None, ("D",)),
    }
    backward_writes = {"dx": (None, ("M", "D"))}

    def __init__(self, **config):
        super().__init__(**config)
        if self.x_dtype == cutlass.Float32:
            self.elem_bytes = 4
        elif self.x_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        self.x_tile_bytes = self.tile_size_M * self.D * self.elem_bytes

        assert self.D >= 32, f"RMSNormOp requires D >= 32, got D={self.D}"
        assert self.x_tile_bytes <= PAGE_SIZE, (
            f"RMSNormOp: tile smem ({self.x_tile_bytes}B) exceeds PAGE_SIZE ({PAGE_SIZE}B). "
            f"Reduce tile_size_M={self.tile_size_M}."
        )

        self.x_nbits_per_row = self.D * self.elem_bytes * 8

    # =========================================================================
    # Forward Load (G->S)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_M, tile_D, x, weight, y, work_mbar):
        """Load x tile from global to shared memory (async G->S).

        Having work_mbar in the signature tells the framework this is an
        async load. The framework wraps non-TMA loads in elect_one(), so
        only one DMA-warp thread executes this body.
        """
        row_start = tile_M * self.tile_size_M

        # Compute total bytes (handle partial M tiles at boundary)
        row_bytes = Int32(self.D * self.elem_bytes)
        total_bytes = Int32(self.tile_size_M) * row_bytes
        remaining = Int32(self.M) - row_start
        if remaining < Int32(self.tile_size_M):
            total_bytes = remaining * row_bytes

        mbar_ptr = cute.make_ptr(
            cutlass.Int64, work_mbar, cute.AddressSpace.smem
        )
        mbarrier_arrive_expect_tx(work_mbar, total_bytes)

        g2s = cute.make_copy_atom(
            CopyBulkG2SOp(), self.x_dtype,
            num_bits_per_copy=self.x_nbits_per_row,
        )

        for local_row in range(self.tile_size_M):
            row_idx = row_start + local_row
            if row_idx < self.M:
                g_x = cute.make_tensor(
                    x.iterator + row_idx * self.D,
                    cute.make_layout((self.D,)),
                )
                s_x = cute.make_tensor(
                    cute.make_ptr(
                        self.x_dtype,
                        page_ptr + Int32(local_row * self.D * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.D,)),
                )
                gsrc, sdst = group_bulk_copy_modes(g_x, s_x)
                cute.copy(g2s, gsrc, sdst, mbar_ptr=mbar_ptr)

    # =========================================================================
    # Forward Compute
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_M, tile_D, x, weight, y):
        """RMSNorm forward: read x from smem, write y to same smem region.

        Weight is loaded from global memory to registers (small, reused
        across all rows in the tile). x and y tensor params are passed
        by the framework (all-or-none) but not used here.
        """
        x_smem = cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem)

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        thr_layout = cute.make_layout(32)

        # Load weight into registers (from global, shared across tiles)
        w_row = cute.make_tensor(
            weight.iterator,
            cute.make_layout(self.D),
        )
        w_part = cute.local_partition(w_row, thr_layout, lane_idx)
        w_reg = cute.make_fragment_like(w_part)
        cute.autovec_copy(w_part, w_reg)

        row_start = tile_M * self.tile_size_M

        # Each warp processes rows in a strided pattern across the tile
        for local_row in range(warp_idx, self.tile_size_M, num_warps):
            row_idx = row_start + local_row

            if row_idx < self.M:
                # Read x from smem
                x_row = cute.make_tensor(
                    x_smem + local_row * self.D,
                    cute.make_layout(self.D),
                )
                x_part = cute.local_partition(x_row, thr_layout, lane_idx)
                x_reg = cute.make_fragment_like(x_part)
                cute.autovec_copy(x_part, x_reg)

                # Sum of squares (accumulate in fp32)
                partial_sq = Float32(0.0)
                for i in range(cute.size(x_reg)):
                    val = x_reg[i].to(Float32)
                    partial_sq = partial_sq + val * val

                # Warp reduction only (no block reduce needed)
                sum_sq = cute.arch.warp_reduction(partial_sq, operator.add)

                # rstd = 1 / sqrt(mean(x²) + eps)
                rstd = cute.math.rsqrt(sum_sq / self.D + RMSNORM_EPS, fastmath=True)

                # y = x * rstd * weight (compute in fp32, store in input dtype)
                y_reg = cute.make_fragment_like(x_reg)
                for i in range(cute.size(x_reg)):
                    val = x_reg[i].to(Float32) * rstd * w_reg[i].to(Float32)
                    y_reg[i] = val.to(self.x_dtype)

                # Write y back to same smem location (x fully consumed into regs)
                y_row = cute.make_tensor(
                    x_smem + local_row * self.D,
                    cute.make_layout(self.D),
                )
                y_part = cute.local_partition(y_row, thr_layout, lane_idx)
                cute.autovec_copy(y_reg, y_part)

    # =========================================================================
    # Forward Store (S->G)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_M, tile_D, x, weight, y):
        """Store y from shared to global memory (S->G).

        Runs on all 32 DMA warp threads (no elect_one).
        """
        s2g = cute.make_copy_atom(
            CopyBulkS2GOp(), self.x_dtype,
            num_bits_per_copy=self.x_nbits_per_row,
        )
        row_start = tile_M * self.tile_size_M

        for local_row in range(self.tile_size_M):
            row_idx = row_start + local_row
            if row_idx < self.M:
                s_y = cute.make_tensor(
                    cute.make_ptr(
                        self.x_dtype,
                        page_ptr + Int32(local_row * self.D * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.D,)),
                )
                g_y = cute.make_tensor(
                    y.iterator + row_idx * self.D,
                    cute.make_layout((self.D,)),
                )
                ssrc, gdst = group_bulk_copy_modes(s_y, g_y)
                cute.copy(s2g, ssrc, gdst)

    # =========================================================================
    # Backward (unchanged — reads from global memory)
    # =========================================================================

    @cute.jit
    def backward_compute(self, page_ptr, tile_M, tile_D, dout, x, weight, dx):
        """RMSNorm backward with warp-parallel row processing.

        dx = (dout * weight - x * rstd² * mean(dout * weight * x)) * rstd

        Each warp processes a separate row independently.
        Reads from global memory (no pipelining — x+dout don't fit in page).
        """
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32

        row_start = tile_M * self.tile_size_M

        # Vectorized path (D >= 32 guaranteed by __init__ assert)
        thr_layout = cute.make_layout(32)

        # Load weight into registers
        w_row = cute.make_tensor(
            weight.iterator,
            cute.make_layout(self.D),
        )
        w_part = cute.local_partition(w_row, thr_layout, lane_idx)
        w_reg = cute.make_fragment_like(w_part)
        cute.autovec_copy(w_part, w_reg)

        for local_row in range(warp_idx, self.tile_size_M, num_warps):
            row_idx = row_start + local_row

            if row_idx < self.M:
                # Load x row
                x_row = cute.make_tensor(
                    x.iterator + row_idx * self.D,
                    cute.make_layout(self.D),
                )
                dout_row = cute.make_tensor(
                    dout.iterator + row_idx * self.D,
                    cute.make_layout(self.D),
                )
                dx_row = cute.make_tensor(
                    dx.iterator + row_idx * self.D,
                    cute.make_layout(self.D),
                )

                x_part = cute.local_partition(x_row, thr_layout, lane_idx)
                dout_part = cute.local_partition(dout_row, thr_layout, lane_idx)
                dx_part = cute.local_partition(dx_row, thr_layout, lane_idx)

                # Load x into registers
                x_reg = cute.make_fragment_like(x_part)
                cute.autovec_copy(x_part, x_reg)

                # Pass 1: Compute rstd (accumulate in fp32)
                partial_sq = Float32(0.0)
                for i in range(cute.size(x_reg)):
                    val = x_reg[i].to(Float32)
                    partial_sq = partial_sq + val * val

                sum_sq = cute.arch.warp_reduction(partial_sq, operator.add)
                rstd = cute.math.rsqrt(sum_sq / self.D + RMSNORM_EPS, fastmath=True)

                # Load dout
                dout_reg = cute.make_fragment_like(dout_part)
                cute.autovec_copy(dout_part, dout_reg)

                # Pass 2: Compute sum(dout * weight * x) (accumulate in fp32)
                partial_grad = Float32(0.0)
                for i in range(cute.size(x_reg)):
                    d = dout_reg[i].to(Float32)
                    w = w_reg[i].to(Float32)
                    x_val = x_reg[i].to(Float32)
                    partial_grad = partial_grad + d * w * x_val

                sum_grad = cute.arch.warp_reduction(partial_grad, operator.add)
                mean_grad = sum_grad / self.D

                # Pass 3: dx = (dout * w - x * rstd² * mean_grad) * rstd
                dx_reg = cute.make_fragment_like(x_reg)
                for i in range(cute.size(x_reg)):
                    d = dout_reg[i].to(Float32)
                    w = w_reg[i].to(Float32)
                    x_val = x_reg[i].to(Float32)
                    dw_x = d * w
                    result = (dw_x - x_val * rstd * rstd * mean_grad) * rstd
                    dx_reg[i] = result.to(self.dout_dtype)

                # Store dx
                cute.autovec_copy(dx_reg, dx_part)


__all__ = ["RMSNormOp"]
