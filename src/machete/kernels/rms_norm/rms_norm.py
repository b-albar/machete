# Copyright (c) 2025, Machete Authors
"""
RMSNorm Op for the Megakernel.

Applies Root Mean Square Layer Normalization:
    y = x / sqrt(mean(x²) + eps) * weight

Optimized for high throughput with warp-parallel execution:
- D >= 32: Each warp processes one row independently (warp reduction only)
- D < 32: Scalar fallback with lane-strided access

Uses:
- Vectorized memory access via CuTe local_partition + autovec_copy
- Minimal synchronization (warp-only or block-level as needed)

Configuration:
    - threads_per_row: from MegakernelConfig.threads_per_block (compile-time constant)
    - tile_size_M: from tile_sizes at schedule time, e.g., tile_sizes={"M": 4} for 4 rows/tile

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
from cutlass import Float32

from machete.megakernel.ops import Op


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

    Parallelism modes based on hidden dimension D:
    - D >= 32: Warp-parallel (each warp processes one row, warp reduction only)
    - D < 32: Scalar fallback (lane-strided access)

    Both modes use vectorized memory access via CuTe local_partition + autovec_copy
    where applicable, with weight loaded directly to registers from global memory.

    Tensor declarations:
        x:      (M, D)  — input tensor (bf16/fp16/fp32)
        weight: (D,)    — per-element scale (bf16/fp16/fp32)
        y:      (M, D)  — output tensor (bf16/fp16/fp32)

    Tiling:
        tile_M indexes row groups (ceil(M / tile_size_M) tiles).
        Each tile processes tile_size_M rows (default 4, matching 4 warps).
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

    # --- Forward (compute phase) ---

    @cute.jit
    def compute(self, page_ptr, tile_M, tile_D, x, weight, y):
        """RMSNorm forward with adaptive parallelism.

        Warp-parallel mode (D >= 32):
        - Each warp processes a separate row independently
        - Warp-level reduction only (no barriers)
        - Maximum parallelism within tile

        Scalar fallback (D < 32):
        - Lane-strided access within a single warp
        """
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32

        row_start = tile_M * self.tile_size_M

        # Branch resolved at trace time since D is compile-time constant
        if cutlass.const_expr(self.D >= 32):
            # === Warp-parallel mode: each warp processes one row ===
            thr_layout = cute.make_layout(32)  # 32 threads per warp

            # Load weight into registers (outside dynamic if for CuTe tracing)
            w_row = cute.make_tensor(
                weight.iterator,
                cute.make_layout(self.D),
            )
            w_part = cute.local_partition(w_row, thr_layout, lane_idx)
            w_reg = cute.make_fragment_like(w_part)
            cute.autovec_copy(w_part, w_reg)

            # Each warp processes rows in a strided pattern across the tile.
            for local_row in range(warp_idx, self.tile_size_M, num_warps):
                row_idx = row_start + local_row

                if row_idx < self.M:
                    # Load x row
                    x_row = cute.make_tensor(
                        x.iterator + row_idx * self.D,
                        cute.make_layout(self.D),
                    )
                    y_row = cute.make_tensor(
                        y.iterator + row_idx * self.D,
                        cute.make_layout(self.D),
                    )

                    x_part = cute.local_partition(x_row, thr_layout, lane_idx)
                    y_part = cute.local_partition(y_row, thr_layout, lane_idx)

                    # Load x into registers (vectorized 128-bit loads)
                    x_reg = cute.make_fragment_like(x_part)
                    cute.autovec_copy(x_part, x_reg)

                    # Local reduction: sum of x² (accumulate in fp32 for precision)
                    partial_sq = Float32(0.0)
                    for i in range(cute.size(x_reg)):
                        val = x_reg[i].to(Float32)
                        partial_sq = partial_sq + val * val

                    # Warp reduction only (no block reduce needed!)
                    sum_sq = cute.arch.warp_reduction(partial_sq, operator.add)

                    # rstd = 1 / sqrt(mean(x²) + eps)
                    rstd = cute.math.rsqrt(sum_sq / self.D + RMSNORM_EPS, fastmath=True)

                    # Normalize: y = x * rstd * weight (compute in fp32, store in input dtype)
                    y_reg = cute.make_fragment_like(x_reg)
                    for i in range(cute.size(x_reg)):
                        val = x_reg[i].to(Float32) * rstd * w_reg[i].to(Float32)
                        y_reg[i] = val.to(self.x_dtype)

                    # Store result (vectorized 128-bit stores)
                    cute.autovec_copy(y_reg, y_part)

        else:
            # === Scalar path: D < 32 (small hidden dim) ===
            # Flatten 2D tensors for scalar indexing
            x = cute.make_tensor(x.iterator, cute.make_layout(self.M * self.D))
            y = cute.make_tensor(y.iterator, cute.make_layout(self.M * self.D))

            for local_row in range(warp_idx, self.tile_size_M, num_warps):
                row_idx = row_start + local_row

                if row_idx < self.M:
                    row_offset = row_idx * self.D

                    # Pass 1: sum of squares (accumulate in fp32)
                    partial_sq = Float32(0.0)
                    for i in range(lane_idx, self.D, 32):
                        val = x[row_offset + i].to(Float32)
                        partial_sq = partial_sq + val * val

                    sum_sq = cute.arch.warp_reduction(partial_sq, operator.add)
                    rstd = cute.math.rsqrt(sum_sq / self.D + RMSNORM_EPS, fastmath=True)

                    # Pass 2: normalize and apply weight (compute in fp32, store in input dtype)
                    for i in range(lane_idx, self.D, 32):
                        val = x[row_offset + i].to(Float32)
                        w = weight[i].to(Float32)
                        y[row_offset + i] = (val * rstd * w).to(self.x_dtype)

    # --- Backward (compute phase) ---

    @cute.jit
    def backward_compute(self, page_ptr, tile_M, tile_D, dout, x, weight, dx):
        """RMSNorm backward with warp-parallel row processing.

        dx = (dout * weight - x * rstd² * mean(dout * weight * x)) * rstd

        Each warp processes a separate row independently.
        """
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32

        row_start = tile_M * self.tile_size_M

        if cutlass.const_expr(self.D >= 32):
            # === Vectorized path ===
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

        else:
            # === Scalar path ===
            # Flatten 2D tensors for scalar indexing
            x = cute.make_tensor(x.iterator, cute.make_layout(self.M * self.D))
            dout = cute.make_tensor(dout.iterator, cute.make_layout(self.M * self.D))
            dx = cute.make_tensor(dx.iterator, cute.make_layout(self.M * self.D))

            for local_row in range(warp_idx, self.tile_size_M, num_warps):
                row_idx = row_start + local_row

                if row_idx < self.M:
                    row_offset = row_idx * self.D

                    # Pass 1: Compute rstd (accumulate in fp32)
                    partial_sq = Float32(0.0)
                    for i in range(lane_idx, self.D, 32):
                        val = x[row_offset + i].to(Float32)
                        partial_sq = partial_sq + val * val

                    sum_sq = cute.arch.warp_reduction(partial_sq, operator.add)
                    rstd = cute.math.rsqrt(sum_sq / self.D + RMSNORM_EPS, fastmath=True)

                    # Pass 2: Compute sum(dout * weight * x)
                    partial_grad = Float32(0.0)
                    for i in range(lane_idx, self.D, 32):
                        d = dout[row_offset + i].to(Float32)
                        w = weight[i].to(Float32)
                        x_val = x[row_offset + i].to(Float32)
                        partial_grad = partial_grad + d * w * x_val

                    sum_grad = cute.arch.warp_reduction(partial_grad, operator.add)
                    mean_grad = sum_grad / self.D

                    # Pass 3: dx = (dout * w - x * rstd² * mean_grad) * rstd
                    for i in range(lane_idx, self.D, 32):
                        d = dout[row_offset + i].to(Float32)
                        w = weight[i].to(Float32)
                        x_val = x[row_offset + i].to(Float32)
                        dw_x = d * w
                        result = (dw_x - x_val * rstd * rstd * mean_grad) * rstd
                        dx[row_offset + i] = result.to(self.dout_dtype)


__all__ = ["RMSNormOp"]
