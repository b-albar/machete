# Copyright (c) 2025, Machete Authors
"""
RMSNorm Op for the Megakernel.

Applies Root Mean Square Layer Normalization:
    y = x / sqrt(mean(x²) + eps) * weight

Optimized for high throughput with adaptive parallelism:
- Small/medium D (< 4096): Warp-parallel (each warp processes one row)
- Large D (>= 4096): Block-parallel (all threads cooperate on one row)

Both modes use:
- Vectorized memory access via CuTe local_partition + autovec_copy
- Minimal synchronization (warp-only or block-level as needed)

Configuration:
    - threads_per_row: from MegakernelConfig.threads_per_block (compile-time constant)
    - tile_size_M: from tile declaration, e.g., tile = (("M", 4),) for 4 rows/tile

Forward:
    rstd = 1 / sqrt(mean(x²) + eps)
    y = x * rstd * weight

Usage:
    from machete.kernels.rms_norm import RMSNormOp
    from machete.megakernel import Megakernel

    x_2d = x.view(-1, D).contiguous()
    ops = [RMSNormOp.schedule(x=x_2d, weight=w, y=y)]
    kernel = Megakernel(ops)
    kernel.run()
"""

import operator

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32

from machete.megakernel.ops import Op
from machete.kernels.utils.reduce import block_reduce


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

    Optimized with warp-parallel row processing and shared memory weight caching:
    - Each warp processes a separate row independently
    - Weight vector cached in shared memory (loaded once per block)
    - Vectorized memory access via CuTe local_partition + autovec_copy
    - Warp-level reduction only (much faster than block reduction)

    Tensor declarations:
        x:      (M, D)  — input tensor, float32
        weight: (D,)    — per-element scale, float32
        y:      (M, D)  — output tensor, float32

    Tiling:
        tile_m indexes row groups (ceil(M / tile_size_M) tiles).
        Each tile processes num_warps rows in parallel (one per warp).

    Kernel config (from MegakernelConfig):
        threads_per_row: Number of threads per block
        tile_size_M: Should match num_warps for optimal performance
    """

    # dtype=None means infer from tensor at schedule time (supports bf16/fp16/fp32)
    reads = {
        "x": (None, "M, D"),
        "weight": (None, "D"),
    }
    writes = {"y": (None, "M, D")}
    tile = (("M", 4),)  # Process 4 rows per tile (matches 4 warps = 128 threads)

    backward_reads = {
        "dout": (None, "M, D"),
        "x": (None, "M, D"),
        "weight": (None, "D"),
    }
    backward_writes = {"dx": (None, "M, D")}

    # --- Forward (compute phase) ---

    @staticmethod
    def compute(
        page_ptr: Int32,
        tile_m: Int32,
        tile_n: Int32,
        tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """RMSNorm forward with adaptive parallelism.

        For large D (>= 4096): Block-parallel mode
        - All threads cooperate on one row at a time
        - Block-level reduction for sum of squares
        - Rows processed serially within tile

        For small/medium D (< 4096): Warp-parallel mode
        - Each warp processes a separate row independently
        - Warp-level reduction only (no barriers)
        - Maximum parallelism within tile

        Uses compile-time constants from init_source:
            threads_per_row: from MegakernelConfig.threads_per_block
            tile_size_M: from tile declaration
            D: hidden dimension (static)
        """
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = threads_per_row // 32

        row_start = tile_m * tile_size_M

        # Branch resolved at trace time since D is compile-time constant
        if cutlass.const_expr(D >= 4096):
            # === Block-parallel mode: all threads cooperate on one row ===
            # Better for large D: D/threads_per_row elements per thread
            # For D=4096, 128 threads: 32 elements per thread (vs 128 for warp-parallel)

            thr_layout = cute.make_layout(threads_per_row)

            # Allocate reduction buffer in shared memory: (1, num_warps)
            reduction_buffer = cute.make_tensor(
                cute.make_ptr(Float32, page_ptr, cute.AddressSpace.smem),
                cute.make_layout((1, num_warps)),
            )

            # Load weight into registers (all threads cooperate)
            w_row = cute.make_tensor(
                cute.make_ptr(x_dtype, weight_ptr_raw, cute.AddressSpace.gmem),
                cute.make_layout(D),
            )
            w_part = cute.local_partition(w_row, thr_layout, tidx)
            w_reg = cute.make_fragment_like(w_part)
            cute.autovec_copy(w_part, w_reg)

            # Process rows serially (tile_size_M is compile-time constant)
            for local_row in range(tile_size_M):
                row_idx = row_start + local_row

                if row_idx < M:
                    # Load x row (all threads cooperate)
                    x_row = cute.make_tensor(
                        cute.make_ptr(x_dtype, x_ptr_raw, cute.AddressSpace.gmem) + row_idx * D,
                        cute.make_layout(D),
                    )
                    y_row = cute.make_tensor(
                        cute.make_ptr(x_dtype, y_ptr_raw, cute.AddressSpace.gmem) + row_idx * D,
                        cute.make_layout(D),
                    )

                    x_part = cute.local_partition(x_row, thr_layout, tidx)
                    y_part = cute.local_partition(y_row, thr_layout, tidx)

                    # Load x into registers
                    x_reg = cute.make_fragment_like(x_part)
                    cute.autovec_copy(x_part, x_reg)

                    # Local reduction: sum of x² (accumulate in fp32 for precision)
                    partial_sq = Float32(0.0)
                    for i in range(cute.size(x_reg)):
                        val = x_reg[i].to(Float32)
                        partial_sq = partial_sq + val * val

                    # Warp reduction first
                    warp_sum = cute.arch.warp_reduction(partial_sq, operator.add)

                    # Block reduction across warps
                    sum_sq = block_reduce(warp_sum, operator.add, reduction_buffer, Float32(0.0))

                    # rstd = 1 / sqrt(mean(x²) + eps)
                    rstd = cute.math.rsqrt(sum_sq / D + RMSNORM_EPS, fastmath=True)

                    # Normalize: y = x * rstd * weight (compute in fp32, store in input dtype)
                    y_reg = cute.make_fragment_like(x_reg)
                    for i in range(cute.size(x_reg)):
                        val = x_reg[i].to(Float32) * rstd * w_reg[i].to(Float32)
                        y_reg[i] = val.to(x_dtype)

                    # Store result
                    cute.autovec_copy(y_reg, y_part)

        elif cutlass.const_expr(D >= 32):
            # === Warp-parallel mode: each warp processes one row ===
            # Better for small/medium D: no cross-warp synchronization

            thr_layout = cute.make_layout(32)  # 32 threads per warp

            # Load weight into registers (outside dynamic if for CuTe tracing)
            w_row = cute.make_tensor(
                cute.make_ptr(x_dtype, weight_ptr_raw, cute.AddressSpace.gmem),
                cute.make_layout(D),
            )
            w_part = cute.local_partition(w_row, thr_layout, lane_idx)
            w_reg = cute.make_fragment_like(w_part)
            cute.autovec_copy(w_part, w_reg)

            # Each warp processes one row (warp_idx determines which row)
            row_idx = row_start + warp_idx

            if warp_idx < tile_size_M and row_idx < M:
                # Load x row
                x_row = cute.make_tensor(
                    cute.make_ptr(x_dtype, x_ptr_raw, cute.AddressSpace.gmem) + row_idx * D,
                    cute.make_layout(D),
                )
                y_row = cute.make_tensor(
                    cute.make_ptr(x_dtype, y_ptr_raw, cute.AddressSpace.gmem) + row_idx * D,
                    cute.make_layout(D),
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
                rstd = cute.math.rsqrt(sum_sq / D + RMSNORM_EPS, fastmath=True)

                # Normalize: y = x * rstd * weight (compute in fp32, store in input dtype)
                y_reg = cute.make_fragment_like(x_reg)
                for i in range(cute.size(x_reg)):
                    val = x_reg[i].to(Float32) * rstd * w_reg[i].to(Float32)
                    y_reg[i] = val.to(x_dtype)

                # Store result (vectorized 128-bit stores)
                cute.autovec_copy(y_reg, y_part)

        else:
            # === Scalar path: D < 32 (small hidden dim) ===
            row_idx = row_start + warp_idx

            if warp_idx < tile_size_M and row_idx < M:
                row_offset = row_idx * D

                # Pass 1: sum of squares (accumulate in fp32)
                partial_sq = Float32(0.0)
                for i in range(lane_idx, D, 32):
                    val = x[row_offset + i].to(Float32)
                    partial_sq = partial_sq + val * val

                sum_sq = cute.arch.warp_reduction(partial_sq, operator.add)
                rstd = cute.math.rsqrt(sum_sq / D + RMSNORM_EPS, fastmath=True)

                # Pass 2: normalize and apply weight (compute in fp32, store in input dtype)
                for i in range(lane_idx, D, 32):
                    val = x[row_offset + i].to(Float32)
                    w = weight[i].to(Float32)
                    y[row_offset + i] = (val * rstd * w).to(x_dtype)

    # --- Backward (compute phase) ---

    @staticmethod
    def backward_compute(
        page_ptr: Int32,
        tile_m: Int32,
        tile_n: Int32,
        tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """RMSNorm backward with warp-parallel row processing.

        dx = (dout * weight - x * rstd² * mean(dout * weight * x)) * rstd

        Each warp processes a separate row independently.
        """
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()

        row_start = tile_m * tile_size_M
        row_idx = row_start + warp_idx

        if warp_idx < tile_size_M and row_idx < M:
            if cutlass.const_expr(D >= 32):
                # === Vectorized path ===
                thr_layout = cute.make_layout(32)

                # Load weight into registers
                w_row = cute.make_tensor(
                    cute.make_ptr(dout_dtype, weight_ptr_raw, cute.AddressSpace.gmem),
                    cute.make_layout(D),
                )
                w_part = cute.local_partition(w_row, thr_layout, lane_idx)
                w_reg = cute.make_fragment_like(w_part)
                cute.autovec_copy(w_part, w_reg)

                # Load x row
                x_row = cute.make_tensor(
                    cute.make_ptr(dout_dtype, x_ptr_raw, cute.AddressSpace.gmem) + row_idx * D,
                    cute.make_layout(D),
                )
                dout_row = cute.make_tensor(
                    cute.make_ptr(dout_dtype, dout_ptr_raw, cute.AddressSpace.gmem) + row_idx * D,
                    cute.make_layout(D),
                )
                dx_row = cute.make_tensor(
                    cute.make_ptr(dout_dtype, dx_ptr_raw, cute.AddressSpace.gmem) + row_idx * D,
                    cute.make_layout(D),
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
                rstd = cute.math.rsqrt(sum_sq / D + RMSNORM_EPS, fastmath=True)

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
                mean_grad = sum_grad / D

                # Pass 3: dx = (dout * w - x * rstd² * mean_grad) * rstd
                dx_reg = cute.make_fragment_like(x_reg)
                for i in range(cute.size(x_reg)):
                    d = dout_reg[i].to(Float32)
                    w = w_reg[i].to(Float32)
                    x_val = x_reg[i].to(Float32)
                    dw_x = d * w
                    result = (dw_x - x_val * rstd * rstd * mean_grad) * rstd
                    dx_reg[i] = result.to(dout_dtype)

                # Store dx
                cute.autovec_copy(dx_reg, dx_part)

            else:
                # === Scalar path ===
                row_offset = row_idx * D

                # Pass 1: Compute rstd (accumulate in fp32)
                partial_sq = Float32(0.0)
                for i in range(lane_idx, D, 32):
                    val = x[row_offset + i].to(Float32)
                    partial_sq = partial_sq + val * val

                sum_sq = cute.arch.warp_reduction(partial_sq, operator.add)
                rstd = cute.math.rsqrt(sum_sq / D + RMSNORM_EPS, fastmath=True)

                # Pass 2: Compute sum(dout * weight * x)
                partial_grad = Float32(0.0)
                for i in range(lane_idx, D, 32):
                    d = dout[row_offset + i].to(Float32)
                    w = weight[i].to(Float32)
                    x_val = x[row_offset + i].to(Float32)
                    partial_grad = partial_grad + d * w * x_val

                sum_grad = cute.arch.warp_reduction(partial_grad, operator.add)
                mean_grad = sum_grad / D

                # Pass 3: dx = (dout * w - x * rstd² * mean_grad) * rstd
                for i in range(lane_idx, D, 32):
                    d = dout[row_offset + i].to(Float32)
                    w = weight[i].to(Float32)
                    x_val = x[row_offset + i].to(Float32)
                    dw_x = d * w
                    result = (dw_x - x_val * rstd * rstd * mean_grad) * rstd
                    dx[row_offset + i] = result.to(dout_dtype)


__all__ = ["RMSNormOp"]
