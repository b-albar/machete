# Copyright (c) 2025, Machete Authors
"""
RMSNorm Op for the Megakernel.

Applies Root Mean Square Layer Normalization:
    y = x / sqrt(mean(x²) + eps) * weight

Both forward and backward passes use vectorized memory access
(local_partition + autovec_copy for 128-bit loads/stores when D >= 256)
and hierarchical reduction (warp_reduction + block_reduce).

Multi-row tiling: each tile processes _ROWS_PER_TILE rows, amortizing
persistent kernel dispatch overhead and caching weight in registers.

Forward:
    rstd = 1 / sqrt(mean(x²) + eps)
    y = x * rstd * weight

Backward:
    dx = (dout * weight - x * rstd² * mean(dout * weight * x)) * rstd

Usage:
    from machete.kernels.rms_norm import RMSNormOp
    from machete.megakernel import Megakernel

    x_2d = x.view(-1, D).contiguous()
    ops = [RMSNormOp.schedule(x=x_2d, weight=w, y=y)]
    kernel = Megakernel(ops)
    kernel.run()
"""

import operator

import cutlass.cute as cute
from cutlass import Int32, Int64, Float32

from machete.megakernel.ops import Op
from machete.kernels.utils.reduce import block_reduce


# =============================================================================
# Constants
# =============================================================================

RMSNORM_EPS = 1e-6

# Number of threads per row — must match MegakernelConfig.threads_per_block.
# Compile-time constant enabling CuTe layout algebra and vectorization.
_THREADS_PER_ROW = 256

# Number of rows processed per tile dispatch. Amortizes persistent kernel
# dispatch overhead and caches weight in registers across rows.
_ROWS_PER_TILE = 2


# =============================================================================
# RMSNorm Op
# =============================================================================


class RMSNormOp(Op):
    """RMSNorm operation for the megakernel framework.

    Applies Root Mean Square Layer Normalization:
        y = x / sqrt(mean(x²) + eps) * weight

    Uses vectorized memory access via CuTe local_partition + autovec_copy
    (128-bit loads/stores) when D >= threads_per_row, with scalar fallback
    for small D. Hierarchical reduction via warp_reduction + block_reduce.

    Multi-row tiling: each tile processes _ROWS_PER_TILE rows to amortize
    persistent kernel dispatch overhead.

    Zero-page op: all data accessed directly from global memory.

    Tensor declarations:
        x:      (M, D)  — input tensor, float32
        weight: (D,)    — per-element scale, float32
        y:      (M, D)  — output tensor, float32

    Tiling:
        tile_m indexes row groups (ceil(M / _ROWS_PER_TILE) tiles).
    """

    reads = {
        "x": (Float32, "M, D"),
        "weight": (Float32, "D"),
    }
    writes = {"y": (Float32, "M, D")}
    tile = ("M",)

    backward_reads = {
        "dout": (Float32, "M, D"),
        "x": (Float32, "M, D"),
        "weight": (Float32, "D"),
    }
    backward_writes = {"dx": (Float32, "M, D")}

    # --- Forward ---

    @staticmethod
    def compute_forward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """RMSNorm forward for _ROWS_PER_TILE rows with vectorized memory access.

        Weight is loaded once into registers and reused across all rows.
        """
        # Shared memory for cross-warp reduction
        reduction_smem = cute.arch.alloc_smem(Float32, 8)
        reduction_buffer = cute.make_tensor(
            reduction_smem, cute.make_layout((1, 8))
        )

        row_start = tile_m * _ROWS_PER_TILE

        # D is a static compile-time int; branch is resolved at trace time
        if D >= _THREADS_PER_ROW:
            # === Vectorized path: D >= 256 threads ===
            thr_layout = cute.make_layout(_THREADS_PER_ROW)

            # Load weight once into registers (reused across all rows)
            w_row = cute.make_tensor(
                cute.make_ptr(Float32, weight_ptr_raw, cute.AddressSpace.gmem),
                cute.make_layout(D),
            )
            w_part = cute.local_partition(w_row, thr_layout, tidx)
            w_reg = cute.make_fragment_like(w_part)
            cute.autovec_copy(w_part, w_reg)

            # Process _ROWS_PER_TILE rows (loop unrolled at trace time)
            for row_off in range(_ROWS_PER_TILE):
                row_idx = row_start + row_off

                if row_idx < M:
                    # Row-view tensors with static-D layout
                    x_row = cute.make_tensor(
                        cute.make_ptr(Float32, x_ptr_raw, cute.AddressSpace.gmem)
                        + row_idx * D,
                        cute.make_layout(D),
                    )
                    y_row = cute.make_tensor(
                        cute.make_ptr(Float32, y_ptr_raw, cute.AddressSpace.gmem)
                        + row_idx * D,
                        cute.make_layout(D),
                    )

                    x_part = cute.local_partition(x_row, thr_layout, tidx)
                    y_part = cute.local_partition(y_row, thr_layout, tidx)

                    # Load x into registers (vectorized 128-bit loads)
                    x_reg = cute.make_fragment_like(x_part)
                    cute.autovec_copy(x_part, x_reg)

                    # Local reduction: sum of x²
                    partial_sq = Float32(0.0)
                    for i in range(cute.size(x_reg)):
                        partial_sq = partial_sq + x_reg[i] * x_reg[i]

                    # Cross-thread reduction: warp → block
                    warp_val = cute.arch.warp_reduction(partial_sq, operator.add)
                    sum_sq = block_reduce(warp_val, operator.add,
                                          reduction_buffer, Float32(0.0))

                    # rstd = 1 / sqrt(mean(x²) + eps)
                    rstd = cute.math.rsqrt(sum_sq / D + RMSNORM_EPS,
                                           fastmath=True)

                    # Normalize: y = x * rstd * weight
                    for i in range(cute.size(x_reg)):
                        x_reg[i] = x_reg[i] * rstd * w_reg[i]

                    # Store result (vectorized 128-bit stores)
                    cute.autovec_copy(x_reg, y_part)

                # Barrier before reusing reduction_buffer in next row
                cute.arch.barrier()

        else:
            # === Scalar path: D < 256 threads (small hidden dim) ===
            for row_off in range(_ROWS_PER_TILE):
                row_idx = row_start + row_off

                if row_idx < M:
                    row_offset = row_idx * D

                    # Pass 1: sum of squares
                    partial_sq = Float32(0.0)
                    for i in range(tidx, D, num_threads):
                        val = x[row_offset + i]
                        partial_sq = partial_sq + val * val

                    warp_val = cute.arch.warp_reduction(partial_sq, operator.add)
                    sum_sq = block_reduce(warp_val, operator.add,
                                          reduction_buffer, Float32(0.0))

                    rstd = cute.math.rsqrt(sum_sq / D + RMSNORM_EPS,
                                           fastmath=True)

                    # Pass 2: normalize and apply weight
                    for i in range(tidx, D, num_threads):
                        val = x[row_offset + i]
                        w = weight[i]
                        y[row_offset + i] = val * rstd * w

                # Barrier before reusing reduction_buffer in next row
                cute.arch.barrier()

    # --- Backward ---

    @staticmethod
    def compute_backward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """RMSNorm backward for _ROWS_PER_TILE rows with vectorized memory access.

        dx = (dout * weight - x * rstd² * mean(dout * weight * x)) * rstd

        Weight is loaded once into registers and reused across all rows.
        """
        # Shared memory for cross-warp reduction
        reduction_smem = cute.arch.alloc_smem(Float32, 8)
        reduction_buffer = cute.make_tensor(
            reduction_smem, cute.make_layout((1, 8))
        )

        row_start = tile_m * _ROWS_PER_TILE

        if D >= _THREADS_PER_ROW:
            # === Vectorized path ===
            thr_layout = cute.make_layout(_THREADS_PER_ROW)

            # Load weight once into registers (reused across all rows)
            w_row = cute.make_tensor(
                cute.make_ptr(Float32, weight_ptr_raw, cute.AddressSpace.gmem),
                cute.make_layout(D),
            )
            w_part = cute.local_partition(w_row, thr_layout, tidx)
            w_reg = cute.make_fragment_like(w_part)
            cute.autovec_copy(w_part, w_reg)

            for row_off in range(_ROWS_PER_TILE):
                row_idx = row_start + row_off

                if row_idx < M:
                    x_row = cute.make_tensor(
                        cute.make_ptr(Float32, x_ptr_raw, cute.AddressSpace.gmem)
                        + row_idx * D,
                        cute.make_layout(D),
                    )
                    dout_row = cute.make_tensor(
                        cute.make_ptr(Float32, dout_ptr_raw, cute.AddressSpace.gmem)
                        + row_idx * D,
                        cute.make_layout(D),
                    )
                    dx_row = cute.make_tensor(
                        cute.make_ptr(Float32, dx_ptr_raw, cute.AddressSpace.gmem)
                        + row_idx * D,
                        cute.make_layout(D),
                    )

                    x_part = cute.local_partition(x_row, thr_layout, tidx)
                    dout_part = cute.local_partition(dout_row, thr_layout, tidx)
                    dx_part = cute.local_partition(dx_row, thr_layout, tidx)

                    # Load x into registers (vectorized)
                    x_reg = cute.make_fragment_like(x_part)
                    cute.autovec_copy(x_part, x_reg)

                    # Pass 1: Compute rstd
                    partial_sq = Float32(0.0)
                    for i in range(cute.size(x_reg)):
                        partial_sq = partial_sq + x_reg[i] * x_reg[i]

                    warp_val = cute.arch.warp_reduction(partial_sq, operator.add)
                    sum_sq = block_reduce(warp_val, operator.add,
                                          reduction_buffer, Float32(0.0))
                    rstd = cute.math.rsqrt(sum_sq / D + RMSNORM_EPS,
                                           fastmath=True)

                    # Load dout (vectorized) — weight already in w_reg
                    dout_reg = cute.make_fragment_like(dout_part)
                    cute.autovec_copy(dout_part, dout_reg)

                    # Pass 2: Compute sum(dout * weight * x)
                    partial_grad = Float32(0.0)
                    for i in range(cute.size(x_reg)):
                        partial_grad = (partial_grad
                                        + dout_reg[i] * w_reg[i] * x_reg[i])

                    # Barrier before reusing reduction_buffer
                    cute.arch.barrier()

                    warp_val2 = cute.arch.warp_reduction(partial_grad,
                                                         operator.add)
                    sum_grad = block_reduce(warp_val2, operator.add,
                                            reduction_buffer, Float32(0.0))
                    mean_grad = sum_grad / D

                    # Pass 3: dx = (dout * w - x * rstd² * mean_grad) * rstd
                    for i in range(cute.size(x_reg)):
                        dw_x = dout_reg[i] * w_reg[i]
                        x_reg[i] = (dw_x - x_reg[i] * rstd * rstd
                                    * mean_grad) * rstd

                    # Store dx (vectorized)
                    cute.autovec_copy(x_reg, dx_part)

                # Barrier before reusing reduction_buffer in next row
                cute.arch.barrier()

        else:
            # === Scalar path ===
            for row_off in range(_ROWS_PER_TILE):
                row_idx = row_start + row_off

                if row_idx < M:
                    row_offset = row_idx * D

                    # Pass 1: Recompute rstd from x
                    partial_sq = Float32(0.0)
                    for i in range(tidx, D, num_threads):
                        val = x[row_offset + i]
                        partial_sq = partial_sq + val * val

                    warp_val = cute.arch.warp_reduction(partial_sq, operator.add)
                    sum_sq = block_reduce(warp_val, operator.add,
                                          reduction_buffer, Float32(0.0))
                    rstd = cute.math.rsqrt(sum_sq / D + RMSNORM_EPS,
                                           fastmath=True)

                    # Pass 2: Compute sum(dout * weight * x)
                    partial_grad = Float32(0.0)
                    for i in range(tidx, D, num_threads):
                        partial_grad = (partial_grad
                                        + dout[row_offset + i]
                                        * weight[i]
                                        * x[row_offset + i])

                    # Barrier before reusing reduction_buffer
                    cute.arch.barrier()

                    warp_val2 = cute.arch.warp_reduction(partial_grad,
                                                         operator.add)
                    sum_grad = block_reduce(warp_val2, operator.add,
                                            reduction_buffer, Float32(0.0))
                    mean_grad = sum_grad / D

                    # Pass 3: Compute dx
                    for i in range(tidx, D, num_threads):
                        dw_x = dout[row_offset + i] * weight[i]
                        x_val = x[row_offset + i]
                        dx[row_offset + i] = (dw_x - x_val * rstd * rstd
                                              * mean_grad) * rstd

                # Barrier before reusing reduction_buffer in next row
                cute.arch.barrier()


# Override compute_tiles to return ceil(M / _ROWS_PER_TILE)
# (metaclass sets compute_tiles = M, we need fewer tiles for multi-row)
@staticmethod
def _rmsnorm_compute_tiles(**tensors):
    M = tensors["x"].shape[0]
    return (M + _ROWS_PER_TILE - 1) // _ROWS_PER_TILE


RMSNormOp.compute_tiles = _rmsnorm_compute_tiles


__all__ = ["RMSNormOp"]
