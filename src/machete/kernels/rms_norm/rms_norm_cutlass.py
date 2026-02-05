# Copyright (c) 2025, Machete Authors
"""
RMSNorm Op for the Megakernel.

Applies Root Mean Square Layer Normalization:
    y = x / sqrt(mean(x²) + eps) * weight

This implementation faithfully follows the NVIDIA CUTLASS RMSNorm pattern
(examples/python/CuTeDSL/blackwell/rmsnorm.py) with:
- local_tile() for 2D tiling of tensors
- TiledCopy abstractions for coalesced 128-bit vectorized access
- Async copy (gmem → smem) to hide memory latency
- Shared memory staging for fast data reuse across passes
- TensorSSA for element-wise operations
- Hierarchical reduction (warp → block → cluster)
- Cluster support for large hidden dimensions (cluster_n > 1)

Two-Pass Architecture:
    Pass 1 (Reduction): Load x to smem via async copy, compute sum(x²), derive rstd
    Pass 2 (Normalize): Reload x from smem (fast!), compute y = x * rstd * w

Cluster Mode (cluster_n > 1):
    - D dimension is partitioned across CTAs in the cluster
    - Each CTA processes D/cluster_n elements
    - Cross-CTA reduction via distributed shared memory and mbarrier

Usage:
    from machete.kernels.rms_norm import RMSNormOp
    from machete.megakernel import Megakernel, MegakernelConfig

    x_2d = x.view(-1, D).contiguous()
    ops = [RMSNormOp.schedule(x=x_2d, weight=w, y=y)]

    # Single CTA mode (default)
    kernel = Megakernel(ops)
    kernel.run()

    # Cluster mode for large D (e.g., cluster_size=2 for D >= 2048)
    config = MegakernelConfig(cluster_size=2)
    kernel = Megakernel(ops, config=config)
    kernel.run()
"""

import operator

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Boolean, Int32, Int64, Float32

from machete.megakernel.ops import Op
from machete.kernels.utils.reduce import block_reduce, row_reduce


# =============================================================================
# Constants
# =============================================================================

RMSNORM_EPS = 1e-6
COPY_BITS = 128  # 128-bit vectorized loads

# Thread configuration (num_threads, warps_per_block, cluster_n, cluster_idx)
# is provided by the megakernel framework as compile-time constants in init source.


# =============================================================================
# Predicate Utility (from CUTLASS reference)
# =============================================================================


@cute.jit
def predicate_k(tXcX: cute.Tensor, limit: int) -> cute.Tensor:
    """Create predicate tensor for bounds checking.

    Following CUTLASS pattern: creates a predicate mask for TiledCopy operations
    to handle cases where the tensor dimension doesn't evenly divide the tile size.
    """
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(tXcX, mode=[0, 1]), cute.size(tXcX, mode=[1]), cute.size(tXcX, mode=[2])),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            tXpX[rest_v, 0, rest_k] = cute.elem_less(tXcX[(0, rest_v), 0, rest_k][1], limit)
    return tXpX


# =============================================================================
# RMSNorm Op
# =============================================================================


class RMSNormOp(Op):
    """RMSNorm operation for the megakernel framework.

    Applies Root Mean Square Layer Normalization:
        y = x / sqrt(mean(x²) + eps) * weight

    This implementation faithfully follows the CUTLASS RMSNorm pattern with:
    - local_tile() for 2D tiling
    - TiledCopy with 2D tv_layout for vectorized memory access
    - Async copy for memory latency hiding
    - Shared memory staging for two-pass algorithm

    Cluster Mode (cluster_n > 1):
        When the megakernel is configured with cluster_size > 1, multiple CTAs
        cooperate on the same tile by dividing the D dimension:
        - Each CTA processes D/cluster_n elements
        - CTA k processes elements [k*D/cluster_n : (k+1)*D/cluster_n]
        - Cross-CTA reduction aggregates partial sums via mbarrier

    Tensor declarations:
        x:      (M, D)  — input tensor, float32
        weight: (D,)    — per-element scale, float32
        y:      (M, D)  — output tensor, float32

    Tiling:
        rows_per_block rows per tile (computed from thread configuration).
        Following CUTLASS: rows_per_block = num_threads // threads_per_row
    """

    reads = {
        "x": (Float32, "M, D"),
        "weight": (Float32, "D"),
    }
    writes = {"y": (Float32, "M, D")}
    tile = ("M",)
    # tile_size_m will be computed dynamically based on thread configuration
    # Default to 4 (typical for 128 threads, 32 threads_per_row)
    tile_size = {"M": 4}

    backward_reads = {
        "dout": (Float32, "M, D"),
        "x": (Float32, "M, D"),
        "weight": (Float32, "D"),
    }
    backward_writes = {"dx": (Float32, "M, D")}

    @staticmethod
    def forward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """RMSNorm forward following CUTLASS pattern exactly.

        Uses local_tile() for 2D tiling and TiledCopy for vectorized memory access.
        """
        # =====================================================================
        # Configuration (computed from compile-time constants)
        # =====================================================================
        vec_size = COPY_BITS // 32  # 4 floats for float32

        # Thread configuration following CUTLASS pattern
        if cutlass.const_expr(D <= 64):
            threads_per_row = 8
        elif cutlass.const_expr(D <= 128):
            threads_per_row = 16
        elif cutlass.const_expr(D <= 3072):
            threads_per_row = 32
        elif cutlass.const_expr(D <= 6144):
            threads_per_row = 64
        elif cutlass.const_expr(D <= 16384):
            threads_per_row = 128
        else:
            threads_per_row = 256

        rows_per_block = num_threads // threads_per_row
        warps_per_row = max(threads_per_row // 32, 1)

        # Compute cols_per_tile based on D and cluster configuration
        if cutlass.const_expr(cluster_n > 1):
            N_per_cta = D // cluster_n
        else:
            N_per_cta = D

        num_vec_blocks = max(
            1, (N_per_cta // vec_size + threads_per_row - 1) // threads_per_row
        )
        cols_per_tile = vec_size * num_vec_blocks * threads_per_row

        # Tile dimensions
        tiler_mn = (rows_per_block, cols_per_tile)

        # Calculate row range for this tile
        start_row = tile_m * rows_per_block

        # =====================================================================
        # Cluster configuration - use const_expr for compile-time branching
        # =====================================================================
        if cutlass.const_expr(cluster_n > 1):
            # === CLUSTER MODE ===
            # Allocate shared memory using SmemAllocator pattern
            smem = utils.SmemAllocator()

            sX = smem.allocate_tensor(
                Float32,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )

            mbar_ptr = smem.allocate_array(Int64, num_elems=1)

            # Initialize mbarrier (thread 0 only)
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()

            # Cluster sync before starting computation
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

            # Create 2D tensor views of global memory
            # assumed_align=16 required for 128-bit async copy
            mX = cute.make_tensor(
                cute.make_ptr(Float32, x_ptr_raw, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((M, D), stride=(D, 1)),
            )
            mY = cute.make_tensor(
                cute.make_ptr(Float32, y_ptr_raw, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((M, D), stride=(D, 1)),
            )
            mW = cute.make_tensor(
                cute.make_ptr(Float32, weight_ptr_raw, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((D,), stride=(1,)),
            )

            # Create identity tensor for coordinate tracking
            idX = cute.make_identity_tensor(mX.shape)

            # Create tiled views using local_tile
            gX = cute.local_tile(mX, tiler_mn, (tile_m, cluster_idx))
            gY = cute.local_tile(mY, tiler_mn, (tile_m, cluster_idx))
            cX = cute.local_tile(idX, tiler_mn, (tile_m, cluster_idx))

            # Expand weight to 2D for tiling
            mW_expanded_layout = cute.prepend(
                mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
            gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_idx))

            # Create TV layout following CUTLASS pattern
            tv_shape = (
                (threads_per_row, rows_per_block),
                (vec_size, num_vec_blocks),
            )
            tv_stride = (
                (vec_size * rows_per_block, 1),
                (rows_per_block, rows_per_block * vec_size * threads_per_row),
            )
            tv_layout = cute.make_layout(tv_shape, stride=tv_stride)

            # Create TiledCopy operations
            copy_atom_load_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                Float32,
                num_bits_per_copy=COPY_BITS,
            )
            copy_atom_load_W = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Float32,
                num_bits_per_copy=COPY_BITS,
            )
            copy_atom_store = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Float32,
                num_bits_per_copy=COPY_BITS,
            )

            tiled_copy_load = cute.make_tiled_copy(copy_atom_load_async, tv_layout, tiler_mn)
            tiled_copy_W = cute.make_tiled_copy(copy_atom_load_W, tv_layout, tiler_mn)
            tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

            thr_copy_X = tiled_copy_load.get_slice(tidx)
            thr_copy_W = tiled_copy_W.get_slice(tidx)
            thr_copy_Y = tiled_copy_store.get_slice(tidx)

            # Partition tensors
            tXgX = thr_copy_X.partition_S(gX)
            tXsX = thr_copy_X.partition_D(sX)
            tXgY = thr_copy_Y.partition_D(gY)
            tXcX = thr_copy_X.partition_S(cX)

            # Register fragments
            tXrX = cute.make_fragment_like(tXgX)
            tXrY = cute.make_fragment_like(tXgY)

            tWgW = thr_copy_W.partition_S(gW)
            tWrW = cute.make_fragment_like(tWgW)
            tXrW = thr_copy_X.retile(tWrW)

            # Bounds checking
            tXpX = predicate_k(tXcX, limit=D)

            row_coord = tXcX[(0, 0), 0, 0]
            row_in_bounds = row_coord[0] < M

            # ========== Async copy global → shared ==========
            if row_in_bounds:
                cute.copy(copy_atom_load_async, tXgX, tXsX, pred=tXpX)

            cute.arch.cp_async_commit_group()

            # Load weight while waiting
            tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=D)
            cute.copy(copy_atom_load_W, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            # ========== Pass 1: Compute sum of squares with cluster reduction ==========
            cute.autovec_copy(tXsX, tXrX)
            x_ssa = tXrX.load().to(Float32)

            x_sq = x_ssa * x_ssa
            sum_sq = row_reduce(
                x_sq,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer,
                mbar_ptr,
                cluster_n,
                Float32(0.0),
            )

            # rstd = 1 / sqrt(mean(x²) + eps)
            mean_sq = sum_sq / D
            rstd = cute.math.rsqrt(mean_sq + RMSNORM_EPS, fastmath=True)

            # Sync after reduction
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

            # ========== Pass 2: Normalize and output ==========
            cute.autovec_copy(tXsX, tXrX)
            x_ssa = tXrX.load().to(Float32)

            y_ssa = x_ssa * rstd

            # Apply weight
            w_ssa = tXrW.load().to(Float32)
            y_ssa = y_ssa * w_ssa

            # Store to global memory
            tXrY.store(y_ssa.to(Float32))

            if row_in_bounds:
                cute.copy(copy_atom_store, tXrY, tXgY, pred=tXpX)

        else:
            # === SINGLE-CTA MODE ===
            # Allocate shared memory using SmemAllocator pattern
            smem = utils.SmemAllocator()

            sX = smem.allocate_tensor(
                Float32,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )

            # Create 2D tensor views of global memory
            # assumed_align=16 required for 128-bit async copy
            mX = cute.make_tensor(
                cute.make_ptr(Float32, x_ptr_raw, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((M, D), stride=(D, 1)),
            )
            mY = cute.make_tensor(
                cute.make_ptr(Float32, y_ptr_raw, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((M, D), stride=(D, 1)),
            )
            mW = cute.make_tensor(
                cute.make_ptr(Float32, weight_ptr_raw, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((D,), stride=(1,)),
            )

            # Create identity tensor for coordinate tracking
            idX = cute.make_identity_tensor(mX.shape)

            # Create tiled views using local_tile
            # tile_m is the tile index from instruction stream
            gX = cute.local_tile(mX, tiler_mn, (tile_m, 0))
            gY = cute.local_tile(mY, tiler_mn, (tile_m, 0))
            cX = cute.local_tile(idX, tiler_mn, (tile_m, 0))

            # Expand weight to 2D for tiling
            mW_expanded_layout = cute.prepend(
                mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
            gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))

            # Create TV layout following CUTLASS pattern
            tv_shape = (
                (threads_per_row, rows_per_block),
                (vec_size, num_vec_blocks),
            )
            tv_stride = (
                (vec_size * rows_per_block, 1),
                (rows_per_block, rows_per_block * vec_size * threads_per_row),
            )
            tv_layout = cute.make_layout(tv_shape, stride=tv_stride)

            # Create TiledCopy operations
            copy_atom_load_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                Float32,
                num_bits_per_copy=COPY_BITS,
            )
            copy_atom_load_W = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Float32,
                num_bits_per_copy=COPY_BITS,
            )
            copy_atom_store = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Float32,
                num_bits_per_copy=COPY_BITS,
            )

            tiled_copy_load = cute.make_tiled_copy(copy_atom_load_async, tv_layout, tiler_mn)
            tiled_copy_W = cute.make_tiled_copy(copy_atom_load_W, tv_layout, tiler_mn)
            tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

            thr_copy_X = tiled_copy_load.get_slice(tidx)
            thr_copy_W = tiled_copy_W.get_slice(tidx)
            thr_copy_Y = tiled_copy_store.get_slice(tidx)

            # Partition tensors
            tXgX = thr_copy_X.partition_S(gX)
            tXsX = thr_copy_X.partition_D(sX)
            tXgY = thr_copy_Y.partition_D(gY)
            tXcX = thr_copy_X.partition_S(cX)

            # Register fragments
            tXrX = cute.make_fragment_like(tXgX)
            tXrY = cute.make_fragment_like(tXgY)

            tWgW = thr_copy_W.partition_S(gW)
            tWrW = cute.make_fragment_like(tWgW)
            tXrW = thr_copy_X.retile(tWrW)

            # Bounds checking
            tXpX = predicate_k(tXcX, limit=D)

            row_coord = tXcX[(0, 0), 0, 0]
            row_in_bounds = row_coord[0] < M

            # ========== Async copy global → shared ==========
            if row_in_bounds:
                cute.copy(copy_atom_load_async, tXgX, tXsX, pred=tXpX)

            cute.arch.cp_async_commit_group()

            # Load weight while waiting
            tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=D)
            cute.copy(copy_atom_load_W, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # ========== Pass 1: Compute sum of squares ==========
            cute.autovec_copy(tXsX, tXrX)
            x_ssa = tXrX.load().to(Float32)

            x_sq = x_ssa * x_ssa
            sum_sq = row_reduce(
                x_sq,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer,
                None,
                1,
                Float32(0.0),
            )

            # rstd = 1 / sqrt(mean(x²) + eps)
            mean_sq = sum_sq / D
            rstd = cute.math.rsqrt(mean_sq + RMSNORM_EPS, fastmath=True)

            cute.arch.barrier()

            # ========== Pass 2: Normalize and output ==========
            cute.autovec_copy(tXsX, tXrX)
            x_ssa = tXrX.load().to(Float32)

            y_ssa = x_ssa * rstd

            # Apply weight
            w_ssa = tXrW.load().to(Float32)
            y_ssa = y_ssa * w_ssa

            # Store to global memory
            tXrY.store(y_ssa.to(Float32))

            if row_in_bounds:
                cute.copy(copy_atom_store, tXrY, tXgY, pred=tXpX)

    @staticmethod
    def backward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """RMSNorm backward using local_partition pattern.

        dx = (dout * weight - x * rstd² * mean(dout * weight * x)) * rstd

        Uses the simpler row-by-row approach with local_partition which is
        proven to work correctly with the megakernel's tile dispatch.
        """
        # =====================================================================
        # Configuration - must match tile_size = {"M": 4} in Op definition
        # =====================================================================
        # Fixed tile size to match megakernel dispatch
        rows_per_tile = 4  # Must match tile_size["M"]
        start_row = tile_m * rows_per_tile
        warps_per_block = num_threads // 32

        if cutlass.const_expr(cluster_n > 1):
            # === CLUSTER MODE ===
            # For cluster mode, all num_threads in each CTA work on each row
            reduction_smem = cute.arch.alloc_smem(Float32, warps_per_block * cluster_n)
            reduction_buffer = cute.make_tensor(
                reduction_smem, cute.make_layout((1, (warps_per_block, cluster_n)))
            )

            mbar_smem = cute.arch.alloc_smem(Int64, 1)
            mbar_ptr = cute.make_tensor(mbar_smem, cute.make_layout(1)).iterator

            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()

            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

            d_per_cta = D // cluster_n
            d_offset = cluster_idx * d_per_cta
            thr_layout = cute.make_layout(num_threads)

            w_row = cute.make_tensor(
                cute.make_ptr(Float32, weight_ptr_raw, cute.AddressSpace.gmem)
                + d_offset,
                cute.make_layout(d_per_cta),
            )
            w_part = cute.local_partition(w_row, thr_layout, tidx)
            w_reg = cute.make_fragment_like(w_part)
            cute.autovec_copy(w_part, w_reg)

            for row_offset in range(rows_per_tile):
                row_idx = start_row + row_offset
                if row_idx < M:
                    x_row = cute.make_tensor(
                        cute.make_ptr(Float32, x_ptr_raw, cute.AddressSpace.gmem)
                        + row_idx * D + d_offset,
                        cute.make_layout(d_per_cta),
                    )
                    dout_row = cute.make_tensor(
                        cute.make_ptr(Float32, dout_ptr_raw, cute.AddressSpace.gmem)
                        + row_idx * D + d_offset,
                        cute.make_layout(d_per_cta),
                    )
                    dx_row = cute.make_tensor(
                        cute.make_ptr(Float32, dx_ptr_raw, cute.AddressSpace.gmem)
                        + row_idx * D + d_offset,
                        cute.make_layout(d_per_cta),
                    )

                    x_part = cute.local_partition(x_row, thr_layout, tidx)
                    dout_part = cute.local_partition(dout_row, thr_layout, tidx)
                    dx_part = cute.local_partition(dx_row, thr_layout, tidx)

                    x_reg = cute.make_fragment_like(x_part)
                    dout_reg = cute.make_fragment_like(dout_part)
                    cute.autovec_copy(x_part, x_reg)
                    cute.autovec_copy(dout_part, dout_reg)

                    # Pass 1: Compute rstd
                    x_ssa = x_reg.load().to(Float32)
                    x_sq = x_ssa * x_ssa

                    sum_sq = row_reduce(
                        x_sq, cute.ReductionOp.ADD, num_threads,
                        reduction_buffer, mbar_ptr, cluster_n, Float32(0.0),
                    )
                    rstd = cute.math.rsqrt(sum_sq / D + RMSNORM_EPS, fastmath=True)

                    # Pass 2: Compute sum(dout * weight * x)
                    dout_ssa = dout_reg.load().to(Float32)
                    w_ssa = w_reg.load().to(Float32)
                    x_ssa = x_reg.load().to(Float32)
                    grad_prod = dout_ssa * w_ssa * x_ssa

                    cute.arch.barrier()

                    sum_grad = row_reduce(
                        grad_prod, cute.ReductionOp.ADD, num_threads,
                        reduction_buffer, mbar_ptr, cluster_n, Float32(0.0),
                    )
                    mean_grad = sum_grad / D

                    cute.arch.barrier()

                    # Pass 3: Compute dx
                    x_ssa = x_reg.load().to(Float32)
                    dout_ssa = dout_reg.load().to(Float32)
                    w_ssa = w_reg.load().to(Float32)

                    dw_x = dout_ssa * w_ssa
                    dx_ssa = (dw_x - x_ssa * rstd * rstd * mean_grad) * rstd

                    dx_reg = cute.make_fragment_like(dx_part)
                    dx_reg.store(dx_ssa.to(Float32))
                    cute.autovec_copy(dx_reg, dx_part)

            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        else:
            # === SINGLE-CTA MODE ===
            warps_per_block = num_threads // 32
            reduction_smem = cute.arch.alloc_smem(Float32, warps_per_block)
            reduction_buffer = cute.make_tensor(
                reduction_smem, cute.make_layout((1, warps_per_block))
            )

            if cutlass.const_expr(D >= num_threads):
                # === Vectorized path: D >= num_threads ===
                thr_layout = cute.make_layout(num_threads)

                w_row = cute.make_tensor(
                    cute.make_ptr(Float32, weight_ptr_raw, cute.AddressSpace.gmem),
                    cute.make_layout(D),
                )
                w_part = cute.local_partition(w_row, thr_layout, tidx)
                w_reg = cute.make_fragment_like(w_part)
                cute.autovec_copy(w_part, w_reg)

                for row_offset in range(rows_per_tile):
                    row_idx = start_row + row_offset
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

                        x_reg = cute.make_fragment_like(x_part)
                        dout_reg = cute.make_fragment_like(dout_part)
                        cute.autovec_copy(x_part, x_reg)
                        cute.autovec_copy(dout_part, dout_reg)

                        # Pass 1: Compute rstd
                        x_ssa = x_reg.load().to(Float32)
                        x_sq = x_ssa * x_ssa

                        sum_sq = row_reduce(
                            x_sq, cute.ReductionOp.ADD, num_threads,
                            reduction_buffer, None, 1, Float32(0.0),
                        )
                        rstd = cute.math.rsqrt(sum_sq / D + RMSNORM_EPS, fastmath=True)

                        # Pass 2: Compute sum(dout * weight * x)
                        dout_ssa = dout_reg.load().to(Float32)
                        w_ssa = w_reg.load().to(Float32)
                        x_ssa = x_reg.load().to(Float32)
                        grad_prod = dout_ssa * w_ssa * x_ssa

                        cute.arch.barrier()

                        sum_grad = row_reduce(
                            grad_prod, cute.ReductionOp.ADD, num_threads,
                            reduction_buffer, None, 1, Float32(0.0),
                        )
                        mean_grad = sum_grad / D

                        cute.arch.barrier()

                        # Pass 3: Compute dx
                        x_ssa = x_reg.load().to(Float32)
                        dout_ssa = dout_reg.load().to(Float32)
                        w_ssa = w_reg.load().to(Float32)

                        dw_x = dout_ssa * w_ssa
                        dx_ssa = (dw_x - x_ssa * rstd * rstd * mean_grad) * rstd

                        dx_reg = cute.make_fragment_like(dx_part)
                        dx_reg.store(dx_ssa.to(Float32))
                        cute.autovec_copy(dx_reg, dx_part)

            else:
                # === Scalar path: D < num_threads ===
                import operator
                for row_offset in range(rows_per_tile):
                    row_idx = start_row + row_offset
                    if row_idx < M:
                        row_mem_offset = row_idx * D

                        # Pass 1: Compute rstd
                        partial_sq = Float32(0.0)
                        for i in range(tidx, D, num_threads):
                            val = x[row_mem_offset + i]
                            partial_sq = partial_sq + val * val

                        warp_val = cute.arch.warp_reduction(partial_sq, operator.add)
                        sum_sq = block_reduce(warp_val, operator.add,
                                              reduction_buffer, Float32(0.0))
                        rstd = cute.math.rsqrt(sum_sq / D + RMSNORM_EPS, fastmath=True)

                        # Pass 2: Compute sum(dout * weight * x)
                        partial_grad = Float32(0.0)
                        for i in range(tidx, D, num_threads):
                            partial_grad = (partial_grad
                                            + dout[row_mem_offset + i]
                                            * weight[i]
                                            * x[row_mem_offset + i])

                        cute.arch.barrier()

                        warp_val2 = cute.arch.warp_reduction(partial_grad, operator.add)
                        sum_grad = block_reduce(warp_val2, operator.add,
                                                reduction_buffer, Float32(0.0))
                        mean_grad = sum_grad / D

                        # Pass 3: Compute dx
                        for i in range(tidx, D, num_threads):
                            dw_x = dout[row_mem_offset + i] * weight[i]
                            x_val = x[row_mem_offset + i]
                            dx[row_mem_offset + i] = (dw_x - x_val * rstd * rstd
                                                      * mean_grad) * rstd


__all__ = ["RMSNormOp", "RMSNORM_EPS"]
