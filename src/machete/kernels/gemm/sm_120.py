# Copyright (c) 2025, Machete Authors
"""
Dense GEMM Op for Blackwell GeForce (SM_120) - Machete Megakernel Framework.

High-performance GEMM implementation using TMA and warp-specialized execution,
adapted from NVIDIA CUTLASS Blackwell GeForce dense_gemm.py.

Architecture:
- TMA for global→shared memory transfers (efficient async copies)
- Warp-specialized execution: MMA warps (compute) + DMA warp (loads)
- PipelineTmaAsync for async pipeline barrier handling
- Multi-stage K-loop with register prefetch overlapping computation
- ldmatrix for efficient smem→register transfers
- Tensor core MMA (mma.sync.m16n8k16)
- Accumulation in fp32, direct store to global memory
- Dynamic warp count based on megakernel threads_per_row

Usage:
    from machete.kernels.gemm import GemmOp
    from machete.megakernel import Megakernel, MegakernelConfig

    scheduled_op = GemmOp.schedule(a=A, b=B, c=C)
    kernel = Megakernel([scheduled_op], config=MegakernelConfig(threads_per_block=160))
    kernel.run()
"""

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Int32, Int64, Float32

from machete.megakernel.ops import Op


# =============================================================================
# Constants
# =============================================================================

# Default tile shape for GEMM
DEFAULT_TILE_M = 128
DEFAULT_TILE_N = 128
DEFAULT_TILE_K = 64

# Number of pipeline stages for K-loop
NUM_STAGES = 2

# Buffer alignment for shared memory
BUFFER_ALIGN_BYTES = 1024


# =============================================================================
# GEMM Op for Megakernel (TMA + Warp-Specialized, Blackwell GeForce)
# =============================================================================


class GemmOp(Op):
    """GEMM operation using TMA and warp specialization for Blackwell GeForce.

    Computes C = A @ B^T where:
    - A has shape (M, K) - input matrix
    - B has shape (N, K) - weight matrix (note: K is the second dimension)
    - C has shape (M, N) - output matrix

    This follows CUTLASS/BLAS conventions where B is stored as (N, K).

    Implementation adapted from NVIDIA CUTLASS Blackwell GeForce dense_gemm.py:
    - TMA for global→shared memory (efficient async copies)
    - Warp specialization: MMA warps (compute) + DMA warp (TMA loads)
    - PipelineTmaAsync for async pipeline barrier handling
    - K-loop with register prefetch for latency hiding
    - ldmatrix for smem→register transfers
    - Tensor core MMA (mma.sync.m16n8k16)
    - Direct global stores for epilogue
    - Dynamic warp count from megakernel's threads_per_row

    Tensor declarations:
        a: (M, K)  - input matrix, fp16/bf16
        b: (N, K)  - weight matrix, fp16/bf16
        c: (M, N)  - output matrix, fp16/bf16
    """

    # Tensor declarations (dtype=None means infer from tensor at schedule time)
    reads = {
        "a": (None, "M, K"),
        "b": (None, "N, K"),
    }
    writes = {"c": (None, "M, N")}

    # Tile over M and N dimensions
    tile = (("M", DEFAULT_TILE_M), ("N", DEFAULT_TILE_N))

    # No shared memory pages managed by framework - we manage our own
    NUM_INPUT_PAGES = 0
    NUM_OUTPUT_PAGES = 0

    @staticmethod
    def compute(
        page_ptr: Int32,
        tile_m: Int32,
        tile_n: Int32,
        tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """GEMM forward pass with TMA and warp specialization (Blackwell GeForce).

        Adapted from NVIDIA CUTLASS Blackwell GeForce dense_gemm.py.

        Auto-generated variables available:
        - tidx, num_threads: thread indexing
        - threads_per_row: compile-time warp count from megakernel
        - a, b, c: CuTe tensor views
        - M, N, K: dimension sizes
        - a_dtype: tensor dtype
        - tile_size_M, tile_size_N: output tile sizes
        """
        # =====================================================================
        # Configuration
        # =====================================================================
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Tile shape configuration
        tile_shape_m = tile_size_M
        tile_shape_n = tile_size_N
        tile_shape_k = DEFAULT_TILE_K
        tile_shape_mnk = (tile_shape_m, tile_shape_n, tile_shape_k)

        # Compute number of warps from threads_per_row (compile-time constant)
        num_warps = threads_per_row // 32

        # Warp specialization: MMA warps for compute, 1 DMA warp for TMA loads
        num_mma_warps = num_warps - 1
        is_mma_warp = warp_idx < num_mma_warps
        is_dma_warp = warp_idx == num_mma_warps

        # MMA instruction shape
        mma_inst_mnk = (16, 8, 16)

        # Configure atom_layout based on MMA warp count
        # Default: spread across N dimension (works for any warp count)
        atom_layout_m = 1
        atom_layout_n = num_mma_warps

        # Override for specific warp counts with better layouts
        if num_mma_warps == 4:
            atom_layout_m = 2
            atom_layout_n = 2
        if num_mma_warps == 8:
            atom_layout_m = 2
            atom_layout_n = 4

        atom_layout = (atom_layout_m, atom_layout_n, 1)

        # Number of K tiles
        k_tile_cnt = (K + tile_shape_k - 1) // tile_shape_k

        # Number of pipeline stages (compile-time constant)
        ab_stage = NUM_STAGES

        # =====================================================================
        # Create TiledMMA for MMA warps
        # =====================================================================
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            a_dtype,
            Float32,  # Accumulator dtype
            mma_inst_mnk,
        )
        tC = cute.make_layout(atom_layout)
        permutation_mnk = (
            atom_layout[0] * mma_inst_mnk[0],
            atom_layout[1] * mma_inst_mnk[1] * 2,
            atom_layout[2] * mma_inst_mnk[2],
        )
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            tC,
            permutation_mnk=permutation_mnk,
        )

        # =====================================================================
        # Create shared memory layouts using CUTLASS helpers
        # =====================================================================
        a_layout = utils.LayoutEnum.ROW_MAJOR
        b_layout = utils.LayoutEnum.ROW_MAJOR

        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )

        # Get single-stage layouts for TMA
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        # Compute TMA copy bytes directly from tile sizes
        tma_copy_bytes = (
            tile_shape_m * tile_shape_k * a_dtype.width // 8
            + tile_shape_n * tile_shape_k * a_dtype.width // 8
        )

        # =====================================================================
        # Create Global Memory Tensors
        # =====================================================================
        mA = cute.make_tensor(
            a.iterator,
            cute.make_layout((M, K), stride=(K, 1)),
        )
        mB = cute.make_tensor(
            b.iterator,
            cute.make_layout((N, K), stride=(K, 1)),
        )
        mC = cute.make_tensor(
            c.iterator,
            cute.make_layout((M, N), stride=(N, 1)),
        )

        # =====================================================================
        # TMA Descriptors for async global→shared copies
        # =====================================================================
        tma_atom_a, mA_tma = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mA,
            a_smem_layout,
            (tile_shape_m, tile_shape_k),
            num_multicast=1,
        )
        tma_atom_b, mB_tma = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mB,
            b_smem_layout,
            (tile_shape_n, tile_shape_k),
            num_multicast=1,
        )

        # =====================================================================
        # Shared Memory Allocation
        # =====================================================================
        smem = cutlass.utils.SmemAllocator()

        # Pipeline barriers (2 barriers per stage) - allocate_array returns a pointer
        mainloop_pipeline_array_ptr = smem.allocate_array(Int64, ab_stage * 2)

        # Allocate staged A and B buffers
        sA = smem.allocate_tensor(
            a_dtype,
            a_smem_layout_staged.outer,
            byte_alignment=BUFFER_ALIGN_BYTES,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            a_dtype,
            b_smem_layout_staged.outer,
            byte_alignment=BUFFER_ALIGN_BYTES,
            swizzle=b_smem_layout_staged.inner,
        )

        # =====================================================================
        # Pipeline Setup
        # =====================================================================
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_mma_warps
        )

        # cta_layout_vmnk: (V, M, N, K) - no cluster, so all 1s
        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        # =====================================================================
        # Local_tile partition global tensors for this CTA
        # =====================================================================
        # (bM, bK, loopK)
        gA_mk = cute.local_tile(
            mA_tma,
            cute.slice_(tile_shape_mnk, (None, 0, None)),
            (tile_m, None),
        )
        # (bN, bK, loopK)
        gB_nk = cute.local_tile(
            mB_tma,
            cute.slice_(tile_shape_mnk, (0, None, None)),
            (tile_n, None),
        )
        # (bM, bN)
        gC_mn = cute.local_tile(mC, (tile_shape_m, tile_shape_n), (tile_m, tile_n))

        # =====================================================================
        # TMA partition for shared/global tensors
        # =====================================================================
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mk, 0, 2),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nk, 0, 2),
        )

        # =====================================================================
        # MMA Partitioning
        # =====================================================================
        thr_mma = tiled_mma.get_slice(tidx)

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

        tCgC = thr_mma.partition_C(gC_mn)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, Float32)

        # Number of K blocks per stage
        num_k_blocks = cute.size(tCrA, mode=[2])

        # =====================================================================
        # ldmatrix Copy Atoms for smem→register
        # =====================================================================
        # Row-major (K-major) layouts: A is not M-major, B is not N-major
        is_a_m_major = False
        is_b_n_major = False

        atom_copy_ldmatrix_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(is_a_m_major, 4),
            a_dtype,
        )
        atom_copy_ldmatrix_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(is_b_n_major, 4),
            a_dtype,
        )

        smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

        thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
        thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)

        tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
        tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
        tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
        tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

        # Sync before starting
        pipeline.sync(barrier_id=1)

        # =====================================================================
        # Create pipeline states
        # =====================================================================
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, ab_stage
        )

        # =====================================================================
        # MMA Warp Group - Compute
        # =====================================================================
        if is_mma_warp:
            # Clear the accumulator
            accumulators.fill(0.0)

            # Reset consumer state
            mainloop_consumer_state.reset_count()

            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_state
                )

            # Wait for first TMA load to complete
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_state, peek_ab_full_status
            )

            # Get partitioned views for current stage
            tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
            tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]

            # Initial load for k_block 0
            cute.copy(
                smem_tiled_copy_A,
                tCsA_p[None, None, 0],
                tCrA_copy_view[None, None, 0],
            )
            cute.copy(
                smem_tiled_copy_B,
                tCsB_p[None, None, 0],
                tCrB_copy_view[None, None, 0],
            )

            # Main K-loop (all but last k_tile)
            for k_tile in range(0, k_tile_cnt - 1, 1, unroll=1):
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )

                    if k_block_idx == num_k_blocks - 1:
                        # Release current stage and advance to next
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                        peek_ab_full_status = cutlass.Boolean(1)
                        peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                            mainloop_consumer_state
                        )

                        # Update views for new stage
                        tCsA_p = tCsA_copy_view[
                            None, None, None, mainloop_consumer_state.index
                        ]
                        tCsB_p = tCsB_copy_view[
                            None, None, None, mainloop_consumer_state.index
                        ]
                        mainloop_pipeline.consumer_wait(
                            mainloop_consumer_state, peek_ab_full_status
                        )

                    # Prefetch next k_block while computing current
                    cute.copy(
                        smem_tiled_copy_A,
                        tCsA_p[None, None, k_block_next],
                        tCrA_copy_view[None, None, k_block_next],
                    )
                    cute.copy(
                        smem_tiled_copy_B,
                        tCsB_p[None, None, k_block_next],
                        tCrB_copy_view[None, None, k_block_next],
                    )

                    # MMA for current k_block
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        tCrA[None, None, k_block_idx],
                        tCrB[None, None, k_block_idx],
                        accumulators,
                    )

            # Last k_tile (hoisted out)
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_next = (
                    0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                )

                if k_block_idx == num_k_blocks - 1:
                    mainloop_pipeline.consumer_release(mainloop_consumer_state)
                    mainloop_consumer_state.advance()

                if k_block_next > 0:
                    cute.copy(
                        smem_tiled_copy_A,
                        tCsA_p[None, None, k_block_next],
                        tCrA_copy_view[None, None, k_block_next],
                    )
                    cute.copy(
                        smem_tiled_copy_B,
                        tCsB_p[None, None, k_block_next],
                        tCrB_copy_view[None, None, k_block_next],
                    )

                # MMA for current k_block
                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA[None, None, k_block_idx],
                    tCrB[None, None, k_block_idx],
                    accumulators,
                )

            # =================================================================
            # Epilogue: Store C
            # =================================================================
            # Convert accumulators from fp32 to output dtype
            acc_out = cute.make_rmem_tensor(accumulators.shape, a_dtype)
            acc_vec = accumulators.load()
            acc_out.store(acc_vec.to(a_dtype))

            # Direct store to global memory
            cute.autovec_copy(acc_out, tCgC)

        # =====================================================================
        # DMA Warp Group - Load
        # =====================================================================
        if is_dma_warp:
            # Prefetch TMA descriptors
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

            mainloop_producer_state.reset_count()

            for k_tile in range(0, k_tile_cnt, 1, unroll=1):
                # Wait for buffer to be empty
                mainloop_pipeline.producer_acquire(mainloop_producer_state)

                # Slice to current k_tile
                tAgA_k = tAgA[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tBgB_k = tBgB[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                # TMA load A
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=0,
                )
                # TMA load B
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=0,
                )

                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

            # Wait for all buffers to be consumed
            mainloop_pipeline.producer_tail(mainloop_producer_state)


# =============================================================================
# High-Level API
# =============================================================================


def gemm_megakernel(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Perform GEMM using the machete megakernel framework.

    Computes C = A @ B^T where:
    - A has shape (M, K)
    - B has shape (N, K)
    - C has shape (M, N)

    Args:
        a: Input tensor A with shape (M, K), fp16/bf16
        b: Input tensor B with shape (N, K), fp16/bf16

    Returns:
        Output tensor C with shape (M, N)
    """
    from machete.megakernel import Megakernel, MegakernelConfig

    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    assert a.dtype == b.dtype, f"Dtype mismatch: {a.dtype} != {b.dtype}"
    assert a.dtype in (torch.float16, torch.bfloat16), "Only fp16/bf16 supported"

    m, k = a.shape
    n, k2 = b.shape
    assert k == k2, f"K dimension mismatch: {k} != {k2}"

    # Create output tensor
    c = torch.empty(m, n, dtype=a.dtype, device=a.device)

    # Schedule GEMM operation
    scheduled_op = GemmOp.schedule(a=a, b=b, c=c)

    ops = [scheduled_op]

    # Use 5 warps: 4 MMA warps + 1 DMA warp = 160 threads
    # This follows the Blackwell GeForce pattern with dedicated DMA warp
    kernel_config = MegakernelConfig(threads_per_block=160)
    kernel = Megakernel(ops, config=kernel_config)
    kernel.run()

    return c


__all__ = [
    "GemmOp",
    "gemm_megakernel",
]
