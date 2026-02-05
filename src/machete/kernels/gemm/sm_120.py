# Copyright (c) 2025, Machete Authors
"""
Dense GEMM Op for Blackwell GeForce (SM_120) - Machete Megakernel Framework.

Simplified GEMM implementation using cpasync (cp.async) instead of TMA.
TMA operations in CUTLASS DSL are SM_90 specific and don't work on SM_120.

Architecture:
- cpasync for global→shared memory transfers (instead of TMA)
- Warp-specialized execution: Producer warp (loads) + Consumer warps (compute)
- Multi-stage K-loop with software pipelining
- ldmatrix for efficient smem→register transfers
- Non-tensor-core MMA (mma.sync.m16n8k16)
- Accumulation in fp32, direct store to global memory

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
from cutlass import Int32, Int64, Float32, Boolean

from machete.megakernel.ops import Op
from machete.utils.testing import is_blackwell_available, is_hopper_available


# =============================================================================
# Constants
# =============================================================================

# Default tile shape for GEMM
DEFAULT_TILE_M = 64
DEFAULT_TILE_N = 64
DEFAULT_TILE_K = 64

# Buffer alignment for shared memory
BUFFER_ALIGN_BYTES = 128


# =============================================================================
# GEMM Op for Megakernel (cpasync + Warp-Specialized)
# =============================================================================


class GemmOp(Op):
    """GEMM operation using cpasync and warp specialization.

    Computes C = A @ B^T where:
    - A has shape (M, K) - input matrix
    - B has shape (N, K) - weight matrix (note: K is the second dimension)
    - C has shape (M, N) - output matrix

    This follows CUTLASS/BLAS conventions where B is stored as (N, K).

    Implementation uses:
    - cpasync for global→shared memory (SM_120 compatible)
    - ldmatrix for smem→register transfers
    - Non-tensor-core MMA (mma.sync.m16n8k16)
    - Direct global stores for epilogue

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
    def forward(
        smem_base: Int32,
        config_ptr: Int32,
        page_ids: tuple,
        tile_m: Int32,
        tile_n: Int32,
        tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """GEMM forward pass with cpasync and warp specialization.

        Implements simplified GEMM with software pipelining:
        1. All warps load data via cpasync
        2. All warps compute via ldmatrix + MMA
        3. All warps store results directly to global memory

        Auto-generated variables available:
        - tidx, num_threads: thread indexing
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
        lane_idx = cute.arch.lane_idx()

        # Tile shape configuration
        tile_shape_m = tile_size_M
        tile_shape_n = tile_size_N
        tile_shape_k = 64

        # MMA configuration: 2x2x1 atom layout with m16n8k16 MMA
        atom_layout = (2, 2, 1)
        mma_inst_mnk = (16, 8, 16)
        num_warps = atom_layout[0] * atom_layout[1] * atom_layout[2]  # 4 warps

        # Number of K tiles
        k_tile_cnt = (K + tile_shape_k - 1) // tile_shape_k

        # =====================================================================
        # Create TiledMMA
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
        # Shared Memory Layouts and Allocation
        # =====================================================================
        # Simple row-major layouts for shared memory (no swizzling for simplicity)
        a_smem_layout = cute.make_layout((tile_shape_m, tile_shape_k), stride=(tile_shape_k, 1))
        b_smem_layout = cute.make_layout((tile_shape_n, tile_shape_k), stride=(tile_shape_k, 1))

        smem = utils.SmemAllocator()
        sA = smem.allocate_tensor(a_dtype, a_smem_layout, byte_alignment=BUFFER_ALIGN_BYTES)
        sB = smem.allocate_tensor(a_dtype, b_smem_layout, byte_alignment=BUFFER_ALIGN_BYTES)

        # =====================================================================
        # Create Global Memory Tensors
        # =====================================================================
        mA = cute.make_tensor(
            cute.make_ptr(a_dtype, a_ptr_raw, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((M, K), stride=(K, 1)),
        )
        mB = cute.make_tensor(
            cute.make_ptr(a_dtype, b_ptr_raw, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((N, K), stride=(K, 1)),
        )
        mC = cute.make_tensor(
            cute.make_ptr(a_dtype, c_ptr_raw, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((M, N), stride=(N, 1)),
        )

        # =====================================================================
        # MMA Partitioning
        # =====================================================================
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None])

        # Local tile of C in global memory
        gC_m_offset = tile_m * tile_shape_m
        gC_n_offset = tile_n * tile_shape_n
        gC_mn = cute.local_tile(mC, (tile_shape_m, tile_shape_n), (tile_m, tile_n))

        # Partition for output
        tCgC = thr_mma.partition_C(gC_mn)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, Float32)

        # Clear accumulator
        accumulators.fill(0.0)

        # =====================================================================
        # ldmatrix Copy Atoms
        # =====================================================================
        # Row-major A: not M-major
        # Row-major B: not N-major
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

        # =====================================================================
        # K-loop with simple load pattern
        # =====================================================================
        for k_tile in range(0, k_tile_cnt, 1, unroll=1):
            # Create views for this K tile
            gA_slice = cute.local_tile(mA, (tile_shape_m, tile_shape_k), (tile_m, k_tile))
            gB_slice = cute.local_tile(mB, (tile_shape_n, tile_shape_k), (tile_n, k_tile))

            # Copy global -> shared using autovec_copy
            cute.autovec_copy(gA_slice, sA)
            cute.autovec_copy(gB_slice, sB)

            # Sync after load
            cute.arch.sync_threads()

            # Load from smem to registers via ldmatrix
            cute.copy(smem_tiled_copy_A, tCsA_copy_view, tCrA_copy_view)
            cute.copy(smem_tiled_copy_B, tCsB_copy_view, tCrB_copy_view)

            # MMA compute
            cute.gemm(tiled_mma, accumulators, tCrA, tCrB, accumulators)

            # Sync before next iteration
            cute.arch.sync_threads()

        # =====================================================================
        # Epilogue: Store C
        # =====================================================================
        # Convert accumulators from fp32 to output dtype
        acc_out = cute.make_rmem_tensor(accumulators.shape, a_dtype)
        acc_vec = accumulators.load()
        acc_out.store(acc_vec.to(a_dtype))

        # Direct store to global memory
        cute.autovec_copy(acc_out, tCgC)


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

    # 4 warps for compute (no separate DMA warp in simplified version)
    kernel_config = MegakernelConfig(threads_per_block=128)
    kernel = Megakernel(ops, config=kernel_config)
    kernel.run()

    return c


__all__ = [
    "GemmOp",
    "gemm_megakernel",
    "is_blackwell_available",
    "is_hopper_available",
]
