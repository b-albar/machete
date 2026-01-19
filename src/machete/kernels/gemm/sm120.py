# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Warp-specialized GEMM kernel for SM120 (Blackwell Geforce).

This kernel follows the Machete L/C/S pattern where:
- setup_host() creates TMA descriptors, layouts, and MMA configurations
- setup_kernel() unpacks host state and creates kernel-side structures
- load_forward() handles TMA loads (loader warps)
- compute_forward() handles MMA operations (consumer warps)
- store_forward() handles epilogue TMA stores (consumer warps)

The megakernel core inlines these methods into a single @cute.kernel,
preserving CuTe type information across the host->kernel boundary.
"""

import torch
from typing import Tuple, List, Any

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils

from quack.cute_dsl_utils import torch2cute_dtype_map
from machete.megakernel.interface import (
    WarpSpecializedKernel,
    TensorParam,
    reads,
    writes,
    warp_role,
    WarpRole,
)
from machete.megakernel.scheduler import WarpConfig
from machete.megakernel.single import SingleKernel


class Sm120GemmWS(WarpSpecializedKernel):
    """Warp-specialized GEMM kernel for SM120 using L/C/S interface.

    This implementation uses the Machete L/C/S pattern:
    1. setup_host() creates TMA atoms, layouts, MMA configs at compile time
    2. setup_kernel() unpacks host state inside the kernel
    3. load_forward/compute_forward/store_forward contain the actual implementation
    4. The megakernel core inlines these into a single @cute.kernel

    Matrix dimensions:
    - A: (L, M, K) row-major
    - B: (L, K, N) row-major
    - C: (L, M, N) row-major
    """

    def __init__(
        self,
        a_dtype: torch.dtype,
        b_dtype: torch.dtype,
        c_dtype: torch.dtype,
        acc_dtype: Any,
        tile_shape_mnk: Tuple[int, int, int],
    ):
        super().__init__()
        self.torch_a_dtype = a_dtype
        self.torch_b_dtype = b_dtype
        self.torch_c_dtype = c_dtype
        self.cute_a_dtype = torch2cute_dtype_map[a_dtype]
        self.cute_b_dtype = torch2cute_dtype_map[b_dtype]
        self.cute_c_dtype = torch2cute_dtype_map[c_dtype]
        self.acc_dtype = torch2cute_dtype_map[acc_dtype] if isinstance(acc_dtype, torch.dtype) else acc_dtype
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.ab_stage = 4
        self.epi_stage = 8
        # Kernel state set by setup_kernel, used by L/C/S methods
        self._kernel_state = None

    @property
    def tensor_params(self) -> List[TensorParam]:
        return [
            TensorParam("c", shape=(0, 1, 2), stride=(0, 1, 2), dtype_attr="cute_c_dtype"),
            TensorParam("a", shape=(0, 1, 2), stride=(0, 1, 2), dtype_attr="cute_a_dtype"),
            TensorParam("b", shape=(0, 1, 2), stride=(0, 1, 2), dtype_attr="cute_b_dtype"),
        ]

    @property
    def warp_config(self) -> WarpConfig:
        return WarpConfig(
            num_consumer_warps=4, num_loader_warps=1, num_storer_warps=0,
            num_launcher_warps=0, num_controller_warps=0
        )

    @property
    def smem_size_fwd(self) -> int:
        tm, tn, tk = self.tile_shape_mnk
        a_bytes = tm * tk * self.ab_stage * (self.cute_a_dtype.width // 8)
        b_bytes = tn * tk * self.ab_stage * (self.cute_b_dtype.width // 8)
        c_bytes = 1024 * self.epi_stage * (self.cute_c_dtype.width // 8)
        return (a_bytes + b_bytes + c_bytes + 32768) // 1024 * 1024

    def grid_fn(self, c, a, b, alpha, beta):
        M, N, K, L = self.shapes["M"], self.shapes["N"], self.shapes["K"], self.shapes["L"]
        tm, tn, _ = self.tile_shape_mnk
        return [(M + tm - 1) // tm * (N + tn - 1) // tn * L, 1, 1]

    def block_fn(self, *args):
        return [5 * 32, 1, 1]

    def _extract_shapes(self, args):
        """Extract M, N, K, L from tensor arguments."""
        c, a, b = args[0], args[1], args[2]
        if a.dim() == 2:
            return {"M": a.shape[0], "K": a.shape[1], "N": b.shape[1], "L": 1}
        else:
            return {"L": a.shape[0], "M": a.shape[1], "K": a.shape[2], "N": b.shape[2]}

    @cute.jit
    def setup_host(self, c, a, b):
        """Host-side setup: create TMA descriptors, layouts, and MMA config.

        This runs at compile time with CuTe tensors.
        Returns structures that maintain their CuTe types when passed to setup_kernel.
        """
        m_val, n_val, k_val, l_val = self.shapes["M"], self.shapes["N"], self.shapes["K"], self.shapes["L"]
        tm, tn, tk = const_expr(self.tile_shape_mnk)
        stages = const_expr(self.ab_stage)
        epi_stage = const_expr(self.epi_stage)

        a_layout = utils.LayoutEnum.ROW_MAJOR
        b_layout = utils.LayoutEnum.ROW_MAJOR
        c_layout = utils.LayoutEnum.ROW_MAJOR

        # Create staged smem layouts for A, B, C
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(a_layout, self.tile_shape_mnk, self.cute_a_dtype, stages)
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(b_layout, self.tile_shape_mnk, self.cute_b_dtype, stages)
        epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.cute_c_dtype, is_cooperative=False
        )
        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(self.cute_c_dtype, c_layout, epi_tile, epi_stage)

        # Define SharedStorage struct
        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, stages * 2]
            sA: cute.struct.Align[cute.struct.MemRange[self.cute_a_dtype, cute.cosize(a_smem_layout_staged)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[self.cute_b_dtype, cute.cosize(b_smem_layout_staged)], 1024]
            sC: cute.struct.Align[cute.struct.MemRange[self.cute_c_dtype, cute.cosize(epi_smem_layout_staged)], 1024]

        # TMA atoms need sliced view of smem (single stage)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        c_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))

        # Create TMA atoms for A, B, C
        t_a_atom, t_a_gmem = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), a, a_smem_layout, (tm, tk)
        )
        t_b_atom, t_b_gmem = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), b, b_smem_layout, (tn, tk)
        )
        t_c_atom, t_c_gmem = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), c, c_smem_layout, epi_tile
        )

        # Create global tile views
        gA_mkl = cute.local_tile(t_a_gmem, cute.slice_(self.tile_shape_mnk, (None, 0, None)), (None, None, None))
        gB_nkl = cute.local_tile(t_b_gmem, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None))
        gC_mnl = cute.local_tile(t_c_gmem, cute.slice_(self.tile_shape_mnk, (None, None, 0)), (None, None, None))

        # MMA configuration
        mma_inst_mnk = (16, 8, 16)
        atom_layout = (2, 2, 1)
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(self.cute_a_dtype, self.acc_dtype, mma_inst_mnk)
        permutation_mnk = (
            atom_layout[0] * mma_inst_mnk[0],
            atom_layout[1] * mma_inst_mnk[1] * 2,
            atom_layout[2] * mma_inst_mnk[2],
        )
        t_mma = cute.make_tiled_mma(mma_op, cute.make_layout(atom_layout), permutation_mnk=permutation_mnk)

        # LDMatrix copy atoms for A and B
        atom_copy_ldmatrix_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(a_layout.is_m_major_a(), 4), self.cute_a_dtype
        )
        atom_copy_ldmatrix_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(b_layout.is_n_major_b(), 4), self.cute_b_dtype
        )
        smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, t_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, t_mma)

        # Epilogue coordination barrier
        epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=4 * 32)

        return (
            SharedStorage,
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
            epi_tile,
            t_a_atom,
            t_b_atom,
            t_c_atom,
            gA_mkl,
            gB_nkl,
            gC_mnl,
            t_mma,
            smem_tiled_copy_A,
            smem_tiled_copy_B,
            epilog_sync_barrier,
            m_val,
            n_val,
            k_val,
            c_layout,
        )

    @cute.jit
    def setup_kernel(self, logical_idx, smem_alloc, host_state):
        """Kernel-side setup: unpack host state and create kernel structures.

        Called inside the generated @cute.kernel before warp dispatch.
        Creates pipeline, partitions tensors, returns state for L/C/S methods.
        """
        # Unpack host state
        (
            SharedStorage,
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
            epi_tile,
            t_a_atom,
            t_b_atom,
            t_c_atom,
            gA_mkl,
            gB_nkl,
            gC_mnl,
            t_mma,
            smem_tiled_copy_A,
            smem_tiled_copy_B,
            epilog_sync_barrier,
            m_val,
            n_val,
            k_val,
            c_layout,
        ) = host_state

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        tm, tn, tk = const_expr(self.tile_shape_mnk)
        stages = const_expr(self.ab_stage)
        epi_stage = const_expr(self.epi_stage)
        num_k_tiles = (k_val + tk - 1) // tk

        # Allocate shared memory
        storage = smem_alloc.allocate(SharedStorage)
        bar_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Create smem tensors with swizzling
        s_a = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        s_b = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        s_c = storage.sC.get_tensor(epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner)

        # Compute tile coordinates from linear block index
        grid_n = (n_val + tn - 1) // tn
        tiles_per_batch = ((m_val + tm - 1) // tm) * grid_n
        tile_in_batch = bidx % tiles_per_batch
        m_idx = tile_in_batch // grid_n
        n_idx = tile_in_batch % grid_n
        b_idx = bidx // tiles_per_batch

        # Get warp ID
        warp_idx_raw = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx_raw)

        # Prefetch TMA descriptors (warp 0 only)
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(t_a_atom)
            cute.nvgpu.cpasync.prefetch_descriptor(t_b_atom)
            cute.nvgpu.cpasync.prefetch_descriptor(t_c_atom)

        # Partition TMA for A and B
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            t_a_atom, Int32(0), cute.make_layout(1), cute.group_modes(s_a, 0, 2), cute.group_modes(gA_mkl, 0, 2)
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            t_b_atom, Int32(0), cute.make_layout(1), cute.group_modes(s_b, 0, 2), cute.group_modes(gB_nkl, 0, 2)
        )

        # Slice to current tile
        tAgA_tile = tAgA[(None, m_idx, None, b_idx)]
        tBgB_tile = tBgB[(None, n_idx, None, b_idx)]

        # Create TMA pipeline
        element_bytes = self.cute_a_dtype.width // 8
        tma_copy_bytes = (tm * tk + tn * tk) * element_bytes
        pipe = pipeline.PipelineTmaAsync.create(
            num_stages=stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
            tx_count=tma_copy_bytes,
            barrier_storage=bar_ptr,
        )

        # MMA partitioning
        thr_mma = t_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(s_a)
        tCsB = thr_mma.partition_B(s_b)
        tCrA = t_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = t_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCgC = thr_mma.partition_C(gC_mnl)
        acc = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)

        # LDMatrix partitioning
        thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
        thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
        tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(s_a)
        tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
        tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(s_b)
        tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)
        num_k_blocks = cute.size(tCrA, mode=[2])

        # Epilogue TMA partition for C - slice to current tile first (static shapes)
        # gC_mnl has shape (tile_M, tile_N, m_tiles, n_tiles, batches) from local_tile
        gC_tile = gC_mnl[(None, None, m_idx, n_idx, b_idx)]  # Single tile with static shape
        sepi_for_tma = cute.group_modes(s_c, 0, 2)
        tcgc_for_tma = cute.zipped_divide(gC_tile, epi_tile)
        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            t_c_atom, Int32(0), cute.make_layout(1), sepi_for_tma, cute.group_modes(tcgc_for_tma, 0, 2)
        )
        # Layout for hierarchical coordinate computation in epilogue loop
        epi_tile_shape = tcgc_for_tma.shape[1]
        epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(1, epi_tile_shape[0]))

        # Return kernel state for L/C/S methods
        return (
            # Pipeline and indices
            pipe, stages, num_k_tiles, tidx, warp_idx, m_idx, n_idx, b_idx,
            # TMA partitions
            t_a_atom, t_b_atom, t_c_atom, tAsA, tAgA_tile, tBsB, tBgB_tile,
            # Smem tensors
            s_a, s_b, s_c,
            # MMA structures
            t_mma, tCsA, tCsB, tCrA, tCrB, acc, num_k_blocks,
            # Copy structures
            smem_tiled_copy_A, smem_tiled_copy_B,
            tCsA_copy_view, tCrA_copy_view, tCsB_copy_view, tCrB_copy_view,
            # Epilogue structures
            gC_mnl, epi_tile, epi_stage, epilog_sync_barrier, c_layout,
            bSG_sD, bSG_gD, epi_tile_layout,
        )

    @warp_role(WarpRole.LOADER)
    @reads("a", "b")
    @cute.jit
    def load_forward(self, kernel_state, logical_idx, smem, c, a, b, alpha, beta):
        """Loader warp: TMA loads for A and B matrices."""
        (
            pipe, stages, num_k_tiles, tidx, warp_idx, m_idx, n_idx, b_idx,
            t_a_atom, t_b_atom, t_c_atom, tAsA, tAgA_tile, tBsB, tBgB_tile,
            s_a, s_b, s_c,
            t_mma, tCsA, tCsB, tCrA, tCrB, acc, num_k_blocks,
            smem_tiled_copy_A, smem_tiled_copy_B,
            tCsA_copy_view, tCrA_copy_view, tCsB_copy_view, tCrB_copy_view,
            gC_mnl, epi_tile, epi_stage, epilog_sync_barrier, c_layout,
        ) = kernel_state

        ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, stages)
        for k in range(num_k_tiles):
            pipe.producer_acquire(ps)
            if (tidx % 32) == 0:
                k_i32 = Int32(k)
                cute.copy(
                    t_a_atom,
                    tAgA_tile[(None, k_i32)],
                    tAsA[(None, ps.index)],
                    tma_bar_ptr=pipe.producer_get_barrier(ps),
                )
                cute.copy(
                    t_b_atom,
                    tBgB_tile[(None, k_i32)],
                    tBsB[(None, ps.index)],
                    tma_bar_ptr=pipe.producer_get_barrier(ps),
                )
            pipe.producer_commit(ps)
            ps.advance()

    @warp_role(WarpRole.CONSUMER)
    @writes("c")
    @cute.jit
    def compute_forward(self, kernel_state, logical_idx, smem, c, a, b, alpha, beta):
        """Consumer warps: MMA operations (mainloop)."""
        (
            pipe, stages, num_k_tiles, tidx, warp_idx, m_idx, n_idx, b_idx,
            t_a_atom, t_b_atom, t_c_atom, tAsA, tAgA_tile, tBsB, tBgB_tile,
            s_a, s_b, s_c,
            t_mma, tCsA, tCsB, tCrA, tCrB, acc, num_k_blocks,
            smem_tiled_copy_A, smem_tiled_copy_B,
            tCsA_copy_view, tCrA_copy_view, tCsB_copy_view, tCrB_copy_view,
            gC_mnl, epi_tile, epi_stage, epilog_sync_barrier, c_layout,
        ) = kernel_state

        acc.fill(0.0)
        cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, stages)

        # Wait for first tile
        pipe.consumer_wait(cs)
        tCsA_p = tCsA_copy_view[None, None, None, cs.index]
        tCsB_p = tCsB_copy_view[None, None, None, cs.index]
        cute.copy(smem_tiled_copy_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
        cute.copy(smem_tiled_copy_B, tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0])

        # Mainloop: k_tiles - 1 iterations with prefetch
        for k_tile in range(num_k_tiles - 1):
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                if k_block_idx == num_k_blocks - 1:
                    pipe.consumer_release(cs)
                    cs.advance()
                    pipe.consumer_wait(cs)
                    tCsA_p = tCsA_copy_view[None, None, None, cs.index]
                    tCsB_p = tCsB_copy_view[None, None, None, cs.index]
                cute.copy(smem_tiled_copy_A, tCsA_p[None, None, k_block_next], tCrA_copy_view[None, None, k_block_next])
                cute.copy(smem_tiled_copy_B, tCsB_p[None, None, k_block_next], tCrB_copy_view[None, None, k_block_next])
                cute.gemm(t_mma, acc, tCrA[None, None, k_block_idx], tCrB[None, None, k_block_idx], acc)

        # Last k_tile (no prefetch needed)
        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
            k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
            if k_block_idx == num_k_blocks - 1:
                pipe.consumer_release(cs)
                cs.advance()
            if k_block_next > 0:
                cute.copy(smem_tiled_copy_A, tCsA_p[None, None, k_block_next], tCrA_copy_view[None, None, k_block_next])
                cute.copy(smem_tiled_copy_B, tCsB_p[None, None, k_block_next], tCrB_copy_view[None, None, k_block_next])
            cute.gemm(t_mma, acc, tCrA[None, None, k_block_idx], tCrB[None, None, k_block_idx], acc)

    @warp_role(WarpRole.CONSUMER)
    @writes("c")
    @cute.jit
    def store_forward(self, kernel_state, logical_idx, smem, c, a, b, alpha, beta):
        """Consumer warps: Epilogue - store results via TMA.

        The TMA store uses the pre-partitioned tensors from setup_kernel.
        We iterate through epilogue tiles within this block's output tile.
        """
        (
            pipe, stages, num_k_tiles, tidx, warp_idx, m_idx, n_idx, b_idx,
            t_a_atom, t_b_atom, t_c_atom, tAsA, tAgA_tile, tBsB, tBgB_tile,
            s_a, s_b, s_c,
            t_mma, tCsA, tCsB, tCrA, tCrB, acc, num_k_blocks,
            smem_tiled_copy_A, smem_tiled_copy_B,
            tCsA_copy_view, tCrA_copy_view, tCsB_copy_view, tCrB_copy_view,
            gC_mnl, epi_tile, epi_stage, epilog_sync_barrier, c_layout,
            bSG_sD, bSG_gD, epi_tile_layout,
        ) = kernel_state

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            c_layout, elem_ty_d=self.cute_c_dtype, elem_ty_acc=self.acc_dtype
        )
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(c_layout.is_m_major_c(), 4), self.cute_c_dtype
        )
        tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, t_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)

        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(s_c)
        tRS_rAcc = tiled_copy_r2s.retile(acc)
        tRS_rD_layout = cute.make_layout(cute.shape(thr_copy_r2s.partition_S(s_c))[:3])
        tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)

        # Iterate through epilogue tiles within this block's output tile
        epi_tile_num = cute.size(tRS_sD, mode=[3])
        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.cute_c_dtype)
            tRS_rD_out.store((tRS_rD.load() * alpha).to(self.cute_c_dtype))

            epi_buffer = epi_idx % epi_stage
            cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)])

            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
            epilog_sync_barrier.arrive_and_wait()

            # Consumer warp 0 issues TMA store
            # Use hierarchical coordinate to index into the pre-partitioned global tensor
            if warp_idx == 0:
                gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                cute.copy(t_c_atom, bSG_sD[(None, epi_buffer)], bSG_gD[(None, gmem_coord)])

    def __call__(self, A, B, C=None, alpha=1.0, beta=0.0):
        if A.dim() == 2:
            A, B = A.unsqueeze(0), B.unsqueeze(0)
        if C is None:
            C = torch.zeros((A.shape[0], A.shape[1], B.shape[2]), dtype=A.dtype, device=A.device)

        # Extract shapes before launching
        self.shapes = self._extract_shapes([C, A, B])

        # Create SingleKernel wrapper and run
        sk = SingleKernel(self, self.grid_fn, self.block_fn)
        return sk.apply_autograd(C, A, B, alpha, beta)


def gemm_sm120(A, B, C=None, alpha=1.0, beta=0.0, tile_m=128, tile_n=128, tile_k=64):
    """Convenience function for SM120 GEMM.

    Args:
        A: Input matrix (M, K) or (L, M, K)
        B: Weight matrix (K, N) or (L, K, N)
        C: Optional output matrix
        alpha: Scale factor
        beta: Output scale (currently unused)
        tile_m, tile_n, tile_k: Tile dimensions

    Returns:
        C = alpha * A @ B
    """
    squeeze_o = A.dim() == 2
    res = Sm120GemmWS(A.dtype, B.dtype, (C.dtype if C is not None else A.dtype), Float32, (tile_m, tile_n, tile_k))(
        A, B, C, alpha, beta
    )
    return res.squeeze(0) if squeeze_o else res
