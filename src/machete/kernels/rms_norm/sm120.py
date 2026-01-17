# Copyright (c) 2025, Machete Authors
"""
RMSNorm kernel for SM 120 (Blackwell Geforce) with L/C/S decomposition.

RMSNorm computes: y = x / sqrt(mean(x²) + eps) * weight

Based on the official CUTLASS RMSNorm example - follows closely for maximum performance.
Uses L/C/S pattern for No Bubbles pipelining.
"""

import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Boolean, Int32

from quack.cute_dsl_utils import torch2cute_dtype_map
from machete.megakernel.interface import FusableKernel
from machete.megakernel.single import SingleKernel

# Import reduction utilities
from machete.kernels.utils import row_reduce, MacheteSmemAllocator


# =============================================================================
# Predicate Utility (same as CUTLASS version)
# =============================================================================


@cute.jit
def predicate_k(tXcX: cute.Tensor, limit: int) -> cute.Tensor:
    """Create predicate tensor for bounds checking."""
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
# RMSNorm Configuration (same as CUTLASS version)
# =============================================================================


class RMSNormConfig:
    """Configuration for the RMSNorm kernel - matches CUTLASS implementation."""

    COPY_BITS = 128  # 128-bit vectorized loads

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        n_hidden: int,
        has_weight: bool = True,
    ):
        self.dtype = dtype
        self.N = n_hidden
        self.has_weight = has_weight

        # Vector size for 128-bit loads
        self.vec_size = self.COPY_BITS // dtype.width

        # No cluster for SM120 (consumer Blackwell)
        self.cluster_n = 1
        self.N_per_cta = n_hidden

        # Thread configuration (matches CUTLASS)
        self.threads_per_row = self._compute_threads_per_row(self.N_per_cta)
        self.num_threads = self._compute_num_threads(self.N_per_cta)

        # Derived values
        self.num_vec_blocks = max(
            1, (self.N_per_cta // self.vec_size + self.threads_per_row - 1) // self.threads_per_row
        )
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

    @staticmethod
    def _compute_threads_per_row(N_per_cta: int) -> int:
        """Compute optimal threads per row based on N per CTA."""
        if N_per_cta <= 64:
            return 8
        elif N_per_cta <= 128:
            return 16
        elif N_per_cta <= 3072:
            return 32
        elif N_per_cta <= 6144:
            return 64
        elif N_per_cta <= 16384:
            return 128
        else:
            return 256

    @staticmethod
    def _compute_num_threads(N_per_cta: int) -> int:
        """Compute total threads per block."""
        return 128 if N_per_cta <= 16384 else 256

    @staticmethod
    def _make_tv_layout(
        threads_per_row: int,
        rows_per_block: int,
        vec_size: int,
        num_vec_blocks: int,
    ) -> tuple:
        """Create Thread-Value layout for coalesced vectorized memory access."""
        shape = (
            (threads_per_row, rows_per_block),
            (vec_size, num_vec_blocks),
        )
        stride = (
            (vec_size * rows_per_block, 1),
            (rows_per_block, rows_per_block * vec_size * threads_per_row),
        )
        return shape, stride

    def smem_size_in_bytes(self) -> int:
        """Calculate shared memory requirement in bytes (max of fwd/bwd)."""
        tile_bytes = self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
        reduction_bytes = self.rows_per_block * self.warps_per_row * 4
        # Backward needs 2x tile (for dout and x)
        bwd_tile_bytes = 2 * tile_bytes
        return bwd_tile_bytes + reduction_bytes


# =============================================================================
# RMSNorm SM120 Kernel with L/C/S decomposition
# =============================================================================


class RMSNormSM120(SingleKernel, FusableKernel):
    """
    RMSNorm kernel with L/C/S decomposition for SM 120 (Blackwell Geforce).

    Directly follows the CUTLASS RMSNorm implementation for maximum performance.
    The kernel uses a single compute phase since the load/compute/store are
    tightly coupled with the async copy pattern.
    """

    def __init__(self, dtype: torch.dtype, n_hidden: int, has_weight: bool = True):
        self.torch_dtype = dtype
        self.cute_dtype = torch2cute_dtype_map[dtype]
        self.n_hidden = n_hidden
        self.has_weight = has_weight

        # Create configuration (matches CUTLASS)
        self.cfg = RMSNormConfig(self.cute_dtype, n_hidden, has_weight)

        # Store layout parameters (created inside JIT)
        self.tv_shape, self.tv_stride = RMSNormConfig._make_tv_layout(
            self.cfg.threads_per_row,
            self.cfg.rows_per_block,
            self.cfg.vec_size,
            self.cfg.num_vec_blocks,
        )
        self.tiler_mn = (self.cfg.rows_per_block, self.cfg.cols_per_tile)

        SingleKernel.__init__(self, self, self.grid_fn, self.block_fn)

    @property
    def smem_size_fwd(self) -> int:
        """Shared memory size in bytes for forward pass."""
        return self.cfg.smem_size_in_bytes()

    @property
    def smem_size_bwd(self) -> int:
        """Shared memory size in bytes for backward pass."""
        return self.cfg.smem_size_in_bytes()

    # ========== Shared Memory Helper ==========

    @cute.jit
    def _partition_smem_fwd(self, smem, dtype):
        """Standardized partitioning for forward pass shared memory."""
        tiler_mn = cutlass.const_expr(self.tiler_mn)
        rows_per_block = tiler_mn[0]
        warps_per_row = cutlass.const_expr(self.cfg.warps_per_row)

        # Use paged allocator
        alloc = MacheteSmemAllocator(smem)
        sX = alloc.allocate_tensor(
            dtype,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer = alloc.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, warps_per_row)),
            byte_alignment=4,
        )
        return sX, reduction_buffer

    @cute.jit
    def _partition_smem_bwd(self, smem, dtype):
        """Standardized partitioning for backward pass shared memory."""
        tiler_mn = cutlass.const_expr(self.tiler_mn)
        rows_per_block = tiler_mn[0]
        warps_per_row = cutlass.const_expr(self.cfg.warps_per_row)

        alloc = MacheteSmemAllocator(smem)
        # Allocate dout and x tiles
        s_dout = alloc.allocate_tensor(
            dtype,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        s_x = alloc.allocate_tensor(
            dtype,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer = alloc.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, warps_per_row)),
            byte_alignment=4,
        )
        return s_dout, s_x, reduction_buffer

    # ========== Forward Pass L/C/S ==========

    @cute.jit
    def load_forward(self, logical_idx, smem, x_ptr, w_ptr, out_ptr, eps, n_rows):
        """
        Load phase: Async copy input x from global to shared memory.
        Also prefetch weight into registers.
        """
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # Create TV layout inside JIT
        tv_layout = cute.make_layout(cutlass.const_expr(self.tv_shape), stride=cutlass.const_expr(self.tv_stride))
        tiler_mn = cutlass.const_expr(self.tiler_mn)

        # Create tensor from pointer with static stride to guarantee alignment for async copy
        # This matches the original CUTLASS pattern
        N = cutlass.const_expr(cfg.N)
        mX = cute.make_tensor(
            x_ptr,
            cute.make_layout((n_rows, N), stride=(N, 1)),
        )

        # Use helper for consistent memory layout
        sX, _ = self._partition_smem_fwd(smem, mX.element_type)

        # Create identity tensor for bounds checking
        idX = cute.make_identity_tensor(mX.shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        # Create async copy atom
        copy_atom_load_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=RMSNormConfig.COPY_BITS,
        )
        tiled_copy_load = cute.make_tiled_copy(copy_atom_load_async, tv_layout, tiler_mn)
        thr_copy_X = tiled_copy_load.get_slice(tidx)

        # Partition tensors
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXcX = thr_copy_X.partition_S(cX)

        # Bounds checking
        tXpX = predicate_k(tXcX, limit=cfg.N)
        row_in_bounds = tXcX[(0, 0), 0, 0][0] < n_rows

        # Async copy global → shared
        if row_in_bounds:
            cute.copy(copy_atom_load_async, tXgX, tXsX, pred=tXpX)

        cute.arch.cp_async_commit_group()

    @cute.jit
    def compute_forward(self, logical_idx, smem, x_ptr, w_ptr, out_ptr, eps, n_rows):
        """
        Compute phase: RMS normalization.
        - Wait for async load
        - Compute sum of squares via reduction
        - Compute rstd and normalize
        - Store result to output tensor (part of compute for this kernel)
        """
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # Create TV layout inside JIT
        tv_layout = cute.make_layout(cutlass.const_expr(self.tv_shape), stride=cutlass.const_expr(self.tv_stride))
        tiler_mn = cutlass.const_expr(self.tiler_mn)
        threads_per_row = tv_layout.shape[0][0]

        # Create tensor from pointer with static stride
        N = cutlass.const_expr(cfg.N)
        mX = cute.make_tensor(
            x_ptr,
            cute.make_layout((n_rows, N), stride=(N, 1)),
        )

        # Use helper for consistent memory layout
        sX, reduction_buffer = self._partition_smem_fwd(smem, mX.element_type)

        # Weights handling - create tensor from pointer
        if cutlass.const_expr(cfg.has_weight and w_ptr is not None):
            mW = cute.make_tensor(w_ptr, cute.make_layout((N,), stride=(1,)))
            mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
            mW_2d = cute.make_tensor(w_ptr, mW_expanded_layout)
            gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))

        # Create copy atoms
        copy_atom_load_W = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=RMSNormConfig.COPY_BITS,
        )
        tiled_copy_load = cute.make_tiled_copy(
            cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=RMSNormConfig.COPY_BITS
            ),
            tv_layout,
            tiler_mn,
        )
        tiled_copy_W = cute.make_tiled_copy(copy_atom_load_W, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)

        # Partition tensors
        tXsX = thr_copy_X.partition_D(sX)
        tXrX = cute.make_fragment_like(tXsX)

        if cutlass.const_expr(cfg.has_weight and w_ptr is not None):
            tXgW = thr_copy_W.partition_S(gW)
            tXrW = cute.make_fragment_like(tXgW)
            tXrW_retiled = thr_copy_X.retile(tXrW)

        # Wait for async load from load_forward
        cute.arch.cp_async_wait_group(0)

        # Load weight while waiting for async copy
        if cutlass.const_expr(cfg.has_weight and w_ptr is not None):
            idX = cute.make_identity_tensor(mX.shape)
            tXcX_W = thr_copy_W.partition_S(cute.local_tile(idX, tiler_mn, (bidx, 0)))
            tWpW = predicate_k(tXcX_W, limit=cfg.N)
            cute.copy(copy_atom_load_W, tXgW, tXrW, pred=tWpW)

        # =====================================================================
        # Pass 1: Compute sum of squares
        # =====================================================================
        cute.autovec_copy(tXsX, tXrX)
        x_val = tXrX.load().to(Float32)

        x_sq = x_val * x_val
        sum_sq = row_reduce(
            x_sq,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer,
            None,
            cutlass.const_expr(1),
            Float32(0.0),
        )

        # rstd = 1 / sqrt(mean(x²) + eps)
        mean_sq = sum_sq / cfg.N
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        # =====================================================================
        # Pass 2: Normalize and compute output
        # =====================================================================
        y_val = x_val * rstd

        # Apply weight if present
        if cutlass.const_expr(cfg.has_weight and w_ptr is not None):
            w = tXrW_retiled.load().to(Float32)
            y_val = y_val * w

        # Write back to shared memory so it can be picked up by store_forward
        tXrX.store(y_val.to(cfg.dtype))
        cute.copy(copy_atom_load_W, tXrX, tXsX)

    @cute.jit
    def store_forward(self, logical_idx, smem, x_ptr, w_ptr, out_ptr, eps, n_rows):
        """
        Store phase: Write the normalized result from shared memory to global memory.
        """
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # Create TV layout inside JIT
        tv_layout = cute.make_layout(cutlass.const_expr(self.tv_shape), stride=cutlass.const_expr(self.tv_stride))
        tiler_mn = cutlass.const_expr(self.tiler_mn)

        # Create tensor from pointer with static stride
        N = cutlass.const_expr(cfg.N)
        mO = cute.make_tensor(
            out_ptr,
            cute.make_layout((n_rows, N), stride=(N, 1)),
        )

        # Allocate shared memory with proper alignment (same as compute_forward)
        sX, _ = self._partition_smem_fwd(smem, mO.element_type)

        # Create identity tensor and partitions
        idX = cute.make_identity_tensor(mO.shape)
        gO = cute.local_tile(mO, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        # Create store copy atom (Universal)
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mO.element_type,
            num_bits_per_copy=RMSNormConfig.COPY_BITS,
        )
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # Partition tensors
        tXsX = thr_copy_O.partition_S(sX)
        tXgO = thr_copy_O.partition_D(gO)
        tXcX = thr_copy_O.partition_S(cX)

        # Bounds checking
        tXpX = predicate_k(tXcX, limit=cfg.N)
        row_in_bounds = tXcX[(0, 0), 0, 0][0] < n_rows

        # Write from shared memory to global memory
        if row_in_bounds:
            cute.copy(copy_atom_store, tXsX, tXgO, pred=tXpX)

    # ========== Backward Pass L/C/S ==========

    @cute.jit
    def load_backward(self, logical_idx, smem, dout_ptr, x_ptr, w_ptr, dx_ptr, eps, n_rows):
        """
        Load phase: Async copy dout and x from global to shared memory.
        """
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # Create TV layout inside JIT
        tv_layout = cute.make_layout(cutlass.const_expr(self.tv_shape), stride=cutlass.const_expr(self.tv_stride))
        tiler_mn = cutlass.const_expr(self.tiler_mn)

        # Create tensors from pointers with static stride
        N = cutlass.const_expr(cfg.N)
        mDout = cute.make_tensor(
            dout_ptr,
            cute.make_layout((n_rows, N), stride=(N, 1)),
        )
        mX = cute.make_tensor(
            x_ptr,
            cute.make_layout((n_rows, N), stride=(N, 1)),
        )

        s_dout, s_x, _ = self._partition_smem_bwd(smem, mDout.element_type)

        # Create identity tensor and partitions
        id_dout = cute.make_identity_tensor(mDout.shape)
        g_dout = cute.local_tile(mDout, tiler_mn, (bidx, 0))
        g_x = cute.local_tile(mX, tiler_mn, (bidx, 0))
        c_dout = cute.local_tile(id_dout, tiler_mn, (bidx, 0))

        # Create async copy atom
        copy_atom_load_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mDout.element_type,
            num_bits_per_copy=RMSNormConfig.COPY_BITS,
        )
        tiled_copy_load = cute.make_tiled_copy(copy_atom_load_async, tv_layout, tiler_mn)
        thr_copy_dout = tiled_copy_load.get_slice(tidx)

        # Partition tensors
        t_g_dout = thr_copy_dout.partition_S(g_dout)
        t_s_dout = thr_copy_dout.partition_D(s_dout)
        t_g_x = thr_copy_dout.partition_S(g_x)
        t_s_x = thr_copy_dout.partition_D(s_x)
        t_c_dout = thr_copy_dout.partition_S(c_dout)

        # Bounds checking
        t_pred = predicate_k(t_c_dout, limit=cfg.N)
        row_in_bounds = t_c_dout[(0, 0), 0, 0][0] < n_rows

        # Async copy global → shared
        if row_in_bounds:
            cute.copy(copy_atom_load_async, t_g_dout, t_s_dout, pred=t_pred)
            cute.copy(copy_atom_load_async, t_g_x, t_s_x, pred=t_pred)

        cute.arch.cp_async_commit_group()

    @cute.jit
    def compute_backward(self, logical_idx, smem, dout_ptr, x_ptr, w_ptr, dx_ptr, eps, n_rows):
        """
        Compute phase: Compute dx = (dout * w - x * rstd² * sum(dout * x * w) / N) * rstd
        """
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # Create TV layout inside JIT
        tv_layout = cute.make_layout(cutlass.const_expr(self.tv_shape), stride=cutlass.const_expr(self.tv_stride))
        tiler_mn = cutlass.const_expr(self.tiler_mn)
        threads_per_row = tv_layout.shape[0][0]

        # Create tensor from pointer with static stride
        N = cutlass.const_expr(cfg.N)
        mDout = cute.make_tensor(
            dout_ptr,
            cute.make_layout((n_rows, N), stride=(N, 1)),
        )

        s_dout, s_x, reduction_buffer = self._partition_smem_bwd(smem, mDout.element_type)

        # Weight handling - create tensor from pointer
        if cutlass.const_expr(cfg.has_weight and w_ptr is not None):
            mW = cute.make_tensor(w_ptr, cute.make_layout((N,), stride=(1,)))
            mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
            mW_2d = cute.make_tensor(w_ptr, mW_expanded_layout)
            g_w = cute.local_tile(mW_2d, tiler_mn, (0, 0))

        # Create copy atoms
        copy_atom_load_W = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mDout.element_type, num_bits_per_copy=RMSNormConfig.COPY_BITS
        )
        tiled_copy_load = cute.make_tiled_copy(
            cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), mDout.element_type, num_bits_per_copy=RMSNormConfig.COPY_BITS
            ),
            tv_layout,
            tiler_mn,
        )
        tiled_copy_W = cute.make_tiled_copy(copy_atom_load_W, tv_layout, tiler_mn)

        thr_copy_dout = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)

        # Partition tensors
        t_s_dout = thr_copy_dout.partition_D(s_dout)
        t_s_x = thr_copy_dout.partition_D(s_x)

        # Register fragments
        t_r_dout = cute.make_fragment_like(t_s_dout)
        t_r_x = cute.make_fragment_like(t_s_x)

        if cutlass.const_expr(cfg.has_weight and w_ptr is not None):
            t_g_w = thr_copy_W.partition_S(g_w)
            t_r_w = cute.make_fragment_like(t_g_w)
            t_r_w_retiled = thr_copy_dout.retile(t_r_w)

        # Wait for async load
        cute.arch.cp_async_wait_group(0)

        # Load weights while sync is happening if needed
        if cutlass.const_expr(cfg.has_weight and w_ptr is not None):
            id_dout = cute.make_identity_tensor(mDout.shape)
            t_c_dout_W = thr_copy_W.partition_S(cute.local_tile(id_dout, tiler_mn, (bidx, 0)))
            t_pred_w = predicate_k(t_c_dout_W, limit=cfg.N)
            cute.copy(copy_atom_load_W, t_g_w, t_r_w, pred=t_pred_w)

        # =====================================================================
        # Pass 1: Compute rstd
        # =====================================================================
        cute.autovec_copy(t_s_x, t_r_x)
        x_val = t_r_x.load().to(Float32)
        sum_sq = row_reduce(
            x_val * x_val,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer,
            None,
            cutlass.const_expr(1),
            Float32(0.0),
        )
        rstd = cute.math.rsqrt((sum_sq / cfg.N) + eps, fastmath=True)
        cute.arch.barrier()

        # =====================================================================
        # Pass 2: Compute sum(dout * weight * x)
        # =====================================================================
        cute.autovec_copy(t_s_dout, t_r_dout)
        dout_val = t_r_dout.load().to(Float32)

        # Term: dout * w
        dw_x = dout_val
        if cutlass.const_expr(cfg.has_weight and w_ptr is not None):
            w_val = t_r_w_retiled.load().to(Float32)
            dw_x = dout_val * w_val

        sum_grad = row_reduce(
            dw_x * x_val,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer,
            None,
            cutlass.const_expr(1),
            Float32(0.0),
        )
        cute.arch.barrier()

        # =====================================================================
        # Pass 3: Compute dx and store back to s_dout
        # =====================================================================
        # dx = (dout * w - x * rstd² * (sum(dout * x * w) / N)) * rstd
        mean_grad = sum_grad / cfg.N
        dx_val = (dw_x - x_val * (rstd * rstd * mean_grad)) * rstd

        t_r_dx = cute.make_fragment_like(t_s_dout)
        t_r_dx.store(dx_val.to(cfg.dtype))
        cute.copy(copy_atom_load_W, t_r_dx, t_s_dout)

    @cute.jit
    def store_backward(self, logical_idx, smem, dout_ptr, x_ptr, w_ptr, dx_ptr, eps, n_rows):
        """
        Store phase: Write dx from shared memory to global memory.
        """
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # Create TV layout inside JIT
        tv_layout = cute.make_layout(cutlass.const_expr(self.tv_shape), stride=cutlass.const_expr(self.tv_stride))
        tiler_mn = cutlass.const_expr(self.tiler_mn)

        # Create tensor from pointer with static stride
        N = cutlass.const_expr(cfg.N)
        mDx = cute.make_tensor(
            dx_ptr,
            cute.make_layout((n_rows, N), stride=(N, 1)),
        )

        s_dx, _, _ = self._partition_smem_bwd(smem, mDx.element_type)

        # Create identity tensor and partitions
        id_dx = cute.make_identity_tensor(mDx.shape)
        g_dx = cute.local_tile(mDx, tiler_mn, (bidx, 0))
        c_dx = cute.local_tile(id_dx, tiler_mn, (bidx, 0))

        # Create store copy atom
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mDx.element_type, num_bits_per_copy=RMSNormConfig.COPY_BITS
        )
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)
        thr_copy_dx = tiled_copy_store.get_slice(tidx)

        # Partition tensors
        t_s_dx = thr_copy_dx.partition_S(s_dx)
        t_g_dx = thr_copy_dx.partition_D(g_dx)
        t_c_dx = thr_copy_dx.partition_S(c_dx)

        # Bounds checking
        t_pred = predicate_k(t_c_dx, limit=cfg.N)
        row_in_bounds = t_c_dx[(0, 0), 0, 0][0] < n_rows

        # Write from shared memory to global memory
        if row_in_bounds:
            cute.copy(copy_atom_store, t_s_dx, t_g_dx, pred=t_pred)

    # ========== Launch Helpers ==========

    def grid_fn(self, *args):
        """Grid: one block per rows_per_block rows."""
        m_x = args[0]
        n_rows = m_x.shape[0]
        rows_per_block = self.cfg.rows_per_block
        # Use Python ceil division instead of cute.ceil_div
        n_blocks = (n_rows + rows_per_block - 1) // rows_per_block
        return [n_blocks, 1, 1]

    def block_fn(self, *args):
        return [self.cfg.num_threads, 1, 1]

    def run_forward(self, ctx, x_2d, weight, out, eps, n_rows):
        ctx.save_for_backward(x_2d, weight)
        ctx.eps = eps
        ctx.n_rows = n_rows

        args = (x_2d, weight, out, eps, n_rows)
        grid = self.grid_fn(*args)
        block = self.block_fn(*args)
        n_blocks = grid[0] * grid[1] * grid[2]

        self._update_or_add(self.mk_fwd, args)
        self.mk_fwd.launch(n_blocks, grid, block)
        return out

    def run_backward(self, ctx, dout_2d):
        x_2d, weight = ctx.saved_tensors
        eps = ctx.eps
        n_rows = ctx.n_rows

        dx = torch.empty_like(x_2d)
        args = (dout_2d, x_2d, weight, dx, eps, n_rows)
        grid = self.grid_fn(*args)
        block = self.block_fn(*args)
        n_blocks = grid[0] * grid[1] * grid[2]

        self._update_or_add(self.mk_bwd, args)
        self.mk_bwd.launch(n_blocks, grid, block)
        # Returns dx, dweight (None for now), out (None), eps (None), n_rows (None)
        return dx, None, None, None, None

    def __call__(self, x: Tensor, weight: Tensor | None = None, eps: float = 1e-6) -> Tensor:
        """Convenience call for forward pass."""
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.n_hidden)
        n_rows = x_2d.shape[0]

        out = torch.empty_like(x_2d)

        # Create dummy weight if not provided
        if weight is None and self.has_weight:
            weight = torch.ones(self.n_hidden, dtype=x.dtype, device=x.device)

        return self.apply_autograd(x_2d, weight, out, Float32(eps), Int32(n_rows)).view(orig_shape)


# Convenience function for direct use
_kernel_cache: dict[tuple, RMSNormSM120] = {}


def rms_norm_sm120(x: Tensor, weight: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    """Convenience function for RMSNorm forward pass."""
    n_hidden = x.shape[-1]
    key = (x.dtype, n_hidden, weight is not None)

    if key not in _kernel_cache:
        _kernel_cache[key] = RMSNormSM120(x.dtype, n_hidden, has_weight=(weight is not None))

    return _kernel_cache[key](x, weight, eps)
