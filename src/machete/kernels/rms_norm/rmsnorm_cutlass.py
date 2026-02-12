# Copyright (c) 2025, Machete Authors
# Ported from NVIDIA CUTLASS example (examples/python/CuTeDSL/blackwell/rmsnorm.py)
# Original: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone CUTLASS RMSNorm kernel with cluster-based reduction.

Ported from the official NVIDIA CUTLASS example for use as a benchmark
reference implementation.

Features:
- Cluster-based reduction for large hidden dimensions (SM90+)
- 128-bit vectorized memory access via TiledCopy
- Hierarchical reduction: warp → block → cluster
- Architecture-specific tuning (SM80/SM90/SM100)

Note: This kernel requires SM90+ (Hopper or newer) GPUs.
"""

import functools
from typing import Optional, Union

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import Boolean, Float32, Int32, Int64
    from cutlass.cute.runtime import make_ptr

    from machete.kernels.utils.reduce import row_reduce

    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False


# =============================================================================
# Architecture Detection
# =============================================================================


def _is_hopper_or_newer():
    """Check if GPU is Hopper (SM90) or newer."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


@functools.lru_cache(maxsize=16)
def get_sm_version(device: Optional[Union[int, torch.device, str]] = None) -> int:
    """Get the SM (compute capability) version of a CUDA device."""
    if not torch.cuda.is_available():
        return 80
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


# =============================================================================
# Predicate Utility
# =============================================================================


if HAS_CUTLASS and _is_hopper_or_newer():

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

    # =========================================================================
    # RMSNorm Configuration
    # =========================================================================

    class RMSNormConfig:
        """Configuration for the RMSNorm kernel.

        Encapsulates all kernel configuration computed once at initialization.
        """

        COPY_BITS = 128  # 128-bit vectorized loads

        def __init__(
            self,
            dtype: type[cutlass.Numeric],
            N: int,
            has_weight: bool = True,
            sm_version: int | None = None,
        ):
            self.dtype = dtype
            self.N = N
            self.has_weight = has_weight
            self.sm_version = sm_version if sm_version is not None else get_sm_version()

            self.vec_size = self.COPY_BITS // dtype.width

            self.cluster_n = self._compute_cluster_n(N, dtype, self.sm_version)
            self.N_per_cta = N // self.cluster_n

            self.threads_per_row = self._compute_threads_per_row(self.N_per_cta)
            self.num_threads = self._compute_num_threads(self.N_per_cta)

            self.num_vec_blocks = max(
                1, (self.N_per_cta // self.vec_size + self.threads_per_row - 1) // self.threads_per_row
            )
            self.rows_per_block = self.num_threads // self.threads_per_row
            self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row
            self.warps_per_row = max(self.threads_per_row // 32, 1)

        @staticmethod
        def _compute_cluster_n(N: int, dtype: type[cutlass.Numeric], sm_version: int) -> int:
            """Compute optimal cluster size based on N and architecture."""
            if sm_version < 90:
                return 1

            if dtype.width == 16:  # FP16/BF16
                if N <= 16 * 1024:
                    return 1
                elif N <= 32 * 1024:
                    return 2
                elif N <= 64 * 1024:
                    return 4
                elif N <= 128 * 1024:
                    return 8
                else:
                    return 16
            else:  # FP32
                if N <= 32 * 1024:
                    return 1
                elif N <= 64 * 1024:
                    return 2
                elif N <= 128 * 1024:
                    return 4
                elif N <= 256 * 1024:
                    return 8
                else:
                    return 16

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
            """Calculate shared memory requirement in bytes."""
            tile_bytes = self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
            reduction_bytes = self.rows_per_block * self.warps_per_row * self.cluster_n * 4
            mbar_bytes = 8 if self.cluster_n > 1 else 0
            return tile_bytes + reduction_bytes + mbar_bytes

    # =========================================================================
    # RMSNorm Kernel
    # =========================================================================

    class RMSNormKernel:
        """RMSNorm kernel with cluster synchronization for large N.

        Features:
        - Cluster-based reduction for large N (SM90+)
        - Multiple CTAs cooperate via mbarrier
        - Single reduction (sum of squares) with cluster-level aggregation
        """

        def __init__(
            self,
            dtype: cutlass.Numeric,
            N: int,
            has_weight: bool = True,
            config: RMSNormConfig | None = None,
        ):
            if config is not None:
                self.cfg = config
            else:
                self.cfg = RMSNormConfig(dtype, N, has_weight)

            self.dtype = self.cfg.dtype
            self.N = self.cfg.N
            self.has_weight = self.cfg.has_weight
            self.cluster_n = self.cfg.cluster_n

        @cute.jit
        def __call__(
            self,
            x_ptr: cute.Pointer,
            w_ptr: cute.Pointer | None,
            o_ptr: cute.Pointer,
            M: Int32,
            eps: Float32,
            stream: cuda.CUstream,
        ):
            """Host function to launch the RMSNorm kernel."""
            cfg = self.cfg

            mX = cute.make_tensor(
                x_ptr,
                cute.make_layout((M, cfg.N), stride=(cfg.N, 1)),
            )
            mO = cute.make_tensor(
                o_ptr,
                cute.make_layout((M, cfg.N), stride=(cfg.N, 1)),
            )

            if cutlass.const_expr(cfg.has_weight and w_ptr is not None):
                mW = cute.make_tensor(
                    w_ptr,
                    cute.make_layout((cfg.N,), stride=(1,)),
                )
            else:
                mW = None

            tv_shape, tv_stride = RMSNormConfig._make_tv_layout(
                cfg.threads_per_row,
                cfg.rows_per_block,
                cfg.vec_size,
                cfg.num_vec_blocks,
            )
            tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
            tiler_mn = (cfg.rows_per_block, cfg.cols_per_tile)

            self.kernel(mX, mW, mO, eps, tv_layout, tiler_mn).launch(
                grid=[cute.ceil_div(M, cfg.rows_per_block), cfg.cluster_n, 1],
                block=[cfg.num_threads, 1, 1],
                cluster=[1, cfg.cluster_n, 1] if cutlass.const_expr(cfg.cluster_n > 1) else None,
                smem=cfg.smem_size_in_bytes(),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor | None,
            mO: cute.Tensor,
            eps: Float32,
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
        ):
            """Device kernel implementing RMSNorm with cluster support."""
            cfg = self.cfg
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()

            if cutlass.const_expr(cfg.cluster_n > 1):
                cluster_y = cute.arch.block_idx()[1]
            else:
                cluster_y = cutlass.const_expr(0)

            M = mX.shape[0]
            threads_per_row = tv_layout.shape[0][0]
            warps_per_row = max(threads_per_row // 32, 1)
            rows_per_block = tiler_mn[0]

            # =================================================================
            # Allocate shared memory
            # =================================================================
            smem = utils.SmemAllocator()

            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

            if cutlass.const_expr(cfg.cluster_n == 1):
                reduction_buffer = smem.allocate_tensor(
                    Float32,
                    cute.make_layout((rows_per_block, warps_per_row)),
                    byte_alignment=4,
                )
                mbar_ptr = None
            else:
                reduction_buffer = smem.allocate_tensor(
                    Float32,
                    cute.make_layout((rows_per_block, (warps_per_row, cfg.cluster_n))),
                    byte_alignment=4,
                )
                mbar_ptr = smem.allocate_array(Int64, num_elems=1)

            # =================================================================
            # Initialize cluster
            # =================================================================
            if cutlass.const_expr(cfg.cluster_n > 1):
                if tidx == 0:
                    cute.arch.mbarrier_init(mbar_ptr, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()

            # =================================================================
            # Create identity tensor and partition
            # =================================================================
            idX = cute.make_identity_tensor(mX.shape)

            gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
            gO = cute.local_tile(mO, tiler_mn, (bidx, cluster_y))
            cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

            if cutlass.const_expr(cfg.has_weight and mW is not None):
                mW_expanded_layout = cute.prepend(
                    mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
                )
                mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
                gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

            # =================================================================
            # Create TiledCopy operations
            # =================================================================
            copy_atom_load_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=RMSNormConfig.COPY_BITS,
            )

            copy_atom_load_W = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                mX.element_type,
                num_bits_per_copy=RMSNormConfig.COPY_BITS,
            )

            copy_atom_store = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                mO.element_type,
                num_bits_per_copy=RMSNormConfig.COPY_BITS,
            )

            tiled_copy_load = cute.make_tiled_copy(copy_atom_load_async, tv_layout, tiler_mn)
            tiled_copy_W = cute.make_tiled_copy(copy_atom_load_W, tv_layout, tiler_mn)
            tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

            thr_copy_X = tiled_copy_load.get_slice(tidx)
            thr_copy_W = tiled_copy_W.get_slice(tidx)
            thr_copy_O = tiled_copy_store.get_slice(tidx)

            # Partition tensors
            tXgX = thr_copy_X.partition_S(gX)
            tXsX = thr_copy_X.partition_D(sX)
            tXgO = thr_copy_O.partition_D(gO)
            tXcX = thr_copy_X.partition_S(cX)

            # Register fragments
            tXrX = cute.make_fragment_like(tXgX)
            tXrO = cute.make_fragment_like(tXgO)

            if cutlass.const_expr(cfg.has_weight and mW is not None):
                tWgW = thr_copy_W.partition_S(gW)
                tWrW = cute.make_fragment_like(tWgW)
                tXrW = thr_copy_X.retile(tWrW)

            # =================================================================
            # Bounds checking
            # =================================================================
            tXpX = predicate_k(tXcX, limit=cfg.N)

            row_coord = tXcX[(0, 0), 0, 0]
            row_in_bounds = row_coord[0] < M

            # =================================================================
            # Async copy global → shared
            # =================================================================
            if row_in_bounds:
                cute.copy(copy_atom_load_async, tXgX, tXsX, pred=tXpX)

            cute.arch.cp_async_commit_group()

            # Load weight while waiting
            if cutlass.const_expr(cfg.has_weight and mW is not None):
                tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=cfg.N)
                cute.copy(copy_atom_load_W, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            # =================================================================
            # Pass 1: Compute sum of squares with cluster reduction
            # =================================================================
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

            x_sq = x * x
            sum_sq = row_reduce(
                x_sq,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer,
                mbar_ptr,
                cfg.cluster_n,
                Float32(0.0),
            )

            # rstd = 1 / sqrt(mean(x²) + eps)
            mean_sq = sum_sq / cfg.N
            rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

            # Sync after reduction
            if cutlass.const_expr(cfg.cluster_n > 1):
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            else:
                cute.arch.barrier()

            # =================================================================
            # Pass 2: Normalize and output
            # =================================================================
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

            y = x * rstd

            # Apply weight if present
            if cutlass.const_expr(cfg.has_weight and mW is not None):
                w = tXrW.load().to(Float32)
                y = y * w

            # Store to global memory
            tXrO.store(y.to(cfg.dtype))

            if row_in_bounds:
                cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

    # =========================================================================
    # Public API
    # =========================================================================

    _torch_to_cutlass_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }

    _compile_cache: dict = {}

    def _get_compiled_kernel(dtype, N, has_weight, stream):
        key = (dtype, N, has_weight)
        if key not in _compile_cache:
            kernel_obj = RMSNormKernel(dtype, N, has_weight)

            compiled = cute.compile(
                kernel_obj,
                make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),  # x
                make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
                if has_weight
                else None,  # w
                make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),  # o
                Int32(1),
                Float32(1e-6),
                stream,
            )

            _compile_cache[key] = compiled
        return _compile_cache[key]

    def rmsnorm_cutlass(x, weight, eps=1e-6):
        """CUTLASS RMSNorm forward (SM90+ only).

        Cluster-based implementation with 128-bit vectorized memory access
        and hierarchical reduction (warp → block → cluster).

        Args:
            x: (M, D) float32/float16/bfloat16, CUDA.
            weight: (D,) same dtype, CUDA.
            eps: float

        Returns:
            y: (M, D) same dtype
        """
        M, N = x.shape
        out = torch.empty_like(x)

        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        dtype = _torch_to_cutlass_dtype.get(x.dtype, cutlass.Float32)
        has_weight = weight is not None

        compiled = _get_compiled_kernel(dtype, N, has_weight, stream)

        x_ptr = make_ptr(dtype, x.data_ptr())
        w_ptr = make_ptr(dtype, weight.data_ptr()) if has_weight else None
        out_ptr = make_ptr(dtype, out.data_ptr())

        compiled(x_ptr, w_ptr, out_ptr, Int32(M), Float32(eps), stream)
        return out

    HAS_CUTLASS_RMSNORM = True

else:
    HAS_CUTLASS_RMSNORM = False

    def rmsnorm_cutlass(x, weight, eps=1e-6):
        raise RuntimeError("CUTLASS RMSNorm requires SM90+ GPU and CUTLASS library")


__all__ = ["rmsnorm_cutlass", "HAS_CUTLASS_RMSNORM"]
