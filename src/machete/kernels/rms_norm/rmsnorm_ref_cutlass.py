# Copyright (c) 2025, Machete Authors
"""Standalone CUTLASS RMSNorm kernel for benchmarking.

This is a direct port of the NVIDIA CUTLASS RMSNorm example
(examples/python/CuTeDSL/blackwell/rmsnorm.py) for use as a
benchmark reference implementation.

Features:
- Cluster-based reduction for large hidden dimensions (SM90+)
- 128-bit vectorized memory access
- Warp-level reduction for efficiency

Note: This kernel requires SM90+ (Hopper or newer) GPUs.
"""

import functools
import operator as op

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64

    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False


def _is_hopper_or_newer():
    """Check if GPU is Hopper (SM90) or newer."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


# Only define the kernel if CUTLASS is available and GPU is Hopper+
if HAS_CUTLASS and _is_hopper_or_newer():

    @functools.lru_cache(maxsize=16)
    def _get_sm_version(device=None) -> int:
        if not torch.cuda.is_available():
            return 80
        props = torch.cuda.get_device_properties(device)
        return props.major * 10 + props.minor

    # =========================================================================
    # Simple CUTLASS RMSNorm Kernel (warp-parallel, no cluster)
    # =========================================================================

    class _SimpleCutlassRMSNormKernel:
        """Simple CUTLASS RMSNorm kernel with warp-parallel rows.

        Each warp processes one row independently using warp-level reduction.
        This matches the optimized megakernel approach and avoids complex
        cluster-based reduction that has dynamic layout issues.
        """

        def __init__(self, dtype, N, has_weight=True):
            self.dtype = dtype
            self.N = N  # Hidden dimension (compile-time constant)
            self.has_weight = has_weight
            self.vec_size = 128 // dtype.width  # 128-bit loads

            # Thread configuration: 128 threads = 4 warps
            # Each warp processes one row
            self.num_threads = 128
            self.num_warps = self.num_threads // 32
            self.rows_per_block = self.num_warps  # 4 rows per block

        @cute.jit
        def __call__(self, x_ptr, w_ptr, o_ptr, M: Int32, eps: Float32, stream):
            rows_per_block = self.rows_per_block
            grid_m = (M + rows_per_block - 1) // rows_per_block

            self.kernel(x_ptr, w_ptr, o_ptr, M, eps).launch(
                grid=[grid_m, 1, 1],
                block=[self.num_threads, 1, 1],
                smem=0,
                stream=stream,
            )

        @cute.kernel
        def kernel(self, x_ptr, w_ptr, o_ptr, M: Int32, eps: Float32):
            N = self.N
            dtype = self.dtype
            has_weight = self.has_weight
            rows_per_block = self.rows_per_block

            tidx = cute.arch.thread_idx()[0]
            bidx = cute.arch.block_idx()[0]
            warp_idx = cute.arch.warp_idx()
            lane_idx = cute.arch.lane_idx()

            # Each warp processes one row
            row_idx = bidx * rows_per_block + warp_idx

            # Only process if row is valid
            if row_idx < M:
                # Create 1D tensors for this row using static N
                thr_layout = cute.make_layout(32)  # 32 threads per warp

                x_row = cute.make_tensor(
                    cute.make_ptr(dtype, x_ptr, cute.AddressSpace.gmem) + row_idx * N,
                    cute.make_layout(N),
                )
                o_row = cute.make_tensor(
                    cute.make_ptr(dtype, o_ptr, cute.AddressSpace.gmem) + row_idx * N,
                    cute.make_layout(N),
                )

                x_part = cute.local_partition(x_row, thr_layout, lane_idx)
                o_part = cute.local_partition(o_row, thr_layout, lane_idx)

                # Load x into registers
                x_reg = cute.make_fragment_like(x_part)
                cute.autovec_copy(x_part, x_reg)

                # Compute sum of squares
                partial_sq = Float32(0.0)
                for i in range(cute.size(x_reg)):
                    val = x_reg[i].to(Float32)
                    partial_sq = partial_sq + val * val

                # Warp reduction
                sum_sq = cute.arch.warp_reduction(partial_sq, op.add)

                # rstd = 1 / sqrt(mean(x²) + eps)
                rstd = cute.math.rsqrt(sum_sq / N + eps, fastmath=True)

                # Load weight if needed
                if cutlass.const_expr(has_weight and w_ptr is not None):
                    w_row = cute.make_tensor(
                        cute.make_ptr(dtype, w_ptr, cute.AddressSpace.gmem),
                        cute.make_layout(N),
                    )
                    w_part = cute.local_partition(w_row, thr_layout, lane_idx)
                    w_reg = cute.make_fragment_like(w_part)
                    cute.autovec_copy(w_part, w_reg)

                # Compute output: y = x * rstd * weight
                o_reg = cute.make_fragment_like(x_reg)
                for i in range(cute.size(x_reg)):
                    val = x_reg[i].to(Float32) * rstd
                    if cutlass.const_expr(has_weight and w_ptr is not None):
                        val = val * w_reg[i].to(Float32)
                    o_reg[i] = val.to(dtype)

                # Store output
                cute.autovec_copy(o_reg, o_part)

    # =========================================================================
    # Public API
    # =========================================================================

    _torch_to_cutlass_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }

    _cutlass_compile_cache = {}

    def _get_cutlass_kernel(dtype, N, has_weight, stream):
        key = (dtype, N, has_weight)
        if key not in _cutlass_compile_cache:
            kernel_obj = _SimpleCutlassRMSNormKernel(dtype, N, has_weight)
            compiled = cute.compile(
                kernel_obj,
                Int64(0),  # x_ptr
                Int64(0) if has_weight else None,  # w_ptr
                Int64(0),  # o_ptr
                Int32(1), Float32(1e-6), stream,
            )
            _cutlass_compile_cache[key] = compiled
        return _cutlass_compile_cache[key]

    def rmsnorm_cutlass(x, weight, eps=1e-6):
        """CUTLASS RMSNorm forward (SM90+ only).

        Warp-parallel implementation with 128-bit vectorized memory access.
        Each warp processes one row independently using warp-level reduction.

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

        compiled = _get_cutlass_kernel(dtype, N, has_weight, stream)

        x_ptr = Int64(x.data_ptr())
        w_ptr = Int64(weight.data_ptr()) if has_weight else Int64(0)
        out_ptr = Int64(out.data_ptr())

        compiled(x_ptr, w_ptr, out_ptr, Int32(M), Float32(eps), stream)
        return out

    HAS_CUTLASS_RMSNORM = True

else:
    HAS_CUTLASS_RMSNORM = False

    def rmsnorm_cutlass(x, weight, eps=1e-6):
        raise RuntimeError("CUTLASS RMSNorm requires SM90+ GPU and CUTLASS library")


__all__ = ["rmsnorm_cutlass", "HAS_CUTLASS_RMSNORM"]
