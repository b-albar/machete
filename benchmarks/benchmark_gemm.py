# Copyright (c) 2025, Machete Authors
"""Benchmark SM120 GEMM kernel against PyTorch."""
import torch
from machete.utils.testing import benchmark_op, clear_kernel_caches


def gemm_pytorch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference GEMM.

    Computes: C = A @ B.T
    Where B is (N, K) - K-major layout (transposed).
    """
    # Handle batched vs non-batched
    if a.dim() == 2:
        # B is (N, K), we need (K, N) for matmul
        return torch.mm(a, b.t())
    else:
        # Batched: A is (L, M, K), B is (L, N, K)
        return torch.bmm(a, b.transpose(-2, -1))


def gemm_pytorch_cublas(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    PyTorch GEMM using torch.matmul (cuBLAS backend).

    This should use the optimized cuBLAS path.
    """
    if a.dim() == 2:
        return a @ b.t()
    else:
        return a @ b.transpose(-2, -1)


def main():
    device = "cuda"
    dtype = torch.float16

    print("=" * 70)
    print("SM120 GEMM Benchmark")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    # Import GEMM kernel
    try:
        from machete.kernels.gemm.sm120 import GemmSm120, gemm_sm120
    except ImportError as e:
        print(f"Failed to import SM120 GEMM kernel: {e}")
        print("Make sure the kernel is properly installed.")
        return

    # Create kernel instances with different tile sizes
    kernel_default = GemmSm120(dtype=dtype)
    kernel_small = GemmSm120(dtype=dtype, tile_m=64, tile_n=64, tile_k=32)

    # --- Square Matrix Benchmark ---
    def get_square_configs():
        """Square matrix configurations (M=N=K)."""
        sizes = [256, 512, 1024, 2048, 4096]

        for size in sizes:
            name = f"M=N=K={size}"
            try:
                a = torch.randn(size, size, device=device, dtype=dtype)
                b = torch.randn(size, size, device=device, dtype=dtype)
                yield name, (a, b)
                del a, b
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                yield name, None
                torch.cuda.empty_cache()
                break

    square_op_map = {
        "PyTorch": lambda a, b: gemm_pytorch(a, b),
        "cuBLAS": lambda a, b: gemm_pytorch_cublas(a, b),
        "SM120-Default": lambda a, b: kernel_default(a, b),
        "SM120-Small": lambda a, b: kernel_small(a, b),
    }

    def numel_square(args):
        a, b = args
        m, k = a.shape
        n = b.shape[0]
        # Memory: A (M*K) + B (N*K) + C (M*N)
        return m * k + n * k + m * n

    benchmark_op("Square GEMM (M=N=K)", get_square_configs(), square_op_map, numel_square)
    clear_kernel_caches()

    # --- Rectangular Matrix Benchmark ---
    def get_rect_configs():
        """Rectangular matrix configurations (typical LLM shapes)."""
        configs = [
            (1024, 4096, 1024),    # Typical attention projection
            (4096, 4096, 1024),    # Larger M
            (1024, 11008, 4096),   # LLaMA-7B FFN shapes
            (4096, 11008, 4096),   # Larger batch
            (8192, 4096, 4096),    # Large batch, medium dims
        ]

        for m, n, k in configs:
            name = f"M={m} N={n} K={k}"
            try:
                a = torch.randn(m, k, device=device, dtype=dtype)
                b = torch.randn(n, k, device=device, dtype=dtype)
                yield name, (a, b)
                del a, b
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                yield name, None
                torch.cuda.empty_cache()
                break

    rect_op_map = {
        "PyTorch": lambda a, b: gemm_pytorch(a, b),
        "cuBLAS": lambda a, b: gemm_pytorch_cublas(a, b),
        "SM120": lambda a, b: gemm_sm120(a, b),
    }

    def numel_rect(args):
        a, b = args
        m, k = a.shape
        n = b.shape[0]
        return m * k + n * k + m * n

    benchmark_op("Rectangular GEMM", get_rect_configs(), rect_op_map, numel_rect)
    clear_kernel_caches()

    # --- Batched GEMM Benchmark ---
    def get_batched_configs():
        """Batched GEMM configurations."""
        configs = [
            (8, 512, 512, 512),     # Small batched
            (16, 512, 512, 512),    # Medium batched
            (32, 256, 256, 256),    # Larger batch, smaller dims
            (4, 1024, 1024, 1024),  # Small batch, large dims
        ]

        for batch, m, n, k in configs:
            name = f"L={batch} M={m} N={n} K={k}"
            try:
                a = torch.randn(batch, m, k, device=device, dtype=dtype)
                b = torch.randn(batch, n, k, device=device, dtype=dtype)
                yield name, (a, b)
                del a, b
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                yield name, None
                torch.cuda.empty_cache()
                break

    batched_op_map = {
        "PyTorch": lambda a, b: gemm_pytorch(a, b),
        "cuBLAS": lambda a, b: gemm_pytorch_cublas(a, b),
        "SM120": lambda a, b: kernel_default(a, b),
    }

    def numel_batched(args):
        a, b = args
        batch, m, k = a.shape
        n = b.shape[1]
        return batch * (m * k + n * k + m * n)

    benchmark_op("Batched GEMM", get_batched_configs(), batched_op_map, numel_batched)
    clear_kernel_caches()

    # --- TFLOPS Comparison ---
    print("\n" + "=" * 70)
    print("TFLOPS Comparison (FP16)")
    print("=" * 70)
    print(f"{'Config':<25} | {'PyTorch TFLOPS':<15} | {'SM120 TFLOPS':<15} | {'Speedup':<10}")
    print("-" * 70)

    try:
        import triton

        tflops_configs = [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ]

        for m, n, k in tflops_configs:
            try:
                a = torch.randn(m, k, device=device, dtype=dtype)
                b = torch.randn(n, k, device=device, dtype=dtype)

                # Compute FLOPs: 2 * M * N * K (multiply-add)
                flops = 2.0 * m * n * k

                # Warmup
                for _ in range(5):
                    _ = gemm_pytorch_cublas(a, b)
                    _ = gemm_sm120(a, b)

                # Benchmark PyTorch
                ms_pytorch = triton.testing.do_bench(lambda: gemm_pytorch_cublas(a, b))
                tflops_pytorch = flops / (ms_pytorch * 1e-3) / 1e12

                # Benchmark SM120
                ms_sm120 = triton.testing.do_bench(lambda: gemm_sm120(a, b))
                tflops_sm120 = flops / (ms_sm120 * 1e-3) / 1e12

                speedup = tflops_sm120 / tflops_pytorch if tflops_pytorch > 0 else 0

                config_name = f"M=N=K={m}"
                print(f"{config_name:<25} | {tflops_pytorch:<15.2f} | {tflops_sm120:<15.2f} | {speedup:<10.2f}x")

                del a, b
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"{'M=N=K=' + str(m):<25} | {'Error':<15} | {'Error':<15} | {'-':<10}")
                print(f"  Error: {e}")

    except ImportError:
        print("triton is required for TFLOPS measurement")

    print()
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
