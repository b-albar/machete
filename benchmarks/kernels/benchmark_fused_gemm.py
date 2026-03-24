#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark 5 chained GEMMs: CUDA graph vs Megakernel fusion.

Models a chain of linear projections: x → W1 → W2 → W3 → W4 → W5 → y
where each GEMM's output is the next one's input. All weight matrices
are square (D×D) so the chain is: [1,M,D] @ [D,D] → [1,M,D] @ [D,D] → ...

Compares:
  1. Sequential torch.matmul calls (cuBLAS)
  2. CUDA graph capturing the sequential chain
  3. Single megakernel fusing all 5 GemmOps with auto-detected dependencies

Usage:
    python benchmarks/kernels/benchmark_fused_gemm.py
"""

import contextlib
import io

import torch

from machete.megakernel import Megakernel
from machete.kernels.gemm import GemmOp
from machete.utils.benchmark import Benchmark

try:
    import cutlass  # noqa: F401
    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False

PAGE_SIZES = [16384, 32768, 49152]

NUM_GEMMS = 5


def is_sm90_or_newer():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 90


def total_bytes(M, D, dtype, page_size):
    """Total bytes for chained GEMMs.

    Each GEMM reads input [1,M,D] + weight [D,D], writes output [1,M,D].
    First GEMM reads x, last writes y, intermediates are read+written.
    """
    elem_bytes = 2 if dtype in ("float16", "bfloat16") else 4
    weight_bytes = D * D * elem_bytes * NUM_GEMMS
    activation_bytes = M * D * elem_bytes * (NUM_GEMMS + 1)  # x + all intermediates
    return weight_bytes + activation_bytes


def total_flops(M, D, dtype, page_size):
    """Total FLOPs for NUM_GEMMS chained GEMMs: each is 2*M*D*D."""
    return 2 * M * D * D * NUM_GEMMS


@Benchmark.parametrize("page_size", PAGE_SIZES)
@Benchmark.parametrize("dtype", ["bfloat16"])
@Benchmark.parametrize("D", [1024, 2048, 4096, 8192])
@Benchmark.parametrize("M", [128, 512, 2048, 4096])
def bench_fused_gemm(M, D, dtype, page_size):
    """Benchmark 5 chained GEMMs: x → W1 → W2 → W3 → W4 → W5 → y."""
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    torch.manual_seed(42)

    # Weight matrices [D, D] for each layer
    Ws = [torch.randn(D, D, dtype=torch_dtype, device="cuda") for _ in range(NUM_GEMMS)]

    # Chain buffers: x → h0 → h1 → h2 → h3 → h4 (=y)
    # We need NUM_GEMMS + 1 activation buffers but can alias with 2 (ping-pong)
    # 2D buffers for cuBLAS / CUDA graph paths
    buf_a = torch.randn(M, D, dtype=torch_dtype, device="cuda")   # input / even steps
    buf_b = torch.zeros(M, D, dtype=torch_dtype, device="cuda")   # odd steps

    funcs = {}

    # --- 1. Sequential torch.matmul (cuBLAS) ---
    def seq_matmul():
        torch.matmul(buf_a, Ws[0], out=buf_b)
        torch.matmul(buf_b, Ws[1], out=buf_a)
        torch.matmul(buf_a, Ws[2], out=buf_b)
        torch.matmul(buf_b, Ws[3], out=buf_a)
        torch.matmul(buf_a, Ws[4], out=buf_b)

    funcs["cublas"] = seq_matmul

    # --- 2. Megakernel fusion ---
    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        Ws_t = [w.t().contiguous() for w in Ws]  # GemmOp expects (N, K) layout

        try:
            mk_a = buf_a.clone().unsqueeze(0)  # (1, M, D)
            mk_b = torch.zeros(1, M, D, dtype=torch_dtype, device="cuda")

            inputs =  [mk_a, mk_b, mk_a, mk_b, mk_a]
            outputs = [mk_b, mk_a, mk_b, mk_a, mk_b]

            all_ops = []
            for i in range(NUM_GEMMS):
                ops = GemmOp.schedule(a=inputs[i], b=Ws_t[i], c=outputs[i], page_size=page_size)
                all_ops.extend(ops)

            config = GemmOp.kernel_config(all_ops[:1])
            kernel = Megakernel(all_ops, config=config)
            with contextlib.redirect_stdout(io.StringIO()):
                kernel.run()
            torch.cuda.synchronize()

            funcs["megakernel"] = kernel.bench_spec(
                setup_fn=lambda mk_a=mk_a, mk_b=mk_b: (mk_a[0].copy_(buf_a), mk_b.zero_()),
                keep_alive=[mk_a, mk_b, Ws_t],
            )
        except Exception:
            pass

    return funcs


if __name__ == "__main__":
    print("=" * 100)
    print(f"Chained GEMM Benchmark: x → W1 → W2 → ... → W{NUM_GEMMS} → y")
    print("cublas     = 5 sequential torch.matmul (CUDA-graph captured by framework)")
    print("megakernel = single persistent kernel with auto dependencies")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print()

    bench_fused_gemm._benchmark.run(
        mode="kernel",
        bytes_fn=total_bytes,
        warmup=25,
        rep=100,
        columns=["cublas", "megakernel"],
    )
