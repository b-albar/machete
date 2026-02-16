#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark Flash Attention: Megakernel vs PyTorch vs torch SDPA vs CuTe DSL FA2.

Compares GPU kernel execution time of:
  - PyTorch manual attention (bmm + softmax + bmm) [fp32]
  - torch.nn.functional.scaled_dot_product_attention [fp32]
  - CuTe DSL FlashAttentionForwardAmpere (tensor core MMA) [fp16]
  - Megakernel FlashAttentionOp (scalar warp-parallel) [fp32]

All implementations use direct CUDA event timing (no CUDA graph capture)
for consistent measurement.

Usage:
    python benchmarks/kernels/benchmark_attention.py
"""

import contextlib
import io
import sys

import torch
import torch.nn.functional as F

from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.attention import FlashAttentionOp
from machete.kernels.attention.ref import flash_attention_pytorch
from machete.megakernel.paged_memory import PAGE_SIZE

try:
    import cutlass
    import cutlass.cute as cute

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False

# CuTe DSL Flash Attention v2 (Ampere tensor core implementation)
CUTE_FA2_AVAILABLE = False
if CUTLASS_AVAILABLE:
    try:
        _fa2_dir = "/home/elentir/Projets/flash-attention/csrc/cutlass/examples/python/CuTeDSL/ampere"
        if _fa2_dir not in sys.path:
            sys.path.insert(0, _fa2_dir)
        from flash_attention_v2 import FlashAttentionForwardAmpere
        import cuda.bindings.driver as cuda_driver
        from cutlass.cute.runtime import from_dlpack

        CUTE_FA2_AVAILABLE = True
    except (ImportError, ModuleNotFoundError) as e:
        print(f"CuTe DSL FA2 not available: {e}")


def is_hopper_or_newer():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _tile_size_M(D, elem_bytes=4):
    """Compute tile_size_M that fits double-buffered K/V in PAGE_SIZE."""
    max_tile = (PAGE_SIZE - 8) // (D * elem_bytes) - 4
    return min(4, max(1, max_tile))


def _fa2_block_sizes(D):
    """Choose FA2 block sizes that fit in SM80 shared memory (96KB).

    smem = (m_blk * D + n_blk * D * 2) * 2 bytes (fp16 = 2 bytes/elem)
    Constraint: (m_blk * 2) % num_threads == 0 with num_threads=128
    """
    if D >= 128:
        # m=128, n=64: (128*128 + 64*128*2)*2 = 65536 <= 98304
        return 128, 64
    else:
        # m=128, n=128: (128*64 + 128*64*2)*2 = 49152 <= 98304
        return 128, 128


def _make_cute_tensor(torch_tensor):
    """Convert a torch fp16 tensor to a CuTe tensor for FA2."""
    return (
        from_dlpack(torch_tensor, assumed_align=16)
        .mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(
            mode=3,
            stride_order=torch_tensor.dim_order(),
            divisibility=8,  # 128 bits / 16 bits per fp16
        )
    )


def _bench_fn(fn, warmup=25, rep=100):
    """Benchmark a callable using direct CUDA event timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def _bench_megakernel(kernel, setup_fn, warmup=25, rep=100):
    """Benchmark a megakernel using its bench_spec."""
    spec = kernel.bench_spec(setup_fn=setup_fn)
    torch_stream, _ = spec.stream
    with torch.cuda.stream(torch_stream):
        for _ in range(warmup):
            spec.launch_fn()
        torch_stream.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(rep):
            start.record(torch_stream)
            spec.launch_fn()
            end.record(torch_stream)
            torch_stream.synchronize()
            times.append(start.elapsed_time(end))
    return sum(times) / len(times)


# =============================================================================
# Benchmark configs
# =============================================================================

CONFIGS = [
    # (BH, M, N, D) — representative transformer attention shapes
    # Decode (M=1, single new token attending to KV cache)
    (32, 1, 128, 128),     # small KV cache, 32 heads
    (32, 1, 512, 128),     # medium KV cache
    (32, 1, 1024, 128),    # large KV cache
    (32, 1, 2048, 128),    # very large KV cache
    (32, 1, 128, 64),      # D=64 variant
    (32, 1, 1024, 64),
    # Prefill (M=N, processing prompt tokens)
    (32, 64, 64, 128),
    (32, 128, 128, 128),
    (32, 256, 256, 128),
    (32, 512, 512, 128),
    (32, 128, 128, 64),
    (32, 256, 256, 64),
    # Chunked prefill (M < N)
    (32, 64, 256, 128),
    (32, 64, 512, 128),
    (32, 128, 512, 128),
]


def run_benchmarks(warmup=25, rep=100):
    """Run attention benchmarks across all configs."""
    has_megakernel = is_hopper_or_newer() and CUTLASS_AVAILABLE
    has_fa2 = CUTE_FA2_AVAILABLE

    # Build dynamic header
    cols = ["pytorch", "sdpa"]
    if has_fa2:
        cols.append("cute_fa2")
    if has_megakernel:
        cols.append("mk_fp32")
        cols.append("mk_fp16")

    header = f"{'Config':<28}"
    for c in cols:
        header += f" {c:>12}"
    if has_megakernel and has_fa2:
        header += f" {'mk16/fa2':>8}"
    if has_megakernel:
        header += f" {'mk16/sdpa':>9}"
    print(header)

    sub = f"{'':<28}"
    for _ in cols:
        sub += f" {'ms':>12}"
    if has_megakernel and has_fa2:
        sub += f" {'ratio':>8}"
    if has_megakernel:
        sub += f" {'ratio':>9}"
    print(sub)
    print("-" * len(header))

    # Cache compiled FA2 kernels per (D, m_blk, n_blk)
    fa2_cache = {}

    for BH, M, N, D in CONFIGS:
        label = f"BH={BH},M={M},N={N},D={D}"
        torch.manual_seed(42)

        # fp32 tensors for pytorch, sdpa, megakernel
        q = torch.randn(BH, M, D, dtype=torch.float32, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")

        # PyTorch manual
        ms_pt = _bench_fn(
            lambda: flash_attention_pytorch(q, k, v),
            warmup=warmup, rep=rep,
        )

        # torch SDPA
        ms_sdpa = _bench_fn(
            lambda: F.scaled_dot_product_attention(q, k, v),
            warmup=warmup, rep=rep,
        )

        # CuTe DSL Flash Attention v2 (fp16, tensor cores)
        ms_fa2 = None
        if has_fa2:
            try:
                m_blk, n_blk = _fa2_block_sizes(D)
                if FlashAttentionForwardAmpere.can_implement(
                    cutlass.Float16, D, m_blk, n_blk, 128, False
                ):
                    # FA2 expects (batch, seq, heads, dim) in fp16
                    batch_size = 1
                    num_heads = BH
                    q16 = torch.randn(
                        batch_size, M, num_heads, D,
                        dtype=torch.float16, device="cuda",
                    )
                    k16 = torch.randn(
                        batch_size, N, num_heads, D,
                        dtype=torch.float16, device="cuda",
                    )
                    v16 = torch.randn(
                        batch_size, N, num_heads, D,
                        dtype=torch.float16, device="cuda",
                    )
                    o16 = torch.zeros(
                        batch_size, M, num_heads, D,
                        dtype=torch.float16, device="cuda",
                    )

                    q_ct = _make_cute_tensor(q16)
                    k_ct = _make_cute_tensor(k16)
                    v_ct = _make_cute_tensor(v16)
                    o_ct = _make_cute_tensor(o16)

                    scale = 1.0 / (D ** 0.5)
                    torch_stream = torch.cuda.current_stream()
                    cu_stream = cuda_driver.CUstream(torch_stream.cuda_stream)

                    # Compile (or reuse cached per unique block config)
                    cache_key = (D, m_blk, n_blk)
                    if cache_key not in fa2_cache:
                        fa2_obj = FlashAttentionForwardAmpere(
                            D, m_blk, n_blk, 128, False,
                        )
                        fa2_cache[cache_key] = cute.compile(
                            fa2_obj, q_ct, k_ct, v_ct, o_ct, scale, cu_stream,
                        )

                    compiled_fa2 = fa2_cache[cache_key]

                    ms_fa2 = _bench_fn(
                        lambda: compiled_fa2(
                            q_ct, k_ct, v_ct, o_ct, scale, cu_stream,
                        ),
                        warmup=warmup, rep=rep,
                    )
            except Exception as e:
                ms_fa2 = None
                print(f"  [{label}] CuTe FA2 error: {e}")

        # Megakernel fp32 (scalar path)
        ms_mk32 = None
        if has_megakernel:
            try:
                tile_m = _tile_size_M(D, q.element_size())
                o = torch.zeros_like(q)
                ops = FlashAttentionOp.schedule(
                    q=q, k=k, v=v, o=o,
                    tile_sizes={"M": tile_m},
                )
                kernel = Megakernel(ops, config=MegakernelConfig())
                with contextlib.redirect_stdout(io.StringIO()):
                    kernel.run()
                torch.cuda.synchronize()

                ms_mk32 = _bench_megakernel(
                    kernel, setup_fn=lambda: o.zero_(),
                    warmup=warmup, rep=rep,
                )
            except Exception as e:
                ms_mk32 = None
                print(f"  [{label}] mk_fp32 error: {e}")

        # Megakernel fp16 (MMA tensor core path)
        ms_mk16 = None
        if has_megakernel:
            try:
                q16 = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
                k16 = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
                v16 = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
                o16 = torch.zeros_like(q16)
                ops16 = FlashAttentionOp.schedule(
                    q=q16, k=k16, v=v16, o=o16,
                )
                kernel16 = Megakernel(ops16, config=MegakernelConfig())
                with contextlib.redirect_stdout(io.StringIO()):
                    kernel16.run()
                torch.cuda.synchronize()

                ms_mk16 = _bench_megakernel(
                    kernel16, setup_fn=lambda: o16.zero_(),
                    warmup=warmup, rep=rep,
                )
            except Exception as e:
                ms_mk16 = None
                print(f"  [{label}] mk_fp16 error: {e}")

        # Print results
        line = f"{label:<28}"
        line += f" {ms_pt:>12.4f}"
        line += f" {ms_sdpa:>12.4f}"
        if has_fa2:
            line += f" {ms_fa2:>12.4f}" if ms_fa2 is not None else f" {'error':>12}"
        if has_megakernel:
            line += f" {ms_mk32:>12.4f}" if ms_mk32 is not None else f" {'error':>12}"
            line += f" {ms_mk16:>12.4f}" if ms_mk16 is not None else f" {'error':>12}"
        if has_megakernel and has_fa2:
            if ms_mk16 is not None and ms_fa2 is not None and ms_fa2 > 0:
                line += f" {ms_mk16 / ms_fa2:>7.2f}x"
            else:
                line += f" {'---':>8}"
        if has_megakernel:
            if ms_mk16 is not None and ms_sdpa > 0:
                line += f" {ms_mk16 / ms_sdpa:>8.2f}x"
            else:
                line += f" {'---':>9}"
        print(line)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 100)
    print("Flash Attention Benchmark: Megakernel vs PyTorch vs SDPA vs CuTe DSL FA2")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"Hopper+: {is_hopper_or_newer()}")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print(f"CuTe FA2: {CUTE_FA2_AVAILABLE}")
    print()
    print("Note: pytorch/sdpa use fp32; cute_fa2/mk_fp16 use fp16 (tensor cores); mk_fp32 uses fp32 (scalar)")
    print()

    run_benchmarks()
