#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark Flash Attention: Megakernel vs PyTorch vs torch SDPA vs CuTe DSL FA2.

Compares GPU kernel execution time of:
  - PyTorch manual attention (bmm + softmax + bmm) [bf16]
  - torch.nn.functional.scaled_dot_product_attention [bf16]
  - CuTe DSL FlashAttentionForwardAmpere (tensor core MMA) [fp16]
  - Megakernel FlashAttentionSm120Op (cooperative cpasync + MMA) [bf16]

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

from machete.megakernel import Megakernel
from machete.kernels.attention import FlashAttentionSm120Op, FlashAttentionSm120BwdOp
from machete.kernels.attention.ref import flash_attention_pytorch
from machete.kernels.utils import SingleOpKernel
from machete.utils.benchmark import Benchmark

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


def _fa2_block_sizes(D):
    """Choose FA2 block sizes that fit in SM80 shared memory (96KB)."""
    if D >= 128:
        return 128, 64
    else:
        return 128, 128


def _make_cute_tensor(torch_tensor):
    """Convert a torch fp16 tensor to a CuTe tensor for FA2."""
    return (
        from_dlpack(torch_tensor, assumed_align=16)
        .mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(
            mode=3,
            stride_order=torch_tensor.dim_order(),
            divisibility=8,
        )
    )


# Module-level cache for compiled FA2 kernels
_fa2_cache = {}


# =============================================================================
# Benchmark configs
# =============================================================================

CONFIGS = [
    # (BH, M, N, D) — realistic LLM attention shapes (D=128 typical)
    #
    # Decode (autoregressive, M=16 minimum MMA tile)
    (32, 16, 2048, 128),
    (32, 16, 4096, 128),
    (32, 16, 8192, 128),
    (32, 16, 16384, 128),
    (32, 16, 32768, 128),
    (32, 16, 65536, 128),
    (32, 16, 131072, 128),
    # Prefill (M=N, processing prompt tokens)
    (32, 1024, 1024, 128),
    (32, 2048, 2048, 128),
    (32, 4096, 4096, 128),
    (32, 8192, 8192, 128),
    # Chunked prefill (M < N, typical chunk sizes)
    (32, 256, 4096, 128),
    (32, 256, 16384, 128),
    (32, 512, 8192, 128),
    (32, 512, 32768, 128),
]


@Benchmark.configs(["BH", "M", "N", "D"], CONFIGS)
def bench_attention(BH, M, N, D):
    """Setup attention benchmark functions for each implementation."""
    torch_dtype = torch.bfloat16

    torch.manual_seed(42)
    q = torch.randn(BH, M, D, dtype=torch_dtype, device="cuda")
    k = torch.randn(BH, N, D, dtype=torch_dtype, device="cuda")
    v = torch.randn(BH, N, D, dtype=torch_dtype, device="cuda")

    funcs = {}

    # PyTorch manual
    funcs["pytorch"] = lambda: flash_attention_pytorch(q, k, v)

    # torch SDPA (4D input so flash attention backend is used)
    q4d = q.unsqueeze(0)  # (1, BH, M, D) → (batch, heads, seq, head_dim)
    k4d = k.unsqueeze(0)
    v4d = v.unsqueeze(0)
    funcs["sdpa"] = lambda: F.scaled_dot_product_attention(q4d, k4d, v4d)

    # CuTe DSL Flash Attention v2 (fp16, tensor cores)
    if CUTE_FA2_AVAILABLE:
        try:
            m_blk, n_blk = _fa2_block_sizes(D)
            if FlashAttentionForwardAmpere.can_implement(
                cutlass.Float16, D, m_blk, n_blk, 128, False
            ):
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
                init_stream = cuda_driver.CUstream(
                    torch.cuda.current_stream().cuda_stream
                )

                cache_key = (D, m_blk, n_blk)
                if cache_key not in _fa2_cache:
                    fa2_obj = FlashAttentionForwardAmpere(
                        D, m_blk, n_blk, 128, False,
                    )
                    _fa2_cache[cache_key] = cute.compile(
                        fa2_obj, q_ct, k_ct, v_ct, o_ct, scale, init_stream,
                    )

                compiled_fa2 = _fa2_cache[cache_key]

                # Use dynamic stream so CUDA graph capture works correctly
                # (the benchmark framework may run on a non-default stream).
                def _run_fa2(f=compiled_fa2, q=q_ct, k=k_ct, v=v_ct,
                             o=o_ct, s=scale):
                    stream = cuda_driver.CUstream(
                        torch.cuda.current_stream().cuda_stream
                    )
                    f(q, k, v, o, s, stream)

                funcs["cute_fa2"] = _run_fa2
        except Exception as e:
            print(f"  CuTe FA2 error: {e}")

    # Megakernel bf16 (cooperative cpasync + MMA)
    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        try:
            o_mk = torch.zeros_like(q)
            ops_mk = FlashAttentionSm120Op.schedule(
                q=q, k=k, v=v, o=o_mk,
            )
            config_mk = FlashAttentionSm120Op.kernel_config(ops_mk)
            kernel_mk = Megakernel(ops_mk, config=config_mk)
            with contextlib.redirect_stdout(io.StringIO()):
                kernel_mk.run()
            torch.cuda.synchronize()

            funcs["megakernel"] = kernel_mk.bench_spec(
                setup_fn=lambda: o_mk.zero_(),
                keep_alive=[q, k, v, o_mk],
            )
        except Exception as e:
            print(f"  megakernel error: {e}")

    # SingleOpKernel bf16
    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        try:
            o_so = torch.zeros_like(q)
            so_ops = FlashAttentionSm120Op.schedule(
                q=q, k=k, v=v, o=o_so,
            )
            so_kernel = SingleOpKernel(so_ops)
            with contextlib.redirect_stdout(io.StringIO()):
                so_kernel.run()
            torch.cuda.synchronize()

            funcs["single_op"] = so_kernel.bench_spec(
                setup_fn=lambda: o_so.zero_(),
                keep_alive=[q, k, v, o_so],
            )
        except Exception as e:
            print(f"  single_op error: {e}")

    return funcs


# =============================================================================
# Backward Benchmark
# =============================================================================

BWD_CONFIGS = [
    # (BH, M, N, D) — subset of forward configs for backward
    # Decode
    (32, 16, 2048, 128),
    (32, 16, 4096, 128),
    (32, 16, 8192, 128),
    (32, 16, 16384, 128),
    # Prefill
    (32, 1024, 1024, 128),
    (32, 2048, 2048, 128),
    (32, 4096, 4096, 128),
    # Chunked prefill
    (32, 256, 4096, 128),
    (32, 512, 8192, 128),
]


@Benchmark.configs(["BH", "M", "N", "D"], BWD_CONFIGS)
def bench_attention_bwd(BH, M, N, D):
    """Setup attention backward benchmark."""
    torch_dtype = torch.bfloat16

    torch.manual_seed(42)
    q = torch.randn(BH, M, D, dtype=torch_dtype, device="cuda")
    k = torch.randn(BH, N, D, dtype=torch_dtype, device="cuda")
    v = torch.randn(BH, N, D, dtype=torch_dtype, device="cuda")
    dout = torch.randn(BH, M, D, dtype=torch_dtype, device="cuda")

    funcs = {}

    # PyTorch autograd backward
    def pytorch_bwd():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        o = flash_attention_pytorch(q_, k_, v_)
        o.backward(dout)
        return q_.grad, k_.grad, v_.grad

    funcs["pytorch"] = pytorch_bwd

    # torch SDPA backward
    dout4d = dout.unsqueeze(0)

    def sdpa_bwd():
        q_ = q.unsqueeze(0).detach().requires_grad_(True)
        k_ = k.unsqueeze(0).detach().requires_grad_(True)
        v_ = v.unsqueeze(0).detach().requires_grad_(True)
        o = F.scaled_dot_product_attention(q_, k_, v_)
        o.backward(dout4d)
        return q_.grad, k_.grad, v_.grad

    funcs["sdpa"] = sdpa_bwd

    # Megakernel backward (FlashAttentionSm120BwdOp)
    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        try:
            # Run forward to get lse
            o_mk = torch.zeros_like(q)
            lse = torch.empty(BH, M, dtype=torch.float32, device="cuda")
            fwd_ops = FlashAttentionSm120Op.schedule_forward(
                q=q, k=k, v=v, o=o_mk, lse=lse,
            )
            fwd_config = FlashAttentionSm120Op.kernel_config(fwd_ops)
            fwd_kernel = Megakernel(fwd_ops, config=fwd_config)
            with contextlib.redirect_stdout(io.StringIO()):
                fwd_kernel.run()
            torch.cuda.synchronize()

            # Setup backward
            dpsum = (dout.float() * o_mk.float()).sum(dim=-1).contiguous()
            dq_accum = torch.zeros(BH, M, D, dtype=torch.float32, device="cuda")
            dk = torch.zeros_like(k)
            dv = torch.zeros_like(v)

            bwd_ops = FlashAttentionSm120BwdOp.schedule_backward(
                k=k, v=v, q=q, dout=dout, lse=lse, dpsum=dpsum,
                dq=dq_accum, dk=dk, dv=dv,
            )
            bwd_config = FlashAttentionSm120BwdOp.kernel_config(bwd_ops)
            bwd_kernel = Megakernel(bwd_ops, config=bwd_config)
            with contextlib.redirect_stdout(io.StringIO()):
                bwd_kernel.run()
            torch.cuda.synchronize()

            def reset_bwd():
                dq_accum.zero_()
                dk.zero_()
                dv.zero_()

            funcs["megakernel"] = bwd_kernel.bench_spec(
                setup_fn=reset_bwd,
                keep_alive=[q, k, v, dout, lse, dpsum, dq_accum, dk, dv],
            )
        except Exception as e:
            print(f"  megakernel bwd error: {e}")

    # SingleOpKernel backward
    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        try:
            # Run forward to get lse
            o_so = torch.zeros_like(q)
            lse_so = torch.empty(BH, M, dtype=torch.float32, device="cuda")
            so_fwd_ops = FlashAttentionSm120Op.schedule_forward(
                q=q, k=k, v=v, o=o_so, lse=lse_so,
            )
            so_fwd_kernel = SingleOpKernel(so_fwd_ops)
            with contextlib.redirect_stdout(io.StringIO()):
                so_fwd_kernel.run()
            torch.cuda.synchronize()

            dpsum_so = (dout.float() * o_so.float()).sum(dim=-1).contiguous()
            dq_so = torch.zeros(BH, M, D, dtype=torch.float32, device="cuda")
            dk_so = torch.zeros_like(k)
            dv_so = torch.zeros_like(v)

            so_bwd_ops = FlashAttentionSm120BwdOp.schedule_backward(
                k=k, v=v, q=q, dout=dout, lse=lse_so, dpsum=dpsum_so,
                dq=dq_so, dk=dk_so, dv=dv_so,
            )
            so_bwd_kernel = SingleOpKernel(so_bwd_ops)
            with contextlib.redirect_stdout(io.StringIO()):
                so_bwd_kernel.run()
            torch.cuda.synchronize()

            def reset_so_bwd():
                dq_so.zero_()
                dk_so.zero_()
                dv_so.zero_()

            funcs["single_op"] = so_bwd_kernel.bench_spec(
                setup_fn=reset_so_bwd,
                keep_alive=[q, k, v, dout, lse_so, dpsum_so, dq_so, dk_so, dv_so],
            )
        except Exception as e:
            print(f"  single_op bwd error: {e}")

    return funcs


# =============================================================================
# Main
# =============================================================================


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
    print("Note: All implementations use bf16. CuTe FA2 uses fp16 (only supported dtype).")
    print()

    bench_attention._benchmark.run(
        mode="kernel",
        warmup=25,
        rep=100,
    )

    print()
    print("=" * 100)
    print("Flash Attention Backward Benchmark: Megakernel vs PyTorch vs SDPA")
    print("=" * 100)
    print()

    bench_attention_bwd._benchmark.run(
        mode="kernel",
        warmup=25,
        rep=100,
    )
