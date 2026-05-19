#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark Flash Decoding vs SDPA for decode workloads.

Compares GPU kernel execution time of:
  - torch.nn.functional.scaled_dot_product_attention [bf16]
  - Megakernel FlashAttentionSm120Op (single CTA per head) [bf16]
  - FlashDecoding split-KV (multi-CTA per head) [bf16]

Usage:
    python benchmarks/kernels/benchmark_flash_decoding.py
"""

import contextlib
import io

import torch
import torch.nn.functional as F

from machete.megakernel import Megakernel
from machete.kernels.attention import FlashAttentionSm120Op
from machete.kernels.attention.flash_decoding import flash_decoding_schedule
from machete.utils.benchmark import Benchmark

try:
    import cutlass  # noqa: F401
    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False

PAGE_SIZES = [16384, 32768, 49152]
H_SIZES = [1, 4, 16, 32]


def is_hopper_or_newer():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _check_close(name, actual, expected):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=2e-2,
        atol=2e-2,
        msg=lambda msg: f"{name} correctness failed:\n{msg}",
    )


# Decode configs: small M, large N
_BASE_CONFIGS = [
    (16, 2048, 128),
    (16, 4096, 128),
    (16, 8192, 128),
    (16, 16384, 128),
    (16, 32768, 128),
    (16, 65536, 128),
    (16, 131072, 128),
]

CONFIGS = [(h,) + c + (ps,) for c in _BASE_CONFIGS for h in H_SIZES for ps in PAGE_SIZES]


@Benchmark.configs(["H", "M", "N", "D", "page_size"], CONFIGS)
def bench_flash_decoding(H, M, N, D, page_size):
    torch_dtype = torch.bfloat16
    torch.manual_seed(42)
    q = torch.randn(1, M, H, D, dtype=torch_dtype, device="cuda")
    k = torch.randn(1, N, H, D, dtype=torch_dtype, device="cuda")
    v = torch.randn(1, N, H, D, dtype=torch_dtype, device="cuda")

    funcs = {}

    # torch SDPA
    q4d = q.permute(0, 2, 1, 3)
    k4d = k.permute(0, 2, 1, 3)
    v4d = v.permute(0, 2, 1, 3)
    funcs["sdpa"] = lambda: F.scaled_dot_product_attention(q4d, k4d, v4d)
    ref = funcs["sdpa"]().permute(0, 2, 1, 3).contiguous()
    torch.cuda.synchronize()

    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        # Standard FA (single CTA per head)
        try:
            o_fa = torch.zeros_like(q)
            ops_fa = FlashAttentionSm120Op.schedule(
                q=q, k=k, v=v, o=o_fa, page_size=page_size,
            )
            config_fa = FlashAttentionSm120Op.kernel_config(ops_fa)
            kernel_fa = Megakernel(ops_fa, config=config_fa)
            with contextlib.redirect_stdout(io.StringIO()):
                kernel_fa.run()
            torch.cuda.synchronize()
            _check_close("fa_mega", o_fa, ref)
            funcs["fa_mega"] = kernel_fa.bench_spec(
                setup_fn=lambda o=o_fa: o.zero_(),
                keep_alive=[q, k, v, o_fa],
            )
        except Exception as e:
            print(f"  FA megakernel failed: {e}")

        # Flash Decoding (split-KV, multi-CTA, cp.async K/V)
        try:
            o_fd = torch.zeros_like(q)
            fd_ops, fd_config = flash_decoding_schedule(
                q=q, k=k, v=v, o=o_fd, page_size=page_size,
            )
            kernel_fd = Megakernel(fd_ops, config=fd_config)
            with contextlib.redirect_stdout(io.StringIO()):
                kernel_fd.run()
            torch.cuda.synchronize()
            _check_close("flash_dec", o_fd, ref)
            funcs["flash_dec"] = kernel_fd.bench_spec(
                setup_fn=lambda o=o_fd: o.zero_(),
                keep_alive=[q, k, v, o_fd],
            )
        except Exception as e:
            print(f"  Flash decoding failed: {e}")

    return funcs


if __name__ == "__main__":
    print("=" * 100)
    print("Flash Decoding Benchmark: SDPA vs FA Megakernel vs Flash Decoding")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"Hopper+: {is_hopper_or_newer()}")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print()

    bench_flash_decoding._benchmark.run(
        mode="kernel",
        warmup=25,
        rep=100,
    )
