#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark Cross-Entropy: Megakernel vs PyTorch vs Liger-Kernel.

Compares GPU kernel execution time of:
  - PyTorch F.cross_entropy [fp32 compute]
  - Liger-Kernel cross-entropy (Triton) [if available]
  - Megakernel CrossEntropyOp (fused fwd+bwd, online softmax) [bf16/fp32]

Usage:
    python benchmarks/kernels/benchmark_cross_entropy.py
"""

import contextlib
import io

import torch
import torch.nn.functional as F

from machete.megakernel import Megakernel
from machete.kernels.cross_entropy import CrossEntropyOp
from machete.kernels.utils import SingleOpKernel
from machete.utils.benchmark import Benchmark

try:
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    print("Liger-Kernel not available, skipping Liger benchmarks")

PAGE_SIZES = [16384, 32768, 49152]


# =============================================================================
# Benchmark configs
# =============================================================================

_BASE_CONFIGS = [
    # (BT, V) — realistic LLM shapes
    (256, 32000),
    (256, 128256),
    (1024, 32000),
    (1024, 128256),
    (4096, 32000),
    (4096, 128256),
    (16384, 32000),
]

CONFIGS = [c + (ps,) for c in _BASE_CONFIGS for ps in PAGE_SIZES]


# =============================================================================
# Benchmark function
# =============================================================================


@Benchmark.configs(["BT", "V", "page_size"], CONFIGS)
def benchmark_cross_entropy(BT, V, page_size):
    torch.manual_seed(42)
    logits = torch.randn(BT, V, dtype=torch.bfloat16, device="cuda")
    targets = torch.randint(0, V, (BT,), device="cuda")

    results = {}

    # --- PyTorch baseline ---
    def pytorch_ce():
        loss = F.cross_entropy(logits.float(), targets, reduction="none")
        # Simulate backward: compute softmax gradient
        softmax = torch.softmax(logits.float(), dim=-1)
        one_hot = torch.zeros_like(softmax)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        grad = (softmax - one_hot).to(logits.dtype)
        return loss, grad

    results["PyTorch"] = pytorch_ce

    # --- Liger-Kernel ---
    if LIGER_AVAILABLE:
        def liger_ce():
            logits_clone = logits.clone()
            return LigerCrossEntropyFunction.apply(
                logits_clone, targets, None,  # weight=None
            )

        results["Liger"] = liger_ce

    # --- Megakernel + SingleOp CrossEntropyOp ---
    try:
        loss_buf = torch.zeros(BT, dtype=torch.float32, device="cuda")
        grad_buf = torch.zeros_like(logits)
        ops = CrossEntropyOp.schedule_forward(
            logits=logits,
            targets=targets.int(),
            loss=loss_buf,
            grad_logits=grad_buf,
            page_size=page_size,
        )
        config = CrossEntropyOp.kernel_config(ops)
        kernel = Megakernel(ops, config=config)
        with contextlib.redirect_stdout(io.StringIO()):
            kernel.run()
        torch.cuda.synchronize()
        results["megakernel"] = kernel.bench_spec()
    except Exception:
        pass

    try:
        loss_so = torch.zeros(BT, dtype=torch.float32, device="cuda")
        grad_so = torch.zeros_like(logits)
        so_ops = CrossEntropyOp.schedule_forward(
            logits=logits,
            targets=targets.int(),
            loss=loss_so,
            grad_logits=grad_so,
            page_size=page_size,
        )
        so_kernel = SingleOpKernel(so_ops)
        with contextlib.redirect_stdout(io.StringIO()):
            so_kernel.run()
        torch.cuda.synchronize()
        results["single_op"] = so_kernel.bench_spec()
    except Exception:
        pass

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    benchmark_cross_entropy._benchmark.run(
        mode="kernel",
        warmup=5,
        rep=20,
    )
