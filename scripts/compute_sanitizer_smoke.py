#!/usr/bin/env python
# Copyright (c) 2026, Machete Authors
"""Compute-sanitizer smoke harness for individual kernel families.

Runs a minimal real megakernel launch per op family. Intended to be invoked
either directly or under compute-sanitizer, ideally one op per fresh process.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections import OrderedDict

import torch


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")


def _major_cc() -> int:
    major, _minor = torch.cuda.get_device_capability()
    return major


def _require_hopper() -> None:
    _require_cuda()
    if _major_cc() < 9:
        raise RuntimeError("SM90+ GPU required")


def _run_kernel(ops, config=None) -> None:
    from machete.megakernel import Megakernel

    kernel = Megakernel(ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()


def activation() -> None:
    _require_hopper()
    from machete.kernels.activation import ActivationOp

    x = torch.randn(2, 64, 1024, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    ops = ActivationOp.schedule(x=x, y=y, activation="silu")
    _run_kernel(ops, config=ActivationOp.kernel_config(ops))


def rmsnorm_fwd() -> None:
    _require_hopper()
    from machete.kernels.rms_norm import RMSNormOp

    x = torch.randn(2, 64, 1024, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(1024, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    ops = RMSNormOp.schedule(x=x, weight=weight, y=y)
    _run_kernel(ops, config=RMSNormOp.kernel_config(ops))


def rmsnorm_bwd() -> None:
    _require_hopper()
    from machete.kernels.rms_norm import RMSNormBwdOp

    dout = torch.randn(2, 64, 1024, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(2, 64, 1024, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(1024, dtype=torch.bfloat16, device="cuda")
    dx = torch.empty_like(x)
    ops = RMSNormBwdOp.schedule(dout=dout, x=x, weight=weight, dx=dx)
    _run_kernel(ops, config=RMSNormBwdOp.kernel_config(ops))


def glu_fwd() -> None:
    _require_hopper()
    from machete.kernels.glu import GLUOp

    x = torch.randn(2, 64, 2048, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(2, 64, 1024, dtype=torch.bfloat16, device="cuda")
    ops = GLUOp.schedule(x=x, y=y, activation="silu")
    _run_kernel(ops, config=GLUOp.kernel_config(ops))


def glu_bwd() -> None:
    _require_hopper()
    from machete.kernels.glu import GLUBwdOp

    x = torch.randn(2, 64, 2048, dtype=torch.bfloat16, device="cuda")
    dy = torch.randn(2, 64, 1024, dtype=torch.bfloat16, device="cuda")
    dx = torch.empty_like(x)
    ops = GLUBwdOp.schedule(dy=dy, x=x, dx=dx, activation="silu")
    _run_kernel(ops, config=GLUBwdOp.kernel_config(ops))


def gemm() -> None:
    _require_hopper()
    from machete.kernels.gemm import GemmOp

    a = torch.randn(1, 128, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda").t().contiguous()
    c = torch.empty(1, 128, 4096, dtype=torch.bfloat16, device="cuda")
    ops = GemmOp.schedule(a=a, b=b, c=c, page_size=32768)
    _run_kernel(ops, config=GemmOp.kernel_config(ops))


def rope_fwd() -> None:
    _require_hopper()
    from machete.kernels.rope import RopeOp

    q = torch.randn(2, 128, 8, 128, dtype=torch.bfloat16, device="cuda")
    cos = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
    sin = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
    ops = RopeOp.schedule(q=q, cos=cos, sin=sin)
    _run_kernel(ops, config=RopeOp.kernel_config(ops))


def rope_bwd() -> None:
    _require_hopper()
    from machete.kernels.rope import RopeBwdOp

    q = torch.randn(2, 128, 8, 128, dtype=torch.bfloat16, device="cuda")
    cos = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
    sin = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
    ops = RopeBwdOp.schedule(q=q, cos=cos, sin=sin)
    _run_kernel(ops, config=RopeBwdOp.kernel_config(ops))


def qknorm_rope() -> None:
    _require_hopper()
    from machete.kernels.qknorm_rope import QKNormRopeOp

    q = torch.randn(32, 8, 256, dtype=torch.float32, device="cuda")
    norm_weight = torch.randn(256, dtype=torch.float32, device="cuda")
    cos = torch.randn(16, 32, dtype=torch.float32, device="cuda")
    sin = torch.randn(16, 32, dtype=torch.float32, device="cuda")
    ops = QKNormRopeOp.schedule(q=q, norm_weight=norm_weight, cos=cos, sin=sin)
    _run_kernel(ops, config=QKNormRopeOp.kernel_config(ops))


def attention_fwd() -> None:
    _require_hopper()
    from machete.kernels.attention import FlashAttentionSm120Op

    q = torch.randn(16, 16, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(16, 128, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(16, 128, 128, dtype=torch.bfloat16, device="cuda")
    o = torch.empty_like(q)
    ops = FlashAttentionSm120Op.schedule(q=q, k=k, v=v, o=o, page_size=32768)
    _run_kernel(ops, config=FlashAttentionSm120Op.kernel_config(ops))


def attention_bwd() -> None:
    _require_hopper()
    from machete.kernels.attention import FlashAttentionSm120BwdOp

    q = torch.randn(1, 16, 16, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 16, 128, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 16, 128, 128, dtype=torch.bfloat16, device="cuda")
    dout = torch.randn_like(q)
    lse = torch.randn(1, 16, 16, dtype=torch.float32, device="cuda")
    dpsum = torch.randn(1, 16, 16, dtype=torch.float32, device="cuda")
    dq = torch.zeros(q.shape, dtype=torch.float32, device="cuda")
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    ops = FlashAttentionSm120BwdOp.schedule(
        q=q, k=k, v=v, dout=dout, lse=lse, dpsum=dpsum, dq=dq, dk=dk, dv=dv, page_size=49152
    )
    _run_kernel(ops, config=FlashAttentionSm120BwdOp.kernel_config(ops))


def flash_decoding() -> None:
    _require_hopper()
    from machete.kernels.attention.flash_decoding import flash_decoding_schedule

    q = torch.randn(8, 16, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(2, 256, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(2, 256, 128, dtype=torch.bfloat16, device="cuda")
    o = torch.empty_like(q)
    ops, config = flash_decoding_schedule(q=q, k=k, v=v, o=o, kv_group_size=4)
    _run_kernel(ops, config=config)


def conv1d_fwd() -> None:
    _require_hopper()
    from machete.kernels.conv1d import Conv1dOp

    x = torch.randn(2, 128, 256, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(256, 4, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    ops = Conv1dOp.schedule(x=x, w=w, y=y, page_size=49152)
    _run_kernel(ops, config=Conv1dOp.kernel_config(ops))


def conv1d_bwd() -> None:
    _require_hopper()
    from machete.kernels.conv1d import Conv1dBwdOp

    dy = torch.randn(2, 128, 256, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(256, 4, dtype=torch.bfloat16, device="cuda")
    dx = torch.empty_like(dy)
    ops = Conv1dBwdOp.schedule(dy=dy, w=w, dx=dx, page_size=49152)
    _run_kernel(ops, config=Conv1dBwdOp.kernel_config(ops))


def cross_entropy() -> None:
    _require_hopper()
    from machete.kernels.cross_entropy import CrossEntropyOp

    logits = torch.randn(8, 4096, dtype=torch.bfloat16, device="cuda")
    targets = torch.randint(0, 4096, (8,), device="cuda", dtype=torch.int64)
    loss = torch.empty(8, dtype=torch.float32, device="cuda")
    grad_logits = torch.empty_like(logits)
    ops = CrossEntropyOp.schedule(
        logits=logits, targets=targets.int(), loss=loss, grad_logits=grad_logits
    )
    _run_kernel(ops, config=CrossEntropyOp.kernel_config(ops))


def rmsnorm_gemm() -> None:
    _require_hopper()
    from machete.kernels.rmsnorm_gemm import RMSNormGemmOp

    a = torch.randn(2, 64, 1024, dtype=torch.bfloat16, device="cuda")
    w_rms = torch.randn(1024, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(4096, 1024, dtype=torch.bfloat16, device="cuda")
    c = torch.empty(2, 64, 4096, dtype=torch.bfloat16, device="cuda")
    ops = RMSNormGemmOp.schedule(a=a, rmsnorm_weight=w_rms, b=b, c=c, page_size=32768)
    _run_kernel(ops, config=RMSNormGemmOp.kernel_config(ops))


OPS = OrderedDict(
    [
        ("activation", activation),
        ("rmsnorm_fwd", rmsnorm_fwd),
        ("rmsnorm_bwd", rmsnorm_bwd),
        ("glu_fwd", glu_fwd),
        ("glu_bwd", glu_bwd),
        ("gemm", gemm),
        ("rope_fwd", rope_fwd),
        ("rope_bwd", rope_bwd),
        ("qknorm_rope", qknorm_rope),
        ("attention_fwd", attention_fwd),
        ("attention_bwd", attention_bwd),
        ("flash_decoding", flash_decoding),
        ("conv1d_fwd", conv1d_fwd),
        ("conv1d_bwd", conv1d_bwd),
        ("cross_entropy", cross_entropy),
        ("rmsnorm_gemm", rmsnorm_gemm),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=list(OPS) + ["all"], default="all")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name in OPS:
            print(name)
        return

    torch.manual_seed(0)
    if args.op == "all":
        for name, fn in OPS.items():
            fn()
            print(f"PASS {name}", flush=True)
        return

    OPS[args.op]()
    print(f"PASS {args.op}", flush=True)


if __name__ == "__main__":
    main()
