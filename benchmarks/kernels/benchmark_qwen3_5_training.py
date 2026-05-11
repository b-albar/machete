#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Showcase Qwen 3.5 forward/backward MLP blocks as Machete megakernels.

This is intentionally not a decode benchmark.  It demonstrates the simple path:
schedule several ops, build one ``Megakernel``, and benchmark it across batch
and sequence lengths.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from machete.kernels.gemm import GemmOp
from machete.kernels.glu import GLUBwdOp, GLUOp
from machete.kernels.qwen_3_5.sm120 import (
    QWEN3_5_EPS,
    QWEN3_5_HIDDEN,
    QWEN3_5_INTERMEDIATE,
)
from machete.kernels.rms_norm import RMSNormBwdOp, RMSNormOp
from machete.megakernel import Megakernel, MegakernelConfig
from machete.utils.benchmark import Benchmark

PAGE_SIZE = 32768
CONFIGS = [
    (1, 128, "forward"),
    (1, 128, "backward"),
    (1, 512, "forward"),
    (1, 512, "backward"),
    (4, 128, "forward"),
    (4, 128, "backward"),
    (4, 512, "forward"),
    (4, 512, "backward"),
]


def _config_for(ops):
    gemm_ops = [op for op in ops if issubclass(op.op_cls, GemmOp)]
    gemm_config = GemmOp.kernel_config(gemm_ops or ops)
    return MegakernelConfig(
        threads_per_block=max(256, gemm_config.threads_per_block),
        page_size=max(op.static_dims.get("page_size", PAGE_SIZE) for op in ops),
        loader_idle_sleep_ns=0,
    )


def _forward_spec(batch: int, seq_len: int):
    dtype = torch.bfloat16
    device = "cuda"
    x = torch.randn(batch, seq_len, QWEN3_5_HIDDEN, dtype=dtype, device=device)
    norm = torch.ones(QWEN3_5_HIDDEN, dtype=dtype, device=device)
    hidden = torch.empty_like(x)
    gate_up = torch.empty(batch, seq_len, 2 * QWEN3_5_INTERMEDIATE, dtype=dtype, device=device)
    mlp = torch.empty(batch, seq_len, QWEN3_5_INTERMEDIATE, dtype=dtype, device=device)
    y = torch.empty_like(x)
    w_gate_up = torch.randn(2 * QWEN3_5_INTERMEDIATE, QWEN3_5_HIDDEN, dtype=dtype, device=device) * 0.02
    w_down = torch.randn(QWEN3_5_HIDDEN, QWEN3_5_INTERMEDIATE, dtype=dtype, device=device) * 0.02

    ops = []
    ops += RMSNormOp.schedule(x=x, weight=norm, y=hidden, tile_sizes={"S": 1}, page_size=PAGE_SIZE, eps=QWEN3_5_EPS)
    ops += GemmOp.schedule(a=hidden, b=w_gate_up, c=gate_up, page_size=PAGE_SIZE)
    ops += GLUOp.schedule(x=gate_up, y=mlp, activation="silu", page_size=PAGE_SIZE)
    ops += GemmOp.schedule(a=mlp, b=w_down, c=y, page_size=PAGE_SIZE)

    kernel = Megakernel(ops, config=_config_for(ops))
    return kernel.bench_spec(keep_alive=[x, norm, hidden, gate_up, mlp, y, w_gate_up, w_down])


def _backward_spec(batch: int, seq_len: int):
    dtype = torch.bfloat16
    device = "cuda"
    x = torch.randn(batch, seq_len, QWEN3_5_HIDDEN, dtype=dtype, device=device)
    hidden = torch.randn_like(x)
    gate_up = torch.randn(batch, seq_len, 2 * QWEN3_5_INTERMEDIATE, dtype=dtype, device=device)
    mlp = torch.randn(batch, seq_len, QWEN3_5_INTERMEDIATE, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    d_mlp = torch.empty_like(mlp)
    d_gate_up = torch.empty_like(gate_up)
    d_hidden = torch.empty_like(hidden)
    dx = torch.empty_like(x)
    norm = torch.ones(QWEN3_5_HIDDEN, dtype=dtype, device=device)
    w_gate_up = torch.randn(2 * QWEN3_5_INTERMEDIATE, QWEN3_5_HIDDEN, dtype=dtype, device=device) * 0.02
    w_down = torch.randn(QWEN3_5_HIDDEN, QWEN3_5_INTERMEDIATE, dtype=dtype, device=device) * 0.02
    dw_gate_up = torch.empty_like(w_gate_up).unsqueeze(0)
    dw_down = torch.empty_like(w_down).unsqueeze(0)

    ops = []
    ops += GemmOp.schedule_backward(dout=dy, a=mlp, b=w_down, da=d_mlp, db=dw_down, page_size=PAGE_SIZE)
    ops += GLUBwdOp.schedule(dy=d_mlp, x=gate_up, dx=d_gate_up, activation="silu", page_size=PAGE_SIZE)
    ops += GemmOp.schedule_backward(dout=d_gate_up, a=hidden, b=w_gate_up, da=d_hidden, db=dw_gate_up, page_size=PAGE_SIZE)
    ops += RMSNormBwdOp.schedule(dout=d_hidden, x=x, weight=norm, dx=dx, page_size=PAGE_SIZE, eps=QWEN3_5_EPS)

    kernel = Megakernel(ops, config=_config_for(ops))
    keep_alive = [x, hidden, gate_up, mlp, dy, d_mlp, d_gate_up, d_hidden, dx, norm, w_gate_up, w_down, dw_gate_up, dw_down]
    return kernel.bench_spec(keep_alive=keep_alive)


@Benchmark.configs(["batch", "seq_len", "direction"], CONFIGS)
def bench_qwen35_block(batch: int, seq_len: int, direction: str):
    if direction == "forward":
        return {"machete": _forward_spec(batch, seq_len)}
    return {"machete": _backward_spec(batch, seq_len)}


if __name__ == "__main__":
    bench_qwen35_block._benchmark.run(mode="kernel", warmup=5, rep=20, export_csv=False)
