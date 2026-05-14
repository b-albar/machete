#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Showcase Qwen 3.5 forward/backward MLP blocks as Machete megakernels.

This is intentionally not a decode benchmark.  It demonstrates the simple path:
schedule several ops, build one ``Megakernel``, and benchmark it across batch
and sequence lengths.
"""

from __future__ import annotations

import argparse
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
from machete.megakernel import Megakernel, MegakernelConfig, OverlapTileScheduler
from machete.utils.benchmark import Benchmark
from machete.utils.benchmark_utils import KernelBenchSpec

DEFAULT_PAGE_SIZE = 32768
CONFIGS = [
    (1, 128, DEFAULT_PAGE_SIZE, "forward"),
    (1, 128, DEFAULT_PAGE_SIZE, "backward"),
    (1, 512, DEFAULT_PAGE_SIZE, "forward"),
    (1, 512, DEFAULT_PAGE_SIZE, "backward"),
    (4, 128, DEFAULT_PAGE_SIZE, "forward"),
    (4, 128, DEFAULT_PAGE_SIZE, "backward"),
    (4, 512, DEFAULT_PAGE_SIZE, "forward"),
    (4, 512, DEFAULT_PAGE_SIZE, "backward"),
]
MACHETE_VARIANTS = ("default",)
POINTWISE_PAGE_SIZE_OVERRIDE = 0
NUM_PAGES_OVERRIDE = 0
FETCH_STRIDE_OVERRIDE = 0
GLOBAL_BARRIER_SLEEP_NS = 8
RMS_FWD_TILE_S_OVERRIDE = 0


def _config_for(ops):
    gemm_ops = [op for op in ops if issubclass(op.op_cls, GemmOp)]
    gemm_config = GemmOp.kernel_config(gemm_ops or ops)
    return MegakernelConfig(
        threads_per_block=max(256, gemm_config.threads_per_block),
        page_size=max(op.static_dims.get("page_size", DEFAULT_PAGE_SIZE) for op in ops),
        loader_idle_sleep_ns=0,
        num_pages=NUM_PAGES_OVERRIDE if NUM_PAGES_OVERRIDE > 0 else None,
        global_barrier_sleep_ns=GLOBAL_BARRIER_SLEEP_NS,
    )


def _scheduler_for_variant(variant: str):
    fetch_stride = FETCH_STRIDE_OVERRIDE if FETCH_STRIDE_OVERRIDE > 0 else None
    if variant == "default":
        return None
    if variant == "overlap":
        return OverlapTileScheduler(fetch_stride=fetch_stride)
    if variant == "overlap-adaptive":
        return OverlapTileScheduler(
            fetch_stride=fetch_stride,
            adaptive_fetch_stride=fetch_stride is None,
        )
    raise ValueError(f"unknown Machete variant: {variant}")


def _backward_pointwise_page_size(batch: int, page_size: int) -> int:
    if POINTWISE_PAGE_SIZE_OVERRIDE > 0:
        return POINTWISE_PAGE_SIZE_OVERRIDE
    return page_size


def _forward_rms_tile_s(batch: int, seq_len: int) -> int:
    if RMS_FWD_TILE_S_OVERRIDE > 0:
        return RMS_FWD_TILE_S_OVERRIDE
    return 4 if batch * seq_len >= 2048 else 1


def _forward_spec(batch: int, seq_len: int, page_size: int, variant: str = "default"):
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
    ops += RMSNormOp.schedule(
        x=x,
        weight=norm,
        y=hidden,
        tile_sizes={"S": _forward_rms_tile_s(batch, seq_len)},
        page_size=page_size,
        eps=QWEN3_5_EPS,
    )
    ops += GemmOp.schedule(a=hidden, b=w_gate_up, c=gate_up, page_size=page_size)
    ops += GLUOp.schedule(x=gate_up, y=mlp, activation="silu", page_size=page_size)
    ops += GemmOp.schedule(a=mlp, b=w_down, c=y, page_size=page_size)

    kernel = Megakernel(ops, config=_config_for(ops), scheduler=_scheduler_for_variant(variant))
    return kernel.bench_spec(keep_alive=[x, norm, hidden, gate_up, mlp, y, w_gate_up, w_down])


def _torch_forward_impl(x, norm, w_gate_up, w_down):
    rstd = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + QWEN3_5_EPS)
    hidden = (x * rstd.to(dtype=x.dtype)) * norm
    gate_up = torch.matmul(hidden, w_gate_up.t())
    gate, up = gate_up.chunk(2, dim=-1)
    mlp = torch.nn.functional.silu(gate) * up
    return torch.matmul(mlp, w_down.t())


def _torch_forward_spec(batch: int, seq_len: int, page_size: int):
    dtype = torch.bfloat16
    device = "cuda"
    x = torch.randn(batch, seq_len, QWEN3_5_HIDDEN, dtype=dtype, device=device)
    norm = torch.ones(QWEN3_5_HIDDEN, dtype=dtype, device=device)
    w_gate_up = torch.randn(2 * QWEN3_5_INTERMEDIATE, QWEN3_5_HIDDEN, dtype=dtype, device=device) * 0.02
    w_down = torch.randn(QWEN3_5_HIDDEN, QWEN3_5_INTERMEDIATE, dtype=dtype, device=device) * 0.02

    compiled = torch.compile(_torch_forward_impl, mode="max-autotune")
    sink = {}
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        sink["out"] = compiled(x, norm, w_gate_up, w_down)
    stream.synchronize()

    def _launch():
        sink["out"] = compiled(x, norm, w_gate_up, w_down)

    return KernelBenchSpec(
        launch_fn=_launch,
        stream=(stream, None),
        _keep_alive=[x, norm, w_gate_up, w_down, sink, compiled],
    )


def _backward_spec(batch: int, seq_len: int, page_size: int, variant: str = "default"):
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
    pointwise_page_size = _backward_pointwise_page_size(batch, page_size)

    ops = []
    ops += GemmOp.schedule_backward(dout=dy, a=mlp, b=w_down, da=d_mlp, db=dw_down, page_size=page_size)
    ops += GLUBwdOp.schedule(
        dy=d_mlp,
        x=gate_up,
        dx=d_gate_up,
        activation="silu",
        page_size=pointwise_page_size,
    )
    ops += GemmOp.schedule_backward(dout=d_gate_up, a=hidden, b=w_gate_up, da=d_hidden, db=dw_gate_up, page_size=page_size)
    ops += RMSNormBwdOp.schedule(
        dout=d_hidden,
        x=x,
        weight=norm,
        dx=dx,
        page_size=pointwise_page_size,
        eps=QWEN3_5_EPS,
    )

    kernel = Megakernel(ops, config=_config_for(ops), scheduler=_scheduler_for_variant(variant))
    keep_alive = [x, hidden, gate_up, mlp, dy, d_mlp, d_gate_up, d_hidden, dx, norm, w_gate_up, w_down, dw_gate_up, dw_down]
    return kernel.bench_spec(keep_alive=keep_alive)


def _torch_backward_impl(x, hidden, gate_up, mlp, dy, norm, w_gate_up, w_down):
    d_mlp = torch.matmul(dy, w_down)

    gate, up = gate_up.chunk(2, dim=-1)
    gate_f = gate.float()
    up_f = up.float()
    d_mlp_f = d_mlp.float()
    sig = torch.sigmoid(gate_f)
    silu = gate_f * sig
    d_silu = sig * (1.0 + gate_f * (1.0 - sig))
    d_gate = d_mlp_f * up_f * d_silu
    d_up = d_mlp_f * silu
    d_gate_up = torch.cat((d_gate, d_up), dim=-1).to(dtype=gate_up.dtype)

    d_hidden = torch.matmul(d_gate_up, w_gate_up)
    dw_down = torch.matmul(
        dy.reshape(-1, QWEN3_5_HIDDEN).t(),
        mlp.reshape(-1, QWEN3_5_INTERMEDIATE),
    )
    dw_gate_up = torch.matmul(
        d_gate_up.reshape(-1, 2 * QWEN3_5_INTERMEDIATE).t(),
        hidden.reshape(-1, QWEN3_5_HIDDEN),
    )

    x_f = x.float()
    a = d_hidden.float() * norm.float()
    rstd = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + QWEN3_5_EPS)
    dot = (a * x_f).mean(dim=-1, keepdim=True)
    dx = (a * rstd - x_f * rstd.square() * rstd * dot).to(dtype=x.dtype)
    return dx, dw_gate_up, dw_down


def _torch_backward_spec(batch: int, seq_len: int, page_size: int):
    dtype = torch.bfloat16
    device = "cuda"
    x = torch.randn(batch, seq_len, QWEN3_5_HIDDEN, dtype=dtype, device=device)
    hidden = torch.randn_like(x)
    gate_up = torch.randn(batch, seq_len, 2 * QWEN3_5_INTERMEDIATE, dtype=dtype, device=device)
    mlp = torch.randn(batch, seq_len, QWEN3_5_INTERMEDIATE, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    norm = torch.ones(QWEN3_5_HIDDEN, dtype=dtype, device=device)
    w_gate_up = torch.randn(2 * QWEN3_5_INTERMEDIATE, QWEN3_5_HIDDEN, dtype=dtype, device=device) * 0.02
    w_down = torch.randn(QWEN3_5_HIDDEN, QWEN3_5_INTERMEDIATE, dtype=dtype, device=device) * 0.02

    compiled = torch.compile(_torch_backward_impl, mode="max-autotune")
    sink = {}
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        sink["out"] = compiled(x, hidden, gate_up, mlp, dy, norm, w_gate_up, w_down)
    stream.synchronize()

    def _launch():
        sink["out"] = compiled(x, hidden, gate_up, mlp, dy, norm, w_gate_up, w_down)

    return KernelBenchSpec(
        launch_fn=_launch,
        stream=(stream, None),
        _keep_alive=[x, hidden, gate_up, mlp, dy, norm, w_gate_up, w_down, sink, compiled],
    )


@Benchmark.configs(["batch", "seq_len", "page_size", "direction"], CONFIGS)
def bench_qwen35_block(batch: int, seq_len: int, page_size: int, direction: str):
    if direction == "forward":
        specs = {
            "torch.compile": _torch_forward_spec(batch, seq_len, page_size),
        }
        for variant in MACHETE_VARIANTS:
            name = "machete" if variant == "default" else f"machete-{variant}"
            specs[name] = _forward_spec(batch, seq_len, page_size, variant)
        return specs
    specs = {
        "torch.compile": _torch_backward_spec(batch, seq_len, page_size),
    }
    for variant in MACHETE_VARIANTS:
        name = "machete" if variant == "default" else f"machete-{variant}"
        specs[name] = _backward_spec(batch, seq_len, page_size, variant)
    return specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--seq-len", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--page-size", type=int, nargs="+", default=[DEFAULT_PAGE_SIZE])
    parser.add_argument(
        "--pointwise-page-size",
        type=int,
        default=0,
        help="Backward GLU/RMS page size; 0 uses --page-size.",
    )
    parser.add_argument(
        "--num-pages",
        type=int,
        default=0,
        help="Megakernel ring pages; 0 uses framework auto-selection.",
    )
    parser.add_argument(
        "--fetch-stride",
        type=int,
        default=0,
        help="Overlap scheduler fetch stride; 0 uses scheduler default.",
    )
    parser.add_argument(
        "--global-barrier-sleep-ns",
        type=int,
        default=8,
        help="Sleep interval used by relaxed global barrier waits.",
    )
    parser.add_argument(
        "--rms-fwd-tile-s",
        type=int,
        default=0,
        help="Override RMSNorm forward S tile size; 0 uses benchmark heuristic.",
    )
    parser.add_argument("--direction", choices=["forward", "backward"], nargs="+", default=["forward", "backward"])
    parser.add_argument(
        "--machete-variant",
        choices=["default", "overlap", "overlap-adaptive"],
        nargs="+",
        default=["default"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    args = parser.parse_args()

    MACHETE_VARIANTS = tuple(args.machete_variant)
    POINTWISE_PAGE_SIZE_OVERRIDE = int(args.pointwise_page_size)
    NUM_PAGES_OVERRIDE = int(args.num_pages)
    FETCH_STRIDE_OVERRIDE = int(args.fetch_stride)
    GLOBAL_BARRIER_SLEEP_NS = int(args.global_barrier_sleep_ns)
    RMS_FWD_TILE_S_OVERRIDE = int(args.rms_fwd_tile_s)
    bench_qwen35_block._benchmark._configs = [
        (batch, seq_len, page_size, direction)
        for batch in args.batch
        for seq_len in args.seq_len
        for page_size in args.page_size
        for direction in args.direction
    ]
    bench_qwen35_block._benchmark.run(
        mode="kernel",
        warmup=args.warmup,
        rep=args.rep,
        export_csv=False,
    )
