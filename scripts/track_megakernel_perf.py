#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Track megakernel cold build time and warm runtime on representative builds.

The script runs each measurement in a fresh subprocess so compile-time numbers
are not polluted by process-local caches:
- ``Megakernel._compiled_kernel_cache``
- noinline function caches
- builder-local warm state

Supported workloads:
- ``decode``: full Qwen decode megakernel with configurable layer count
- ``prefill``: full Qwen prefill benchmark path built from fused layer kernels
- ``backward``: full Qwen activation-backward megakernel with configurable layer count
- ``layer-fwd``: single Qwen layer forward megakernel
- ``layer-bwd``: single Qwen layer backward megakernel

Examples:
    python scripts/track_megakernel_perf.py decode --layers 8 12 24 36
    python scripts/track_megakernel_perf.py prefill --seq-lens 128 512 1024
    python scripts/track_megakernel_perf.py layer-fwd --seq-lens 128 512 1024
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    root_path = str(REPO_ROOT)
    existing = env.get("PYTHONPATH")
    joined = f"{root_path}:{src_path}"
    env["PYTHONPATH"] = joined if not existing else f"{joined}:{existing}"
    return env


def _run_worker(args: list[str]) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", *args]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_worker_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "worker failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"worker produced no output: {' '.join(cmd)}")
    return json.loads(lines[-1])


def _format_ms(value: float) -> str:
    return f"{value:.3f}"


def _print_table(rows: list[dict], key_fields: list[str]) -> None:
    headers = key_fields + ["build_ms", "runtime_ms"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in key_fields:
            widths[header] = max(widths[header], len(str(row[header])))
        widths["build_ms"] = max(widths["build_ms"], len(_format_ms(row["build_ms"])))
        widths["runtime_ms"] = max(widths["runtime_ms"], len(_format_ms(row["runtime_ms"])))

    def _fmt(header: str, value: str) -> str:
        return value.ljust(widths[header])

    print("  ".join(_fmt(h, h) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        values = [str(row[h]) for h in key_fields]
        values += [_format_ms(row["build_ms"]), _format_ms(row["runtime_ms"])]
        print("  ".join(_fmt(h, v) for h, v in zip(headers, values)))


def _track_decode(args: argparse.Namespace) -> None:
    rows = []
    for layers in args.layers:
        worker_args = [
            "decode-worker",
            "--layers", str(layers),
            "--batch", str(args.batch),
            "--context-len", str(args.context_len),
            "--page-size", str(args.page_size),
            "--warmup", str(args.warmup),
            "--rep", str(args.rep),
        ]
        if args.qwen_sm120:
            worker_args.append("--qwen-sm120")
        rows.append(_run_worker(worker_args))
    _print_table(rows, ["mode", "layers", "batch", "context_len", "page_size"])


def _track_prefill(args: argparse.Namespace) -> None:
    rows = []
    for seq_len in args.seq_lens:
        worker_args = [
            "prefill-worker",
            "--batch", str(args.batch),
            "--seq-len", str(seq_len),
            "--page-size", str(args.page_size),
            "--warmup", str(args.warmup),
            "--rep", str(args.rep),
        ]
        rows.append(_run_worker(worker_args))
    _print_table(rows, ["mode", "batch", "seq_len", "page_size"])


def _track_backward(args: argparse.Namespace) -> None:
    rows = []
    for layers in args.layers:
        rows.append(_run_worker([
            "backward-worker",
            "--layers", str(layers),
            "--batch", str(args.batch),
            "--seq-len", str(args.seq_len),
            "--page-size", str(args.page_size),
            "--warmup", str(args.warmup),
            "--rep", str(args.rep),
        ]))
    _print_table(rows, ["mode", "layers", "batch", "seq_len", "page_size"])


def _track_layer(args: argparse.Namespace, mode: str) -> None:
    rows = []
    for seq_len in args.seq_lens:
        rows.append(_run_worker([
            mode,
            "--batch", str(args.batch),
            "--seq-len", str(seq_len),
            "--page-size", str(args.page_size),
            "--warmup", str(args.warmup),
            "--rep", str(args.rep),
        ]))
    _print_table(rows, ["mode", "batch", "seq_len", "page_size"])


def _measure_decode(args: argparse.Namespace) -> dict:
    import torch

    from benchmarks.kernels.benchmark_qwen3_5_decode import (
        DECODE_S,
        HIDDEN,
        allocate_kv_cache,
        allocate_model_weights,
        megakernel_decode_build,
    )
    from machete.utils.benchmark import Benchmark

    torch.manual_seed(42)
    dtype = torch.bfloat16
    weights = allocate_model_weights(dtype=dtype, device="cuda")
    k_caches, v_caches = allocate_kv_cache(args.batch, args.context_len + DECODE_S, dtype=dtype, device="cuda")
    for i in range(len(k_caches)):
        k_caches[i][:, :, :args.context_len, :].normal_()
        v_caches[i][:, :, :args.context_len, :].normal_()
    x = torch.randn(args.batch, DECODE_S, HIDDEN, dtype=dtype, device="cuda")
    residual = torch.zeros(args.batch, DECODE_S, HIDDEN, dtype=dtype, device="cuda")

    t0 = time.perf_counter()
    spec, _logits, _keep_alive = megakernel_decode_build(
        args.batch,
        args.context_len,
        k_caches,
        v_caches,
        weights,
        x_init=x,
        residual_init=residual,
        page_size=args.page_size,
        num_layers=args.layers,
        use_qwen_sm120_ops=args.qwen_sm120,
    )
    torch.cuda.synchronize()
    build_ms = (time.perf_counter() - t0) * 1000.0
    runtime_ms = Benchmark()._bench_kernel_func(spec, warmup=args.warmup, rep=args.rep)
    return {
        "mode": "decode-qwen-sm120" if args.qwen_sm120 else "decode",
        "layers": args.layers,
        "batch": args.batch,
        "context_len": args.context_len,
        "page_size": args.page_size,
        "build_ms": build_ms,
        "runtime_ms": runtime_ms,
    }


def _measure_prefill(args: argparse.Namespace) -> dict:
    import torch

    from benchmarks.kernels.benchmark_qwen3_5_decode import HIDDEN, allocate_model_weights
    from benchmarks.kernels.benchmark_qwen3_5_prefill import megakernel_prefill_build
    from machete.utils.benchmark import Benchmark

    torch.manual_seed(42)
    dtype = torch.bfloat16
    weights = allocate_model_weights(dtype=dtype, device="cuda")
    x = torch.randn(args.batch, args.seq_len, HIDDEN, dtype=dtype, device="cuda")
    residual = torch.randn(args.batch, args.seq_len, HIDDEN, dtype=dtype, device="cuda")

    t0 = time.perf_counter()
    spec, _logits, _residual = megakernel_prefill_build(
        args.batch,
        args.seq_len,
        x,
        residual,
        weights,
        page_size=args.page_size,
    )
    torch.cuda.synchronize()
    build_ms = (time.perf_counter() - t0) * 1000.0
    runtime_ms = Benchmark()._bench_kernel_func(spec, warmup=args.warmup, rep=args.rep)
    return {
        "mode": "prefill",
        "batch": args.batch,
        "seq_len": args.seq_len,
        "page_size": args.page_size,
        "build_ms": build_ms,
        "runtime_ms": runtime_ms,
    }


def _measure_backward(args: argparse.Namespace) -> dict:
    import torch

    from benchmarks.kernels.benchmark_qwen3_5_backward import megakernel_backward_build
    from benchmarks.kernels.benchmark_qwen3_5_decode import HIDDEN, VOCAB_SIZE, allocate_model_weights
    from machete.utils.benchmark import Benchmark

    torch.manual_seed(42)
    dtype = torch.bfloat16
    weights = allocate_model_weights(dtype=dtype, device="cuda")
    x = torch.randn(args.batch, args.seq_len, HIDDEN, dtype=dtype, device="cuda")
    residual = torch.randn(args.batch, args.seq_len, HIDDEN, dtype=dtype, device="cuda")
    d_logits = torch.randn(args.batch, args.seq_len, VOCAB_SIZE, dtype=dtype, device="cuda")

    t0 = time.perf_counter()
    spec, _dx, _dres = megakernel_backward_build(
        args.batch,
        args.seq_len,
        x,
        residual,
        weights,
        d_logits,
        page_size=args.page_size,
        num_layers=args.layers,
    )
    torch.cuda.synchronize()
    build_ms = (time.perf_counter() - t0) * 1000.0
    runtime_ms = Benchmark()._bench_kernel_func(spec, warmup=args.warmup, rep=args.rep)
    return {
        "mode": "backward",
        "layers": args.layers,
        "batch": args.batch,
        "seq_len": args.seq_len,
        "page_size": args.page_size,
        "build_ms": build_ms,
        "runtime_ms": runtime_ms,
    }


def _measure_layer_fwd(args: argparse.Namespace) -> dict:
    import torch

    from benchmarks.kernels.benchmark_qwen3_5_layer import (
        D2,
        HIDDEN,
        HEAD_DIM,
        INTERMEDIATE,
        KV_DIM,
        Q_DIM,
        megakernel_forward_build,
    )
    from machete.utils.benchmark import Benchmark

    torch.manual_seed(42)
    dtype = torch.bfloat16
    batch = args.batch
    seq_len = args.seq_len

    x = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device="cuda")
    residual = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device="cuda")
    w_attn_norm = torch.randn(HIDDEN, dtype=dtype, device="cuda")
    W_q = torch.randn(Q_DIM, HIDDEN, dtype=dtype, device="cuda") * 0.02
    W_k = torch.randn(KV_DIM, HIDDEN, dtype=dtype, device="cuda") * 0.02
    W_v = torch.randn(KV_DIM, HIDDEN, dtype=dtype, device="cuda") * 0.02
    w_q_norm = torch.ones(HEAD_DIM, dtype=dtype, device="cuda")
    w_k_norm = torch.ones(HEAD_DIM, dtype=dtype, device="cuda")
    cos = torch.randn(seq_len, D2, dtype=dtype, device="cuda")
    sin = torch.randn(seq_len, D2, dtype=dtype, device="cuda")
    W_o = torch.randn(HIDDEN, Q_DIM, dtype=dtype, device="cuda") * 0.02
    w_mlp_norm = torch.randn(HIDDEN, dtype=dtype, device="cuda")
    W_gate_up = torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=dtype, device="cuda") * 0.02
    W_down = torch.randn(HIDDEN, INTERMEDIATE, dtype=dtype, device="cuda") * 0.02

    t0 = time.perf_counter()
    spec, _out, _res = megakernel_forward_build(
        batch, seq_len, x, residual, w_attn_norm, W_q, W_k, W_v,
        w_q_norm, w_k_norm, cos, sin, W_o, w_mlp_norm, W_gate_up, W_down,
        page_size=args.page_size,
    )
    torch.cuda.synchronize()
    build_ms = (time.perf_counter() - t0) * 1000.0
    runtime_ms = Benchmark()._bench_kernel_func(spec, warmup=args.warmup, rep=args.rep)
    return {
        "mode": "layer-fwd",
        "batch": batch,
        "seq_len": seq_len,
        "page_size": args.page_size,
        "build_ms": build_ms,
        "runtime_ms": runtime_ms,
    }


def _measure_layer_bwd(args: argparse.Namespace) -> dict:
    import torch

    from benchmarks.kernels.benchmark_qwen3_5_layer import (
        D2,
        HIDDEN,
        HEAD_DIM,
        INTERMEDIATE,
        KV_DIM,
        Q_DIM,
        megakernel_layer_bwd_build,
    )
    from machete.utils.benchmark import Benchmark

    torch.manual_seed(42)
    dtype = torch.bfloat16
    batch = args.batch
    seq_len = args.seq_len

    x = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device="cuda")
    residual = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device="cuda")
    w_attn_norm = torch.randn(HIDDEN, dtype=dtype, device="cuda")
    W_q = torch.randn(Q_DIM, HIDDEN, dtype=dtype, device="cuda") * 0.02
    W_k = torch.randn(KV_DIM, HIDDEN, dtype=dtype, device="cuda") * 0.02
    W_v = torch.randn(KV_DIM, HIDDEN, dtype=dtype, device="cuda") * 0.02
    w_q_norm = torch.ones(HEAD_DIM, dtype=dtype, device="cuda")
    w_k_norm = torch.ones(HEAD_DIM, dtype=dtype, device="cuda")
    cos = torch.randn(seq_len, D2, dtype=dtype, device="cuda")
    sin = torch.randn(seq_len, D2, dtype=dtype, device="cuda")
    W_o = torch.randn(HIDDEN, Q_DIM, dtype=dtype, device="cuda") * 0.02
    w_mlp_norm = torch.randn(HIDDEN, dtype=dtype, device="cuda")
    W_gate_up = torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=dtype, device="cuda") * 0.02
    W_down = torch.randn(HIDDEN, INTERMEDIATE, dtype=dtype, device="cuda") * 0.02

    t0 = time.perf_counter()
    spec, _errs = megakernel_layer_bwd_build(
        batch, seq_len, x, residual, w_attn_norm, W_q, W_k, W_v,
        w_q_norm, w_k_norm, cos, sin, W_o, w_mlp_norm, W_gate_up, W_down,
        page_size=args.page_size,
    )
    torch.cuda.synchronize()
    build_ms = (time.perf_counter() - t0) * 1000.0
    runtime_ms = Benchmark()._bench_kernel_func(spec, warmup=args.warmup, rep=args.rep)
    return {
        "mode": "layer-bwd",
        "batch": batch,
        "seq_len": seq_len,
        "page_size": args.page_size,
        "build_ms": build_ms,
        "runtime_ms": runtime_ms,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="mode", required=True)

    decode = sub.add_parser("decode")
    decode.add_argument("--layers", nargs="+", type=int, default=[8, 12, 24, 36])
    decode.add_argument("--batch", type=int, default=1)
    decode.add_argument("--context-len", type=int, default=128)
    decode.add_argument("--page-size", type=int, default=32768)
    decode.add_argument("--warmup", type=int, default=5)
    decode.add_argument("--rep", type=int, default=20)
    decode.add_argument("--qwen-sm120", action="store_true")

    prefill = sub.add_parser("prefill")
    prefill.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512, 1024])
    prefill.add_argument("--batch", type=int, default=1)
    prefill.add_argument("--page-size", type=int, default=32768)
    prefill.add_argument("--warmup", type=int, default=5)
    prefill.add_argument("--rep", type=int, default=20)

    backward = sub.add_parser("backward")
    backward.add_argument("--layers", nargs="+", type=int, default=[1, 2, 4])
    backward.add_argument("--batch", type=int, default=1)
    backward.add_argument("--seq-len", type=int, default=128)
    backward.add_argument("--page-size", type=int, default=32768)
    backward.add_argument("--warmup", type=int, default=5)
    backward.add_argument("--rep", type=int, default=20)

    layer_fwd = sub.add_parser("layer-fwd")
    layer_fwd.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512, 1024])
    layer_fwd.add_argument("--batch", type=int, default=1)
    layer_fwd.add_argument("--page-size", type=int, default=32768)
    layer_fwd.add_argument("--warmup", type=int, default=5)
    layer_fwd.add_argument("--rep", type=int, default=20)

    layer_bwd = sub.add_parser("layer-bwd")
    layer_bwd.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512, 1024])
    layer_bwd.add_argument("--batch", type=int, default=1)
    layer_bwd.add_argument("--page-size", type=int, default=32768)
    layer_bwd.add_argument("--warmup", type=int, default=5)
    layer_bwd.add_argument("--rep", type=int, default=20)

    worker_decode = sub.add_parser("decode-worker")
    worker_decode.add_argument("--layers", type=int, required=True)
    worker_decode.add_argument("--batch", type=int, required=True)
    worker_decode.add_argument("--context-len", type=int, required=True)
    worker_decode.add_argument("--page-size", type=int, required=True)
    worker_decode.add_argument("--warmup", type=int, required=True)
    worker_decode.add_argument("--rep", type=int, required=True)
    worker_decode.add_argument("--qwen-sm120", action="store_true")

    worker_prefill = sub.add_parser("prefill-worker")
    worker_prefill.add_argument("--batch", type=int, required=True)
    worker_prefill.add_argument("--seq-len", type=int, required=True)
    worker_prefill.add_argument("--page-size", type=int, required=True)
    worker_prefill.add_argument("--warmup", type=int, required=True)
    worker_prefill.add_argument("--rep", type=int, required=True)

    worker_backward = sub.add_parser("backward-worker")
    worker_backward.add_argument("--layers", type=int, required=True)
    worker_backward.add_argument("--batch", type=int, required=True)
    worker_backward.add_argument("--seq-len", type=int, required=True)
    worker_backward.add_argument("--page-size", type=int, required=True)
    worker_backward.add_argument("--warmup", type=int, required=True)
    worker_backward.add_argument("--rep", type=int, required=True)

    worker_layer_fwd = sub.add_parser("layer-fwd-worker")
    worker_layer_fwd.add_argument("--batch", type=int, required=True)
    worker_layer_fwd.add_argument("--seq-len", type=int, required=True)
    worker_layer_fwd.add_argument("--page-size", type=int, required=True)
    worker_layer_fwd.add_argument("--warmup", type=int, required=True)
    worker_layer_fwd.add_argument("--rep", type=int, required=True)

    worker_layer_bwd = sub.add_parser("layer-bwd-worker")
    worker_layer_bwd.add_argument("--batch", type=int, required=True)
    worker_layer_bwd.add_argument("--seq-len", type=int, required=True)
    worker_layer_bwd.add_argument("--page-size", type=int, required=True)
    worker_layer_bwd.add_argument("--warmup", type=int, required=True)
    worker_layer_bwd.add_argument("--rep", type=int, required=True)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.worker:
        if args.mode == "decode-worker":
            print(json.dumps(_measure_decode(args)))
            return
        if args.mode == "prefill-worker":
            print(json.dumps(_measure_prefill(args)))
            return
        if args.mode == "backward-worker":
            print(json.dumps(_measure_backward(args)))
            return
        if args.mode == "layer-fwd-worker":
            print(json.dumps(_measure_layer_fwd(args)))
            return
        if args.mode == "layer-bwd-worker":
            print(json.dumps(_measure_layer_bwd(args)))
            return
        raise SystemExit(f"unsupported worker mode: {args.mode}")

    if args.mode == "decode":
        _track_decode(args)
        return
    if args.mode == "prefill":
        _track_prefill(args)
        return
    if args.mode == "backward":
        _track_backward(args)
        return
    if args.mode == "layer-fwd":
        _track_layer(args, "layer-fwd-worker")
        return
    if args.mode == "layer-bwd":
        _track_layer(args, "layer-bwd-worker")
        return
    raise SystemExit(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
