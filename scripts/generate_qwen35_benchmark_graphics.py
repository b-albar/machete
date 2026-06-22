#!/usr/bin/env python
"""Generate Qwen 3.5 benchmark graphics.

This harness covers two comparison groups:

* Decode: optional llama.cpp, Luce megakernel, and Machete decode rows.
* Full Qwen layer megakernel: torch.compile/PyTorch sequential vs Machete for
  the source-backed one-layer benchmark.

The script can run the local benchmark entry points, or it can plot a prior
JSON file with ``--plot-only``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "benchmark_results" / "qwen35_graphics"
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(paths + ([existing] if existing else []))
    return env


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> str:
    print("+ " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(map(str, cmd))}")
    return proc.stdout


def _parse_first(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"could not parse {label} from command output")
    return float(match.group(1))


def _parse_llama_bench_json(text: str) -> float:
    marker = text.rfind('"build_commit"')
    if marker != -1:
        start = text.rfind("[", 0, marker)
        end = text.find("\n]", marker)
        if end != -1:
            payload = text[start : end + 2]
        else:
            payload = text[start:]
    else:
        payload = text[text.find("[") :] if "[" in text else text
    data = json.loads(payload)
    if isinstance(data, dict):
        data = [data]
    candidates = []
    for row in data:
        if not isinstance(row, dict):
            continue
        for key in ("tokens_second", "tokens_per_second", "tps", "tok_s", "avg_ts"):
            value = row.get(key)
            if isinstance(value, (int, float)) and value > 0:
                candidates.append(float(value))
    if not candidates:
        raise RuntimeError("could not parse llama-bench JSON throughput")
    return candidates[-1]


def _system_order(name: str) -> tuple[int, str]:
    """Order benchmark bars with Machete last."""
    lower = name.lower()
    return (1 if "machete" in lower else 0, lower)


def run_machete_decode(args: argparse.Namespace, context_len: int) -> dict[str, Any]:
    bench_name = "benchmark_qwen3_5_mxfp4_simt_decode.py"
    system_name = "Machete MXFP4"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "benchmarks" / "kernels" / bench_name),
        "--context-len",
        str(context_len),
        "--iters",
        str(args.decode_rep),
        "--top-partitions",
        str(args.machete_top_partitions),
        "--page-size",
        str(args.decode_page_size),
        "--num-pages",
        str(args.decode_num_pages),
        "--threads",
        str(args.machete_threads),
    ]
    cmd.append("--dummy-weights" if args.machete_dummy_weights else "--no-dummy-weights")
    if args.machete_no_final:
        cmd.append("--no-final")
    out = _run(cmd)
    ms = _parse_first(r"machete_qwen_mxfp4_SIMT_decode:\s*([0-9.]+)\s*ms/token", out, "Machete ms/token")
    tok_s = _parse_first(r"machete_qwen_mxfp4_SIMT_decode:.*?,\s*([0-9.]+)\s*tok/s", out, "Machete tok/s")
    return {"system": system_name, "context_len": context_len, "ms_per_token": ms, "tok_s": tok_s}


def run_luce_decode(args: argparse.Namespace, context_len: int) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(args.luce_dir / "megakernel" / "final_bench.py"),
        "--backend",
        args.luce_backend,
        "--prompt-tokens",
        str(context_len),
        "--gen-tokens",
        str(args.decode_tokens),
        "--skip-hf",
    ]
    out = _run(cmd, cwd=args.luce_dir / "megakernel")
    tok_s = _parse_first(r"^tg[0-9]+:\s*([0-9.]+)\s*tok/s", out, "Luce tg tok/s")
    return {"system": "Luce megakernel", "context_len": context_len, "tok_s": tok_s, "ms_per_token": 1000.0 / tok_s}


def run_llamacpp_decode(args: argparse.Namespace, context_len: int) -> dict[str, Any]:
    if args.llamacpp_model is None:
        raise RuntimeError("llama.cpp decode needs --llamacpp-model or --skip-llamacpp")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "benchmark_llamacpp_forward.py"),
        "--model",
        str(args.llamacpp_model),
        "--prompt",
        "0",
        "--gen",
        str(args.decode_tokens),
        "--depth",
        str(context_len),
        "--batch",
        "1",
        "--ubatch",
        "1",
        "--reps",
        str(args.llamacpp_reps),
        "--json",
    ]
    out = _run(cmd)
    tok_s = _parse_llama_bench_json(out)
    return {"system": args.llamacpp_label, "context_len": context_len, "tok_s": tok_s, "ms_per_token": 1000.0 / tok_s}


def run_decode(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context_len in args.decode_context_len:
        if not args.skip_llamacpp:
            if args.llamacpp_model is None:
                msg = "skipping llama.cpp decode: pass --llamacpp-model /path/to/qwen3.5.gguf to enable it"
                if args.strict:
                    raise RuntimeError(msg)
                print(msg, flush=True)
            else:
                rows.append(run_llamacpp_decode(args, context_len))
        if not args.skip_luce:
            luce_bench = args.luce_dir / "megakernel" / "final_bench.py"
            if not luce_bench.exists():
                msg = f"skipping Luce decode: {luce_bench} does not exist"
                if args.strict:
                    raise RuntimeError(msg)
                print(msg, flush=True)
            else:
                try:
                    rows.append(run_luce_decode(args, context_len))
                except Exception as exc:
                    if args.strict:
                        raise
                    print(f"skipping Luce decode after failure: {exc}", flush=True)
        if not args.skip_machete_decode:
            rows.append(run_machete_decode(args, context_len))
    return rows


def run_training(args: argparse.Namespace) -> list[dict[str, Any]]:
    return run_qwen_layer_training(args)


def _load_qwen35_layer_module():
    module_name = "benchmarks.kernels.benchmark_qwen3_5_layer"
    if module_name in sys.modules:
        return sys.modules[module_name]
    py_path = REPO_ROOT / "benchmarks" / "kernels" / "benchmark_qwen3_5_layer.py"
    if not py_path.exists():
        raise RuntimeError(
            "Qwen full-layer graphics require a source-backed benchmark at "
            f"{py_path}. The old MLP-only training benchmark is intentionally "
            "not used for these graphics."
        )
    return __import__(module_name, fromlist=["*"])


def run_qwen_layer_training(args: argparse.Namespace) -> list[dict[str, Any]]:
    layer = _load_qwen35_layer_module()
    if int(args.num_pages) > 0:
        from machete.megakernel import MegakernelConfig as RealMegakernelConfig

        def _forced_config(*config_args, **config_kwargs):
            config_kwargs["page_size"] = int(args.training_page_size)
            config_kwargs["num_pages"] = int(args.num_pages)
            return RealMegakernelConfig(*config_args, **config_kwargs)

        layer.MegakernelConfig = _forced_config
    rows: list[dict[str, Any]] = []
    config_names = ["seq_len", "batch", "page_size"]

    directions = []
    if "forward" in args.training_direction:
        directions.append(("forward", layer.bench_qwen35_layer_fwd))
    if "backward" in args.training_direction:
        directions.append(("backward", layer.bench_qwen35_layer_bwd))

    for direction, bench_fn in directions:
        bench = bench_fn._benchmark
        bench._config_names = config_names
        bench._configs = [
            (seq_len, batch, args.training_page_size)
            for batch in args.training_batch
            for seq_len in args.training_seq_len
        ]
        results = bench.run(
            mode="kernel",
            warmup=args.training_warmup,
            rep=args.training_rep,
            export_csv=False,
            print_summary=True,
        )
        for params_text, systems in results.items():
            params = ast.literal_eval(params_text)
            for system, metrics in systems.items():
                if system == "sequential":
                    label = "PyTorch sequential"
                elif system == "torch_compile":
                    label = "torch.compile"
                elif system == "megakernel":
                    label = "Machete full megakernel"
                else:
                    label = system
                time_ms = float(metrics.get("time_ms", 0.0))
                rows.append(
                    {
                        "system": label,
                        "batch": int(params["batch"]),
                        "seq_len": int(params["seq_len"]),
                        "page_size": int(params["page_size"]),
                        "direction": direction,
                        "time_ms": time_ms,
                        "tokens_per_s": (int(params["batch"]) * int(params["seq_len"]) * 1000.0 / time_ms)
                        if time_ms > 0
                        else 0.0,
                    }
                )
            if direction == "backward" and "torch_compile" not in systems:
                time_ms = _bench_compiled_qwen_layer_backward(
                    layer,
                    batch=int(params["batch"]),
                    seq_len=int(params["seq_len"]),
                    warmup=args.training_warmup,
                    rep=args.training_rep,
                )
                rows.append(
                    {
                        "system": "torch.compile",
                        "batch": int(params["batch"]),
                        "seq_len": int(params["seq_len"]),
                        "page_size": int(params["page_size"]),
                        "direction": direction,
                        "time_ms": time_ms,
                        "tokens_per_s": (int(params["batch"]) * int(params["seq_len"]) * 1000.0 / time_ms)
                        if time_ms > 0
                        else 0.0,
                    }
                )
    return rows


def _bench_compiled_qwen_layer_backward(layer: Any, *, batch: int, seq_len: int, warmup: int, rep: int) -> float:
    import torch

    dtype = torch.bfloat16
    device = "cuda"
    hidden = int(layer.HIDDEN)
    q_dim = int(layer.Q_DIM)
    kv_dim = int(layer.KV_DIM)
    head_dim = int(layer.HEAD_DIM)
    rotary_d2 = int(layer.D2)
    intermediate = int(layer.INTERMEDIATE)

    tensors = (
        torch.randn(batch, seq_len, hidden, dtype=dtype, device=device),
        torch.randn(batch, seq_len, hidden, dtype=dtype, device=device),
        torch.randn(hidden, dtype=dtype, device=device),
        torch.randn(q_dim, hidden, dtype=dtype, device=device) * 0.02,
        torch.randn(kv_dim, hidden, dtype=dtype, device=device) * 0.02,
        torch.randn(kv_dim, hidden, dtype=dtype, device=device) * 0.02,
        torch.ones(head_dim, dtype=dtype, device=device),
        torch.ones(head_dim, dtype=dtype, device=device),
        torch.randn(seq_len, rotary_d2, dtype=dtype, device=device),
        torch.randn(seq_len, rotary_d2, dtype=dtype, device=device),
        torch.randn(hidden, q_dim, dtype=dtype, device=device) * 0.02,
        torch.randn(hidden, dtype=dtype, device=device),
        torch.randn(2 * intermediate, hidden, dtype=dtype, device=device) * 0.02,
        torch.randn(hidden, intermediate, dtype=dtype, device=device) * 0.02,
        torch.randn(batch, seq_len, hidden, dtype=dtype, device=device),
    )
    compiled = torch.compile(layer.sequential_layer_bwd, mode="max-autotune")
    compiled(*tensors)
    torch.cuda.synchronize()
    for _ in range(warmup):
        compiled(*tensors)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        compiled(*tensors)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / rep)


def plot_decode(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    systems = sorted({row["system"] for row in rows}, key=_system_order)
    contexts = sorted({int(row["context_len"]) for row in rows})
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(contexts))

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for idx, system in enumerate(systems):
        values = []
        for context in contexts:
            match = next((row for row in rows if row["system"] == system and int(row["context_len"]) == context), None)
            values.append(float(match["tok_s"]) if match else np.nan)
        rects = ax.bar(x + (idx - (len(systems) - 1) / 2) * width, values, width, label=system)
        ax.bar_label(rects, fmt="%.0f", padding=3)
    ax.set_title("Qwen 3.5 Decode Throughput\nhigher is better")
    ax.set_xlabel("KV context length / prompt depth")
    ax.set_ylabel("tokens/s")
    ax.set_xticks(x, [str(v) for v in contexts])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "qwen35_decode_tok_s.png", dpi=180)
    plt.close(fig)


def plot_training(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    for direction in sorted({row["direction"] for row in rows}):
        for batch in sorted({int(row["batch"]) for row in rows if row["direction"] == direction}):
            subset = [row for row in rows if row["direction"] == direction and int(row["batch"]) == batch]
            systems = sorted({row["system"] for row in subset}, key=_system_order)
            seq_lens = sorted({int(row["seq_len"]) for row in subset})
            width = 0.8 / max(1, len(systems))
            x = np.arange(len(seq_lens))
            fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
            for idx, system in enumerate(systems):
                values = []
                for seq_len in seq_lens:
                    match = next(
                        (row for row in subset if row["system"] == system and int(row["seq_len"]) == seq_len),
                        None,
                    )
                    values.append(float(match["time_ms"]) if match else np.nan)
                rects = ax.bar(x + (idx - (len(systems) - 1) / 2) * width, values, width, label=system)
                ax.bar_label(rects, fmt="%.2f", padding=3)
            ax.set_title(f"Qwen 3.5 Full Layer Megakernel {direction.title()} Latency, batch={batch}\nlower is better")
            ax.set_xlabel("sequence length")
            ax.set_ylabel("time (ms)")
            ax.set_xticks(x, [str(v) for v in seq_lens])
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
            fig.savefig(out_dir / f"qwen35_{direction}_b{batch}_time_ms.png", dpi=180)
            plt.close(fig)


def load_or_run(args: argparse.Namespace) -> dict[str, Any]:
    if args.plot_only:
        return json.loads(args.plot_only.read_text())

    payload: dict[str, Any] = {"decode": [], "training": []}
    if args.group in ("all", "decode"):
        payload["decode"] = run_decode(args)
    if args.group in ("all", "training"):
        payload["training"] = run_training(args)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("all", "decode", "training"), default="all")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--plot-only", type=Path, help="Plot an existing qwen35_benchmark_results.json file")
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping unavailable external backends")

    parser.add_argument("--decode-context-len", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--decode-warmup", type=int, default=2)
    parser.add_argument("--decode-rep", type=int, default=5)
    parser.add_argument(
        "--decode-page-size",
        type=int,
        default=32768,
        help="Machete decode page size. Use 65536 for 8K+ context.",
    )
    parser.add_argument(
        "--decode-num-pages",
        type=int,
        default=3,
        help="Machete decode ring pages.",
    )
    parser.add_argument("--skip-llamacpp", action="store_true")
    parser.add_argument("--skip-luce", action="store_true")
    parser.add_argument("--skip-machete-decode", action="store_true")
    parser.add_argument("--llamacpp-model", type=Path)
    parser.add_argument("--llamacpp-label", default="llama.cpp")
    parser.add_argument("--llamacpp-reps", type=int, default=5)
    parser.add_argument("--luce-dir", type=Path, default=Path("/tmp/lucebox-hub"))
    parser.add_argument("--luce-backend", choices=("auto", "bf16", "nvfp4"), default="auto")
    parser.add_argument("--machete-scheduler", choices=("default", "overlap", "overlap-adaptive"), default="overlap")
    parser.add_argument(
        "--machete-decode-bench",
        choices=("mxfp4",),
        default="mxfp4",
        help="Machete decode benchmark entry point used for decode graphics.",
    )
    parser.add_argument("--machete-dummy-weights", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--machete-no-final", action="store_true")
    parser.add_argument("--machete-top-partitions", type=int, default=70)
    parser.add_argument("--machete-threads", type=int, default=512)

    parser.add_argument("--training-batch", type=int, nargs="+", default=[1])
    parser.add_argument("--training-seq-len", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--training-page-size", type=int, default=32768)
    parser.add_argument("--training-direction", choices=("forward", "backward"), nargs="+", default=["forward", "backward"])
    parser.add_argument("--training-warmup", type=int, default=5)
    parser.add_argument("--training-rep", type=int, default=20)
    parser.add_argument("--num-pages", type=int, default=0)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_or_run(args)
    results_path = args.out_dir / "qwen35_benchmark_results.json"
    results_path.write_text(json.dumps(payload, indent=2) + "\n")
    plot_decode(payload.get("decode", []), args.out_dir)
    plot_training(payload.get("training", []), args.out_dir)
    print(f"wrote {results_path}")
    print(f"wrote PNGs under {args.out_dir}")


if __name__ == "__main__":
    main()
