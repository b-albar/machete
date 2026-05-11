#!/usr/bin/env python
# Copyright (c) 2026, Machete Authors
"""Build and run an isolated llama.cpp decode-forward benchmark.

This harness keeps llama.cpp outside the Machete tree and measures its normal
CUDA forward path with ``llama-bench``.  For Llama-1B low-latency comparisons,
use ``--prompt 0`` and ``--gen`` tokens; ``--depth`` can prefill the KV cache
before timing generation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO = "https://github.com/ggml-org/llama.cpp"
DEFAULT_LLAMA_DIR = Path("/tmp/llama.cpp")
DEFAULT_BUILD_DIR = DEFAULT_LLAMA_DIR / "build-cuda"
DEFAULT_MODEL_REPO = "bartowski/Llama-3.2-1B-Instruct-GGUF"
DEFAULT_MODEL_FILE = "Llama-3.2-1B-Instruct-f16.gguf"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_llama_cpp(llama_dir: Path) -> None:
    if llama_dir.exists():
        return
    run(["git", "clone", "--depth", "1", REPO, str(llama_dir)])


def build_llama_bench(llama_dir: Path, build_dir: Path) -> Path:
    ensure_llama_cpp(llama_dir)
    run(
        [
            "cmake",
            "-S",
            str(llama_dir),
            "-B",
            str(build_dir),
            "-DGGML_CUDA=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",
        ]
    )
    run(["cmake", "--build", str(build_dir), "--target", "llama-bench", f"-j{os.cpu_count() or 1}"])
    bench = build_dir / "bin" / "llama-bench"
    if not bench.exists():
        raise FileNotFoundError(f"llama-bench was not built at {bench}")
    return bench


def download_model(cache_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub or pass --model /path/to/model.gguf") from exc

    return Path(
        hf_hub_download(
            repo_id=DEFAULT_MODEL_REPO,
            filename=DEFAULT_MODEL_FILE,
            local_dir=cache_dir,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-dir", type=Path, default=DEFAULT_LLAMA_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--model", type=Path, help="Path to a Llama-3.2-1B GGUF file")
    parser.add_argument("--download-model", action="store_true", help=f"Download {DEFAULT_MODEL_REPO}:{DEFAULT_MODEL_FILE}")
    parser.add_argument("--model-cache", type=Path, default=Path("/tmp/llama-models"))
    parser.add_argument("--prompt", type=int, default=0, help="Prompt tokens timed by llama-bench")
    parser.add_argument("--gen", type=int, default=128, help="Single-token decode forwards to time")
    parser.add_argument("--depth", type=int, default=128, help="KV-cache depth prepared before timed generation")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--ubatch", type=int, default=1)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--ngl", type=int, default=99, help="GPU layers to offload")
    parser.add_argument("--flash-attn", type=int, default=1)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit llama-bench JSON")
    args = parser.parse_args()

    bench = build_llama_bench(args.llama_dir, args.build_dir)

    model = args.model
    if model is None and args.download_model:
        model = download_model(args.model_cache)
    if model is None:
        raise SystemExit("Pass --model /path/to/model.gguf or --download-model")
    if not model.exists():
        raise FileNotFoundError(model)

    cmd = [
        str(bench),
        "-m",
        str(model),
        "-ngl",
        str(args.ngl),
        "-p",
        str(args.prompt),
        "-n",
        str(args.gen),
        "-d",
        str(args.depth),
        "-b",
        str(args.batch),
        "-ub",
        str(args.ubatch),
        "-fa",
        str(args.flash_attn),
        "-r",
        str(args.reps),
    ]
    if args.no_warmup:
        cmd.append("--no-warmup")
    if args.json:
        cmd += ["-o", "json"]

    if not shutil.which("nvidia-smi"):
        print("warning: nvidia-smi not found; CUDA benchmark may fail", file=sys.stderr)
    run(cmd)


if __name__ == "__main__":
    main()
