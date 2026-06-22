#!/usr/bin/env python
"""Measure per-DMA-warp register caps for the Qwen 3.5 forward megakernel."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from machete.utils.benchmark import Benchmark

import benchmarks.kernels.benchmark_qwen3_5_layer as qwen_bench


def _parse_caps(text: str) -> tuple[int, int, int]:
    parts = [int(x) for x in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected controller,loader,store")
    return parts[0], parts[1], parts[2]


def _install_config_override(controller: int, loader: int, store: int) -> None:
    base_config_for = qwen_bench._config_for

    def _config_for_with_caps(*args, **kwargs):
        config = base_config_for(*args, **kwargs)
        config.controller_reg_count = controller
        config.loader_reg_count = loader
        config.store_reg_count = store
        return config

    qwen_bench._config_for = _config_for_with_caps


def _build_spec(args: argparse.Namespace):
    controller, loader, store = args.caps
    _install_config_override(controller, loader, store)
    layer_args = qwen_bench._alloc_layer(args.batch, args.seq_len)
    scheduler = qwen_bench._scheduler(args.scheduler, args.fetch_stride)
    spec, _, _ = qwen_bench.megakernel_forward_build(
        *layer_args,
        page_size=args.page_size,
        scheduler=scheduler,
        num_pages=args.num_pages,
        page_free_extra_slots=args.page_free_extra_slots,
        adaptive_gemm_tiling=args.adaptive_gemm_tiling,
        use_qwen_projection=args.use_qwen_projection,
        use_fused_rms_proj=args.use_fused_rms_proj,
    )
    return spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", type=_parse_caps, required=True, help="controller,loader,store register caps")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=qwen_bench.DEFAULT_PAGE_SIZE)
    parser.add_argument("--num-pages", type=int, default=None)
    parser.add_argument("--page-free-extra-slots", type=int, default=0)
    parser.add_argument("--scheduler", choices=("default", "overlap", "overlap-adaptive"), default="overlap-adaptive")
    parser.add_argument("--fetch-stride", type=int, default=None)
    parser.add_argument("--adaptive-gemm-tiling", action="store_true")
    parser.add_argument("--use-qwen-projection", action="store_true")
    parser.add_argument("--use-fused-rms-proj", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--profile-only", action="store_true", help="run one warmup and one profiled launch for NCU")
    args = parser.parse_args()

    spec = _build_spec(args)
    torch.cuda.synchronize()
    if args.profile_only:
        if spec.setup_fn is not None:
            spec.setup_fn()
        spec.launch_fn()
        torch.cuda.synchronize()
        if spec.setup_fn is not None:
            spec.setup_fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        spec.launch_fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print(f"caps={args.caps[0]},{args.caps[1]},{args.caps[2]} profile_launches=1")
        return

    ms = Benchmark()._bench_kernel_func(spec, warmup=args.warmup, rep=args.rep)
    print(f"caps={args.caps[0]},{args.caps[1]},{args.caps[2]} time_ms={ms:.6f}")


if __name__ == "__main__":
    main()
