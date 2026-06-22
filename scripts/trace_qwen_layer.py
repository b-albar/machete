#!/usr/bin/env python
"""Write cutedsl-trace files for Qwen 3.5 layer forward/backward."""

import argparse
import contextlib
import io
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from benchmarks.kernels.benchmark_qwen3_5_layer import (
    _alloc_layer,
    megakernel_forward_build,
    megakernel_layer_bwd_build,
)
from machete.kernels.qwen_3_5.qwen_3_5_forward import Qwen3_5ForwardOverlapScheduler
from machete.megakernel import (
    OverlapTileScheduler,
    TimingAwareOverlapScheduler,
    TimingProfile,
    scheduler_from_autotune_summary,
)


def _alloc_qwen_layer(batch: int, seq_len: int):
    return _alloc_layer(batch, seq_len)


def _default_perfetto_name(args) -> str:
    parts = [
        "qwen35_layer",
        args.mode,
        f"b{args.batch}",
        f"s{args.seq_len}",
        args.scheduler,
    ]
    if args.mode == "fwd":
        parts.append(f"ops{args.forward_op_limit}" if args.forward_op_limit > 0 else "full")
        parts.append("packed_qk" if args.packed_qk else "separate_qk")
        if args.adaptive_gemm_tiling:
            parts.append("adaptive_gemm")
        if args.qwen_projection:
            parts.append("qwen_projection")
        if args.packed_qkv_projection:
            parts.append("packed_qkv")
        parts.append("qknorm4d" if args.qknorm_4d else "flat_qknorm")
        if args.fused_rms_proj:
            parts.append("fused_rms_proj")
        if args.rms_tile_s is not None:
            parts.append(f"rmsS{args.rms_tile_s}")
    if args.fetch_stride > 0:
        parts.append(f"stride{args.fetch_stride}")
    if args.dependency_slack_waves > 0:
        parts.append(f"slack{args.dependency_slack_waves}")
    return "_".join(parts) + ".perfetto.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fwd", "bwd"], default="fwd")
    parser.add_argument(
        "--scheduler",
        choices=["default", "overlap", "qwen-forward", "qwen-causal-rank", "timing", "autotuned"],
        default="qwen-causal-rank",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=32768)
    qk_group = parser.add_mutually_exclusive_group()
    qk_group.add_argument("--packed-qk", dest="packed_qk", action="store_true", default=True)
    qk_group.add_argument("--separate-qk", dest="packed_qk", action="store_false")
    parser.add_argument("--rms-tile-s", type=int, default=None)
    parser.add_argument("--adaptive-gemm-tiling", action="store_true")
    parser.add_argument(
        "--qwen-projection",
        action="store_true",
        help="Use the Qwen-specific packed Q/K projection op for forward packed-QK.",
    )
    qkv_group = parser.add_mutually_exclusive_group()
    qkv_group.add_argument(
        "--packed-qkv-projection",
        dest="packed_qkv_projection",
        action="store_true",
        default=True,
        help="Use one Qwen-local QKV projection buffer/op instead of separate packed QK + V projection.",
    )
    qkv_group.add_argument(
        "--separate-v-projection",
        dest="packed_qkv_projection",
        action="store_false",
        help="Use the older packed-QK plus separate V projection path.",
    )
    rms_proj_group = parser.add_mutually_exclusive_group()
    rms_proj_group.add_argument(
        "--fused-rms-proj",
        dest="fused_rms_proj",
        action="store_true",
        default=True,
        help="Fuse the first RMSNorm into the first projection op for forward mode.",
    )
    rms_proj_group.add_argument(
        "--no-fused-rms-proj",
        dest="fused_rms_proj",
        action="store_false",
        help="Keep the first RMSNorm as a separate op.",
    )
    attn_group = parser.add_mutually_exclusive_group()
    attn_group.add_argument(
        "--tma-attention",
        dest="attention_path",
        action="store_const",
        const="tma",
        default="tma",
        help="Use Qwen-local full attention with compute-issued TMA K/V loads.",
    )
    attn_group.add_argument(
        "--split-attention",
        dest="attention_path",
        action="store_const",
        const="split",
        help="Use split-KV attention experiment.",
    )
    parser.add_argument(
        "--two-page-attention",
        action="store_true",
        help="With --tma-attention, use separate Q and KV/O shared-memory pages.",
    )
    parser.add_argument(
        "--three-page-attention",
        action="store_true",
        default=True,
        help="With --tma-attention, use separate Q/O, K, and V shared-memory pages.",
    )
    parser.add_argument(
        "--split-attention-splits",
        type=int,
        default=1,
        help="Number of split-KV chunks for --split-attention.",
    )
    parser.add_argument(
        "--attention-tile-m",
        type=int,
        default=None,
        help="Override Qwen forward attention M tile for overlap experiments.",
    )
    qknorm_group = parser.add_mutually_exclusive_group()
    qknorm_group.add_argument(
        "--qknorm-4d",
        dest="qknorm_4d",
        action="store_true",
        default=True,
        help="Use the Qwen-local 4D packed QKNorm dependency path.",
    )
    qknorm_group.add_argument(
        "--flat-qknorm",
        dest="qknorm_4d",
        action="store_false",
        help="Use the older flattened packed QKNorm path.",
    )
    parser.add_argument(
        "--forward-op-limit",
        type=int,
        default=0,
        help="For forward mode, schedule only the first N Qwen forward ops. Default 0 schedules the full layer.",
    )
    parser.add_argument(
        "--fetch-stride",
        type=int,
        default=0,
        help="Overlap scheduler fetch stride; 0 uses the scheduler default.",
    )
    parser.add_argument("--attention-bwd-batch-window", type=int, default=4)
    parser.add_argument(
        "--projection-bwd-input-tile-s",
        type=int,
        default=0,
        help="Use smaller S tiles for Q/K/V projection dA GEMMs in backward. Default 0 disables the split.",
    )
    parser.add_argument(
        "--projection-bwd-reduce-tile-n",
        type=int,
        default=0,
        help="Split Q/K/V projection dA over reduction chunks and atomic-accumulate dA. Default 0 disables it.",
    )
    parser.add_argument("--dependency-slack-waves", type=int, default=0)
    parser.add_argument(
        "--timing-profile",
        default=None,
        help="Perfetto JSON trace used by --scheduler timing.",
    )
    parser.add_argument(
        "--autotune-summary",
        default=None,
        help="Autotune summary JSON used by --scheduler autotuned.",
    )
    parser.add_argument(
        "--timing-mode",
        choices=["critical", "compute", "data_wait", "short", "avoid_data_wait"],
        default="critical",
        help="Which per-op timing score to prioritize for --scheduler timing.",
    )
    parser.add_argument(
        "--timing-position",
        choices=["late", "before_op", "after_resource"],
        default="late",
        help="Where to place timing in the ready-tile priority tuple.",
    )
    parser.add_argument(
        "--dependency-slack-op-idx",
        action="append",
        type=int,
        default=None,
        help="Apply dependency slack only to this op index. May be repeated.",
    )
    parser.add_argument("--output", default="traces/qwen_layer.nanotrace")
    parser.add_argument("--perfetto-output", default=None)
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Directory for generated Perfetto traces. Ignored when --perfetto-output is set.",
    )
    parser.add_argument(
        "--deps-output-prefix",
        default=None,
        help="Write static dependency graph CSVs as PREFIX.op_deps.csv and PREFIX.tile_deps.csv.",
    )
    args = parser.parse_args()
    args.split_attention = args.attention_path == "split"
    args.tma_attention = args.attention_path == "tma"
    if args.two_page_attention:
        args.three_page_attention = False
    if not args.packed_qk:
        args.packed_qkv_projection = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.scheduler == "autotuned":
        if args.autotune_summary is None:
            raise ValueError("--scheduler autotuned requires --autotune-summary")
        profile = (
            TimingProfile.from_perfetto(args.timing_profile)
            if args.timing_profile is not None
            else None
        )
        scheduler = scheduler_from_autotune_summary(
            args.autotune_summary,
            timing_profile=profile,
        )
    elif args.scheduler == "timing":
        if args.timing_profile is None:
            raise ValueError("--scheduler timing requires --timing-profile")
        scheduler = TimingAwareOverlapScheduler(
            timing_profile=TimingProfile.from_perfetto(args.timing_profile),
            timing_mode=args.timing_mode,
            timing_position=args.timing_position,
            fetch_stride=args.fetch_stride if args.fetch_stride > 0 else None,
            dependency_slack_waves=args.dependency_slack_waves,
            dependency_slack_op_indices=set(args.dependency_slack_op_idx) if args.dependency_slack_op_idx else None,
        )
    elif args.scheduler == "overlap":
        scheduler = OverlapTileScheduler(
            fetch_stride=args.fetch_stride if args.fetch_stride > 0 else None,
            dependency_slack_waves=args.dependency_slack_waves,
            dependency_slack_op_indices=set(args.dependency_slack_op_idx) if args.dependency_slack_op_idx else None,
        )
    elif args.scheduler == "qwen-forward":
        if args.mode != "fwd":
            raise ValueError("--scheduler qwen-forward is only valid for --mode fwd")
        scheduler = Qwen3_5ForwardOverlapScheduler(
            fetch_stride=args.fetch_stride if args.fetch_stride > 0 else None,
            dependency_slack_waves=args.dependency_slack_waves,
            dependency_slack_op_indices=set(args.dependency_slack_op_idx) if args.dependency_slack_op_idx else None,
        )
    elif args.scheduler == "qwen-causal-rank":
        if args.mode != "fwd":
            raise ValueError("--scheduler qwen-causal-rank is only valid for --mode fwd")
        from machete.kernels.qwen_3_5.qwen_3_5_forward import Qwen3_5ForwardCausalRankScheduler

        scheduler = Qwen3_5ForwardCausalRankScheduler(
            fetch_stride=args.fetch_stride if args.fetch_stride > 0 else None,
            dependency_slack_waves=args.dependency_slack_waves,
            dependency_slack_op_indices=set(args.dependency_slack_op_idx) if args.dependency_slack_op_idx else None,
        )
    else:
        scheduler = None
    qwen_args = _alloc_qwen_layer(args.batch, args.seq_len)
    build = megakernel_forward_build if args.mode == "fwd" else megakernel_layer_bwd_build

    with contextlib.redirect_stdout(io.StringIO()):
        result = build(
            *qwen_args,
            page_size=args.page_size,
            scheduler=scheduler,
            tracing=True,
            **({"rms_tile_s": args.rms_tile_s} if args.mode == "fwd" and args.rms_tile_s is not None else {}),
            **({"adaptive_gemm_tiling": args.adaptive_gemm_tiling} if args.mode == "fwd" else {}),
            **({"use_packed_qk": args.packed_qk} if args.mode == "fwd" else {}),
            **({"use_qwen_projection": args.qwen_projection} if args.mode == "fwd" else {}),
            **({"use_packed_qkv_projection": args.packed_qkv_projection} if args.mode == "fwd" else {}),
            **({"use_fused_rms_proj": args.fused_rms_proj} if args.mode == "fwd" else {}),
            **({"use_qknorm_4d": args.qknorm_4d} if args.mode == "fwd" else {}),
            **({"use_tma_attention": args.tma_attention} if args.mode == "fwd" else {}),
            **({"use_two_page_attention": args.two_page_attention} if args.mode == "fwd" else {}),
            **({"use_three_page_attention": args.three_page_attention} if args.mode == "fwd" else {}),
            **({"use_split_attention": args.split_attention} if args.mode == "fwd" else {}),
            **({"split_attention_splits": args.split_attention_splits} if args.mode == "fwd" else {}),
            **({"attention_tile_m": args.attention_tile_m} if args.mode == "fwd" else {}),
            **(
                {"forward_op_limit": args.forward_op_limit}
                if args.mode == "fwd" and args.forward_op_limit > 0
                else {}
            ),
            **(
                {"attention_bwd_batch_window": args.attention_bwd_batch_window}
                if args.mode == "bwd"
                else {}
            ),
            **(
                {
                    "projection_bwd_input_tile_s": (
                        None
                        if args.projection_bwd_input_tile_s == 0
                        else args.projection_bwd_input_tile_s
                    )
                }
                if args.mode == "bwd"
                else {}
            ),
            **(
                {
                    "projection_bwd_reduce_tile_n": (
                        None
                        if args.projection_bwd_reduce_tile_n == 0
                        else args.projection_bwd_reduce_tile_n
                    )
                }
                if args.mode == "bwd"
                else {}
            ),
        )
    spec = result[0]
    kernel = next((obj for obj in spec._keep_alive if hasattr(obj, "write_trace_perfetto")), None)
    if kernel is None:
        raise RuntimeError("could not find megakernel object in benchmark spec")

    if spec.setup_fn is not None:
        spec.setup_fn()
    spec.launch_fn()
    torch.cuda.synchronize()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    kernel.write_trace(args.output)
    print(f"wrote {args.output}")

    perfetto_output = args.perfetto_output
    if perfetto_output is None and args.trace_dir:
        perfetto_output = str(Path(args.trace_dir) / _default_perfetto_name(args))
    if perfetto_output:
        os.makedirs(os.path.dirname(perfetto_output) or ".", exist_ok=True)
        kernel.write_trace_perfetto(perfetto_output)
        print(f"wrote {perfetto_output}")

    if args.deps_output_prefix:
        op_deps = f"{args.deps_output_prefix}.op_deps.csv"
        tile_deps = f"{args.deps_output_prefix}.tile_deps.csv"
        kernel.write_dependency_graph_csv(op_deps, tile_deps)
        print(f"wrote {op_deps}")
        print(f"wrote {tile_deps}")


if __name__ == "__main__":
    main()
