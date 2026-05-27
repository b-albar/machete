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
from machete.megakernel import OverlapTileScheduler


def _alloc_qwen_layer(batch: int, seq_len: int):
    return _alloc_layer(batch, seq_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fwd", "bwd"], default="fwd")
    parser.add_argument("--scheduler", choices=["default", "overlap"], default="overlap")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=32768)
    parser.add_argument("--rms-tile-s", type=int, default=None)
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
        "--dependency-slack-op",
        action="append",
        default=None,
        help="Apply dependency slack only to this op class name. May be repeated.",
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
        "--deps-output-prefix",
        default=None,
        help="Write static dependency graph CSVs as PREFIX.op_deps.csv and PREFIX.tile_deps.csv.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    scheduler = (
        OverlapTileScheduler(
            dependency_slack_waves=args.dependency_slack_waves,
            dependency_slack_op_names=set(args.dependency_slack_op) if args.dependency_slack_op else None,
            dependency_slack_op_indices=set(args.dependency_slack_op_idx) if args.dependency_slack_op_idx else None,
            use_controller_waits_for_readiness=True,
        )
        if args.scheduler == "overlap"
        else None
    )
    qwen_args = _alloc_qwen_layer(args.batch, args.seq_len)
    build = megakernel_forward_build if args.mode == "fwd" else megakernel_layer_bwd_build

    with contextlib.redirect_stdout(io.StringIO()):
        result = build(
            *qwen_args,
            page_size=args.page_size,
            scheduler=scheduler,
            tracing=True,
            **({"rms_tile_s": args.rms_tile_s} if args.mode == "fwd" and args.rms_tile_s is not None else {}),
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

    if args.perfetto_output:
        os.makedirs(os.path.dirname(args.perfetto_output) or ".", exist_ok=True)
        kernel.write_trace_perfetto(args.perfetto_output)
        print(f"wrote {args.perfetto_output}")

    if args.deps_output_prefix:
        op_deps = f"{args.deps_output_prefix}.op_deps.csv"
        tile_deps = f"{args.deps_output_prefix}.tile_deps.csv"
        kernel.write_dependency_graph_csv(op_deps, tile_deps)
        print(f"wrote {op_deps}")
        print(f"wrote {tile_deps}")


if __name__ == "__main__":
    main()
