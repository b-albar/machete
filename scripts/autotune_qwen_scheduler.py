#!/usr/bin/env python
"""Autotune Qwen layer scheduler choices by measuring real runtimes."""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
from pathlib import Path
import sys
import time
from typing import Any

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
from machete.megakernel import TimingProfile
from machete.megakernel.scheduler_config import scheduler_from_config


def _trace_summary(path: Path) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        events = json.load(f)
    spans = [e for e in events if e.get("ph") == "X"]
    start = min(e["ts"] for e in spans)
    end = max(e["ts"] + e["dur"] for e in spans)
    totals = {
        "makespan_us": end - start,
        "data_wait_us": 0.0,
        "dep_wait_us": 0.0,
        "compute_wait_us": 0.0,
        "ring_full_wait_us": 0.0,
    }
    for event in spans:
        name = event.get("name")
        if name == "data wait":
            totals["data_wait_us"] += float(event["dur"])
        elif name == "dep wait":
            totals["dep_wait_us"] += float(event["dur"])
        elif name == "compute wait":
            totals["compute_wait_us"] += float(event["dur"])
        elif name == "ring full wait":
            totals["ring_full_wait_us"] += float(event["dur"])
    return totals


def _runtime_repeats(spec, *, warmup: int, repeats: int) -> list[dict[str, float]]:
    """Measure a benchmark spec with synchronized wall-clock timings."""
    if spec.setup_fn is not None:
        spec.setup_fn()
    for _ in range(max(0, warmup)):
        spec.launch_fn()
    torch.cuda.synchronize()

    timings = []
    for _ in range(max(1, repeats)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        spec.launch_fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append({"runtime_us": float((end - start) * 1_000_000.0)})
    return timings


def _scheduler_from_candidate(candidate: dict[str, Any], profile: TimingProfile | None):
    return scheduler_from_config(candidate, timing_profile=profile)


def _candidates(
    num_sms: int,
    include_timing: bool,
    fetch_stride_candidates: list[int] | None = None,
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [
        {"name": "overlap", "kind": "overlap", "kwargs": {}},
        {"name": "adaptive", "kind": "overlap", "kwargs": {"adaptive_fetch_stride": True}},
        {"name": "fetch_half", "kind": "overlap", "kwargs": {"fetch_stride": max(1, num_sms // 2)}},
        {"name": "fetch_quarter", "kind": "overlap", "kwargs": {"fetch_stride": max(1, num_sms // 4)}},
        {"name": "fetch_eighth", "kind": "overlap", "kwargs": {"fetch_stride": max(1, num_sms // 8)}},
        {"name": "fetch_2x", "kind": "overlap", "kwargs": {"fetch_stride": 2 * num_sms}},
        {
            "name": "slack_op19",
            "kind": "overlap",
            "kwargs": {"dependency_slack_waves": 1, "dependency_slack_op_indices": [19]},
        },
    ]
    if include_timing:
        configs.extend(
            [
                {
                    "name": "timing_critical_late",
                    "kind": "timing",
                    "kwargs": {"timing_mode": "critical", "timing_position": "late"},
                },
                {
                    "name": "timing_critical_before_op",
                    "kind": "timing",
                    "kwargs": {"timing_mode": "critical", "timing_position": "before_op"},
                },
                {
                    "name": "timing_avoid_data_late",
                    "kind": "timing",
                    "kwargs": {"timing_mode": "avoid_data_wait", "timing_position": "late"},
                },
            ]
        )
    for stride in fetch_stride_candidates or []:
        stride = max(1, int(stride))
        configs.append(
            {
                "name": f"fetch_{stride}",
                "kind": "overlap",
                "kwargs": {"fetch_stride": stride},
            }
        )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fwd", "bwd"], default="bwd")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=32768)
    qk_group = parser.add_mutually_exclusive_group()
    qk_group.add_argument("--packed-qk", dest="packed_qk", action="store_true", default=True)
    qk_group.add_argument("--separate-qk", dest="packed_qk", action="store_false")
    parser.add_argument("--num-sms", type=int, default=70)
    parser.add_argument(
        "--fetch-stride-candidate",
        action="append",
        type=int,
        default=None,
        help="Additional explicit overlap fetch stride to benchmark. May be repeated.",
    )
    parser.add_argument("--timing-profile", default=None)
    parser.add_argument("--out-dir", default="/tmp/machete_scheduler_autotune")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measure", choices=["runtime", "trace"], default="runtime")
    parser.add_argument(
        "--write-best-trace",
        action="store_true",
        help="After event autotune, emit a Perfetto trace for the best candidate.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help="Run only this candidate name. May be repeated.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build = megakernel_forward_build if args.mode == "fwd" else megakernel_layer_bwd_build

    profile = TimingProfile.from_perfetto(args.timing_profile) if args.timing_profile else None
    results = []
    candidates = _candidates(
        args.num_sms,
        include_timing=profile is not None,
        fetch_stride_candidates=args.fetch_stride_candidate,
    )
    if args.candidate:
        wanted = set(args.candidate)
        candidates = [candidate for candidate in candidates if candidate["name"] in wanted]
        missing = wanted - {candidate["name"] for candidate in candidates}
        if missing:
            raise ValueError(f"unknown candidate(s): {sorted(missing)}")

    for candidate in candidates:
        print(f"run {candidate['name']}", flush=True)
        spec = kernel = result = scheduler = qwen_args = None
        try:
            gc.collect()
            torch.cuda.empty_cache()
            qwen_args = _alloc_layer(args.batch, args.seq_len)
            scheduler = _scheduler_from_candidate(candidate, profile)
            with contextlib.redirect_stdout(io.StringIO()):
                result = build(
                    *qwen_args,
                    page_size=args.page_size,
                    scheduler=scheduler,
                    tracing=args.measure == "trace",
                    **({"use_packed_qk": args.packed_qk} if args.mode == "fwd" else {}),
                )
            spec = result[0]
            if args.measure == "trace":
                kernel = next(
                    obj for obj in spec._keep_alive if hasattr(obj, "write_trace_perfetto")
                )
                repeat_summaries = []
                trace_path = None
                for repeat in range(max(1, args.repeats)):
                    if spec.setup_fn is not None:
                        spec.setup_fn()
                    spec.launch_fn()
                    torch.cuda.synchronize()

                    suffix = (
                        candidate["name"]
                        if args.repeats <= 1
                        else f"{candidate['name']}_r{repeat}"
                    )
                    trace_path = out_dir / f"{args.mode}_b{args.batch}_s{args.seq_len}_{suffix}.perfetto.json"
                    kernel.write_trace_perfetto(str(trace_path))
                    repeat_summaries.append(_trace_summary(trace_path))

                measurement_key = "makespan_us"
            else:
                repeat_summaries = _runtime_repeats(
                    spec,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                trace_path = None
                measurement_key = "runtime_us"

            repeat_summaries.sort(key=lambda item: item[measurement_key])
            best_repeat = repeat_summaries[0]
            median_repeat = repeat_summaries[len(repeat_summaries) // 2]
            summary = dict(best_repeat)
            selection_key = f"median_{measurement_key}"
            summary[selection_key] = median_repeat[measurement_key]
            summary["measurement_key"] = measurement_key
            summary["metric_key"] = selection_key
            summary["repeat_count"] = len(repeat_summaries)
            row = {
                "candidate": candidate,
                "trace": str(trace_path) if trace_path is not None else None,
                "repeats": repeat_summaries,
                **summary,
            }
            results.append(row)
            if measurement_key == "makespan_us":
                print(
                    f"  makespan={summary['makespan_us']:.3f}us "
                    f"median={summary['median_makespan_us']:.3f}us "
                    f"data_wait={summary['data_wait_us']:.1f}us "
                    f"dep_wait={summary['dep_wait_us']:.1f}us",
                    flush=True,
                )
            else:
                print(
                    f"  runtime={summary['runtime_us']:.3f}us "
                    f"median={summary['median_runtime_us']:.3f}us",
                    flush=True,
                )
        except Exception as exc:
            torch.cuda.synchronize()
            row = {
                "candidate": candidate,
                "error": f"{type(exc).__name__}: {exc}",
            }
            results.append(row)
            print(f"  failed: {row['error']}", flush=True)
        finally:
            del spec, kernel, result, scheduler, qwen_args
            gc.collect()
            torch.cuda.empty_cache()

    successful = [row for row in results if "metric_key" in row]
    if not successful:
        raise RuntimeError("all scheduler candidates failed")
    successful.sort(key=lambda row: row[row["metric_key"]])
    if args.write_best_trace and args.measure == "runtime":
        best_candidate = successful[0]["candidate"]
        gc.collect()
        torch.cuda.empty_cache()
        qwen_args = _alloc_layer(args.batch, args.seq_len)
        scheduler = _scheduler_from_candidate(best_candidate, profile)
        with contextlib.redirect_stdout(io.StringIO()):
            result = build(
                *qwen_args,
                page_size=args.page_size,
                scheduler=scheduler,
                tracing=True,
                **({"use_packed_qk": args.packed_qk} if args.mode == "fwd" else {}),
            )
        spec = result[0]
        kernel = next(obj for obj in spec._keep_alive if hasattr(obj, "write_trace_perfetto"))
        if spec.setup_fn is not None:
            spec.setup_fn()
        spec.launch_fn()
        torch.cuda.synchronize()
        trace_path = out_dir / f"{args.mode}_b{args.batch}_s{args.seq_len}_best.perfetto.json"
        kernel.write_trace_perfetto(str(trace_path))
        successful[0]["trace"] = str(trace_path)

    summary_path = out_dir / f"{args.mode}_b{args.batch}_s{args.seq_len}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"best": successful[0], "results": results}, f, indent=2)
    metric_key = successful[0]["metric_key"]
    print(
        f"best={successful[0]['candidate']['name']} "
        f"{metric_key}={successful[0][metric_key]:.3f}us"
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
