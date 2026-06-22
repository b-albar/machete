#!/usr/bin/env python
# Copyright (c) 2026, Machete Authors
"""Run Nsight Compute comparisons for the L2 locality probe."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


METRICS = [
    "lts__t_sector_hit_rate.pct",
    "lts__t_bytes.sum",
    "dram__bytes.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
]
LONG_SCOREBOARD_PCT = "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"


CASES = [
    ("baseline", "chunk"),
    ("global", "chunk"),
    ("domain", "chunk"),
    ("domain", "random"),
]


def _parse_metric_value(raw: str) -> float | None:
    raw = raw.strip().replace(",", "")
    if raw == "n/a" or not raw:
        return None
    return float(raw)


def _parse_csv(path: Path) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(line for line in f if line.startswith('"')):
            if "locality_stream_kernel" not in row["Kernel Name"]:
                continue
            metric = row["Metric Name"]
            if metric in METRICS:
                values[metric] = _parse_metric_value(row["Metric Value"])
            elif metric == LONG_SCOREBOARD_PCT:
                values[LONG_SCOREBOARD_PCT] = _parse_metric_value(row["Metric Value"])
    return values


def _run_case(args: argparse.Namespace, mode: str, owner: str, out_dir: Path) -> dict[str, float | None]:
    csv_path = out_dir / f"ncu_l2_locality_{mode}_{owner}.csv"
    cmd = [
        "ncu",
        "--csv",
        "--target-processes",
        "all",
        "--kernel-name",
        "regex:locality_stream_kernel",
        "--log-file",
        str(csv_path),
        "--metrics",
        ",".join(METRICS),
        "python",
        "benchmarks/kernels/benchmark_l2_locality.py",
        "--mode",
        mode,
        "--owner",
        owner,
        "--chunks",
        str(args.chunks),
        "--chunk-mb",
        str(args.chunk_mb),
        "--repeats",
        str(args.repeats),
        "--inner-iters",
        str(args.inner_iters),
        "--num-domains",
        str(args.num_domains),
        "--num-blocks",
        str(args.num_blocks),
        "--threads",
        str(args.threads),
        "--warmup",
        "0",
        "--iters",
        "1",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return _parse_csv(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/tmp/machete_l2_locality_ncu")
    parser.add_argument("--num-domains", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=0)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--chunks", type=int, default=128)
    parser.add_argument("--chunk-mb", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--inner-iters", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode, owner in CASES:
        metrics = _run_case(args, mode, owner, out_dir)
        l2 = metrics.get("lts__t_bytes.sum")
        dram = metrics.get("dram__bytes.sum")
        ratio = None if l2 is None or dram in (None, 0.0) else l2 / dram
        rows.append((mode, owner, metrics, ratio))

    print("\ncase,l2_hit_pct,l2_bytes,dram_bytes,l2_dram_ratio,dram_pct,sm_pct,long_scoreboard")
    for mode, owner, metrics, ratio in rows:
        print(
            ",".join(
                [
                    f"{mode}:{owner}",
                    str(metrics.get("lts__t_sector_hit_rate.pct")),
                    str(metrics.get("lts__t_bytes.sum")),
                    str(metrics.get("dram__bytes.sum")),
                    str(ratio),
                    str(metrics.get("dram__throughput.avg.pct_of_peak_sustained_elapsed")),
                    str(metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")),
                    str(metrics.get(LONG_SCOREBOARD_PCT)),
                ]
            )
        )
    print(f"\nCSV files: {out_dir}")


if __name__ == "__main__":
    main()
