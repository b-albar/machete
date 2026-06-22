#!/usr/bin/env python
"""Summarize per-op timings from a megakernel Perfetto trace."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from machete.megakernel import TimingProfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("perfetto_json")
    parser.add_argument("--top", type=int, default=16)
    parser.add_argument(
        "--sort",
        choices=["total", "compute", "data_wait", "dep_wait", "compute_wait", "ring_full"],
        default="total",
    )
    args = parser.parse_args()

    profile = TimingProfile.from_perfetto(args.perfetto_json)
    key_by_sort = {
        "total": lambda item: item[1].total_us,
        "compute": lambda item: item[1].compute_us,
        "data_wait": lambda item: item[1].data_wait_us,
        "dep_wait": lambda item: item[1].dep_wait_us,
        "compute_wait": lambda item: item[1].compute_wait_us,
        "ring_full": lambda item: item[1].ring_full_wait_us,
    }
    rows = sorted(profile.ops.items(), key=key_by_sort[args.sort], reverse=True)
    print(
        "op_idx,total_us,compute_us,data_wait_us,dep_wait_us,"
        "compute_wait_us,ring_full_wait_us,event_count"
    )
    for op_idx, timing in rows[: args.top]:
        print(
            f"{op_idx},{timing.total_us:.3f},{timing.compute_us:.3f},"
            f"{timing.data_wait_us:.3f},{timing.dep_wait_us:.3f},"
            f"{timing.compute_wait_us:.3f},{timing.ring_full_wait_us:.3f},"
            f"{timing.event_count}"
        )


if __name__ == "__main__":
    main()
