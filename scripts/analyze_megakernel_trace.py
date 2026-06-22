#!/usr/bin/env python3
"""Summarize Machete cutedsl-trace Perfetto JSON exports."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _kind(name: str) -> str:
    if "compute" in name:
        return "compute"
    if "load" in name:
        return "load"
    if "store" in name:
        return "store"
    if "wait" in name:
        return "wait"
    return "other"


def _op_name(name: str) -> str:
    for suffix in (" compute", " load", " store"):
        name = name.replace(suffix, "")
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, help="Perfetto JSON trace written by Megakernel.write_trace_perfetto")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--gap-threshold-us", type=float, default=5.0)
    args = parser.parse_args()

    data = json.loads(args.trace.read_text())
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    spans = [e for e in events if e.get("ph") == "X"]
    if not spans:
        raise SystemExit("no span events found")

    t0 = min(e["ts"] for e in spans)
    t1 = max(e["ts"] + e["dur"] for e in spans)
    makespan = t1 - t0
    print(f"trace={args.trace}")
    print(f"spans={len(spans)} makespan_us={makespan:.3f}")

    by_lane: dict[int | str, list[dict]] = defaultdict(list)
    by_op_kind: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_name: dict[str, list[dict]] = defaultdict(list)
    by_tid: dict[int, list[dict]] = defaultdict(list)
    for event in spans:
        lane = event.get("args", {}).get("lane_id", "?")
        kind = _kind(event["name"])
        by_lane[lane].append(event)
        by_op_kind[(_op_name(event["name"]), kind)].append(event)
        by_name[event["name"]].append(event)
        by_tid[event["tid"]].append(event)

    print("\nlanes:")
    for lane, lane_events in sorted(by_lane.items(), key=lambda item: str(item[0])):
        total = sum(e["dur"] for e in lane_events)
        counts = Counter(_kind(e["name"]) for e in lane_events)
        print(
            f"  lane={lane} events={len(lane_events)} total_us={total:.1f} "
            f"util_vs_makespan={total / makespan:.3f} {dict(counts)}"
        )

    print("\ntop op/kind totals:")
    rows = []
    for (op, kind), op_events in by_op_kind.items():
        durs = [e["dur"] for e in op_events]
        rows.append((sum(durs), len(durs), op, kind, statistics.mean(durs), max(durs)))
    for total, count, op, kind, mean, max_dur in sorted(rows, reverse=True)[: args.top]:
        print(f"  {kind:7} {op:48} total_us={total:9.1f} n={count:5d} mean={mean:7.3f} max={max_dur:7.3f}")

    wait_rows = [
        (sum(e["dur"] for e in events_for_name), len(events_for_name), name, max(e["dur"] for e in events_for_name))
        for name, events_for_name in by_name.items()
        if "wait" in name
    ]
    print("\nwait totals:")
    if wait_rows:
        for total, count, name, max_dur in sorted(wait_rows, reverse=True):
            print(f"  {name:48} total_us={total:9.1f} n={count:5d} max={max_dur:7.3f}")
    else:
        print("  none")

    gaps = []
    gap_after = Counter()
    gap_before = Counter()
    big_gaps = []
    for tid, tid_events in by_tid.items():
        tid_events.sort(key=lambda e: e["ts"])
        for prev, nxt in zip(tid_events, tid_events[1:]):
            gap = nxt["ts"] - (prev["ts"] + prev["dur"])
            if gap <= 0:
                continue
            gaps.append(gap)
            prev_op = _op_name(prev["name"])
            next_op = _op_name(nxt["name"])
            gap_after[prev_op] += gap
            gap_before[next_op] += gap
            if gap >= args.gap_threshold_us:
                big_gaps.append((gap, tid, prev_op, next_op, prev["ts"] + prev["dur"]))

    print("\nper-track idle gaps:")
    if gaps:
        gaps_sorted = sorted(gaps)
        p95 = gaps_sorted[int(0.95 * (len(gaps_sorted) - 1))]
        print(
            f"  count={len(gaps)} total_us={sum(gaps):.1f} mean={statistics.mean(gaps):.3f} "
            f"p95={p95:.3f} max={max(gaps):.3f}"
        )
        print("  top gap after:")
        for op, total in gap_after.most_common(args.top):
            print(f"    {op:48} total_us={total:9.1f}")
        print("  top gap before:")
        for op, total in gap_before.most_common(args.top):
            print(f"    {op:48} total_us={total:9.1f}")
        print(f"  largest gaps >= {args.gap_threshold_us:g} us:")
        for gap, tid, prev_op, next_op, end_ts in sorted(big_gaps, reverse=True)[: args.top]:
            print(f"    gap={gap:8.3f} tid={tid:4d} after={prev_op} before={next_op} prev_end_ts={end_ts:.3f}")
    else:
        print("  none")


if __name__ == "__main__":
    main()
