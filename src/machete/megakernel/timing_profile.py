# Copyright (c) 2025, Machete Authors
"""Timing profiles derived from megakernel Perfetto traces."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Mapping


@dataclass(frozen=True)
class OpTiming:
    """Aggregated timing for one scheduled op index."""

    compute_us: float = 0.0
    data_wait_us: float = 0.0
    dep_wait_us: float = 0.0
    compute_wait_us: float = 0.0
    ring_full_wait_us: float = 0.0
    event_count: int = 0

    @property
    def total_wait_us(self) -> float:
        return (
            self.data_wait_us
            + self.dep_wait_us
            + self.compute_wait_us
            + self.ring_full_wait_us
        )

    @property
    def total_us(self) -> float:
        return self.compute_us + self.total_wait_us


@dataclass(frozen=True)
class TimingProfile:
    """Per-op timing scores used by host-side tile schedulers.

    The profile is intentionally compact: it keeps totals by ``op_idx`` from a
    previous trace.  Schedulers can use these as priority hints, while the
    dependency formulas still enforce correctness.
    """

    ops: Mapping[int, OpTiming] = field(default_factory=dict)

    @classmethod
    def from_perfetto(cls, path: str | Path) -> "TimingProfile":
        """Build a profile from a Perfetto JSON trace."""
        with open(path, "r", encoding="utf-8") as f:
            events = json.load(f)

        mutable: Dict[int, Dict[str, float]] = {}
        for event in events:
            if event.get("ph") != "X":
                continue
            args = event.get("args") or {}
            op_idx = args.get("param0")
            if op_idx is None:
                continue
            op_idx = int(op_idx)
            row = mutable.setdefault(
                op_idx,
                {
                    "compute_us": 0.0,
                    "data_wait_us": 0.0,
                    "dep_wait_us": 0.0,
                    "compute_wait_us": 0.0,
                    "ring_full_wait_us": 0.0,
                    "event_count": 0.0,
                },
            )
            name = event.get("name", "")
            dur = float(event.get("dur", 0.0))
            if name.endswith(" compute"):
                row["compute_us"] += dur
            elif name == "data wait":
                row["data_wait_us"] += dur
            elif name == "dep wait":
                row["dep_wait_us"] += dur
            elif name == "compute wait":
                row["compute_wait_us"] += dur
            elif name == "ring full wait":
                row["ring_full_wait_us"] += dur
            else:
                continue
            row["event_count"] += 1.0

        return cls(
            {
                op_idx: OpTiming(
                    compute_us=row["compute_us"],
                    data_wait_us=row["data_wait_us"],
                    dep_wait_us=row["dep_wait_us"],
                    compute_wait_us=row["compute_wait_us"],
                    ring_full_wait_us=row["ring_full_wait_us"],
                    event_count=int(row["event_count"]),
                )
                for op_idx, row in mutable.items()
            }
        )

    def score(self, op_idx: int, mode: str = "critical") -> int:
        """Return an integer priority score for ``op_idx``.

        Modes:
        - ``critical``: prioritize high total observed time.
        - ``compute``: prioritize high compute time.
        - ``data_wait``: prioritize ops that previously suffered data waits.
        - ``short``: prioritize short ops first by negating total time.
        - ``avoid_data_wait``: de-prioritize ops with high observed data wait.
        """
        timing = self.ops.get(int(op_idx))
        if timing is None:
            return 0
        if mode == "compute":
            value = timing.compute_us
        elif mode == "data_wait":
            value = timing.data_wait_us
        elif mode == "short":
            value = -timing.total_us
        elif mode == "avoid_data_wait":
            value = -timing.data_wait_us
        else:
            value = timing.total_us
        # Keep tuple comparisons cheap and stable; microsecond resolution is
        # enough for the coarse host-side priority choice.
        return int(round(value))


__all__ = ["OpTiming", "TimingProfile"]
