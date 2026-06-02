# Copyright (c) 2025, Machete Authors
"""Helpers for constructing schedulers from JSON-compatible configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .scheduling import OverlapTileScheduler, TimingAwareOverlapScheduler, TileScheduler
from .timing_profile import TimingProfile


def _normalize_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(kwargs)
    if "dependency_slack_op_indices" in normalized:
        normalized["dependency_slack_op_indices"] = {
            int(idx) for idx in normalized["dependency_slack_op_indices"]
        }
    return normalized


def scheduler_from_config(
    config: Mapping[str, Any] | None,
    *,
    timing_profile: TimingProfile | None = None,
) -> TileScheduler | None:
    """Construct a scheduler from a JSON-compatible candidate config.

    ``config`` is expected to look like the candidate object written by
    ``scripts/autotune_qwen_scheduler.py``:

    ``{"kind": "overlap", "kwargs": {...}}`` or
    ``{"kind": "timing", "kwargs": {...}}``.
    """
    if config is None:
        return None
    kind = config.get("kind", "overlap")
    kwargs = _normalize_kwargs(config.get("kwargs", {}))
    if kind == "default":
        return None
    if kind == "overlap":
        return OverlapTileScheduler(**kwargs)
    if kind == "timing":
        if timing_profile is None:
            raise ValueError("timing scheduler config requires a TimingProfile")
        return TimingAwareOverlapScheduler(
            timing_profile=timing_profile,
            **kwargs,
        )
    raise ValueError(f"unknown scheduler kind: {kind}")


def scheduler_from_autotune_summary(
    path: str | Path,
    *,
    timing_profile: TimingProfile | None = None,
) -> TileScheduler | None:
    """Construct the best scheduler from an autotune summary JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    best = summary.get("best") or {}
    return scheduler_from_config(
        best.get("candidate"),
        timing_profile=timing_profile,
    )


__all__ = ["scheduler_from_config", "scheduler_from_autotune_summary"]
