# Copyright (c) 2025, Machete Authors
"""Small helpers for declaring persistent regions."""

from __future__ import annotations

from typing import Iterable

from .ops import PersistentRegion, PipelineSpec, ScheduledOp


def region(
    name: str,
    ops: Iterable[ScheduledOp],
    *,
    pipeline: PipelineSpec | None = None,
    generated_op: ScheduledOp | None = None,
) -> PersistentRegion:
    """Declare one persistent region.

    ``ops`` are the semantic schedule for dependencies and validation.  If
    ``generated_op`` is provided, the region lowers to that single specialized
    op; otherwise it replays the listed ops while preserving the region boundary.
    """
    return PersistentRegion.from_ops(
        name,
        tuple(ops),
        pipeline=pipeline,
        lowering=(
            PersistentRegion.LOWER_HANDLER
            if generated_op is not None
            else PersistentRegion.LOWER_REPLAY
        ),
        generated_op=generated_op,
    )


__all__ = [
    "region",
]
