# Copyright (c) 2025, Machete Authors
"""Shared helpers for megakernel framework tests."""

from __future__ import annotations


def get_nop_op():
    """Return a test-only no-op op without importing CUTLASS at module import time."""
    from machete.megakernel.ops import Op

    class _NOPOp(Op):
        pass

    return _NOPOp
