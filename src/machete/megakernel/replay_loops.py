# Copyright (c) 2026, Machete Authors
"""Compatibility exports for persistent megakernel replay loops."""

from .compute_only_replay import build_compute_only_kernel_loop
from .full_replay import build_ring_kernel_loop

__all__ = ["build_compute_only_kernel_loop", "build_ring_kernel_loop"]
