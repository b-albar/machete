# Copyright (c) 2025, Machete Authors
"""Shared test-environment helpers."""

from __future__ import annotations

import torch


def is_cuda_available() -> bool:
    """Whether CUDA is available for tests."""
    return torch.cuda.is_available()


def is_sm90_available() -> bool:
    """Whether an SM90+ GPU is available."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def is_hopper_available() -> bool:
    """Alias for older Hopper naming in tests."""
    return is_sm90_available()
