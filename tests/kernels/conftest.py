# Copyright (c) 2025, Machete Authors
"""Shared fixtures for kernel tests."""

import pytest
import torch


@pytest.fixture(autouse=True)
def cuda_cleanup():
    """Sync CUDA and clear kernel cache between tests.

    TMA descriptors are baked into compiled kernels by cute.compile().
    Clearing the cache ensures fresh TMA descriptors are created when
    tensors are reallocated between tests.
    """
    from machete.megakernel import Megakernel

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    Megakernel._compiled_kernel_cache.clear()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
