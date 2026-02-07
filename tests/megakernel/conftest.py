# Copyright (c) 2025, Machete Authors
"""Shared fixtures for megakernel tests."""

import pytest
import torch


@pytest.fixture(autouse=True)
def cuda_cleanup():
    """Sync CUDA and clear kernel cache between tests.

    Persistent megakernels run until all instructions are processed. If a
    previous test was interrupted (Ctrl+C, timeout), the old kernel may still
    be occupying the GPU. Synchronizing before launch ensures the device is
    idle, and synchronizing after ensures completion before the next test.

    Clearing the kernel cache prevents stale compiled kernels from being
    reused across tests with different configurations.
    """
    from machete.megakernel import Megakernel

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    Megakernel._compiled_kernel_cache.clear()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
