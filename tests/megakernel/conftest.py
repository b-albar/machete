# Copyright (c) 2025, Machete Authors
"""
Pytest fixtures for megakernel tests.

Provides common fixtures for CUDA testing and nanotrace export.
"""

import os
import pytest
from pathlib import Path

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False
    torch = None


@pytest.fixture
def cuda_device():
    """Fixture providing a CUDA device for tests."""
    if not HAS_CUDA:
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def trace_output_dir(request, tmp_path) -> Path:
    """Fixture providing a directory for trace output.

    By default, uses a temporary directory. Set MACHETE_TRACE_DIR environment
    variable to persist traces to a specific directory for debugging.

    The directory structure is:
        <trace_dir>/<test_module>/<test_name>.nanotrace
    """
    env_dir = os.environ.get("MACHETE_TRACE_DIR")
    if env_dir:
        base_dir = Path(env_dir)
    else:
        base_dir = tmp_path

    # Create subdirectory based on test module name
    module_name = request.module.__name__.split(".")[-1]
    trace_dir = base_dir / module_name
    trace_dir.mkdir(parents=True, exist_ok=True)

    return trace_dir


@pytest.fixture
def trace_file(request, trace_output_dir) -> str:
    """Fixture providing a trace file path for the current test.

    Returns a path like: <trace_dir>/<test_name>.nanotrace

    Usage:
        def test_my_kernel(cuda_device, trace_file):
            mk = Megakernel(name="test")
            mk.add(...)
            mk.launch_logical(block=(...), trace_file=trace_file)
    """
    test_name = request.node.name
    # Sanitize test name for filesystem
    safe_name = test_name.replace("[", "_").replace("]", "_").replace("/", "_")
    return str(trace_output_dir / f"{safe_name}.nanotrace")


@pytest.fixture
def enable_tracing(request) -> bool:
    """Fixture to check if tracing should be enabled.

    Tracing is enabled when:
    - MACHETE_TRACE_DIR environment variable is set, OR
    - --trace-kernels pytest option is passed

    Usage:
        def test_my_kernel(cuda_device, trace_file, enable_tracing):
            mk = Megakernel(name="test")
            mk.add(...)
            trace_path = trace_file if enable_tracing else None
            mk.launch_logical(block=(...), trace_file=trace_path)
    """
    # Check environment variable
    if os.environ.get("MACHETE_TRACE_DIR"):
        return True

    # Check pytest marker
    marker = request.node.get_closest_marker("trace")
    if marker is not None:
        return True

    # Check if --trace-kernels was passed (requires conftest.py hook)
    return getattr(request.config, "_trace_kernels", False)


def pytest_addoption(parser):
    """Add custom pytest options for tracing."""
    parser.addoption(
        "--trace-kernels",
        action="store_true",
        default=False,
        help="Enable nanotrace export for all kernel tests",
    )
    parser.addoption(
        "--trace-dir",
        action="store",
        default=None,
        help="Directory to store nanotrace files (overrides MACHETE_TRACE_DIR)",
    )


def pytest_configure(config):
    """Configure pytest based on command line options."""
    config._trace_kernels = config.getoption("--trace-kernels")

    # Override env var with command line option
    trace_dir = config.getoption("--trace-dir")
    if trace_dir:
        os.environ["MACHETE_TRACE_DIR"] = trace_dir

    # Register custom markers
    config.addinivalue_line(
        "markers", "trace: mark test to always export nanotrace file"
    )
