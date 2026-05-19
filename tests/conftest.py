# Copyright (c) 2025, Machete Authors
"""Shared pytest configuration for the maintained test suites."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
for path in (_REPO_ROOT, _SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    import cutlass  # noqa: F401

    _CUTLASS_AVAILABLE = True
except ImportError:
    _CUTLASS_AVAILABLE = False


def _suite_markers_for_path(path: Path) -> tuple[str, ...]:
    parts = path.parts
    if "kernels" in parts:
        markers = ["kernels", "gpu"]
        if path.name == "test_attention_sm_120.py":
            markers.append("arch_sm120")
        if path.name in {
            "test_activation.py",
            "test_glu.py",
            "test_rope.py",
        }:
            markers.append("smoke")
        if path.name in {
            "test_gemm.py",
            "test_rmsnorm.py",
            "test_attention_sm_120.py",
            "test_flash_decoding.py",
            "test_gated_delta_net.py",
            "test_moe_gemm.py",
            "test_moe_gemm_bwd.py",
            "test_rmsnorm_gemm.py",
        }:
            markers.append("integration")
        if path.name in {
            "test_gemm.py",
            "test_rmsnorm.py",
            "test_attention_sm_120.py",
            "test_gated_delta_net.py",
            "test_gemm_kblock_multipass.py",
            "test_flash_decoding.py",
        }:
            markers.append("slow")
        return tuple(markers)
    if "megakernel" in parts:
        markers = ["megakernel", "gpu"]
        if "deps" in parts:
            markers.extend(["deps", "unit"])
            return tuple(markers)
        if path.name in {
            "test_ops.py",
            "test_autograd.py",
            "test_cute_tensor_params.py",
            "test_mul_add_smem_ops.py",
        }:
            markers.append("unit")
        if path.name in {
            "test_megakernel.py",
            "test_persistent_kernel.py",
            "test_tma_megakernel.py",
            "test_fused_rmsnorm_rope.py",
        }:
            markers.append("smoke")
        if path.name in {
            "test_communicate.py",
            "test_cluster_launch_control.py",
            "test_integration_gpu.py",
        }:
            markers.append("integration")
        if path.name in {
            "test_communicate.py",
            "test_cluster_launch_control.py",
            "test_integration_gpu.py",
        }:
            markers.append("slow")
        return tuple(markers)
    if "patching" in parts:
        markers = ["patching"]
        if path.name in {"test_linear.py", "test_rope.py", "test_attention.py"}:
            markers.append("unit")
        if path.name in {"test_qwen.py", "test_glm4.py", "test_llama.py"}:
            markers.append("integration")
        if path.name in {"test_qwen.py", "test_glm4.py"}:
            markers.append("slow")
        return tuple(markers)
    return ()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Attach stable suite markers based on the test file location."""
    for item in items:
        path = Path(str(item.fspath))
        for marker in _suite_markers_for_path(path):
            item.add_marker(getattr(pytest.mark, marker))


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    """Skip GPU framework suites entirely when optional deps are unavailable."""
    del config
    parts = collection_path.parts
    if _CUTLASS_AVAILABLE:
        return False
    return "tests" in parts and ("kernels" in parts or "megakernel" in parts)


@pytest.fixture(autouse=True)
def cuda_cleanup(request: pytest.FixtureRequest):
    """Synchronize CUDA and clear compiled-kernel cache for GPU suites only."""
    if not any(
        request.node.get_closest_marker(marker_name)
        for marker_name in ("kernels", "megakernel", "gpu")
    ):
        yield
        return

    from machete.megakernel import Megakernel

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    Megakernel._compiled_kernel_cache.clear()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
