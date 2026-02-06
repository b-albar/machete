# Copyright (c) 2025, Machete Authors
"""
Test: cute.Tensor as kernel parameters — parameter limit discovery.

Tests how many cute.Tensor objects can be passed as direct kernel parameters
before hitting the CUDA 4KB kernel parameter limit.

Also validates that cute.Tensor params work correctly for read/write access
inside @cute.kernel functions.
"""

import linecache
import textwrap

import pytest
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

from machete.utils.testing import is_hopper_available


# =============================================================================
# Kernel generators via exec() — dynamic number of cute.Tensor params
# =============================================================================

_gen_counter = 0


def _make_write_kernel_class(n_tensors, use_dynamic_layout=False):
    """Generate a kernel class that writes to N cute.Tensor parameters.

    Each tensor gets written with value (tensor_index + 1) * 100 at position 0.
    This validates that all tensor parameters are accessible inside the kernel.

    Args:
        n_tensors: Number of cute.Tensor parameters.
        use_dynamic_layout: If True, mark dim 0 as dynamic.

    Returns:
        (kernel_class_instance, make_tensors_fn)
    """
    global _gen_counter
    _gen_counter += 1

    # Build parameter lists
    # _comma variants have a leading ", " for appending after other params
    # _bare variant has no leading comma for standalone use
    tensor_params_call = ", ".join(f"t{i}: cute.Tensor" for i in range(n_tensors))
    tensor_params_kernel = ", ".join(f"t{i}: cute.Tensor" for i in range(n_tensors))
    tensor_args_bare = ", ".join(f"t{i}" for i in range(n_tensors))

    if tensor_params_call:
        tensor_params_call = ", " + tensor_params_call
        tensor_params_kernel = ", " + tensor_params_kernel
    tensor_args = (", " + tensor_args_bare) if tensor_args_bare else ""

    # Build write statements (thread 0 writes marker to each tensor)
    # 12 spaces = class body (4) + method body (4) + if body (4)
    write_lines = []
    for i in range(n_tensors):
        write_lines.append(f"            t{i}[Int32(0)] = Float32({(i + 1) * 100}.0)")
    write_body = "\n".join(write_lines) if write_lines else "            pass"

    source = (
        f"class WriteKernel_{_gen_counter}:\n"
        f"    @cute.jit\n"
        f"    def __call__(self, stream: cuda.CUstream{tensor_params_call}):\n"
        f"        self.kernel({tensor_args_bare}).launch(\n"
        f"            grid=[1, 1, 1],\n"
        f"            block=[1, 1, 1],\n"
        f"            stream=stream,\n"
        f"        )\n"
        f"\n"
        f"    @cute.kernel\n"
        f"    def kernel(self{tensor_params_kernel}):\n"
        f"        tidx = cute.arch.thread_idx()[0]\n"
        f"        if tidx == Int32(0):\n"
        f"{write_body}\n"
    )

    filename = f"<write_kernel_{_gen_counter}>"
    linecache.cache[filename] = (
        len(source), None, source.splitlines(True), filename,
    )

    exec_globals = {
        "cute": cute,
        "cuda": cuda,
        "Int32": Int32,
        "Float32": Float32,
    }
    code = compile(source, filename, "exec")
    exec(code, exec_globals)

    kernel_cls = exec_globals[f"WriteKernel_{_gen_counter}"]

    def make_tensors():
        tensors_torch = [
            torch.zeros(4, dtype=torch.float32, device="cuda")
            for _ in range(n_tensors)
        ]
        tensors_cute = []
        for t in tensors_torch:
            ct = from_dlpack(t, assumed_align=16)
            if use_dynamic_layout:
                ct = ct.mark_layout_dynamic()
            tensors_cute.append(ct)
        return tensors_torch, tensors_cute

    return kernel_cls(), make_tensors


def _make_write_kernel_class_2d(n_tensors):
    """Generate a kernel class with N 2D cute.Tensor parameters.

    Each tensor has shape (rows, cols). Writes marker at [0,0].
    Tests parameter size with 2D layout metadata.
    """
    global _gen_counter
    _gen_counter += 1

    tensor_params_call = ", ".join(f"t{i}: cute.Tensor" for i in range(n_tensors))
    tensor_params_kernel = ", ".join(f"t{i}: cute.Tensor" for i in range(n_tensors))
    tensor_args_bare = ", ".join(f"t{i}" for i in range(n_tensors))

    if tensor_params_call:
        tensor_params_call = ", " + tensor_params_call
        tensor_params_kernel = ", " + tensor_params_kernel
    tensor_args = (", " + tensor_args_bare) if tensor_args_bare else ""

    write_lines = []
    for i in range(n_tensors):
        write_lines.append(f"            t{i}[Int32(0)] = Float32({(i + 1) * 100}.0)")
    write_body = "\n".join(write_lines) if write_lines else "            pass"

    source = (
        f"class WriteKernel2D_{_gen_counter}:\n"
        f"    @cute.jit\n"
        f"    def __call__(self, stream: cuda.CUstream{tensor_params_call}):\n"
        f"        self.kernel({tensor_args_bare}).launch(\n"
        f"            grid=[1, 1, 1],\n"
        f"            block=[1, 1, 1],\n"
        f"            stream=stream,\n"
        f"        )\n"
        f"\n"
        f"    @cute.kernel\n"
        f"    def kernel(self{tensor_params_kernel}):\n"
        f"        tidx = cute.arch.thread_idx()[0]\n"
        f"        if tidx == Int32(0):\n"
        f"{write_body}\n"
    )

    filename = f"<write_kernel_2d_{_gen_counter}>"
    linecache.cache[filename] = (
        len(source), None, source.splitlines(True), filename,
    )

    exec_globals = {
        "cute": cute,
        "cuda": cuda,
        "Int32": Int32,
        "Float32": Float32,
    }
    code = compile(source, filename, "exec")
    exec(code, exec_globals)

    kernel_cls = exec_globals[f"WriteKernel2D_{_gen_counter}"]

    def make_tensors():
        tensors_torch = [
            torch.zeros(4, 8, dtype=torch.float32, device="cuda")
            for _ in range(n_tensors)
        ]
        tensors_cute = []
        for t in tensors_torch:
            ct = from_dlpack(t, assumed_align=16)
            ct = ct.mark_layout_dynamic(leading_dim=1)
            tensors_cute.append(ct)
        return tensors_torch, tensors_cute

    return kernel_cls(), make_tensors


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestCuteTensorParams:
    """Test passing cute.Tensor objects as kernel parameters."""

    def _get_stream(self):
        torch_stream = torch.cuda.current_stream()
        return cuda.CUstream(torch_stream.cuda_stream)

    def test_single_tensor_param(self):
        """Verify a single cute.Tensor can be passed and written to."""
        kernel, make_tensors = _make_write_kernel_class(1)
        tensors_torch, tensors_cute = make_tensors()

        stream = self._get_stream()
        kernel(stream, *tensors_cute)
        torch.cuda.synchronize()

        assert tensors_torch[0][0].item() == 100.0

    def test_multiple_tensor_params(self):
        """Verify multiple cute.Tensors can be passed and each written to."""
        n = 5
        kernel, make_tensors = _make_write_kernel_class(n)
        tensors_torch, tensors_cute = make_tensors()

        stream = self._get_stream()
        kernel(stream, *tensors_cute)
        torch.cuda.synchronize()

        for i in range(n):
            expected = float((i + 1) * 100)
            actual = tensors_torch[i][0].item()
            assert actual == expected, f"tensor {i}: expected {expected}, got {actual}"

    def test_2d_tensor_params(self):
        """Verify 2D cute.Tensors with dynamic layout work as params."""
        n = 5
        kernel, make_tensors = _make_write_kernel_class_2d(n)
        tensors_torch, tensors_cute = make_tensors()

        stream = self._get_stream()
        kernel(stream, *tensors_cute)
        torch.cuda.synchronize()

        for i in range(n):
            expected = float((i + 1) * 100)
            # [0] in linear indexing = element [0,0] in 2D
            actual = tensors_torch[i].flatten()[0].item()
            assert actual == expected, f"tensor {i}: expected {expected}, got {actual}"

    def test_dynamic_layout_tensor_params(self):
        """Verify cute.Tensors with dynamic layouts work."""
        n = 3
        kernel, make_tensors = _make_write_kernel_class(n, use_dynamic_layout=True)
        tensors_torch, tensors_cute = make_tensors()

        stream = self._get_stream()
        kernel(stream, *tensors_cute)
        torch.cuda.synchronize()

        for i in range(n):
            expected = float((i + 1) * 100)
            actual = tensors_torch[i][0].item()
            assert actual == expected, f"tensor {i}: expected {expected}, got {actual}"

    def test_param_limit_1d(self):
        """Find the maximum number of 1D cute.Tensor params.

        Uses binary search to find the limit. Reports the result.
        CUDA has a 4KB kernel parameter limit.
        """
        lo, hi = 1, 500  # Binary search bounds
        max_working = 0

        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                kernel, make_tensors = _make_write_kernel_class(mid)
                tensors_torch, tensors_cute = make_tensors()
                stream = self._get_stream()

                # cute.compile to trigger compilation without launch
                compiled = cute.compile(kernel, stream, *tensors_cute)
                compiled(stream, *tensors_cute)
                torch.cuda.synchronize()

                # Verify correctness
                ok = True
                for i in range(mid):
                    if tensors_torch[i][0].item() != float((i + 1) * 100):
                        ok = False
                        break

                if ok:
                    max_working = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            except Exception:
                hi = mid - 1

        print(f"\n{'=' * 60}")
        print(f"RESULT: Maximum 1D cute.Tensor parameters = {max_working}")
        print(f"{'=' * 60}")
        assert max_working >= 5, f"Expected at least 5 tensor params, got {max_working}"

    def test_param_limit_2d_dynamic(self):
        """Find the maximum number of 2D dynamic cute.Tensor params.

        2D tensors with dynamic layout carry more metadata per param,
        so the limit should be lower than 1D.
        """
        lo, hi = 1, 400
        max_working = 0

        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                kernel, make_tensors = _make_write_kernel_class_2d(mid)
                tensors_torch, tensors_cute = make_tensors()
                stream = self._get_stream()

                compiled = cute.compile(kernel, stream, *tensors_cute)
                compiled(stream, *tensors_cute)
                torch.cuda.synchronize()

                ok = True
                for i in range(mid):
                    if tensors_torch[i].flatten()[0].item() != float((i + 1) * 100):
                        ok = False
                        break

                if ok:
                    max_working = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            except Exception:
                hi = mid - 1

        print(f"\n{'=' * 60}")
        print(f"RESULT: Maximum 2D dynamic cute.Tensor parameters = {max_working}")
        print(f"{'=' * 60}")
        assert max_working >= 5, f"Expected at least 5 tensor params, got {max_working}"


if __name__ == "__main__":
    if not is_hopper_available():
        print("Hopper (SM90+) GPU required")
        exit(1)

    print("Testing cute.Tensor as kernel parameters")
    print("=" * 60)

    test = TestCuteTensorParams()

    print("\n--- test_single_tensor_param ---")
    test.test_single_tensor_param()
    print("PASSED")

    print("\n--- test_multiple_tensor_params ---")
    test.test_multiple_tensor_params()
    print("PASSED")

    print("\n--- test_2d_tensor_params ---")
    test.test_2d_tensor_params()
    print("PASSED")

    print("\n--- test_dynamic_layout_tensor_params ---")
    test.test_dynamic_layout_tensor_params()
    print("PASSED")

    print("\n--- test_param_limit_1d ---")
    test.test_param_limit_1d()

    print("\n--- test_param_limit_2d_dynamic ---")
    test.test_param_limit_2d_dynamic()
