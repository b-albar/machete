# Copyright (c) 2025, Machete Authors
"""Tests for kernel fusion in megakernels.

This file demonstrates fusing multiple kernels into a single megakernel launch.
The TensorDef API allows each kernel to declare its tensors independently,
and the megakernel automatically maps them to global slots at launch time.
"""

import pytest
import torch
import cutlass.cute as cute
from cutlass import Int32

from machete.megakernel.core import Megakernel
from machete.megakernel.interface import FusableKernel, TensorDef

HAS_CUDA = torch.cuda.is_available()


class AddKernel(FusableKernel):
    """Simple kernel: output = input + bias."""

    tensor_defs = [
        TensorDef("input_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("bias_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("output_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("n_elements", is_scalar=True),
    ]

    def get_logical_grid_size(self, *args) -> int:
        return (args[0].numel() + 255) // 256

    def compute_forward(self, idx, input_t, bias_t, output_t, n_elements):
        if idx < n_elements:
            output_t[idx] = input_t[idx] + bias_t[idx]


class ScaleKernel(FusableKernel):
    """Simple kernel: output = input * scale."""

    tensor_defs = [
        TensorDef("input_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("scale_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("output_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("n_elements", is_scalar=True),
    ]

    def get_logical_grid_size(self, *args) -> int:
        return (args[0].numel() + 255) // 256

    def compute_forward(self, idx, input_t, scale_t, output_t, n_elements):
        if idx < n_elements:
            output_t[idx] = input_t[idx] * scale_t[idx]


class ReluKernel(FusableKernel):
    """Simple kernel: output = max(input, 0)."""

    tensor_defs = [
        TensorDef("input_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("output_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("n_elements", is_scalar=True),
    ]

    def get_logical_grid_size(self, *args) -> int:
        return (args[0].numel() + 255) // 256

    def compute_forward(self, idx, input_t, output_t, n_elements):
        if idx < n_elements:
            val = input_t[idx]
            if val > 0.0:
                output_t[idx] = val
            else:
                output_t[idx] = 0.0


@pytest.fixture
def cuda_device():
    if not HAS_CUDA:
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestTwoKernelFusion:
    """Test fusing two kernels into one megakernel."""

    def test_add_then_scale_fusion(self, cuda_device):
        """Test fusing Add + Scale: output = (input + bias) * scale."""
        size = 1024

        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        scale_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        intermediate = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        n_elements = Int32(size)

        mk = Megakernel(name="test_add_scale", mode="forward", strategy="sequential")

        # Op 1: Add (input + bias -> intermediate)
        add_kernel = AddKernel()
        mk.add(add_kernel, input_t, bias_t, intermediate, n_elements)

        # Op 2: Scale (intermediate * scale -> output)
        scale_kernel = ScaleKernel()
        mk.add(scale_kernel, intermediate, scale_t, output_t, n_elements)

        # Launch - grid size covers all elements
        grid_size = add_kernel.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (256, 1, 1))

        # Verify: output = (input + bias) * scale
        expected = (input_t + bias_t) * scale_t
        torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-5)

    def test_add_then_relu_fusion(self, cuda_device):
        """Test fusing Add + ReLU: output = max(input + bias, 0)."""
        size = 2048

        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        intermediate = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        n_elements = Int32(size)

        mk = Megakernel(name="test_add_relu", mode="forward", strategy="sequential")

        # Op 1: Add
        add_kernel = AddKernel()
        mk.add(add_kernel, input_t, bias_t, intermediate, n_elements)

        # Op 2: ReLU
        relu_kernel = ReluKernel()
        mk.add(relu_kernel, intermediate, output_t, n_elements)

        grid_size = add_kernel.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (256, 1, 1))

        # Verify: output = relu(input + bias)
        expected = torch.relu(input_t + bias_t)
        torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-5)

    def test_scale_then_add_fusion(self, cuda_device):
        """Test fusing Scale + Add: output = (input * scale) + bias."""
        size = 1024

        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        scale_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        intermediate = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        n_elements = Int32(size)

        mk = Megakernel(name="test_scale_add", mode="forward", strategy="sequential")

        # Op 1: Scale
        scale_kernel = ScaleKernel()
        mk.add(scale_kernel, input_t, scale_t, intermediate, n_elements)

        # Op 2: Add
        add_kernel = AddKernel()
        mk.add(add_kernel, intermediate, bias_t, output_t, n_elements)

        grid_size = scale_kernel.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (256, 1, 1))

        # Verify: output = (input * scale) + bias
        expected = (input_t * scale_t) + bias_t
        torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestThreeKernelFusion:
    """Test fusing three kernels into one megakernel."""

    def test_add_scale_relu_fusion(self, cuda_device):
        """Test fusing Add + Scale + ReLU: output = relu((input + bias) * scale)."""
        size = 1024

        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        scale_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        inter1 = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        inter2 = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        n_elements = Int32(size)

        mk = Megakernel(name="test_add_scale_relu", mode="forward", strategy="sequential")

        # Op 1: Add (input + bias -> inter1)
        add_kernel = AddKernel()
        mk.add(add_kernel, input_t, bias_t, inter1, n_elements)

        # Op 2: Scale (inter1 * scale -> inter2)
        scale_kernel = ScaleKernel()
        mk.add(scale_kernel, inter1, scale_t, inter2, n_elements)

        # Op 3: ReLU (inter2 -> output)
        relu_kernel = ReluKernel()
        mk.add(relu_kernel, inter2, output_t, n_elements)

        grid_size = add_kernel.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (256, 1, 1))

        # Verify: output = relu((input + bias) * scale)
        expected = torch.relu((input_t + bias_t) * scale_t)
        torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestFusionWithLargerData:
    """Test fusion with larger tensor sizes."""

    def test_large_tensor_fusion(self, cuda_device):
        """Test fusion with larger tensors (64K elements)."""
        size = 65536

        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        scale_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        intermediate = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        n_elements = Int32(size)

        mk = Megakernel(name="test_large", mode="forward", strategy="sequential")

        add_kernel = AddKernel()
        mk.add(add_kernel, input_t, bias_t, intermediate, n_elements)

        scale_kernel = ScaleKernel()
        mk.add(scale_kernel, intermediate, scale_t, output_t, n_elements)

        grid_size = add_kernel.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (256, 1, 1))

        expected = (input_t + bias_t) * scale_t
        torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestTensorSharing:
    """Test that tensor sharing/reuse works correctly in fusion."""

    def test_shared_n_elements(self, cuda_device):
        """Test that the same n_elements scalar can be shared across ops."""
        size = 512

        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        scale_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        inter = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float32)

        # Use the same n_elements for all ops
        n_elements = Int32(size)

        mk = Megakernel(name="test_shared", mode="forward", strategy="sequential")

        add_kernel = AddKernel()
        mk.add(add_kernel, input_t, bias_t, inter, n_elements)

        scale_kernel = ScaleKernel()
        mk.add(scale_kernel, inter, scale_t, output_t, n_elements)

        grid_size = add_kernel.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (256, 1, 1))

        expected = (input_t + bias_t) * scale_t
        torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-5)


class SequentialAddKernel(FusableKernel):
    """Add kernel that explicitly requests sequential execution."""

    tensor_defs = [
        TensorDef("input_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("bias_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("output_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("n_elements", is_scalar=True),
    ]

    @property
    def execution_mode(self) -> str:
        return "sequential"

    def get_logical_grid_size(self, *args) -> int:
        return (args[0].numel() + 255) // 256

    def compute_forward(self, idx, input_t, bias_t, output_t, n_elements):
        if idx < n_elements:
            output_t[idx] = input_t[idx] + bias_t[idx]


class WarpSpecializedScaleKernel(FusableKernel):
    """Scale kernel that explicitly requests warp-specialized execution."""

    tensor_defs = [
        TensorDef("input_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("scale_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("output_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("n_elements", is_scalar=True),
    ]

    @property
    def execution_mode(self) -> str:
        return "warp_specialized"

    def get_logical_grid_size(self, *args) -> int:
        return (args[0].numel() + 255) // 256

    def compute_forward(self, idx, input_t, scale_t, output_t, n_elements):
        if idx < n_elements:
            output_t[idx] = input_t[idx] * scale_t[idx]


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestMixedExecutionModeFusion:
    """Test fusing kernels with different execution modes."""

    def test_mixed_modes_uses_megakernel_strategy(self, cuda_device):
        """When kernels have mixed execution_modes, megakernel strategy is used."""
        size = 1024

        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        scale_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        inter = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        n_elements = Int32(size)

        # Megakernel uses sequential strategy
        mk = Megakernel(name="test_mixed", mode="forward", strategy="sequential")

        # Op 1: Sequential Add kernel
        seq_add = SequentialAddKernel()
        mk.add(seq_add, input_t, bias_t, inter, n_elements)

        # Op 2: Warp-specialized Scale kernel
        warp_scale = WarpSpecializedScaleKernel()
        mk.add(warp_scale, inter, scale_t, output_t, n_elements)

        # Verify the instructions have different execution modes
        assert mk.instructions[0]["execution_mode"] == "sequential"
        assert mk.instructions[1]["execution_mode"] == "warp_specialized"

        # Launch with sequential block size (megakernel strategy)
        grid_size = seq_add.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (256, 1, 1))

        # Verify result
        expected = (input_t + bias_t) * scale_t
        torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-5)

    def test_all_sequential_kernels_in_fusion(self, cuda_device):
        """When all fused kernels are sequential, sequential mode is used."""
        size = 1024

        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias1 = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias2 = torch.randn(size, device=cuda_device, dtype=torch.float32)
        inter = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        n_elements = Int32(size)

        # Even though megakernel defaults to warp_specialized
        mk = Megakernel(name="test_all_seq", mode="forward", strategy="warp_specialized")

        # Both kernels are sequential
        seq_add1 = SequentialAddKernel()
        mk.add(seq_add1, input_t, bias1, inter, n_elements)

        seq_add2 = SequentialAddKernel()
        mk.add(seq_add2, inter, bias2, output_t, n_elements)

        # Both should be sequential
        assert mk.instructions[0]["execution_mode"] == "sequential"
        assert mk.instructions[1]["execution_mode"] == "sequential"

        grid_size = seq_add1.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (256, 1, 1))

        # Verify result: ((input + bias1) + bias2)
        expected = (input_t + bias1) + bias2
        torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-5)

    def test_all_warp_specialized_kernels_in_fusion(self, cuda_device):
        """When all fused kernels are warp_specialized, warp mode is used."""
        size = 2048

        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        scale1 = torch.randn(size, device=cuda_device, dtype=torch.float32)
        scale2 = torch.randn(size, device=cuda_device, dtype=torch.float32)
        inter = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        n_elements = Int32(size)

        # Megakernel defaults to sequential, but all ops are warp_specialized
        mk = Megakernel(name="test_all_warp", mode="forward", strategy="sequential")

        warp_scale1 = WarpSpecializedScaleKernel()
        mk.add(warp_scale1, input_t, scale1, inter, n_elements)

        warp_scale2 = WarpSpecializedScaleKernel()
        mk.add(warp_scale2, inter, scale2, output_t, n_elements)

        # Both should be warp_specialized
        assert mk.instructions[0]["execution_mode"] == "warp_specialized"
        assert mk.instructions[1]["execution_mode"] == "warp_specialized"

        # Launch with warp-specialized block size
        num_consumer_threads = mk.warp_config.num_consumer_warps * 32
        grid_size = (size + num_consumer_threads - 1) // num_consumer_threads
        mk.launch(grid_size, (grid_size, 1, 1), (mk.warp_config.total_threads, 1, 1))

        # Verify result: ((input * scale1) * scale2)
        expected = (input_t * scale1) * scale2
        torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-5)

    def test_three_ops_with_mixed_modes(self, cuda_device):
        """Test three-op fusion with mixed execution modes."""
        size = 1024

        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        scale_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        inter1 = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        inter2 = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        n_elements = Int32(size)

        mk = Megakernel(name="test_three_mixed", mode="forward", strategy="sequential")

        # Op 1: Sequential Add
        seq_add = SequentialAddKernel()
        mk.add(seq_add, input_t, bias_t, inter1, n_elements)

        # Op 2: Warp-specialized Scale
        warp_scale = WarpSpecializedScaleKernel()
        mk.add(warp_scale, inter1, scale_t, inter2, n_elements)

        # Op 3: Regular ReLU (inherits megakernel's sequential strategy)
        relu = ReluKernel()
        mk.add(relu, inter2, output_t, n_elements)

        # Verify execution modes
        assert mk.instructions[0]["execution_mode"] == "sequential"
        assert mk.instructions[1]["execution_mode"] == "warp_specialized"
        assert mk.instructions[2]["execution_mode"] == "sequential"  # inherits from megakernel

        grid_size = seq_add.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (256, 1, 1))

        # Verify result: relu((input + bias) * scale)
        expected = torch.relu((input_t + bias_t) * scale_t)
        torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
