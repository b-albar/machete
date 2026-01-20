# Copyright (c) 2025, Machete Authors
"""Simple example tests for megakernel execution strategies.

This file demonstrates the TensorDef API for writing clean FusableKernel
implementations. The megakernel automatically:
- Creates CuTe tensors from Int64 pointers based on tensor_defs
- Inlines the user's compute_forward body into the JIT wrapper
- Handles the verbose 32-slot signature internally
"""

import pytest
import torch
import cutlass.cute as cute
from cutlass import Int32

from machete.megakernel.core import Megakernel
from machete.megakernel.interface import FusableKernel, TensorDef

HAS_CUDA = torch.cuda.is_available()


class AddTensorsKernel(FusableKernel):
    """Simple kernel using TensorDef API: output = input + bias.

    The TensorDef API automatically:
    - Assigns slot indices based on definition order
    - Creates CuTe tensors from Int64 pointers
    - Inlines compute_forward into a JIT wrapper

    Note: compute_forward should NOT have @cute.jit decorator.
    The wrapper handles JIT compilation with tensor creation.
    """

    # Declarative tensor definitions - slots assigned by order (0, 1, 2, 3)
    tensor_defs = [
        TensorDef("input_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("bias_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("output_t", cute.Float32, shape=(0,), stride=(0,)),
        TensorDef("n_elements", is_scalar=True),
    ]

    def get_logical_grid_size(self, *args) -> int:
        # args[0] is input tensor
        return (args[0].numel() + 255) // 256

    # Parameters: idx (global thread index), then tensor_defs names
    def compute_forward(self, idx, input_t, bias_t, output_t, n_elements):
        """Clean compute logic - tensors are ready to use!"""
        if idx < n_elements:
            output_t[idx] = input_t[idx] + bias_t[idx]


@pytest.fixture
def cuda_device():
    if not HAS_CUDA:
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestTensorDefAPI:
    """Test kernels using the clean TensorDef API."""

    def test_sequential_strategy(self, cuda_device):
        """Test TensorDef kernel with sequential execution strategy."""
        size = 1024

        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        n_elements = Int32(size)

        mk = Megakernel(name="test_add_seq", mode="forward", strategy="sequential")
        kernel = AddTensorsKernel()
        mk.add(kernel, input_t, bias_t, output_t, n_elements)

        grid_size = kernel.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (256, 1, 1))

        expected = input_t + bias_t
        torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-5)

    def test_warp_specialized_strategy(self, cuda_device):
        """Test TensorDef kernel with warp-specialized execution strategy.

        In warp-specialized mode, only consumer warps (num_consumer_warps * 32 threads)
        run compute_forward, so grid_size must be calculated based on consumer threads.
        """
        size = 2048

        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        n_elements = Int32(size)

        mk = Megakernel(name="test_add_warp", mode="forward", strategy="warp_specialized")
        kernel = AddTensorsKernel()
        mk.add(kernel, input_t, bias_t, output_t, n_elements)

        # Grid size based on consumer threads (not all block threads)
        num_consumer_threads = mk.warp_config.num_consumer_warps * 32  # 12 * 32 = 384
        grid_size = (size + num_consumer_threads - 1) // num_consumer_threads
        mk.launch(grid_size, (grid_size, 1, 1), (mk.warp_config.total_threads, 1, 1))

        expected = input_t + bias_t
        torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-5)

    def test_large_tensor(self, cuda_device):
        """Test TensorDef kernel with larger tensor size."""
        size = 16384

        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        n_elements = Int32(size)

        mk = Megakernel(name="test_large", mode="forward", strategy="sequential")
        kernel = AddTensorsKernel()
        mk.add(kernel, input_t, bias_t, output_t, n_elements)

        grid_size = kernel.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (256, 1, 1))

        expected = input_t + bias_t
        torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
