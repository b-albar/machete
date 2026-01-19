# Copyright (c) 2025, Machete Authors
"""Simple example tests for megakernel execution strategies."""

import pytest
import torch
import cutlass.cute as cute
from cutlass import Int32, const_expr

from machete.megakernel.core import Megakernel
from machete.megakernel.interface import FusableKernel

HAS_CUDA = torch.cuda.is_available()


class AddTensorsKernel(FusableKernel):
    """Simple kernel that adds two tensors element-wise: output = input + bias."""

    def get_logical_grid_size(self, *args) -> int:
        # args[0] is input tensor
        return (args[0].numel() + 255) // 256

    @cute.jit
    def load_forward(
        self, logical_idx, smem,
        t0, t1, t2, t3, t4, t5, t6, t7,
        t8, t9, t10, t11, t12, t13, t14, t15,
        t16, t17, t18, t19, t20, t21, t22, t23,
        t24, t25, t26, t27, t28, t29, t30, t31,
    ):
        pass

    @cute.jit
    def compute_forward(
        self, logical_idx, smem,
        t0, t1, t2, t3, t4, t5, t6, t7,
        t8, t9, t10, t11, t12, t13, t14, t15,
        t16, t17, t18, t19, t20, t21, t22, t23,
        t24, t25, t26, t27, t28, t29, t30, t31,
    ):
        # t0 = input, t1 = bias, t2 = output, t3 = n_elements
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = const_expr(256)

        input_ptr = cute.make_ptr(cute.Float32, t0, cute.AddressSpace.gmem)
        bias_ptr = cute.make_ptr(cute.Float32, t1, cute.AddressSpace.gmem)
        output_ptr = cute.make_ptr(cute.Float32, t2, cute.AddressSpace.gmem)

        input_tensor = cute.make_tensor(input_ptr, cute.make_layout((t3,)))
        bias_tensor = cute.make_tensor(bias_ptr, cute.make_layout((t3,)))
        output_tensor = cute.make_tensor(output_ptr, cute.make_layout((t3,)))

        idx = logical_idx * num_threads + tidx
        if idx < t3:
            val = input_tensor[idx]
            b = bias_tensor[idx]
            output_tensor[idx] = val + b

    @cute.jit
    def store_forward(
        self, logical_idx, smem,
        t0, t1, t2, t3, t4, t5, t6, t7,
        t8, t9, t10, t11, t12, t13, t14, t15,
        t16, t17, t18, t19, t20, t21, t22, t23,
        t24, t25, t26, t27, t28, t29, t30, t31,
    ):
        pass


@pytest.fixture
def cuda_device():
    if not HAS_CUDA:
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestAddTensorsKernel:
    """Test the AddTensorsKernel with both execution strategies."""

    def test_sequential_strategy(self, cuda_device):
        """Test AddTensorsKernel with sequential execution strategy."""
        size = 1024

        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        n_elements = Int32(size)

        # Create megakernel with sequential strategy
        mk = Megakernel(name="test_add_seq", mode="forward", strategy="sequential")
        kernel = AddTensorsKernel()
        mk.add(kernel, input_t, bias_t, output_t, n_elements)

        # Launch
        grid_size = kernel.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (256, 1, 1))

        # Verify
        expected = input_t + bias_t
        torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-5)

    def test_warp_specialized_strategy(self, cuda_device):
        """Test AddTensorsKernel with warp-specialized execution strategy."""
        size = 2048

        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        n_elements = Int32(size)

        # Create megakernel with warp specialization
        mk = Megakernel(name="test_add_warp", mode="forward", strategy="warp_specialized")
        kernel = AddTensorsKernel()
        mk.add(kernel, input_t, bias_t, output_t, n_elements)

        # Launch with full warp config threads
        grid_size = kernel.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (mk.warp_config.total_threads, 1, 1))

        # Verify
        expected = input_t + bias_t
        torch.testing.assert_close(output_t, expected, rtol=1e-5, atol=1e-5)

    def test_both_strategies_match(self, cuda_device):
        """Verify both strategies produce identical results."""
        size = 4096

        # Same input for both
        input_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        bias_t = torch.randn(size, device=cuda_device, dtype=torch.float32)
        output_seq = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        output_warp = torch.zeros(size, device=cuda_device, dtype=torch.float32)
        n_elements = Int32(size)

        # Sequential
        mk_seq = Megakernel(name="test_seq", mode="forward", strategy="sequential")
        kernel_seq = AddTensorsKernel()
        mk_seq.add(kernel_seq, input_t, bias_t, output_seq, n_elements)
        grid_size = kernel_seq.get_logical_grid_size(input_t)
        mk_seq.launch(grid_size, (grid_size, 1, 1), (256, 1, 1))

        # Warp-specialized
        mk_warp = Megakernel(name="test_warp", mode="forward", strategy="warp_specialized")
        kernel_warp = AddTensorsKernel()
        mk_warp.add(kernel_warp, input_t, bias_t, output_warp, n_elements)
        mk_warp.launch(grid_size, (grid_size, 1, 1), (mk_warp.warp_config.total_threads, 1, 1))

        # Both should match
        torch.testing.assert_close(output_seq, output_warp, rtol=1e-5, atol=1e-5)

        # And both should match expected
        expected = input_t + bias_t
        torch.testing.assert_close(output_seq, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
