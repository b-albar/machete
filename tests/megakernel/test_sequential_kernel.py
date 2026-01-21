# Copyright (c) 2025, Machete Authors
"""
Test sequential execution mode for MacheteKernel.

This test demonstrates a simple kernel that runs in sequential mode (default),
where all threads execute load -> compute -> store with sync_threads() barriers.
"""

import pytest
import torch
from typing import Dict, Tuple

try:
    import cutlass.cute as cute
    from cutlass import Float32, const_expr

    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False
    cute = None
    Float32 = None
    const_expr = None

from machete.megakernel import MacheteKernel, TensorSpec, WarpConfig

try:
    from quack.cute_dsl_utils import torch2cute_dtype_map
except ImportError:
    torch2cute_dtype_map = None


class VectorAddKernel(MacheteKernel):
    """Simple vector addition kernel for testing sequential execution.

    Computes: output = input_a + input_b

    This kernel demonstrates the sequential execution pattern where:
    - All threads participate in compute phase (direct global memory access)
    - No shared memory is used for simplicity
    """

    NUM_THREADS = 256
    TILE_SIZE = 256

    def __init__(self, dtype: torch.dtype):
        self.torch_dtype = dtype
        self.element_size = dtype.itemsize
        # Set cute_dtype if available
        if torch2cute_dtype_map is not None:
            self.cute_dtype = torch2cute_dtype_map[dtype]
        else:
            self.cute_dtype = None

    @property
    def smem_size_fwd(self) -> int:
        """No shared memory needed for direct global memory access."""
        return 0

    def declare_tensors(self) -> Dict[str, TensorSpec]:
        """Declare input and output tensors."""
        return {
            "input_a": TensorSpec(
                name="input_a",
                dtype=self.cute_dtype,
                shape_expr=("n_elements",),
                is_input=True,
            ),
            "input_b": TensorSpec(
                name="input_b",
                dtype=self.cute_dtype,
                shape_expr=("n_elements",),
                is_input=True,
            ),
            "output": TensorSpec(
                name="output",
                dtype=self.cute_dtype,
                shape_expr=("n_elements",),
                is_input=False,
                is_output=True,
            ),
        }

    def declare_scalars(self) -> Tuple[str, ...]:
        """Declare scalar parameters."""
        return ("n_elements",)

    def get_logical_grid_size(self, input_a, input_b, output, n_elements) -> int:
        """One logical block per tile."""
        return (n_elements + self.TILE_SIZE - 1) // self.TILE_SIZE

    @property
    def uses_warp_specialization(self) -> bool:
        """Sequential mode - all threads do all phases."""
        return False

    def setup_kernel(self, logical_idx, smem, input_a, input_b, output, n_elements):
        """Setup per-block state."""
        self.tile_start = logical_idx * const_expr(self.TILE_SIZE)

    def load_forward(self, logical_idx, smem, input_a, input_b, output, n_elements):
        """Load phase - no-op for direct global memory kernel."""
        pass

    def compute_forward(self, logical_idx, smem, input_a, input_b, output, n_elements):
        """Compute element-wise addition directly in global memory.

        Each thread processes elements strided by NUM_THREADS.
        """
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = const_expr(self.NUM_THREADS)
        tile_size = const_expr(self.TILE_SIZE)
        tile_start = self.tile_start

        # Each thread computes elements at stride NUM_THREADS
        for i in range(tidx, tile_size, num_threads):
            global_idx = tile_start + i
            if global_idx < n_elements:
                a_val = input_a[global_idx].to(Float32)
                b_val = input_b[global_idx].to(Float32)
                output[global_idx] = (a_val + b_val).to(self.cute_dtype)

    def store_forward(self, logical_idx, smem, input_a, input_b, output, n_elements):
        """Store phase - no-op since compute writes directly to global."""
        pass


class TestSequentialKernel:
    """Test suite for sequential kernel execution."""

    def test_kernel_interface(self):
        """Test that the kernel interface is correctly defined."""
        kernel = VectorAddKernel(torch.float16)

        # Check basic properties
        assert kernel.uses_warp_specialization is False
        assert kernel.NUM_STAGES == 1  # Default for sequential
        assert kernel.smem_size_fwd == 0  # No smem needed

        # Check tensor declarations
        tensors = kernel.declare_tensors()
        assert "input_a" in tensors
        assert "input_b" in tensors
        assert "output" in tensors
        assert tensors["input_a"].is_input is True
        assert tensors["output"].is_output is True

        # Check scalar declarations
        scalars = kernel.declare_scalars()
        assert "n_elements" in scalars

    def test_warp_config_default(self):
        """Test that warp config returns default for sequential kernels."""
        kernel = VectorAddKernel(torch.float16)
        config = kernel.warp_config

        # Default WarpConfig values
        assert isinstance(config, WarpConfig)
        assert config.num_consumer_warps > 0

    def test_logical_grid_size(self):
        """Test logical grid size calculation."""
        kernel = VectorAddKernel(torch.float16)

        # Create dummy tensors for API compatibility
        n_elements = 1000

        # Grid size should be ceil(n_elements / TILE_SIZE)
        expected_blocks = (n_elements + kernel.TILE_SIZE - 1) // kernel.TILE_SIZE
        actual_blocks = kernel.get_logical_grid_size(None, None, None, n_elements)

        assert actual_blocks == expected_blocks

    def test_smem_size(self):
        """Test shared memory size calculation."""
        kernel_f16 = VectorAddKernel(torch.float16)
        kernel_f32 = VectorAddKernel(torch.float32)

        # No smem needed for direct global memory access
        assert kernel_f16.smem_size_fwd == 0
        assert kernel_f32.smem_size_fwd == 0

    def test_kernel_signature(self):
        """Test kernel signature generation."""
        kernel = VectorAddKernel(torch.float16)
        sig = kernel.get_kernel_signature()

        assert len(sig.tensors) == 3
        assert len(sig.scalars) == 1
        assert "n_elements" in sig.scalars

        # Check input/output classification
        inputs = sig.get_inputs()
        outputs = sig.get_outputs()
        assert len(inputs) == 2
        assert len(outputs) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
