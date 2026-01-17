# Copyright (c) 2025, Machete Authors
"""
CUDA Integration Tests for Megakernel Execution.

These tests verify:
1. Correct execution order of L/C/S phases
2. Synchronization between producer/consumer operations
3. Warp-specialized execution with proper role dispatch
4. Data correctness after megakernel execution

Note: These tests require CUDA and will be skipped if not available.
"""

import pytest
from typing import Tuple
import cutlass.cute as cute

# Try to import torch and check for CUDA
try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False
    torch = None

from machete.megakernel.interface import (
    FusableKernel,
    WarpSpecializedKernel,
    WarpConfig,
    WarpRole,
    reads,
    writes,
    warp_role,
)

# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


# ============================================================================
# Test Kernels
# ============================================================================


class SimpleAddKernel(FusableKernel):
    """Simple kernel: output = input + scalar."""

    TILE_SIZE = 256

    def __init__(self, dtype=cute.Float16):
        self.cute_dtype = dtype

    @property
    def smem_size(self) -> int:
        return 0

    @property
    def needs_block_sync(self) -> bool:
        return False

    def get_logical_grid_size(self, input_tensor, scalar, output_tensor, n_elements) -> int:
        return (n_elements + self.TILE_SIZE - 1) // self.TILE_SIZE

    def get_logical_coord(self, logical_idx: int, *args) -> Tuple[int, ...]:
        return (logical_idx,)

    def get_logical_coord_names(self) -> Tuple[str, ...]:
        return ("chunk",)

    @reads("input", "scalar")
    @cute.jit
    def load_forward(self, logical_idx, input_tensor, scalar, output_tensor, n_elements):
        pass

    @writes("output")
    @cute.jit
    def compute_forward(self, logical_idx, input_tensor, scalar, output_tensor, n_elements):
        tidx, _, _ = cute.arch.thread_idx()
        elem_idx = logical_idx * self.TILE_SIZE + tidx
        if elem_idx < n_elements:
            output_tensor[elem_idx] = input_tensor[elem_idx] + scalar[0]

    @cute.jit
    def store_forward(self, logical_idx, input_tensor, scalar, output_tensor, n_elements):
        pass

    def grid_fn(self, input_tensor, scalar, output_tensor, n_elements):
        return (self.get_logical_grid_size(input_tensor, scalar, output_tensor, n_elements), 1, 1)

    def block_fn(self, *args):
        return (self.TILE_SIZE, 1, 1)


class SimpleMulKernel(FusableKernel):
    """Simple kernel: output = input * scalar."""

    TILE_SIZE = 256

    def __init__(self, dtype=cute.Float16):
        self.cute_dtype = dtype

    @property
    def smem_size(self) -> int:
        return 0

    @property
    def needs_block_sync(self) -> bool:
        return False

    def get_logical_grid_size(self, input_tensor, scalar, output_tensor, n_elements) -> int:
        return (n_elements + self.TILE_SIZE - 1) // self.TILE_SIZE

    def get_logical_coord(self, logical_idx: int, *args) -> Tuple[int, ...]:
        return (logical_idx,)

    @reads("input", "scalar")
    @cute.jit
    def load_forward(self, logical_idx, input_tensor, scalar, output_tensor, n_elements):
        pass

    @writes("output")
    @cute.jit
    def compute_forward(self, logical_idx, input_tensor, scalar, output_tensor, n_elements):
        tidx, _, _ = cute.arch.thread_idx()
        elem_idx = logical_idx * self.TILE_SIZE + tidx
        if elem_idx < n_elements:
            output_tensor[elem_idx] = input_tensor[elem_idx] * scalar[0]

    @cute.jit
    def store_forward(self, logical_idx, input_tensor, scalar, output_tensor, n_elements):
        pass

    def grid_fn(self, input_tensor, scalar, output_tensor, n_elements):
        return (self.get_logical_grid_size(input_tensor, scalar, output_tensor, n_elements), 1, 1)

    def block_fn(self, *args):
        return (self.TILE_SIZE, 1, 1)


class SharedMemKernel(FusableKernel):
    """Kernel using shared memory to verify L/C/S ordering.

    Load phase writes to smem, compute reads from smem.
    This verifies __syncthreads() is correctly placed.
    """

    TILE_SIZE = 256

    def __init__(self, dtype=cute.Float16):
        self.cute_dtype = dtype

    @property
    def smem_size(self) -> int:
        return self.TILE_SIZE * 2  # fp16

    @property
    def needs_block_sync(self) -> bool:
        return True  # Critical: must sync between L and C

    def get_logical_grid_size(self, input_tensor, output_tensor, n_elements) -> int:
        return (n_elements + self.TILE_SIZE - 1) // self.TILE_SIZE

    def get_logical_coord(self, logical_idx: int, *args) -> Tuple[int, ...]:
        return (logical_idx,)

    @reads("input")
    @cute.jit
    def load_forward(self, logical_idx, smem, input_tensor, output_tensor, n_elements):
        """Load data into shared memory."""
        tidx, _, _ = cute.arch.thread_idx()
        elem_idx = logical_idx * self.TILE_SIZE + tidx
        if elem_idx < n_elements and tidx < self.TILE_SIZE:
            smem[tidx] = input_tensor[elem_idx]

    @writes("output")
    @cute.jit
    def compute_forward(self, logical_idx, smem, input_tensor, output_tensor, n_elements):
        """Read from smem and write to output."""
        tidx, _, _ = cute.arch.thread_idx()
        elem_idx = logical_idx * self.TILE_SIZE + tidx
        if elem_idx < n_elements and tidx < self.TILE_SIZE:
            # Read from smem (must be synced after load)
            val = smem[tidx]
            # Double the value to verify we read correct data
            output_tensor[elem_idx] = val + val

    @cute.jit
    def store_forward(self, logical_idx, smem, input_tensor, output_tensor, n_elements):
        pass

    def grid_fn(self, input_tensor, output_tensor, n_elements):
        return (self.get_logical_grid_size(input_tensor, output_tensor, n_elements), 1, 1)

    def block_fn(self, *args):
        return (self.TILE_SIZE, 1, 1)


class WarpSpecializedTestKernel(WarpSpecializedKernel):
    """Warp-specialized kernel for testing role dispatch."""

    TILE_SIZE = 256

    def __init__(self, dtype=cute.Float16):
        self.cute_dtype = dtype

    @property
    def warp_config(self) -> WarpConfig:
        # Use fewer warps for testing
        return WarpConfig(num_consumer_warps=4)

    @property
    def smem_size(self) -> int:
        return self.TILE_SIZE * 2

    def get_logical_grid_size(self, input_tensor, scalar, output_tensor, n_elements) -> int:
        return (n_elements + self.TILE_SIZE - 1) // self.TILE_SIZE

    def get_logical_coord(self, logical_idx: int, *args) -> Tuple[int, ...]:
        return (logical_idx,)

    @warp_role(WarpRole.LOADER)
    @reads("input")
    @cute.jit
    def load_forward(self, logical_idx, smem, input_tensor, scalar, output_tensor, n_elements):
        """Loader warp loads data to smem."""
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        tile_start = logical_idx * self.TILE_SIZE

        # Each thread in loader warp loads multiple elements
        for i in range(self.TILE_SIZE // 32):
            elem_idx = tile_start + lane + i * 32
            smem_idx = lane + i * 32
            if elem_idx < n_elements and smem_idx < self.TILE_SIZE:
                smem[smem_idx] = input_tensor[elem_idx]

    @warp_role(WarpRole.CONSUMER)
    @writes("output")
    @cute.jit
    def compute_forward(self, logical_idx, smem, input_tensor, scalar, output_tensor, n_elements):
        """Consumer warps compute output = smem * scalar."""
        tidx, _, _ = cute.arch.thread_idx()
        warp_id = tidx // 32
        lane = tidx % 32
        tile_start = logical_idx * self.TILE_SIZE

        # Each consumer warp handles a portion
        elements_per_warp = self.TILE_SIZE // 4  # 4 consumer warps
        start = warp_id * elements_per_warp

        for i in range((elements_per_warp + 31) // 32):
            local_idx = start + lane + i * 32
            elem_idx = tile_start + local_idx
            if local_idx < start + elements_per_warp and elem_idx < n_elements:
                val = smem[local_idx]
                output_tensor[elem_idx] = val * scalar[0]

    @warp_role(WarpRole.STORER)
    @cute.jit
    def store_forward(self, logical_idx, smem, input_tensor, scalar, output_tensor, n_elements):
        pass

    def grid_fn(self, input_tensor, scalar, output_tensor, n_elements):
        return (self.get_logical_grid_size(input_tensor, scalar, output_tensor, n_elements), 1, 1)

    def block_fn(self, *args):
        return (self.warp_config.total_threads, 1, 1)


# ============================================================================
# Test Classes
# ============================================================================

# Note: cuda_device, trace_file, and trace_output_dir fixtures are defined in conftest.py


class TestSimpleExecution:
    """Test basic megakernel execution."""

    def test_single_add_kernel(self, cuda_device):
        """Test single AddKernel execution."""
        from machete.megakernel.core import Megakernel

        # Create test data
        n = 1024
        input_tensor = torch.ones(n, dtype=torch.float16, device=cuda_device)
        scalar = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        # Create and run megakernel
        kernel = SimpleAddKernel()
        mk = Megakernel(name="test_add")
        mk.add(kernel, input_tensor, scalar, output, n)
        mk.launch_logical(block=(256, 1, 1))

        # Verify result
        expected = input_tensor + scalar[0]
        torch.testing.assert_close(output, expected)

    def test_single_mul_kernel(self, cuda_device):
        """Test single MulKernel execution."""
        from machete.megakernel.core import Megakernel

        n = 1024
        input_tensor = torch.full((n,), 3.0, dtype=torch.float16, device=cuda_device)
        scalar = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel = SimpleMulKernel()
        mk = Megakernel(name="test_mul")
        mk.add(kernel, input_tensor, scalar, output, n)
        mk.launch_logical(block=(256, 1, 1))

        expected = input_tensor * scalar[0]
        torch.testing.assert_close(output, expected)

    def test_non_power_of_two_size(self, cuda_device):
        """Test with non-power-of-two tensor size."""
        from machete.megakernel.core import Megakernel

        n = 1000  # Not divisible by 256
        input_tensor = torch.randn(n, dtype=torch.float16, device=cuda_device)
        scalar = torch.tensor([1.5], dtype=torch.float16, device=cuda_device)
        output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel = SimpleAddKernel()
        mk = Megakernel(name="test_non_pow2")
        mk.add(kernel, input_tensor, scalar, output, n)
        mk.launch_logical(block=(256, 1, 1))

        expected = input_tensor + scalar[0]
        torch.testing.assert_close(output, expected)


class TestPipelineExecution:
    """Test fused pipeline execution (multiple kernels)."""

    def test_add_then_mul_pipeline(self, cuda_device):
        """Test Add -> Mul pipeline: output = (input + a) * b."""
        from machete.megakernel.core import Megakernel

        n = 1024
        input_tensor = torch.ones(n, dtype=torch.float16, device=cuda_device)
        add_scalar = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        mul_scalar = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        intermediate = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        add_kernel = SimpleAddKernel()
        mul_kernel = SimpleMulKernel()

        mk = Megakernel(name="add_mul_pipeline")
        mk.add(add_kernel, input_tensor, add_scalar, intermediate, n)
        mk.add(mul_kernel, intermediate, mul_scalar, output, n)
        mk.launch_logical(block=(256, 1, 1))

        # Expected: (1 + 1) * 2 = 4
        expected = torch.full((n,), 4.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(output, expected)

    def test_mul_then_add_pipeline(self, cuda_device):
        """Test Mul -> Add pipeline: output = input * a + b."""
        from machete.megakernel.core import Megakernel

        n = 1024
        input_tensor = torch.full((n,), 3.0, dtype=torch.float16, device=cuda_device)
        mul_scalar = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        add_scalar = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        intermediate = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mul_kernel = SimpleMulKernel()
        add_kernel = SimpleAddKernel()

        mk = Megakernel(name="mul_add_pipeline")
        mk.add(mul_kernel, input_tensor, mul_scalar, intermediate, n)
        mk.add(add_kernel, intermediate, add_scalar, output, n)
        mk.launch_logical(block=(256, 1, 1))

        # Expected: 3 * 2 + 1 = 7
        expected = torch.full((n,), 7.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(output, expected)

    def test_three_stage_pipeline(self, cuda_device):
        """Test three-stage pipeline: A -> B -> C."""
        from machete.megakernel.core import Megakernel

        n = 1024
        x = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        s1 = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)
        s2 = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        s3 = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        t1 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        t2 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        out = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="three_stage")
        mk.add(SimpleMulKernel(), x, s1, t1, n)  # t1 = x * 3 = 6
        mk.add(SimpleAddKernel(), t1, s2, t2, n)  # t2 = t1 + 2 = 8
        mk.add(SimpleMulKernel(), t2, s3, out, n)  # out = t2 * 1 = 8
        mk.launch_logical(block=(256, 1, 1))

        expected = torch.full((n,), 8.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(out, expected)


class TestSharedMemorySynchronization:
    """Test shared memory synchronization between L/C/S phases."""

    def test_smem_load_compute_sync(self, cuda_device):
        """Test that compute phase sees data loaded in load phase."""
        from machete.megakernel.core import Megakernel

        n = 1024
        input_tensor = torch.randn(n, dtype=torch.float16, device=cuda_device)
        output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel = SharedMemKernel()
        mk = Megakernel(name="smem_sync_test")
        mk.add(kernel, input_tensor, output, n)
        mk.launch_logical(block=(256, 1, 1))

        # Kernel doubles input via smem
        expected = input_tensor * 2
        torch.testing.assert_close(output, expected)


class TestWarpSpecialization:
    """Test warp-specialized kernel execution."""

    def test_warp_specialized_basic(self, cuda_device):
        """Test basic warp-specialized kernel execution."""
        from machete.megakernel.core import Megakernel

        n = 1024
        input_tensor = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        scalar = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)
        output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel = WarpSpecializedTestKernel()
        mk = Megakernel(name="warp_spec_test")
        mk.add(kernel, input_tensor, scalar, output, n)

        # Use warp config thread count
        block_size = kernel.warp_config.total_threads
        mk.launch_logical(block=(block_size, 1, 1))

        # Expected: 2 * 3 = 6
        expected = torch.full((n,), 6.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(output, expected)

    def test_warp_config_detection(self, cuda_device):
        """Test that warp specialization is detected correctly."""
        from machete.megakernel.core import Megakernel

        kernel = WarpSpecializedTestKernel()
        mk = Megakernel(name="warp_detect_test")

        # Create dummy tensors
        n = 256
        x = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        s = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk.add(kernel, x, s, y, n)

        # Check instruction has warp specialization info
        assert len(mk.instructions) == 1
        inst = mk.instructions[0]
        assert inst.get("uses_warp_specialization") is True
        assert inst.get("warp_config") is not None
        assert inst["warp_config"].num_consumer_warps == 4


class TestMixedModeExecution:
    """Test mixed sequential + warp-specialized kernel fusion."""

    def test_sequential_then_warp_specialized(self, cuda_device):
        """Test fusing a sequential kernel followed by a warp-specialized kernel."""
        from machete.megakernel.core import Megakernel

        n = 1024
        x = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        add_scalar = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        mul_scalar = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)
        intermediate = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        # Sequential kernel (no warp specialization)
        add_kernel = SimpleAddKernel()
        # Warp-specialized kernel
        warp_kernel = WarpSpecializedTestKernel()

        mk = Megakernel(name="seq_then_warp")
        mk.add(add_kernel, x, add_scalar, intermediate, n)  # intermediate = 2 + 1 = 3
        mk.add(warp_kernel, intermediate, mul_scalar, output, n)  # output = 3 * 3 = 9

        # Use warp config thread count for the warp-specialized kernel
        block_size = warp_kernel.warp_config.total_threads
        mk.launch_logical(block=(block_size, 1, 1))

        expected = torch.full((n,), 9.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(output, expected)

    def test_warp_specialized_then_sequential(self, cuda_device):
        """Test fusing a warp-specialized kernel followed by a sequential kernel."""
        from machete.megakernel.core import Megakernel

        n = 1024
        x = torch.full((n,), 4.0, dtype=torch.float16, device=cuda_device)
        mul_scalar = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        add_scalar = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        intermediate = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        # Warp-specialized kernel
        warp_kernel = WarpSpecializedTestKernel()
        # Sequential kernel (no warp specialization)
        add_kernel = SimpleAddKernel()

        mk = Megakernel(name="warp_then_seq")
        mk.add(warp_kernel, x, mul_scalar, intermediate, n)  # intermediate = 4 * 2 = 8
        mk.add(add_kernel, intermediate, add_scalar, output, n)  # output = 8 + 1 = 9

        # Use warp config thread count for the warp-specialized kernel
        block_size = warp_kernel.warp_config.total_threads
        mk.launch_logical(block=(block_size, 1, 1))

        expected = torch.full((n,), 9.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(output, expected)

    def test_mixed_three_stage_pipeline(self, cuda_device):
        """Test three-stage pipeline: sequential -> warp-spec -> sequential."""
        from machete.megakernel.core import Megakernel

        n = 1024
        x = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        s1 = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)  # add
        s2 = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)  # mul (warp-spec)
        s3 = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)  # add
        t1 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        t2 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        out = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        add1 = SimpleAddKernel()
        warp_mul = WarpSpecializedTestKernel()
        add2 = SimpleAddKernel()

        mk = Megakernel(name="mixed_three_stage")
        mk.add(add1, x, s1, t1, n)  # t1 = 2 + 1 = 3
        mk.add(warp_mul, t1, s2, t2, n)  # t2 = 3 * 2 = 6
        mk.add(add2, t2, s3, out, n)  # out = 6 + 3 = 9

        block_size = warp_mul.warp_config.total_threads
        mk.launch_logical(block=(block_size, 1, 1))

        expected = torch.full((n,), 9.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(out, expected)


class TestDataCorrectness:
    """Test data correctness with various inputs."""

    def test_random_inputs(self, cuda_device):
        """Test with random input values."""
        from machete.megakernel.core import Megakernel

        n = 4096
        for _ in range(5):  # Multiple random trials
            input_tensor = torch.randn(n, dtype=torch.float16, device=cuda_device)
            scalar = torch.randn(1, dtype=torch.float16, device=cuda_device)
            output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

            kernel = SimpleAddKernel()
            mk = Megakernel(name="random_test")
            mk.add(kernel, input_tensor, scalar, output, n)
            mk.launch_logical(block=(256, 1, 1))

            expected = input_tensor + scalar[0]
            torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)

    def test_large_tensor(self, cuda_device):
        """Test with large tensor."""
        from machete.megakernel.core import Megakernel

        n = 1024 * 1024  # 1M elements
        input_tensor = torch.randn(n, dtype=torch.float16, device=cuda_device)
        scalar = torch.tensor([0.5], dtype=torch.float16, device=cuda_device)
        output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel = SimpleMulKernel()
        mk = Megakernel(name="large_tensor_test")
        mk.add(kernel, input_tensor, scalar, output, n)
        mk.launch_logical(block=(256, 1, 1))

        expected = input_tensor * scalar[0]
        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)

    def test_edge_values(self, cuda_device):
        """Test with edge case values (zeros, ones, negative)."""
        from machete.megakernel.core import Megakernel

        n = 1024
        test_cases = [
            (torch.zeros(n), 1.0),  # zeros + 1
            (torch.ones(n), 0.0),  # ones + 0
            (torch.ones(n), -1.0),  # ones - 1
            (-torch.ones(n), 1.0),  # -ones + 1
        ]

        for input_vals, scalar_val in test_cases:
            input_tensor = input_vals.to(dtype=torch.float16, device=cuda_device)
            scalar = torch.tensor([scalar_val], dtype=torch.float16, device=cuda_device)
            output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

            kernel = SimpleAddKernel()
            mk = Megakernel(name="edge_test")
            mk.add(kernel, input_tensor, scalar, output, n)
            mk.launch_logical(block=(256, 1, 1))

            expected = input_tensor + scalar[0]
            torch.testing.assert_close(output, expected)


class TestLogicalBlocksExecution:
    """Test logical blocks coordinate system."""

    def test_logical_grid_calculation(self, cuda_device):
        """Test that logical grid size is calculated correctly."""
        from machete.megakernel.core import Megakernel

        n = 1000  # Requires ceil(1000/256) = 4 blocks
        input_tensor = torch.randn(n, dtype=torch.float16, device=cuda_device)
        scalar = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel = SimpleAddKernel()
        mk = Megakernel(name="logical_grid_test")
        mk.add(kernel, input_tensor, scalar, output, n)

        # Calculate logical blocks
        total_logical = mk._calculate_logical_blocks()
        assert total_logical == 4

        # Verify execution still works
        mk.launch_logical(block=(256, 1, 1))
        expected = input_tensor + scalar[0]
        torch.testing.assert_close(output, expected)


class TestTraceExport:
    """Test trace export for manual verification of operation order.

    These tests export .nanotrace files that can be viewed with cutedsl-trace tools.
    The trace includes:
    - L/C/S phase timing for each operation
    - Fine-grained semaphore wait/signal events
    - Page acquisition/release events (for paged mode)

    To export traces to a persistent directory:
        MACHETE_TRACE_DIR=/path/to/traces pytest tests/megakernel/test_cuda_execution.py

    Or use pytest options:
        pytest --trace-kernels --trace-dir=/path/to/traces tests/megakernel/
    """

    @pytest.mark.trace
    def test_single_kernel_trace(self, cuda_device, trace_file):
        """Export trace for a single kernel to verify L/C/S ordering."""
        from machete.megakernel.core import Megakernel
        from pathlib import Path

        n = 1024
        input_tensor = torch.ones(n, dtype=torch.float16, device=cuda_device)
        scalar = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel = SimpleAddKernel()
        mk = Megakernel(name="trace_single")
        mk.add(kernel, input_tensor, scalar, output, n)
        mk.launch_logical(block=(256, 1, 1), trace_file=trace_file)

        # Verify trace file was created (with .nanotrace extension)
        actual_path = Path(trace_file).with_suffix(".nanotrace")
        assert actual_path.exists(), f"Trace file not found: {actual_path}"

        # Verify data correctness
        expected = input_tensor + scalar[0]
        torch.testing.assert_close(output, expected)

    @pytest.mark.trace
    def test_pipeline_trace(self, cuda_device, trace_file):
        """Export trace for Add -> Mul pipeline to verify execution order."""
        from machete.megakernel.core import Megakernel
        from pathlib import Path

        n = 1024
        input_tensor = torch.ones(n, dtype=torch.float16, device=cuda_device)
        add_scalar = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        mul_scalar = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        intermediate = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        add_kernel = SimpleAddKernel()
        mul_kernel = SimpleMulKernel()

        mk = Megakernel(name="trace_pipeline")
        mk.add(add_kernel, input_tensor, add_scalar, intermediate, n)
        mk.add(mul_kernel, intermediate, mul_scalar, output, n)
        mk.launch_logical(block=(256, 1, 1), trace_file=trace_file)

        actual_path = Path(trace_file).with_suffix(".nanotrace")
        assert actual_path.exists(), f"Trace file not found: {actual_path}"

        # Verify: (1 + 1) * 2 = 4
        expected = torch.full((n,), 4.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(output, expected)

    @pytest.mark.trace
    def test_smem_sync_trace(self, cuda_device, trace_file):
        """Export trace for shared memory kernel to verify sync points."""
        from machete.megakernel.core import Megakernel
        from pathlib import Path

        n = 1024
        input_tensor = torch.randn(n, dtype=torch.float16, device=cuda_device)
        output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel = SharedMemKernel()
        mk = Megakernel(name="trace_smem")
        mk.add(kernel, input_tensor, output, n)
        mk.launch_logical(block=(256, 1, 1), trace_file=trace_file)

        actual_path = Path(trace_file).with_suffix(".nanotrace")
        assert actual_path.exists(), f"Trace file not found: {actual_path}"

        # Verify correctness (doubling via smem)
        expected = input_tensor * 2
        torch.testing.assert_close(output, expected)

    @pytest.mark.trace
    def test_warp_specialized_trace(self, cuda_device, trace_file):
        """Export trace for warp-specialized kernel to verify role dispatch."""
        from machete.megakernel.core import Megakernel
        from pathlib import Path

        n = 1024
        input_tensor = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        scalar = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)
        output = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel = WarpSpecializedTestKernel()
        mk = Megakernel(name="trace_warp_spec")
        mk.add(kernel, input_tensor, scalar, output, n)

        block_size = kernel.warp_config.total_threads
        mk.launch_logical(block=(block_size, 1, 1), trace_file=trace_file)

        actual_path = Path(trace_file).with_suffix(".nanotrace")
        assert actual_path.exists(), f"Trace file not found: {actual_path}"

        # Verify: 2 * 3 = 6
        expected = torch.full((n,), 6.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(output, expected)

    @pytest.mark.trace
    def test_three_stage_trace(self, cuda_device, trace_file):
        """Export trace for three-stage pipeline to verify complex ordering."""
        from machete.megakernel.core import Megakernel
        from pathlib import Path

        n = 1024
        x = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        s1 = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)
        s2 = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        s3 = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        t1 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        t2 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        out = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="trace_three_stage")
        mk.add(SimpleMulKernel(), x, s1, t1, n)  # t1 = x * 3 = 6
        mk.add(SimpleAddKernel(), t1, s2, t2, n)  # t2 = t1 + 2 = 8
        mk.add(SimpleMulKernel(), t2, s3, out, n)  # out = t2 * 1 = 8
        mk.launch_logical(block=(256, 1, 1), trace_file=trace_file)

        actual_path = Path(trace_file).with_suffix(".nanotrace")
        assert actual_path.exists(), f"Trace file not found: {actual_path}"

        expected = torch.full((n,), 8.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(out, expected)


class TestTraceToFixedPath:
    """Test trace export to a fixed path for easy manual inspection.

    These tests export traces to a known location for manual analysis
    using cutedsl-trace visualization tools.

    Run with:
        MACHETE_TRACE_DIR=/path/to/traces pytest -v tests/megakernel/test_cuda_execution.py::TestTraceToFixedPath
    """

    @pytest.mark.trace
    def test_comprehensive_trace_export(self, cuda_device, trace_output_dir):
        """Export multiple traces for comprehensive manual verification.

        The trace files can be viewed with cutedsl-trace visualization:
            cutedsl-trace view <trace_file.nanotrace>
        """
        from machete.megakernel.core import Megakernel

        # Test 1: Simple single kernel
        n = 512
        x1 = torch.randn(n, dtype=torch.float16, device=cuda_device)
        s1 = torch.tensor([1.5], dtype=torch.float16, device=cuda_device)
        y1 = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk1 = Megakernel(name="single_op")
        mk1.add(SimpleAddKernel(), x1, s1, y1, n)
        mk1.launch_logical(block=(256, 1, 1), trace_file=str(trace_output_dir / "01_single_op.nanotrace"))

        # Test 2: Two-stage pipeline
        x2 = torch.randn(n, dtype=torch.float16, device=cuda_device)
        s2a = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        s2b = torch.tensor([0.5], dtype=torch.float16, device=cuda_device)
        t2 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        y2 = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk2 = Megakernel(name="two_stage")
        mk2.add(SimpleMulKernel(), x2, s2a, t2, n)
        mk2.add(SimpleAddKernel(), t2, s2b, y2, n)
        mk2.launch_logical(block=(256, 1, 1), trace_file=str(trace_output_dir / "02_two_stage.nanotrace"))

        # Test 3: Shared memory sync
        x3 = torch.randn(n, dtype=torch.float16, device=cuda_device)
        y3 = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk3 = Megakernel(name="smem_sync")
        mk3.add(SharedMemKernel(), x3, y3, n)
        mk3.launch_logical(block=(256, 1, 1), trace_file=str(trace_output_dir / "03_smem_sync.nanotrace"))

        # Test 4: Warp specialized
        x4 = torch.randn(n, dtype=torch.float16, device=cuda_device)
        s4 = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        y4 = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel4 = WarpSpecializedTestKernel()
        mk4 = Megakernel(name="warp_spec")
        mk4.add(kernel4, x4, s4, y4, n)
        mk4.launch_logical(
            block=(kernel4.warp_config.total_threads, 1, 1),
            trace_file=str(trace_output_dir / "04_warp_spec.nanotrace"),
        )

        # Print summary
        print("\n" + "=" * 60)
        print("TRACE FILES EXPORTED FOR MANUAL INSPECTION")
        print("=" * 60)
        print(f"Directory: {trace_output_dir}")
        print("\nFiles:")
        for f in sorted(trace_output_dir.glob("*.nanotrace")):
            print(f"  - {f.name}")
        print("\nUse cutedsl-trace visualization tools to inspect:")
        print("  cutedsl-trace view <trace_file.nanotrace>")
        print("=" * 60)


# ============================================================================
# Additional Test Kernels for Comprehensive Testing
# ============================================================================


class IndependentAddKernel(FusableKernel):
    """Add kernel that writes to a DIFFERENT output (no dependency on prev output).

    Used to test early load optimization when ops are independent.
    """

    TILE_SIZE = 256

    def __init__(self, output_name: str = "out_a", dtype=cute.Float16):
        self.cute_dtype = dtype
        self._output_name = output_name

    @property
    def smem_size(self) -> int:
        return 0

    def get_logical_grid_size(self, input_tensor, scalar, output_tensor, n_elements) -> int:
        return (n_elements + self.TILE_SIZE - 1) // self.TILE_SIZE

    @reads("input")
    @cute.jit
    def load_forward(self, logical_idx, input_tensor, scalar, output_tensor, n_elements):
        pass

    @writes("out_a")  # Different output name - no dependency
    @cute.jit
    def compute_forward(self, logical_idx, input_tensor, scalar, output_tensor, n_elements):
        tidx, _, _ = cute.arch.thread_idx()
        elem_idx = logical_idx * self.TILE_SIZE + tidx
        if elem_idx < n_elements:
            output_tensor[elem_idx] = input_tensor[elem_idx] + scalar[0]

    @cute.jit
    def store_forward(self, logical_idx, input_tensor, scalar, output_tensor, n_elements):
        pass

    def grid_fn(self, *args):
        return (self.get_logical_grid_size(*args), 1, 1)

    def block_fn(self, *args):
        return (self.TILE_SIZE, 1, 1)


class IndependentMulKernel(FusableKernel):
    """Mul kernel that writes to a DIFFERENT output (no dependency on prev output)."""

    TILE_SIZE = 256

    def __init__(self, output_name: str = "out_b", dtype=cute.Float16):
        self.cute_dtype = dtype
        self._output_name = output_name

    @property
    def smem_size(self) -> int:
        return 0

    def get_logical_grid_size(self, input_tensor, scalar, output_tensor, n_elements) -> int:
        return (n_elements + self.TILE_SIZE - 1) // self.TILE_SIZE

    @reads("input2")  # Different input name
    @cute.jit
    def load_forward(self, logical_idx, input_tensor, scalar, output_tensor, n_elements):
        pass

    @writes("out_b")  # Different output name - no dependency
    @cute.jit
    def compute_forward(self, logical_idx, input_tensor, scalar, output_tensor, n_elements):
        tidx, _, _ = cute.arch.thread_idx()
        elem_idx = logical_idx * self.TILE_SIZE + tidx
        if elem_idx < n_elements:
            output_tensor[elem_idx] = input_tensor[elem_idx] * scalar[0]

    @cute.jit
    def store_forward(self, logical_idx, input_tensor, scalar, output_tensor, n_elements):
        pass

    def grid_fn(self, *args):
        return (self.get_logical_grid_size(*args), 1, 1)

    def block_fn(self, *args):
        return (self.TILE_SIZE, 1, 1)


class WarpSpecIndependentKernel(WarpSpecializedKernel):
    """Warp-specialized kernel with no dependency on previous ops."""

    TILE_SIZE = 256

    def __init__(self, dtype=cute.Float16):
        self.cute_dtype = dtype

    @property
    def warp_config(self) -> WarpConfig:
        return WarpConfig(num_consumer_warps=4)

    @property
    def smem_size(self) -> int:
        return self.TILE_SIZE * 2

    def get_logical_grid_size(self, input_tensor, scalar, output_tensor, n_elements) -> int:
        return (n_elements + self.TILE_SIZE - 1) // self.TILE_SIZE

    @warp_role(WarpRole.LOADER)
    @reads("warp_input")  # Different input name
    @cute.jit
    def load_forward(self, logical_idx, smem, input_tensor, scalar, output_tensor, n_elements):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        tile_start = logical_idx * self.TILE_SIZE

        for i in range(self.TILE_SIZE // 32):
            elem_idx = tile_start + lane + i * 32
            smem_idx = lane + i * 32
            if elem_idx < n_elements and smem_idx < self.TILE_SIZE:
                smem[smem_idx] = input_tensor[elem_idx]

    @warp_role(WarpRole.CONSUMER)
    @writes("warp_output")  # Different output name
    @cute.jit
    def compute_forward(self, logical_idx, smem, input_tensor, scalar, output_tensor, n_elements):
        tidx, _, _ = cute.arch.thread_idx()
        warp_id = tidx // 32
        lane = tidx % 32
        tile_start = logical_idx * self.TILE_SIZE

        elements_per_warp = self.TILE_SIZE // 4
        start = warp_id * elements_per_warp

        for i in range((elements_per_warp + 31) // 32):
            local_idx = start + lane + i * 32
            elem_idx = tile_start + local_idx
            if local_idx < start + elements_per_warp and elem_idx < n_elements:
                val = smem[local_idx]
                output_tensor[elem_idx] = val * scalar[0]

    @warp_role(WarpRole.STORER)
    @cute.jit
    def store_forward(self, logical_idx, smem, input_tensor, scalar, output_tensor, n_elements):
        pass

    def grid_fn(self, *args):
        return (self.get_logical_grid_size(*args), 1, 1)

    def block_fn(self, *args):
        return (self.warp_config.total_threads, 1, 1)


# ============================================================================
# Comprehensive Dependency and Synchronization Tests
# ============================================================================


class TestDependencyScenarios:
    """Test various dependency scenarios between operations."""

    def test_two_independent_sequential_ops(self, cuda_device):
        """Test two independent sequential ops (should allow early load)."""
        from machete.megakernel.core import Megakernel

        n = 1024
        # Two completely independent operations
        x1 = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        x2 = torch.full((n,), 3.0, dtype=torch.float16, device=cuda_device)
        s1 = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        s2 = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        y1 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        y2 = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel1 = IndependentAddKernel()
        kernel2 = IndependentMulKernel()

        mk = Megakernel(name="independent_seq_ops")
        mk.add(kernel1, x1, s1, y1, n)
        mk.add(kernel2, x2, s2, y2, n)
        mk.launch_logical(block=(256, 1, 1))

        # Verify both outputs
        torch.testing.assert_close(y1, x1 + s1[0])
        torch.testing.assert_close(y2, x2 * s2[0])

    def test_two_dependent_sequential_ops(self, cuda_device):
        """Test two sequential ops with dependency (no early load)."""
        from machete.megakernel.core import Megakernel

        n = 1024
        x = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        s1 = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        s2 = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)
        intermediate = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        out = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="dependent_seq_ops")
        mk.add(SimpleAddKernel(), x, s1, intermediate, n)  # intermediate = 2 + 1 = 3
        mk.add(SimpleMulKernel(), intermediate, s2, out, n)  # out = 3 * 3 = 9
        mk.launch_logical(block=(256, 1, 1))

        expected = torch.full((n,), 9.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(out, expected)

    def test_warp_warp_independent(self, cuda_device):
        """Test two independent warp-specialized ops."""
        from machete.megakernel.core import Megakernel

        n = 1024
        x1 = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        x2 = torch.full((n,), 4.0, dtype=torch.float16, device=cuda_device)
        s1 = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)
        s2 = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        y1 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        y2 = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel1 = WarpSpecializedTestKernel()
        kernel2 = WarpSpecIndependentKernel()

        mk = Megakernel(name="warp_warp_independent")
        mk.add(kernel1, x1, s1, y1, n)
        mk.add(kernel2, x2, s2, y2, n)

        block_size = kernel1.warp_config.total_threads
        mk.launch_logical(block=(block_size, 1, 1))

        # y1 = 2 * 3 = 6, y2 = 4 * 2 = 8
        torch.testing.assert_close(y1, torch.full((n,), 6.0, dtype=torch.float16, device=cuda_device))
        torch.testing.assert_close(y2, torch.full((n,), 8.0, dtype=torch.float16, device=cuda_device))

    def test_warp_warp_dependent(self, cuda_device):
        """Test two warp-specialized ops with dependency."""
        from machete.megakernel.core import Megakernel

        n = 1024
        x = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        s1 = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)
        s2 = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        intermediate = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        out = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel1 = WarpSpecializedTestKernel()
        kernel2 = WarpSpecializedTestKernel()

        mk = Megakernel(name="warp_warp_dependent")
        mk.add(kernel1, x, s1, intermediate, n)  # intermediate = 2 * 3 = 6
        mk.add(kernel2, intermediate, s2, out, n)  # out = 6 * 2 = 12
        mk.launch_logical(block=(kernel1.warp_config.total_threads, 1, 1))

        expected = torch.full((n,), 12.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(out, expected)

    def test_four_stage_alternating_modes(self, cuda_device):
        """Test four-stage pipeline: seq -> warp -> seq -> warp."""
        from machete.megakernel.core import Megakernel

        n = 1024
        x = torch.full((n,), 1.0, dtype=torch.float16, device=cuda_device)
        s1 = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)  # add
        s2 = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)  # mul (warp)
        s3 = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)  # add
        s4 = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)  # mul (warp)
        t1 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        t2 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        t3 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        out = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        warp_kernel = WarpSpecializedTestKernel()

        mk = Megakernel(name="four_stage_alternating")
        mk.add(SimpleAddKernel(), x, s1, t1, n)  # t1 = 1 + 1 = 2
        mk.add(warp_kernel, t1, s2, t2, n)  # t2 = 2 * 2 = 4
        mk.add(SimpleAddKernel(), t2, s3, t3, n)  # t3 = 4 + 3 = 7
        mk.add(warp_kernel, t3, s4, out, n)  # out = 7 * 2 = 14
        mk.launch_logical(block=(warp_kernel.warp_config.total_threads, 1, 1))

        expected = torch.full((n,), 14.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(out, expected)


class TestEarlyLoadOptimization:
    """Test early load optimization scenarios."""

    def test_early_load_enabled_for_independent(self, cuda_device):
        """Verify early load is enabled when ops are independent."""
        from machete.megakernel.core import Megakernel

        n = 1024
        x1 = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        x2 = torch.full((n,), 3.0, dtype=torch.float16, device=cuda_device)
        s1 = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        s2 = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        y1 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        y2 = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel1 = IndependentAddKernel()
        kernel2 = IndependentMulKernel()

        mk = Megakernel(name="early_load_test")
        mk.add(kernel1, x1, s1, y1, n)
        mk.add(kernel2, x2, s2, y2, n)

        # Build graph and check early load eligibility
        graph = mk._build_graph()
        assert graph.can_move_load_early(1), "Op 1 should allow early load (no dependency)"

        # Execute to verify correctness
        mk.launch_logical(block=(256, 1, 1))
        torch.testing.assert_close(y1, x1 + s1[0])
        torch.testing.assert_close(y2, x2 * s2[0])

    def test_early_load_disabled_for_dependent(self, cuda_device):
        """Verify early load is disabled when ops have dependency."""
        from machete.megakernel.core import Megakernel

        n = 1024
        x = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        s1 = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        s2 = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)
        intermediate = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        out = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="no_early_load_test")
        mk.add(SimpleAddKernel(), x, s1, intermediate, n)
        mk.add(SimpleMulKernel(), intermediate, s2, out, n)

        # Build graph and check early load eligibility
        graph = mk._build_graph()
        assert not graph.can_move_load_early(1), "Op 1 should NOT allow early load (has dependency)"

        # Execute to verify correctness
        mk.launch_logical(block=(256, 1, 1))
        expected = torch.full((n,), 9.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(out, expected)


class TestSmemSyncScenarios:
    """Test shared memory synchronization in various scenarios."""

    def test_smem_kernel_then_sequential(self, cuda_device):
        """Test smem kernel followed by sequential kernel."""
        from machete.megakernel.core import Megakernel

        n = 1024
        x = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        s = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        intermediate = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        out = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        smem_kernel = SharedMemKernel()
        add_kernel = SimpleAddKernel()

        mk = Megakernel(name="smem_then_seq")
        mk.add(smem_kernel, x, intermediate, n)  # intermediate = 2 * 2 = 4 (via smem)
        mk.add(add_kernel, intermediate, s, out, n)  # out = 4 + 1 = 5
        mk.launch_logical(block=(256, 1, 1))

        expected = torch.full((n,), 5.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(out, expected)

    def test_sequential_then_smem_kernel(self, cuda_device):
        """Test sequential kernel followed by smem kernel."""
        from machete.megakernel.core import Megakernel

        n = 1024
        x = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        s = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        intermediate = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        out = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        add_kernel = SimpleAddKernel()
        smem_kernel = SharedMemKernel()

        mk = Megakernel(name="seq_then_smem")
        mk.add(add_kernel, x, s, intermediate, n)  # intermediate = 2 + 1 = 3
        mk.add(smem_kernel, intermediate, out, n)  # out = 3 * 2 = 6 (via smem)
        mk.launch_logical(block=(256, 1, 1))

        expected = torch.full((n,), 6.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(out, expected)

    def test_smem_then_warp_specialized(self, cuda_device):
        """Test smem kernel followed by warp-specialized kernel."""
        from machete.megakernel.core import Megakernel

        n = 1024
        x = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        s = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)
        intermediate = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        out = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        smem_kernel = SharedMemKernel()
        warp_kernel = WarpSpecializedTestKernel()

        mk = Megakernel(name="smem_then_warp")
        mk.add(smem_kernel, x, intermediate, n)  # intermediate = 2 * 2 = 4 (via smem)
        mk.add(warp_kernel, intermediate, s, out, n)  # out = 4 * 3 = 12
        mk.launch_logical(block=(warp_kernel.warp_config.total_threads, 1, 1))

        expected = torch.full((n,), 12.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(out, expected)


class TestConcurrencyStress:
    """Stress tests for concurrent block execution with dependencies."""

    def test_many_blocks_pipeline(self, cuda_device):
        """Test pipeline with many logical blocks to stress inter-block sync."""
        from machete.megakernel.core import Megakernel

        n = 16 * 1024  # 64 logical blocks
        x = torch.full((n,), 1.0, dtype=torch.float16, device=cuda_device)
        s1 = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        s2 = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        intermediate = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        out = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="many_blocks_pipeline")
        mk.add(SimpleAddKernel(), x, s1, intermediate, n)  # intermediate = 1 + 1 = 2
        mk.add(SimpleMulKernel(), intermediate, s2, out, n)  # out = 2 * 2 = 4
        mk.launch_logical(block=(256, 1, 1))

        expected = torch.full((n,), 4.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(out, expected)

    def test_many_ops_chain(self, cuda_device):
        """Test long chain of dependent operations."""
        from machete.megakernel.core import Megakernel

        n = 1024
        num_ops = 8
        tensors = [torch.full((n,), 1.0, dtype=torch.float16, device=cuda_device)]
        scalars = [torch.tensor([1.0], dtype=torch.float16, device=cuda_device) for _ in range(num_ops)]

        for _ in range(num_ops):
            tensors.append(torch.zeros(n, dtype=torch.float16, device=cuda_device))

        mk = Megakernel(name="many_ops_chain")
        for i in range(num_ops):
            mk.add(SimpleAddKernel(), tensors[i], scalars[i], tensors[i + 1], n)
        mk.launch_logical(block=(256, 1, 1))

        # 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 9
        expected = torch.full((n,), 1.0 + num_ops, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(tensors[-1], expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
