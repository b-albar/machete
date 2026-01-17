# Copyright (c) 2025, Machete Authors
"""
Tests for shared memory synchronization with full L/C/S pipeline.

These tests verify correct synchronization when data flows through all three
L/C/S phases with shared memory. The kernels are intentionally simple - the
goal is to test synchronization and paging, not kernel logic.

Test progression (simple to complex):
1. Single kernel with L->C->S through smem
2. Single kernel with all threads participating
3. Two fused kernels with smem
4. Three fused kernels with smem
5. Warp-specialized kernel with smem
6. Multiple blocks stress test
7. Repeated execution for race condition detection
"""

import pytest
import cutlass.cute as cute

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

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")

# Note: cuda_device and trace_file fixtures are defined in conftest.py


# ============================================================================
# Simple Test Kernels - Focus on L/C/S data flow, not compute logic
# ============================================================================


class SmemDoubleKernel(FusableKernel):
    """Simplest L/C/S kernel: Load to smem, double in compute, store from smem.

    Data flow:
    - Load: global[i] -> smem[i]
    - Compute: smem[i] = smem[i] * 2
    - Store: smem[i] -> global[i]
    """

    TILE_SIZE = 256

    def __init__(self, dtype=cute.Float16):
        self.cute_dtype = dtype

    @property
    def smem_size(self) -> int:
        return self.TILE_SIZE * 2  # fp16

    def get_logical_grid_size(self, input_t, output_t, n) -> int:
        return (n + self.TILE_SIZE - 1) // self.TILE_SIZE

    @reads("input")
    @cute.jit
    def load_forward(self, logical_idx, smem, input_t, output_t, n):
        tidx, _, _ = cute.arch.thread_idx()
        idx = logical_idx * self.TILE_SIZE + tidx
        if idx < n and tidx < self.TILE_SIZE:
            smem[tidx] = input_t[idx]

    @writes("smem")
    @cute.jit
    def compute_forward(self, logical_idx, smem, input_t, output_t, n):
        tidx, _, _ = cute.arch.thread_idx()
        idx = logical_idx * self.TILE_SIZE + tidx
        if idx < n and tidx < self.TILE_SIZE:
            smem[tidx] = smem[tidx] + smem[tidx]  # Double

    @writes("output")
    @cute.jit
    def store_forward(self, logical_idx, smem, input_t, output_t, n):
        tidx, _, _ = cute.arch.thread_idx()
        idx = logical_idx * self.TILE_SIZE + tidx
        if idx < n and tidx < self.TILE_SIZE:
            output_t[idx] = smem[tidx]


class SmemAddScalarKernel(FusableKernel):
    """L/C/S kernel: Load to smem, add scalar in compute, store from smem."""

    TILE_SIZE = 256

    def __init__(self, dtype=cute.Float16):
        self.cute_dtype = dtype

    @property
    def smem_size(self) -> int:
        return self.TILE_SIZE * 2

    def get_logical_grid_size(self, input_t, scalar, output_t, n) -> int:
        return (n + self.TILE_SIZE - 1) // self.TILE_SIZE

    @reads("input")
    @cute.jit
    def load_forward(self, logical_idx, smem, input_t, scalar, output_t, n):
        tidx, _, _ = cute.arch.thread_idx()
        idx = logical_idx * self.TILE_SIZE + tidx
        if idx < n and tidx < self.TILE_SIZE:
            smem[tidx] = input_t[idx]

    @reads("scalar")
    @writes("smem")
    @cute.jit
    def compute_forward(self, logical_idx, smem, input_t, scalar, output_t, n):
        tidx, _, _ = cute.arch.thread_idx()
        idx = logical_idx * self.TILE_SIZE + tidx
        if idx < n and tidx < self.TILE_SIZE:
            smem[tidx] = smem[tidx] + scalar[0]

    @writes("output")
    @cute.jit
    def store_forward(self, logical_idx, smem, input_t, scalar, output_t, n):
        tidx, _, _ = cute.arch.thread_idx()
        idx = logical_idx * self.TILE_SIZE + tidx
        if idx < n and tidx < self.TILE_SIZE:
            output_t[idx] = smem[tidx]


class SmemMulScalarKernel(FusableKernel):
    """L/C/S kernel: Load to smem, multiply by scalar in compute, store from smem."""

    TILE_SIZE = 256

    def __init__(self, dtype=cute.Float16):
        self.cute_dtype = dtype

    @property
    def smem_size(self) -> int:
        return self.TILE_SIZE * 2

    def get_logical_grid_size(self, input_t, scalar, output_t, n) -> int:
        return (n + self.TILE_SIZE - 1) // self.TILE_SIZE

    @reads("input")
    @cute.jit
    def load_forward(self, logical_idx, smem, input_t, scalar, output_t, n):
        tidx, _, _ = cute.arch.thread_idx()
        idx = logical_idx * self.TILE_SIZE + tidx
        if idx < n and tidx < self.TILE_SIZE:
            smem[tidx] = input_t[idx]

    @reads("scalar")
    @writes("smem")
    @cute.jit
    def compute_forward(self, logical_idx, smem, input_t, scalar, output_t, n):
        tidx, _, _ = cute.arch.thread_idx()
        idx = logical_idx * self.TILE_SIZE + tidx
        if idx < n and tidx < self.TILE_SIZE:
            smem[tidx] = smem[tidx] * scalar[0]

    @writes("output")
    @cute.jit
    def store_forward(self, logical_idx, smem, input_t, scalar, output_t, n):
        tidx, _, _ = cute.arch.thread_idx()
        idx = logical_idx * self.TILE_SIZE + tidx
        if idx < n and tidx < self.TILE_SIZE:
            output_t[idx] = smem[tidx]


class WarpSpecSmemKernel(WarpSpecializedKernel):
    """Warp-specialized kernel with smem through L/C/S."""

    TILE_SIZE = 256

    def __init__(self, dtype=cute.Float16):
        self.cute_dtype = dtype

    @property
    def warp_config(self) -> WarpConfig:
        return WarpConfig(num_consumer_warps=4)

    @property
    def smem_size(self) -> int:
        return self.TILE_SIZE * 2

    def get_logical_grid_size(self, input_t, scalar, output_t, n) -> int:
        return (n + self.TILE_SIZE - 1) // self.TILE_SIZE

    @warp_role(WarpRole.LOADER)
    @reads("input")
    @cute.jit
    def load_forward(self, logical_idx, smem, input_t, scalar, output_t, n):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        base = logical_idx * self.TILE_SIZE
        # Loader warp loads all elements
        for i in range(self.TILE_SIZE // 32):
            idx = base + lane + i * 32
            smem_idx = lane + i * 32
            if idx < n and smem_idx < self.TILE_SIZE:
                smem[smem_idx] = input_t[idx]

    @warp_role(WarpRole.CONSUMER)
    @reads("scalar")
    @writes("smem")
    @cute.jit
    def compute_forward(self, logical_idx, smem, input_t, scalar, output_t, n):
        tidx, _, _ = cute.arch.thread_idx()
        warp_id = tidx // 32
        lane = tidx % 32
        base = logical_idx * self.TILE_SIZE
        # Each consumer warp handles a portion
        elems_per_warp = self.TILE_SIZE // 4
        start = warp_id * elems_per_warp
        for i in range((elems_per_warp + 31) // 32):
            local = start + lane + i * 32
            idx = base + local
            if local < start + elems_per_warp and idx < n:
                smem[local] = smem[local] * scalar[0]

    @warp_role(WarpRole.STORER)
    @writes("output")
    @cute.jit
    def store_forward(self, logical_idx, smem, input_t, scalar, output_t, n):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        base = logical_idx * self.TILE_SIZE
        # Storer warp writes all elements
        for i in range(self.TILE_SIZE // 32):
            idx = base + lane + i * 32
            smem_idx = lane + i * 32
            if idx < n and smem_idx < self.TILE_SIZE:
                output_t[idx] = smem[smem_idx]


# ============================================================================
# Test Classes - Simple to Complex
# ============================================================================


class TestLevel1_SingleKernelBasic:
    """Level 1: Single kernel, basic L/C/S synchronization."""

    def test_single_block(self, cuda_device):
        """Simplest case: single block, aligned size."""
        from machete.megakernel.core import Megakernel

        n = 256  # Exactly one block
        x = torch.randn(n, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l1_single_block")
        mk.add(SmemDoubleKernel(), x, y, n)
        mk.launch_logical(block=(256, 1, 1))

        torch.testing.assert_close(y, x * 2)

    def test_two_blocks(self, cuda_device):
        """Two blocks: verify inter-block independence."""
        from machete.megakernel.core import Megakernel

        n = 512  # Two blocks
        x = torch.randn(n, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l1_two_blocks")
        mk.add(SmemDoubleKernel(), x, y, n)
        mk.launch_logical(block=(256, 1, 1))

        torch.testing.assert_close(y, x * 2)

    def test_non_aligned_size(self, cuda_device):
        """Non-aligned size: boundary handling."""
        from machete.megakernel.core import Megakernel

        n = 300  # Not divisible by 256
        x = torch.randn(n, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l1_non_aligned")
        mk.add(SmemDoubleKernel(), x, y, n)
        mk.launch_logical(block=(256, 1, 1))

        torch.testing.assert_close(y, x * 2)


class TestLevel2_SingleKernelWithScalar:
    """Level 2: Single kernel with scalar parameter."""

    def test_add_scalar(self, cuda_device):
        """Add scalar through smem pipeline."""
        from machete.megakernel.core import Megakernel

        n = 512
        x = torch.randn(n, dtype=torch.float16, device=cuda_device)
        s = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l2_add_scalar")
        mk.add(SmemAddScalarKernel(), x, s, y, n)
        mk.launch_logical(block=(256, 1, 1))

        torch.testing.assert_close(y, x + s[0])

    def test_mul_scalar(self, cuda_device):
        """Multiply by scalar through smem pipeline."""
        from machete.megakernel.core import Megakernel

        n = 512
        x = torch.randn(n, dtype=torch.float16, device=cuda_device)
        s = torch.tensor([2.5], dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l2_mul_scalar")
        mk.add(SmemMulScalarKernel(), x, s, y, n)
        mk.launch_logical(block=(256, 1, 1))

        torch.testing.assert_close(y, x * s[0], rtol=1e-3, atol=1e-3)


class TestLevel3_TwoKernelsFused:
    """Level 3: Two fused kernels, both using smem."""

    def test_double_then_add(self, cuda_device):
        """Pipeline: x*2 -> +scalar."""
        from machete.megakernel.core import Megakernel

        n = 512
        x = torch.randn(n, dtype=torch.float16, device=cuda_device)
        s = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        tmp = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l3_double_add")
        mk.add(SmemDoubleKernel(), x, tmp, n)
        mk.add(SmemAddScalarKernel(), tmp, s, y, n)
        mk.launch_logical(block=(256, 1, 1))

        expected = x * 2 + s[0]
        torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-3)

    def test_add_then_mul(self, cuda_device):
        """Pipeline: x+a -> *b."""
        from machete.megakernel.core import Megakernel

        n = 512
        x = torch.ones(n, dtype=torch.float16, device=cuda_device)
        a = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        b = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)
        tmp = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l3_add_mul")
        mk.add(SmemAddScalarKernel(), x, a, tmp, n)
        mk.add(SmemMulScalarKernel(), tmp, b, y, n)
        mk.launch_logical(block=(256, 1, 1))

        # (1 + 2) * 3 = 9
        expected = torch.full((n,), 9.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(y, expected)

    def test_mul_then_add(self, cuda_device):
        """Pipeline: x*a -> +b."""
        from machete.megakernel.core import Megakernel

        n = 512
        x = torch.full((n,), 4.0, dtype=torch.float16, device=cuda_device)
        a = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        b = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        tmp = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l3_mul_add")
        mk.add(SmemMulScalarKernel(), x, a, tmp, n)
        mk.add(SmemAddScalarKernel(), tmp, b, y, n)
        mk.launch_logical(block=(256, 1, 1))

        # 4 * 2 + 1 = 9
        expected = torch.full((n,), 9.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(y, expected)


class TestLevel4_ThreeKernelsFused:
    """Level 4: Three fused kernels, complex pipeline."""

    def test_three_stage_pipeline(self, cuda_device):
        """Pipeline: x*2 -> +a -> *b."""
        from machete.megakernel.core import Megakernel

        n = 512
        x = torch.ones(n, dtype=torch.float16, device=cuda_device)
        a = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        b = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        t1 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        t2 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l4_three_stage")
        mk.add(SmemDoubleKernel(), x, t1, n)  # t1 = 1*2 = 2
        mk.add(SmemAddScalarKernel(), t1, a, t2, n)  # t2 = 2+1 = 3
        mk.add(SmemMulScalarKernel(), t2, b, y, n)  # y = 3*2 = 6
        mk.launch_logical(block=(256, 1, 1))

        expected = torch.full((n,), 6.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(y, expected)

    def test_four_stage_pipeline(self, cuda_device):
        """Pipeline: x*2 -> *a -> +b -> *c."""
        from machete.megakernel.core import Megakernel

        n = 512
        x = torch.ones(n, dtype=torch.float16, device=cuda_device)
        a = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)
        b = torch.tensor([2.0], dtype=torch.float16, device=cuda_device)
        c = torch.tensor([0.5], dtype=torch.float16, device=cuda_device)
        t1 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        t2 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        t3 = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l4_four_stage")
        mk.add(SmemDoubleKernel(), x, t1, n)  # t1 = 1*2 = 2
        mk.add(SmemMulScalarKernel(), t1, a, t2, n)  # t2 = 2*3 = 6
        mk.add(SmemAddScalarKernel(), t2, b, t3, n)  # t3 = 6+2 = 8
        mk.add(SmemMulScalarKernel(), t3, c, y, n)  # y = 8*0.5 = 4
        mk.launch_logical(block=(256, 1, 1))

        expected = torch.full((n,), 4.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(y, expected)


class TestLevel5_WarpSpecialized:
    """Level 5: Warp-specialized kernel with smem."""

    def test_warp_specialized_basic(self, cuda_device):
        """Basic warp-specialized execution."""
        from machete.megakernel.core import Megakernel

        n = 512
        x = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        s = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel = WarpSpecSmemKernel()
        mk = Megakernel(name="l5_warp_spec")
        mk.add(kernel, x, s, y, n)
        mk.launch_logical(block=(kernel.warp_config.total_threads, 1, 1))

        # 2 * 3 = 6
        expected = torch.full((n,), 6.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(y, expected)

    def test_warp_specialized_random(self, cuda_device):
        """Warp-specialized with random values."""
        from machete.megakernel.core import Megakernel

        n = 1024
        x = torch.randn(n, dtype=torch.float16, device=cuda_device)
        s = torch.randn(1, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel = WarpSpecSmemKernel()
        mk = Megakernel(name="l5_warp_random")
        mk.add(kernel, x, s, y, n)
        mk.launch_logical(block=(kernel.warp_config.total_threads, 1, 1))

        torch.testing.assert_close(y, x * s[0], rtol=1e-3, atol=1e-3)


class TestLevel6_ManyBlocks:
    """Level 6: Many blocks stress test."""

    def test_16_blocks(self, cuda_device):
        """16 blocks execution."""
        from machete.megakernel.core import Megakernel

        n = 256 * 16
        x = torch.randn(n, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l6_16_blocks")
        mk.add(SmemDoubleKernel(), x, y, n)
        mk.launch_logical(block=(256, 1, 1))

        torch.testing.assert_close(y, x * 2, rtol=1e-3, atol=1e-3)

    def test_64_blocks(self, cuda_device):
        """64 blocks execution."""
        from machete.megakernel.core import Megakernel

        n = 256 * 64
        x = torch.randn(n, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l6_64_blocks")
        mk.add(SmemDoubleKernel(), x, y, n)
        mk.launch_logical(block=(256, 1, 1))

        torch.testing.assert_close(y, x * 2, rtol=1e-3, atol=1e-3)

    def test_256_blocks(self, cuda_device):
        """256 blocks execution."""
        from machete.megakernel.core import Megakernel

        n = 256 * 256
        x = torch.randn(n, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l6_256_blocks")
        mk.add(SmemDoubleKernel(), x, y, n)
        mk.launch_logical(block=(256, 1, 1))

        torch.testing.assert_close(y, x * 2, rtol=1e-3, atol=1e-3)

    def test_many_blocks_fused(self, cuda_device):
        """Many blocks with fused kernels."""
        from machete.megakernel.core import Megakernel

        n = 256 * 32
        x = torch.ones(n, dtype=torch.float16, device=cuda_device)
        s = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        t = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l6_many_fused")
        mk.add(SmemDoubleKernel(), x, t, n)
        mk.add(SmemAddScalarKernel(), t, s, y, n)
        mk.launch_logical(block=(256, 1, 1))

        # 1*2 + 1 = 3
        expected = torch.full((n,), 3.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(y, expected)


class TestLevel7_StressAndRace:
    """Level 7: Stress tests for race condition detection."""

    def test_repeated_execution_10x(self, cuda_device):
        """Repeat 10 times to catch intermittent races."""
        from machete.megakernel.core import Megakernel

        n = 1024
        for i in range(10):
            x = torch.randn(n, dtype=torch.float16, device=cuda_device)
            y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

            mk = Megakernel(name=f"l7_repeat_{i}")
            mk.add(SmemDoubleKernel(), x, y, n)
            mk.launch_logical(block=(256, 1, 1))

            torch.testing.assert_close(y, x * 2, msg=f"Failed on iteration {i}")

    def test_repeated_fused_10x(self, cuda_device):
        """Repeat fused pipeline 10 times."""
        from machete.megakernel.core import Megakernel

        n = 1024
        for i in range(10):
            x = torch.randn(n, dtype=torch.float16, device=cuda_device)
            s = torch.randn(1, dtype=torch.float16, device=cuda_device)
            t = torch.zeros(n, dtype=torch.float16, device=cuda_device)
            y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

            mk = Megakernel(name=f"l7_fused_{i}")
            mk.add(SmemDoubleKernel(), x, t, n)
            mk.add(SmemMulScalarKernel(), t, s, y, n)
            mk.launch_logical(block=(256, 1, 1))

            expected = x * 2 * s[0]
            torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-3, msg=f"Failed on iteration {i}")

    def test_alternating_kernels(self, cuda_device):
        """Alternate between different kernel types."""
        from machete.megakernel.core import Megakernel

        n = 512
        x = torch.ones(n, dtype=torch.float16, device=cuda_device)

        for i in range(5):
            s = torch.tensor([float(i + 1)], dtype=torch.float16, device=cuda_device)
            y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

            if i % 2 == 0:
                mk = Megakernel(name=f"l7_alt_add_{i}")
                mk.add(SmemAddScalarKernel(), x, s, y, n)
                expected = x + s[0]
            else:
                mk = Megakernel(name=f"l7_alt_mul_{i}")
                mk.add(SmemMulScalarKernel(), x, s, y, n)
                expected = x * s[0]

            mk.launch_logical(block=(256, 1, 1))
            torch.testing.assert_close(y, expected, msg=f"Failed on iteration {i}")


class TestLevel8_EdgeCases:
    """Level 8: Edge cases and boundary conditions."""

    def test_single_element(self, cuda_device):
        """Single element tensor."""
        from machete.megakernel.core import Megakernel

        n = 1
        x = torch.tensor([5.0], dtype=torch.float16, device=cuda_device)
        y = torch.zeros(1, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l8_single_elem")
        mk.add(SmemDoubleKernel(), x, y, n)
        mk.launch_logical(block=(256, 1, 1))

        torch.testing.assert_close(y, x * 2)

    def test_prime_size(self, cuda_device):
        """Prime number size (worst case for alignment)."""
        from machete.megakernel.core import Megakernel

        n = 997  # Prime
        x = torch.randn(n, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l8_prime")
        mk.add(SmemDoubleKernel(), x, y, n)
        mk.launch_logical(block=(256, 1, 1))

        torch.testing.assert_close(y, x * 2)

    def test_zero_scalar(self, cuda_device):
        """Zero scalar multiplication."""
        from machete.megakernel.core import Megakernel

        n = 512
        x = torch.randn(n, dtype=torch.float16, device=cuda_device)
        s = torch.tensor([0.0], dtype=torch.float16, device=cuda_device)
        y = torch.ones(n, dtype=torch.float16, device=cuda_device)  # Non-zero init

        mk = Megakernel(name="l8_zero_scalar")
        mk.add(SmemMulScalarKernel(), x, s, y, n)
        mk.launch_logical(block=(256, 1, 1))

        expected = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(y, expected)

    def test_negative_values(self, cuda_device):
        """Negative input values."""
        from machete.megakernel.core import Megakernel

        n = 512
        x = -torch.ones(n, dtype=torch.float16, device=cuda_device)
        s = torch.tensor([-2.0], dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="l8_negative")
        mk.add(SmemMulScalarKernel(), x, s, y, n)
        mk.launch_logical(block=(256, 1, 1))

        # (-1) * (-2) = 2
        expected = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(y, expected)


class TestTraceExport:
    """Trace export for debugging synchronization issues.

    These tests export .nanotrace files that can be viewed with cutedsl-trace tools.
    The trace includes fine-grained synchronization events (semaphore waits/signals)
    in addition to L/C/S phase timing.

    To export traces to a persistent directory:
        MACHETE_TRACE_DIR=/path/to/traces pytest tests/megakernel/test_smem_synchronization.py

    Or use pytest options:
        pytest --trace-kernels --trace-dir=/path/to/traces tests/megakernel/
    """

    @pytest.mark.trace
    def test_single_kernel_trace(self, cuda_device, trace_file):
        """Export trace for single smem kernel."""
        from machete.megakernel.core import Megakernel

        n = 256
        x = torch.randn(n, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="trace_single_smem")
        mk.add(SmemDoubleKernel(), x, y, n)
        mk.launch_logical(block=(256, 1, 1), trace_file=trace_file)

        # Verify trace file was created (with .nanotrace extension)
        from pathlib import Path

        trace_path = Path(trace_file)
        # The core.py ensures .nanotrace extension
        actual_path = trace_path.with_suffix(".nanotrace")
        assert actual_path.exists(), f"Trace file not found: {actual_path}"
        torch.testing.assert_close(y, x * 2)

    @pytest.mark.trace
    def test_fused_trace(self, cuda_device, trace_file):
        """Export trace for fused smem kernels."""
        from machete.megakernel.core import Megakernel

        n = 256
        x = torch.ones(n, dtype=torch.float16, device=cuda_device)
        s = torch.tensor([1.0], dtype=torch.float16, device=cuda_device)
        t = torch.zeros(n, dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        mk = Megakernel(name="trace_fused_smem")
        mk.add(SmemDoubleKernel(), x, t, n)
        mk.add(SmemAddScalarKernel(), t, s, y, n)
        mk.launch_logical(block=(256, 1, 1), trace_file=trace_file)

        from pathlib import Path

        actual_path = Path(trace_file).with_suffix(".nanotrace")
        assert actual_path.exists(), f"Trace file not found: {actual_path}"
        expected = torch.full((n,), 3.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(y, expected)

    @pytest.mark.trace
    def test_warp_spec_trace(self, cuda_device, trace_file):
        """Export trace for warp-specialized smem kernel."""
        from machete.megakernel.core import Megakernel

        n = 256
        x = torch.full((n,), 2.0, dtype=torch.float16, device=cuda_device)
        s = torch.tensor([3.0], dtype=torch.float16, device=cuda_device)
        y = torch.zeros(n, dtype=torch.float16, device=cuda_device)

        kernel = WarpSpecSmemKernel()
        mk = Megakernel(name="trace_warp_smem")
        mk.add(kernel, x, s, y, n)
        mk.launch_logical(block=(kernel.warp_config.total_threads, 1, 1), trace_file=trace_file)

        from pathlib import Path

        actual_path = Path(trace_file).with_suffix(".nanotrace")
        assert actual_path.exists(), f"Trace file not found: {actual_path}"
        expected = torch.full((n,), 6.0, dtype=torch.float16, device=cuda_device)
        torch.testing.assert_close(y, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
