# Copyright (c) 2025, Machete Authors
"""
Tests for example Megakernel implementations using the Logical Blocks API.

This module tests the example kernels that demonstrate fusable kernels
combined in a megakernel pipeline:
1. AddKernel: output = input + bias
2. MulKernel: output = input * scale
3. RoPEExampleKernel: 3D logical coordinates example
4. WarpSpecializedAddMul: No Bubbles pattern example
"""

import pytest
from typing import Tuple
import cutlass.cute as cute

from machete.megakernel.interface import (
    FusableKernel,
    WarpSpecializedKernel,
    WarpConfig,
    WarpRole,
    reads,
    writes,
    warp_role,
    LogicalGridInfo,
)


# ============================================================================
# Example Kernel Definitions
# ============================================================================


class AddKernel(FusableKernel):
    """Example kernel: output = input + bias.

    Uses Logical Blocks to process data in tiles of TILE_SIZE elements.
    The logical coordinate is a simple 1D chunk index.
    """

    TILE_SIZE = 256  # Elements per logical block

    def __init__(self, dtype=cute.Float16):
        self.cute_dtype = dtype

    @property
    def smem_size(self) -> int:
        """No shared memory needed for this simple kernel."""
        return 0

    # ========== Logical Blocks API ==========

    def get_logical_grid_size(self, input_tensor, bias, output_tensor) -> int:
        """Return total number of logical blocks (tiles) to process."""
        total_elements = input_tensor.numel()
        return (total_elements + self.TILE_SIZE - 1) // self.TILE_SIZE

    def get_logical_coord(self, logical_idx: int, input_tensor, bias, output_tensor) -> Tuple[int, ...]:
        """Map logical index to (chunk_idx,) coordinate."""
        return (logical_idx,)

    def get_logical_coord_names(self) -> Tuple[str, ...]:
        return ("chunk",)

    def get_logical_grid_info(self, input_tensor, bias, output_tensor) -> LogicalGridInfo:
        grid_size = self.get_logical_grid_size(input_tensor, bias, output_tensor)
        return LogicalGridInfo(
            logical_grid_size=grid_size,
            coord_names=("chunk",),
            coord_dims=(grid_size,),
        )

    # ========== L/C/S Methods ==========

    @reads("input", "bias")
    @cute.jit
    def load_forward(self, paged_pool, page_idx, logical_idx, input_tensor, bias, output_tensor):
        """Load phase - no-op for this kernel (direct global memory access)."""
        pass

    @writes("output")
    @cute.jit
    def compute_forward(self, logical_idx, input_tensor, bias, output_tensor):
        """Compute: output = input + bias."""
        tidx, _, _ = cute.arch.thread_idx()

        tile_start = logical_idx * self.TILE_SIZE
        elem_idx = tile_start + tidx

        total_elements = input_tensor.size()
        if elem_idx < total_elements:
            val = input_tensor[elem_idx]
            b = bias[elem_idx % bias.size()]
            output_tensor[elem_idx] = val + b

    @cute.jit
    def store_forward(self, paged_pool, page_idx, logical_idx, input_tensor, bias, output_tensor):
        """Store phase - no-op (results written directly in compute)."""
        pass

    def grid_fn(self, input_tensor, bias, output_tensor):
        grid_size = self.get_logical_grid_size(input_tensor, bias, output_tensor)
        return (grid_size, 1, 1)

    def block_fn(self, input_tensor, bias, output_tensor):
        return (self.TILE_SIZE, 1, 1)


class MulKernel(FusableKernel):
    """Example kernel: output = input * scale.

    Uses Logical Blocks with a 2D interpretation: (batch, chunk).
    """

    TILE_SIZE = 256

    def __init__(self, dtype=cute.Float16):
        self.cute_dtype = dtype

    @property
    def smem_size(self) -> int:
        return 0

    def get_logical_grid_size(self, input_tensor, scale, output_tensor, batch_size: int = 1) -> int:
        """Return total logical blocks: batch_size * chunks_per_batch."""
        elements_per_batch = input_tensor.numel() // batch_size
        chunks_per_batch = (elements_per_batch + self.TILE_SIZE - 1) // self.TILE_SIZE
        return batch_size * chunks_per_batch

    def get_logical_coord(
        self, logical_idx: int, input_tensor, scale, output_tensor, batch_size: int = 1
    ) -> Tuple[int, int]:
        """Map logical index to (batch_idx, chunk_idx) coordinate."""
        elements_per_batch = input_tensor.numel() // batch_size
        chunks_per_batch = (elements_per_batch + self.TILE_SIZE - 1) // self.TILE_SIZE

        batch_idx = logical_idx // chunks_per_batch
        chunk_idx = logical_idx % chunks_per_batch
        return (batch_idx, chunk_idx)

    def get_logical_coord_names(self) -> Tuple[str, ...]:
        return ("batch", "chunk")

    @reads("input", "scale")
    @cute.jit
    def load_forward(self, paged_pool, page_idx, logical_idx, input_tensor, scale, output_tensor, batch_size: int = 1):
        pass

    @writes("output")
    @cute.jit
    def compute_forward(self, logical_idx, input_tensor, scale, output_tensor, batch_size: int = 1):
        """Compute: output = input * scale."""
        tidx, _, _ = cute.arch.thread_idx()

        total_elements = input_tensor.size()
        elements_per_batch = total_elements // batch_size
        chunks_per_batch = (elements_per_batch + self.TILE_SIZE - 1) // self.TILE_SIZE

        batch_idx = logical_idx // chunks_per_batch
        chunk_idx = logical_idx % chunks_per_batch

        tile_start = batch_idx * elements_per_batch + chunk_idx * self.TILE_SIZE
        elem_idx = tile_start + tidx

        batch_end = (batch_idx + 1) * elements_per_batch
        if elem_idx < batch_end and elem_idx < total_elements:
            val = input_tensor[elem_idx]
            s = scale[0]
            output_tensor[elem_idx] = val * s

    @cute.jit
    def store_forward(self, paged_pool, page_idx, logical_idx, input_tensor, scale, output_tensor, batch_size: int = 1):
        pass

    def grid_fn(self, input_tensor, scale, output_tensor, batch_size: int = 1):
        grid_size = self.get_logical_grid_size(input_tensor, scale, output_tensor, batch_size)
        return (grid_size, 1, 1)

    def block_fn(self, input_tensor, scale, output_tensor, batch_size: int = 1):
        return (self.TILE_SIZE, 1, 1)


class RoPEExampleKernel(FusableKernel):
    """Example RoPE-style kernel demonstrating 3D logical coordinates.

    Logical coordinates: (batch, seq_chunk, head)
    """

    TILE_SIZE = 64

    def __init__(self, dtype=cute.Float16, head_dim: int = 64):
        self.cute_dtype = dtype
        self.head_dim = head_dim

    @property
    def smem_size(self) -> int:
        return self.head_dim * 2 * 2

    # needs_block_sync inherited from FusableKernel (returns True by default)

    def get_logical_grid_size(self, q_tensor, k_tensor, cos, sin, batch: int, seq_len: int, num_heads: int) -> int:
        seq_chunks = (seq_len + self.TILE_SIZE - 1) // self.TILE_SIZE
        return batch * seq_chunks * num_heads

    def get_logical_coord(
        self, logical_idx: int, q_tensor, k_tensor, cos, sin, batch: int, seq_len: int, num_heads: int
    ) -> Tuple[int, int, int]:
        seq_chunks = (seq_len + self.TILE_SIZE - 1) // self.TILE_SIZE
        head_idx = logical_idx % num_heads
        seq_chunk_idx = (logical_idx // num_heads) % seq_chunks
        batch_idx = logical_idx // (num_heads * seq_chunks)
        return (batch_idx, seq_chunk_idx, head_idx)

    def get_logical_coord_names(self) -> Tuple[str, ...]:
        return ("batch", "seq_chunk", "head")

    def get_logical_grid_info(
        self, q_tensor, k_tensor, cos, sin, batch: int, seq_len: int, num_heads: int
    ) -> LogicalGridInfo:
        seq_chunks = (seq_len + self.TILE_SIZE - 1) // self.TILE_SIZE
        return LogicalGridInfo(
            logical_grid_size=self.get_logical_grid_size(q_tensor, k_tensor, cos, sin, batch, seq_len, num_heads),
            coord_names=("batch", "seq_chunk", "head"),
            coord_dims=(batch, seq_chunks, num_heads),
        )

    @reads("cos", "sin")
    @cute.jit
    def load_forward(
        self,
        paged_pool,
        page_idx,
        logical_idx,
        smem,
        q_tensor,
        k_tensor,
        cos,
        sin,
        batch: int,
        seq_len: int,
        num_heads: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        seq_chunks = (seq_len + self.TILE_SIZE - 1) // self.TILE_SIZE
        seq_chunk_idx = (logical_idx // num_heads) % seq_chunks

        half_dim = self.head_dim // 2
        if tidx < half_dim:
            seq_pos = seq_chunk_idx * self.TILE_SIZE
            if seq_pos < seq_len:
                smem[tidx] = cos[seq_pos * half_dim + tidx]
                smem[half_dim + tidx] = sin[seq_pos * half_dim + tidx]

    @writes("q", "k")
    @cute.jit
    def compute_forward(
        self, logical_idx, smem, q_tensor, k_tensor, cos, sin, batch: int, seq_len: int, num_heads: int
    ):
        tidx, _, _ = cute.arch.thread_idx()

        seq_chunks = (seq_len + self.TILE_SIZE - 1) // self.TILE_SIZE
        head_idx = logical_idx % num_heads
        seq_chunk_idx = (logical_idx // num_heads) % seq_chunks
        batch_idx = logical_idx // (num_heads * seq_chunks)

        seq_pos = seq_chunk_idx * self.TILE_SIZE + (tidx // self.head_dim)
        dim_idx = tidx % self.head_dim

        if seq_pos < seq_len and dim_idx < self.head_dim:
            half_dim = self.head_dim // 2
            cos_val = smem[dim_idx % half_dim]
            sin_val = smem[half_dim + dim_idx % half_dim]

            offset = batch_idx * seq_len * num_heads * self.head_dim
            offset += seq_pos * num_heads * self.head_dim
            offset += head_idx * self.head_dim
            offset += dim_idx

            q_val = q_tensor[offset]
            k_val = k_tensor[offset]

            if dim_idx < half_dim:
                q_rotated = q_val * cos_val - q_tensor[offset + half_dim] * sin_val
                k_rotated = k_val * cos_val - k_tensor[offset + half_dim] * sin_val
            else:
                q_rotated = q_tensor[offset - half_dim] * sin_val + q_val * cos_val
                k_rotated = k_tensor[offset - half_dim] * sin_val + k_val * cos_val

            q_tensor[offset] = q_rotated
            k_tensor[offset] = k_rotated

    @cute.jit
    def store_forward(
        self,
        paged_pool,
        page_idx,
        logical_idx,
        smem,
        q_tensor,
        k_tensor,
        cos,
        sin,
        batch: int,
        seq_len: int,
        num_heads: int,
    ):
        pass

    def grid_fn(self, q_tensor, k_tensor, cos, sin, batch: int, seq_len: int, num_heads: int):
        grid_size = self.get_logical_grid_size(q_tensor, k_tensor, cos, sin, batch, seq_len, num_heads)
        return (grid_size, 1, 1)

    def block_fn(self, q_tensor, k_tensor, cos, sin, batch: int, seq_len: int, num_heads: int):
        threads = min(self.TILE_SIZE * self.head_dim, 256)
        return (threads, 1, 1)


class WarpSpecializedAddMul(WarpSpecializedKernel):
    """Example warp-specialized kernel combining Add and Mul operations.

    Demonstrates the No Bubbles pattern where:
    - Loader warps handle data movement (Global -> Shared)
    - Consumer warps execute compute (Add + Mul)
    - Storer warps handle results (Shared -> Global)
    """

    TILE_SIZE = 256

    def __init__(self, dtype=cute.Float16):
        self.cute_dtype = dtype

    @property
    def warp_config(self) -> WarpConfig:
        return WarpConfig(num_consumer_warps=12)

    @property
    def smem_size(self) -> int:
        return self.TILE_SIZE * 2 * 2

    def get_logical_grid_size(self, input_tensor, bias, scale, output_tensor) -> int:
        total_elements = input_tensor.numel()
        return (total_elements + self.TILE_SIZE - 1) // self.TILE_SIZE

    def get_logical_coord(self, logical_idx: int, input_tensor, bias, scale, output_tensor) -> Tuple[int, ...]:
        return (logical_idx,)

    def get_logical_coord_names(self) -> Tuple[str, ...]:
        return ("chunk",)

    @warp_role(WarpRole.LOADER)
    @reads("input", "bias", "scale")
    @cute.jit
    def load_forward(self, paged_pool, page_idx, logical_idx, smem, input_tensor, bias, scale, output_tensor):
        tidx, _, _ = cute.arch.thread_idx()

        tile_start = logical_idx * self.TILE_SIZE
        lane_in_warp = tidx % 32

        elements_per_thread = self.TILE_SIZE // 32
        for i in range(elements_per_thread):
            elem_idx = tile_start + lane_in_warp * elements_per_thread + i
            if elem_idx < input_tensor.size():
                smem_idx = lane_in_warp * elements_per_thread + i
                smem[smem_idx] = input_tensor[elem_idx]

    @warp_role(WarpRole.CONSUMER)
    @writes("output")
    @cute.jit
    def compute_forward(self, logical_idx, smem, input_tensor, bias, scale, output_tensor):
        tidx, _, _ = cute.arch.thread_idx()

        warp_id = tidx // 32
        lane_in_warp = tidx % 32

        tile_start = logical_idx * self.TILE_SIZE
        total_elements = input_tensor.size()

        elements_per_warp = (self.TILE_SIZE + 11) // 12
        warp_start = warp_id * elements_per_warp

        for i in range(elements_per_warp // 32 + 1):
            local_idx = warp_start + lane_in_warp + i * 32
            elem_idx = tile_start + local_idx

            if local_idx < self.TILE_SIZE and elem_idx < total_elements:
                val = smem[local_idx]
                b = bias[elem_idx % bias.size()]
                s = scale[0]
                output_tensor[elem_idx] = (val + b) * s

    @warp_role(WarpRole.STORER)
    @cute.jit
    def store_forward(self, paged_pool, page_idx, logical_idx, smem, input_tensor, bias, scale, output_tensor):
        pass

    def grid_fn(self, input_tensor, bias, scale, output_tensor):
        grid_size = self.get_logical_grid_size(input_tensor, bias, scale, output_tensor)
        return (grid_size, 1, 1)

    def block_fn(self, input_tensor, bias, scale, output_tensor):
        return (self.warp_config.total_threads, 1, 1)


# ============================================================================
# Tests
# ============================================================================


class TestAddKernel:
    """Tests for AddKernel."""

    def test_logical_grid_size(self):
        """Test logical grid size calculation."""
        kernel = AddKernel()

        # Mock tensor with numel() method
        class MockTensor:
            def numel(self):
                return 1024

        tensor = MockTensor()
        grid_size = kernel.get_logical_grid_size(tensor, tensor, tensor)

        # 1024 elements / 256 tile size = 4 blocks
        assert grid_size == 4

    def test_logical_grid_size_non_divisible(self):
        """Test logical grid size with non-divisible element count."""
        kernel = AddKernel()

        class MockTensor:
            def numel(self):
                return 1000  # Not divisible by 256

        tensor = MockTensor()
        grid_size = kernel.get_logical_grid_size(tensor, tensor, tensor)

        # ceil(1000 / 256) = 4 blocks
        assert grid_size == 4

    def test_logical_coord(self):
        """Test logical coordinate mapping."""
        kernel = AddKernel()

        class MockTensor:
            def numel(self):
                return 1024

        tensor = MockTensor()

        # Each logical idx maps to (chunk_idx,)
        assert kernel.get_logical_coord(0, tensor, tensor, tensor) == (0,)
        assert kernel.get_logical_coord(1, tensor, tensor, tensor) == (1,)
        assert kernel.get_logical_coord(3, tensor, tensor, tensor) == (3,)

    def test_logical_coord_names(self):
        """Test logical coordinate names."""
        kernel = AddKernel()
        assert kernel.get_logical_coord_names() == ("chunk",)


class TestMulKernel:
    """Tests for MulKernel."""

    def test_logical_grid_size_single_batch(self):
        """Test logical grid size with single batch."""
        kernel = MulKernel()

        class MockTensor:
            def numel(self):
                return 512

        tensor = MockTensor()
        grid_size = kernel.get_logical_grid_size(tensor, tensor, tensor, batch_size=1)

        # 512 / 256 = 2 blocks
        assert grid_size == 2

    def test_logical_grid_size_multi_batch(self):
        """Test logical grid size with multiple batches."""
        kernel = MulKernel()

        class MockTensor:
            def numel(self):
                return 1024

        tensor = MockTensor()
        grid_size = kernel.get_logical_grid_size(tensor, tensor, tensor, batch_size=2)

        # 1024 total / 2 batches = 512 per batch
        # 512 / 256 = 2 chunks per batch
        # 2 batches * 2 chunks = 4 blocks
        assert grid_size == 4

    def test_logical_coord_2d(self):
        """Test 2D logical coordinate mapping."""
        kernel = MulKernel()

        class MockTensor:
            def numel(self):
                return 1024

        tensor = MockTensor()

        # With batch_size=2: 512 elements per batch, 2 chunks per batch
        # logical_idx 0 -> batch 0, chunk 0
        # logical_idx 1 -> batch 0, chunk 1
        # logical_idx 2 -> batch 1, chunk 0
        # logical_idx 3 -> batch 1, chunk 1
        assert kernel.get_logical_coord(0, tensor, tensor, tensor, batch_size=2) == (0, 0)
        assert kernel.get_logical_coord(1, tensor, tensor, tensor, batch_size=2) == (0, 1)
        assert kernel.get_logical_coord(2, tensor, tensor, tensor, batch_size=2) == (1, 0)
        assert kernel.get_logical_coord(3, tensor, tensor, tensor, batch_size=2) == (1, 1)

    def test_logical_coord_names(self):
        """Test 2D logical coordinate names."""
        kernel = MulKernel()
        assert kernel.get_logical_coord_names() == ("batch", "chunk")


class TestRoPEExampleKernel:
    """Tests for RoPEExampleKernel."""

    def test_logical_grid_size_3d(self):
        """Test 3D logical grid size."""
        kernel = RoPEExampleKernel(head_dim=64)

        class MockTensor:
            pass

        tensor = MockTensor()
        batch, seq_len, num_heads = 2, 128, 8

        grid_size = kernel.get_logical_grid_size(tensor, tensor, tensor, tensor, batch, seq_len, num_heads)

        # seq_chunks = ceil(128 / 64) = 2
        # total = 2 batches * 2 seq_chunks * 8 heads = 32
        assert grid_size == 32

    def test_logical_coord_3d(self):
        """Test 3D logical coordinate mapping."""
        kernel = RoPEExampleKernel(head_dim=64)

        class MockTensor:
            pass

        tensor = MockTensor()
        batch, seq_len, num_heads = 2, 128, 8

        # Decomposition: idx = b * (seq_chunks * heads) + s * heads + h
        # seq_chunks = 2, heads = 8
        # idx 0 -> (0, 0, 0)
        # idx 1 -> (0, 0, 1)
        # idx 8 -> (0, 1, 0)
        # idx 16 -> (1, 0, 0)
        coord_0 = kernel.get_logical_coord(0, tensor, tensor, tensor, tensor, batch, seq_len, num_heads)
        coord_1 = kernel.get_logical_coord(1, tensor, tensor, tensor, tensor, batch, seq_len, num_heads)
        coord_8 = kernel.get_logical_coord(8, tensor, tensor, tensor, tensor, batch, seq_len, num_heads)
        coord_16 = kernel.get_logical_coord(16, tensor, tensor, tensor, tensor, batch, seq_len, num_heads)

        assert coord_0 == (0, 0, 0)  # batch=0, seq_chunk=0, head=0
        assert coord_1 == (0, 0, 1)  # batch=0, seq_chunk=0, head=1
        assert coord_8 == (0, 1, 0)  # batch=0, seq_chunk=1, head=0
        assert coord_16 == (1, 0, 0)  # batch=1, seq_chunk=0, head=0

    def test_logical_coord_names(self):
        """Test 3D logical coordinate names."""
        kernel = RoPEExampleKernel()
        assert kernel.get_logical_coord_names() == ("batch", "seq_chunk", "head")

    def test_smem_size(self):
        """Test shared memory size calculation."""
        kernel = RoPEExampleKernel(head_dim=64)
        # half_dim * 2 (cos+sin) * 2 bytes
        assert kernel.smem_size == 64 * 2 * 2


class TestWarpSpecializedAddMul:
    """Tests for WarpSpecializedAddMul."""

    def test_warp_config(self):
        """Test warp configuration."""
        kernel = WarpSpecializedAddMul()
        config = kernel.warp_config

        assert config.num_consumer_warps == 12
        assert config.num_loader_warps == 1
        assert config.num_storer_warps == 1
        assert config.num_launcher_warps == 1
        assert config.num_controller_warps == 1
        assert config.total_warps == 16
        assert config.total_threads == 512

    def test_uses_warp_specialization(self):
        """Test warp specialization flag."""
        kernel = WarpSpecializedAddMul()
        assert kernel.uses_warp_specialization is True

    def test_smem_size(self):
        """Test shared memory size."""
        kernel = WarpSpecializedAddMul()
        # 2 tiles * 256 elements * 2 bytes
        assert kernel.smem_size == 256 * 2 * 2

    def test_logical_grid_size(self):
        """Test logical grid size."""
        kernel = WarpSpecializedAddMul()

        class MockTensor:
            def numel(self):
                return 1024

        tensor = MockTensor()
        grid_size = kernel.get_logical_grid_size(tensor, tensor, tensor, tensor)
        assert grid_size == 4  # 1024 / 256


class TestLogicalGridInfo:
    """Tests for LogicalGridInfo generation."""

    def test_add_kernel_grid_info(self):
        """Test AddKernel grid info."""
        kernel = AddKernel()

        class MockTensor:
            def numel(self):
                return 512

        tensor = MockTensor()
        info = kernel.get_logical_grid_info(tensor, tensor, tensor)

        assert info.logical_grid_size == 2
        assert info.coord_names == ("chunk",)
        assert info.coord_dims == (2,)

    def test_rope_kernel_grid_info(self):
        """Test RoPEExampleKernel grid info."""
        kernel = RoPEExampleKernel()

        class MockTensor:
            pass

        tensor = MockTensor()
        info = kernel.get_logical_grid_info(tensor, tensor, tensor, tensor, batch=2, seq_len=128, num_heads=8)

        assert info.logical_grid_size == 32
        assert info.coord_names == ("batch", "seq_chunk", "head")
        assert info.coord_dims == (2, 2, 8)


class TestWarpRoleDecorators:
    """Tests for warp role decorators on methods."""

    def test_load_forward_has_loader_role(self):
        """Test that load_forward has LOADER role."""
        kernel = WarpSpecializedAddMul()
        role = getattr(kernel.load_forward, "_machete_warp_role", None)
        assert role == WarpRole.LOADER

    def test_compute_forward_has_consumer_role(self):
        """Test that compute_forward has CONSUMER role."""
        kernel = WarpSpecializedAddMul()
        role = getattr(kernel.compute_forward, "_machete_warp_role", None)
        assert role == WarpRole.CONSUMER

    def test_store_forward_has_storer_role(self):
        """Test that store_forward has STORER role."""
        kernel = WarpSpecializedAddMul()
        role = getattr(kernel.store_forward, "_machete_warp_role", None)
        assert role == WarpRole.STORER


class TestDependencyDecorators:
    """Tests for reads/writes decorators."""

    def test_add_kernel_load_reads(self):
        """Test AddKernel load reads annotation."""
        kernel = AddKernel()
        reads_set = getattr(kernel.load_forward, "_machete_reads", set())
        assert reads_set == {"input", "bias"}

    def test_add_kernel_compute_writes(self):
        """Test AddKernel compute writes annotation."""
        kernel = AddKernel()
        writes_set = getattr(kernel.compute_forward, "_machete_writes", set())
        assert writes_set == {"output"}

    def test_warp_specialized_load_reads(self):
        """Test WarpSpecializedAddMul load reads annotation."""
        kernel = WarpSpecializedAddMul()
        reads_set = getattr(kernel.load_forward, "_machete_reads", set())
        assert reads_set == {"input", "bias", "scale"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
