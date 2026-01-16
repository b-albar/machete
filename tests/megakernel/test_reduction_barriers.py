# Copyright (c) 2025, Machete Authors
"""
Tests for reduction barrier functionality with real cute.jit kernels.

Tests the many-to-one dependency pattern where:
- Producer: (batch, head, seq) -> e.g., 8192 blocks
- Consumer: (batch, head) -> e.g., 16 blocks
- Consumer must wait for ALL seq blocks per (batch, head) to complete

This tests the two-level barrier system:
1. Fine-grained barriers: Producer signals per logical block (existing)
2. Reduction barriers: Accumulated counter for consumer to wait on
"""

import pytest
import torch
from typing import Tuple
import cutlass.cute as cute
from cutlass import Int32

from machete.megakernel.interface import FusableKernel, LogicalGridInfo
from machete.megakernel.scheduler import (
    DimensionMapping,
    InterOpDependency,
    DependencyGranularity,
    ReductionBarrierConfig,
)


# ============================================================================
# Test Kernels for Reduction Pattern
# ============================================================================


class ProducerKernel(FusableKernel):
    """Producer kernel with (batch, head, seq) logical grid.

    Simulates an attention-like kernel that produces outputs for each
    (batch, head, seq) combination.
    """

    TILE_SIZE = 32  # Threads per block

    def __init__(self, batch_size: int, num_heads: int, seq_len: int, dtype=cute.Float16):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.cute_dtype = dtype

    @property
    def smem_size(self) -> int:
        return 0

    def get_logical_grid_size(self, input_tensor, output_tensor) -> int:
        """Total blocks = batch * head * seq."""
        return self.batch_size * self.num_heads * self.seq_len

    def get_logical_coord(self, logical_idx: int, input_tensor, output_tensor) -> Tuple[int, int, int]:
        """Map linear index to (batch, head, seq)."""
        seq_idx = logical_idx % self.seq_len
        head_idx = (logical_idx // self.seq_len) % self.num_heads
        batch_idx = logical_idx // (self.seq_len * self.num_heads)
        return (batch_idx, head_idx, seq_idx)

    def get_logical_coord_names(self) -> Tuple[str, ...]:
        return ("batch", "head", "seq")

    def get_logical_grid_info(self, input_tensor, output_tensor) -> LogicalGridInfo:
        return LogicalGridInfo(
            logical_grid_size=self.get_logical_grid_size(input_tensor, output_tensor),
            coord_names=("batch", "head", "seq"),
            coord_dims=(self.batch_size, self.num_heads, self.seq_len),
        )

    @cute.jit
    def load_forward(self, paged_pool, page_idx, logical_idx, input_tensor, output_tensor):
        pass

    @cute.jit
    def compute_forward(self, logical_idx, input_tensor, output_tensor):
        """Write logical_idx + 1 to output so we can verify execution."""
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            output_tensor[logical_idx] = input_tensor[logical_idx] + Int32(1)

    @cute.jit
    def store_forward(self, paged_pool, page_idx, logical_idx, input_tensor, output_tensor):
        pass

    def grid_fn(self, input_tensor, output_tensor):
        return (self.get_logical_grid_size(input_tensor, output_tensor), 1, 1)

    def block_fn(self, input_tensor, output_tensor):
        return (self.TILE_SIZE, 1, 1)


class ConsumerKernel(FusableKernel):
    """Consumer kernel with (batch, head) logical grid.

    Simulates an output projection-like kernel that consumes the reduced
    outputs from the producer. Each (batch, head) block must wait for ALL
    seq blocks from the producer.
    """

    TILE_SIZE = 32

    def __init__(self, batch_size: int, num_heads: int, seq_len: int, dtype=cute.Float16):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len  # Used to know how many producer blocks to expect
        self.cute_dtype = dtype

    @property
    def smem_size(self) -> int:
        return 0

    def get_logical_grid_size(self, producer_output, final_output) -> int:
        """Total blocks = batch * head."""
        return self.batch_size * self.num_heads

    def get_logical_coord(self, logical_idx: int, producer_output, final_output) -> Tuple[int, int]:
        """Map linear index to (batch, head)."""
        head_idx = logical_idx % self.num_heads
        batch_idx = logical_idx // self.num_heads
        return (batch_idx, head_idx)

    def get_logical_coord_names(self) -> Tuple[str, ...]:
        return ("batch", "head")

    def get_logical_grid_info(self, producer_output, final_output) -> LogicalGridInfo:
        return LogicalGridInfo(
            logical_grid_size=self.get_logical_grid_size(producer_output, final_output),
            coord_names=("batch", "head"),
            coord_dims=(self.batch_size, self.num_heads),
        )

    @cute.jit
    def load_forward(self, paged_pool, page_idx, logical_idx, producer_output, final_output):
        pass

    @cute.jit
    def compute_forward(self, logical_idx, producer_output, final_output):
        """Sum all seq values for this (batch, head) and store result.

        This verifies that all producer blocks for this (batch, head) have completed.
        """
        tidx, _, _ = cute.arch.thread_idx()

        if tidx == 0:
            # logical_idx = batch * num_heads + head
            # Producer indices for this (batch, head) are:
            # [batch * num_heads * seq_len + head * seq_len + s for s in range(seq_len)]
            batch_idx = logical_idx // self.num_heads
            head_idx = logical_idx % self.num_heads

            total = Int32(0)
            base_idx = (batch_idx * self.num_heads + head_idx) * self.seq_len
            for s in range(self.seq_len):
                total = total + producer_output[base_idx + s]

            final_output[logical_idx] = total

    @cute.jit
    def store_forward(self, paged_pool, page_idx, logical_idx, producer_output, final_output):
        pass

    def grid_fn(self, producer_output, final_output):
        return (self.get_logical_grid_size(producer_output, final_output), 1, 1)

    def block_fn(self, producer_output, final_output):
        return (self.TILE_SIZE, 1, 1)


# ============================================================================
# Tests
# ============================================================================


class TestDimensionMapping:
    """Test DimensionMapping for reduction patterns."""

    def test_reduction_trailing_axis(self):
        """Test reduction over the trailing seq axis."""
        mapping = DimensionMapping(
            producer_dims=(2, 8, 512),  # (batch, head, seq)
            consumer_dims=(2, 8),  # (batch, head)
        )

        assert mapping.is_reduction
        assert mapping.reduction_axes == (2,)
        assert mapping.blocks_per_consumer == 512

        # Consumer block 0 (batch=0, head=0) -> producer blocks 0..511
        producer_blocks = mapping.get_producer_blocks(0)
        assert len(producer_blocks) == 512
        assert producer_blocks[0] == 0
        assert producer_blocks[511] == 511

        # Consumer block 1 (batch=0, head=1) -> producer blocks 512..1023
        producer_blocks = mapping.get_producer_blocks(1)
        assert len(producer_blocks) == 512
        assert producer_blocks[0] == 512
        assert producer_blocks[511] == 1023

    def test_contiguous_range_optimization(self):
        """Test that trailing-axis reductions are detected as contiguous."""
        mapping = DimensionMapping(
            producer_dims=(2, 8, 512),
            consumer_dims=(2, 8),
        )

        # Should be contiguous (reduction over last axis)
        start, end, stride = mapping.get_producer_block_range(0)
        assert start == 0
        assert end == 512
        assert stride == 1

        start, end, stride = mapping.get_producer_block_range(5)
        assert start == 5 * 512
        assert end == 6 * 512
        assert stride == 1


class TestReductionBarrierConfig:
    """Test ReductionBarrierConfig for managing reduction barriers."""

    def test_add_reduction(self):
        """Test adding a reduction dependency."""
        config = ReductionBarrierConfig()

        dim_mapping = DimensionMapping(
            producer_dims=(2, 8, 512),
            consumer_dims=(2, 8),
        )

        config.add_reduction(
            producer_op_idx=0,
            consumer_op_idx=1,
            dim_mapping=dim_mapping,
        )

        assert len(config.reductions) == 1
        assert config.total_consumer_blocks == 16  # 2 * 8

    def test_get_barrier_index(self):
        """Test getting barrier index for consumer blocks."""
        config = ReductionBarrierConfig()

        dim_mapping = DimensionMapping(
            producer_dims=(2, 8, 512),
            consumer_dims=(2, 8),
        )

        config.add_reduction(0, 1, dim_mapping)

        # Consumer blocks should map to sequential barrier indices
        assert config.get_barrier_index(1, 0) == 0
        assert config.get_barrier_index(1, 5) == 5
        assert config.get_barrier_index(1, 15) == 15

    def test_get_consumer_logical_from_producer(self):
        """Test reverse mapping from producer to consumer block."""
        config = ReductionBarrierConfig()

        dim_mapping = DimensionMapping(
            producer_dims=(2, 8, 512),
            consumer_dims=(2, 8),
        )

        config.add_reduction(0, 1, dim_mapping)

        # Producer blocks 0..511 map to consumer block 0
        assert config.get_consumer_logical_from_producer(0, 1, 0) == 0
        assert config.get_consumer_logical_from_producer(0, 1, 100) == 0
        assert config.get_consumer_logical_from_producer(0, 1, 511) == 0

        # Producer blocks 512..1023 map to consumer block 1
        assert config.get_consumer_logical_from_producer(0, 1, 512) == 1
        assert config.get_consumer_logical_from_producer(0, 1, 700) == 1
        assert config.get_consumer_logical_from_producer(0, 1, 1023) == 1

    def test_get_wait_count(self):
        """Test getting wait count for consumer operations."""
        config = ReductionBarrierConfig()

        dim_mapping = DimensionMapping(
            producer_dims=(2, 8, 512),
            consumer_dims=(2, 8),
        )

        config.add_reduction(0, 1, dim_mapping)

        # Consumer should wait for 512 producer blocks
        assert config.get_wait_count(1) == 512


class TestInterOpDependencyReduction:
    """Test InterOpDependency with reduction dimension mapping."""

    def test_auto_infer_reduction_granularity(self):
        """Test that granularity is auto-inferred for reduction."""
        dep = InterOpDependency(
            producer_op="attention",
            consumer_op="output_proj",
            dim_mapping=DimensionMapping(
                producer_dims=(2, 8, 512),
                consumer_dims=(2, 8),
            ),
        )

        assert dep.granularity == DependencyGranularity.REDUCTION

    def test_get_producer_logical_blocks(self):
        """Test getting all producer blocks for a consumer."""
        dep = InterOpDependency(
            producer_op="attention",
            consumer_op="output_proj",
            dim_mapping=DimensionMapping(
                producer_dims=(2, 8, 512),
                consumer_dims=(2, 8),
            ),
        )

        # Consumer block 0 should depend on 512 producer blocks
        producer_blocks = dep.get_producer_logical_blocks(0)
        assert len(producer_blocks) == 512

    def test_get_wait_count(self):
        """Test getting wait count for reduction."""
        dep = InterOpDependency(
            producer_op="attention",
            consumer_op="output_proj",
            dim_mapping=DimensionMapping(
                producer_dims=(2, 8, 512),
                consumer_dims=(2, 8),
            ),
        )

        assert dep.get_wait_count(0) == 512


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestReductionBarrierExecution:
    """Integration tests for reduction barriers with real kernel execution.

    These tests require CUDA and verify that:
    1. Reduction barriers are correctly allocated
    2. Producer blocks signal the reduction barrier
    3. Consumer blocks wait for all producer blocks
    """

    def test_producer_consumer_reduction(self):
        """Test producer -> consumer with reduction dependency."""
        from machete.megakernel.core import Megakernel

        batch_size = 2
        num_heads = 4
        seq_len = 8  # Small for testing

        # Create kernels
        producer = ProducerKernel(batch_size, num_heads, seq_len)
        consumer = ConsumerKernel(batch_size, num_heads, seq_len)

        # Create tensors
        device = torch.device("cuda")
        producer_input = torch.ones(batch_size * num_heads * seq_len, dtype=torch.int32, device=device)
        producer_output = torch.zeros(batch_size * num_heads * seq_len, dtype=torch.int32, device=device)
        consumer_output = torch.zeros(batch_size * num_heads, dtype=torch.int32, device=device)

        # Create megakernel
        mk = Megakernel(name="reduction_test")

        # Add operations
        mk.add(producer, producer_input, producer_output)
        mk.add(consumer, producer_output, consumer_output)

        # Add reduction dependency
        mk.add_reduction_dependency(
            producer_op_idx=0,
            consumer_op_idx=1,
            producer_dims=(batch_size, num_heads, seq_len),
            consumer_dims=(batch_size, num_heads),
        )

        # Launch with logical blocks
        # Grid size = max(producer_blocks, consumer_blocks)
        producer_blocks = batch_size * num_heads * seq_len
        consumer_blocks = batch_size * num_heads
        n_blocks = max(producer_blocks, consumer_blocks)

        mk.launch_logical(block=(32, 1, 1), grid=(n_blocks, 1, 1))

        # Verify producer output: each element should be input + 1 = 2
        expected_producer = torch.ones_like(producer_output) * 2
        torch.testing.assert_close(producer_output, expected_producer)

        # Verify consumer output: each (batch, head) sums seq_len producer values
        # Each producer value is 2, so sum = seq_len * 2
        expected_consumer = torch.ones_like(consumer_output) * (seq_len * 2)
        torch.testing.assert_close(consumer_output, expected_consumer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
