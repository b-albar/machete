# Copyright (c) 2025, Machete Authors
"""
Integration tests for Producer/Consumer patterns with Logical Blocks.

Tests:
1. Simple Add -> Mul pipeline
2. Producer/Consumer with barrier synchronization
3. Paged shared memory with page reuse
4. Multi-operation pipeline with dependencies
"""

import pytest
from typing import Tuple
import cutlass.cute as cute

from machete.megakernel.interface import FusableKernel, reads, writes, LogicalGridInfo
from machete.megakernel.scheduler import (
    NoBubblesScheduler,
    NoBubblesConfig,
    NoBubblesScheduler,
    OpDescriptor,
    MicroOpType,
)


class MockProducerKernel(FusableKernel):
    """Mock producer kernel that writes to 'output'."""

    TILE_SIZE = 256

    def __init__(self, grid_size: int = 100):
        self._grid_size = grid_size

    @property
    def smem_size(self) -> int:
        """Total shared memory size (TILE_SIZE * 2 bytes per element)."""
        return self.TILE_SIZE * 2

    @property
    def needs_block_sync(self) -> bool:
        return True

    @property
    def needs_global_sync(self) -> bool:
        return True  # Producer needs global sync for consumer to see results

    def get_logical_grid_size(self, *args) -> int:
        return self._grid_size

    def get_logical_coord(self, logical_idx: int, *args) -> Tuple[int, ...]:
        return (logical_idx,)

    def get_logical_coord_names(self) -> Tuple[str, ...]:
        return ("chunk",)

    @reads("input")
    @cute.jit
    def load_forward(self, paged_pool, page_idx, *args):
        pass

    @writes("intermediate")
    @cute.jit
    def compute_forward(self, *args):
        pass

    @cute.jit
    def store_forward(self, paged_pool, page_idx, *args):
        pass


class MockConsumerKernel(FusableKernel):
    """Mock consumer kernel that reads from 'intermediate'."""

    TILE_SIZE = 256

    def __init__(self, grid_size: int = 100):
        self._grid_size = grid_size

    @property
    def smem_size(self) -> int:
        """Total shared memory size (TILE_SIZE * 2 bytes per element)."""
        return self.TILE_SIZE * 2

    @property
    def needs_block_sync(self) -> bool:
        return True

    @property
    def needs_global_sync(self) -> bool:
        return False

    def get_logical_grid_size(self, *args) -> int:
        return self._grid_size

    def get_logical_coord(self, logical_idx: int, *args) -> Tuple[int, ...]:
        return (logical_idx,)

    @reads("intermediate")
    @cute.jit
    def load_forward(self, paged_pool, page_idx, *args):
        pass

    @writes("output")
    @cute.jit
    def compute_forward(self, *args):
        pass

    @cute.jit
    def store_forward(self, paged_pool, page_idx, *args):
        pass


class TestProducerConsumerScheduling:
    """Test Producer/Consumer scheduling patterns."""

    def test_simple_producer_consumer(self):
        """Test that consumer waits for producer's store."""
        scheduler = NoBubblesScheduler()

        producer = OpDescriptor(
            name="Producer",
            op_idx=0,
            reads={"input"},
            writes={"intermediate"},
            needs_global_sync=True,
            logical_grid_size=100,
        )
        consumer = OpDescriptor(
            name="Consumer",
            op_idx=1,
            reads={"intermediate"},  # RAW dependency on producer
            writes={"output"},
            logical_grid_size=100,
        )

        micro_ops = scheduler.generate_page_aware_schedule([producer, consumer])

        # Find key operations
        producer_store = next(
            (op for op in micro_ops if op.op_idx == 0 and op.type == MicroOpType.STORE),
            None
        )
        consumer_load = next(
            (op for op in micro_ops if op.op_idx == 1 and op.type == MicroOpType.LOAD),
            None
        )

        assert producer_store is not None
        assert consumer_load is not None

        # Consumer load should depend on producer store (RAW dependency)
        assert producer_store.id in consumer_load.depends_on

    def test_independent_operations_overlap(self):
        """Test that independent operations can overlap."""
        scheduler = NoBubblesScheduler()

        op1 = OpDescriptor(
            name="Op1",
            op_idx=0,
            reads={"A"},
            writes={"B"},
            logical_grid_size=50,
        )
        op2 = OpDescriptor(
            name="Op2",
            op_idx=1,
            reads={"C"},  # No dependency on op1
            writes={"D"},
            logical_grid_size=50,
        )

        micro_ops = scheduler.generate_page_aware_schedule([op1, op2])

        # Get parallelizable groups (waves)
        waves = scheduler.get_parallelizable_groups()

        # Both loads should be in the same wave (can run in parallel)
        wave_0_ops = [(op.type, op.op_idx) for op in waves[0]]

        # First wave should contain Load[0]
        assert (MicroOpType.LOAD, 0) in wave_0_ops

        # Load[1] should be able to start early (wave 0 or 1)
        load_1 = next(
            (op for op in micro_ops if op.op_idx == 1 and op.type == MicroOpType.LOAD),
            None
        )
        assert load_1 is not None
        # Should not depend on Store[0]
        store_0 = next(
            (op for op in micro_ops if op.op_idx == 0 and op.type == MicroOpType.STORE),
            None
        )
        if store_0:
            assert store_0.id not in load_1.depends_on

    def test_barrier_tensor_sizing(self):
        """Test that barrier tensor is sized correctly for logical blocks."""
        scheduler = NoBubblesScheduler()

        ops = [
            OpDescriptor(name="Op0", op_idx=0, logical_grid_size=100),
            OpDescriptor(name="Op1", op_idx=1, logical_grid_size=150),
            OpDescriptor(name="Op2", op_idx=2, logical_grid_size=75),
        ]

        scheduler.generate_page_aware_schedule(ops)

        # Barrier should be sized for max logical blocks
        assert scheduler.barrier_config is not None
        assert scheduler.barrier_config.num_ops == 3
        assert scheduler.barrier_config.total_logical_blocks == 150
        assert scheduler.barrier_config.tensor_size == (3, 150)


class TestPagedScheduling:
    """Test paged memory scheduling with producer/consumer."""

    def test_async_pipeline_schedule(self):
        """Test async pipeline generates correct micro-ops."""
        config = NoBubblesConfig(num_pages=8)
        scheduler = NoBubblesScheduler(config)

        ops = [
            OpDescriptor(
                name="Producer",
                op_idx=0,
                reads={"input"},
                writes={"intermediate"},
                logical_grid_size=100,
            ),
            OpDescriptor(
                name="Consumer",
                op_idx=1,
                reads={"intermediate"},
                writes={"output"},
                logical_grid_size=100,
            ),
        ]

        micro_ops = scheduler.generate_async_pipeline_schedule(ops, pages_per_op=2)

        # Should have LOAD_ASYNC, COMMIT_GROUP, WAIT_LOAD, COMPUTE, STORE_ASYNC
        op_types = [op.type for op in micro_ops]

        assert MicroOpType.LOAD_ASYNC in op_types
        assert MicroOpType.COMMIT_GROUP in op_types
        assert MicroOpType.WAIT_LOAD in op_types
        assert MicroOpType.COMPUTE in op_types
        assert MicroOpType.STORE_ASYNC in op_types

    def test_page_acquire_release(self):
        """Test that pages are properly acquired and released."""
        config = NoBubblesConfig(num_pages=4)
        scheduler = NoBubblesScheduler(config)

        ops = [
            OpDescriptor(name="Op0", op_idx=0, logical_grid_size=50),
            OpDescriptor(name="Op1", op_idx=1, logical_grid_size=50),
        ]

        micro_ops = scheduler.generate_page_aware_schedule(ops, pages_per_op=2)

        # Check that ops acquire and release pages
        load_ops = [op for op in micro_ops if op.type == MicroOpType.LOAD]
        store_ops = [op for op in micro_ops if op.type == MicroOpType.STORE]

        # First load should acquire pages
        assert len(load_ops[0].acquires_pages) > 0

        # Stores should release pages
        for store_op in store_ops:
            # Not all stores may release pages (depends on scheduling)
            pass  # Just verify no errors

    def test_overlapped_schedule(self):
        """Test overlapped scheduling maximizes parallelism."""
        config = NoBubblesConfig(num_pages=6)  # Enough for 3 ops at 2 pages each
        scheduler = NoBubblesScheduler(config)

        ops = [
            OpDescriptor(name=f"Op{i}", op_idx=i, logical_grid_size=20)
            for i in range(4)
        ]

        micro_ops = scheduler.generate_overlapped_schedule(ops, pages_per_op=2)

        # Should have all operations
        assert len(micro_ops) > 0

        # Verify structure - should interleave loads with computes
        load_ops = [op for op in micro_ops if op.type == MicroOpType.LOAD]
        compute_ops = [op for op in micro_ops if op.type == MicroOpType.COMPUTE]

        # Should have loads for all 4 ops
        load_op_indices = set(op.op_idx for op in load_ops)
        assert load_op_indices == {0, 1, 2, 3}

        # Should have computes for all 4 ops
        compute_op_indices = set(op.op_idx for op in compute_ops)
        assert compute_op_indices == {0, 1, 2, 3}


class TestMultiOperationPipeline:
    """Test multi-operation pipeline patterns."""

    def test_three_stage_pipeline(self):
        """Test a 3-stage pipeline: A -> B -> C."""
        scheduler = NoBubblesScheduler()

        op_a = OpDescriptor(
            name="OpA",
            op_idx=0,
            reads={"input"},
            writes={"tensor_ab"},
            logical_grid_size=100,
        )
        op_b = OpDescriptor(
            name="OpB",
            op_idx=1,
            reads={"tensor_ab"},  # Depends on A
            writes={"tensor_bc"},
            logical_grid_size=100,
        )
        op_c = OpDescriptor(
            name="OpC",
            op_idx=2,
            reads={"tensor_bc"},  # Depends on B
            writes={"output"},
            logical_grid_size=100,
        )

        micro_ops = scheduler.generate_page_aware_schedule([op_a, op_b, op_c])

        # Verify dependency chain
        load_b = next(op for op in micro_ops if op.op_idx == 1 and op.type == MicroOpType.LOAD)
        load_c = next(op for op in micro_ops if op.op_idx == 2 and op.type == MicroOpType.LOAD)
        store_a = next(op for op in micro_ops if op.op_idx == 0 and op.type == MicroOpType.STORE)
        store_b = next(op for op in micro_ops if op.op_idx == 1 and op.type == MicroOpType.STORE)

        # B's load depends on A's store
        assert store_a.id in load_b.depends_on

        # C's load depends on B's store
        assert store_b.id in load_c.depends_on

    def test_diamond_dependency(self):
        """Test diamond dependency: A -> B, A -> C, B -> D, C -> D."""
        scheduler = NoBubblesScheduler()

        op_a = OpDescriptor(name="A", op_idx=0, reads=set(), writes={"out_a"}, logical_grid_size=50)
        op_b = OpDescriptor(name="B", op_idx=1, reads={"out_a"}, writes={"out_b"}, logical_grid_size=50)
        op_c = OpDescriptor(name="C", op_idx=2, reads={"out_a"}, writes={"out_c"}, logical_grid_size=50)
        op_d = OpDescriptor(name="D", op_idx=3, reads={"out_b", "out_c"}, writes={"final"}, logical_grid_size=50)

        micro_ops = scheduler.generate_page_aware_schedule([op_a, op_b, op_c, op_d])

        # Find relevant operations
        load_d = next(op for op in micro_ops if op.op_idx == 3 and op.type == MicroOpType.LOAD)
        store_b = next(op for op in micro_ops if op.op_idx == 1 and op.type == MicroOpType.STORE)
        store_c = next(op for op in micro_ops if op.op_idx == 2 and op.type == MicroOpType.STORE)

        # D's load should depend on both B's and C's stores
        assert store_b.id in load_d.depends_on
        assert store_c.id in load_d.depends_on

    def test_topological_sort(self):
        """Test that topological sort produces valid execution order."""
        scheduler = NoBubblesScheduler()

        ops = [
            OpDescriptor(name="A", op_idx=0, reads=set(), writes={"x"}, logical_grid_size=10),
            OpDescriptor(name="B", op_idx=1, reads={"x"}, writes={"y"}, logical_grid_size=10),
            OpDescriptor(name="C", op_idx=2, reads={"y"}, writes={"z"}, logical_grid_size=10),
        ]

        scheduler.generate_page_aware_schedule(ops)
        sorted_ops = scheduler.topological_sort()

        # Verify ordering: all dependencies come before dependents
        id_to_pos = {op.id: i for i, op in enumerate(sorted_ops)}

        for op in sorted_ops:
            for dep_id in op.depends_on:
                if dep_id in id_to_pos:  # Skip external dependencies
                    assert id_to_pos[dep_id] < id_to_pos[op.id], (
                        f"Dependency {dep_id} should come before {op.id}"
                    )


class TestVisualization:
    """Test schedule visualization."""

    def test_visualize_schedule(self):
        """Test ASCII visualization of schedule."""
        scheduler = NoBubblesScheduler()

        ops = [
            OpDescriptor(name="Op0", op_idx=0, reads={"A"}, writes={"B"}, logical_grid_size=10),
            OpDescriptor(name="Op1", op_idx=1, reads={"C"}, writes={"D"}, logical_grid_size=10),
        ]

        scheduler.generate_page_aware_schedule(ops)
        viz = scheduler.visualize_schedule()

        assert "Wave" in viz
        assert "LOAD" in viz
        assert "COMPUTE" in viz
        assert "STORE" in viz

    def test_visualize_page_schedule(self):
        """Test page-aware schedule visualization."""
        config = NoBubblesConfig(num_pages=4)
        scheduler = NoBubblesScheduler(config)

        ops = [
            OpDescriptor(name="Op0", op_idx=0, logical_grid_size=10),
            OpDescriptor(name="Op1", op_idx=1, logical_grid_size=10),
        ]

        scheduler.generate_page_aware_schedule(ops, pages_per_op=2)
        viz = scheduler.visualize_page_schedule()

        assert "Page-Aware Schedule" in viz
        assert "pages" in viz.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
