# Copyright (c) 2025, Machete Authors
"""
Tests for Shared Memory Page Dependencies.

Tests the No Bubbles pattern's page semaphore protocol:
1. Page semaphore configuration and indexing
2. sem_arrived / sem_finished protocol
3. Warp-specialized schedule generation
4. Load[N+1] can start when Compute[N] finishes (not Store[N])
5. Page reuse and circular buffer pattern
6. Device-aware page count calculation
"""

import pytest
from typing import List

from machete.megakernel.scheduler import (
    PageSemaphoreConfig,
    NoBubblesScheduler,
    PageAwareMicroOp,
    NoBubblesConfig,
    OpDescriptor,
    MicroOpType,
    WarpRole,
    WarpConfig,
)


class TestPageSemaphoreConfig:
    """Test PageSemaphoreConfig for semaphore indexing."""

    def test_basic_config(self):
        config = PageSemaphoreConfig(num_pages=4)
        assert config.num_pages == 4
        assert config.total_semaphores == 12  # 3 per page (arrived, compute_done, finished)
        assert config.semaphore_size_bytes == 8  # Default mbarrier size

    def test_arrived_semaphore_index(self):
        """Test sem_arrived index calculation."""
        config = PageSemaphoreConfig(num_pages=4)

        # sem_arrived for page 0 = 0
        assert config.get_arrived_semaphore_idx(0) == 0
        # sem_arrived for page 1 = 3
        assert config.get_arrived_semaphore_idx(1) == 3
        # sem_arrived for page 2 = 6
        assert config.get_arrived_semaphore_idx(2) == 6
        # sem_arrived for page 3 = 9
        assert config.get_arrived_semaphore_idx(3) == 9

    def test_finished_semaphore_index(self):
        """Test sem_finished index calculation."""
        config = PageSemaphoreConfig(num_pages=4)

        # sem_finished for page 0 = 2
        assert config.get_finished_semaphore_idx(0) == 2
        # sem_finished for page 1 = 5
        assert config.get_finished_semaphore_idx(1) == 5
        # sem_finished for page 2 = 8
        assert config.get_finished_semaphore_idx(2) == 8
        # sem_finished for page 3 = 11
        assert config.get_finished_semaphore_idx(3) == 11

    def test_total_size_bytes(self):
        """Test total shared memory needed for semaphores."""
        config = PageSemaphoreConfig(num_pages=8, semaphore_size_bytes=8)
        # 8 pages * 3 semaphores/page * 8 bytes/semaphore = 192 bytes
        assert config.total_size_bytes == 192

    def test_semaphore_pairing(self):
        """Test that arrived and finished semaphores are correctly paired."""
        config = PageSemaphoreConfig(num_pages=4)

        for page_id in range(4):
            arrived = config.get_arrived_semaphore_idx(page_id)
            finished = config.get_finished_semaphore_idx(page_id)
            # finished should be arrived + 2 (they're separated by compute_done)
            assert finished == arrived + 2


class TestWarpSpecializedSchedule:
    """Test warp-specialized schedule with page semaphores."""

    def test_basic_warp_schedule(self):
        """Test basic warp-specialized schedule generation."""
        config = NoBubblesConfig(num_pages=4)
        scheduler = NoBubblesScheduler(config)

        ops = [
            OpDescriptor(name="Op0", op_idx=0, logical_grid_size=10),
            OpDescriptor(name="Op1", op_idx=1, logical_grid_size=10),
        ]

        micro_ops = scheduler.generate_warp_specialized_schedule(ops, pages_per_op=2)

        # Should have operations for both ops
        assert len(micro_ops) > 0

        # Verify warp roles are assigned
        loader_ops = [op for op in micro_ops if op.warp_role == WarpRole.LOADER]
        consumer_ops = [op for op in micro_ops if op.warp_role == WarpRole.CONSUMER]
        storer_ops = [op for op in micro_ops if op.warp_role == WarpRole.STORER]

        assert len(loader_ops) > 0, "Should have loader operations"
        assert len(consumer_ops) > 0, "Should have consumer operations"
        assert len(storer_ops) > 0, "Should have storer operations"

    def test_semaphore_signals_in_schedule(self):
        """Test that semaphores are correctly assigned to micro-ops."""
        config = NoBubblesConfig(num_pages=4)
        scheduler = NoBubblesScheduler(config)

        ops = [
            OpDescriptor(name="Op0", op_idx=0, logical_grid_size=10),
        ]

        micro_ops = scheduler.generate_warp_specialized_schedule(ops, pages_per_op=2)

        # Find the load operation for Op0
        load_op = next(
            (op for op in micro_ops if op.op_idx == 0 and op.type == MicroOpType.LOAD_ASYNC),
            None,
        )
        assert load_op is not None

        # Load should signal sem_arrived (data is ready)
        assert len(load_op.sem_signals) > 0, "Load should signal sem_arrived"

        # For first iteration, load should NOT wait (pages are free initially)
        assert len(load_op.sem_waits) == 0, "First load shouldn't wait for sem_finished"

        # Find the compute operation
        compute_op = next(
            (op for op in micro_ops if op.op_idx == 0 and op.type == MicroOpType.COMPUTE),
            None,
        )
        assert compute_op is not None

        # Compute should wait for sem_arrived (data ready) and signal sem_finished (page free)
        assert len(compute_op.sem_waits) > 0, "Compute should wait for sem_arrived"
        assert len(compute_op.sem_signals) > 0, "Compute should signal sem_finished"

    def test_load_waits_for_compute_not_store(self):
        """Test that Load[N+1] waits for Compute[N], not Store[N].

        This is the key insight of the No Bubbles pattern:
        - Load[N+1] can start as soon as Compute[N] signals sem_finished
        - Store[N] can run in parallel with Load[N+1]
        """
        config = NoBubblesConfig(num_pages=4)
        scheduler = NoBubblesScheduler(config)

        # More ops than can fit in pages at once (4 pages / 2 per op = 2 concurrent)
        ops = [
            OpDescriptor(name="Op0", op_idx=0, logical_grid_size=10),
            OpDescriptor(name="Op1", op_idx=1, logical_grid_size=10),
            OpDescriptor(name="Op2", op_idx=2, logical_grid_size=10),  # Must reuse pages from Op0
        ]

        micro_ops = scheduler.generate_warp_specialized_schedule(ops, pages_per_op=2)

        # Find Load[2] - it should depend on Compute[0], not Store[0]
        load_2 = next(
            (op for op in micro_ops if op.op_idx == 2 and op.type == MicroOpType.LOAD_ASYNC),
            None,
        )
        compute_0 = next(
            (op for op in micro_ops if op.op_idx == 0 and op.type == MicroOpType.COMPUTE),
            None,
        )
        store_0 = next(
            (op for op in micro_ops if op.op_idx == 0 and op.type == MicroOpType.STORE_ASYNC),
            None,
        )

        assert load_2 is not None
        assert compute_0 is not None
        assert store_0 is not None

        # Load[2] should depend on Store[0] (via sem_finished signaled by Storer)
        # Structural dependency is on store_0.id
        assert store_0.id in load_2.depends_on, "Load[2] should depend on Store[0]"

        # Load[2] should NOT directly depend on Compute[0] for page reuse
        # (It waits for the full L/C/S cycle to finish)
        assert compute_0.id not in load_2.depends_on, "Load[2] should depend on Store[0], not Compute[0]"

        # Load[2] should wait for sem_finished (page freed by Compute[0])
        assert len(load_2.sem_waits) > 0, "Load[2] should wait for sem_finished"

    def test_circular_page_allocation(self):
        """Test that pages are allocated circularly."""
        config = NoBubblesConfig(num_pages=4)
        scheduler = NoBubblesScheduler(config)

        ops = [OpDescriptor(name=f"Op{i}", op_idx=i, logical_grid_size=10) for i in range(4)]

        micro_ops = scheduler.generate_warp_specialized_schedule(ops, pages_per_op=2)

        # Extract page allocations for each load
        page_allocations = {}
        for op in micro_ops:
            if op.type == MicroOpType.LOAD_ASYNC:
                page_allocations[op.op_idx] = op.acquires_pages

        # Op0: pages [0, 1]
        assert page_allocations[0] == [0, 1]
        # Op1: pages [2, 3]
        assert page_allocations[1] == [2, 3]
        # Op2: pages [0, 1] (reuse from Op0)
        assert page_allocations[2] == [0, 1]
        # Op3: pages [2, 3] (reuse from Op1)
        assert page_allocations[3] == [2, 3]

    def test_semaphore_protocol_correctness(self):
        """Verify the complete semaphore protocol for page reuse."""
        config = NoBubblesConfig(num_pages=4)
        scheduler = NoBubblesScheduler(config)

        ops = [
            OpDescriptor(name="Op0", op_idx=0, logical_grid_size=10),
            OpDescriptor(name="Op1", op_idx=1, logical_grid_size=10),
            OpDescriptor(name="Op2", op_idx=2, logical_grid_size=10),
        ]

        micro_ops = scheduler.generate_warp_specialized_schedule(ops, pages_per_op=2)

        # Get semaphore config
        sem_config = scheduler.semaphore_config
        assert sem_config is not None

        # Op0 uses pages [0, 1]
        sem_arrived_pages_01 = [sem_config.get_arrived_semaphore_idx(0), sem_config.get_arrived_semaphore_idx(1)]
        sem_compute_done_pages_01 = [
            sem_config.get_compute_done_semaphore_idx(0),
            sem_config.get_compute_done_semaphore_idx(1),
        ]
        sem_finished_pages_01 = [sem_config.get_finished_semaphore_idx(0), sem_config.get_finished_semaphore_idx(1)]

        # Find Load[0], Compute[0], and Store[0]
        load_0 = next(op for op in micro_ops if op.op_idx == 0 and op.type == MicroOpType.LOAD_ASYNC)
        compute_0 = next(op for op in micro_ops if op.op_idx == 0 and op.type == MicroOpType.COMPUTE)
        store_0 = next(op for op in micro_ops if op.op_idx == 0 and op.type == MicroOpType.STORE_ASYNC)

        # Load[0] signals sem_arrived for pages [0, 1]
        assert set(load_0.sem_signals) == set(sem_arrived_pages_01)

        # Compute[0] waits for sem_arrived and signals sem_compute_done
        assert set(compute_0.sem_waits) == set(sem_arrived_pages_01)
        assert set(compute_0.sem_signals) == set(sem_compute_done_pages_01)

        # Store[0] waits for sem_compute_done and signals sem_finished
        assert set(store_0.sem_waits) == set(sem_compute_done_pages_01)
        assert set(store_0.sem_signals) == set(sem_finished_pages_01)

        # Op2 reuses pages [0, 1], so Load[2] should wait for sem_finished[0, 1]
        load_2 = next(op for op in micro_ops if op.op_idx == 2 and op.type == MicroOpType.LOAD_ASYNC)
        assert set(load_2.sem_waits) == set(sem_finished_pages_01)


class TestPageDependencies:
    """Test page-level dependencies between operations."""

    def test_data_dependency_with_page_reuse(self):
        """Test that data dependencies are respected even with page reuse."""
        config = NoBubblesConfig(num_pages=4)
        scheduler = NoBubblesScheduler(config)

        # Op1 reads what Op0 writes (RAW dependency)
        ops = [
            OpDescriptor(name="Producer", op_idx=0, writes={"tensor_A"}, logical_grid_size=10),
            OpDescriptor(name="Consumer", op_idx=1, reads={"tensor_A"}, logical_grid_size=10),
        ]

        micro_ops = scheduler.generate_warp_specialized_schedule(ops, pages_per_op=2)

        # Find Load[1] and Store[0]
        load_1 = next(op for op in micro_ops if op.op_idx == 1 and op.type == MicroOpType.LOAD_ASYNC)
        store_0 = next(op for op in micro_ops if op.op_idx == 0 and op.type == MicroOpType.STORE_ASYNC)

        # Load[1] should depend on Store[0] due to RAW dependency
        assert store_0.id in load_1.depends_on

    def test_independent_ops_use_different_pages(self):
        """Test that independent ops can use different pages concurrently."""
        config = NoBubblesConfig(num_pages=4)
        scheduler = NoBubblesScheduler(config)

        # These ops are independent (no data dependencies)
        ops = [
            OpDescriptor(name="Op0", op_idx=0, reads={"A"}, writes={"B"}, logical_grid_size=10),
            OpDescriptor(name="Op1", op_idx=1, reads={"C"}, writes={"D"}, logical_grid_size=10),
        ]

        micro_ops = scheduler.generate_warp_specialized_schedule(ops, pages_per_op=2)

        # Find loads for both ops
        load_0 = next(op for op in micro_ops if op.op_idx == 0 and op.type == MicroOpType.LOAD_ASYNC)
        load_1 = next(op for op in micro_ops if op.op_idx == 1 and op.type == MicroOpType.LOAD_ASYNC)

        # They should use different pages
        assert set(load_0.acquires_pages) != set(load_1.acquires_pages)

        # Load[1] should NOT depend on any Op0 operation (no data dependency)
        op0_ids = {op.id for op in micro_ops if op.op_idx == 0}
        assert not (load_1.depends_on & op0_ids)

    def test_page_wait_chain(self):
        """Test that page waits form a proper chain for circular buffer."""
        config = NoBubblesConfig(num_pages=4)
        scheduler = NoBubblesScheduler(config)

        # 6 ops with 2 pages each = must reuse pages multiple times
        ops = [OpDescriptor(name=f"Op{i}", op_idx=i, logical_grid_size=10) for i in range(6)]

        micro_ops = scheduler.generate_warp_specialized_schedule(ops, pages_per_op=2)

        # max_concurrent = 4 / 2 = 2 ops at a time
        # Op2 waits for Compute[0], Op3 waits for Compute[1]
        # Op4 waits for Compute[2], Op5 waits for Compute[3]

        for i in range(2, 6):
            load_i = next(op for op in micro_ops if op.op_idx == i and op.type == MicroOpType.LOAD_ASYNC)
            store_prev = next(op for op in micro_ops if op.op_idx == i - 2 and op.type == MicroOpType.STORE_ASYNC)

            # Load[i] should depend on Store[i-2] (the one that frees the pages)
            assert store_prev.id in load_i.depends_on, f"Load[{i}] should depend on Store[{i - 2}]"


class TestDeviceAwarePageCount:
    """Test device-aware page count calculation."""

    def test_compute_pages_from_smem(self):
        """Test computing num_pages from shared memory size."""
        # H100 has ~227KB shared memory
        # With 16KB pages, that's ~14 pages (leaving some for other uses)
        h100_smem = 227 * 1024  # 227 KB
        page_size = 16 * 1024  # 16 KB

        # Reserve some for other uses (semaphores, instruction buffer, etc.)
        reserved_smem = 4 * 1024  # 4 KB reserved
        available_smem = h100_smem - reserved_smem

        num_pages = available_smem // page_size
        assert num_pages == 13  # (227 - 4) / 16 = 13.9 -> 13

    def test_from_device_smem_h100(self):
        """Test NoBubblesConfig.from_device_smem for H100."""
        h100_smem = 227 * 1024  # 227 KB
        config = NoBubblesConfig.from_device_smem(h100_smem)

        assert config.num_pages == 13
        assert config.page_size_bytes == 16384
        assert config.reserved_smem_bytes == 4096

    def test_from_device_smem_a100(self):
        """Test NoBubblesConfig.from_device_smem for A100."""
        a100_smem = 164 * 1024  # 164 KB
        config = NoBubblesConfig.from_device_smem(a100_smem)

        # (164 - 4) / 16 = 10 pages
        assert config.num_pages == 10

    def test_from_device_smem_custom_page_size(self):
        """Test from_device_smem with custom page size."""
        smem = 128 * 1024  # 128 KB
        config = NoBubblesConfig.from_device_smem(smem, page_size_bytes=8192)

        # (128 - 4) / 8 = 15.5 -> 15 pages
        assert config.num_pages == 15
        assert config.page_size_bytes == 8192

    def test_from_device_smem_custom_reserved(self):
        """Test from_device_smem with custom reserved memory."""
        smem = 227 * 1024
        config = NoBubblesConfig.from_device_smem(smem, reserved_smem_bytes=8192)

        # (227 - 8) / 16 = 13.6 -> 13 pages
        assert config.num_pages == 13
        assert config.reserved_smem_bytes == 8192

    def test_from_device_smem_insufficient_memory(self):
        """Test from_device_smem raises error for insufficient memory."""
        # Only 20KB - not enough for 2 pages of 16KB each
        smem = 20 * 1024

        with pytest.raises(ValueError, match="Not enough shared memory"):
            NoBubblesConfig.from_device_smem(smem)

    def test_total_smem_properties(self):
        """Test total_smem_bytes and total_page_smem_bytes properties."""
        config = NoBubblesConfig(num_pages=13, page_size_bytes=16384, reserved_smem_bytes=4096)

        # 13 pages * 16KB = 208KB
        assert config.total_page_smem_bytes == 13 * 16384

        # 208KB + 4KB = 212KB
        assert config.total_smem_bytes == 13 * 16384 + 4096

    def test_config_with_custom_page_size(self):
        """Test NoBubblesConfig with custom page size."""
        # Smaller pages for smaller operations
        config = NoBubblesConfig(num_pages=26, page_size_bytes=8192)  # 8KB pages
        scheduler = NoBubblesScheduler(config)

        assert scheduler.config.num_pages == 26
        assert scheduler.config.page_size_bytes == 8192
        assert scheduler.semaphore_config.num_pages == 26

    def test_pages_per_op_based_on_smem_requirement(self):
        """Test that pages_per_op can be computed from operation smem requirements."""
        page_size = 16384  # 16KB

        # An operation that needs 48KB of shared memory
        op_smem_requirement = 48 * 1024

        # Would need ceil(48KB / 16KB) = 3 pages
        pages_needed = (op_smem_requirement + page_size - 1) // page_size
        assert pages_needed == 3


class TestPageScheduleVisualization:
    """Test schedule visualization with page info."""

    def test_visualize_warp_schedule(self):
        """Test warp schedule visualization includes all info."""
        config = NoBubblesConfig(num_pages=4)
        scheduler = NoBubblesScheduler(config)

        ops = [
            OpDescriptor(name="Op0", op_idx=0, logical_grid_size=10),
            OpDescriptor(name="Op1", op_idx=1, logical_grid_size=10),
        ]

        scheduler.generate_warp_specialized_schedule(ops, pages_per_op=2)
        viz = scheduler.visualize_warp_schedule()

        # Check that visualization includes key information
        assert "Warp-Specialized" in viz
        assert "Pages:" in viz
        assert "Semaphores:" in viz
        assert "LOADER" in viz
        assert "CONSUMER" in viz
        assert "STORER" in viz
        assert "signal(" in viz or "wait(" in viz


class TestEdgeCases:
    """Test edge cases for page dependencies."""

    def test_single_operation(self):
        """Test schedule with single operation."""
        config = NoBubblesConfig(num_pages=4)
        scheduler = NoBubblesScheduler(config)

        ops = [OpDescriptor(name="Op0", op_idx=0, logical_grid_size=10)]

        micro_ops = scheduler.generate_warp_specialized_schedule(ops, pages_per_op=2)

        # Should have load, compute, store
        types = {op.type for op in micro_ops}
        assert MicroOpType.LOAD_ASYNC in types
        assert MicroOpType.COMPUTE in types
        assert MicroOpType.STORE_ASYNC in types

        # Single op load should not wait for any semaphores
        load_0 = next(op for op in micro_ops if op.type == MicroOpType.LOAD_ASYNC)
        assert len(load_0.sem_waits) == 0

    def test_many_ops_limited_pages(self):
        """Test many operations with very limited pages."""
        config = NoBubblesConfig(num_pages=2)  # Only 2 pages
        scheduler = NoBubblesScheduler(config)

        ops = [OpDescriptor(name=f"Op{i}", op_idx=i, logical_grid_size=10) for i in range(5)]

        # With 2 pages and 2 pages per op, only 1 op at a time
        micro_ops = scheduler.generate_warp_specialized_schedule(ops, pages_per_op=2)

        # All ops should use the same pages [0, 1]
        for i in range(5):
            load_i = next(op for op in micro_ops if op.op_idx == i and op.type == MicroOpType.LOAD_ASYNC)
            assert load_i.acquires_pages == [0, 1]

            # All ops after op 0 should wait for sem_finished
            if i > 0:
                assert len(load_i.sem_waits) > 0

    def test_no_ops(self):
        """Test empty operation list."""
        config = NoBubblesConfig(num_pages=4)
        scheduler = NoBubblesScheduler(config)

        micro_ops = scheduler.generate_warp_specialized_schedule([], pages_per_op=2)
        assert len(micro_ops) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
