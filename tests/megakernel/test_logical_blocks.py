# Copyright (c) 2025, Machete Authors
"""
Unit tests for the Logical Blocks abstraction.

Tests:
1. LogicalCoord and LogicalGridInfo data structures
2. get_logical_grid_size() computation
3. get_logical_coord() mapping
4. Scheduler integration with logical blocks
5. BarrierConfig tensor sizing
"""

import pytest
from machete.megakernel.scheduler import (
    LogicalCoord,
    LogicalGridInfo,
    WarpRole,
    WarpConfig,
    NoBubblesScheduler,
    NoBubblesConfig,
    PageAwareScheduler,
    OpDescriptor,
    BarrierConfig,
    reads,
    writes,
    independent,
    warp_role,
    get_method_dependencies,
    get_method_warp_role,
    build_op_descriptor_from_kernel,
)


class TestLogicalCoord:
    """Test LogicalCoord data structure."""

    def test_basic_creation(self):
        coord = LogicalCoord(values=(1, 2, 3))
        assert coord.values == (1, 2, 3)
        assert len(coord) == 3
        assert coord[0] == 1
        assert coord[1] == 2
        assert coord[2] == 3

    def test_with_names(self):
        coord = LogicalCoord(values=(2, 5, 8), names=("batch", "seq", "head"))
        assert coord.names == ("batch", "seq", "head")
        assert "batch=2" in repr(coord)
        assert "seq=5" in repr(coord)
        assert "head=8" in repr(coord)

    def test_repr_without_names(self):
        coord = LogicalCoord(values=(1, 2))
        assert "LogicalCoord(1, 2)" in repr(coord)


class TestLogicalGridInfo:
    """Test LogicalGridInfo data structure."""

    def test_basic_creation(self):
        info = LogicalGridInfo(logical_grid_size=100)
        assert info.logical_grid_size == 100
        assert info.coord_names == ()
        assert info.coord_dims == ()

    def test_with_coord_info(self):
        info = LogicalGridInfo(
            logical_grid_size=1024,
            coord_names=("batch", "seq_chunk", "head"),
            coord_dims=(2, 64, 8),
        )
        assert info.logical_grid_size == 1024
        assert info.coord_names == ("batch", "seq_chunk", "head")
        assert info.coord_dims == (2, 64, 8)


class TestWarpConfig:
    """Test WarpConfig for warp specialization."""

    def test_default_config(self):
        config = WarpConfig()
        assert config.num_consumer_warps == 16
        assert config.num_loader_warps == 1
        assert config.num_storer_warps == 1
        assert config.num_launcher_warps == 1
        assert config.num_controller_warps == 1
        assert config.total_warps == 20
        assert config.total_threads == 640

    def test_custom_config(self):
        config = WarpConfig(num_consumer_warps=12, num_loader_warps=2)
        assert config.total_warps == 17
        assert config.total_threads == 544

    def test_get_warp_role(self):
        config = WarpConfig(num_consumer_warps=4)

        # Consumer warps: 0-3
        for i in range(4):
            assert config.get_warp_role(i) == WarpRole.CONSUMER

        # System warps
        assert config.get_warp_role(4) == WarpRole.LOADER
        assert config.get_warp_role(5) == WarpRole.STORER
        assert config.get_warp_role(6) == WarpRole.LAUNCHER
        assert config.get_warp_role(7) == WarpRole.CONTROLLER


class TestBarrierConfig:
    """Test BarrierConfig for logical block synchronization."""

    def test_basic_config(self):
        config = BarrierConfig(num_ops=3, total_logical_blocks=100)
        assert config.num_ops == 3
        assert config.total_logical_blocks == 100
        assert config.tensor_size == (3, 100)
        assert config.total_counters == 300

    def test_barrier_index(self):
        config = BarrierConfig(num_ops=4, total_logical_blocks=50)

        # Index for op 0, logical block 0
        assert config.get_barrier_index(0, 0) == 0

        # Index for op 0, logical block 10
        assert config.get_barrier_index(0, 10) == 10

        # Index for op 1, logical block 0
        assert config.get_barrier_index(1, 0) == 50

        # Index for op 2, logical block 25
        assert config.get_barrier_index(2, 25) == 125


class TestDependencyDecorators:
    """Test @reads, @writes, @independent, @warp_role decorators."""

    def test_reads_decorator(self):
        @reads("input", "weight")
        def my_func():
            pass

        reads_set, writes_set, is_indep = get_method_dependencies(my_func)
        assert reads_set == {"input", "weight"}
        assert writes_set == set()
        assert not is_indep

    def test_writes_decorator(self):
        @writes("output")
        def my_func():
            pass

        reads_set, writes_set, is_indep = get_method_dependencies(my_func)
        assert reads_set == set()
        assert writes_set == {"output"}

    def test_combined_decorators(self):
        @reads("input")
        @writes("output")
        def my_func():
            pass

        reads_set, writes_set, is_indep = get_method_dependencies(my_func)
        assert reads_set == {"input"}
        assert writes_set == {"output"}

    def test_independent_decorator(self):
        @independent()
        def my_func():
            pass

        reads_set, writes_set, is_indep = get_method_dependencies(my_func)
        assert is_indep

    def test_warp_role_decorator(self):
        @warp_role(WarpRole.LOADER)
        def my_load():
            pass

        @warp_role(WarpRole.CONSUMER)
        def my_compute():
            pass

        assert get_method_warp_role(my_load) == WarpRole.LOADER
        assert get_method_warp_role(my_compute) == WarpRole.CONSUMER


class TestNoBubblesScheduler:
    """Test NoBubblesScheduler with logical blocks."""

    def test_calculate_total_logical_blocks(self):
        scheduler = NoBubblesScheduler()

        ops = [
            OpDescriptor(name="Op0", op_idx=0, logical_grid_size=100),
            OpDescriptor(name="Op1", op_idx=1, logical_grid_size=50),
            OpDescriptor(name="Op2", op_idx=2, logical_grid_size=200),
        ]

        total = scheduler.calculate_total_logical_blocks(ops)
        assert total == 200  # max of all ops

    def test_configure_barriers(self):
        scheduler = NoBubblesScheduler()
        scheduler.configure_barriers(num_ops=3, total_logical_blocks=100)

        assert scheduler.barrier_config is not None
        assert scheduler.barrier_config.num_ops == 3
        assert scheduler.barrier_config.total_logical_blocks == 100

    def test_dependency_aware_schedule_with_logical(self):
        scheduler = NoBubblesScheduler()

        ops = [
            OpDescriptor(
                name="Op0",
                op_idx=0,
                reads={"input"},
                writes={"intermediate"},
                logical_grid_size=100,
            ),
            OpDescriptor(
                name="Op1",
                op_idx=1,
                reads={"intermediate"},
                writes={"output"},
                logical_grid_size=100,
            ),
        ]

        micro_ops = scheduler.generate_dependency_aware_schedule(ops)

        # Should have barrier config set
        assert scheduler.barrier_config is not None
        assert scheduler.barrier_config.total_logical_blocks == 100

        # Should have micro-ops for both operations
        assert len(micro_ops) > 0

    def test_no_dependency_allows_early_load(self):
        """Test that Load[1] can start early when there's no RAW dependency."""
        scheduler = NoBubblesScheduler()

        ops = [
            OpDescriptor(name="Op0", op_idx=0, reads={"A"}, writes={"B"}),
            OpDescriptor(name="Op1", op_idx=1, reads={"C"}, writes={"D"}),  # No RAW dep on Op0
        ]

        scheduler.generate_dependency_aware_schedule(ops)

        # Find Load[1] - it should have no dependencies on Store[0]
        load_1 = next((op for op in scheduler.micro_ops if op.op_idx == 1 and "Load" in op.desc), None)
        assert load_1 is not None

        # Load[1] should only depend on sync, not on Store[0]
        store_0 = next((op for op in scheduler.micro_ops if op.op_idx == 0 and "Store" in op.desc), None)
        if store_0:
            assert store_0.id not in load_1.depends_on

    def test_raw_dependency_blocks_load(self):
        """Test that Load[1] waits for Store[0] when there's a RAW dependency."""
        scheduler = NoBubblesScheduler()

        ops = [
            OpDescriptor(name="Op0", op_idx=0, reads={"A"}, writes={"B"}),
            OpDescriptor(name="Op1", op_idx=1, reads={"B"}, writes={"C"}),  # RAW: reads B written by Op0
        ]

        scheduler.generate_dependency_aware_schedule(ops)

        # Find Load[1] and Store[0]
        load_1 = next((op for op in scheduler.micro_ops if op.op_idx == 1 and "Load" in op.desc), None)
        store_0 = next((op for op in scheduler.micro_ops if op.op_idx == 0 and "Store" in op.desc), None)

        assert load_1 is not None
        assert store_0 is not None
        assert store_0.id in load_1.depends_on


class TestPageAwareScheduler:
    """Test PageAwareScheduler with logical blocks."""

    def test_async_pipeline_with_logical(self):
        config = NoBubblesConfig(num_pages=8)
        scheduler = PageAwareScheduler(config)

        ops = [
            OpDescriptor(name="Op0", op_idx=0, logical_grid_size=50),
            OpDescriptor(name="Op1", op_idx=1, logical_grid_size=50),
        ]

        micro_ops = scheduler.generate_async_pipeline_schedule(ops, pages_per_op=2)

        # Should configure barriers
        assert scheduler.barrier_config is not None
        assert scheduler.barrier_config.total_logical_blocks == 50

        # Should have async load/store operations
        assert len(micro_ops) > 0

    def test_page_reuse(self):
        """Test that pages are reused when operations complete."""
        config = NoBubblesConfig(num_pages=4)  # Only 4 pages
        scheduler = PageAwareScheduler(config)

        # More ops than can fit in pages at once
        ops = [
            OpDescriptor(name=f"Op{i}", op_idx=i, logical_grid_size=10)
            for i in range(4)
        ]

        micro_ops = scheduler.generate_page_aware_schedule(ops, pages_per_op=2)

        # Later ops should wait for earlier ops to release pages
        assert len(micro_ops) > 0


class MockFusableKernel:
    """Mock FusableKernel for testing build_op_descriptor_from_kernel."""

    needs_block_sync = True
    needs_global_sync = False

    def __init__(self, logical_grid_size: int = 100):
        self._logical_grid_size = logical_grid_size

    def get_logical_grid_size(self, *args):
        return self._logical_grid_size

    @reads("input", "weight")
    def load_forward(self, paged_pool, page_idx, *args):
        pass

    @writes("output")
    def store_forward(self, paged_pool, page_idx, *args):
        pass

    def compute_forward(self, *args):
        pass


class TestBuildOpDescriptor:
    """Test build_op_descriptor_from_kernel function."""

    def test_extracts_dependencies(self):
        kernel = MockFusableKernel()
        desc = build_op_descriptor_from_kernel(kernel, op_idx=0, mode="forward")

        assert desc.op_idx == 0
        assert "input" in desc.reads
        assert "weight" in desc.reads
        assert "output" in desc.writes
        assert desc.needs_block_sync is True
        assert desc.needs_global_sync is False

    def test_detects_logical_grid(self):
        kernel = MockFusableKernel(logical_grid_size=200)
        desc = build_op_descriptor_from_kernel(kernel, op_idx=1, mode="forward")

        # Logical grid size is detected but deferred (-1 sentinel)
        assert desc.logical_grid_size == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
