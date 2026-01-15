# Copyright (c) 2025, Machete Authors
"""
Test dependency-aware scheduler with decorated L/C/S methods.
"""

# pytest imported for test discovery
import cutlass.cute as cute

from machete.megakernel.scheduler import (
    NoBubblesScheduler,
    NoBubblesConfig,
    OpDescriptor,
    MicroOpType,
    reads,
    writes,
    independent,
    get_method_dependencies,
    build_op_descriptor_from_kernel,
    PagedMemoryAllocator,
    PageAwareScheduler,
)
from machete.megakernel.interface import FusableKernel


# ============================================================================
# Simple test kernels with decorated L/C/S methods
# ============================================================================


class KernelA(FusableKernel):
    """Reads from 'x', writes to 'y'."""

    @property
    def num_tensors(self):
        return 2

    @reads("x")
    @cute.jit
    def load_forward(self, paged_pool, page_idx, x, y):
        pass

    @cute.jit
    def compute_forward(self, x, y):
        pass

    @writes("y")
    @cute.jit
    def store_forward(self, paged_pool, page_idx, x, y):
        pass


class KernelB(FusableKernel):
    """Reads from 'y' (depends on KernelA), writes to 'z'."""

    @property
    def num_tensors(self):
        return 2

    @reads("y")  # This creates a RAW dependency on KernelA's store
    @cute.jit
    def load_forward(self, paged_pool, page_idx, y, z):
        pass

    @cute.jit
    def compute_forward(self, y, z):
        pass

    @writes("z")
    @cute.jit
    def store_forward(self, paged_pool, page_idx, y, z):
        pass


class KernelC(FusableKernel):
    """Reads from 'a' (independent of A and B), writes to 'b'."""

    @property
    def num_tensors(self):
        return 2

    @reads("a")  # No dependency on x, y, z - can run early!
    @cute.jit
    def load_forward(self, paged_pool, page_idx, a, b):
        pass

    @cute.jit
    def compute_forward(self, a, b):
        pass

    @writes("b")
    @cute.jit
    def store_forward(self, paged_pool, page_idx, a, b):
        pass


class IndependentKernel(FusableKernel):
    """Explicitly marked as independent."""

    @property
    def num_tensors(self):
        return 1

    @independent()
    @cute.jit
    def load_forward(self, paged_pool, page_idx, x):
        pass

    @cute.jit
    def compute_forward(self, x):
        pass

    @cute.jit
    def store_forward(self, paged_pool, page_idx, x):
        pass


# ============================================================================
# Tests for decorators
# ============================================================================


def test_reads_decorator():
    """Test that @reads decorator sets metadata correctly."""
    method = KernelA().load_forward
    reads_set, writes_set, is_independent = get_method_dependencies(method)
    assert reads_set == {"x"}
    assert writes_set == set()
    assert is_independent is False


def test_writes_decorator():
    """Test that @writes decorator sets metadata correctly."""
    method = KernelA().store_forward
    reads_set, writes_set, is_independent = get_method_dependencies(method)
    assert reads_set == set()
    assert writes_set == {"y"}
    assert is_independent is False


def test_independent_decorator():
    """Test that @independent decorator sets metadata correctly."""
    method = IndependentKernel().load_forward
    reads_set, writes_set, is_independent = get_method_dependencies(method)
    assert is_independent is True


def test_build_op_descriptor():
    """Test building OpDescriptor from decorated kernel."""
    kernel = KernelA()
    desc = build_op_descriptor_from_kernel(kernel, 0, "forward")

    assert desc.name == "KernelA"
    assert desc.op_idx == 0
    assert desc.reads == {"x"}
    assert desc.writes == {"y"}


# ============================================================================
# Tests for dependency-aware scheduling
# ============================================================================


def test_simple_chain_with_dependency():
    """Test A -> B chain where B depends on A (RAW on 'y')."""
    scheduler = NoBubblesScheduler()

    ops = [
        OpDescriptor("A", 0, reads={"x"}, writes={"y"}),
        OpDescriptor("B", 1, reads={"y"}, writes={"z"}),  # Depends on A
    ]

    scheduler.generate_dependency_aware_schedule(ops)
    _ = scheduler.get_parallelizable_groups()  # Verify it works

    # Expected:
    # Wave 0: Load[0] (can start immediately)
    # Wave 1: Compute[0]
    # Wave 2: Store[0]
    # Wave 3: BlockSync[0], Load[1] (Load[1] depends on Store[0])
    # Wave 4: Compute[1]
    # Wave 5: Store[1]
    # Wave 6: BlockSync[1]

    # Verify Load[1] depends on Store[0]
    load_1 = next(op for op in scheduler.micro_ops if op.type == MicroOpType.LOAD and op.op_idx == 1)
    store_0 = next(op for op in scheduler.micro_ops if op.type == MicroOpType.STORE and op.op_idx == 0)
    assert store_0.id in load_1.depends_on, "Load[1] should depend on Store[0] (RAW dependency on 'y')"


def test_independent_ops_parallel():
    """Test A and C can run in parallel (no shared memory regions)."""
    scheduler = NoBubblesScheduler()

    ops = [
        OpDescriptor("A", 0, reads={"x"}, writes={"y"}, needs_block_sync=False),
        OpDescriptor("C", 1, reads={"a"}, writes={"b"}, needs_block_sync=False),  # Independent!
    ]

    scheduler.generate_dependency_aware_schedule(ops)
    waves = scheduler.get_parallelizable_groups()

    # Both loads should be in wave 0 since they're independent
    wave_0_types = [(op.type, op.op_idx) for op in waves[0]]
    assert (MicroOpType.LOAD, 0) in wave_0_types
    assert (MicroOpType.LOAD, 1) in wave_0_types, "Load[1] should be in Wave 0 (independent of A)"


def test_three_op_mixed_deps():
    """Test A -> B (dependent) + C (independent) schedule."""
    scheduler = NoBubblesScheduler()

    ops = [
        OpDescriptor("A", 0, reads={"x"}, writes={"y"}, needs_block_sync=False),
        OpDescriptor("B", 1, reads={"y"}, writes={"z"}, needs_block_sync=False),  # Depends on A
        OpDescriptor("C", 2, reads={"a"}, writes={"b"}, needs_block_sync=False),  # Independent
    ]

    scheduler.generate_dependency_aware_schedule(ops)

    # Load[0] and Load[2] should be independent (Wave 0)
    # Load[1] should depend on Store[0]
    load_0 = next(op for op in scheduler.micro_ops if op.type == MicroOpType.LOAD and op.op_idx == 0)
    load_1 = next(op for op in scheduler.micro_ops if op.type == MicroOpType.LOAD and op.op_idx == 1)
    load_2 = next(op for op in scheduler.micro_ops if op.type == MicroOpType.LOAD and op.op_idx == 2)

    assert len(load_0.depends_on) == 0, "Load[0] should have no dependencies"
    assert len(load_2.depends_on) == 0, "Load[2] should have no dependencies (independent)"
    assert len(load_1.depends_on) > 0, "Load[1] should depend on Store[0]"


def test_topological_sort():
    """Test that topological sort produces valid execution order."""
    scheduler = NoBubblesScheduler()

    ops = [
        OpDescriptor("A", 0, reads={"x"}, writes={"y"}, needs_block_sync=False),
        OpDescriptor("B", 1, reads={"y"}, writes={"z"}, needs_block_sync=False),
    ]

    scheduler.generate_dependency_aware_schedule(ops)
    sorted_ops = scheduler.topological_sort()

    # Verify all dependencies come before their dependents
    position = {op.id: i for i, op in enumerate(sorted_ops)}
    for op in sorted_ops:
        for dep_id in op.depends_on:
            assert position[dep_id] < position[op.id], f"Dependency {dep_id} should come before {op.id}"


def test_visualize_schedule():
    """Test schedule visualization output."""
    scheduler = NoBubblesScheduler()

    ops = [
        OpDescriptor("A", 0, reads={"x"}, writes={"y"}, needs_block_sync=False),
        OpDescriptor("C", 1, reads={"a"}, writes={"b"}, needs_block_sync=False),
    ]

    scheduler.generate_dependency_aware_schedule(ops)
    viz = scheduler.visualize_schedule()

    assert "Wave 0" in viz
    assert "LOAD" in viz
    print("\n" + viz)  # For manual inspection


def test_from_kernel_integration():
    """Test full integration: decorated kernels -> scheduler."""
    scheduler = NoBubblesScheduler()

    kernel_a = KernelA()
    kernel_c = KernelC()  # Independent of A

    ops = [
        build_op_descriptor_from_kernel(kernel_a, 0, "forward"),
        build_op_descriptor_from_kernel(kernel_c, 1, "forward"),
    ]

    # Disable block sync for cleaner test
    ops[0].needs_block_sync = False
    ops[1].needs_block_sync = False

    scheduler.generate_dependency_aware_schedule(ops)
    waves = scheduler.get_parallelizable_groups()

    # Both loads should be parallel since they read different memory
    wave_0_ops = [(op.type, op.op_idx) for op in waves[0]]
    assert (MicroOpType.LOAD, 0) in wave_0_ops
    assert (MicroOpType.LOAD, 1) in wave_0_ops


# ============================================================================
# Tests for paged memory allocator
# ============================================================================


def test_paged_allocator_basic():
    """Test basic page allocation and release."""
    allocator = PagedMemoryAllocator(num_pages=4)

    # Acquire 2 pages for op 0
    pages = allocator.try_acquire(0, 2)
    assert pages == [0, 1]
    assert allocator.page_owner[0] == 0
    assert allocator.page_owner[1] == 0

    # Acquire 2 more pages for op 1
    pages = allocator.try_acquire(1, 2)
    assert pages == [2, 3]

    # Try to acquire more - should fail (no pages left)
    pages = allocator.try_acquire(2, 2)
    assert pages is None

    # Release op 0's pages
    released = allocator.release(0)
    assert set(released) == {0, 1}

    # Now we can acquire for op 2
    pages = allocator.try_acquire(2, 2)
    assert pages == [0, 1]


def test_paged_allocator_blocking_ops():
    """Test finding which ops block acquisition."""
    allocator = PagedMemoryAllocator(num_pages=4)

    # Op 0 takes 2 pages
    allocator.try_acquire(0, 2)
    # Op 1 takes 2 pages
    allocator.try_acquire(1, 2)

    # No pages available - who's blocking?
    blocking = allocator.get_blocking_ops(2)
    # Should return at least one of the ops holding pages
    assert len(blocking) >= 1
    assert blocking.issubset({0, 1})


def test_page_aware_scheduler_basic():
    """Test page-aware scheduling with enough pages."""
    config = NoBubblesConfig(num_pages=4)  # 4 pages, 2 per op = 2 ops can overlap
    scheduler = PageAwareScheduler(config)

    ops = [
        OpDescriptor("A", 0, reads={"x"}, writes={"y"}, needs_block_sync=False),
        OpDescriptor("B", 1, reads={"a"}, writes={"b"}, needs_block_sync=False),  # Independent
    ]

    scheduler.generate_page_aware_schedule(ops, pages_per_op=2)

    # With 4 pages and 2 per op, both loads should be able to start
    load_0 = next(op for op in scheduler.micro_ops if op.type == MicroOpType.LOAD and op.op_idx == 0)
    load_1 = next(op for op in scheduler.micro_ops if op.type == MicroOpType.LOAD and op.op_idx == 1)

    # Load[1] should have no page dependencies (enough pages available)
    # Note: it may still have data dependencies if there's a RAW
    assert len(load_0.acquires_pages) == 2
    assert len(load_1.acquires_pages) == 2


def test_page_aware_scheduler_page_contention():
    """Test page-aware scheduling when pages are limited."""
    config = NoBubblesConfig(num_pages=2)  # Only 2 pages!
    scheduler = PageAwareScheduler(config)

    ops = [
        OpDescriptor("A", 0, reads={"x"}, writes={"y"}, needs_block_sync=False),
        OpDescriptor("B", 1, reads={"a"}, writes={"b"}, needs_block_sync=False),
        OpDescriptor("C", 2, reads={"p"}, writes={"q"}, needs_block_sync=False),
    ]

    scheduler.generate_page_aware_schedule(ops, pages_per_op=2)

    # With only 2 pages and 2 needed per op:
    # - Load[0] acquires [0,1] immediately
    # - Load[1] must WAIT for Store[0] to release pages
    # - Load[2] must WAIT for Store[1] to release pages

    load_1 = next(op for op in scheduler.micro_ops if op.type == MicroOpType.LOAD and op.op_idx == 1)
    store_0 = next(op for op in scheduler.micro_ops if op.type == MicroOpType.STORE and op.op_idx == 0)

    # Load[1] should depend on Store[0] due to page contention
    assert store_0.id in load_1.depends_on, "Load[1] should wait for Store[0] to release pages"

    # Visualize
    print("\n" + scheduler.visualize_page_schedule())


def test_page_aware_scheduler_visualization():
    """Test visualization with page info."""
    config = NoBubblesConfig(num_pages=4)
    scheduler = PageAwareScheduler(config)

    ops = [
        OpDescriptor("A", 0, reads={"x"}, writes={"y"}, needs_block_sync=False),
        OpDescriptor("B", 1, reads={"a"}, writes={"b"}, needs_block_sync=False),
    ]

    scheduler.generate_page_aware_schedule(ops, pages_per_op=2)
    viz = scheduler.visualize_page_schedule()

    assert "ACQUIRE" in viz
    assert "RELEASE" in viz
    print("\n" + viz)


def test_async_pipeline_schedule():
    """Test async pipeline scheduling with proper producer/consumer sync."""
    from machete.megakernel.scheduler import MicroOpType

    config = NoBubblesConfig(num_pages=4)
    scheduler = PageAwareScheduler(config)

    # 4 independent operations
    ops = [OpDescriptor(f"Op{i}", i, reads={f"in{i}"}, writes={f"out{i}"}) for i in range(4)]

    scheduler.generate_async_pipeline_schedule(ops, pages_per_op=2)

    # Check we have the right micro-op types
    op_types = [op.type for op in scheduler.micro_ops]

    # Should have LOAD_ASYNC, COMMIT_GROUP, WAIT_LOAD, COMPUTE, STORE_ASYNC
    assert MicroOpType.LOAD_ASYNC in op_types
    assert MicroOpType.COMMIT_GROUP in op_types
    assert MicroOpType.WAIT_LOAD in op_types
    assert MicroOpType.COMPUTE in op_types
    assert MicroOpType.STORE_ASYNC in op_types

    # First two loads should be issued before first compute (max_concurrent=2)
    load_async_count_before_compute = 0
    for op in scheduler.micro_ops:
        if op.type == MicroOpType.COMPUTE:
            break
        if op.type == MicroOpType.LOAD_ASYNC:
            load_async_count_before_compute += 1

    assert load_async_count_before_compute == 2, (
        f"Expected 2 async loads before first compute, got {load_async_count_before_compute}"
    )

    print("Async pipeline schedule:")
    print(scheduler.visualize_page_schedule())


if __name__ == "__main__":
    test_reads_decorator()
    test_writes_decorator()
    test_independent_decorator()
    test_build_op_descriptor()
    test_simple_chain_with_dependency()
    test_independent_ops_parallel()
    test_three_op_mixed_deps()
    test_topological_sort()
    test_visualize_schedule()
    test_from_kernel_integration()
    test_paged_allocator_basic()
    test_paged_allocator_blocking_ops()
    test_page_aware_scheduler_basic()
    test_page_aware_scheduler_page_contention()
    test_page_aware_scheduler_visualization()
    test_async_pipeline_schedule()
    print("\nAll scheduler tests passed!")
