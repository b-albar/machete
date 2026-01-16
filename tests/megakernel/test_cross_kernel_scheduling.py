# Copyright (c) 2025, Machete Authors
"""
Tests for cross-kernel scheduling and fine-grained inter-operation dependencies.

Tests the "No Bubbles" optimization where:
1. Load[i+1] can start during Compute[i] if there's no data dependency
2. Dependencies are at logical block granularity, not kernel level
3. Async loads enable maximum overlap
4. Dimension mapping supports reduction (many-to-one) and broadcast (one-to-many)
"""

import pytest
from machete.megakernel.scheduler import (
    # Core classes
    NoBubblesScheduler,
    NoBubblesConfig,
    OpDescriptor,
    MicroOpType,
    # Decorators
    reads,
    writes,
    async_load,
    prefetchable,
    depends_on,
    warp_role,
    WarpRole,
    # Helper functions
    build_op_descriptor_from_kernel,
    get_method_dependencies,
    is_async_load,
    get_prefetchable_regions,
    get_cross_op_dependencies,
    # Cross-kernel types
    DependencyGranularity,
    InterOpDependency,
    DimensionMapping,
    GlobalBarrierTensor,
    PrefetchDescriptor,
)


# =============================================================================
# Test Decorators
# =============================================================================


class TestDecorators:
    """Test the decorator functions for declaring dependencies."""

    def test_reads_decorator(self):
        """Test @reads decorator extracts read dependencies."""

        @reads("input", "weights")
        def load_fn():
            pass

        reads_set, writes_set = get_method_dependencies(load_fn)
        assert reads_set == {"input", "weights"}
        assert writes_set == set()

    def test_writes_decorator(self):
        """Test @writes decorator extracts write dependencies."""

        @writes("output")
        def store_fn():
            pass

        reads_set, writes_set = get_method_dependencies(store_fn)
        assert reads_set == set()
        assert writes_set == {"output"}

    def test_combined_decorators(self):
        """Test combining @reads and @writes decorators."""

        @reads("input")
        @writes("smem")
        def compute_fn():
            pass

        reads_set, writes_set = get_method_dependencies(compute_fn)
        assert reads_set == {"input"}
        assert writes_set == {"smem"}

    def test_async_load_decorator(self):
        """Test @async_load decorator marks load as non-blocking."""

        @async_load()
        @reads("weights")
        def load_fn():
            pass

        assert is_async_load(load_fn)

        # Non-decorated function should return False
        def regular_load():
            pass

        assert not is_async_load(regular_load)

    def test_prefetchable_decorator(self):
        """Test @prefetchable decorator marks regions for early loading."""

        @prefetchable("weights", "bias")
        @async_load()
        @reads("weights", "bias", "activations")
        def load_fn():
            pass

        prefetch_regions = get_prefetchable_regions(load_fn)
        assert prefetch_regions == {"weights", "bias"}
        # activations not in prefetchable (depends on previous op)

    def test_depends_on_decorator(self):
        """Test @depends_on decorator declares cross-op dependencies."""

        @depends_on("qkv_projection", granularity="logical_block")
        @depends_on("layer_norm", granularity="kernel")
        @reads("Q", "K", "V")
        def load_attention():
            pass

        deps = get_cross_op_dependencies(load_attention)
        assert len(deps) == 2
        assert deps[0].op_name == "qkv_projection"
        assert deps[0].granularity == "logical_block"
        assert deps[1].op_name == "layer_norm"
        assert deps[1].granularity == "kernel"

    def test_depends_on_with_dims(self):
        """Test @depends_on with producer_dims/consumer_dims for dimension mapping."""

        @depends_on(
            "attention",
            producer_dims=(2, 8, 512),  # (batch, head, seq)
            consumer_dims=(2, 8),  # (batch, head)
        )
        @reads("attn_output")
        def load_output_proj():
            pass

        deps = get_cross_op_dependencies(load_output_proj)
        assert len(deps) == 1
        assert deps[0].op_name == "attention"
        assert deps[0].producer_dims == (2, 8, 512)
        assert deps[0].consumer_dims == (2, 8)

        # Should create a DimensionMapping
        dim_mapping = deps[0].to_dimension_mapping()
        assert dim_mapping is not None
        assert dim_mapping.is_reduction
        assert dim_mapping.reduction_axes == (2,)

    def test_warp_role_decorator(self):
        """Test @warp_role decorator assigns warp specialization."""
        from machete.megakernel.scheduler import get_method_warp_role

        @warp_role(WarpRole.LOADER)
        def load_fn():
            pass

        @warp_role(WarpRole.CONSUMER)
        def compute_fn():
            pass

        @warp_role(WarpRole.STORER)
        def store_fn():
            pass

        assert get_method_warp_role(load_fn) == WarpRole.LOADER
        assert get_method_warp_role(compute_fn) == WarpRole.CONSUMER
        assert get_method_warp_role(store_fn) == WarpRole.STORER


# =============================================================================
# Test DimensionMapping
# =============================================================================


class TestDimensionMapping:
    """Test the DimensionMapping class for reduction and broadcast patterns."""

    def test_one_to_one_mapping(self):
        """Test 1:1 mapping with same dimensions."""
        mapping = DimensionMapping(
            producer_dims=(2, 8),
            consumer_dims=(2, 8),
        )

        assert mapping.is_one_to_one
        assert not mapping.is_reduction
        assert not mapping.is_broadcast

        # Block 5 depends on block 5
        assert mapping.get_producer_blocks(5) == [5]

    def test_reduction_trailing_axis(self):
        """Test reduction over trailing axis (most common case)."""
        # Producer: (batch, head, seq) -> Consumer: (batch, head)
        mapping = DimensionMapping(
            producer_dims=(2, 8, 512),
            consumer_dims=(2, 8),
        )

        assert mapping.is_reduction
        assert not mapping.is_broadcast
        assert mapping.reduction_axes == (2,)
        assert mapping.blocks_per_consumer == 512

        # Consumer block 0 (batch=0, head=0) depends on producer blocks 0..511
        producer_blocks = mapping.get_producer_blocks(0)
        assert len(producer_blocks) == 512
        assert producer_blocks[0] == 0
        assert producer_blocks[511] == 511

        # Consumer block 1 (batch=0, head=1) depends on producer blocks 512..1023
        producer_blocks = mapping.get_producer_blocks(1)
        assert len(producer_blocks) == 512
        assert producer_blocks[0] == 512
        assert producer_blocks[511] == 1023

        # Contiguous range optimization
        start, end, stride = mapping.get_producer_block_range(0)
        assert start == 0
        assert end == 512
        assert stride == 1

        start, end, stride = mapping.get_producer_block_range(1)
        assert start == 512
        assert end == 1024
        assert stride == 1

    def test_broadcast_trailing_axis(self):
        """Test broadcast over trailing axis."""
        # Producer: (batch, head) -> Consumer: (batch, head, seq)
        mapping = DimensionMapping(
            producer_dims=(2, 8),
            consumer_dims=(2, 8, 512),
        )

        assert mapping.is_broadcast
        assert not mapping.is_reduction
        assert mapping.broadcast_axes == (2,)
        assert mapping.blocks_per_consumer == 1  # Each consumer waits for 1 producer

        # Consumer block 0 (batch=0, head=0, seq=0) depends on producer block 0
        assert mapping.get_producer_blocks(0) == [0]

        # Consumer block 511 (batch=0, head=0, seq=511) depends on producer block 0
        assert mapping.get_producer_blocks(511) == [0]

        # Consumer block 512 (batch=0, head=1, seq=0) depends on producer block 1
        assert mapping.get_producer_blocks(512) == [1]

    def test_infer_granularity(self):
        """Test automatic granularity inference."""
        # 1:1 -> LOGICAL_BLOCK
        one_to_one = DimensionMapping(
            producer_dims=(2, 8),
            consumer_dims=(2, 8),
        )
        assert one_to_one.infer_granularity() == DependencyGranularity.LOGICAL_BLOCK

        # Reduction -> REDUCTION
        reduction = DimensionMapping(
            producer_dims=(2, 8, 512),
            consumer_dims=(2, 8),
        )
        assert reduction.infer_granularity() == DependencyGranularity.REDUCTION

        # Broadcast -> BROADCAST
        broadcast = DimensionMapping(
            producer_dims=(2, 8),
            consumer_dims=(2, 8, 512),
        )
        assert broadcast.infer_granularity() == DependencyGranularity.BROADCAST

    def test_dimension_mismatch_error(self):
        """Test that mismatched dimensions raise an error."""
        with pytest.raises(ValueError):
            DimensionMapping(
                producer_dims=(2, 8, 512),
                consumer_dims=(4, 16),  # Doesn't match after reduction
            )


# =============================================================================
# Test InterOpDependency
# =============================================================================


class TestInterOpDependency:
    """Test the InterOpDependency class."""

    def test_default_logical_mapping(self):
        """Test default 1:1 logical block mapping."""
        dep = InterOpDependency(
            producer_op="op1",
            consumer_op="op2",
            granularity=DependencyGranularity.LOGICAL_BLOCK,
        )

        # Default: consumer block i depends on producer block i
        assert dep.get_producer_logical_block(0) == 0
        assert dep.get_producer_logical_block(5) == 5
        assert dep.get_producer_logical_block(100) == 100

    def test_custom_logical_mapping(self):
        """Test custom logical block mapping (e.g., for broadcast)."""
        # Example: all consumer blocks depend on producer block 0 (broadcast)
        dep = InterOpDependency(
            producer_op="broadcast_op",
            consumer_op="consumer_op",
            granularity=DependencyGranularity.LOGICAL_BLOCK,
            logical_mapping=lambda consumer_id: 0,  # All depend on block 0
        )

        assert dep.get_producer_logical_block(0) == 0
        assert dep.get_producer_logical_block(5) == 0
        assert dep.get_producer_logical_block(100) == 0

    def test_reduction_with_dim_mapping(self):
        """Test reduction dependency with DimensionMapping."""
        dep = InterOpDependency(
            producer_op="attention",
            consumer_op="output_proj",
            dim_mapping=DimensionMapping(
                producer_dims=(2, 8, 512),
                consumer_dims=(2, 8),
            ),
        )

        # Granularity should be auto-inferred
        assert dep.granularity == DependencyGranularity.REDUCTION

        # Consumer block 0 depends on 512 producer blocks
        producer_blocks = dep.get_producer_logical_blocks(0)
        assert len(producer_blocks) == 512
        assert dep.get_wait_count(0) == 512

        # Contiguous check
        assert dep.is_contiguous_reduction()

    def test_broadcast_with_dim_mapping(self):
        """Test broadcast dependency with DimensionMapping."""
        dep = InterOpDependency(
            producer_op="weights",
            consumer_op="attention",
            dim_mapping=DimensionMapping(
                producer_dims=(2, 8),
                consumer_dims=(2, 8, 512),
            ),
        )

        # Granularity should be auto-inferred
        assert dep.granularity == DependencyGranularity.BROADCAST

        # Consumer block 0 depends on producer block 0
        assert dep.get_producer_logical_blocks(0) == [0]
        assert dep.get_wait_count(0) == 1


# =============================================================================
# Test GlobalBarrierTensor
# =============================================================================


class TestGlobalBarrierTensor:
    """Test the GlobalBarrierTensor configuration."""

    def test_shape_and_counters(self):
        """Test barrier tensor shape calculation."""
        barrier = GlobalBarrierTensor(
            num_ops=4,
            total_logical_blocks=32,
        )

        assert barrier.shape == (4, 32)
        assert barrier.num_counters == 128


# =============================================================================
# Test NoBubblesScheduler with Cross-Kernel Features
# =============================================================================


class TestNoBubblesSchedulerCrossKernel:
    """Test NoBubblesScheduler with cross-kernel scheduling features."""

    def test_add_operations_and_dependencies(self):
        """Test adding operations and dependencies to scheduler."""
        scheduler = NoBubblesScheduler()

        op1 = OpDescriptor(name="qkv_proj", op_idx=0, logical_grid_size=8)
        op2 = OpDescriptor(name="attention", op_idx=1, logical_grid_size=8)

        scheduler.add_operation(op1)
        scheduler.add_operation(op2)

        assert len(scheduler.operations) == 2

        # Add dependency
        scheduler.add_dependency(
            InterOpDependency(
                producer_op="qkv_proj",
                consumer_op="attention",
                granularity=DependencyGranularity.LOGICAL_BLOCK,
            )
        )

        assert len(scheduler.inter_op_dependencies) == 1

    def test_build_cross_kernel_dependency_graph(self):
        """Test building the cross-kernel dependency graph."""
        scheduler = NoBubblesScheduler()

        scheduler.add_operation(OpDescriptor(name="op1", op_idx=0))
        scheduler.add_operation(OpDescriptor(name="op2", op_idx=1))
        scheduler.add_operation(OpDescriptor(name="op3", op_idx=2))

        scheduler.add_dependency(
            InterOpDependency(producer_op="op1", consumer_op="op2")
        )
        scheduler.add_dependency(
            InterOpDependency(
                producer_op="op2",
                consumer_op="op3",
                granularity=DependencyGranularity.KERNEL,
            )
        )

        graph = scheduler.build_cross_kernel_dependency_graph()

        assert len(graph[0]) == 0  # op1 has no deps
        assert len(graph[1]) == 1  # op2 depends on op1
        assert len(graph[2]) == 1  # op3 depends on op2

    def test_get_producer_blocks_for_consumer(self):
        """Test getting producer blocks for a consumer with reduction."""
        scheduler = NoBubblesScheduler()

        scheduler.add_operation(
            OpDescriptor(name="attention", op_idx=0, logical_grid_size=8192)
        )
        scheduler.add_operation(
            OpDescriptor(name="output_proj", op_idx=1, logical_grid_size=16)
        )

        scheduler.add_dependency(
            InterOpDependency(
                producer_op="attention",
                consumer_op="output_proj",
                dim_mapping=DimensionMapping(
                    producer_dims=(2, 8, 512),
                    consumer_dims=(2, 8),
                ),
            )
        )

        # Consumer block 0 should wait for producer blocks 0..511
        producer_map = scheduler.get_producer_blocks_for_consumer(1, 0)
        assert 0 in producer_map
        assert len(producer_map[0]) == 512

    def test_can_prefetch_op(self):
        """Test checking if an operation can prefetch."""
        scheduler = NoBubblesScheduler()

        scheduler.add_operation(OpDescriptor(name="op1", op_idx=0))
        scheduler.add_operation(OpDescriptor(name="op2", op_idx=1))
        scheduler.add_operation(OpDescriptor(name="op3", op_idx=2))

        # op2 depends on op1 with fine-grained sync -> can prefetch
        scheduler.add_dependency(
            InterOpDependency(
                producer_op="op1",
                consumer_op="op2",
                granularity=DependencyGranularity.LOGICAL_BLOCK,
            )
        )

        # op3 depends on op2 with coarse sync -> cannot prefetch
        scheduler.add_dependency(
            InterOpDependency(
                producer_op="op2",
                consumer_op="op3",
                granularity=DependencyGranularity.KERNEL,
            )
        )

        assert scheduler.can_prefetch_op(0) is True  # No deps
        assert scheduler.can_prefetch_op(1) is True  # Fine-grained
        assert scheduler.can_prefetch_op(2) is False  # Kernel-level

    def test_schedule_with_async_ops(self):
        """Test generating a schedule with async load operations."""
        config = NoBubblesConfig(num_pages=4)
        scheduler = NoBubblesScheduler(config)

        ops = [
            OpDescriptor(
                name="qkv_proj",
                op_idx=0,
                reads={"input"},
                writes={"Q", "K", "V"},
                async_load=True,
                prefetchable_regions={"weights"},
                logical_grid_size=8,
            ),
            OpDescriptor(
                name="attention",
                op_idx=1,
                reads={"Q", "K", "V"},
                writes={"attn_out"},
                async_load=True,
                prefetchable_regions={"attn_weights"},
                logical_grid_size=8,
            ),
        ]

        # Generate schedule
        micro_ops = scheduler.generate_page_aware_schedule(ops, pages_per_op=2)

        # Verify schedule was generated
        assert len(micro_ops) > 0

        # Check that we have the expected operation types
        op_types = [op.type for op in micro_ops]
        assert MicroOpType.LOAD in op_types
        assert MicroOpType.COMPUTE in op_types
        assert MicroOpType.STORE in op_types


# =============================================================================
# Test Transformer-like Pipeline with Dimension Mapping
# =============================================================================


class TestTransformerPipelineWithDimMapping:
    """Test scheduling a transformer pipeline with reduction/broadcast patterns.

    This simulates:
    LayerNorm -> QKV Projection -> Attention (per seq) -> Output Projection

    With dimension mappings:
    - Attention produces (batch, head, seq) = 8192 blocks
    - Output proj consumes (batch, head) = 16 blocks
    - Output proj must wait for ALL seq blocks per (batch, head)
    """

    def test_attention_to_output_proj_reduction(self):
        """Test reduction dependency from attention to output projection."""
        scheduler = NoBubblesScheduler()

        # Attention: (batch=2, head=8, seq=512) = 8192 logical blocks
        attention = OpDescriptor(
            name="attention",
            op_idx=0,
            reads={"Q", "K", "V"},
            writes={"attn_out"},
            logical_grid_size=2 * 8 * 512,
        )

        # Output projection: (batch=2, head=8) = 16 logical blocks
        output_proj = OpDescriptor(
            name="output_proj",
            op_idx=1,
            reads={"attn_out"},
            writes={"projected"},
            logical_grid_size=2 * 8,
        )

        scheduler.add_operation(attention)
        scheduler.add_operation(output_proj)

        # Add reduction dependency
        scheduler.add_dependency(
            InterOpDependency(
                producer_op="attention",
                consumer_op="output_proj",
                dim_mapping=DimensionMapping(
                    producer_dims=(2, 8, 512),
                    consumer_dims=(2, 8),
                ),
            )
        )

        # Build graph
        scheduler.build_cross_kernel_dependency_graph()

        # Output proj block 0 (batch=0, head=0) waits for attention blocks 0..511
        wait_counts = scheduler.get_wait_count_for_consumer(1, 0)
        assert wait_counts[0] == 512

        # Output proj block 1 (batch=0, head=1) waits for attention blocks 512..1023
        producer_blocks = scheduler.get_producer_blocks_for_consumer(1, 1)
        assert len(producer_blocks[0]) == 512
        assert producer_blocks[0][0] == 512
        assert producer_blocks[0][-1] == 1023

    def test_weights_broadcast_to_attention(self):
        """Test broadcast dependency from weights to attention."""
        scheduler = NoBubblesScheduler()

        # Weights: (batch=2, head=8) = 16 logical blocks
        weights = OpDescriptor(
            name="weights",
            op_idx=0,
            writes={"W_qkv"},
            logical_grid_size=2 * 8,
        )

        # Attention: (batch=2, head=8, seq=512) = 8192 logical blocks
        attention = OpDescriptor(
            name="attention",
            op_idx=1,
            reads={"W_qkv", "input"},
            writes={"attn_out"},
            logical_grid_size=2 * 8 * 512,
        )

        scheduler.add_operation(weights)
        scheduler.add_operation(attention)

        # Add broadcast dependency
        scheduler.add_dependency(
            InterOpDependency(
                producer_op="weights",
                consumer_op="attention",
                dim_mapping=DimensionMapping(
                    producer_dims=(2, 8),
                    consumer_dims=(2, 8, 512),
                ),
            )
        )

        # Build graph
        scheduler.build_cross_kernel_dependency_graph()

        # Attention block 0 (batch=0, head=0, seq=0) waits for weights block 0
        producer_blocks = scheduler.get_producer_blocks_for_consumer(1, 0)
        assert producer_blocks[0] == [0]

        # Attention block 511 (batch=0, head=0, seq=511) waits for weights block 0
        producer_blocks = scheduler.get_producer_blocks_for_consumer(1, 511)
        assert producer_blocks[0] == [0]

        # Attention block 512 (batch=0, head=1, seq=0) waits for weights block 1
        producer_blocks = scheduler.get_producer_blocks_for_consumer(1, 512)
        assert producer_blocks[0] == [1]

        # Each attention block waits for only 1 weight block
        wait_counts = scheduler.get_wait_count_for_consumer(1, 0)
        assert wait_counts[0] == 1


# =============================================================================
# Test SchedulingMode and MixedModeScheduler
# =============================================================================


class TestSchedulingMode:
    """Test SchedulingMode enum and per-operation mode detection."""

    def test_scheduling_mode_enum(self):
        """Test SchedulingMode enum values."""
        from machete.megakernel.scheduler import SchedulingMode

        assert SchedulingMode.SEQUENTIAL is not None
        assert SchedulingMode.ASYNC is not None
        assert SchedulingMode.WARP_SPECIALIZED is not None

    def test_op_descriptor_default_mode(self):
        """Test that OpDescriptor defaults to SEQUENTIAL mode."""
        from machete.megakernel.scheduler import SchedulingMode

        desc = OpDescriptor(name="test", op_idx=0)
        assert desc.scheduling_mode == SchedulingMode.SEQUENTIAL

    def test_op_descriptor_infer_warp_specialized(self):
        """Test that warp specialization is detected."""
        from machete.megakernel.scheduler import SchedulingMode

        desc = OpDescriptor(name="test", op_idx=0, uses_warp_specialization=True)
        assert desc.infer_scheduling_mode() == SchedulingMode.WARP_SPECIALIZED

    def test_op_descriptor_infer_async(self):
        """Test that async load is detected."""
        from machete.megakernel.scheduler import SchedulingMode

        desc = OpDescriptor(name="test", op_idx=0, async_load=True)
        assert desc.infer_scheduling_mode() == SchedulingMode.ASYNC

    def test_build_op_descriptor_sets_mode(self):
        """Test that build_op_descriptor_from_kernel sets scheduling mode."""
        from machete.megakernel.scheduler import SchedulingMode

        # Create a mock kernel with warp specialization
        class WarpSpecKernel:
            uses_warp_specialization = True

            def load_forward(self):
                pass

            def compute_forward(self):
                pass

            def store_forward(self):
                pass

        kernel = WarpSpecKernel()
        desc = build_op_descriptor_from_kernel(kernel, 0, "forward")

        # Should infer WARP_SPECIALIZED mode
        assert desc.scheduling_mode == SchedulingMode.WARP_SPECIALIZED


class TestMixedModeScheduler:
    """Test MixedModeScheduler for heterogeneous kernel support."""

    def test_group_ops_by_mode(self):
        """Test grouping consecutive ops by scheduling mode."""
        from machete.megakernel.scheduler import MixedModeScheduler, SchedulingMode

        scheduler = MixedModeScheduler()

        # Create ops with different modes
        ops = [
            OpDescriptor(name="op1", op_idx=0),  # SEQUENTIAL
            OpDescriptor(name="op2", op_idx=1),  # SEQUENTIAL
            OpDescriptor(name="op3", op_idx=2, uses_warp_specialization=True),  # WARP_SPEC
            OpDescriptor(name="op4", op_idx=3, uses_warp_specialization=True),  # WARP_SPEC
            OpDescriptor(name="op5", op_idx=4),  # SEQUENTIAL
        ]

        # Update modes
        for desc in ops:
            desc.scheduling_mode = desc.infer_scheduling_mode()

        groups = scheduler._group_ops_by_mode(ops)

        assert len(groups) == 3
        assert groups[0] == (SchedulingMode.SEQUENTIAL, [0, 1])
        assert groups[1] == (SchedulingMode.WARP_SPECIALIZED, [2, 3])
        assert groups[2] == (SchedulingMode.SEQUENTIAL, [4])

    def test_single_mode_no_grouping(self):
        """Test that single-mode operations form one group."""
        from machete.megakernel.scheduler import MixedModeScheduler, SchedulingMode

        scheduler = MixedModeScheduler()

        ops = [
            OpDescriptor(name="op1", op_idx=0),
            OpDescriptor(name="op2", op_idx=1),
            OpDescriptor(name="op3", op_idx=2),
        ]

        for desc in ops:
            desc.scheduling_mode = desc.infer_scheduling_mode()

        groups = scheduler._group_ops_by_mode(ops)

        assert len(groups) == 1
        assert groups[0] == (SchedulingMode.SEQUENTIAL, [0, 1, 2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
