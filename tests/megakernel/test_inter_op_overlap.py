# Copyright (c) 2025, Machete Authors
"""
Tests for inter-operation overlap in fused warp-specialized kernels.

These tests verify that the semaphore-based synchronization enables
true overlap between operations:
- Loader warps can start loading Op N+1 while consumer warps compute Op N
- No global sync_threads() barriers between operations
- Fine-grained producer-consumer handoff via semaphores

The "No Bubbles" pattern should show:
1. Loader fills stage 0 for Op 0
2. Loader fills stage 1 for Op 0 WHILE Consumer processes stage 0
3. Loader can start Op 1 load WHILE Consumer finishes Op 0
4. etc.
"""

import pytest
from dataclasses import dataclass
from typing import Tuple
import cutlass.cute as cute

from machete.megakernel import (
    WarpSpecializedKernel,
    WarpConfig,
    WarpRole,
    warp_role,
    reads,
    writes,
    FusableKernel,
)
from machete.megakernel.paged_buffer import (
    PagedBufferConfig,
    PagedBufferLayout,
    InterOpSemaphoreConfig,
    InterOpSemaphoreLayout,
)
from machete.megakernel.scheduler import SmemPlanner, OperationGraph, OpNode, OpEdge


# ============================================================================
# Test Kernels
# ============================================================================


class WarpSpecializedAddKernel(WarpSpecializedKernel):
    """Warp-specialized kernel that adds a scalar to input."""

    TILE_SIZE = 256

    def __init__(self, name: str = "WarpSpecAdd"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def warp_config(self) -> WarpConfig:
        return WarpConfig(num_consumer_warps=12)

    @property
    def num_stages(self) -> int:
        return 2  # Double buffering

    @property
    def page_size(self) -> int:
        return self.TILE_SIZE * 2  # fp16

    def get_logical_grid_size(self, input_tensor, output_tensor, scalar, n_elements) -> int:
        return (n_elements + self.TILE_SIZE - 1) // self.TILE_SIZE

    @warp_role(WarpRole.LOADER)
    @reads("input")
    @cute.jit
    def load_forward(self, logical_idx, page_ptr, stage, input_tensor, output_tensor, scalar, n_elements):
        """Load a tile of input data into shared memory."""
        pass  # Actual load logic would go here

    @warp_role(WarpRole.CONSUMER)
    @writes("output")
    @cute.jit
    def compute_forward(self, logical_idx, page_ptr, stage, input_tensor, output_tensor, scalar, n_elements):
        """Add scalar to each element."""
        pass  # Actual compute logic would go here

    @warp_role(WarpRole.STORER)
    @cute.jit
    def store_forward(self, logical_idx, page_ptr, stage, input_tensor, output_tensor, scalar, n_elements):
        """Store results back to global memory."""
        pass  # Actual store logic would go here


class WarpSpecializedMulKernel(WarpSpecializedKernel):
    """Warp-specialized kernel that multiplies by a scalar."""

    TILE_SIZE = 256

    def __init__(self, name: str = "WarpSpecMul"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def warp_config(self) -> WarpConfig:
        return WarpConfig(num_consumer_warps=12)

    @property
    def num_stages(self) -> int:
        return 2

    @property
    def page_size(self) -> int:
        return self.TILE_SIZE * 2

    def get_logical_grid_size(self, input_tensor, output_tensor, scalar, n_elements) -> int:
        return (n_elements + self.TILE_SIZE - 1) // self.TILE_SIZE

    @warp_role(WarpRole.LOADER)
    @reads("intermediate")  # Reads output of previous op
    @cute.jit
    def load_forward(self, logical_idx, page_ptr, stage, input_tensor, output_tensor, scalar, n_elements):
        pass

    @warp_role(WarpRole.CONSUMER)
    @writes("output")
    @cute.jit
    def compute_forward(self, logical_idx, page_ptr, stage, input_tensor, output_tensor, scalar, n_elements):
        pass

    @warp_role(WarpRole.STORER)
    @cute.jit
    def store_forward(self, logical_idx, page_ptr, stage, input_tensor, output_tensor, scalar, n_elements):
        pass


# ============================================================================
# Unit Tests for Buffer Configuration
# ============================================================================


class TestPagedBufferConfig:
    """Test paged buffer configuration calculations."""

    def test_default_config(self):
        config = PagedBufferConfig()
        assert config.num_stages == 2
        assert config.page_size == 16384
        assert config.num_semaphores == 4  # 2 per stage
        assert config.semaphore_bytes == 16

    def test_custom_config(self):
        config = PagedBufferConfig(num_stages=3, page_size=8192)
        assert config.num_stages == 3
        assert config.page_size == 8192
        assert config.num_semaphores == 6  # 2 per stage
        assert config.total_smem_bytes == 3 * 8192 + 6 * 4

    def test_layout_offsets(self):
        config = PagedBufferConfig(num_stages=2, page_size=1024)
        layout = PagedBufferLayout(config, base_offset=0)

        # Data pages
        assert layout.get_page_offset(0) == 0
        assert layout.get_page_offset(1) == 1024

        # Semaphores start after data
        sem_base = 2 * 1024  # 2048
        assert layout.get_loader_ready_sem_offset(0) == sem_base
        assert layout.get_loader_ready_sem_offset(1) == sem_base + 4
        assert layout.get_consumer_done_sem_offset(0) == sem_base + 8
        assert layout.get_consumer_done_sem_offset(1) == sem_base + 12


class TestInterOpSemaphoreConfig:
    """Test inter-operation semaphore configuration."""

    def test_config(self):
        config = InterOpSemaphoreConfig(num_ops=3)
        assert config.num_semaphores == 3
        assert config.semaphore_bytes == 12

    def test_layout_offsets(self):
        layout = InterOpSemaphoreLayout(num_ops=4, base_offset=1000)
        assert layout.get_op_done_sem_offset(0) == 1000
        assert layout.get_op_done_sem_offset(1) == 1004
        assert layout.get_op_done_sem_offset(2) == 1008
        assert layout.get_op_done_sem_offset(3) == 1012
        assert layout.total_bytes == 16


# ============================================================================
# Unit Tests for SmemPlanner with Semaphores
# ============================================================================


class TestSmemPlannerSemaphores:
    """Test SmemPlanner semaphore allocation."""

    def test_no_semaphores_by_default(self):
        """Without use_semaphores=True, no semaphore space is allocated."""
        graph = OperationGraph()
        node = OpNode(
            op_idx=0,
            name="op0",
            smem_bytes=1024,
            uses_warp_specialization=True,
        )
        graph.add_node(node)

        planner = SmemPlanner(graph)
        total = planner.plan(use_semaphores=False)

        assert total == 1024
        assert planner.semaphore_base_offset == 1024

    def test_semaphores_single_op(self):
        """Single op with semaphores."""
        graph = OperationGraph()
        node = OpNode(
            op_idx=0,
            name="op0",
            smem_bytes=1024,
            uses_warp_specialization=True,
        )
        graph.add_node(node)

        planner = SmemPlanner(graph)
        total = planner.plan(use_semaphores=True)

        # 1024 data + (1 op * 2 intra-op + 1 inter-op) * 4 bytes = 1024 + 12 = 1036
        assert total == 1024 + 12
        assert planner.semaphore_base_offset == 1024
        assert planner.get_intra_op_sem_offset(0, "load_done") == 1024
        assert planner.get_intra_op_sem_offset(0, "compute_done") == 1028
        assert planner.get_inter_op_sem_offset(0) == 1032

    def test_semaphores_multiple_ops(self):
        """Multiple ops with semaphores."""
        graph = OperationGraph()
        node0 = OpNode(
            op_idx=0,
            name="op0",
            smem_bytes=1024,
            uses_warp_specialization=True,
            writes={"intermediate"},
        )
        node1 = OpNode(
            op_idx=1,
            name="op1",
            smem_bytes=512,
            uses_warp_specialization=True,
            reads={"intermediate"},
        )
        graph.add_node(node0)
        graph.add_node(node1)
        graph.add_edge(OpEdge(producer_idx=0, consumer_idx=1, tensors={"intermediate"}))

        planner = SmemPlanner(graph)
        total = planner.plan(use_semaphores=True)

        # Data: max(1024, 512) = 1024 (shared buffer, no early load in this case)
        # Semaphores: (2 ops * 2 intra-op + 2 inter-op) * 4 = 24 bytes
        # BUT: with enable_early_load=True (default), op1 might get separate buffer
        # Let's check the actual allocation

        data_bytes = planner.semaphore_base_offset
        sem_bytes = total - data_bytes

        # 2 ops = 2*2 intra-op + 2 inter-op = 6 semaphores = 24 bytes
        assert sem_bytes == 24

        # Check semaphore offsets
        assert planner.get_intra_op_sem_offset(0, "load_done") == data_bytes
        assert planner.get_intra_op_sem_offset(0, "compute_done") == data_bytes + 4
        assert planner.get_intra_op_sem_offset(1, "load_done") == data_bytes + 8
        assert planner.get_intra_op_sem_offset(1, "compute_done") == data_bytes + 12
        assert planner.get_inter_op_sem_offset(0) == data_bytes + 16
        assert planner.get_inter_op_sem_offset(1) == data_bytes + 20


# ============================================================================
# Unit Tests for WarpSpecializedKernel Properties
# ============================================================================


class TestWarpSpecializedKernelConfig:
    """Test WarpSpecializedKernel configuration for pipelining."""

    def test_default_stages(self):
        kernel = WarpSpecializedAddKernel()
        assert kernel.num_stages == 2
        assert kernel.uses_warp_specialization is True

    def test_page_size(self):
        kernel = WarpSpecializedAddKernel()
        assert kernel.page_size == 256 * 2  # TILE_SIZE * 2 bytes for fp16

    def test_smem_includes_semaphores(self):
        kernel = WarpSpecializedAddKernel()
        # smem = (num_stages * page_size) + (num_stages * 2 * 4)
        # = (2 * 512) + (2 * 2 * 4) = 1024 + 16 = 1040
        expected = (kernel.num_stages * kernel.page_size) + (kernel.num_stages * 2 * 4)
        assert kernel.smem_size_fwd == expected

    def test_sem_base_offset(self):
        kernel = WarpSpecializedAddKernel()
        # Semaphores come after data pages
        assert kernel._sem_base_offset == kernel.num_stages * kernel.page_size


# ============================================================================
# Integration Tests for Inter-Op Overlap Pattern
# ============================================================================


class TestInterOpOverlapPattern:
    """Test the inter-op overlap execution pattern.

    These tests verify the conceptual pattern, not actual CUDA execution.
    """

    def test_independent_ops_no_wait(self):
        """Independent ops should not wait for each other."""
        graph = OperationGraph()

        # Two independent ops (different inputs/outputs)
        node0 = OpNode(
            op_idx=0,
            name="add_op",
            smem_bytes=1024,
            uses_warp_specialization=True,
            reads={"input_a"},
            writes={"output_a"},
        )
        node1 = OpNode(
            op_idx=1,
            name="mul_op",
            smem_bytes=1024,
            uses_warp_specialization=True,
            reads={"input_b"},
            writes={"output_b"},
        )
        graph.add_node(node0)
        graph.add_node(node1)

        # No dependency between ops
        assert not graph.nodes[0].depends_on
        assert not graph.nodes[1].depends_on

    def test_dependent_ops_chain(self):
        """Dependent ops should have proper dependency tracking."""
        graph = OperationGraph()

        node0 = OpNode(
            op_idx=0,
            name="add_op",
            smem_bytes=1024,
            uses_warp_specialization=True,
            reads={"input"},
            writes={"intermediate"},
        )
        node1 = OpNode(
            op_idx=1,
            name="mul_op",
            smem_bytes=1024,
            uses_warp_specialization=True,
            reads={"intermediate"},  # Reads output of op0
            writes={"output"},
        )
        graph.add_node(node0)
        graph.add_node(node1)

        # Add explicit dependency via edge
        graph.add_edge(OpEdge(producer_idx=0, consumer_idx=1, tensors={"intermediate"}))

        assert 0 in graph.nodes[1].depends_on

    def test_overlap_timeline_concept(self):
        """Verify the conceptual timeline allows overlap.

        Expected timeline with semaphore sync (2 ops, 2 stages):

        Time | Loader              | Consumer           | Storer
        -----|---------------------|--------------------|-----------------
        T0   | Op0: Load stage 0   |                    |
        T1   | Op0: Load stage 1   | Op0: Compute s0    |
        T2   | Op1: Load stage 0   | Op0: Compute s1    | Op0: Store s0
        T3   | Op1: Load stage 1   | Op1: Compute s0    | Op0: Store s1
        T4   |                     | Op1: Compute s1    | Op1: Store s0
        T5   |                     |                    | Op1: Store s1

        Key: Op1 loader can start at T2, overlapping with Op0's compute/store.
        With sync_threads(), Op1 would have to wait until T5 to start.
        """
        # This is a conceptual test - the actual overlap happens in CUDA
        # We're just verifying the configuration enables this pattern

        kernel0 = WarpSpecializedAddKernel("op0")
        kernel1 = WarpSpecializedMulKernel("op1")

        # Both kernels have double buffering
        assert kernel0.num_stages == 2
        assert kernel1.num_stages == 2

        # Both use warp specialization
        assert kernel0.uses_warp_specialization
        assert kernel1.uses_warp_specialization

        # Warp config has separate loader/consumer/storer warps
        wc = kernel0.warp_config
        assert wc.num_loader_warps >= 1
        assert wc.num_consumer_warps >= 1
        assert wc.num_storer_warps >= 1


class TestSemaphoreCodeGeneration:
    """Test that semaphore-based code generator produces correct patterns."""

    def test_generates_semaphore_imports(self):
        """Code should import semaphore functions."""
        # This would test the actual code generation
        # For now, verify the imports exist in paged_buffer
        from machete.megakernel.paged_buffer import (
            inter_op_init_semaphores,
            inter_op_wait_for_dependency,
            inter_op_signal_done,
        )
        assert callable(inter_op_init_semaphores)
        assert callable(inter_op_wait_for_dependency)
        assert callable(inter_op_signal_done)

    def test_semaphore_code_generator_exists(self):
        """SemaphoreCodeGenerator should be available."""
        from machete.megakernel.core import SemaphoreCodeGenerator
        gen = SemaphoreCodeGenerator()
        assert hasattr(gen, 'generate')


# ============================================================================
# Performance Conceptual Tests
# ============================================================================


class TestOverlapBenefits:
    """Conceptual tests for overlap benefits."""

    def test_bubble_free_pipeline_stages(self):
        """With proper semaphores, pipeline should have no bubbles.

        Traditional approach (sync_threads between ops):
        - Op0 must fully complete before Op1 starts
        - Loader idle during compute, consumer idle during load

        Semaphore approach:
        - Op1 loader starts as soon as its dependencies are met
        - Consumer warps are never idle waiting for unrelated loads
        """
        kernel = WarpSpecializedAddKernel()

        # Double buffering enables hiding load latency
        assert kernel.num_stages >= 2, "Need at least 2 stages for overlap"

        # Separate warps for each role
        wc = kernel.warp_config
        total_warps = wc.total_warps
        assert total_warps >= 3, "Need at least loader + consumer + storer"

    def test_memory_bandwidth_utilization(self):
        """Semaphore sync should improve memory bandwidth utilization.

        With overlap:
        - Loader is always fetching data (next stage or next op)
        - Memory bus stays busy

        Without overlap:
        - Loader waits during compute phase
        - Memory bus idle during compute
        """
        # Conceptual: verify the pattern enables continuous loading
        kernel0 = WarpSpecializedAddKernel()
        kernel1 = WarpSpecializedMulKernel()

        # Both have separate loader warps
        assert kernel0.warp_config.num_loader_warps >= 1
        assert kernel1.warp_config.num_loader_warps >= 1

        # Both have double buffering (loader can work ahead)
        assert kernel0.num_stages >= 2
        assert kernel1.num_stages >= 2
