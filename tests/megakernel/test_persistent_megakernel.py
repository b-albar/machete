# Copyright (c) 2025, Machete Authors
"""
Tests for the persistent megakernel infrastructure.

This module tests:
1. Scheduler data structures (WarpConfig, PageConfig, BarrierConfig, Instruction)
2. InstructionScheduler dependency tracking
3. Megakernel single-op and multi-op execution
4. Dynamic page count calculation
"""

import pytest
import dataclasses
from typing import Tuple, Any

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False
    torch = None

import cutlass.cute as cute
from cutlass import Int32

from machete.megakernel.scheduler import (
    WarpRole,
    WarpConfig,
    PageConfig,
    PageSemaphores,
    BarrierConfig,
    Instruction,
    InstructionScheduler,
    TensorDependency,
    reads,
    writes,
    warp_role,
    async_load,
)
from machete.megakernel.interface import FusableKernel, WarpSpecializedKernel
from machete.megakernel.core import Megakernel


# ============================================================================
# Tests for WarpConfig
# ============================================================================


class TestWarpConfig:
    """Tests for WarpConfig data structure."""

    def test_default_config(self):
        """Test default warp configuration."""
        config = WarpConfig()
        assert config.num_consumer_warps == 12
        assert config.num_loader_warps == 1
        assert config.num_storer_warps == 1
        assert config.num_launcher_warps == 1
        assert config.num_controller_warps == 1

    def test_total_warps(self):
        """Test total warp count."""
        config = WarpConfig()
        assert config.total_warps == 16  # 12 + 1 + 1 + 1 + 1

    def test_total_threads(self):
        """Test total thread count."""
        config = WarpConfig()
        assert config.total_threads == 512  # 16 warps * 32 threads

    def test_custom_config(self):
        """Test custom warp configuration."""
        config = WarpConfig(num_consumer_warps=8, num_loader_warps=2)
        assert config.total_warps == 13  # 8 + 2 + 1 + 1 + 1
        assert config.total_threads == 416

    def test_get_role_consumer(self):
        """Test warp role mapping for consumer warps."""
        config = WarpConfig(num_consumer_warps=4)
        assert config.get_role(0) == WarpRole.CONSUMER
        assert config.get_role(1) == WarpRole.CONSUMER
        assert config.get_role(3) == WarpRole.CONSUMER

    def test_get_role_loader(self):
        """Test warp role mapping for loader warp."""
        config = WarpConfig(num_consumer_warps=4, num_loader_warps=1)
        assert config.get_role(4) == WarpRole.LOADER

    def test_get_role_storer(self):
        """Test warp role mapping for storer warp."""
        config = WarpConfig(num_consumer_warps=4, num_loader_warps=1, num_storer_warps=1)
        assert config.get_role(5) == WarpRole.STORER

    def test_get_role_launcher(self):
        """Test warp role mapping for launcher warp."""
        config = WarpConfig(num_consumer_warps=4)
        assert config.get_role(6) == WarpRole.LAUNCHER

    def test_get_role_controller(self):
        """Test warp role mapping for controller warp."""
        config = WarpConfig(num_consumer_warps=4)
        assert config.get_role(7) == WarpRole.CONTROLLER

    def test_warp_start_indices(self):
        """Test warp start index properties."""
        config = WarpConfig(num_consumer_warps=4)
        assert config.loader_warp_start == 4
        assert config.storer_warp_start == 5
        assert config.launcher_warp_start == 6
        assert config.controller_warp_start == 7


# ============================================================================
# Tests for PageConfig
# ============================================================================


class TestPageConfig:
    """Tests for PageConfig data structure."""

    def test_default_config(self):
        """Test default page configuration."""
        config = PageConfig()
        assert config.page_size == 16384  # 16KB
        assert config.alignment == 128
        assert config.reserved_smem == 4096

    def test_get_num_pages_h100(self):
        """Test page count for H100-like device (227KB)."""
        config = PageConfig(page_size=16384)
        h100_smem = 227 * 1024  # ~227KB
        num_pages = config.get_num_pages(h100_smem)
        # (227KB - 4KB reserved) / 16KB = ~13 pages
        assert num_pages == 13

    def test_get_num_pages_a100(self):
        """Test page count for A100-like device (164KB)."""
        config = PageConfig(page_size=16384)
        a100_smem = 164 * 1024  # ~164KB
        num_pages = config.get_num_pages(a100_smem)
        # (164KB - 4KB reserved) / 16KB = 10 pages
        assert num_pages == 10

    def test_get_num_pages_rtx4090(self):
        """Test page count for RTX 4090-like device (100KB)."""
        config = PageConfig(page_size=16384)
        rtx4090_smem = 100 * 1024  # ~100KB
        num_pages = config.get_num_pages(rtx4090_smem)
        # (100KB - 4KB reserved) / 16KB = 6 pages
        assert num_pages == 6

    def test_get_num_pages_minimum(self):
        """Test minimum page count is 1."""
        config = PageConfig(page_size=16384)
        tiny_smem = 8192  # Less than page_size + reserved
        num_pages = config.get_num_pages(tiny_smem)
        assert num_pages == 1  # Minimum of 1 page

    def test_get_total_smem_for_pages(self):
        """Test total shared memory calculation for pages."""
        config = PageConfig(page_size=16384)
        assert config.get_total_smem_for_pages(4) == 65536  # 4 * 16KB = 64KB
        assert config.get_total_smem_for_pages(8) == 131072  # 8 * 16KB = 128KB

    def test_custom_page_size(self):
        """Test custom page size configuration."""
        config = PageConfig(page_size=8192)  # 8KB pages
        h100_smem = 227 * 1024
        num_pages = config.get_num_pages(h100_smem)
        # (227KB - 4KB reserved) / 8KB = ~27 pages
        assert num_pages == 27


# ============================================================================
# Tests for BarrierConfig
# ============================================================================


class TestBarrierConfig:
    """Tests for BarrierConfig data structure."""

    def test_basic_config(self):
        """Test basic barrier configuration."""
        config = BarrierConfig(num_ops=3, num_chunks=8)
        assert config.num_ops == 3
        assert config.num_chunks == 8

    def test_total_size(self):
        """Test total barrier array size."""
        config = BarrierConfig(num_ops=3, num_chunks=8)
        assert config.total_size == 24  # 3 * 8

    def test_get_index(self):
        """Test barrier index computation."""
        config = BarrierConfig(num_ops=3, num_chunks=8)
        # Bar[op_id][chunk_id] = op_id * num_chunks + chunk_id
        assert config.get_index(0, 0) == 0
        assert config.get_index(0, 1) == 1
        assert config.get_index(0, 7) == 7
        assert config.get_index(1, 0) == 8
        assert config.get_index(1, 5) == 13
        assert config.get_index(2, 3) == 19


# ============================================================================
# Tests for Instruction
# ============================================================================


class TestInstruction:
    """Tests for Instruction encoding/decoding."""

    def test_basic_instruction(self):
        """Test basic instruction creation."""
        inst = Instruction(
            opcode=0,
            logical_idx=5,
            input_pages=[0],
            output_pages=[1],
            wait_barriers=[],
            signal_barriers=[0],
        )
        assert inst.opcode == 0
        assert inst.logical_idx == 5
        assert inst.input_pages == [0]
        assert inst.output_pages == [1]
        assert inst.local_deps == []
        assert inst.needs_page_wait is False

    def test_encode_simple(self):
        """Test encoding simple instruction."""
        inst = Instruction(
            opcode=1,
            logical_idx=3,
            input_pages=[0],
            output_pages=[1],
            wait_barriers=[],
            signal_barriers=[3],
        )
        encoded = inst.encode()
        # [opcode, logical_idx, flags, num_in, in_pages..., num_out, out_pages...,
        #  num_wait, wait_bars..., num_signal, signal_bars..., num_local, local_deps...]
        expected = [1, 3, 0, 1, 0, 1, 1, 0, 1, 3, 0, 0]
        assert encoded == expected

    def test_encode_with_dependencies(self):
        """Test encoding instruction with cross-block dependencies."""
        inst = Instruction(
            opcode=2,
            logical_idx=7,
            input_pages=[0, 1],
            output_pages=[2],
            wait_barriers=[0, 8],
            signal_barriers=[16],
            local_deps=[],
            needs_page_wait=False,
        )
        encoded = inst.encode()
        expected = [2, 7, 0, 2, 0, 1, 1, 2, 2, 0, 8, 1, 16, 0, 0]
        assert encoded == expected

    def test_encode_with_local_deps_and_backpressure(self):
        """Test encoding instruction with local deps and backpressure."""
        inst = Instruction(
            opcode=1,
            logical_idx=10,
            input_pages=[2],
            output_pages=[3],
            wait_barriers=[],
            signal_barriers=[10],
            local_deps=[0],
            needs_page_wait=True,
        )
        encoded = inst.encode()
        # flags = 1 (needs_page_wait)
        expected = [1, 10, 1, 1, 2, 1, 3, 0, 1, 10, 1, 0, 0]
        assert encoded == expected

    def test_decode_simple(self):
        """Test decoding simple instruction."""
        data = [1, 3, 0, 1, 0, 1, 1, 0, 1, 3, 0, 0]
        inst, next_pos = Instruction.decode(data)
        assert inst.opcode == 1
        assert inst.logical_idx == 3
        assert inst.input_pages == [0]
        assert inst.output_pages == [1]
        assert inst.wait_barriers == []
        assert inst.signal_barriers == [3]
        assert inst.local_deps == []
        assert inst.needs_page_wait is False
        assert next_pos == len(data)

    def test_decode_with_dependencies(self):
        """Test decoding instruction with cross-block dependencies."""
        data = [2, 7, 0, 2, 0, 1, 1, 2, 2, 0, 8, 1, 16, 0, 0]
        inst, next_pos = Instruction.decode(data)
        assert inst.opcode == 2
        assert inst.logical_idx == 7
        assert inst.input_pages == [0, 1]
        assert inst.output_pages == [2]
        assert inst.wait_barriers == [0, 8]
        assert inst.signal_barriers == [16]
        assert inst.local_deps == []
        assert inst.needs_page_wait is False
        assert next_pos == len(data)

    def test_decode_with_local_deps_and_backpressure(self):
        """Test decoding instruction with local deps and backpressure."""
        data = [1, 10, 1, 1, 2, 1, 3, 0, 1, 10, 1, 0, 0]
        inst, next_pos = Instruction.decode(data)
        assert inst.opcode == 1
        assert inst.logical_idx == 10
        assert inst.input_pages == [2]
        assert inst.output_pages == [3]
        assert inst.wait_barriers == []
        assert inst.signal_barriers == [10]
        assert inst.local_deps == [0]
        assert inst.needs_page_wait is True
        assert next_pos == len(data)

    def test_encode_decode_roundtrip(self):
        """Test encode-decode roundtrip."""
        original = Instruction(
            opcode=5,
            logical_idx=42,
            input_pages=[3, 4, 5],
            output_pages=[6, 7],
            wait_barriers=[10, 20, 30],
            signal_barriers=[40, 50],
            local_deps=[1, 2],
            needs_page_wait=True,
        )
        encoded = original.encode()
        decoded, _ = Instruction.decode(encoded)
        assert decoded.opcode == original.opcode
        assert decoded.logical_idx == original.logical_idx
        assert decoded.input_pages == original.input_pages
        assert decoded.output_pages == original.output_pages
        assert decoded.wait_barriers == original.wait_barriers
        assert decoded.signal_barriers == original.signal_barriers
        assert decoded.local_deps == original.local_deps
        assert decoded.needs_page_wait == original.needs_page_wait


# ============================================================================
# Tests for InstructionScheduler
# ============================================================================


class TestInstructionScheduler:
    """Tests for InstructionScheduler dependency tracking."""

    def test_single_op_schedule(self):
        """Test scheduling a single operation."""
        barrier_config = BarrierConfig(num_ops=1, num_chunks=4)
        page_config = PageConfig(page_size=16384)
        scheduler = InstructionScheduler(barrier_config, page_config, num_pages=8)

        scheduler.add_op(
            op_id=0,
            op=None,
            logical_grid_size=4,
            reads_list=["input"],
            writes_list=["output"],
        )

        instructions = scheduler.schedule()
        assert len(instructions) == 4  # One per chunk

        # Check signal barriers (all should signal their own Bar[0][chunk_id])
        for i, inst in enumerate(instructions):
            assert inst.opcode == 0
            assert inst.logical_idx == i
            assert inst.signal_barriers == [i]  # Bar[0][i]
            assert inst.wait_barriers == []  # No cross-block dependencies for first op
            assert inst.local_deps == []  # No local deps for first op

    def test_two_op_schedule_with_local_dependency(self):
        """Test scheduling two sequential ops - dependency should be local."""
        barrier_config = BarrierConfig(num_ops=2, num_chunks=4)
        page_config = PageConfig(page_size=16384)
        scheduler = InstructionScheduler(barrier_config, page_config, num_pages=8)

        # Op 0 writes "intermediate"
        scheduler.add_op(
            op_id=0,
            op=None,
            logical_grid_size=4,
            reads_list=["input"],
            writes_list=["intermediate"],
        )

        # Op 1 reads "intermediate" (depends on op 0)
        scheduler.add_op(
            op_id=1,
            op=None,
            logical_grid_size=4,
            reads_list=["intermediate"],
            writes_list=["output"],
        )

        instructions = scheduler.schedule()
        assert len(instructions) == 8  # 4 chunks * 2 ops

        # Check op 1 has LOCAL dependency on op 0 (not cross-block)
        for i in range(4):
            op0_inst = instructions[i]
            op1_inst = instructions[i + 4]

            # Op 0 signals Bar[0][i] (for potential cross-block consumers)
            assert op0_inst.signal_barriers == [i]
            assert op0_inst.local_deps == []

            # Op 1 has LOCAL dependency on op 0 (same block, sequential)
            assert op1_inst.wait_barriers == []  # No cross-block wait needed
            assert op1_inst.local_deps == [0]  # Local dep on op 0
            assert op1_inst.signal_barriers == [4 + i]  # Signal Bar[1][i]

    def test_page_allocation_round_robin(self):
        """Test that pages are allocated round-robin."""
        barrier_config = BarrierConfig(num_ops=1, num_chunks=10)
        page_config = PageConfig(page_size=16384)
        scheduler = InstructionScheduler(barrier_config, page_config, num_pages=4)

        scheduler.add_op(
            op_id=0,
            op=None,
            logical_grid_size=10,
            reads_list=[],
            writes_list=["output"],
        )

        instructions = scheduler.schedule()

        # Check round-robin page assignment
        for i, inst in enumerate(instructions):
            expected_input_page = i % 4
            expected_output_page = (i + 1) % 4
            assert inst.input_pages == [expected_input_page]
            assert inst.output_pages == [expected_output_page]

    def test_backpressure_tracking(self):
        """Test that backpressure is tracked when chunks exceed page count."""
        barrier_config = BarrierConfig(num_ops=1, num_chunks=10)
        page_config = PageConfig(page_size=16384)
        num_pages = 4
        scheduler = InstructionScheduler(barrier_config, page_config, num_pages=num_pages)

        scheduler.add_op(
            op_id=0,
            op=None,
            logical_grid_size=10,
            reads_list=[],
            writes_list=["output"],
        )

        instructions = scheduler.schedule()

        # First num_pages chunks don't need backpressure wait
        for i in range(num_pages):
            assert instructions[i].needs_page_wait is False

        # Chunks >= num_pages need backpressure wait (page may be in use)
        for i in range(num_pages, 10):
            assert instructions[i].needs_page_wait is True

    def test_analyze_dependencies(self):
        """Test dependency analysis."""
        barrier_config = BarrierConfig(num_ops=2, num_chunks=10)
        page_config = PageConfig(page_size=16384)
        num_pages = 4
        scheduler = InstructionScheduler(barrier_config, page_config, num_pages=num_pages)

        scheduler.add_op(
            op_id=0,
            op=None,
            logical_grid_size=10,
            reads_list=["input"],
            writes_list=["intermediate"],
        )
        scheduler.add_op(
            op_id=1,
            op=None,
            logical_grid_size=10,
            reads_list=["intermediate"],
            writes_list=["output"],
        )

        scheduler.schedule()
        analysis = scheduler.analyze_dependencies()

        assert analysis["total_instructions"] == 20  # 10 chunks * 2 ops
        assert analysis["local_dependencies"] == 10  # Op 1 has local dep on Op 0 for each chunk
        assert analysis["cross_block_dependencies"] == 0  # All deps are local
        assert analysis["backpressure_waits"] == 12  # Chunks 4-9 for both ops
        assert analysis["num_pages"] == 4

    def test_get_encoded_instructions(self):
        """Test getting flattened instruction buffer."""
        barrier_config = BarrierConfig(num_ops=1, num_chunks=2)
        page_config = PageConfig(page_size=16384)
        scheduler = InstructionScheduler(barrier_config, page_config, num_pages=8)

        scheduler.add_op(
            op_id=0,
            op=None,
            logical_grid_size=2,
            reads_list=[],
            writes_list=["output"],
        )

        scheduler.schedule()
        encoded = scheduler.get_encoded_instructions()

        # Should be a flat list of ints
        assert isinstance(encoded, list)
        assert all(isinstance(x, int) for x in encoded)
        assert len(encoded) > 0


class TestPageSemaphores:
    """Tests for PageSemaphores."""

    def test_semaphore_offsets(self):
        """Test semaphore offset calculations."""
        sems = PageSemaphores(num_pages=4)

        # sem_loaded at even indices, sem_consumed at odd
        assert sems.get_sem_loaded_offset(0) == 0
        assert sems.get_sem_consumed_offset(0) == 1
        assert sems.get_sem_loaded_offset(1) == 2
        assert sems.get_sem_consumed_offset(1) == 3
        assert sems.get_sem_loaded_offset(3) == 6
        assert sems.get_sem_consumed_offset(3) == 7

    def test_total_semaphores(self):
        """Test total semaphore count."""
        sems = PageSemaphores(num_pages=4)
        assert sems.total_semaphores == 8  # 2 per page

    def test_smem_size(self):
        """Test shared memory size for semaphores."""
        sems = PageSemaphores(num_pages=4)
        assert sems.smem_size == 64  # 8 semaphores * 8 bytes

    def test_page_config_creates_semaphores(self):
        """Test PageConfig creates semaphores."""
        config = PageConfig(page_size=16384)
        sems = config.get_semaphores(num_pages=8)
        assert sems.num_pages == 8
        assert sems.total_semaphores == 16


# ============================================================================
# Tests for Decorators
# ============================================================================


class TestDecorators:
    """Tests for dependency and warp role decorators."""

    def test_reads_decorator(self):
        """Test @reads decorator."""

        @reads("input", "weights")
        def dummy_load():
            pass

        assert hasattr(dummy_load, "_machete_deps")
        assert hasattr(dummy_load, "_machete_reads")
        assert dummy_load._machete_reads == {"input", "weights"}
        deps = dummy_load._machete_deps
        assert len(deps) == 2
        assert all(isinstance(d, TensorDependency) for d in deps)
        assert all(d.is_read for d in deps)

    def test_writes_decorator(self):
        """Test @writes decorator."""

        @writes("output")
        def dummy_store():
            pass

        assert hasattr(dummy_store, "_machete_deps")
        assert hasattr(dummy_store, "_machete_writes")
        assert dummy_store._machete_writes == {"output"}
        deps = dummy_store._machete_deps
        assert len(deps) == 1
        assert deps[0].name == "output"
        assert deps[0].is_read is False

    def test_combined_decorators(self):
        """Test combining @reads and @writes decorators."""

        @reads("input")
        @writes("output")
        def dummy_compute():
            pass

        assert hasattr(dummy_compute, "_machete_reads")
        assert hasattr(dummy_compute, "_machete_writes")
        assert dummy_compute._machete_reads == {"input"}
        assert dummy_compute._machete_writes == {"output"}

    def test_warp_role_decorator(self):
        """Test @warp_role decorator."""

        @warp_role(WarpRole.LOADER)
        def dummy_load():
            pass

        assert hasattr(dummy_load, "_machete_warp_role")
        assert dummy_load._machete_warp_role == WarpRole.LOADER

    def test_async_load_decorator(self):
        """Test @async_load decorator."""

        @async_load
        def dummy_tma_load():
            pass

        assert hasattr(dummy_tma_load, "_machete_async_load")
        assert dummy_tma_load._machete_async_load is True


# ============================================================================
# Tests for Megakernel (with CUDA)
# ============================================================================


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestMegakernelCuda:
    """Tests for Megakernel execution on CUDA devices."""

    def test_megakernel_creation(self, cuda_device):
        """Test Megakernel creation."""
        mk = Megakernel(name="test", mode="forward")
        assert mk.name == "test"
        assert mk.mode == "forward"
        assert mk.warp_config is not None
        assert mk.page_config is not None

    def test_megakernel_custom_config(self, cuda_device):
        """Test Megakernel with custom configuration."""
        warp_config = WarpConfig(num_consumer_warps=8)
        page_config = PageConfig(page_size=8192)
        mk = Megakernel(
            name="custom",
            mode="forward",
            warp_config=warp_config,
            page_config=page_config,
        )
        assert mk.warp_config.num_consumer_warps == 8
        assert mk.page_config.page_size == 8192

    def test_megakernel_empty_launch(self, cuda_device):
        """Test launching megakernel with no operations."""
        mk = Megakernel(name="empty")
        # Should not raise, just return early
        mk.launch(0, (1, 1, 1), (256, 1, 1))


# ============================================================================
# Test Kernel for CUDA Execution
# ============================================================================


class SimpleAddKernel(FusableKernel):
    """Simple add kernel for testing megakernel execution."""

    def __extract_mlir_values__(self):
        return []

    def __new_from_mlir_values__(self, values):
        return self

    def __c_pointers__(self):
        return []

    @property
    def smem_size_fwd(self) -> int:
        return 0

    @property
    def smem_size_bwd(self) -> int:
        return 0

    def get_logical_grid_size(self, *args) -> int:
        # args[0] is input tensor
        return (args[0].numel() + 256 - 1) // 256

    @cute.jit
    def load_forward(
        self,
        logical_idx,
        smem,
        t0,
        t1,
        t2,
        t3,
        t4,
        t5,
        t6,
        t7,
        t8,
        t9,
        t10,
        t11,
        t12,
        t13,
        t14,
        t15,
        t16,
        t17,
        t18,
        t19,
        t20,
        t21,
        t22,
        t23,
        t24,
        t25,
        t26,
        t27,
        t28,
        t29,
        t30,
        t31,
    ):
        pass

    @cute.jit
    def compute_forward(
        self,
        logical_idx,
        smem,
        t0,
        t1,
        t2,
        t3,
        t4,
        t5,
        t6,
        t7,
        t8,
        t9,
        t10,
        t11,
        t12,
        t13,
        t14,
        t15,
        t16,
        t17,
        t18,
        t19,
        t20,
        t21,
        t22,
        t23,
        t24,
        t25,
        t26,
        t27,
        t28,
        t29,
        t30,
        t31,
    ):
        from cutlass import Int32, const_expr

        input_ptr = t0
        bias_ptr = t1
        output_ptr = t2
        n_elements = t3

        tidx, _, _ = cute.arch.thread_idx()
        num_threads = const_expr(256)

        input_ptr_c = cute.make_ptr(cute.Float16, t0, cute.AddressSpace.gmem)
        bias_ptr_c = cute.make_ptr(cute.Float16, t1, cute.AddressSpace.gmem)
        output_ptr_c = cute.make_ptr(cute.Float16, t2, cute.AddressSpace.gmem)

        input_tensor = cute.make_tensor(input_ptr_c, cute.make_layout((t3,)))
        bias = cute.make_tensor(bias_ptr_c, cute.make_layout((t3,)))
        output_tensor = cute.make_tensor(output_ptr_c, cute.make_layout((t3,)))

        base_idx = logical_idx * num_threads + tidx
        if base_idx < n_elements:
            val = input_tensor[base_idx]
            b = bias[base_idx]
            output_tensor[base_idx] = val + b

    @cute.jit
    def store_forward(
        self,
        logical_idx,
        smem,
        t0,
        t1,
        t2,
        t3,
        t4,
        t5,
        t6,
        t7,
        t8,
        t9,
        t10,
        t11,
        t12,
        t13,
        t14,
        t15,
        t16,
        t17,
        t18,
        t19,
        t20,
        t21,
        t22,
        t23,
        t24,
        t25,
        t26,
        t27,
        t28,
        t29,
        t30,
        t31,
    ):
        pass


class SimpleScaleKernel(FusableKernel):
    """Simple scale (multiply) kernel for testing multi-op fusion."""

    def __extract_mlir_values__(self):
        return []

    def __new_from_mlir_values__(self, values):
        return self

    @property
    def smem_size_fwd(self) -> int:
        return 0

    @property
    def smem_size_bwd(self) -> int:
        return 0

    def get_logical_grid_size(self, *args) -> int:
        # args[0] is input tensor
        return (args[0].numel() + 256 - 1) // 256

    @cute.jit
    def load_forward(
        self,
        logical_idx,
        smem,
        t0, t1, t2, t3, t4, t5, t6, t7,
        t8, t9, t10, t11, t12, t13, t14, t15,
        t16, t17, t18, t19, t20, t21, t22, t23,
        t24, t25, t26, t27, t28, t29, t30, t31,
    ):
        pass

    @cute.jit
    def compute_forward(
        self,
        logical_idx,
        smem,
        t0, t1, t2, t3, t4, t5, t6, t7,
        t8, t9, t10, t11, t12, t13, t14, t15,
        t16, t17, t18, t19, t20, t21, t22, t23,
        t24, t25, t26, t27, t28, t29, t30, t31,
    ):
        from cutlass import const_expr

        # t0 = input, t1 = scale_factor, t2 = output, t3 = n_elements
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = const_expr(256)

        input_ptr = cute.make_ptr(cute.Float16, t0, cute.AddressSpace.gmem)
        scale_ptr = cute.make_ptr(cute.Float16, t1, cute.AddressSpace.gmem)
        output_ptr = cute.make_ptr(cute.Float16, t2, cute.AddressSpace.gmem)

        input_tensor = cute.make_tensor(input_ptr, cute.make_layout((t3,)))
        scale_tensor = cute.make_tensor(scale_ptr, cute.make_layout((t3,)))
        output_tensor = cute.make_tensor(output_ptr, cute.make_layout((t3,)))

        base_idx = logical_idx * num_threads + tidx
        if base_idx < t3:
            val = input_tensor[base_idx]
            s = scale_tensor[base_idx]
            output_tensor[base_idx] = val * s

    @cute.jit
    def store_forward(
        self,
        logical_idx,
        smem,
        t0, t1, t2, t3, t4, t5, t6, t7,
        t8, t9, t10, t11, t12, t13, t14, t15,
        t16, t17, t18, t19, t20, t21, t22, t23,
        t24, t25, t26, t27, t28, t29, t30, t31,
    ):
        pass


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestMegakernelExecution:
    """Tests for Megakernel kernel execution."""

    def test_single_op_execution(self, cuda_device):
        """Test executing a single operation through Megakernel."""
        # Create test data
        size = 1024
        input_t = torch.randn(size, device=cuda_device, dtype=torch.float16)
        bias = torch.randn(size, device=cuda_device, dtype=torch.float16)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float16)
        n_elements = Int32(size)

        # Create megakernel with sequential strategy
        # (warp_specialized requires atomic synchronization which has issues)
        mk = Megakernel(name="test_add", mode="forward", strategy="sequential")
        kernel = SimpleAddKernel()
        mk.add(kernel, input_t, bias, output_t, n_elements)

        # Launch
        grid_size = kernel.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (256, 1, 1))

        # Verify
        expected = input_t + bias
        torch.testing.assert_close(output_t, expected, rtol=1e-3, atol=1e-3)

    def test_single_op_warp_specialized(self, cuda_device):
        """Test single op with warp-specialized execution strategy."""
        size = 2048
        input_t = torch.randn(size, device=cuda_device, dtype=torch.float16)
        bias = torch.randn(size, device=cuda_device, dtype=torch.float16)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float16)
        n_elements = Int32(size)

        # Create megakernel with warp specialization
        mk = Megakernel(name="test_warp_spec", mode="forward", strategy="warp_specialized")
        kernel = SimpleAddKernel()
        mk.add(kernel, input_t, bias, output_t, n_elements)

        # Launch with full warp config threads
        grid_size = kernel.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (mk.warp_config.total_threads, 1, 1))

        # Verify
        expected = input_t + bias
        torch.testing.assert_close(output_t, expected, rtol=1e-3, atol=1e-3)

    def test_single_op_sequential(self, cuda_device):
        """Test single op with sequential execution strategy."""
        size = 2048
        input_t = torch.randn(size, device=cuda_device, dtype=torch.float16)
        bias = torch.randn(size, device=cuda_device, dtype=torch.float16)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float16)
        n_elements = Int32(size)

        # Create megakernel with sequential strategy
        mk = Megakernel(name="test_sequential", mode="forward", strategy="sequential")
        kernel = SimpleAddKernel()
        mk.add(kernel, input_t, bias, output_t, n_elements)

        # Launch
        grid_size = kernel.get_logical_grid_size(input_t)
        mk.launch(grid_size, (grid_size, 1, 1), (256, 1, 1))

        # Verify
        expected = input_t + bias
        torch.testing.assert_close(output_t, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestMultiOpFusion:
    """Tests for multi-op kernel fusion."""

    @pytest.mark.skip(reason="Multi-op fusion not yet implemented in sequential kernel")
    def test_two_op_fusion_sequential(self, cuda_device):
        """Test fusing two operations: add + scale (sequential strategy)."""
        size = 1024
        input_t = torch.randn(size, device=cuda_device, dtype=torch.float16)
        bias = torch.randn(size, device=cuda_device, dtype=torch.float16)
        scale = torch.randn(size, device=cuda_device, dtype=torch.float16)
        intermediate = torch.zeros(size, device=cuda_device, dtype=torch.float16)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float16)
        n_elements = Int32(size)

        # Create megakernel with two ops (sequential strategy for easier testing)
        mk = Megakernel(name="test_fusion", mode="forward", strategy="sequential")

        # Op 1: Add (input + bias -> intermediate)
        add_kernel = SimpleAddKernel()
        mk.add(add_kernel, input_t, bias, intermediate, n_elements)

        # Op 2: Scale (intermediate * scale -> output)
        scale_kernel = SimpleScaleKernel()
        mk.add(scale_kernel, intermediate, scale, output_t, n_elements)

        # Launch
        grid_size = add_kernel.get_logical_grid_size(input_t)
        total_instructions = grid_size * 2  # 2 ops
        mk.launch(total_instructions, (grid_size, 1, 1), (256, 1, 1))

        # Verify: output = (input + bias) * scale
        expected = (input_t + bias) * scale
        torch.testing.assert_close(output_t, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.skip(reason="Warp-specialized strategy requires atomic_add_i32 fix")
    def test_two_op_fusion_warp_specialized(self, cuda_device):
        """Test fusing two operations with warp specialization."""
        size = 1024
        input_t = torch.randn(size, device=cuda_device, dtype=torch.float16)
        bias = torch.randn(size, device=cuda_device, dtype=torch.float16)
        scale = torch.randn(size, device=cuda_device, dtype=torch.float16)
        intermediate = torch.zeros(size, device=cuda_device, dtype=torch.float16)
        output_t = torch.zeros(size, device=cuda_device, dtype=torch.float16)
        n_elements = Int32(size)

        # Create megakernel with warp specialization
        mk = Megakernel(name="test_ws_fusion", mode="forward", strategy="warp_specialized")

        # Op 1: Add
        add_kernel = SimpleAddKernel()
        mk.add(add_kernel, input_t, bias, intermediate, n_elements)

        # Op 2: Scale
        scale_kernel = SimpleScaleKernel()
        mk.add(scale_kernel, intermediate, scale, output_t, n_elements)

        # Launch
        grid_size = add_kernel.get_logical_grid_size(input_t)
        total_instructions = grid_size * 2
        mk.launch(total_instructions, (grid_size, 1, 1), (mk.warp_config.total_threads, 1, 1))

        # Verify: output = (input + bias) * scale
        expected = (input_t + bias) * scale
        torch.testing.assert_close(output_t, expected, rtol=1e-2, atol=1e-2)

    def test_scheduler_with_multi_op(self, cuda_device):
        """Test that scheduler correctly generates instructions for multi-op."""
        barrier_config = BarrierConfig(num_ops=2, num_chunks=4)
        page_config = PageConfig(page_size=16384)
        scheduler = InstructionScheduler(barrier_config, page_config, num_pages=8)

        # Add two ops with dependency
        scheduler.add_op(
            op_id=0,
            op=None,
            logical_grid_size=4,
            reads_list=["input"],
            writes_list=["intermediate"],
        )
        scheduler.add_op(
            op_id=1,
            op=None,
            logical_grid_size=4,
            reads_list=["intermediate"],
            writes_list=["output"],
        )

        instructions = scheduler.schedule()

        # Should have 8 instructions total (4 chunks * 2 ops)
        assert len(instructions) == 8

        # Verify dependency structure
        analysis = scheduler.analyze_dependencies()
        assert analysis["total_instructions"] == 8
        assert analysis["local_dependencies"] == 4  # Op 1 depends on Op 0 for each chunk
        assert analysis["cross_block_dependencies"] == 0  # All local


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestWarpSpecialization:
    """Tests for warp specialization infrastructure."""

    def test_warp_role_assignment(self):
        """Test warp roles are correctly assigned."""
        config = WarpConfig(num_consumer_warps=12)

        # Consumer warps: 0-11
        for i in range(12):
            assert config.get_role(i) == WarpRole.CONSUMER

        # Special warps
        assert config.get_role(12) == WarpRole.LOADER
        assert config.get_role(13) == WarpRole.STORER
        assert config.get_role(14) == WarpRole.LAUNCHER
        assert config.get_role(15) == WarpRole.CONTROLLER

    def test_total_warp_threads(self):
        """Test thread count matches warp config."""
        config = WarpConfig(num_consumer_warps=12)
        assert config.total_warps == 16
        assert config.total_threads == 512  # 16 * 32

    def test_custom_warp_config(self):
        """Test custom warp configuration."""
        config = WarpConfig(
            num_consumer_warps=8,
            num_loader_warps=2,
            num_storer_warps=1,
            num_launcher_warps=1,
            num_controller_warps=1,
        )
        assert config.total_warps == 13
        assert config.loader_warp_start == 8
        assert config.storer_warp_start == 10


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestGlobalScoreboard:
    """Tests for global scoreboard (barrier) synchronization."""

    def test_barrier_config_indexing(self):
        """Test barrier array indexing."""
        config = BarrierConfig(num_ops=3, num_chunks=10)

        # Bar[op_id][chunk_id]
        assert config.get_index(0, 0) == 0
        assert config.get_index(0, 9) == 9
        assert config.get_index(1, 0) == 10
        assert config.get_index(2, 5) == 25

    def test_barrier_total_size(self):
        """Test barrier array total size."""
        config = BarrierConfig(num_ops=4, num_chunks=8)
        assert config.total_size == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
