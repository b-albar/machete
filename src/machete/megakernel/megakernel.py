# Copyright (c) 2025, Machete Authors
"""
Persistent Megakernel with Instruction Stream, Paged Memory, and Op Dispatch.

This module implements a "No Bubbles" megakernel architecture:

1. All blocks are persistent (launched once, process entire computation)
2. Each block fetches instructions from a global instruction stream
3. Fine-grained barriers at TILE level enable pipeline overlap between ops
4. Shared memory is managed via a paged memory system (circular buffer)
5. Ops are dispatched at compile time (inlined branches per op index)
6. Barrier wait/signal logic is baked into op handlers at compile time

Usage:
    from machete.megakernel import Megakernel, ScheduledOp, NOPOp

    ops = [
        ScheduledOp(NOPOp, tiles_m=32),
        ScheduledOp(NOPOp, tiles_m=32),
    ]

    kernel = Megakernel(ops)
    kernel.run()
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from .ops import (
    ScheduledOp,
    InstructionStreamBuilder,
    TileInstruction,
)
from .compile import (
    compile_load_async,
    compile_compute,
    compile_store,
    compile_backward_load_async,
    compile_backward_compute,
    compile_backward_store,
)
from .interpreter import (
    global_barrier_wait,
    global_barrier_signal,
    check_barrier_ready,
    load_instruction_to_smem,
    ld_global_i64,
    cp_async_wait_group,
)
from .paged_memory import (
    PAGE_SIZE,
    NPageLayout,
    st_shared_i32,
    ld_shared_i32,
)
from machete.kernels.utils.async_copy import (
    cp_async_commit,
    cp_async_wait_all,
)


# =============================================================================
# Shared Memory Pointer Access
# =============================================================================


@dsl_user_op
def get_smem_base_ptr(*, loc=None, ip=None) -> Int32:
    """Get the base pointer to shared memory using PTX.

    Returns:
        32-bit unsigned integer address of shared memory base.
    """
    result = llvm.inline_asm(
        T.i32(),
        [],
        """
        {
            .reg .u64 smem_ptr64;
            cvta.shared.u64 smem_ptr64, 0;
            cvt.u32.u64 $0, smem_ptr64;
        }
        """,
        "=r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


# =============================================================================
# Megakernel Configuration
# =============================================================================


@dataclass
class MegakernelConfig:
    """Configuration for the persistent megakernel.

    Attributes:
        threads_per_block: Threads per block (default: 256)
        num_sms: Number of SMs to use (None = auto-detect)
        page_size: Size of each page in bytes (default: 16KB)
        num_pages: Number of pages for pipelining (None = auto-detect max).
            With N pages, up to N-1 loads can overlap with compute.
            Minimum is 2 (double-buffering).
        tracing: Enable cutedsl-trace instrumentation (default: False).
            When False, all trace calls are eliminated at compile time
            via constexpr (zero overhead).
    """

    threads_per_block: int = 256
    num_sms: Optional[int] = None
    page_size: int = PAGE_SIZE
    num_pages: Optional[int] = None  # None = auto-detect max for GPU
    tracing: bool = False

    @property
    def warps_per_block(self) -> int:
        """Number of warps per block."""
        return self.threads_per_block // 32


# =============================================================================
# Megakernel Implementation
# =============================================================================


class Megakernel:
    """Persistent megakernel with instruction stream, paged memory, and op dispatch.

    Caching:
        Compiled kernels are cached at the class level to avoid recompilation
        when creating multiple Megakernel instances with the same configuration.
        The cache key is based on: (op_classes, static_dims, config_params, backward).

    Architecture:
    - All SMs launched as persistent blocks
    - Each block fetches instructions from global memory in a strided pattern
    - Fine-grained barriers enable tile-level dependencies between operations
    - Barrier wait/signal logic is baked into op handlers at compile time
    - Shared memory is divided into pages managed by a circular buffer
    - Ops are dispatched via compile-time inlined branches

    Work Distribution:
        Block 0: instr 0 -> instr num_sms -> instr 2*num_sms -> ...
        Block 1: instr 1 -> instr num_sms+1 -> instr 2*num_sms+1 -> ...

    When backward=True, the kernel dispatches to each op's backward() method
    instead of forward(). This enables using the same op definitions and
    scheduling infrastructure for gradient computation.

    Example:
        ops = [
            ScheduledOp(RMSNormOp, tiles_m=32),
            ScheduledOp(MatVecOp, tiles_m=32),  # Can overlap with RMSNorm!
        ]
        kernel = Megakernel(ops)
        kernel.run()

        # Backward pass with same ops
        bwd_kernel = Megakernel(ops, backward=True)
        bwd_kernel.run()
    """

    # Class-level cache for compiled kernels to avoid recompilation
    # Key: (op_classes_tuple, static_dims_tuple, config_key, backward)
    _compiled_kernel_cache: Dict[Tuple, Any] = {}

    def __init__(
        self,
        ops: List[ScheduledOp],
        config: Optional[MegakernelConfig] = None,
        device: str = "cuda",
        backward: bool = False,
    ):
        self.ops = ops
        self.config = config or MegakernelConfig()
        self.device = device
        self.backward = backward

        # Detect SM count if not specified
        if self.config.num_sms is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Megakernel requires a CUDA GPU.")
            props = torch.cuda.get_device_properties(device)
            self.config.num_sms = props.multi_processor_count

        # Create N-page layout (auto-detect max pages or use user-specified)
        if self.config.num_pages is not None:
            # User specified number of pages
            self._layout = NPageLayout(
                num_pages=self.config.num_pages,
                page_size=self.config.page_size,
            )
        else:
            # Auto-detect maximum pages that fit in shared memory
            self._layout = NPageLayout.for_device(
                page_size=self.config.page_size,
                min_pages=2,
            )
            # Store computed num_pages back to config for cache key
            self.config.num_pages = self._layout.num_pages

        # Build instruction stream
        # Pass ScheduledOp directly to preserve tensor_ptrs for automatic dependency detection
        self._builder = InstructionStreamBuilder()
        for op in ops:
            self._builder.add_op(op)

        self._instructions_tensor: Optional[torch.Tensor] = None
        self._barriers_tensor: Optional[torch.Tensor] = None
        self._op_configs_tensor: Optional[torch.Tensor] = None
        self._num_instructions: Optional[int] = None
        self._compiled_kernel = None

        # Trace setup
        self._trace_builder = None
        self._trace_types = {}
        self._trace_block_type = None
        self._trace_format_ids = []  # format_id per op index
        if self.config.tracing:
            self._setup_tracing()

    @property
    def num_sms(self) -> int:
        """Number of SMs used."""
        return self.config.num_sms

    @property
    def total_tiles(self) -> int:
        """Total number of work tiles."""
        return self._builder.total_tiles

    @property
    def num_barriers(self) -> int:
        """Number of barriers needed."""
        return self._builder.num_barriers

    @property
    def smem_size(self) -> int:
        """Total shared memory size in bytes."""
        return self._layout.total_size

    @property
    def grid(self) -> Tuple[int, int, int]:
        """Grid dimensions for kernel launch."""
        return (self.config.num_sms, 1, 1)

    @property
    def block(self) -> Tuple[int, int, int]:
        """Block dimensions for kernel launch."""
        return (self.config.threads_per_block, 1, 1)

    def _prepare_tensors(self) -> None:
        """Prepare instruction, barrier, and config tensors on GPU."""
        if self._instructions_tensor is None:
            self._instructions_tensor = self._builder.build_tensor(self.device)
            self._num_instructions = self._instructions_tensor.shape[0]
            self._num_instructions_i32 = Int32(self._num_instructions)

        if self._barriers_tensor is None:
            self._barriers_tensor = torch.zeros(self.num_barriers, dtype=torch.int32, device=self.device)

        if self._op_configs_tensor is None:
            # One Int64 pointer per op. Ops with no config get pointer 0.
            config_ptrs = []
            for op in self.ops:
                if op.config_data is not None:
                    config_ptrs.append(op.config_data.data_ptr())
                else:
                    config_ptrs.append(0)
            self._op_configs_tensor = torch.tensor(config_ptrs, dtype=torch.int64, device=self.device)

    def _validate_requirements(self) -> None:
        """Validate GPU requirements."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        major, _ = props.major, props.minor

        if major < 9:
            raise RuntimeError(
                f"Megakernel requires Hopper (SM90+) GPU. Current GPU has compute capability sm_{major}x."
            )

        max_smem = props.shared_memory_per_block_optin
        required_smem = self._layout.total_size
        if required_smem > max_smem:
            raise RuntimeError(
                f"Megakernel requires {required_smem // 1024}KB shared memory "
                f"({self._layout.num_pages} pages × {self.config.page_size // 1024}KB + scratch), "
                f"but the GPU only supports {max_smem // 1024}KB per block. "
                f"Reduce num_pages or page_size in MegakernelConfig."
            )

    def _setup_tracing(self):
        """Set up cutedsl-trace builder and trace types."""
        import math
        from cutedsl_trace import (
            TraceType,
            BlockType,
            TrackType,
            DynamicTraceBuilder,
        )
        from cutedsl_trace.types import LaneType

        # One TraceType per unique op class
        seen_classes = {}
        for i, op in enumerate(self.ops):
            cls_name = op.op_cls.__name__
            if cls_name not in seen_classes:
                tt = TraceType(
                    name=cls_name,
                    label_string=cls_name,
                    tooltip_string=f"{cls_name} op_idx={{0}}",
                    param_count=1,
                    lane_type=LaneType.DYNAMIC,
                )
                seen_classes[cls_name] = tt
                self._trace_types[cls_name] = tt
            self._trace_format_ids.append(seen_classes[cls_name].id)

        self._trace_block_type = BlockType(
            name="CTA",
            label_string="CTA {blockLinear}",
            tooltip_string="CTA {blockLinear} on SM {smId}",
        )

        track_type = TrackType(
            name="SM",
            label_string="SM {lane}",
            tooltip_string="SM lane {lane}",
        )

        # 1 lane per CTA, enough events for the instruction stream
        max_events = math.ceil(self.total_tiles / self.num_sms) * 2 + 16
        self._trace_builder = DynamicTraceBuilder(
            num_lanes=1,
            max_events_per_lane=max_events,
            grid_dims=(self.num_sms, 1, 1),
        )
        self._trace_builder.set_track_type(track_type, lane=0)

    def write_trace(self, filename: str) -> None:
        """Write trace to .nanotrace file. Only valid after run() with tracing=True."""
        if self._trace_builder is None:
            raise RuntimeError("Tracing not enabled. Set MegakernelConfig(tracing=True).")
        from cutedsl_trace import TraceWriter

        self._trace_builder.copy_to_host()
        writer = TraceWriter("megakernel")
        writer.set_block_type(self._trace_block_type)
        writer.add_tensor(self._trace_builder)
        for tt in self._trace_types.values():
            writer.register_trace_type(tt)
        writer.write(filename)

    def _build_pipelined_dispatch_fns(self):
        """Build dispatch functions for pipelined execution phases.

        All ops use the pipelined interface (load_async/compute/store).
        Phases that are just `pass` compile to no-ops automatically.

        Returns:
            (dispatch_load_async, dispatch_compute, dispatch_store)
        """
        ops = self.ops
        use_backward = self.backward

        # Compile each phase for each op
        load_fns = []
        compute_fns = []
        store_fns = []

        for i, op in enumerate(ops):
            # Generate init source for this op
            init_source = None
            if hasattr(op.op_cls, "gen_init_source") and (op.static_dims or op.tensor_dtypes):
                kernel_config = {"threads_per_row": self.config.threads_per_block}
                init_source = op.op_cls.gen_init_source(
                    op.static_dims,
                    backward=use_backward,
                    kernel_config=kernel_config,
                    tensor_dtypes=op.tensor_dtypes,
                )

            # Compile all three phases - pass-only methods become no-ops
            # Use backward phase methods when in backward mode
            if use_backward:
                load_fns.append(compile_backward_load_async(op.op_cls, init_source))
                compute_fns.append(compile_backward_compute(op.op_cls, init_source))
                store_fns.append(compile_backward_store(op.op_cls, init_source))
            else:
                load_fns.append(compile_load_async(op.op_cls, init_source))
                compute_fns.append(compile_compute(op.op_cls, init_source))
                store_fns.append(compile_store(op.op_cls, init_source))

        # Build dispatch_load_async
        @cute.jit
        def dispatch_load_async(
            op_idx: Int32,
            page_ptr: Int32,
            tile_m: Int32,
            tile_n: Int32,
            tile_l: Int32,
            op_config_ptr: Int64,
        ) -> None:
            for i, fn in enumerate(load_fns):
                if op_idx == Int32(i):
                    fn(page_ptr, tile_m, tile_n, tile_l, op_config_ptr)

        # Build dispatch_compute
        @cute.jit
        def dispatch_compute(
            op_idx: Int32,
            page_ptr: Int32,
            tile_m: Int32,
            tile_n: Int32,
            tile_l: Int32,
            op_config_ptr: Int64,
        ) -> None:
            for i, fn in enumerate(compute_fns):
                if op_idx == Int32(i):
                    fn(page_ptr, tile_m, tile_n, tile_l, op_config_ptr)

        # Build dispatch_store
        @cute.jit
        def dispatch_store(
            op_idx: Int32,
            page_ptr: Int32,
            tile_m: Int32,
            tile_n: Int32,
            tile_l: Int32,
            op_config_ptr: Int64,
        ) -> None:
            for i, fn in enumerate(store_fns):
                if op_idx == Int32(i):
                    fn(page_ptr, tile_m, tile_n, tile_l, op_config_ptr)

        return dispatch_load_async, dispatch_compute, dispatch_store

    def _build_dependency_checker(self):
        """Build function to check if tile's dependencies are satisfied (non-blocking).

        Returns a @cute.jit function that returns 1 if all wait barriers for
        the given tile are satisfied, 0 otherwise. Uses relaxed memory ordering
        for reads (no blocking).
        """
        barrier_formulas = self._builder.get_op_barrier_formulas()

        @cute.jit
        def check_dependencies_ready(
            op_idx: Int32,
            tile_m: Int32,
            tile_n: Int32,
            tile_l: Int32,
            barriers_ptr: Int64,
        ) -> Int32:
            """Returns 1 if all wait barriers are satisfied, 0 otherwise."""
            result = Int32(1)

            for i, (wait_formulas, _) in barrier_formulas.items():
                if op_idx == Int32(i):
                    for wf in wait_formulas:
                        # Compute barrier index (with divisors for tile size ratios)
                        barrier_idx = (
                            Int32(wf.base)
                            + (Int32(wf.coeff_m) * tile_m) // Int32(wf.div_m)
                            + (Int32(wf.coeff_n) * tile_n) // Int32(wf.div_n)
                            + (Int32(wf.coeff_l) * tile_l) // Int32(wf.div_l)
                        )

                        # Compute linear index for guard check
                        _linear = (
                            Int32(wf.coeff_m) * tile_m
                            + Int32(wf.coeff_n) * tile_n
                            + Int32(wf.coeff_l) * tile_l
                        )

                        if _linear < Int32(wf.guard_max):
                            # Non-blocking check
                            ready = check_barrier_ready(
                                barriers_ptr, barrier_idx, Int32(wf.expected)
                            )
                            if ready == Int32(0):
                                result = Int32(0)

            return result

        return check_dependencies_ready

    def _build_wait_barriers(self):
        """Build function to wait on barriers (blocking).

        Returns a @cute.jit function that blocks until all wait barriers
        for the given tile are satisfied.
        """
        barrier_formulas = self._builder.get_op_barrier_formulas()

        @cute.jit
        def wait_barriers(
            op_idx: Int32,
            tile_m: Int32,
            tile_n: Int32,
            tile_l: Int32,
            barriers_ptr: Int64,
        ) -> None:
            """Block until all wait barriers are satisfied."""
            for i, (wait_formulas, _) in barrier_formulas.items():
                if op_idx == Int32(i):
                    for wf in wait_formulas:
                        # Compute linear index for guard check
                        _linear = (
                            Int32(wf.coeff_m) * tile_m
                            + Int32(wf.coeff_n) * tile_n
                            + Int32(wf.coeff_l) * tile_l
                        )
                        if _linear < Int32(wf.guard_max):
                            # Compute barrier index
                            barrier_idx = (
                                Int32(wf.base)
                                + (Int32(wf.coeff_m) * tile_m) // Int32(wf.div_m)
                                + (Int32(wf.coeff_n) * tile_n) // Int32(wf.div_n)
                                + (Int32(wf.coeff_l) * tile_l) // Int32(wf.div_l)
                            )
                            global_barrier_wait(barriers_ptr, barrier_idx, Int32(wf.expected))

        return wait_barriers

    def _build_signal_barriers(self):
        """Build function to signal barriers after tile completion.

        Returns a @cute.jit function that signals all barriers for the
        given tile after it completes.
        """
        barrier_formulas = self._builder.get_op_barrier_formulas()

        @cute.jit
        def signal_barriers(
            op_idx: Int32,
            tile_m: Int32,
            tile_n: Int32,
            tile_l: Int32,
            barriers_ptr: Int64,
        ) -> None:
            """Signal all barriers for the completed tile."""
            for i, (_, signal_formulas) in barrier_formulas.items():
                if op_idx == Int32(i):
                    for sf in signal_formulas:
                        # Compute barrier index
                        barrier_idx = (
                            Int32(sf.base)
                            + (Int32(sf.coeff_m) * tile_m) // Int32(sf.div_m)
                            + (Int32(sf.coeff_n) * tile_n) // Int32(sf.div_n)
                            + (Int32(sf.coeff_l) * tile_l) // Int32(sf.div_l)
                        )
                        global_barrier_signal(barriers_ptr, barrier_idx)

        return signal_barriers

    def _create_kernel(self):
        """Create the persistent pipelined kernel with N-page buffered execution."""
        num_sms = self.config.num_sms
        threads_per_block = self.config.threads_per_block
        layout = self._layout
        smem_size = layout.total_size
        tracing = self.config.tracing
        num_pages = layout.num_pages

        # Capture layout offsets as compile-time constants
        scratch_offset = layout.instr_offset
        flags_offset = layout.flags_offset
        ring_state_offset = layout.ring_state_offset
        pages_start = layout.pages_start
        aligned_page_size = layout.aligned_page_size

        # Build pipelined dispatch functions
        dispatch_load_async, dispatch_compute, dispatch_store = self._build_pipelined_dispatch_fns()

        # Build barrier functions
        check_deps = self._build_dependency_checker()
        wait_barriers = self._build_wait_barriers()
        signal_barriers = self._build_signal_barriers()

        # Helper to get page pointer by index - uses arithmetic (no branching needed)
        # page_ptr = smem_base + pages_start + page_idx * aligned_page_size
        @cute.jit
        def _get_page_ptr(smem_base: Int32, page_idx: Int32) -> Int32:
            """Get page pointer for given index using direct arithmetic."""
            return smem_base + Int32(pages_start) + page_idx * Int32(aligned_page_size)

        # N-page pipelined kernel loop
        @cute.jit
        def _kernel_loop_n_pipelined(
            instructions_ptr: Int64,
            barriers_ptr: Int64,
            op_configs_ptr: Int64,
            num_instructions: Int32,
            tidx: Int32,
            block_id: Int32,
            num_blocks: Int32,
            smem_base: Int32,
        ) -> None:
            """N-page pipelined persistent loop with ring buffer of pages.

            With N pages, we can have up to N-1 loads in flight overlapping
            with compute. Uses a ring buffer: compute_ptr points to the oldest
            loaded page, acquire_ptr points to the next page to load into.
            """
            # Ring buffer state
            acquire_ptr = Int32(0)   # Next page index to acquire for loading
            compute_ptr = Int32(0)   # Next page index to compute (oldest ready)
            tiles_in_flight = Int32(0)  # Number of tiles loaded but not yet computed

            # Current instruction index for this block
            next_instr_idx = block_id

            # Scratch pointer for instruction decode
            scratch_ptr = smem_base + Int32(scratch_offset)
            flags_ptr = smem_base + Int32(flags_offset)

            # ========== PROLOGUE: Fill pipeline with up to N-1 tiles ==========
            max_in_flight = Int32(num_pages - 1)
            done_filling = Int32(0)  # Flag to track if we should stop filling

            # Use simple counter loop - check conditions inside with nested ifs
            for _fill_iter in range(num_pages - 1):
                # Check all conditions using nested ifs (avoid & operator on Booleans)
                if tiles_in_flight < max_in_flight:
                    if next_instr_idx < num_instructions:
                        if done_filling == Int32(0):
                            # Fetch instruction
                            if tidx == Int32(0):
                                load_instruction_to_smem(instructions_ptr, next_instr_idx, scratch_ptr)
                            cute.arch.barrier()

                            op_idx = ld_shared_i32(scratch_ptr)
                            tile_m = ld_shared_i32(scratch_ptr + 4)
                            tile_n = ld_shared_i32(scratch_ptr + 8)
                            tile_l = ld_shared_i32(scratch_ptr + 12)

                            # Check for end marker - if found, stop filling
                            if op_idx == Int32(TileInstruction.END_MARKER):
                                done_filling = Int32(1)

                            if op_idx != Int32(TileInstruction.END_MARKER):
                                # Check if dependencies are ready (non-blocking)
                                if tidx == Int32(0):
                                    ready = check_deps(op_idx, tile_m, tile_n, tile_l, barriers_ptr)
                                    st_shared_i32(flags_ptr, ready)
                                cute.arch.barrier()
                                deps_ready = ld_shared_i32(flags_ptr)

                                # If deps not ready and nothing in flight, wait blocking
                                if deps_ready == Int32(0):
                                    if tiles_in_flight == Int32(0):
                                        if tidx == Int32(0):
                                            wait_barriers(op_idx, tile_m, tile_n, tile_l, barriers_ptr)
                                        cute.arch.barrier()
                                    else:
                                        # Can't speculate further - stop filling
                                        done_filling = Int32(1)

                                # Only load if we haven't decided to stop
                                if done_filling == Int32(0):
                                    # Get page pointer and issue load
                                    page_ptr = _get_page_ptr(smem_base, acquire_ptr)
                                    op_config_ptr = ld_global_i64(op_configs_ptr, op_idx)
                                    dispatch_load_async(op_idx, page_ptr, tile_m, tile_n, tile_l, op_config_ptr)
                                    cp_async_commit()

                                    # Store tile info
                                    tile_info_ptr = smem_base + Int32(ring_state_offset) + acquire_ptr * Int32(16)
                                    if tidx == Int32(0):
                                        st_shared_i32(tile_info_ptr, op_idx)
                                        st_shared_i32(tile_info_ptr + 4, tile_m)
                                        st_shared_i32(tile_info_ptr + 8, tile_n)
                                        st_shared_i32(tile_info_ptr + 12, tile_l)

                                    # Advance ring buffer
                                    acquire_ptr = (acquire_ptr + Int32(1)) % Int32(num_pages)
                                    tiles_in_flight = tiles_in_flight + Int32(1)
                                    next_instr_idx = next_instr_idx + num_blocks

            # ========== MAIN LOOP: Process tiles and try to load more ==========
            while tiles_in_flight > Int32(0):
                # Wait for the oldest load to complete (keep others in flight)
                cp_async_wait_group(tiles_in_flight - Int32(1))
                cute.arch.barrier()

                # Get compute page pointer and tile info
                compute_page_ptr = _get_page_ptr(smem_base, compute_ptr)
                tile_info_ptr = smem_base + Int32(ring_state_offset) + compute_ptr * Int32(16)
                curr_op_idx = ld_shared_i32(tile_info_ptr)
                curr_tile_m = ld_shared_i32(tile_info_ptr + 4)
                curr_tile_n = ld_shared_i32(tile_info_ptr + 8)
                curr_tile_l = ld_shared_i32(tile_info_ptr + 12)
                curr_op_config_ptr = ld_global_i64(op_configs_ptr, curr_op_idx)

                # ===== TRY TO LOAD MORE TILES (speculative) =====
                can_try_load = tiles_in_flight < max_in_flight
                can_try_load = can_try_load & (next_instr_idx < num_instructions)

                if can_try_load:
                    # Fetch next instruction
                    if tidx == Int32(0):
                        load_instruction_to_smem(instructions_ptr, next_instr_idx, scratch_ptr)
                    cute.arch.barrier()

                    next_op_idx = ld_shared_i32(scratch_ptr)
                    if next_op_idx != Int32(TileInstruction.END_MARKER):
                        next_tile_m = ld_shared_i32(scratch_ptr + 4)
                        next_tile_n = ld_shared_i32(scratch_ptr + 8)
                        next_tile_l = ld_shared_i32(scratch_ptr + 12)

                        # Check dependencies (non-blocking)
                        if tidx == Int32(0):
                            ready = check_deps(next_op_idx, next_tile_m, next_tile_n, next_tile_l, barriers_ptr)
                            st_shared_i32(flags_ptr, ready)
                        cute.arch.barrier()

                        if ld_shared_i32(flags_ptr) == Int32(1):
                            # Dependencies ready - issue load
                            next_page_ptr = _get_page_ptr(smem_base, acquire_ptr)
                            next_op_config_ptr = ld_global_i64(op_configs_ptr, next_op_idx)
                            dispatch_load_async(
                                next_op_idx, next_page_ptr, next_tile_m, next_tile_n, next_tile_l, next_op_config_ptr
                            )
                            cp_async_commit()

                            # Store tile info
                            next_tile_info_ptr = smem_base + Int32(ring_state_offset) + acquire_ptr * Int32(16)
                            if tidx == Int32(0):
                                st_shared_i32(next_tile_info_ptr, next_op_idx)
                                st_shared_i32(next_tile_info_ptr + 4, next_tile_m)
                                st_shared_i32(next_tile_info_ptr + 8, next_tile_n)
                                st_shared_i32(next_tile_info_ptr + 12, next_tile_l)

                            acquire_ptr = (acquire_ptr + Int32(1)) % Int32(num_pages)
                            tiles_in_flight = tiles_in_flight + Int32(1)
                            next_instr_idx = next_instr_idx + num_blocks

                # ===== COMPUTE CURRENT TILE =====
                dispatch_compute(curr_op_idx, compute_page_ptr, curr_tile_m, curr_tile_n, curr_tile_l, curr_op_config_ptr)
                cute.arch.barrier()

                # ===== STORE CURRENT TILE =====
                dispatch_store(curr_op_idx, compute_page_ptr, curr_tile_m, curr_tile_n, curr_tile_l, curr_op_config_ptr)
                cute.arch.barrier()

                # ===== SIGNAL COMPLETION =====
                if tidx == Int32(0):
                    signal_barriers(curr_op_idx, curr_tile_m, curr_tile_n, curr_tile_l, barriers_ptr)

                # Release page and advance compute pointer
                compute_ptr = (compute_ptr + Int32(1)) % Int32(num_pages)
                tiles_in_flight = tiles_in_flight - Int32(1)

            # ========== EPILOGUE: Handle remaining tiles sequentially ==========
            # The epilogue only runs if the main loop couldn't fill enough tiles.
            # This happens when dependencies blocked speculation. Process remaining
            # tiles one at a time with blocking waits.
            done_epilogue = Int32(0)

            # Use the main loop pattern - simple condition that CuTe DSL can handle
            while tiles_in_flight >= Int32(0):  # Always true, exit via done_epilogue
                # Check if we should continue - use nested ifs to avoid & operator
                if next_instr_idx < num_instructions:
                    if done_epilogue == Int32(0):
                        # Fetch instruction
                        if tidx == Int32(0):
                            load_instruction_to_smem(instructions_ptr, next_instr_idx, scratch_ptr)
                        cute.arch.barrier()

                        op_idx = ld_shared_i32(scratch_ptr)

                        # Check for end marker
                        if op_idx == Int32(TileInstruction.END_MARKER):
                            done_epilogue = Int32(1)

                        if op_idx != Int32(TileInstruction.END_MARKER):
                            tile_m = ld_shared_i32(scratch_ptr + 4)
                            tile_n = ld_shared_i32(scratch_ptr + 8)
                            tile_l = ld_shared_i32(scratch_ptr + 12)

                            # Wait for dependencies (blocking)
                            if tidx == Int32(0):
                                wait_barriers(op_idx, tile_m, tile_n, tile_l, barriers_ptr)
                            cute.arch.barrier()

                            # Use page 0 for sequential fallback
                            page_ptr = _get_page_ptr(smem_base, Int32(0))
                            op_config_ptr = ld_global_i64(op_configs_ptr, op_idx)

                            # Load synchronously
                            dispatch_load_async(op_idx, page_ptr, tile_m, tile_n, tile_l, op_config_ptr)
                            cp_async_commit()
                            cp_async_wait_all()
                            cute.arch.barrier()

                            # Compute
                            dispatch_compute(op_idx, page_ptr, tile_m, tile_n, tile_l, op_config_ptr)
                            cute.arch.barrier()

                            # Store
                            dispatch_store(op_idx, page_ptr, tile_m, tile_n, tile_l, op_config_ptr)
                            cute.arch.barrier()

                            # Signal
                            if tidx == Int32(0):
                                signal_barriers(op_idx, tile_m, tile_n, tile_l, barriers_ptr)

                            next_instr_idx = next_instr_idx + num_blocks
                    else:
                        # done_epilogue is set, exit
                        tiles_in_flight = Int32(-1)
                else:
                    # No more instructions, exit the loop
                    tiles_in_flight = Int32(-1)

        if tracing:
            raise NotImplementedError("Tracing not yet supported with pipelined kernel")

        class PersistentKernel:
            def __init__(self):
                self.num_sms = num_sms
                self.threads_per_block = threads_per_block
                self.smem_size = smem_size

            @cute.jit
            def __call__(
                self,
                instructions_ptr: Int64,
                barriers_ptr: Int64,
                op_configs_ptr: Int64,
                trace_buffer_ptr: Int64,
                num_instructions: Int32,
                stream,
            ):
                self.kernel(
                    instructions_ptr,
                    barriers_ptr,
                    op_configs_ptr,
                    num_instructions,
                ).launch(
                    grid=[self.num_sms, 1, 1],
                    block=[self.threads_per_block, 1, 1],
                    smem=self.smem_size,
                    stream=stream,
                )

            @cute.kernel
            def kernel(
                self,
                instructions_ptr: Int64,
                barriers_ptr: Int64,
                op_configs_ptr: Int64,
                num_instructions: Int32,
            ):
                tidx = cute.arch.thread_idx()[0]
                block_id = cute.arch.block_idx()[0]
                num_blocks = cute.arch.grid_dim()[0]
                smem_base = get_smem_base_ptr()

                _kernel_loop_n_pipelined(
                    instructions_ptr,
                    barriers_ptr,
                    op_configs_ptr,
                    num_instructions,
                    tidx,
                    block_id,
                    num_blocks,
                    smem_base,
                )

        return PersistentKernel()

    def _make_cache_key(self) -> Tuple:
        """Create a cache key for the compiled kernel.

        The key includes all parameters that affect kernel compilation:
        - Op classes, their static dimensions, tensor dtypes, and tile counts
        - Config parameters (threads, pages, etc.)
        - Backward flag

        Tile counts (tiles_m, tiles_n, tiles_l) are included because barrier
        formulas are baked into the kernel at compile time. Different tile
        counts produce different barrier formulas and instruction streams.
        """
        op_keys = []
        for op in self.ops:
            static_dims_tuple = tuple(sorted(op.static_dims.items())) if op.static_dims else ()
            # Include tensor dtypes - different dtypes require different compiled code
            # Convert dtypes to their names for hashing (CUTLASS dtype objects aren't hashable directly)
            tensor_dtypes_tuple = (
                tuple(sorted((k, v.__name__) for k, v in op.tensor_dtypes.items()))
                if op.tensor_dtypes
                else ()
            )
            # Include tile counts - barrier formulas are baked into the kernel
            tile_counts = (op.tiles_m, op.tiles_n, op.tiles_l)
            op_keys.append((op.op_cls, static_dims_tuple, tensor_dtypes_tuple, tile_counts))

        config_key = (
            self.config.threads_per_block,
            self.config.page_size,
            self.config.num_pages,
            self.config.tracing,
        )

        return (tuple(op_keys), config_key, self.backward)

    def compile(self) -> None:
        """Compile the kernel without running it.

        Triggers JIT compilation so that subsequent run() calls have no
        compilation overhead. Safe to call multiple times (no-op after first).

        Uses a class-level cache to avoid recompilation when multiple Megakernel
        instances have the same configuration (same ops, static_dims, config).
        """
        # _prepare_tensors is idempotent (checks for None internally)
        self._prepare_tensors()

        if self._compiled_kernel is None:
            # Check class-level cache first
            cache_key = self._make_cache_key()
            if cache_key in Megakernel._compiled_kernel_cache:
                self._compiled_kernel = Megakernel._compiled_kernel_cache[cache_key]
                return

            self._validate_requirements()
            from cutedsl_trace.config import set_tracing_enabled

            set_tracing_enabled(self.config.tracing)

            mode = "backward" if self.backward else "forward"
            tracing_str = " [traced]" if self.config.tracing else ""
            print(
                f"Compiling persistent kernel ({mode}{tracing_str}) for "
                f"{len(self.ops)} ops, "
                f"{self.total_tiles} tiles, {self.num_sms} SMs, "
                f"{self.smem_size // 1024}KB smem..."
            )
            self._compiled_kernel = self._create_kernel()

            # Force upfront JIT compilation with cute.compile()
            # This avoids lazy compilation on first run()
            import cuda.bindings.driver as cuda
            torch_stream = torch.cuda.current_stream()
            cu_stream = cuda.CUstream(torch_stream.cuda_stream)
            self._compiled_kernel = cute.compile(
                self._compiled_kernel,
                Int64(self._instructions_tensor.data_ptr()),
                Int64(self._barriers_tensor.data_ptr()),
                Int64(self._op_configs_tensor.data_ptr()),
                Int64(0),  # trace_buffer_ptr
                self._num_instructions_i32,
                cu_stream,
            )

            # Store in class-level cache
            Megakernel._compiled_kernel_cache[cache_key] = self._compiled_kernel
            print("Compilation complete.")

    def run(self, stream=None, sync: bool = True) -> None:
        """Run the persistent megakernel.

        Args:
            stream: CUDA stream (optional). If None, uses current stream.
            sync: If True (default), synchronize after launch. Set to False
                for benchmarking or when managing synchronization externally.
        """
        # Compile on first call (no-op if already compiled).
        # Always call compile() to ensure tensors are prepared, even when
        # _compiled_kernel was injected externally (e.g., autograd cache).
        self.compile()

        if stream is None:
            import cuda.bindings.driver as cuda

            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

        # Reset barriers for this run
        self._barriers_tensor.zero_()

        # Launch (trace_buffer_ptr is always passed; trace calls compile
        # to nothing when tracing is disabled via constexpr elimination)
        if self.config.tracing:
            self._trace_builder.reset()
            trace_buffer_ptr = Int64(self._trace_builder._buffer.data_ptr())
        else:
            trace_buffer_ptr = Int64(0)

        self._compiled_kernel(
            Int64(self._instructions_tensor.data_ptr()),
            Int64(self._barriers_tensor.data_ptr()),
            Int64(self._op_configs_tensor.data_ptr()),
            trace_buffer_ptr,
            self._num_instructions_i32,
            stream,
        )

        if sync:
            torch.cuda.synchronize()

    def bench_spec(self, setup_fn=None, keep_alive=None):
        """Create a KernelBenchSpec for raw GPU kernel timing.

        Returns a spec that can be passed to the benchmark framework for
        precise kernel-only timing (excludes CPU overhead, tensor copies,
        barrier resets, etc.).

        Args:
            setup_fn: Optional callable invoked before each timed iteration
                to reset input tensors or other state. Runs outside the
                timed region.
            keep_alive: Optional list of tensors/objects to prevent from being
                garbage collected. The kernel references GPU memory via raw
                pointers in op_configs_tensor — if the original tensors are
                freed, those pointers become dangling.

        Returns:
            KernelBenchSpec ready for use with Benchmark.run(mode="kernel").

        Example:
            kernel = Megakernel(ops, config=MegakernelConfig())
            kernel.compile()

            q_orig = q.clone()
            def reset():
                q.copy_(q_orig)

            spec = kernel.bench_spec(setup_fn=reset, keep_alive=[q, cos, sin])
        """
        import cuda.bindings.driver as cuda
        from cutlass.cute.testing import JitArguments
        from machete.utils.benchmark_utils import KernelBenchSpec

        self.compile()

        bench_stream = torch.cuda.Stream()
        cu_stream = cuda.CUstream(bench_stream.cuda_stream)

        # Capture references to internal state (stable after compile)
        compiled_kernel = self._compiled_kernel
        instructions_tensor = self._instructions_tensor
        barriers_tensor = self._barriers_tensor
        op_configs_tensor = self._op_configs_tensor
        num_instructions_i32 = self._num_instructions_i32

        def gen_workspace():
            if setup_fn is not None:
                setup_fn()
            barriers_tensor.zero_()
            return JitArguments(
                Int64(instructions_tensor.data_ptr()),
                Int64(barriers_tensor.data_ptr()),
                Int64(op_configs_tensor.data_ptr()),
                Int64(0),  # trace_buffer_ptr (tracing disabled for benchmarks)
                num_instructions_i32,
                cu_stream,
            )

        return KernelBenchSpec(
            compiled_kernel=compiled_kernel,
            workspace_generator=gen_workspace,
            stream=(bench_stream, cu_stream),
            workspace_count=1,
            _keep_alive=(self, keep_alive),  # prevent GC from freeing GPU memory
        )

    def __repr__(self) -> str:
        op_names = ", ".join(f"{op.op_cls.__name__}({op.total_tiles})" for op in self.ops)
        mode = "backward" if self.backward else "forward"
        return (
            f"Megakernel(\n"
            f"  mode={mode},\n"
            f"  ops=[{op_names}],\n"
            f"  total_tiles={self.total_tiles},\n"
            f"  num_sms={self.num_sms},\n"
            f"  num_barriers={self.num_barriers},\n"
            f"  smem={self.smem_size // 1024}KB\n"
            f")"
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_megakernel(
    ops: List[ScheduledOp],
    num_sms: Optional[int] = None,
    backward: bool = False,
    **kwargs,
) -> Megakernel:
    """Create a megakernel for the given operations.

    Args:
        ops: List of scheduled operations
        num_sms: Number of SMs (default: all available)
        backward: If True, use backward methods instead of forward
        **kwargs: Additional arguments passed to MegakernelConfig

    Returns:
        Configured Megakernel instance
    """
    config = MegakernelConfig(num_sms=num_sms, **kwargs)
    return Megakernel(ops, config=config, backward=backward)


__all__ = [
    "MegakernelConfig",
    "Megakernel",
    "create_megakernel",
    # Utilities
    "get_smem_base_ptr",
]
