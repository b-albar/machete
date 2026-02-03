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
from cutlass import Int32, Int64, Boolean
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from .ops import (
    ScheduledOp,
    ExecutionMode,
    InstructionStreamBuilder,
    TileInstruction,
)
from .compile import compile_sequential, compile_warp_specialized
from .interpreter import (
    global_barrier_wait,
    global_barrier_signal,
    load_instruction_to_smem,
    ld_global_i64,
)
from .paged_memory import (
    PAGE_SIZE,
    SharedMemoryLayout,
    init_page_table,
    init_page_states,
    st_shared_i32,
    ld_shared_i32,
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
# Per-Phase Handler Factories (Pre / Tile / Post)
# =============================================================================


def _make_pre_handler(n_pages, wait_formulas):
    """Create a @cute.jit pre-execution handler for a single op.

    Handles barrier wait and page acquire (thread 0 only).
    Does NOT include a trailing cute.arch.barrier() — the dispatch
    function handles synchronization between phases.

    Args:
        n_pages: Number of pages this op needs (0 for NOPOp)
        wait_formulas: List of BarrierFormula for dependencies to wait on
    """

    if n_pages == 0:

        @cute.jit
        def pre_handler(
            scratch_ptr: Int32,
            page_state_ptr: Int32,
            free_list_ptr: Int32,
            free_list_head_ptr: Int32,
            tile_m: Int32,
            tile_n: Int32,
            tile_l: Int32,
            tidx: Int32,
            instr_idx: Int32,
            num_pages_val: Int32,
            barriers_ptr: Int64,
        ) -> None:
            if tidx == Int32(0):
                for wf in wait_formulas:
                    # Compute linear index (without divisors) for guard check
                    _linear = Int32(wf.coeff_m) * tile_m + Int32(wf.coeff_n) * tile_n + Int32(wf.coeff_l) * tile_l
                    if _linear < Int32(wf.guard_max):
                        # Compute barrier index (with divisors for tile size ratios)
                        _idx = (
                            Int32(wf.base)
                            + (Int32(wf.coeff_m) * tile_m) // Int32(wf.div_m)
                            + (Int32(wf.coeff_n) * tile_n) // Int32(wf.div_n)
                            + (Int32(wf.coeff_l) * tile_l) // Int32(wf.div_l)
                        )
                        global_barrier_wait(barriers_ptr, _idx, Int32(wf.expected))

        return pre_handler

    @cute.jit
    def pre_handler(
        scratch_ptr: Int32,
        page_state_ptr: Int32,
        free_list_ptr: Int32,
        free_list_head_ptr: Int32,
        tile_m: Int32,
        tile_n: Int32,
        tile_l: Int32,
        tidx: Int32,
        instr_idx: Int32,
        num_pages_val: Int32,
        barriers_ptr: Int64,
    ) -> None:
        if tidx == Int32(0):
            for wf in wait_formulas:
                # Compute linear index (without divisors) for guard check
                _linear = Int32(wf.coeff_m) * tile_m + Int32(wf.coeff_n) * tile_n + Int32(wf.coeff_l) * tile_l
                if _linear < Int32(wf.guard_max):
                    # Compute barrier index (with divisors for tile size ratios)
                    _idx = (
                        Int32(wf.base)
                        + (Int32(wf.coeff_m) * tile_m) // Int32(wf.div_m)
                        + (Int32(wf.coeff_n) * tile_n) // Int32(wf.div_n)
                        + (Int32(wf.coeff_l) * tile_l) // Int32(wf.div_l)
                    )
                    global_barrier_wait(barriers_ptr, _idx, Int32(wf.expected))

            # Batch acquire: read head once, acquire all pages, write head once
            _head = ld_shared_i32(free_list_head_ptr)
            for _j in range(n_pages):
                _idx = (_head + Int32(_j)) % num_pages_val
                _pid = ld_shared_i32(free_list_ptr + _idx * 4)
                st_shared_i32(page_state_ptr + _pid * 16, instr_idx)
                st_shared_i32(scratch_ptr + Int32(_j) * 4, _pid)
            _new_head = (_head + Int32(n_pages)) % num_pages_val
            st_shared_i32(free_list_head_ptr, _new_head)

    return pre_handler


def _make_post_handler(n_pages, signal_formulas):
    """Create a @cute.jit post-execution handler for a single op.

    Handles page release and barrier signal (thread 0 only).
    Called after a cute.arch.barrier() in the dispatch function.

    Args:
        n_pages: Number of pages this op needs (0 for NOPOp)
        signal_formulas: List of BarrierFormula for barriers to signal
    """

    if n_pages == 0:

        @cute.jit
        def post_handler(
            scratch_ptr: Int32,
            page_state_ptr: Int32,
            free_list_ptr: Int32,
            free_list_tail_ptr: Int32,
            tile_m: Int32,
            tile_n: Int32,
            tile_l: Int32,
            tidx: Int32,
            num_pages_val: Int32,
            barriers_ptr: Int64,
        ) -> None:
            if tidx == Int32(0):
                for sf in signal_formulas:
                    # Compute barrier index (with divisors for tile size ratios)
                    _sidx = (
                        Int32(sf.base)
                        + (Int32(sf.coeff_m) * tile_m) // Int32(sf.div_m)
                        + (Int32(sf.coeff_n) * tile_n) // Int32(sf.div_n)
                        + (Int32(sf.coeff_l) * tile_l) // Int32(sf.div_l)
                    )
                    global_barrier_signal(barriers_ptr, _sidx)

        return post_handler

    @cute.jit
    def post_handler(
        scratch_ptr: Int32,
        page_state_ptr: Int32,
        free_list_ptr: Int32,
        free_list_tail_ptr: Int32,
        tile_m: Int32,
        tile_n: Int32,
        tile_l: Int32,
        tidx: Int32,
        num_pages_val: Int32,
        barriers_ptr: Int64,
    ) -> None:
        if tidx == Int32(0):
            # Batch release: read tail once, release all pages, write tail once
            _tail = ld_shared_i32(free_list_tail_ptr)
            for _j in range(n_pages):
                _pid = ld_shared_i32(scratch_ptr + Int32(_j) * 4)
                st_shared_i32(page_state_ptr + _pid * 16, Int32(-1))
                _slot = (_tail + Int32(_j)) % num_pages_val
                st_shared_i32(free_list_ptr + _slot * 4, _pid)
            _new_tail = (_tail + Int32(n_pages)) % num_pages_val
            st_shared_i32(free_list_tail_ptr, _new_tail)

            for sf in signal_formulas:
                # Compute barrier index (with divisors for tile size ratios)
                _sidx = (
                    Int32(sf.base)
                    + (Int32(sf.coeff_m) * tile_m) // Int32(sf.div_m)
                    + (Int32(sf.coeff_n) * tile_n) // Int32(sf.div_n)
                    + (Int32(sf.coeff_l) * tile_l) // Int32(sf.div_l)
                )
                global_barrier_signal(barriers_ptr, _sidx)

    return post_handler


def _make_tile_caller(tile_fn, n_pages):
    """Create a @cute.jit function that reads page IDs and calls a tile function.

    The n_pages branch is resolved at Python level (not inside @cute.jit)
    because CuTe DSL's AST transformer traces both branches of dynamic if
    statements, and ops that access page_ids would fail if traced with an
    empty tuple.

    Args:
        tile_fn: Compiled @cute.jit function from compile.py
        n_pages: Number of pages this op needs (0 for NOPOp)
    """

    if n_pages == 0:

        @cute.jit
        def tile_caller(
            smem_base: Int32,
            config_ptr: Int32,
            scratch_ptr: Int32,
            tile_m: Int32,
            tile_n: Int32,
            tile_l: Int32,
            op_config_ptr: Int64,
        ) -> None:
            tile_fn(smem_base, config_ptr, (), tile_m, tile_n, tile_l, op_config_ptr)

        return tile_caller

    @cute.jit
    def tile_caller(
        smem_base: Int32,
        config_ptr: Int32,
        scratch_ptr: Int32,
        tile_m: Int32,
        tile_n: Int32,
        tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        page_ids = tuple(ld_shared_i32(scratch_ptr + Int32(k) * 4) for k in range(n_pages))
        tile_fn(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr)

    return tile_caller


# =============================================================================
# Megakernel Configuration
# =============================================================================


@dataclass
class MegakernelConfig:
    """Configuration for the persistent megakernel.

    Attributes:
        threads_per_block: Threads per block (default: 256)
        num_sms: Number of SMs to use (None = auto-detect)
        num_pages: Number of shared memory pages (default: 4)
        page_size: Size of each page in bytes (default: 16KB)
        tracing: Enable cutedsl-trace instrumentation (default: False).
            When False, all trace calls are eliminated at compile time
            via constexpr (zero overhead).
    """

    threads_per_block: int = 256
    num_sms: Optional[int] = None
    num_pages: int = 4
    page_size: int = PAGE_SIZE
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

    When backward=True, the kernel dispatches to each op's backward methods
    (load_backward, compute_backward, store_backward, init_backward) instead
    of the forward methods. This enables using the same op definitions and
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

        # Ensure enough pages for the most demanding op
        max_pages_needed = max(
            (op.op_cls.NUM_INPUT_PAGES + op.op_cls.NUM_OUTPUT_PAGES for op in ops),
            default=0,
        )
        if self.config.num_pages < max_pages_needed:
            self.config.num_pages = max_pages_needed

        # Compute shared memory layout
        self._layout = SharedMemoryLayout(
            num_pages=self.config.num_pages,
            page_size=self.config.page_size,
        )

        # Build instruction stream
        self._builder = InstructionStreamBuilder()
        for op in ops:
            self._builder.add_op(
                op.op_cls,
                tiles_m=op.tiles_m,
                tiles_n=op.tiles_n,
                tiles_l=op.tiles_l,
                dim_names=op.dim_names if op.dim_names else None,
                **op.params,
            )

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
                f"({self.config.num_pages} pages × {self.config.page_size // 1024}KB + metadata), "
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

    def _build_dispatch_fn(self):
        """Build a compile-time 3-phase op dispatch function.

        Separates barrier/page management (per op index) from tile execution
        (per unique op class) to avoid duplicating heavy tile code when the
        same op class appears multiple times with different barrier formulas.

        Phase 1 (N branches): Pre-execution — barrier wait + page acquire
        Phase 2 (K branches): Tile execution — deduplicated by op class
        Phase 3 (N branches): Post-execution — page release + barrier signal

        Tile functions are compiled once per unique (op_cls, execution_mode,
        num_producer_warps) key. Pre/post handlers are lightweight (barrier
        atomics + shared memory loads/stores) so N branches have minimal
        code bloat.

        When self.backward is True, the backward methods (load_backward,
        compute_backward, store_backward, init_backward) are used instead
        of the forward methods.
        """
        ops = self.ops
        use_backward = self.backward

        # Get per-op barrier formulas
        barrier_formulas = self._builder.get_op_barrier_formulas()

        # Pre-compute page counts per op (known at JIT time)
        op_page_counts = [op.op_cls.NUM_INPUT_PAGES + op.op_cls.NUM_OUTPUT_PAGES for op in ops]

        # --- Build per-op pre/post handlers (N each) ---
        pre_handlers = []
        post_handlers = []
        for i, op in enumerate(ops):
            wait_formulas, signal_formulas = barrier_formulas.get(i, ([], []))
            pre_handlers.append(_make_pre_handler(op_page_counts[i], wait_formulas))
            post_handlers.append(_make_post_handler(op_page_counts[i], signal_formulas))

        # --- Deduplicate tile functions by (op_cls, exec_mode, producer_warps, static_dims) ---
        # dedup_key → (class_idx, tile_caller)
        dedup_map = {}
        # class_idx → list of op indices that use this class
        class_to_ops = {}

        # Select forward or backward method names
        load_attr = "load_backward" if use_backward else "load_forward"
        compute_attr = "compute_backward" if use_backward else "compute_forward"
        store_attr = "store_backward" if use_backward else "store_forward"
        init_attr = "init_backward" if use_backward else "init_forward"

        for i, op in enumerate(ops):
            # Include static dims and tensor dtypes in key: different values → different compiled code
            static_dims_key = tuple(sorted(op.static_dims.items())) if op.static_dims else ()
            # Convert dtypes to their names for hashing (CUTLASS dtype objects aren't hashable directly)
            tensor_dtypes_key = (
                tuple(sorted((k, v.__name__) for k, v in op.tensor_dtypes.items()))
                if op.tensor_dtypes
                else ()
            )
            key = (
                op.op_cls, op.execution_mode, op.num_producer_warps,
                use_backward, static_dims_key, tensor_dtypes_key,
            )
            if key not in dedup_map:
                load_fn = getattr(op.op_cls, load_attr)
                compute_fn = getattr(op.op_cls, compute_attr)
                store_fn = getattr(op.op_cls, store_attr)

                # Generate init source with static dims baked as compile-time constants.
                # Falls back to init_fn for ops without gen_init_source (e.g., NOPOp).
                init_source = None
                init_fn = None
                if hasattr(op.op_cls, "gen_init_source") and (op.static_dims or op.tensor_dtypes):
                    # Pass kernel config parameters (threads_per_row from threads_per_block)
                    kernel_config = {
                        "threads_per_row": self.config.threads_per_block,
                    }
                    init_source = op.op_cls.gen_init_source(
                        op.static_dims,
                        backward=use_backward,
                        kernel_config=kernel_config,
                        tensor_dtypes=op.tensor_dtypes,
                    )
                else:
                    init_fn = getattr(op.op_cls, init_attr, None)

                # Compile tile function once for this (class, static_dims) combo
                if op.execution_mode == ExecutionMode.WARP_SPECIALIZED:
                    tile_fn = compile_warp_specialized(
                        load_fn,
                        compute_fn,
                        store_fn,
                        init_fn=init_fn,
                        init_source=init_source,
                        num_producer_warps=op.num_producer_warps,
                        warps_per_block=self.config.warps_per_block,
                    )
                else:
                    tile_fn = compile_sequential(
                        load_fn,
                        compute_fn,
                        store_fn,
                        init_fn=init_fn,
                        init_source=init_source,
                    )

                class_idx = len(dedup_map)
                tile_caller = _make_tile_caller(tile_fn, op_page_counts[i])
                dedup_map[key] = (class_idx, tile_caller)
                class_to_ops[class_idx] = []

            class_idx, _ = dedup_map[key]
            class_to_ops[class_idx].append(i)

        # Build tile dispatch list: [(tile_caller, [op_indices]), ...]
        tile_dispatch = [(dedup_map[key][1], class_to_ops[dedup_map[key][0]]) for key in dedup_map]

        @cute.jit
        def dispatch_op(
            op_idx: Int32,
            smem_base: Int32,
            config_ptr: Int32,
            scratch_ptr: Int32,
            page_state_ptr: Int32,
            free_list_ptr: Int32,
            free_list_head_ptr: Int32,
            free_list_tail_ptr: Int32,
            tile_m: Int32,
            tile_n: Int32,
            tile_l: Int32,
            tidx: Int32,
            instr_idx: Int32,
            num_pages_val: Int32,
            barriers_ptr: Int64,
            op_configs_ptr: Int64,
        ) -> None:
            """3-phase dispatch: pre (N) → tile (K) → post (N).

            Phase 1 and 3 are keyed by op_idx (barrier formulas differ per
            op instance). Phase 2 is keyed by unique op class (tile code
            is shared across instances of the same class).
            """
            # Read per-op config pointer from global memory
            op_config_ptr = ld_global_i64(op_configs_ptr, op_idx)

            # Phase 1: Pre-execution (barrier wait + page acquire)
            for i, pre in enumerate(pre_handlers):
                if op_idx == Int32(i):
                    pre(
                        scratch_ptr,
                        page_state_ptr,
                        free_list_ptr,
                        free_list_head_ptr,
                        tile_m,
                        tile_n,
                        tile_l,
                        tidx,
                        instr_idx,
                        num_pages_val,
                        barriers_ptr,
                    )

            cute.arch.barrier()

            # Phase 2: Tile execution (deduplicated by op class)
            for tile_caller, op_indices in tile_dispatch:
                _match = Boolean(False)
                for i in op_indices:
                    if op_idx == Int32(i):
                        _match = Boolean(True)
                if _match:
                    tile_caller(
                        smem_base,
                        config_ptr,
                        scratch_ptr,
                        tile_m,
                        tile_n,
                        tile_l,
                        op_config_ptr,
                    )

            cute.arch.barrier()

            # Phase 3: Post-execution (page release + barrier signal)
            for i, post in enumerate(post_handlers):
                if op_idx == Int32(i):
                    post(
                        scratch_ptr,
                        page_state_ptr,
                        free_list_ptr,
                        free_list_tail_ptr,
                        tile_m,
                        tile_n,
                        tile_l,
                        tidx,
                        num_pages_val,
                        barriers_ptr,
                    )

        return dispatch_op

    def _create_kernel(self):
        """Create the persistent kernel with paged memory and op dispatch."""
        num_sms = self.config.num_sms
        threads_per_block = self.config.threads_per_block
        smem_size = self._layout.total_size
        layout = self._layout
        tracing = self.config.tracing

        # Capture layout offsets as compile-time constants
        control_offset = layout.control_offset
        page_table_offset = layout.page_table_offset
        page_data_offset = layout.page_data_offset
        num_pages = layout.num_pages
        page_size = layout.page_size

        # Build the dispatch function (captures ops + barrier formulas at JIT time)
        dispatch_op = self._build_dispatch_fn()

        # Common kernel body as a @cute.jit function (shared between traced/non-traced)
        @cute.jit
        def _kernel_loop(
            instructions_ptr: Int64,
            barriers_ptr: Int64,
            op_configs_ptr: Int64,
            num_instructions: Int32,
            tidx: Int32,
            block_id: Int32,
            num_blocks: Int32,
            smem_base: Int32,
            config_ptr: Int32,
            scratch_ptr: Int32,
            page_state_ptr: Int32,
            free_list_ptr: Int32,
            free_list_head_ptr: Int32,
            free_list_tail_ptr: Int32,
        ) -> None:
            """Persistent loop: fetch instructions, dispatch ops."""
            instr_idx = block_id
            continue_processing = Boolean(True)

            while continue_processing:
                if instr_idx >= num_instructions:
                    continue_processing = Boolean(False)
                else:
                    if tidx == Int32(0):
                        load_instruction_to_smem(instructions_ptr, instr_idx, scratch_ptr)

                    cute.arch.barrier()

                    op_idx = ld_shared_i32(scratch_ptr)

                    if op_idx == Int32(TileInstruction.END_MARKER):
                        continue_processing = Boolean(False)
                    else:
                        tile_m = ld_shared_i32(scratch_ptr + 4)
                        tile_n = ld_shared_i32(scratch_ptr + 8)
                        tile_l = ld_shared_i32(scratch_ptr + 12)

                        dispatch_op(
                            op_idx,
                            smem_base,
                            config_ptr,
                            scratch_ptr,
                            page_state_ptr,
                            free_list_ptr,
                            free_list_head_ptr,
                            free_list_tail_ptr,
                            tile_m,
                            tile_n,
                            tile_l,
                            tidx,
                            instr_idx,
                            Int32(num_pages),
                            barriers_ptr,
                            op_configs_ptr,
                        )

                        instr_idx = instr_idx + num_blocks

        @cute.jit
        def _init_smem(smem_base: Int32, tidx: Int32) -> None:
            """Initialize page table and page states (thread 0 only)."""
            if tidx == 0:
                init_page_table(
                    smem_base,
                    Int32(page_table_offset),
                    Int32(num_pages),
                    Int32(page_size),
                    Int32(page_data_offset),
                )
                init_page_states(
                    smem_base,
                    Int32(page_table_offset),
                    Int32(num_pages),
                )

        @cute.jit
        def _smem_pointers(smem_base: Int32):
            """Compute shared memory region pointers."""
            config_ptr = smem_base + Int32(page_table_offset)
            scratch_ptr = smem_base + Int32(control_offset)
            page_state_ptr = config_ptr + 12
            free_list_ptr = page_state_ptr + Int32(num_pages) * 16
            free_list_head_ptr = free_list_ptr + Int32(num_pages) * 4
            free_list_tail_ptr = free_list_head_ptr + 4
            return (config_ptr, scratch_ptr, page_state_ptr, free_list_ptr, free_list_head_ptr, free_list_tail_ptr)

        if tracing:
            from cutedsl_trace.device import (
                start as trace_start,
                begin_lane_dynamic_raw,
                end_event_dynamic_raw_1,
                finish_lane_dynamic_raw,
            )

            row_stride_bytes = self._trace_builder.row_stride_bytes
            op_format_ids = self._trace_format_ids

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
                        trace_buffer_ptr,
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
                    trace_buffer_ptr: Int64,
                    num_instructions: Int32,
                ):
                    tidx = cute.arch.thread_idx()[0]
                    block_id = cute.arch.block_idx()[0]
                    num_blocks = cute.arch.grid_dim()[0]
                    smem_base = get_smem_base_ptr()

                    _init_smem(smem_base, tidx)
                    cute.arch.barrier()

                    (config_ptr, scratch_ptr, page_state_ptr, free_list_ptr, free_list_head_ptr, free_list_tail_ptr) = (
                        _smem_pointers(smem_base)
                    )

                    # Trace: init lane
                    trace_buffer = cute.make_tensor(
                        cute.make_ptr(cute.Uint8, trace_buffer_ptr),
                        cute.make_layout(1 << 24),
                    )
                    lane = begin_lane_dynamic_raw(
                        Int32(1),
                        Int32(row_stride_bytes),
                        block_id,
                        Int32(0),
                        tidx == Int32(0),
                    )

                    # Persistent loop with tracing
                    instr_idx = block_id
                    continue_processing = Boolean(True)

                    while continue_processing:
                        if instr_idx >= num_instructions:
                            continue_processing = Boolean(False)
                        else:
                            if tidx == Int32(0):
                                load_instruction_to_smem(instructions_ptr, instr_idx, scratch_ptr)

                            cute.arch.barrier()
                            op_idx = ld_shared_i32(scratch_ptr)

                            if op_idx == Int32(TileInstruction.END_MARKER):
                                continue_processing = Boolean(False)
                            else:
                                tile_m = ld_shared_i32(scratch_ptr + 4)
                                tile_n = ld_shared_i32(scratch_ptr + 8)
                                tile_l = ld_shared_i32(scratch_ptr + 12)

                                _ts = trace_start()

                                dispatch_op(
                                    op_idx,
                                    smem_base,
                                    config_ptr,
                                    scratch_ptr,
                                    page_state_ptr,
                                    free_list_ptr,
                                    free_list_head_ptr,
                                    free_list_tail_ptr,
                                    tile_m,
                                    tile_n,
                                    tile_l,
                                    tidx,
                                    instr_idx,
                                    Int32(num_pages),
                                    barriers_ptr,
                                    op_configs_ptr,
                                )

                                for _i, _fmt_id in enumerate(op_format_ids):
                                    if op_idx == Int32(_i):
                                        lane = end_event_dynamic_raw_1(
                                            _ts,
                                            trace_buffer,
                                            Int32(row_stride_bytes),
                                            lane,
                                            Int32(_fmt_id),
                                            op_idx,
                                        )

                                instr_idx = instr_idx + num_blocks

                    finish_lane_dynamic_raw(trace_buffer, lane)

        else:

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

                    _init_smem(smem_base, tidx)
                    cute.arch.barrier()

                    (config_ptr, scratch_ptr, page_state_ptr, free_list_ptr, free_list_head_ptr, free_list_tail_ptr) = (
                        _smem_pointers(smem_base)
                    )

                    _kernel_loop(
                        instructions_ptr,
                        barriers_ptr,
                        op_configs_ptr,
                        num_instructions,
                        tidx,
                        block_id,
                        num_blocks,
                        smem_base,
                        config_ptr,
                        scratch_ptr,
                        page_state_ptr,
                        free_list_ptr,
                        free_list_head_ptr,
                        free_list_tail_ptr,
                    )

        return PersistentKernel()

    def _make_cache_key(self) -> Tuple:
        """Create a cache key for the compiled kernel.

        The key includes all parameters that affect kernel compilation:
        - Op classes and their static dimensions
        - Config parameters (threads, pages, etc.)
        - Backward flag
        """
        op_keys = []
        for op in self.ops:
            static_dims_tuple = tuple(sorted(op.static_dims.items())) if op.static_dims else ()
            op_keys.append((op.op_cls, static_dims_tuple))

        config_key = (
            self.config.threads_per_block,
            self.config.num_pages,
            self.config.page_size,
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
