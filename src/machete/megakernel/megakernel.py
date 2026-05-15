# Copyright (c) 2025, Machete Authors
"""Persistent megakernel runtime and launch plumbing.

The current implementation is handler-based:
- blocks stay resident and pull work from a global instruction stream
- op scheduling and barrier formulas are prepared on the host
- load/store/compute/communicate dispatch uses handler indices plus compact
  runtime metadata tables
- shared memory is managed as a paged circular buffer

This file owns host-side runtime state, launch caching, and the generated
persistent kernel shell. It does not define individual op math.
"""

import ctypes
from dataclasses import dataclass
import inspect
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple
import torch

import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr, range_constexpr

from .ops import (
    MAX_TILE_DIMS,
    Op,
    ScheduledOp,
    build_op_config,
)
from .registries import (
    TensorRegistry,
    TMARegistry,
    PeerBufferRegistry,
    PeerTMARegistry,
    validate_op_compatibility,
)
from .scheduling import (
    INSTR_BARRIER_META_IDX,
    INSTR_RANGE_END,
    INSTR_RANGE_META,
    INSTR_TILE_01,
    INSTR_TILE_23,
    InstructionStreamBuilder,
    TileInstruction,
)
from .compile import exec_generated_source
from .backend import build_backend
from .backend_ir import PHASE_NAMES
from .interpreter import (
    global_barrier_signal,
    global_barrier_signal_gpu,
    global_barrier_wait,
    global_barrier_wait_relaxed,
    load_instruction_to_smem,
    prefetch_instruction,
    ld_global_i32,
    ld_global_i64,
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_arrive,
    mbarrier_wait,
    named_barrier_sync,
    get_smem_base_ptr,
    nanosleep,
)
from .paged_memory import (
    MAX_PAGES,
    NPageLayout,
    PipelinePageLayout,
    st_shared_i32,
    ld_shared_i32,
    st_shared_i64,
    ld_shared_i64,
    st_shared_release_cta_i32,
    ld_shared_acquire_cta_i32,
    ld_shared_v2_b32,
    st_shared_v2_b32,
    FLAG_DISPATCH_LOAD,
    FLAG_PRODUCE_IDX,
    FLAG_STORE_IDX,
    FLAG_LOAD_DONE,
    FLAG_DATA_RELEASE_IDX,
    FLAG_DATA_PRODUCE_IDX,
    TILE_INFO_HANDLER_IDX as _TILE_INFO_HANDLER_IDX,
    TILE_INFO_INSTRUCTION_IDX as _TILE_INFO_INSTRUCTION_IDX,
    TILE_INFO_OP_CONFIG as _TILE_INFO_OP_CONFIG,
    TILE_INFO_PAGE_ID as _TILE_INFO_PAGE_ID,
    TILE_INFO_TILE_0 as _TILE_INFO_TILE_0,
    TILE_INFO_TILE_1 as _TILE_INFO_TILE_1,
    TILE_INFO_TILE_2 as _TILE_INFO_TILE_2,
    TILE_INFO_TILE_3 as _TILE_INFO_TILE_3,
)


# =============================================================================
# Megakernel Configuration
# =============================================================================

# Three DMA warps: controller (fetch + barrier wait), loader (TMA dispatch), store (S→G).
NUM_DMA_WARPS = 3

# PTX minimum register count for setmaxnreg (idle warp floor).
MIN_IDLE_REGS = 24

_PHASE_LOCAL_IDX_PTR_ATTRS = {
    "load": "load_local_idx_ptr",
    "compute": "compute_local_idx_ptr",
    "store": "store_local_idx_ptr",
    "communicate": "communicate_local_idx_ptr",
}
_PHASE_TRANSPORT_PTR_ATTRS = {
    "load": "load_local_transport_positions_ptr",
    "compute": "compute_local_transport_positions_ptr",
    "store": "store_local_transport_positions_ptr",
    "communicate": "communicate_local_transport_positions_ptr",
}
_PHASE_DESC_SLOT_PTR_ATTRS = {
    "load": "load_local_desc_slots_ptr",
    "compute": "compute_local_desc_slots_ptr",
    "store": "store_local_desc_slots_ptr",
    "communicate": "communicate_local_desc_slots_ptr",
}

# Per-op metadata layout (int32 entries).
_OP_META_NUM_WARPS = 0
_OP_META_STRIDE_0 = 1
_OP_META_STRIDE_1 = 2
_OP_META_STRIDE_2 = 3
_OP_META_STRIDE_3 = 4
_OP_META_COUNT_0 = 5
_OP_META_COUNT_1 = 6
_OP_META_COUNT_2 = 7
_OP_META_COUNT_3 = 8
_OP_META_HANDLER_IDX = 9
_OP_META_LOAD_LOCAL_IDX = 10
_OP_META_COMPUTE_LOCAL_IDX = 11
_OP_META_STORE_LOCAL_IDX = 12
_OP_META_COMM_LOCAL_IDX = 13
_OP_META_WAIT_COUNT = 14
_OP_META_COMPUTE_WAIT_COUNT = 15
_OP_META_SIGNAL_COUNT = 16
_OP_META_WAIT_ACQUIRE = 17
_OP_META_PHASE_MASK = 18
_OP_META_STRIDE = 19

_OP_PHASE_LOAD = 1
_OP_PHASE_COMPUTE = 2
_OP_PHASE_STORE = 4
_OP_PHASE_COMMUNICATE = 8

_INSTR_NO_SMEM_PAGE_BIT = 13

# Per-signal-formula metadata layout (int32 entries).
_SIGNAL_META_BASE = 0
_SIGNAL_META_GUARD_MAX = 1
_SIGNAL_META_COEFF_0 = 2
_SIGNAL_META_DIV_0 = 7
_SIGNAL_META_STRIDE = 12

@dataclass
class MegakernelConfig:
    """Configuration for the persistent megakernel.

    Kernel geometry:
        threads_per_block: Total threads (MMA warps + DMA warps). Default 256.
        num_sms: SMs to occupy (None = auto-detect from device).
        page_size: Shared memory page size in bytes (default: 48KB).
        num_pages: Ring buffer pages (None = auto-detect max that fits smem).
            With N pages, up to N-1 loads overlap with compute. N=1 is valid
            but serializes load→compute→store (no pipelining).

    Register budget:
        dma_reg_count: Registers per DMA warp thread (default: 40).
        mma_reg_count: Registers per MMA warp thread (default: 232).
            Total must fit: num_mma_warps * mma_reg_count + NUM_DMA_WARPS * dma_reg_count
            <= 65536 registers per SM.

    Compilation:
        opt_level: LLVM optimization level 0-3 (default: 2).
        tracing: Enable cutedsl-trace instrumentation (default: False).
            When False, trace blocks are stripped at source level (zero overhead).

    Multi-GPU (peer TMA):
        peer_buffers: {tensor_name: [peer0_tensor, peer1_tensor, ...]}
        peer_barriers: torch.Tensor for cross-GPU barrier signaling.
        device_idx: This GPU's index in the peer group.
        num_devices: Total devices in peer group.
    """

    threads_per_block: int = 256
    num_sms: Optional[int] = None
    page_size: int = 49152
    num_pages: Optional[int] = None
    tracing: bool = False
    dma_reg_count: int = 40
    mma_reg_count: int = 232
    loader_idle_sleep_ns: int = 100
    relaxed_global_barriers: bool = True
    global_barrier_sleep_ns: int = 8
    opt_level: int = 2

    # Multi-GPU communication
    peer_buffers: Optional[Dict[str, List[Any]]] = None
    peer_barriers: Optional[Any] = None
    device_idx: int = 0
    num_devices: int = 1


@dataclass
class _LaunchState:
    """Stable kernel launch arguments cached after compile()."""

    instructions_ptr: Int64
    barriers_ptr: Int64
    op_configs_ptr: Int64
    wait_info_ptr: Int64
    compute_wait_info_ptr: Int64
    op_meta_ptr: Int64
    load_local_idx_ptr: Int64
    compute_local_idx_ptr: Int64
    store_local_idx_ptr: Int64
    communicate_local_idx_ptr: Int64
    load_local_transport_positions_ptr: Int64
    compute_local_transport_positions_ptr: Int64
    store_local_transport_positions_ptr: Int64
    communicate_local_transport_positions_ptr: Int64
    load_local_desc_slots_ptr: Int64
    compute_local_desc_slots_ptr: Int64
    store_local_desc_slots_ptr: Int64
    communicate_local_desc_slots_ptr: Int64
    local_tma_desc_pool_ptr: Int64
    peer_tma_desc_pool_ptr: Int64
    signal_meta_ptr: Int64
    peer_signal_ptr: Int64
    trace_buffer_ptr: Int64
    cute_tensors: List[Any]
    tma_tensor_args: List[Any]
    peer_tma_tensor_args: List[Any]


def _unload_compiled_jit_module(compiled) -> None:
    jit_module = getattr(compiled, "jit_module", None)
    if jit_module is not None and not jit_module.is_unloaded():
        jit_module.unload()


def _sync_tma_desc_init_stream(stream) -> None:
    """Drain descriptor initialization before kernels consume runtime TMA maps."""
    import cuda.bindings.driver as cuda

    err, = cuda.cuStreamSynchronize(stream)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuStreamSynchronize failed after TMA descriptor init: {err}")


class _CompiledKernelCache(dict):
    """Compiled-kernel cache that drops Python launch caches on explicit clear."""

    def clear(self) -> None:
        super().clear()
        owner = globals().get("Megakernel")
        if owner is not None:
            owner._dispatch_inputs_cache.clear()


# =============================================================================
# Megakernel Implementation
# =============================================================================


class Megakernel:
    """Persistent megakernel with instruction stream, paged memory, and op dispatch.

    Caching:
        Compiled kernels are cached at the class level to avoid recompilation
        when creating multiple Megakernel instances with the same configuration.
        The cache key is based on: (op_classes, static_dims, config_params).

    Architecture:
    - all SMs launch persistent blocks
    - each block fetches instructions from global memory in a strided pattern
    - fine-grained barriers enable tile-level dependencies between ops
    - shared memory is divided into pages managed by a circular buffer
    - phase dispatch uses the handler backend plus runtime metadata tables

    Work Distribution:
        Block 0: instr 0 -> instr num_sms -> instr 2*num_sms -> ...
        Block 1: instr 1 -> instr num_sms+1 -> instr 2*num_sms+1 -> ...

    Example:
        ops = [
            ScheduledOp(RMSNormOp, tile_counts=(32,)),
            ScheduledOp(MatVecOp, tile_counts=(32,)),  # Can overlap with RMSNorm!
        ]
        kernel = Megakernel(ops)
        kernel.run()
    """

    # Class-level cache for compiled kernels to avoid recompilation
    # Key: (op_classes_tuple, static_dims_tuple, config_key)
    _compiled_kernel_cache: Dict[Tuple, Any] = _CompiledKernelCache()
    _dispatch_inputs_cache: Dict[Tuple, Any] = {}
    _active_tma_jit_modules: List[Any] = []

    @staticmethod
    def _cache_key_has_tma(cache_key: Tuple) -> bool:
        return bool(cache_key[-2] or cache_key[-1])

    @classmethod
    def _drop_cached_tma_kernels(cls) -> None:
        """Drop compiled TMA kernels before compiling a new TMA signature.

        Keep the JIT modules loaded. Explicitly unloading a live CuTe TMA
        module can corrupt subsequent TMA kernels in the same process; dropping
        the Python cache entry is enough to avoid reusing incompatible launch
        wrappers while preserving the CUDA module lifetime.
        """
        stale_keys = [
            key for key in cls._compiled_kernel_cache
            if cls._cache_key_has_tma(key)
        ]
        for key in stale_keys:
            cls._compiled_kernel_cache.pop(key, None)
        for key in list(cls._dispatch_inputs_cache):
            if key and key[0] == "dispatch_inputs" and cls._cache_key_has_tma(key[1]):
                cls._dispatch_inputs_cache.pop(key, None)

    @classmethod
    def _unload_active_tma_modules(cls) -> None:
        cls._active_tma_jit_modules = [
            jit_module for jit_module in cls._active_tma_jit_modules
            if jit_module is not None and not jit_module.is_unloaded()
        ]

    @classmethod
    def _track_active_tma_module(cls, compiled) -> None:
        jit_module = getattr(compiled, "jit_module", None)
        if jit_module is not None and not jit_module.is_unloaded():
            cls._active_tma_jit_modules.append(jit_module)

    @staticmethod
    def _compiled_kernel_module_is_unloaded(compiled) -> bool:
        jit_module = getattr(compiled, "jit_module", None)
        return jit_module is not None and jit_module.is_unloaded()

    @staticmethod
    def _pipeline_for_cls(op_cls):
        return getattr(op_cls, "pipeline", None)

    @staticmethod
    def _pipeline_static_dim(op: ScheduledOp, name: str):
        return op.static_dims.get(f"pipeline_{name}")

    @classmethod
    def _pipeline_spec_for_op(cls, op: ScheduledOp):
        pipeline = cls._pipeline_for_cls(op.op_cls)
        if pipeline is None:
            return None
        return pipeline.with_overrides(
            page_count=cls._pipeline_static_dim(op, "page_count"),
            page_bytes=cls._pipeline_static_dim(op, "page_bytes"),
            semaphore_count=cls._pipeline_static_dim(op, "semaphore_count"),
            scratch_bytes=cls._pipeline_static_dim(op, "scratch_bytes"),
            input_stages=cls._pipeline_static_dim(op, "input_stages"),
            output_stages=cls._pipeline_static_dim(op, "output_stages"),
            stage_pages=cls._pipeline_static_dim(op, "stage_pages"),
        )

    @classmethod
    def _inject_pipeline_resource_static_dims(cls, op: ScheduledOp, pipeline) -> None:
        """Expose op-owned page/semaphore offsets as compile-time op attrs."""
        protocol = getattr(op.op_cls, "pipeline_page_protocol", None)
        if protocol is None:
            protocol = pipeline.page_protocol()
        if protocol is None:
            return
        if protocol.page_count != pipeline.page_count:
            raise ValueError(
                f"Op {op.op_cls.__name__} pipeline protocol declares "
                f"{protocol.page_count} pages, but PipelineSpec declares "
                f"{pipeline.page_count}."
            )
        if protocol.semaphore_count != pipeline.semaphore_count:
            raise ValueError(
                f"Op {op.op_cls.__name__} pipeline protocol declares "
                f"{protocol.semaphore_count} semaphores, but PipelineSpec declares "
                f"{pipeline.semaphore_count}."
            )
        layout = PipelinePageLayout(
            page_count=pipeline.page_count,
            page_bytes=pipeline.page_bytes,
            semaphore_count=pipeline.semaphore_count,
            scratch_bytes=pipeline.scratch_bytes,
        )
        op.static_dims.setdefault("machete_pipeline_page_count", pipeline.page_count)
        op.static_dims.setdefault("machete_pipeline_page_bytes", pipeline.page_bytes)
        op.static_dims.setdefault("machete_pipeline_semaphore_count", pipeline.semaphore_count)
        op.static_dims.setdefault("machete_pipeline_scratch_offset", layout.scratch_offset)
        op.static_dims.setdefault("machete_pipeline_total_size", layout.total_size)
        for page_idx in range(pipeline.page_count):
            op.static_dims.setdefault(
                f"machete_pipeline_page_{page_idx}_offset",
                layout.page_offset(page_idx),
            )
        for sem_idx in range(pipeline.semaphore_count):
            op.static_dims.setdefault(
                f"machete_pipeline_semaphore_{sem_idx}_offset",
                layout.semaphore_offset(sem_idx),
            )

    @staticmethod
    def _active_range_phase_ownership(instance, range_end_param: str) -> Dict[str, bool]:
        """Return which active phases explicitly consume the coalesced range end.

        A staged op may use a framework-owned range, where the runtime expands
        each subtile through load/compute/store, or an op-owned range, where
        every active phase understands the same range end. Mixing the two is
        ambiguous: one phase sees a range while another still sees only the
        first subtile.
        """
        active: Dict[str, bool] = {}
        for phase_name in PHASE_NAMES:
            method = getattr(instance.__class__, phase_name, None)
            base_method = getattr(Op, phase_name, None)
            if phase_name == "compute" or method is not base_method:
                active[phase_name] = range_end_param in inspect.signature(
                    getattr(instance, phase_name)
                ).parameters
        return active

    def _framework_expands_range(self, op_idx: int, range_end_axis: int) -> bool:
        """Return whether the persistent shell expands a coalesced op range.

        If every active phase accepts the range-end tile parameter, the op owns
        the full range and the shell emits a single ring item. If no active
        phase accepts it, the shell expands the range into per-subtile work.
        Metadata rows must match that runtime choice.
        """
        if op_idx < 0 or op_idx >= len(self.ops):
            return True
        if range_end_axis < 0:
            return False
        if range_end_axis >= MAX_TILE_DIMS:
            return True
        op = self.ops[op_idx]
        if bool(op.static_dims.get("framework_owned_ranges", False)) or bool(
            getattr(op.op_cls, "framework_owned_ranges", False)
        ):
            return True

        threads_per_block = self.config.threads_per_block
        num_dma_warps = 0 if self._use_compute_only_replay() else NUM_DMA_WARPS
        num_compute_threads = threads_per_block - num_dma_warps * 32
        kernel_config = {"threads_per_row": num_compute_threads}
        config = build_op_config(op, kernel_config=kernel_config)
        instance = op.op_cls(**config)
        range_phase_owners = self._active_range_phase_ownership(instance, f"tile_{range_end_axis}")
        range_owner_count = sum(1 for owns in range_phase_owners.values() if owns)
        if (
            len(range_phase_owners) > 1
            and range_owner_count > 0
            and range_owner_count != len(range_phase_owners)
        ):
            owners = ", ".join(
                f"{phase}={'range' if owns else 'single'}"
                for phase, owns in range_phase_owners.items()
            )
            raise ValueError(
                f"{op.op_cls.__name__} has partial coalesced-range ownership "
                f"({owners}). Staged range ops must either let the framework "
                "expand every subtile through all phases, or every active "
                f"phase must accept tile_{range_end_axis}."
            )
        return range_owner_count == 0

    def __init__(
        self,
        ops: List[ScheduledOp],
        config: Optional[MegakernelConfig] = None,
        device: str = "cuda",
        scheduler: Optional["TileScheduler"] = None,
    ):
        """Construct a megakernel instance and build its host-side runtime state."""
        self.scheduled_items = tuple(ops)
        self.ops = list(ops)
        self.config = config or MegakernelConfig()
        self.device = device
        self._scheduler = scheduler
        ops = self.ops
        if any(getattr(op.op_cls, "pipeline", None) is not None for op in ops):
            Megakernel._unload_active_tma_modules()
            Megakernel._drop_cached_tma_kernels()
        self._has_page_free_ops = (
            os.environ.get("MACHETE_ENABLE_PAGE_FREE_OPS", "0") == "1"
            and any(not bool(getattr(op.op_cls, "uses_smem_page", True)) for op in ops)
        )
        # Keep instruction slots separate from physical shared-memory pages for
        # multi-page kernels. This prevents slot metadata/mbarrier phase reuse
        # from being coupled to page reuse.
        self._use_physical_page_ring = self._has_page_free_ops

        # Detect resident block count if not specified.
        #
        # Occupying every SM is wasteful for small workloads: the persistent
        # shell cost scales with resident blocks even when there are only a few
        # tiles to process. Cap the default grid by available work so short
        # sequences and small fused graphs do not launch dozens of idle
        # persistent blocks.
        if self.config.num_sms is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Megakernel requires a CUDA GPU.")
            props = torch.cuda.get_device_properties(device)
            total_tiles = max(1, sum(op.total_tiles for op in ops))
            self.config.num_sms = min(props.multi_processor_count, total_tiles)
        if self._scheduler is not None and hasattr(self._scheduler, "bind_num_blocks"):
            self._scheduler.bind_num_blocks(self.config.num_sms)

        # Create N-page layout (auto-detect max pages or use user-specified)
        if self.config.num_pages is not None:
            # User specified number of pages
            num_slots = self.config.num_pages
            if self.config.num_pages > 1 or self._has_page_free_ops:
                num_slots = min(MAX_PAGES, self.config.num_pages + 2)
                self._use_physical_page_ring = True
            self._layout = NPageLayout(
                num_pages=self.config.num_pages,
                num_slots=num_slots,
                page_size=self.config.page_size,
            )
        else:
            # Auto-detect maximum pages that fit in shared memory
            self._layout = NPageLayout.for_device(
                page_size=self.config.page_size,
                min_pages=1,
            )
            if self._layout.num_pages > 1 or self._has_page_free_ops:
                self._use_physical_page_ring = True
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(torch.cuda.current_device())
                    max_smem = props.shared_memory_per_block_optin
                else:
                    max_smem = 228 * 1024
                for n in range(self._layout.num_pages, 0, -1):
                    num_slots = min(MAX_PAGES, n + 2) if n > 1 or self._has_page_free_ops else n
                    candidate = NPageLayout(
                        num_pages=n,
                        num_slots=num_slots,
                        page_size=self.config.page_size,
                    )
                    if candidate.total_size <= max_smem:
                        self._layout = candidate
                        self._use_physical_page_ring = n > 1 or self._has_page_free_ops
                        break
            # Store computed num_pages back to config for cache key
            self.config.num_pages = self._layout.num_pages

        # Validate that config page_size is large enough for all ops
        for op in ops:
            op_page = op.static_dims.get('page_size')
            if op_page is not None and op_page > self.config.page_size:
                raise ValueError(
                    f"Op {op.op_cls.__name__} was scheduled for "
                    f"page_size={op_page}B but megakernel config has "
                    f"page_size={self.config.page_size}B. Use "
                    f"kernel_config(ops) or increase config.page_size."
                )
            pipeline = self._pipeline_spec_for_op(op)
            if pipeline is not None:
                self._inject_pipeline_resource_static_dims(op, pipeline)
                if "inner_iter_idx" in inspect.signature(op.op_cls.load).parameters:
                    raise ValueError(
                        f"Op {op.op_cls.__name__} load phases must own staged "
                        "loops inside the op body."
                    )
                required = int(pipeline.resource_bytes)
                effective_page = int(op_page or self.config.page_size)
                if required > effective_page:
                    raise ValueError(
                        f"Op {op.op_cls.__name__} declares pipeline "
                        f"resources requiring {required}B inside one page, "
                        f"but effective page_size is {effective_page}B. "
                        f"Increase page_size or reduce the pipeline page/scratch shape."
                    )

        # Build instruction stream
        # Pass ScheduledOp directly to preserve tensor_ptrs for automatic dependency detection
        self._builder = InstructionStreamBuilder()
        for op in ops:
            self._builder.add_op(op)

        self._instructions_tensor: Optional[torch.Tensor] = None
        self._barriers_tensor: Optional[torch.Tensor] = None
        self._op_configs_tensor: Optional[torch.Tensor] = None
        self._num_instructions: Optional[int] = None
        self._num_instructions_i32: Optional[Int32] = None
        self._compiled_kernel = None
        self._has_pending_async_launch = False
        self._needs_tma_desc_pool_init = True
        self._op_metadata_tensor: Optional[torch.Tensor] = None
        self._signal_metadata_tensor: Optional[torch.Tensor] = None
        self._wait_info: Optional[torch.Tensor] = None
        self._compute_wait_info: Optional[torch.Tensor] = None
        self._phase_local_idx_tensors: Dict[str, Optional[torch.Tensor]] = {
            phase: None for phase in PHASE_NAMES
        }
        self._phase_local_transport_position_tensors: Dict[str, Optional[torch.Tensor]] = {
            phase: None for phase in PHASE_NAMES
        }
        self._phase_local_transport_position_widths: Dict[str, int] = {
            phase: 0 for phase in PHASE_NAMES
        }
        self._phase_local_desc_slot_tensors: Dict[str, Optional[torch.Tensor]] = {
            phase: None for phase in PHASE_NAMES
        }
        self._phase_local_desc_slot_widths: Dict[str, int] = {
            phase: 0 for phase in PHASE_NAMES
        }
        self._local_tma_desc_pool: Optional[torch.Tensor] = None
        self._peer_tma_desc_pool: Optional[torch.Tensor] = None
        self._peer_signal_tensor: Optional[torch.Tensor] = None
        self._max_signal_formulas: int = 1

        # Tensor parameter mode: build registry, validate compatibility, prepare tensors
        self._tensor_registry = TensorRegistry.from_ops(ops)
        validate_op_compatibility(ops, self._tensor_registry)
        self._cute_tensors: Optional[List] = None  # torch.Tensor objects for kernel params

        # TMA parameter mode: build TMA registry for descriptor management
        self._tma_registry = TMARegistry.from_ops(ops, self._tensor_registry)
        self._tma_cute_tensors: Optional[List] = None  # CuTe tensors with static layout for TMA

        # Peer TMA parameter mode: build peer registries for multi-GPU communication
        if self.config.peer_buffers:
            self._peer_buffer_registry = PeerBufferRegistry.from_config(
                self.config.peer_buffers, self._tensor_registry, ops
            )
            self._peer_tma_registry = PeerTMARegistry.from_ops(ops, self._tensor_registry, self._peer_buffer_registry)
        else:
            self._peer_buffer_registry = PeerBufferRegistry(buffers=[], num_peers=0)
            self._peer_tma_registry = PeerTMARegistry(descriptors=[], op_mappings={}, num_peers=0)
        self._peer_tma_cute_tensors: Optional[List] = None
        self._tma_runtime_layout_cache: Dict[int, Any] = {}
        self._backend_ir, self._backend = build_backend(self)
        self._launch_state: Optional[_LaunchState] = None
        self._cached_cu_stream = None
        self._cached_torch_stream_id = None
        self._max_wait_formulas: int = 1
        self._max_compute_wait_formulas: int = 1

        # Validate barrier formulas eagerly (catches incompatible tile sizes early)
        _ = self._builder.num_barriers

        # Trace setup
        from .tracing import setup_tracing

        self._tracing_state = None
        if self.config.tracing:
            from cutedsl_trace.config import set_tracing_enabled

            set_tracing_enabled(True)
            self._tracing_state = setup_tracing(
                self.ops, self.num_sms, self.total_tiles, device=self.device
            )

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
        """Prepare instruction, barrier, wait_info, and config tensors on GPU."""
        if self._instructions_tensor is None:
            instructions = self._builder.build(scheduler=self._scheduler)
            framework_range_cache: Dict[Tuple[int, int], bool] = {}

            def _framework_expands_instruction_range(instr: TileInstruction) -> bool:
                if instr.range_axis < 0:
                    return False
                key = (instr.op_idx, instr.range_end_axis)
                if key not in framework_range_cache:
                    framework_range_cache[key] = self._framework_expands_range(
                        instr.op_idx,
                        instr.range_end_axis,
                    )
                return framework_range_cache[key]

            instructions = self._builder.coalesce_pipeline_instructions(
                instructions,
                num_blocks=self.num_sms,
                framework_expands_predicate=_framework_expands_instruction_range,
            )
            (
                barrier_meta_indices,
                expanded_instructions,
            ) = self._builder.pipeline_barrier_meta_indices(
                instructions,
                expand_predicate=_framework_expands_instruction_range,
            )

            def _runtime_instruction(instr: TileInstruction) -> TileInstruction:
                if (
                    instr.op_idx == TileInstruction.END_MARKER
                    or instr.range_axis < 0
                    or _framework_expands_instruction_range(instr)
                ):
                    return instr
                tiles = list(instr.tiles) + [0] * (MAX_TILE_DIMS - len(instr.tiles))
                if 0 <= instr.range_end_axis < MAX_TILE_DIMS:
                    tiles[instr.range_end_axis] = instr.range_end
                return TileInstruction(instr.op_idx, tuple(tiles[:MAX_TILE_DIMS]))

            instructions = [_runtime_instruction(instr) for instr in instructions]
            self._wait_info = self._builder.build_wait_info_tensor(
                expanded_instructions,
                self.device,
                num_blocks=self.num_sms,
            )
            self._max_wait_formulas = max(1, self._wait_info.shape[1] // 2)
            self._compute_wait_info = self._builder.build_compute_wait_info_tensor(
                expanded_instructions,
                self.device,
                num_blocks=self.num_sms,
            )
            self._max_compute_wait_formulas = max(
                1, self._compute_wait_info.shape[1] // 2
            )
            self._signal_metadata_tensor = self._builder.build_signal_info_tensor(
                expanded_instructions, self.device
            )
            self._max_signal_formulas = max(1, self._signal_metadata_tensor.shape[1])
            if self._op_metadata_tensor is None:
                self._prepare_op_metadata_tensors()
            self._instructions_tensor = self._builder.build_tensor(
                self.device,
                scheduler=self._scheduler,
                instructions=instructions,
                barrier_meta_indices=barrier_meta_indices,
            )
            self._num_instructions = len(instructions)
            self._num_instructions_i32 = Int32(self._num_instructions)
        elif self._op_metadata_tensor is None:
            self._prepare_op_metadata_tensors()

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

    @staticmethod
    def _tile_linear_strides(tile_counts: Tuple[int, ...]) -> Tuple[int, ...]:
        """Return row-major strides padded to MAX_TILE_DIMS."""
        strides = [0] * MAX_TILE_DIMS
        stride = 1
        for i in range(len(tile_counts) - 1, -1, -1):
            strides[i] = stride
            stride *= tile_counts[i]
        return tuple(strides)

    def _prepare_op_metadata_tensors(self) -> None:
        """Build runtime metadata tables for the persistent shell."""
        formulas = self._builder.get_op_barrier_formulas()
        threads_per_block = self.config.threads_per_block
        num_dma_warps = 0 if self._use_compute_only_replay() else NUM_DMA_WARPS
        num_compute_threads = threads_per_block - num_dma_warps * 32
        default_num_mma_warps = num_compute_threads // 32

        op_meta = []
        has_communicate = self._peer_tma_registry.has_peer_tma
        peer_signal_offsets = [] if has_communicate else None
        peer_barrier_offset = 0
        op_handler_indices = self._backend.handler_indices()
        op_phase_local_indices = {
            phase: self._backend.phase_local_indices(phase)
            for phase in PHASE_NAMES
        }
        from .backend_dispatch import (
            _build_tma_runtime_layout,
        )
        tma_layout = _build_tma_runtime_layout(self._backend, self)
        runtime_transport_records = getattr(self._backend, "runtime_transport_records", False)
        if runtime_transport_records:
            op_phase_local_indices = {
                phase: list(tma_layout.runtime_op_phase_transport_indices[phase])
                for phase in PHASE_NAMES
            }

        for op_idx, op in enumerate(self.ops):
            kernel_config = {"threads_per_row": num_compute_threads}
            config = build_op_config(op, kernel_config=kernel_config)
            instance = op.op_cls(**config)
            if "inner_iter_idx" in inspect.signature(instance.load).parameters:
                raise ValueError(
                    f"{op.op_cls.__name__}.load uses inner_iter_idx, but "
                    "runtime load loops must live inside the op load body."
                )
            tile_strides = self._tile_linear_strides(op.tile_counts)
            _wait_formulas, _signal_formulas = formulas.get(op_idx, ([], []))
            wait_count = self._builder._op_wait_counts.get(op_idx, len(_wait_formulas))
            compute_wait_count = self._builder._op_compute_wait_counts.get(op_idx, 0)
            signal_count = self._builder._op_signal_counts.get(op_idx, len(_signal_formulas))
            phase_mask = _OP_PHASE_COMPUTE
            if getattr(op.op_cls, "load") is not Op.load:
                phase_mask |= _OP_PHASE_LOAD
            if getattr(op.op_cls, "store") is not Op.store:
                phase_mask |= _OP_PHASE_STORE
            if getattr(op.op_cls, "communicate") is not Op.communicate:
                phase_mask |= _OP_PHASE_COMMUNICATE
            instruction_phase_mask = phase_mask
            uses_smem_page = bool(getattr(op.op_cls, "uses_smem_page", True))
            if not uses_smem_page:
                if phase_mask != _OP_PHASE_COMPUTE:
                    raise ValueError(
                        f"{op.op_cls.__name__} declares uses_smem_page=False but "
                        "has active load/store/communicate phases. Page-free ops "
                        "must be compute-only."
                    )
                instruction_phase_mask |= 1 << _INSTR_NO_SMEM_PAGE_BIT
            num_warps = int(getattr(instance, "num_mma_warps", default_num_mma_warps))
            handler_idx = int(op_handler_indices[op_idx])
            op_meta.extend(
                [
                    num_warps,
                    *tile_strides,
                    *tuple(op.tile_counts) + (1,) * (MAX_TILE_DIMS - len(op.tile_counts)),
                    handler_idx,
                    *self._phase_local_indices_for_op(op_phase_local_indices, op_idx),
                    wait_count,
                    compute_wait_count,
                    signal_count,
                    int(op.static_dims.get("barrier_wait_acquire", 0)),
                    instruction_phase_mask,
                ]
            )

            if has_communicate:
                if self._op_has_peer_barriers(op):
                    peer_signal_offsets.append(peer_barrier_offset)
                    peer_barrier_offset += op.total_tiles
                else:
                    peer_signal_offsets.append(-1)

        self._op_metadata_tensor = torch.tensor(
            op_meta, dtype=torch.int32, device=self.device
        )
        for phase in PHASE_NAMES:
            if tma_layout.phase_uses_local_idx.get(phase, False):
                self._phase_local_idx_tensors[phase] = torch.tensor(
                    op_phase_local_indices[phase], dtype=torch.int32, device=self.device
                )
            else:
                self._phase_local_idx_tensors[phase] = None

            local_transport_positions = tma_layout.runtime_phase_local_transport_positions[phase]
            width, flat_positions = self._encode_phase_selector_table(local_transport_positions)
            self._set_phase_i32_table(
                phase,
                tma_layout.phase_uses_transport_selector[phase],
                flat_positions,
                width,
                self._phase_local_transport_position_widths,
                self._phase_local_transport_position_tensors,
            )

            local_desc_slots = tma_layout.runtime_phase_local_desc_slots[phase]
            width, flat_slots = self._encode_phase_selector_table(local_desc_slots)
            self._set_phase_i32_table(
                phase,
                tma_layout.phase_uses_desc_slot_selector[phase],
                flat_slots,
                width,
                self._phase_local_desc_slot_widths,
                self._phase_local_desc_slot_tensors,
            )
        if has_communicate:
            self._peer_signal_tensor = torch.tensor(
                peer_signal_offsets, dtype=torch.int32, device=self.device
            )
        else:
            self._peer_signal_tensor = None
        self._local_tma_desc_pool = torch.empty(
            max(1, 128 * len(self._tma_registry.descriptors)),
            dtype=torch.uint8,
            device=self.device,
        )
        self._peer_tma_desc_pool = torch.empty(
            max(1, 128 * len(self._peer_tma_registry.descriptors)),
            dtype=torch.uint8,
            device=self.device,
        )

    def _prepare_cute_tensors(self) -> None:
        """Prepare tensor objects for kernel parameters.

        Pre-converts all tensors to CuTe _Tensor objects via from_dlpack +
        mark_layout_dynamic. This avoids the expensive TensorAdapter
        conversion (from_dlpack) on every kernel launch — the CuTe _Tensor
        objects have a cached __c_pointers__() that costs ~0.03us per tensor
        vs ~3us per tensor for TensorAdapter re-creation.

        Tensors are threaded as runtime parameters through:
        PersistentKernel -> kernel_loop -> dispatch -> phase_fn.
        """
        if self._cute_tensors is not None:
            return

        from cutlass.cute.runtime import from_dlpack

        self._cute_tensors = []
        for canonical_name, tensor, dtype in self._tensor_registry.tensors:
            # detach() for gradient-tracking tensors.  Do NOT call
            # .contiguous() — ops that store per-dim strides in
            # static_dims (e.g., FlashAttention K/V) rely on the
            # original strided layout.  from_dlpack handles strided
            # tensors via DLPack strides.
            t = tensor.detach()
            # Convert all tensors to CuTe _Tensor objects with explicit
            # leading_dim to handle ambiguous strides (size-1 dims).
            cute_t = from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
            self._cute_tensors.append(cute_t)

    def _registry_tensor_for_canonical(self, canonical_name: str) -> torch.Tensor:
        """Look up a tensor by canonical registry name."""
        for registry_name, tensor, _dtype in self._tensor_registry.tensors:
            if registry_name == canonical_name:
                return tensor
        raise KeyError(f"Unknown tensor canonical name: {canonical_name}")

    def _reshape_tensor_for_tma(self, tensor: torch.Tensor, target_ndim: int) -> torch.Tensor:
        """Match a tensor's rank to a TMA descriptor rank.

        CuTe may merge contiguous modes when importing tensors. Reshaping the
        leading dimensions before descriptor creation keeps the imported rank
        aligned with the descriptor's tile shape.
        """
        if tensor.ndim <= target_ndim:
            return tensor

        keep_trailing_dims = target_ndim - 1
        if keep_trailing_dims > 0:
            return tensor.reshape(-1, *tensor.shape[-keep_trailing_dims:])
        return tensor.reshape(-1)

    def _permute_tensor_for_tma(self, tensor: torch.Tensor, dim_perm: Tuple[int, ...] = ()) -> torch.Tensor:
        """Permute tensor dims into a TMA-safe stride order."""
        if tensor.ndim < 2:
            return tensor
        perm = dim_perm if dim_perm and len(dim_perm) == tensor.ndim else tuple(reversed(range(tensor.ndim)))
        return tensor.permute(perm)

    def _to_static_cute_tensor(self, tensor: torch.Tensor):
        """Convert a tensor to a CuTe tensor with dynamic layout metadata.

        TMA creation only needs a stable rank, stride order, and innermost
        contiguous mode. Marking the layout dynamic keeps batch/shape extents
        out of the compiled memref type so kernels can be reused across
        reallocated tensors and dynamic batch sizes with the same rank and
        stride pattern.
        """
        from cutlass.cute.runtime import from_dlpack

        cute_t = from_dlpack(tensor, assumed_align=16)
        one_stride_dims = [i for i, s in enumerate(tensor.stride()) if s == 1]
        if len(one_stride_dims) == 1:
            leading_dim = one_stride_dims[0]
        else:
            live_dims = [i for i in one_stride_dims if tensor.shape[i] > 1]
            if len(live_dims) == 1:
                leading_dim = live_dims[0]
            elif tensor.ndim > 0:
                leading_dim = min(range(tensor.ndim), key=lambda i: tensor.stride(i))
            else:
                leading_dim = 0
        return cute_t.mark_layout_dynamic(leading_dim=leading_dim)

    def _resolve_local_tma_tensor(self, desc) -> torch.Tensor:
        """Resolve the tensor backing a local TMA descriptor."""
        if desc.original_tensor is not None:
            tensor = desc.original_tensor.detach()
        else:
            tensor = self._registry_tensor_for_canonical(desc.tensor_canonical).detach()
            if desc.tensor_shape and tuple(tensor.shape) != desc.tensor_shape:
                tensor = tensor.reshape(desc.tensor_shape)
        return self._reshape_tensor_for_tma(tensor, len(desc.tile_shape))

    def _prepare_tma_tensors(self) -> None:
        """Prepare CuTe tensors with static layout for TMA descriptor creation.

        TMA requires tensors with static layout (not mark_layout_dynamic).
        We create these separately from regular tensor params using explicit
        from_dlpack with assumed_align=16.

        These tensors are passed to the kernel's __call__ method where
        make_tiled_tma_atom creates the TMA descriptors.
        """
        if self._tma_cute_tensors is not None:
            return
        if not self._tma_registry.has_tma:
            self._tma_cute_tensors = []
            return

        self._tma_cute_tensors = []
        # Collect unique tensors needed for TMA, deduped by (canonical_name, ndim).
        # Different ops may reference the same underlying storage with different
        # TMA dimensionalities (e.g., GEMM 2D store vs GDN 3D load on the same
        # data_ptr). Each unique ndim needs its own CuTe tensor with the correct
        # reshape so make_tiled_tma_atom can compose the tile.
        seen = set()
        for desc in self._tma_registry.descriptors:
            ndim = len(desc.tile_shape)
            key = (desc.tensor_canonical, ndim)
            if key in seen:
                continue
            seen.add(key)

            tensor = self._resolve_local_tma_tensor(desc)
            tensor = self._permute_tensor_for_tma(tensor, desc.dim_perm)
            cute_t = self._to_static_cute_tensor(tensor)
            self._tma_cute_tensors.append(
                (f"{desc.tensor_canonical}_{ndim}d", cute_t))

    def _prepare_peer_tma_tensors(self) -> None:
        """Prepare CuTe tensors with static layout for peer TMA descriptor creation.

        Same pattern as _prepare_tma_tensors but uses peer GPU buffers.
        Each peer buffer is permuted and converted via from_dlpack for TMA.
        """
        if self._peer_tma_cute_tensors is not None:
            return
        if not self._peer_tma_registry.has_peer_tma:
            self._peer_tma_cute_tensors = []
            return

        self._peer_tma_cute_tensors = []
        seen = set()
        for desc in self._peer_tma_registry.descriptors:
            key = (desc.tensor_canonical, desc.peer_idx)
            if key in seen:
                continue
            seen.add(key)

            # Get peer tensor from registry
            peer_tensors = self._peer_buffer_registry.get_peer_tensors(desc.tensor_canonical)
            if peer_tensors is None:
                continue

            tensor = peer_tensors[desc.peer_idx].detach()
            tensor = self._permute_tensor_for_tma(tensor)
            cute_t = self._to_static_cute_tensor(tensor)
            self._peer_tma_cute_tensors.append((desc.tensor_canonical, desc.peer_idx, cute_t))

    def _validate_requirements(self) -> None:
        """Validate GPU requirements."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        if self.config.threads_per_block % 32 != 0:
            raise RuntimeError(
                "MegakernelConfig.threads_per_block must be a multiple of 32 "
                f"(got {self.config.threads_per_block})."
            )

        compute_only = self._use_compute_only_replay()
        num_dma_warps = 0 if compute_only else NUM_DMA_WARPS
        num_mma_warps = self.config.threads_per_block // 32 - num_dma_warps
        if not compute_only:
            if num_mma_warps < 1:
                raise RuntimeError(
                    "TMA megakernel replay requires at least one compute warp "
                    f"in addition to the {NUM_DMA_WARPS} DMA warps. "
                    f"Increase MegakernelConfig.threads_per_block to at least "
                    f"{(NUM_DMA_WARPS + 1) * 32}; got {self.config.threads_per_block}."
                )
        else:
            reg_budget = num_mma_warps * 32 * self.config.mma_reg_count
            max_compute_only_regs = 58 * 1024
            if reg_budget > max_compute_only_regs:
                raise RuntimeError(
                    "Compute-only megakernel register budget is too high and can deadlock "
                    "during setmaxnreg/replay. Lower MegakernelConfig.mma_reg_count or "
                    "threads_per_block. "
                    f"Requested {reg_budget} registers "
                    f"({num_mma_warps} warps * 32 lanes * {self.config.mma_reg_count}); "
                    f"limit is {max_compute_only_regs}."
                )

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

    def write_trace(self, filename: str) -> None:
        """Write trace to .nanotrace file. Only valid after run() with tracing=True."""
        from .tracing import write_trace

        write_trace(self._tracing_state, filename)

    def write_trace_perfetto(self, filename: str) -> None:
        """Write trace as Perfetto JSON. Only valid after run() with tracing=True."""
        from .tracing import write_trace_perfetto

        write_trace_perfetto(self._tracing_state, filename)

    def _build_pipelined_dispatch_fns(self):
        """Build dispatch functions for pipelined execution phases.

        Each phase function receives an op-specific subset of tensor params
        (e.g., t0, t1, t2 for RMSNorm) and TMA params (tma0_atom, tma0_gmem).
        The dispatch functions accept ALL canonical names and route the correct
        subset to each phase fn.

        Returns:
            (dispatch_load, dispatch_compute, dispatch_store)
        """
        cache_key = ("dispatch_inputs", self._make_cache_key())
        dispatch_inputs = Megakernel._dispatch_inputs_cache.get(cache_key)
        if dispatch_inputs is None:
            dispatch_inputs = self._backend.compile_phase_dispatch_inputs(self)
            Megakernel._dispatch_inputs_cache[cache_key] = dispatch_inputs
        return (
            dispatch_inputs["dispatch_load"],
            dispatch_inputs["dispatch_compute"],
            dispatch_inputs["dispatch_store"],
            dispatch_inputs["dispatch_communicate"],
            dispatch_inputs["phase_uses_handler_local_idx"],
            dispatch_inputs["phase_uses_runtime_transport_selector"],
            dispatch_inputs["phase_uses_desc_slot_selector"],
            dispatch_inputs["has_communicate"],
            dispatch_inputs["per_op_warps"],
            dispatch_inputs["phase_tensor_names"],
            dispatch_inputs["phase_tma_names"],
            dispatch_inputs["all_tma_canonical"],
        )

    @staticmethod
    def _signature_suffix(param_names: List[str]) -> str:
        """Return a comma-prefixed signature fragment for optional params."""
        return f", {', '.join(param_names)}" if param_names else ""

    def _build_kernel_loop_source(
        self,
        kernel_loop_fn,
        *,
        tensor_sig: str,
        tma_sig: str,
        peer_tma_sig: str,
        has_communicate: bool,
        tracing: bool,
        phase_uses_handler_local_idx: Dict[str, bool],
        dispatch_extra_params: Dict[str, str],
    ) -> str:
        """Render the generated `_kernel_loop` source."""
        from .compile import _extract_body

        body = _extract_body(kernel_loop_fn)
        if dispatch_extra_params:
            def _rewrite_dispatch(match):
                fn_name = match.group(1)
                call_args = match.group(2).rstrip().rstrip(",")
                phase_name = fn_name.removeprefix("dispatch_")
                extra_params = dispatch_extra_params.get(phase_name, "")
                if not extra_params:
                    return f"{fn_name}({call_args})"
                return f"{fn_name}({call_args}, {extra_params})"

            body = re.sub(
                r"(dispatch_(?:load|compute|store|communicate))\(([^)]*)\)",
                _rewrite_dispatch,
                body,
            )

        peer_signal_sig = ", peer_signal_ptr" if has_communicate else ""
        peer_signal_init = "" if has_communicate else "    peer_signal_ptr = Int64(0)\n"
        static_sig = ""
        trace_sig = ", trace_buffer_ptr" if tracing else ""
        trace_init = "" if tracing else "    trace_buffer_ptr = Int64(0)\n"
        local_sig_parts = []
        selector_sig_parts = []
        desc_slot_sig_parts = []
        local_init = ""
        selector_init = ""
        desc_slot_init = ""
        for phase in PHASE_NAMES:
            ptr_name = f"{phase}_local_idx_ptr"
            if self._phase_local_idx_tensors[phase] is not None:
                local_sig_parts.append(ptr_name)
            else:
                local_init += f"    {ptr_name} = Int64(0)\n"
            selector_ptr_name = f"{phase}_local_transport_positions_ptr"
            if self._phase_local_transport_position_tensors[phase] is not None:
                selector_sig_parts.append(selector_ptr_name)
            else:
                selector_init += f"    {selector_ptr_name} = Int64(0)\n"
            desc_slot_ptr_name = f"{phase}_local_desc_slots_ptr"
            if self._phase_local_desc_slot_tensors[phase] is not None:
                desc_slot_sig_parts.append(desc_slot_ptr_name)
            else:
                desc_slot_init += f"    {desc_slot_ptr_name} = Int64(0)\n"
        local_sig = f", {', '.join(local_sig_parts)}" if local_sig_parts else ""
        selector_sig = f", {', '.join(selector_sig_parts)}" if selector_sig_parts else ""
        desc_slot_sig = f", {', '.join(desc_slot_sig_parts)}" if desc_slot_sig_parts else ""
        return (
            "@cute.jit\n"
            "def _kernel_loop(instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            f"                  op_meta_ptr{local_sig}{selector_sig}{desc_slot_sig}, signal_meta_ptr{peer_signal_sig}{static_sig},\n"
            "                  num_instructions, tidx, block_id, num_blocks,\n"
            f"                  smem_base{trace_sig}, wait_info_ptr, compute_wait_info_ptr{tensor_sig}{tma_sig}{peer_tma_sig}):\n"
            f"{local_init}"
            f"{selector_init}"
            f"{desc_slot_init}"
            f"{peer_signal_init}"
            f"{trace_init}"
            + textwrap.indent(body, "    ")
            + "\n"
        )

    def _build_kernel_exec_globals(
        self,
        *,
        dispatch_load,
        dispatch_compute,
        dispatch_store,
        signal_barriers,
        get_page_ptr_fn,
        num_pages: int,
        num_slots: int,
        iq_offset: int,
        flags_offset: int,
        ring_state_offset: int,
        extra_exec_globals=None,
    ) -> Dict[str, Any]:
        """Build exec globals for the generated `_kernel_loop`."""
        exec_globals = {
            "cute": cute,
            "Int32": Int32,
            "Int64": Int64,
            "range_constexpr": __import__("cutlass").range_constexpr,
            "const_expr": __import__("cutlass").const_expr,
            "tracing": bool(self.config.tracing),
            "TileInstruction": TileInstruction,
            "dispatch_load": dispatch_load,
            "dispatch_compute": dispatch_compute,
            "dispatch_store": dispatch_store,
            "signal_barriers": signal_barriers,
            "_get_page_ptr": get_page_ptr_fn,
            "ld_shared_i32": ld_shared_i32,
            "st_shared_i32": st_shared_i32,
            "st_shared_release_cta_i32": st_shared_release_cta_i32,
            "ld_shared_acquire_cta_i32": ld_shared_acquire_cta_i32,
            "load_instruction_to_smem": load_instruction_to_smem,
            "prefetch_instruction": prefetch_instruction,
            "ld_global_i64": ld_global_i64,
            "mbarrier_init": mbarrier_init,
            "mbarrier_init_fence": mbarrier_init_fence,
            "mbarrier_arrive": mbarrier_arrive,
            "mbarrier_wait": mbarrier_wait,
            "nanosleep": nanosleep,
            "named_barrier_sync": named_barrier_sync,
            "global_barrier_signal": global_barrier_signal,
            "global_barrier_signal_gpu": global_barrier_signal_gpu,
            "num_pages": num_pages,
            "num_slots": num_slots,
            "iq_offset": iq_offset,
            "flags_offset": flags_offset,
            "ring_state_offset": ring_state_offset,
        }
        if extra_exec_globals:
            exec_globals.update(extra_exec_globals)
        return exec_globals

    def _collect_tma_tensor_names(self, tma_registry) -> List[str]:
        """Return unique parameter names for local static-layout TMA tensors."""
        tensor_names: List[str] = []
        seen = set()
        for desc in tma_registry.descriptors:
            ndim = len(desc.tile_shape)
            tensor_name = f"tma_{desc.tensor_canonical}_{ndim}d"
            if tensor_name in seen:
                continue
            seen.add(tensor_name)
            tensor_names.append(tensor_name)
        return tensor_names

    def _collect_peer_tma_tensor_names(self, peer_tma_registry) -> List[str]:
        """Return unique parameter names for peer static-layout TMA tensors."""
        tensor_names: List[str] = []
        seen = set()
        for desc in peer_tma_registry.descriptors:
            tensor_name = f"ptma_{desc.tensor_canonical}_p{desc.peer_idx}"
            if tensor_name in seen:
                continue
            seen.add(tensor_name)
            tensor_names.append(tensor_name)
        return tensor_names

    @staticmethod
    def _render_tma_creation_expr(desc, tensor_source: str) -> str:
        """Render the `make_tiled_tma_atom` expression for one descriptor."""
        direction = getattr(desc, "direction", "s2g")
        if direction == "g2s":
            copy_op = "CopyBulkTensorTileG2SOp()"
        elif direction == "s2g":
            copy_op = "CopyBulkTensorTileS2GOp()"
        elif direction == "s2g_reduce":
            copy_op = "CopyReduceBulkTensorTileS2GOp(reduction_kind=ReductionOp.ADD)"
        else:
            raise ValueError(f"Unknown TMA direction: {direction}")

        shape_str = ", ".join(str(s) for s in desc.tile_shape)
        smem_layout_code = desc.smem_layout_src or f"cute.make_layout(({shape_str},))"
        return (
            "cute.nvgpu.cpasync.make_tiled_tma_atom(\n"
            f"        {copy_op},\n"
            f"        {tensor_source},\n"
            f"        {smem_layout_code},\n"
            f"        ({shape_str},),\n"
            "        num_multicast=1,\n"
            "    )"
        )

    def _append_tma_descriptor_code(
        self,
        tma_creation_lines,
        desc_pool_init_specs,
        desc,
        tensor_source: str,
        helper_name: Optional[str] = None,
        *,
        create_atom_binding: bool,
        create_gmem_binding: bool,
        pool_name: str,
        pool_slot: int,
    ) -> None:
        """Append source that constructs one TMA descriptor pair."""
        concrete_atom_name = desc.canonical_atom if create_atom_binding else f"_{desc.canonical_desc}_atom"
        concrete_gmem_name = desc.canonical_gmem if create_gmem_binding else f"_{desc.canonical_desc}_gmem"
        concrete_desc_name = desc.canonical_desc
        if helper_name is not None:
            tma_creation_lines.append(
                f"        {concrete_atom_name}, {concrete_gmem_name}, {concrete_desc_name} = "
                f"{helper_name}({tensor_source})"
            )
        else:
            tma_creation_lines.append(
                f"        {concrete_atom_name}, {concrete_gmem_name} = "
                f"{self._render_tma_creation_expr(desc, tensor_source)}"
            )
            tma_creation_lines.append(f"        {concrete_desc_name} = {concrete_atom_name}._trait.desc_ptr")
        desc_pool_init_specs.append((concrete_desc_name, pool_name, pool_slot))
        if create_atom_binding and concrete_atom_name != desc.canonical_atom:
            tma_creation_lines.append(f"        {desc.canonical_atom} = {concrete_atom_name}")
        if create_gmem_binding and concrete_gmem_name != desc.canonical_gmem:
            tma_creation_lines.append(f"        {desc.canonical_gmem} = {concrete_gmem_name}")

    def _build_tma_kernel_components(self, tma_registry, peer_tma_registry) -> Dict[str, Any]:
        """Assemble the TMA-specific signature fragments and descriptor setup code."""
        tma_tensor_names = self._collect_tma_tensor_names(tma_registry)
        peer_tma_tensor_names = self._collect_peer_tma_tensor_names(peer_tma_registry)
        tma_creation_lines: List[str] = []
        desc_pool_init_specs: List[Tuple[str, str, int]] = []
        helper_sources: List[str] = []
        helper_name_by_key: Dict[Tuple[Any, ...], str] = {}
        seen_atoms = set()
        seen_gmems = set()

        def _helper_name_for_desc(desc) -> str:
            key = (
                getattr(desc, "direction", "s2g"),
                tuple(desc.tile_shape),
                desc.smem_layout_src or "",
                tuple(getattr(desc, "dim_perm", ()) or ()),
            )
            helper_name = helper_name_by_key.get(key)
            if helper_name is not None:
                return helper_name
            helper_name = f"_make_tma_helper_{len(helper_name_by_key)}"
            helper_name_by_key[key] = helper_name
            direction = getattr(desc, "direction", "s2g")
            if direction == "g2s":
                copy_op = "CopyBulkTensorTileG2SOp()"
            elif direction == "s2g":
                copy_op = "CopyBulkTensorTileS2GOp()"
            elif direction == "s2g_reduce":
                copy_op = "CopyReduceBulkTensorTileS2GOp(reduction_kind=ReductionOp.ADD)"
            else:
                raise ValueError(f"Unknown TMA direction: {direction}")
            shape_str = ", ".join(str(s) for s in desc.tile_shape)
            smem_layout_code = desc.smem_layout_src or f"cute.make_layout(({shape_str},))"
            helper_sources.append(
                "@cute.jit\n"
                f"def {helper_name}(tensor):\n"
                "    atom, gmem = make_runtime_desc_tma_atom(\n"
                f"        {copy_op},\n"
                "        tensor,\n"
                f"        {smem_layout_code},\n"
                f"        ({shape_str},),\n"
                "        num_multicast=1,\n"
                "    )\n"
                "    return atom, gmem, atom._trait.desc_ptr\n"
            )
            return helper_name

        for slot, desc in enumerate(tma_registry.descriptors):
            ndim = len(desc.tile_shape)
            self._append_tma_descriptor_code(
                tma_creation_lines,
                desc_pool_init_specs,
                desc,
                f"tma_{desc.tensor_canonical}_{ndim}d",
                helper_name=_helper_name_for_desc(desc),
                create_atom_binding=desc.canonical_atom not in seen_atoms,
                create_gmem_binding=desc.canonical_gmem not in seen_gmems,
                pool_name="local_tma_desc_pool_ptr",
                pool_slot=slot,
            )
            seen_atoms.add(desc.canonical_atom)
            seen_gmems.add(desc.canonical_gmem)

        for slot, desc in enumerate(peer_tma_registry.descriptors):
            self._append_tma_descriptor_code(
                tma_creation_lines,
                desc_pool_init_specs,
                desc,
                f"ptma_{desc.tensor_canonical}_p{desc.peer_idx}",
                helper_name=_helper_name_for_desc(desc),
                create_atom_binding=desc.canonical_atom not in seen_atoms,
                create_gmem_binding=desc.canonical_gmem not in seen_gmems,
                pool_name="peer_tma_desc_pool_ptr",
                pool_slot=slot,
            )
            seen_atoms.add(desc.canonical_atom)
            seen_gmems.add(desc.canonical_gmem)

        tma_creation_code = "\n".join(tma_creation_lines)
        if tma_creation_code:
            tma_creation_code = "\n" + tma_creation_code + "\n"

        init_param_names: List[str] = []
        for desc_name, _pool_name, _slot in desc_pool_init_specs:
            if desc_name not in init_param_names:
                init_param_names.append(desc_name)
        desc_pool_init_params = ", ".join(init_param_names)
        desc_pool_init_sig = self._signature_suffix(init_param_names)
        desc_pool_init_body_lines = []
        for desc_name, pool_name, slot in desc_pool_init_specs:
            desc_pool_init_body_lines.append(
                f"        copy_runtime_desc_to_pool({desc_name}, {pool_name}, Int32({slot}))"
            )
        if desc_pool_init_body_lines:
            desc_pool_init_body_lines.append("        fence_runtime_desc_pool()")
        desc_pool_init_body = "\n".join(desc_pool_init_body_lines)

        return {
            "desc_pool_sig": ", local_tma_desc_pool_ptr, peer_tma_desc_pool_ptr",
            "tma_tensor_sig": self._signature_suffix(tma_tensor_names),
            "peer_tma_tensor_input_sig": self._signature_suffix(peer_tma_tensor_names),
            "helper_definitions_code": "\n".join(helper_sources) + ("\n" if helper_sources else ""),
            "tma_creation_code": tma_creation_code,
            "desc_pool_init_sig": desc_pool_init_sig,
            "desc_pool_init_params": desc_pool_init_params,
            "desc_pool_init_body": desc_pool_init_body,
        }

    def _build_persistent_kernel_source(
        self,
        *,
        num_sms: int,
        threads_per_block: int,
        smem_size: int,
        tensor_sig: str,
        kernel_tma_sig: str,
        tma_components: Dict[str, Any],
        has_communicate: bool,
        tracing: bool,
        phase_uses_handler_local_idx: Dict[str, bool],
    ) -> str:
        """Render the `PersistentKernel` class source."""
        peer_signal_sig = ", peer_signal_ptr" if has_communicate else ""
        peer_signal_arg = ", peer_signal_ptr" if has_communicate else ""
        peer_signal_init = "" if has_communicate else "        peer_signal_ptr = Int64(0)\n"
        static_sig = ""
        static_arg = ""
        trace_sig = ", trace_buffer_ptr" if tracing else ""
        trace_arg = ", trace_buffer_ptr" if tracing else ""
        trace_init = "" if tracing else "        trace_buffer_ptr = Int64(0)\n"
        local_sig_parts = []
        local_arg_parts = []
        selector_sig_parts = []
        selector_arg_parts = []
        desc_slot_sig_parts = []
        desc_slot_arg_parts = []
        local_init = ""
        selector_init = ""
        desc_slot_init = ""
        for phase in PHASE_NAMES:
            ptr_name = f"{phase}_local_idx_ptr"
            if self._phase_local_idx_tensors[phase] is not None:
                local_sig_parts.append(ptr_name)
                local_arg_parts.append(ptr_name)
            else:
                local_init += f"        {ptr_name} = Int64(0)\n"
            selector_ptr_name = f"{phase}_local_transport_positions_ptr"
            if self._phase_local_transport_position_tensors[phase] is not None:
                selector_sig_parts.append(selector_ptr_name)
                selector_arg_parts.append(selector_ptr_name)
            else:
                selector_init += f"        {selector_ptr_name} = Int64(0)\n"
            desc_slot_ptr_name = f"{phase}_local_desc_slots_ptr"
            if self._phase_local_desc_slot_tensors[phase] is not None:
                desc_slot_sig_parts.append(desc_slot_ptr_name)
                desc_slot_arg_parts.append(desc_slot_ptr_name)
            else:
                desc_slot_init += f"        {desc_slot_ptr_name} = Int64(0)\n"
        local_sig = f", {', '.join(local_sig_parts)}" if local_sig_parts else ""
        local_arg = f", {', '.join(local_arg_parts)}" if local_arg_parts else ""
        selector_sig = f", {', '.join(selector_sig_parts)}" if selector_sig_parts else ""
        selector_arg = f", {', '.join(selector_arg_parts)}" if selector_arg_parts else ""
        desc_slot_sig = f", {', '.join(desc_slot_sig_parts)}" if desc_slot_sig_parts else ""
        desc_slot_arg = f", {', '.join(desc_slot_arg_parts)}" if desc_slot_arg_parts else ""
        return (
            f"{tma_components['helper_definitions_code']}"
            "class PersistentKernel:\n"
            "    def __init__(self):\n"
            f"        self.num_sms = {num_sms}\n"
            f"        self.threads_per_block = {threads_per_block}\n"
            f"        self.smem_size = {smem_size}\n"
            "\n"
            "    @cute.kernel\n"
            f"    def init_tma_desc_pool(self, local_tma_desc_pool_ptr, peer_tma_desc_pool_ptr"
            f"{tma_components['desc_pool_init_sig']}):\n"
            f"{tma_components['desc_pool_init_body'] or '        return\n'}"
            "\n"
            "    @cute.jit\n"
            "    def __call__(self, instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            f"                 op_meta_ptr{local_sig}{selector_sig}{desc_slot_sig}, signal_meta_ptr{peer_signal_sig}{static_sig},\n"
            f"                 wait_info_ptr, compute_wait_info_ptr, num_instructions{trace_sig}"
            f"{tensor_sig}{tma_components['desc_pool_sig']}{tma_components['tma_tensor_sig']}{tma_components['peer_tma_tensor_input_sig']}, desc_pool_init_needed, stream):\n"
            f"{local_init}"
            f"{selector_init}"
            f"{desc_slot_init}"
            f"{peer_signal_init}"
            f"{trace_init}"
            f"{tma_components['tma_creation_code']}"
            "        if desc_pool_init_needed:\n"
            "            self.init_tma_desc_pool(\n"
            "                local_tma_desc_pool_ptr,\n"
            "                peer_tma_desc_pool_ptr"
            f"{', ' if tma_components['desc_pool_init_params'] else ''}{tma_components['desc_pool_init_params']}\n"
            "            ).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)\n"
            "        if not desc_pool_init_needed:\n"
            "            self.kernel(instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            f"                        op_meta_ptr{local_arg}{selector_arg}{desc_slot_arg}, signal_meta_ptr{peer_signal_arg}{static_arg},\n"
            f"                        wait_info_ptr, compute_wait_info_ptr,\n"
            f"                        num_instructions{trace_arg}{tensor_sig}{kernel_tma_sig}).launch(\n"
            "                grid=[self.num_sms, 1, 1],\n"
            "                block=[self.threads_per_block, 1, 1],\n"
            "                smem=self.smem_size,\n"
            "                stream=stream,\n"
            "                min_blocks_per_mp=1,\n"
            "            )\n"
            "\n"
            "    @cute.kernel\n"
            "    def kernel(self, instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            f"               op_meta_ptr{local_sig}{selector_sig}{desc_slot_sig}, signal_meta_ptr{peer_signal_sig}{static_sig},\n"
            f"               wait_info_ptr, compute_wait_info_ptr, num_instructions{trace_sig}{tensor_sig}{kernel_tma_sig}):\n"
            f"{local_init}"
            f"{selector_init}"
            f"{desc_slot_init}"
            "        tidx = cute.arch.thread_idx()[0]\n"
            "        block_id = cute.arch.block_idx()[0]\n"
            "        num_blocks = cute.arch.grid_dim()[0]\n"
            "        smem_base = get_smem_base_ptr()\n"
            "        _kernel_loop(instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            f"                     op_meta_ptr{local_arg}{selector_arg}{desc_slot_arg}, signal_meta_ptr{peer_signal_arg}{static_arg},\n"
            "                     num_instructions, tidx, block_id, num_blocks,\n"
             f"                     smem_base{trace_arg}, wait_info_ptr, compute_wait_info_ptr{tensor_sig}{kernel_tma_sig})\n"
        )

    def _build_persistent_kernel_globals(self, tma_registry, peer_tma_registry, kernel_loop) -> Dict[str, Any]:
        """Build exec globals for the generated `PersistentKernel` class."""
        cutlass = __import__("cutlass")
        pk_globals = {
            "cute": cute,
            "cutlass": cutlass,
            "Int32": Int32,
            "Int64": Int64,
            "range_constexpr": cutlass.range_constexpr,
            "const_expr": cutlass.const_expr,
            "get_smem_base_ptr": get_smem_base_ptr,
            "_kernel_loop": kernel_loop,
            "_sync_tma_desc_init_stream": _sync_tma_desc_init_stream,
        }
        from .transport import copy_runtime_desc_to_pool, fence_runtime_desc_pool, make_runtime_desc_tma_atom
        pk_globals["copy_runtime_desc_to_pool"] = copy_runtime_desc_to_pool
        pk_globals["fence_runtime_desc_pool"] = fence_runtime_desc_pool
        pk_globals["make_runtime_desc_tma_atom"] = make_runtime_desc_tma_atom

        if tma_registry.has_tma:
            from cutlass.cute.nvgpu.cpasync import (
                CopyBulkTensorTileG2SOp,
                CopyBulkTensorTileS2GOp,
            )
            pk_globals["CopyBulkTensorTileG2SOp"] = CopyBulkTensorTileG2SOp
            pk_globals["CopyBulkTensorTileS2GOp"] = CopyBulkTensorTileS2GOp

            if self._registry_uses_reduce_store(tma_registry):
                from cutlass.cute.nvgpu.cpasync import CopyReduceBulkTensorTileS2GOp
                from cutlass.cute.tensor import ReductionOp

                pk_globals["CopyReduceBulkTensorTileS2GOp"] = CopyReduceBulkTensorTileS2GOp
                pk_globals["ReductionOp"] = ReductionOp

        if peer_tma_registry.has_peer_tma and not tma_registry.has_tma:
            from cutlass.cute.nvgpu.cpasync import CopyBulkTensorTileS2GOp

            pk_globals["CopyBulkTensorTileS2GOp"] = CopyBulkTensorTileS2GOp

        if peer_tma_registry.has_peer_tma and self._registry_uses_reduce_store(peer_tma_registry):
            from cutlass.cute.nvgpu.cpasync import CopyReduceBulkTensorTileS2GOp
            from cutlass.cute.tensor import ReductionOp

            pk_globals["CopyReduceBulkTensorTileS2GOp"] = CopyReduceBulkTensorTileS2GOp
            pk_globals["ReductionOp"] = ReductionOp

        return pk_globals

    @staticmethod
    def _registry_uses_reduce_store(registry) -> bool:
        """Return whether any descriptor in the registry performs a reduce-store."""
        return any(getattr(desc, "direction", "s2g") == "s2g_reduce" for desc in registry.descriptors)

    @staticmethod
    def _op_has_peer_barriers(op: ScheduledOp) -> bool:
        """Return whether an op emits cross-GPU barrier signals."""
        return bool(
            getattr(op.op_cls, "_PEER_STORES", set())
            | getattr(op.op_cls, "_PEER_REDUCE_STORES", set())
        )

    @staticmethod
    def _op_has_phased_replay(op: ScheduledOp) -> bool:
        """Return whether an op needs the load/store replay shell."""
        op_cls = op.op_cls
        return (
            getattr(op_cls, "load_phase", None) is not None
            or getattr(op_cls, "store_phase", None) is not None
            or getattr(op_cls, "communicate_phase", None) is not None
            or getattr(op_cls, "load") is not Op.load
            or getattr(op_cls, "store") is not Op.store
            or getattr(op_cls, "communicate", None) is not getattr(Op, "communicate", None)
        )

    def _use_compute_only_replay(self) -> bool:
        """Use the lighter replay loop when no TMA or peer phases exist."""
        return (
            not self._tma_registry.has_tma
            and not self._peer_tma_registry.has_peer_tma
            and not any(self._op_has_phased_replay(op) for op in self.ops)
        )

    def _sync_compute_warps_after_tile(self) -> bool:
        """Return whether any op requires a CTA sync after compute."""
        return any(
            bool(getattr(op.op_cls, "sync_compute_warps_after_tile", False))
            for op in self.ops
        )

    @staticmethod
    def _resolve_warp_register_api():
        """Resolve the CuTe warp register allocation API, with a no-op fallback."""
        try:
            from cutlass.cute.arch import (
                setmaxregister_decrease,
                setmaxregister_increase,
            )

            return setmaxregister_increase, setmaxregister_decrease
        except ImportError:
            try:
                from cutlass.cute.arch import (
                    warpgroup_reg_alloc as setmaxregister_increase,
                    warpgroup_reg_dealloc as setmaxregister_decrease,
                )

                return setmaxregister_increase, setmaxregister_decrease
            except ImportError:

                def _setmaxregister_noop(_n):
                    """Fallback when the local CuTe build exposes no register API."""

                return _setmaxregister_noop, _setmaxregister_noop

    def _kernel_static_config(self) -> Dict[str, Any]:
        """Collect the compile-time constants used to build the persistent kernel."""
        layout = self._layout
        threads_per_block = self.config.threads_per_block
        num_dma_warps = 0 if self._use_compute_only_replay() else NUM_DMA_WARPS
        num_mma_warps = (threads_per_block // 32) - num_dma_warps

        return {
            "num_sms": self.config.num_sms,
            "threads_per_block": threads_per_block,
            "smem_size": layout.total_size,
            "tracing": self.config.tracing,
            "num_pages": layout.num_pages,
            "num_slots": layout.num_slots,
            "has_page_free_ops": self._use_physical_page_ring and not self._use_compute_only_replay(),
            "iq_offset": layout.iq_offset,
            "flags_offset": layout.flags_offset,
            "ring_state_offset": layout.ring_state_offset,
            "pages_start": layout.pages_start,
            "aligned_page_size": layout.aligned_page_size,
            "work_notify_mbar_offset_0": layout.work_notify_mbar_offset(0),
            "compute_done_mbar_offset_0": layout.compute_done_mbar_offset(0),
            "num_mma_warps": num_mma_warps,
            "num_compute_threads": num_mma_warps * 32,
            "num_dma_warps": num_dma_warps,
            "dma_reg_count": self.config.dma_reg_count,
            "mma_reg_count": self.config.mma_reg_count,
            "actual_threads_per_block": threads_per_block,
            "mbarrier_stride": NPageLayout._MBARRIER_SIZE,
            "tile_info_bytes": NPageLayout._TILE_INFO_SIZE,
            "peer_barriers_data_ptr": (
                self.config.peer_barriers.data_ptr()
                if self.config.peer_barriers is not None
                else 0
            ),
        }

    def _kernel_runtime_components(self, kernel_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Build reusable runtime helpers needed by `_create_kernel`."""
        setmaxregister_increase, setmaxregister_decrease = self._resolve_warp_register_api()
        (
            dispatch_load,
            dispatch_compute,
            dispatch_store,
            dispatch_communicate,
            phase_uses_handler_local_idx,
            phase_uses_runtime_transport_selector,
            phase_uses_desc_slot_selector,
            has_communicate,
            per_op_warps,
            phase_tensor_names,
            phase_tma_names,
            all_tma_canonical,
        ) = self._build_pipelined_dispatch_fns()

        if self.config.tracing:
            from .tracing import get_trace_exec_globals
            trace_exec_globals = get_trace_exec_globals(self._tracing_state)
        else:
            trace_exec_globals = {}

        @cute.jit
        def _get_page_ptr(smem_base: Int32, page_idx: Int32) -> Int32:
            """Return the shared-memory base pointer for a page slot."""
            return (
                smem_base
                + Int32(kernel_cfg["pages_start"])
                + page_idx * Int32(kernel_cfg["aligned_page_size"])
            )

        @cute.jit
        def _work_notify_mbar(smem_base: Int32, slot: Int32) -> Int32:
            """Return the work-available mbarrier address for one slot."""
            return (
                smem_base
                + Int32(kernel_cfg["work_notify_mbar_offset_0"])
                + slot * Int32(kernel_cfg["mbarrier_stride"])
            )

        @cute.jit
        def _compute_done_mbar(smem_base: Int32, page_idx: Int32) -> Int32:
            """Return the compute-complete mbarrier address for one page."""
            return (
                smem_base
                + Int32(kernel_cfg["compute_done_mbar_offset_0"])
                + page_idx * Int32(kernel_cfg["mbarrier_stride"])
            )

        @cute.jit
        def _op_meta_i32(op_meta_ptr: Int64, op_idx: Int32, field: Int32) -> Int32:
            return ld_global_i32(
                op_meta_ptr, op_idx * Int32(_OP_META_STRIDE) + field
            )

        @cute.jit
        def _op_meta_base(op_idx: Int32) -> Int32:
            return op_idx * Int32(_OP_META_STRIDE)

        @cute.jit
        def _op_meta_i32_base(op_meta_ptr: Int64, op_meta_base: Int32, field: Int32) -> Int32:
            return ld_global_i32(op_meta_ptr, op_meta_base + field)

        @cute.jit
        def _decompose_tile(op_meta_ptr: Int64, op_meta_base: Int32, linear_idx: Int32):
            rem = linear_idx
            t0 = Int32(0)
            t1 = Int32(0)
            t2 = Int32(0)
            t3 = Int32(0)
            c0 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_0))
            c1 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_1))
            c2 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_2))
            c3 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_3))
            s0 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_STRIDE_0))
            s1 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_STRIDE_1))
            s2 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_STRIDE_2))
            s3 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_STRIDE_3))
            if c0 > Int32(1):
                if s0 > Int32(1):
                    t0 = rem // s0
                    rem = rem % s0
                else:
                    t0 = rem
            if c1 > Int32(1):
                if s1 > Int32(1):
                    t1 = rem // s1
                    rem = rem % s1
                else:
                    t1 = rem
            if c2 > Int32(1):
                if s2 > Int32(1):
                    t2 = rem // s2
                    rem = rem % s2
                else:
                    t2 = rem
            if c3 > Int32(1):
                if s3 > Int32(1):
                    t3 = rem // s3
                    rem = rem % s3
                else:
                    t3 = rem
            return t0, t1, t2, t3

        @cute.jit
        def _advance_tile(op_meta_ptr: Int64, op_meta_base: Int32, t0: Int32, t1: Int32, t2: Int32, t3: Int32):
            carry = Int32(1)
            c3 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_3))
            if carry != Int32(0) and c3 > Int32(1):
                t3 = t3 + Int32(1)
                if t3 < c3:
                    carry = Int32(0)
                else:
                    t3 = Int32(0)
            c2 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_2))
            if carry != Int32(0) and c2 > Int32(1):
                t2 = t2 + Int32(1)
                if t2 < c2:
                    carry = Int32(0)
                else:
                    t2 = Int32(0)
            c1 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_1))
            if carry != Int32(0) and c1 > Int32(1):
                t1 = t1 + Int32(1)
                if t1 < c1:
                    carry = Int32(0)
                else:
                    t1 = Int32(0)
            c0 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_0))
            if carry != Int32(0) and c0 > Int32(1):
                t0 = t0 + Int32(1)
                if t0 >= c0:
                    t0 = Int32(0)
            return t0, t1, t2, t3


        @cute.jit
        def _signal_barriers_from_meta(
            signal_meta_ptr: Int64,
            instruction_idx: Int32,
            signal_count: Int32,
            barriers_ptr: Int64,
        ):
            done_signals = Int32(0)
            for sig_idx in range_constexpr(self._max_signal_formulas):
                if done_signals == Int32(0):
                    if sig_idx < signal_count:
                        barrier_idx = ld_global_i32(
                            signal_meta_ptr,
                            instruction_idx * Int32(self._max_signal_formulas) + Int32(sig_idx),
                        )
                        if barrier_idx >= Int32(0):
                            global_barrier_signal_gpu(barriers_ptr, barrier_idx)
                        else:
                            done_signals = Int32(1)
                    else:
                        done_signals = Int32(1)

        if has_communicate:
            @cute.jit
            def _signal_peer_barriers_from_meta(
                peer_signal_ptr: Int64,
                op_idx: Int32,
                linear_tile_idx: Int32,
                peer_barriers_ptr: Int64,
            ):
                barrier_offset = ld_global_i32(peer_signal_ptr, op_idx)
                if barrier_offset >= Int32(0):
                    global_barrier_signal(
                        peer_barriers_ptr, barrier_offset + linear_tile_idx
                    )
        else:
            _signal_peer_barriers_from_meta = None

        return {
            "setmaxregister_increase": setmaxregister_increase,
            "setmaxregister_decrease": setmaxregister_decrease,
            "dispatch_load": dispatch_load,
            "dispatch_compute": dispatch_compute,
            "dispatch_store": dispatch_store,
            "dispatch_communicate": dispatch_communicate,
            "phase_uses_handler_local_idx": phase_uses_handler_local_idx,
            "phase_uses_runtime_transport_selector": phase_uses_runtime_transport_selector,
            "phase_uses_desc_slot_selector": phase_uses_desc_slot_selector,
            "has_communicate": has_communicate,
            "needs_warp_transition": any(w < kernel_cfg["num_mma_warps"] for w in per_op_warps),
            "max_waits": self._max_wait_formulas,
            "max_compute_waits": self._max_compute_wait_formulas,
            "signal_barriers": _signal_barriers_from_meta,
            "signal_peer_barriers": _signal_peer_barriers_from_meta,
            "trace_exec_globals": trace_exec_globals,
            "decompose_tile": _decompose_tile,
            "_op_meta_i32": _op_meta_i32,
            "_op_meta_base": _op_meta_base,
            "_op_meta_i32_base": _op_meta_i32_base,
            "_get_page_ptr": _get_page_ptr,
            "_work_notify_mbar": _work_notify_mbar,
            "_compute_done_mbar": _compute_done_mbar,
            "phase_tensor_names": phase_tensor_names,
            "phase_tma_names": phase_tma_names,
            "all_tma_canonical": all_tma_canonical,
            "advance_tile": _advance_tile,
        }

    @staticmethod
    def _op_meta_exec_globals() -> Dict[str, int]:
        """Return per-op metadata indices injected into generated kernel code."""
        return {
            "_OP_META_NUM_WARPS": _OP_META_NUM_WARPS,
            "_OP_META_STRIDE_0": _OP_META_STRIDE_0,
            "_OP_META_STRIDE_1": _OP_META_STRIDE_1,
            "_OP_META_STRIDE_2": _OP_META_STRIDE_2,
            "_OP_META_STRIDE_3": _OP_META_STRIDE_3,
            "_OP_META_COUNT_0": _OP_META_COUNT_0,
            "_OP_META_COUNT_1": _OP_META_COUNT_1,
            "_OP_META_COUNT_2": _OP_META_COUNT_2,
            "_OP_META_COUNT_3": _OP_META_COUNT_3,
            "_OP_META_HANDLER_IDX": _OP_META_HANDLER_IDX,
            "_OP_META_LOAD_LOCAL_IDX": _OP_META_LOAD_LOCAL_IDX,
            "_OP_META_COMPUTE_LOCAL_IDX": _OP_META_COMPUTE_LOCAL_IDX,
            "_OP_META_STORE_LOCAL_IDX": _OP_META_STORE_LOCAL_IDX,
            "_OP_META_COMM_LOCAL_IDX": _OP_META_COMM_LOCAL_IDX,
            "_OP_META_WAIT_COUNT": _OP_META_WAIT_COUNT,
            "_OP_META_COMPUTE_WAIT_COUNT": _OP_META_COMPUTE_WAIT_COUNT,
            "_OP_META_SIGNAL_COUNT": _OP_META_SIGNAL_COUNT,
            "_OP_META_WAIT_ACQUIRE": _OP_META_WAIT_ACQUIRE,
            "_OP_META_PHASE_MASK": _OP_META_PHASE_MASK,
            "_INSTR_NO_SMEM_PAGE_BIT": _INSTR_NO_SMEM_PAGE_BIT,
        }

    @staticmethod
    def _tile_info_exec_globals() -> Dict[str, int]:
        """Return shared tile-info layout constants for generated kernel code."""
        return {
            "_TILE_INFO_HANDLER_IDX": _TILE_INFO_HANDLER_IDX,
            "_TILE_INFO_INSTRUCTION_IDX": _TILE_INFO_INSTRUCTION_IDX,
            "_TILE_INFO_TILE_0": _TILE_INFO_TILE_0,
            "_TILE_INFO_TILE_1": _TILE_INFO_TILE_1,
            "_TILE_INFO_TILE_2": _TILE_INFO_TILE_2,
            "_TILE_INFO_TILE_3": _TILE_INFO_TILE_3,
            "_TILE_INFO_OP_CONFIG": _TILE_INFO_OP_CONFIG,
            "_TILE_INFO_PAGE_ID": _TILE_INFO_PAGE_ID,
            "_INSTR_TILE_01": INSTR_TILE_01,
            "_INSTR_TILE_23": INSTR_TILE_23,
            "_INSTR_BARRIER_META_IDX": INSTR_BARRIER_META_IDX,
            "_INSTR_RANGE_META": INSTR_RANGE_META,
            "_INSTR_RANGE_END": INSTR_RANGE_END,
        }

    def _kernel_extra_exec_globals(
        self,
        kernel_cfg: Dict[str, Any],
        runtime: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the extra globals consumed by the generated kernel loop."""
        exec_globals = {
            "_work_notify_mbar": runtime["_work_notify_mbar"],
            "_compute_done_mbar": runtime["_compute_done_mbar"],
            "decompose_tile": runtime["decompose_tile"],
            "advance_tile": runtime["advance_tile"],
            "ld_shared_v2_b32": ld_shared_v2_b32,
            "st_shared_v2_b32": st_shared_v2_b32,
            "ld_shared_i64": ld_shared_i64,
            "st_shared_i64": st_shared_i64,
            "num_mma_warps": kernel_cfg["num_mma_warps"],
            "num_compute_threads": kernel_cfg["num_compute_threads"],
            "threads_per_block": kernel_cfg["actual_threads_per_block"],
            "setmaxregister_increase": runtime["setmaxregister_increase"],
            "setmaxregister_decrease": runtime["setmaxregister_decrease"],
            "dma_reg_count": kernel_cfg["dma_reg_count"],
            "mma_reg_count": kernel_cfg["mma_reg_count"],
            "tile_info_bytes": kernel_cfg["tile_info_bytes"],
            "FLAG_DISPATCH_LOAD": FLAG_DISPATCH_LOAD,
            "FLAG_PRODUCE_IDX": FLAG_PRODUCE_IDX,
            "FLAG_STORE_IDX": FLAG_STORE_IDX,
            "FLAG_LOAD_DONE": FLAG_LOAD_DONE,
            "FLAG_DATA_RELEASE_IDX": FLAG_DATA_RELEASE_IDX,
            "FLAG_DATA_PRODUCE_IDX": FLAG_DATA_PRODUCE_IDX,
            "_op_meta_i32": runtime["_op_meta_i32"],
            "_op_meta_base": runtime["_op_meta_base"],
            "_op_meta_i32_base": runtime["_op_meta_i32_base"],
            "max_waits": runtime["max_waits"],
            "max_compute_waits": runtime["max_compute_waits"],
            "max_signal_formulas": self._max_signal_formulas,
            "global_barrier_wait": global_barrier_wait,
            "global_barrier_wait_relaxed": global_barrier_wait_relaxed,
            "relaxed_global_barriers": self.config.relaxed_global_barriers,
            "global_barrier_sleep_ns": int(self.config.global_barrier_sleep_ns),
            "ld_global_i32": ld_global_i32,
            "has_communicate": runtime["has_communicate"],
            "needs_warp_transition": runtime["needs_warp_transition"],
            "sync_compute_warps_after_tile": self._sync_compute_warps_after_tile(),
            "loader_idle_sleep_ns": int(self.config.loader_idle_sleep_ns),
            "has_page_free_ops": bool(kernel_cfg["has_page_free_ops"]),
            "dispatch_load_uses_handler_local_idx": runtime["phase_uses_handler_local_idx"]["load"],
            "dispatch_compute_uses_handler_local_idx": runtime["phase_uses_handler_local_idx"]["compute"],
            "dispatch_store_uses_handler_local_idx": runtime["phase_uses_handler_local_idx"]["store"],
            "dispatch_communicate_uses_handler_local_idx": runtime["phase_uses_handler_local_idx"]["communicate"],
            "MIN_IDLE_REGS": MIN_IDLE_REGS,
            "_OP_PHASE_LOAD": _OP_PHASE_LOAD,
            "_OP_PHASE_STORE": _OP_PHASE_STORE,
            "_OP_PHASE_COMMUNICATE": _OP_PHASE_COMMUNICATE,
            **self._op_meta_exec_globals(),
            **self._tile_info_exec_globals(),
            **runtime["trace_exec_globals"],
        }
        if runtime["has_communicate"]:
            exec_globals.update(
                {
                    "dispatch_communicate": runtime["dispatch_communicate"],
                    "signal_peer_barriers": runtime["signal_peer_barriers"],
                    "_peer_barriers_data_ptr": kernel_cfg["peer_barriers_data_ptr"],
                }
            )
        return exec_globals

    def _build_ring_kernel_loop(self, kernel_cfg: Dict[str, Any], runtime: Dict[str, Any]):
        """Build the warp-specialized ring-buffer loop used by the persistent kernel.

        Returns a flat function whose body contains all dispatch_load/compute/store
        calls directly, so that _build_kernel_loop_source's regex can inject
        tensor and TMA parameters into every call site.
        """
        # Unpack into locals so they are captured by the closure below.
        # _build_kernel_loop_source extracts the function body as source text
        # and uses a regex to append tensor/TMA params to dispatch_* calls.
        # This only works when dispatch calls appear in the body itself (not
        # hidden inside nested @cute.jit sub-functions).
        def _kernel_loop_ring(
            instructions_ptr: Int64,
            barriers_ptr: Int64,
            op_configs_ptr: Int64,
            op_meta_ptr: Int64,
            signal_meta_ptr: Int64,
            peer_signal_ptr: Int64,
            num_instructions: Int32,
            tidx: Int32,
            block_id: Int32,
            num_blocks: Int32,
            smem_base: Int32,
            trace_buffer_ptr: Int64,
            wait_info_ptr: Int64,
            compute_wait_info_ptr: Int64,
        ) -> None:
            """Warp-specialized ring buffer loop.

            Warp roles (NUM_DMA_WARPS=3, W = num_mma_warps):
              Warps 0..W-1:  MMA warps — compute from ring buffer pages
              Warp W:        Controller — fetches instructions, pre-computed barrier wait
              Warp W+1:      Loader — dispatches TMA loads
              Warp W+2:      Store — waits for compute_done, dispatches TMA stores

            Mbarrier phases alternate 0/1 with each use (hardware auto-reset).
            """
            warp_id = tidx // Int32(32)
            lane_id = tidx % Int32(32)

            is_store_warp = warp_id == Int32(num_mma_warps + 2)

            if warp_id >= Int32(num_mma_warps):
                setmaxregister_decrease(dma_reg_count)
            if warp_id < Int32(num_mma_warps):
                setmaxregister_increase(mma_reg_count)

            iq_base = smem_base + Int32(iq_offset)
            flags_ptr = smem_base + Int32(flags_offset)

            if const_expr(tracing):
                _trace_buf = cute.make_tensor(
                    cute.make_ptr(cute.Uint8, trace_buffer_ptr),
                    cute.make_layout(1 << 24),
                )

            # ========== INIT (controller warp thread 0) ==========
            if warp_id == Int32(num_mma_warps):
                with cute.arch.elect_one():
                    st_shared_i32(flags_ptr + FLAG_DISPATCH_LOAD, Int32(-1))
                    st_shared_i32(flags_ptr + FLAG_PRODUCE_IDX, Int32(0))
                    st_shared_i32(flags_ptr + FLAG_STORE_IDX, Int32(0))
                    st_shared_i32(flags_ptr + FLAG_LOAD_DONE, Int32(0))
                    st_shared_i32(flags_ptr + FLAG_DATA_RELEASE_IDX, Int32(0))
                    st_shared_i32(flags_ptr + FLAG_DATA_PRODUCE_IDX, Int32(0))
                    for _ip in range(num_slots):
                        _slot_ti = smem_base + Int32(ring_state_offset) + Int32(_ip) * Int32(tile_info_bytes)
                        st_shared_i32(_slot_ti, Int32(-1))
                        mbarrier_init(
                            _work_notify_mbar(smem_base, Int32(_ip)),
                            Int32(1),
                        )
                        mbarrier_init(
                            _compute_done_mbar(smem_base, Int32(_ip)),
                            Int32(num_mma_warps),
                        )
                    mbarrier_init_fence()

            named_barrier_sync(Int32(0), Int32(threads_per_block))

            # ========== CONTROLLER WARP (fetch + pre-computed barrier wait) ==========
            if warp_id == Int32(num_mma_warps):
                if const_expr(tracing):
                    _ctrl_lane = begin_lane_dynamic_raw(
                        Int32(4),
                        Int32(trace_row_stride),
                        block_id,
                        Int32(3),
                        lane_id == Int32(0),
                    )
                produce_idx = Int32(0)
                _fetch_idx = block_id
                _fetch_limit = num_instructions
                _fetch_stride = num_blocks
                _ctrl_done = Int32(0)
                _ctrl_cached_config = Int64(0)
                _ctrl_cached_config_idx = Int32(-2)
                _ctrl_cached_handler = Int32(0)
                _ctrl_cached_wait_count = Int32(0)
                _ctrl_cached_wait_acquire = Int32(0)
                _ctrl_cached_barrier_meta_idx = Int32(0)
                _ctrl_cached_wait_barrier = Int32(-2)
                _ctrl_cached_wait_expected = Int32(-1)
                _ctrl_cached_op = Int32(TileInstruction.END_MARKER)
                _ctrl_cached_phase_mask = Int32(0)
                _ctrl_cached_t0 = Int32(0)
                _ctrl_cached_t1 = Int32(0)
                _ctrl_cached_t2 = Int32(0)
                _ctrl_cached_t3 = Int32(0)
                _ctrl_range_axis = Int32(-1)
                _ctrl_range_pos = Int32(0)
                _ctrl_range_end = Int32(0)
                _ctrl_range_offset = Int32(0)
                _ctrl_range_stride = Int32(1)
                _ctrl_range_active = Int32(0)

                produce_idx_ptr = flags_ptr + FLAG_PRODUCE_IDX
                store_idx_ptr = flags_ptr + FLAG_STORE_IDX
                load_done_ptr = flags_ptr + FLAG_LOAD_DONE
                _temp_instr = iq_base

                while _ctrl_done == Int32(0):
                    if lane_id == Int32(0):
                        _instr_op = Int32(TileInstruction.END_MARKER)
                        _instr_word0 = Int32(0)
                        if _ctrl_range_active != Int32(0):
                            _instr_op = _ctrl_cached_op
                        if _ctrl_range_active == Int32(0) and _fetch_idx < _fetch_limit:
                            load_instruction_to_smem(instructions_ptr, _fetch_idx, _temp_instr)
                            _instr_word0 = ld_shared_i32(_temp_instr)
                            _instr_op = _instr_word0 & Int32(65535)
                            if _instr_op == Int32(65535):
                                _instr_op = Int32(TileInstruction.END_MARKER)
                            if _instr_op == Int32(TileInstruction.END_MARKER):
                                _fetch_idx = _fetch_limit
                            if _instr_op != Int32(TileInstruction.END_MARKER):
                                _next_fetch_idx = _fetch_idx + _fetch_stride
                                if _next_fetch_idx < _fetch_limit:
                                    prefetch_instruction(instructions_ptr, _next_fetch_idx)
                                _fetch_idx = _fetch_idx + _fetch_stride

                        if _instr_op >= Int32(0) and _ctrl_range_active == Int32(0):
                            _ctrl_meta_base = _op_meta_base(_instr_op)
                            _phase_mask = _ctrl_cached_phase_mask
                            if _instr_op != _ctrl_cached_op:
                                _ctrl_cached_op = _instr_op
                                _ctrl_cached_handler = _op_meta_i32_base(
                                    op_meta_ptr, _ctrl_meta_base, Int32(_OP_META_HANDLER_IDX)
                                )
                                _ctrl_cached_wait_count = _op_meta_i32_base(
                                    op_meta_ptr, _ctrl_meta_base, Int32(_OP_META_WAIT_COUNT)
                                )
                                _ctrl_cached_wait_acquire = _op_meta_i32_base(
                                    op_meta_ptr, _ctrl_meta_base, Int32(_OP_META_WAIT_ACQUIRE)
                                )
                                _phase_mask = _op_meta_i32_base(
                                    op_meta_ptr, _ctrl_meta_base, Int32(_OP_META_PHASE_MASK)
                                )
                                _ctrl_cached_phase_mask = _phase_mask
                            _ctrl_cached_barrier_meta_idx = ld_shared_i32(
                                _temp_instr + Int32(4 * _INSTR_BARRIER_META_IDX)
                            )
                            if _instr_op != _ctrl_cached_config_idx:
                                _ctrl_cached_config = ld_global_i64(op_configs_ptr, _instr_op)
                                _ctrl_cached_config_idx = _instr_op

                            _ctrl_tile_01 = ld_shared_i32(_temp_instr + Int32(4 * _INSTR_TILE_01))
                            _ctrl_tile_23 = ld_shared_i32(_temp_instr + Int32(4 * _INSTR_TILE_23))
                            _ctrl_cached_t0 = _ctrl_tile_01 & Int32(65535)
                            _ctrl_cached_t1 = (_ctrl_tile_01 >> Int32(16)) & Int32(65535)
                            _ctrl_cached_t2 = _ctrl_tile_23 & Int32(65535)
                            _ctrl_cached_t3 = (_ctrl_tile_23 >> Int32(16)) & Int32(65535)
                            _ctrl_range_stride = Int32(1)
                            _ctrl_range_offset = Int32(0)
                            _ctrl_range_meta = (_instr_word0 >> Int32(16)) & Int32(65535)
                            _ctrl_range_axis = Int32(-1)
                            _ctrl_range_pos = Int32(0)
                            _ctrl_range_end = Int32(1)
                            if _ctrl_range_meta != Int32(0):
                                _ctrl_range_axis = (_ctrl_range_meta % Int32(16)) - Int32(1)
                                if _ctrl_range_axis == Int32(0):
                                    _ctrl_range_pos = _ctrl_cached_t0
                                if _ctrl_range_axis == Int32(1):
                                    _ctrl_range_pos = _ctrl_cached_t1
                                if _ctrl_range_axis == Int32(2):
                                    _ctrl_range_pos = _ctrl_cached_t2
                                if _ctrl_range_axis == Int32(3):
                                    _ctrl_range_pos = _ctrl_cached_t3
                                _ctrl_range_end = ld_shared_i32(
                                    _temp_instr + Int32(4 * _INSTR_RANGE_END)
                                ) & Int32(65535)
                            if _ctrl_range_axis < Int32(0) or _ctrl_range_end <= _ctrl_range_pos:
                                _ctrl_range_end = _ctrl_range_pos + Int32(1)
                                _ctrl_range_stride = Int32(1)
                            _ctrl_range_active = Int32(1)

                        if _instr_op >= Int32(0):
                            _ctrl_current_meta_idx = (
                                _ctrl_cached_barrier_meta_idx
                                + _ctrl_range_offset
                            )
                            if _ctrl_cached_wait_count > Int32(0):
                                _done_waits = Int32(0)
                                for _w in range_constexpr(max_waits):
                                    if _done_waits == Int32(0):
                                        if _w < _ctrl_cached_wait_count:
                                            _wi_off = (
                                                _ctrl_current_meta_idx * Int32(max_waits * 2)
                                                + Int32(_w * 2)
                                            )
                                            _bar_idx = ld_global_i32(wait_info_ptr, _wi_off)
                                            if _bar_idx >= Int32(0):
                                                _bar_exp = ld_global_i32(wait_info_ptr, _wi_off + Int32(1))
                                                if (
                                                    _bar_idx != _ctrl_cached_wait_barrier
                                                    or _bar_exp != _ctrl_cached_wait_expected
                                                ):
                                                    if const_expr(relaxed_global_barriers):
                                                        if const_expr(tracing):
                                                            _tdw = trace_start()
                                                        if _ctrl_cached_wait_acquire != Int32(0):
                                                            global_barrier_wait(barriers_ptr, _bar_idx, _bar_exp)
                                                        else:
                                                            global_barrier_wait_relaxed(
                                                                barriers_ptr,
                                                                _bar_idx,
                                                                _bar_exp,
                                                                Int32(global_barrier_sleep_ns),
                                                            )
                                                        if const_expr(tracing):
                                                            _ctrl_lane = end_event_dynamic_raw_1(
                                                                _tdw,
                                                                _trace_buf,
                                                                Int32(trace_row_stride),
                                                                _ctrl_lane,
                                                                Int32(trace_dep_wait_fmt),
                                                                _instr_op,
                                                            )
                                                    else:
                                                        if const_expr(tracing):
                                                            _tdw = trace_start()
                                                        global_barrier_wait(barriers_ptr, _bar_idx, _bar_exp)
                                                        if const_expr(tracing):
                                                            _ctrl_lane = end_event_dynamic_raw_1(
                                                                _tdw,
                                                                _trace_buf,
                                                                Int32(trace_row_stride),
                                                                _ctrl_lane,
                                                                Int32(trace_dep_wait_fmt),
                                                                _instr_op,
                                                            )
                                                    _ctrl_cached_wait_barrier = _bar_idx
                                                    _ctrl_cached_wait_expected = _bar_exp
                                            else:
                                                _done_waits = Int32(1)
                                        else:
                                            _done_waits = Int32(1)

                            _si = ld_shared_i32(store_idx_ptr)
                            while (produce_idx - _si) >= Int32(num_slots):
                                _wait_slot = produce_idx % Int32(num_slots)
                                _wait_phase = ((produce_idx // Int32(num_slots)) + Int32(1)) % Int32(2)
                                if const_expr(tracing):
                                    _trfw = trace_start()
                                mbarrier_wait(
                                    _compute_done_mbar(smem_base, _wait_slot), _wait_phase
                                )
                                if const_expr(tracing):
                                    _ctrl_lane = end_event_dynamic_raw_1(
                                        _trfw,
                                        _trace_buf,
                                        Int32(trace_row_stride),
                                        _ctrl_lane,
                                        Int32(trace_ring_full_wait_fmt),
                                        _instr_op,
                                    )
                                _si = ld_shared_i32(store_idx_ptr)

                            _p_slot = produce_idx % Int32(num_slots)
                            _p_ti = smem_base + Int32(ring_state_offset) + _p_slot * Int32(tile_info_bytes)
                            st_shared_i32(
                                _p_ti + Int32(4 * _TILE_INFO_PAGE_ID),
                                _p_slot % Int32(num_pages),
                            )
                            _p_t0 = _ctrl_cached_t0
                            _p_t1 = _ctrl_cached_t1
                            _p_t2 = _ctrl_cached_t2
                            _p_t3 = _ctrl_cached_t3
                            if _ctrl_range_axis == Int32(0):
                                _p_t0 = _ctrl_range_pos
                            if _ctrl_range_axis == Int32(1):
                                _p_t1 = _ctrl_range_pos
                            if _ctrl_range_axis == Int32(2):
                                _p_t2 = _ctrl_range_pos
                            if _ctrl_range_axis == Int32(3):
                                _p_t3 = _ctrl_range_pos
                            st_shared_i32(_p_ti, _instr_op)
                            st_shared_i32(
                                _p_ti + Int32(4 * _TILE_INFO_HANDLER_IDX),
                                _ctrl_cached_handler,
                            )
                            st_shared_i32(_p_ti + Int32(4 * _TILE_INFO_TILE_0), _p_t0)
                            st_shared_i32(_p_ti + Int32(4 * _TILE_INFO_TILE_1), _p_t1)
                            st_shared_i32(_p_ti + Int32(4 * _TILE_INFO_TILE_2), _p_t2)
                            st_shared_i32(_p_ti + Int32(4 * _TILE_INFO_TILE_3), _p_t3)
                            st_shared_i32(
                                _p_ti + Int32(4 * _TILE_INFO_INSTRUCTION_IDX),
                                _ctrl_current_meta_idx,
                            )
                            st_shared_i64(
                                _p_ti + Int32(4 * _TILE_INFO_OP_CONFIG),
                                _ctrl_cached_config,
                            )
                            produce_idx = produce_idx + Int32(1)
                            st_shared_release_cta_i32(produce_idx_ptr, produce_idx)
                            _ctrl_range_offset = _ctrl_range_offset + Int32(1)
                            _ctrl_range_pos = _ctrl_range_pos + _ctrl_range_stride
                            if _ctrl_range_pos >= _ctrl_range_end:
                                _ctrl_range_active = Int32(0)

                        if (
                            _instr_op == Int32(TileInstruction.END_MARKER)
                            and _ctrl_range_active == Int32(0)
                        ):
                            if _fetch_idx >= _fetch_limit:
                                _store_idx_done = ld_shared_i32(store_idx_ptr)
                                if (produce_idx - _store_idx_done) < Int32(num_slots):
                                    _sent = produce_idx % Int32(num_slots)
                                    st_shared_i32(
                                        smem_base + Int32(ring_state_offset) + _sent * Int32(tile_info_bytes),
                                        Int32(TileInstruction.END_MARKER),
                                    )
                                    mbarrier_arrive(_work_notify_mbar(smem_base, _sent))
                                    st_shared_i32(load_done_ptr, Int32(1))

                    _ctrl_done = ld_shared_i32(load_done_ptr)
                if const_expr(tracing):
                    finish_lane_dynamic_raw(_trace_buf, _ctrl_lane)

            # ========== LOADER WARP (TMA dispatch) ==========
            if warp_id == Int32(num_mma_warps + 1):
                if const_expr(tracing):
                    _dma_lane = begin_lane_dynamic_raw(
                        Int32(4),
                        Int32(trace_row_stride),
                        block_id,
                        Int32(0),
                        lane_id == Int32(0),
                    )

                _ldr_done = Int32(0)
                _ldr_load_done_ptr = flags_ptr + FLAG_LOAD_DONE
                _ldr_produce_ptr = flags_ptr + FLAG_PRODUCE_IDX
                _ldr_idx = Int32(0)

                while _ldr_done == Int32(0):
                    _p_idx = ld_shared_acquire_cta_i32(_ldr_produce_ptr)
                    if _ldr_idx < _p_idx:
                        _dl_slot = _ldr_idx % Int32(num_slots)
                        _dl_ti = smem_base + Int32(ring_state_offset) + _dl_slot * Int32(tile_info_bytes)
                        _dl_op = ld_shared_i32(_dl_ti)
                        if _dl_op != Int32(TileInstruction.END_MARKER):
                            _dl_meta_base = _op_meta_base(_dl_op)
                            _dl_handler = ld_shared_i32(_dl_ti + Int32(4 * _TILE_INFO_HANDLER_IDX))
                            _dl_handler_local = Int32(0)
                            if const_expr(dispatch_load_uses_handler_local_idx):
                                _dl_handler_local = _op_meta_i32_base(
                                    op_meta_ptr, _dl_meta_base, Int32(_OP_META_LOAD_LOCAL_IDX)
                                )
                            _dl_0 = ld_shared_i32(_dl_ti + Int32(4 * _TILE_INFO_TILE_0))
                            _dl_1 = ld_shared_i32(_dl_ti + Int32(4 * _TILE_INFO_TILE_1))
                            _dl_2 = ld_shared_i32(_dl_ti + Int32(4 * _TILE_INFO_TILE_2))
                            _dl_3 = ld_shared_i32(_dl_ti + Int32(4 * _TILE_INFO_TILE_3))
                            _dl_config = ld_shared_i64(_dl_ti + Int32(4 * _TILE_INFO_OP_CONFIG))
                            _dl_mbar = _work_notify_mbar(smem_base, _dl_slot)
                            if const_expr(tracing):
                                _tl = trace_start()
                            if const_expr(has_page_free_ops):
                                _dl_page = ld_shared_i32(_dl_ti + Int32(4 * _TILE_INFO_PAGE_ID))
                                _dl_pp = _get_page_ptr(smem_base, _dl_page)
                            else:
                                _dl_pp = _get_page_ptr(smem_base, _dl_slot)
                            if const_expr(dispatch_load_uses_handler_local_idx):
                                dispatch_load(
                                    _dl_handler,
                                    _dl_handler_local,
                                    _dl_pp,
                                    _dl_0,
                                    _dl_1,
                                    _dl_2,
                                    _dl_3,
                                    _dl_config,
                                    _dl_mbar,
                                )
                            else:
                                dispatch_load(
                                    _dl_handler,
                                    _dl_pp,
                                    _dl_0,
                                    _dl_1,
                                    _dl_2,
                                    _dl_3,
                                    _dl_config,
                                    _dl_mbar,
                                )
                            if const_expr(tracing):
                                _dma_lane = end_event_dynamic_raw_1(
                                    _tl,
                                    _trace_buf,
                                    Int32(trace_row_stride),
                                    _dma_lane,
                                    ld_global_i32(trace_load_fmt_ptr, _dl_op),
                                    _dl_op,
                                )
                        _ldr_idx = _ldr_idx + Int32(1)
                    else:
                        nanosleep(Int32(loader_idle_sleep_ns))

                    _ldr_done = ld_shared_i32(_ldr_load_done_ptr)
                    if _ldr_done == Int32(1) and _ldr_idx < ld_shared_acquire_cta_i32(_ldr_produce_ptr):
                        _ldr_done = Int32(0)

                if const_expr(tracing):
                    finish_lane_dynamic_raw(_trace_buf, _dma_lane)

            # ========== STORE WARP LOOP ==========
            if is_store_warp:
                if const_expr(tracing):
                    _store_lane = begin_lane_dynamic_raw(
                        Int32(4),
                        Int32(trace_row_stride),
                        block_id,
                        Int32(2),
                        lane_id == Int32(0),
                    )
                _sw_done = Int32(0)
                store_idx_ptr = flags_ptr + FLAG_STORE_IDX
                produce_idx_ptr = flags_ptr + FLAG_PRODUCE_IDX
                load_done_ptr = flags_ptr + FLAG_LOAD_DONE
                while _sw_done == Int32(0):
                    _s_idx = ld_shared_i32(store_idx_ptr)
                    _p_idx = ld_shared_i32(produce_idx_ptr)

                    if _s_idx < _p_idx:
                        _s_slot = _s_idx % Int32(num_slots)
                        _s_phase = (_s_idx // Int32(num_slots)) % Int32(2)

                        _ds_ti = smem_base + Int32(ring_state_offset) + _s_slot * Int32(tile_info_bytes)
                        _ds_op = ld_shared_i32(_ds_ti)
                        _ds_meta_base = _op_meta_base(_ds_op)
                        _ds_handler = ld_shared_i32(_ds_ti + Int32(4 * _TILE_INFO_HANDLER_IDX))
                        _ds_signal_count = _op_meta_i32_base(
                            op_meta_ptr, _ds_meta_base, Int32(_OP_META_SIGNAL_COUNT)
                        )
                        _ds_handler_local = Int32(0)
                        if const_expr(dispatch_store_uses_handler_local_idx):
                            _ds_handler_local = _op_meta_i32_base(
                                op_meta_ptr, _ds_meta_base, Int32(_OP_META_STORE_LOCAL_IDX)
                            )
                        _dc_handler_local = Int32(0)
                        if const_expr(dispatch_communicate_uses_handler_local_idx):
                            _dc_handler_local = _op_meta_i32_base(
                                op_meta_ptr, _ds_meta_base, Int32(_OP_META_COMM_LOCAL_IDX)
                            )
                        _ds_0 = ld_shared_i32(_ds_ti + Int32(4 * _TILE_INFO_TILE_0))
                        _ds_1 = ld_shared_i32(_ds_ti + Int32(4 * _TILE_INFO_TILE_1))
                        _ds_2 = ld_shared_i32(_ds_ti + Int32(4 * _TILE_INFO_TILE_2))
                        _ds_3 = ld_shared_i32(_ds_ti + Int32(4 * _TILE_INFO_TILE_3))
                        _ds_instruction_idx = ld_shared_i32(
                            _ds_ti + Int32(4 * _TILE_INFO_INSTRUCTION_IDX)
                        )
                        _ds_config = ld_shared_i64(_ds_ti + Int32(4 * _TILE_INFO_OP_CONFIG))
                        if const_expr(has_page_free_ops):
                            _ds_page = ld_shared_i32(_ds_ti + Int32(4 * _TILE_INFO_PAGE_ID))
                        else:
                            _ds_page = _s_slot

                        if const_expr(tracing):
                            _tsw = trace_start()
                        mbarrier_wait(_compute_done_mbar(smem_base, _s_slot), _s_phase)
                        if const_expr(tracing):
                            _store_lane = end_event_dynamic_raw_1(
                                _tsw,
                                _trace_buf,
                                Int32(trace_row_stride),
                                _store_lane,
                                Int32(trace_compute_wait_fmt),
                                _ds_op,
                            )

                        if const_expr(tracing):
                            _tss = trace_start()
                        if const_expr(has_page_free_ops):
                            _ds_pp = _get_page_ptr(smem_base, _ds_page)
                        else:
                            _ds_pp = _get_page_ptr(smem_base, _s_slot)
                        if const_expr(dispatch_store_uses_handler_local_idx):
                            dispatch_store(
                                _ds_handler,
                                _ds_handler_local,
                                _ds_pp,
                                _ds_0,
                                _ds_1,
                                _ds_2,
                                _ds_3,
                                _ds_config,
                            )
                        else:
                            dispatch_store(
                                _ds_handler,
                                _ds_pp,
                                _ds_0,
                                _ds_1,
                                _ds_2,
                                _ds_3,
                                _ds_config,
                            )
                        if const_expr(has_communicate):
                            if const_expr(dispatch_communicate_uses_handler_local_idx):
                                dispatch_communicate(
                                    _ds_handler,
                                    _dc_handler_local,
                                    _ds_pp,
                                    _ds_0,
                                    _ds_1,
                                    _ds_2,
                                    _ds_3,
                                    _ds_config,
                                )
                            else:
                                dispatch_communicate(
                                    _ds_handler,
                                    _ds_pp,
                                    _ds_0,
                                    _ds_1,
                                    _ds_2,
                                    _ds_3,
                                    _ds_config,
                                )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        if const_expr(tracing):
                            _store_lane = end_event_dynamic_raw_1(
                                _tss,
                                _trace_buf,
                                Int32(trace_row_stride),
                                _store_lane,
                                ld_global_i32(trace_store_fmt_ptr, _ds_op),
                                _ds_op,
                            )

                        with cute.arch.elect_one():
                            if _ds_signal_count > Int32(0):
                                if _ds_signal_count == Int32(1):
                                    _sig_barrier = ld_global_i32(
                                        signal_meta_ptr,
                                        _ds_instruction_idx * Int32(max_signal_formulas),
                                    )
                                    if _sig_barrier >= Int32(0):
                                        global_barrier_signal_gpu(barriers_ptr, _sig_barrier)
                                else:
                                    signal_barriers(
                                        signal_meta_ptr,
                                        _ds_instruction_idx,
                                        _ds_signal_count,
                                        barriers_ptr,
                                    )
                            if const_expr(has_communicate):
                                _ds_lin = Int32(0)
                                _ds_s0 = _op_meta_i32_base(
                                    op_meta_ptr, _ds_meta_base, Int32(_OP_META_STRIDE_0)
                                )
                                _ds_s1 = _op_meta_i32_base(
                                    op_meta_ptr, _ds_meta_base, Int32(_OP_META_STRIDE_1)
                                )
                                _ds_s2 = _op_meta_i32_base(
                                    op_meta_ptr, _ds_meta_base, Int32(_OP_META_STRIDE_2)
                                )
                                _ds_s3 = _op_meta_i32_base(
                                    op_meta_ptr, _ds_meta_base, Int32(_OP_META_STRIDE_3)
                                )
                                _ds_lin = (
                                    _ds_0 * _ds_s0
                                    + _ds_1 * _ds_s1
                                    + _ds_2 * _ds_s2
                                    + _ds_3 * _ds_s3
                                )
                                signal_peer_barriers(
                                    peer_signal_ptr,
                                    _ds_op,
                                    _ds_lin,
                                    Int64(_peer_barriers_data_ptr),
                                )
                            st_shared_i32(store_idx_ptr, _s_idx + Int32(1))

                    if _s_idx >= _p_idx:
                        if ld_shared_i32(load_done_ptr) == Int32(1):
                            _sw_done = Int32(1)
                        if ld_shared_i32(load_done_ptr) != Int32(1):
                            _sw_next_slot = _s_idx % Int32(num_slots)
                            _sw_next_phase = (_s_idx // Int32(num_slots)) % Int32(2)
                            mbarrier_wait(
                                _work_notify_mbar(smem_base, _sw_next_slot),
                                _sw_next_phase,
                            )

                if const_expr(tracing):
                    finish_lane_dynamic_raw(_trace_buf, _store_lane)

            # ========== MMA WARP LOOP ==========
            if warp_id < Int32(num_mma_warps):
                if const_expr(tracing):
                    _mma_lane = begin_lane_dynamic_raw(
                        Int32(4),
                        Int32(trace_row_stride),
                        block_id,
                        Int32(1),
                        (warp_id == Int32(0)) & (lane_id == Int32(0)),
                    )

                consume_ptr = Int32(0)
                mma_running = Int32(1)
                _cached_op_idx = Int32(-1)
                _active_op_warps = Int32(num_mma_warps)
                _cached_compute_wait_count = Int32(0)
                _cached_compute_wait_acquire = Int32(0)
                _cached_compute_wait_barrier = Int32(-2)
                _cached_compute_wait_expected = Int32(-1)

                while mma_running == Int32(1):
                    slot = consume_ptr % Int32(num_slots)

                    _wn_phase = (consume_ptr // Int32(num_slots)) % Int32(2)
                    if const_expr(tracing):
                        _tw = trace_start()
                    mbarrier_wait(_work_notify_mbar(smem_base, slot), _wn_phase)

                    tile_info_ptr = smem_base + Int32(ring_state_offset) + slot * Int32(tile_info_bytes)
                    op_idx = ld_shared_i32(tile_info_ptr)

                    if op_idx == Int32(TileInstruction.END_MARKER):
                        mma_running = Int32(0)

                    if op_idx != Int32(TileInstruction.END_MARKER):
                        if const_expr(tracing):
                            _mma_lane = end_event_dynamic_raw_1(
                                _tw,
                                _trace_buf,
                                Int32(trace_row_stride),
                                _mma_lane,
                                Int32(trace_data_wait_fmt),
                                op_idx,
                            )

                        tile_0 = ld_shared_i32(tile_info_ptr + Int32(4 * _TILE_INFO_TILE_0))
                        tile_1 = ld_shared_i32(tile_info_ptr + Int32(4 * _TILE_INFO_TILE_1))
                        tile_2 = ld_shared_i32(tile_info_ptr + Int32(4 * _TILE_INFO_TILE_2))
                        tile_3 = ld_shared_i32(tile_info_ptr + Int32(4 * _TILE_INFO_TILE_3))
                        _op_meta_base_cached = _op_meta_base(op_idx)
                        _handler_idx = ld_shared_i32(
                            tile_info_ptr + Int32(4 * _TILE_INFO_HANDLER_IDX)
                        )
                        _handler_local_idx = Int32(0)
                        if const_expr(dispatch_compute_uses_handler_local_idx):
                            _handler_local_idx = _op_meta_i32_base(
                                op_meta_ptr, _op_meta_base_cached, Int32(_OP_META_COMPUTE_LOCAL_IDX)
                            )
                        _op_config = ld_shared_i64(tile_info_ptr + Int32(4 * _TILE_INFO_OP_CONFIG))
                        if const_expr(has_page_free_ops):
                            _page_id = ld_shared_i32(tile_info_ptr + Int32(4 * _TILE_INFO_PAGE_ID))
                        else:
                            _page_id = slot
                            if op_idx != _cached_op_idx:
                                _cached_op_idx = op_idx

                            if const_expr(needs_warp_transition):
                                _active_op_warps = _op_meta_i32_base(
                                    op_meta_ptr, _op_meta_base_cached, Int32(_OP_META_NUM_WARPS)
                                )
                                if warp_id >= _active_op_warps:
                                    setmaxregister_decrease(MIN_IDLE_REGS)
                                named_barrier_sync(
                                    Int32(1), Int32(num_compute_threads))
                                if warp_id < _active_op_warps:
                                    setmaxregister_increase(mma_reg_count)
                            if const_expr(max_compute_waits > 0):
                                _cached_compute_wait_count = _op_meta_i32_base(
                                    op_meta_ptr,
                                    _op_meta_base_cached,
                                    Int32(_OP_META_COMPUTE_WAIT_COUNT),
                                )
                                _cached_compute_wait_acquire = _op_meta_i32_base(
                                    op_meta_ptr,
                                    _op_meta_base_cached,
                                    Int32(_OP_META_WAIT_ACQUIRE),
                                )

                        if const_expr(max_compute_waits > 0):
                            if _cached_compute_wait_count > Int32(0):
                                _barrier_meta_idx = ld_shared_i32(
                                    tile_info_ptr + Int32(4 * _TILE_INFO_INSTRUCTION_IDX)
                                )
                                if warp_id == Int32(0) and lane_id == Int32(0):
                                    _done_compute_waits = Int32(0)
                                    for _cw in range_constexpr(max_compute_waits):
                                        if _done_compute_waits == Int32(0):
                                            if _cw < _cached_compute_wait_count:
                                                _cwi_off = (
                                                    _barrier_meta_idx
                                                    * Int32(max_compute_waits * 2)
                                                    + Int32(_cw * 2)
                                                )
                                                _cbar_idx = ld_global_i32(
                                                    compute_wait_info_ptr, _cwi_off
                                                )
                                                if _cbar_idx >= Int32(0):
                                                    _cbar_exp = ld_global_i32(
                                                        compute_wait_info_ptr,
                                                        _cwi_off + Int32(1),
                                                    )
                                                    if (
                                                        _cbar_idx != _cached_compute_wait_barrier
                                                        or _cbar_exp != _cached_compute_wait_expected
                                                    ):
                                                        if const_expr(tracing):
                                                            _tdw = trace_start()
                                                        if const_expr(relaxed_global_barriers):
                                                            if _cached_compute_wait_acquire != Int32(0):
                                                                global_barrier_wait(
                                                                    barriers_ptr, _cbar_idx, _cbar_exp
                                                                )
                                                            else:
                                                                global_barrier_wait_relaxed(
                                                                    barriers_ptr,
                                                                    _cbar_idx,
                                                                    _cbar_exp,
                                                                    Int32(global_barrier_sleep_ns),
                                                                )
                                                        else:
                                                            global_barrier_wait(
                                                                barriers_ptr, _cbar_idx, _cbar_exp
                                                            )
                                                        if const_expr(tracing):
                                                            _mma_lane = end_event_dynamic_raw_1(
                                                                _tdw,
                                                                _trace_buf,
                                                                Int32(trace_row_stride),
                                                                _mma_lane,
                                                                Int32(trace_dep_wait_fmt),
                                                                op_idx,
                                                            )
                                                        _cached_compute_wait_barrier = _cbar_idx
                                                        _cached_compute_wait_expected = _cbar_exp
                                                else:
                                                    _done_compute_waits = Int32(1)
                                            else:
                                                _done_compute_waits = Int32(1)
                                named_barrier_sync(Int32(1), Int32(num_compute_threads))

                        if const_expr(tracing):
                            _tc = trace_start()

                        if warp_id < _active_op_warps:
                            if const_expr(has_page_free_ops):
                                page_ptr = _get_page_ptr(smem_base, _page_id)
                            else:
                                page_ptr = _get_page_ptr(smem_base, slot)
                            if const_expr(dispatch_compute_uses_handler_local_idx):
                                dispatch_compute(
                                    _handler_idx,
                                    _handler_local_idx,
                                    page_ptr,
                                    tile_0,
                                    tile_1,
                                    tile_2,
                                    tile_3,
                                    _op_config,
                                )
                            else:
                                dispatch_compute(
                                    _handler_idx,
                                    page_ptr,
                                    tile_0,
                                    tile_1,
                                    tile_2,
                                    tile_3,
                                    _op_config,
                                )

                        if const_expr(sync_compute_warps_after_tile):
                            named_barrier_sync(Int32(1), Int32(num_compute_threads))

                        if const_expr(tracing):
                            _mma_lane = end_event_dynamic_raw_1(
                                _tc,
                                _trace_buf,
                                Int32(trace_row_stride),
                                _mma_lane,
                                ld_global_i32(trace_compute_fmt_ptr, op_idx),
                                op_idx,
                            )

                        with cute.arch.elect_one():
                            mbarrier_arrive(_compute_done_mbar(smem_base, slot))

                        consume_ptr = consume_ptr + Int32(1)

                if const_expr(tracing):
                    finish_lane_dynamic_raw(_trace_buf, _mma_lane)

        return _kernel_loop_ring

    def _build_compute_only_kernel_loop(self, kernel_cfg, runtime):
        """Build a lighter replay loop for graphs with compute phases only."""
        tracing = bool(self.config.tracing)
        dispatch_compute = runtime["dispatch_compute"]
        signal_barriers = runtime["signal_barriers"]
        max_waits = runtime["max_waits"]
        max_signal_formulas = self._max_signal_formulas
        num_mma_warps = kernel_cfg["num_mma_warps"]
        num_compute_threads = kernel_cfg["num_compute_threads"]
        mma_reg_count = kernel_cfg["mma_reg_count"]
        sync_compute_warps_after_tile = self._sync_compute_warps_after_tile()
        relaxed_global_barriers = bool(self.config.relaxed_global_barriers)
        global_barrier_sleep_ns = int(self.config.global_barrier_sleep_ns)

        @cute.jit
        def _kernel_loop_compute_only(
            instructions_ptr: Int64,
            barriers_ptr: Int64,
            op_configs_ptr: Int64,
            op_meta_ptr: Int64,
            signal_meta_ptr: Int64,
            num_instructions: Int32,
            tidx: Int32,
            block_id: Int32,
            num_blocks: Int32,
            smem_base: Int32,
            trace_buffer_ptr: Int64,
            wait_info_ptr: Int64,
            compute_wait_info_ptr: Int64,
        ):
            warp_id = tidx // Int32(32)
            lane_id = tidx % Int32(32)
            if const_expr(tracing):
                _trace_buf = cute.make_tensor(
                    cute.make_ptr(cute.Uint8, trace_buffer_ptr),
                    cute.make_layout(1 << 24),
                )
                _mma_lane = begin_lane_dynamic_raw(
                    Int32(4),
                    Int32(trace_row_stride),
                    block_id,
                    Int32(1),
                    (warp_id == Int32(0)) & (lane_id == Int32(0)),
                )

            if warp_id < Int32(num_mma_warps):
                setmaxregister_increase(mma_reg_count)

            iq_base = smem_base + Int32(iq_offset)
            _fetch_idx = block_id
            _fetch_limit = num_instructions
            _fetch_stride = num_blocks

            _cached_wait_count = Int32(0)
            _cached_wait_acquire = Int32(0)
            _cached_signal_count = Int32(0)
            _cached_wait_barrier = Int32(-2)
            _cached_wait_expected = Int32(-1)
            _running = Int32(1)
            _cached_op_idx = Int32(TileInstruction.END_MARKER)
            _cached_handler = Int32(0)
            _cached_phase_mask = Int32(0)
            _cached_config_idx = Int32(-2)
            _cached_config = Int64(0)

            while _running == Int32(1):
                if warp_id == Int32(0) and lane_id == Int32(0):
                    _instr_op = Int32(TileInstruction.END_MARKER)
                    if _fetch_idx < _fetch_limit:
                        load_instruction_to_smem(instructions_ptr, _fetch_idx, iq_base)
                        _instr_word0 = ld_shared_i32(iq_base)
                        _instr_op = _instr_word0 & Int32(65535)
                        if _instr_op == Int32(65535):
                            _instr_op = Int32(TileInstruction.END_MARKER)
                        if _instr_op == Int32(TileInstruction.END_MARKER):
                            _fetch_idx = _fetch_limit
                        else:
                            _next_fetch_idx = _fetch_idx + _fetch_stride
                            if _next_fetch_idx < _fetch_limit:
                                prefetch_instruction(instructions_ptr, _next_fetch_idx)
                            _fetch_idx = _fetch_idx + _fetch_stride
                    else:
                        st_shared_i32(iq_base, Int32(65535))
                named_barrier_sync(Int32(1), Int32(num_compute_threads))

                _op_word0 = ld_shared_i32(iq_base)
                op_idx = _op_word0 & Int32(65535)
                if op_idx == Int32(65535):
                    op_idx = Int32(TileInstruction.END_MARKER)
                if op_idx == Int32(TileInstruction.END_MARKER):
                    _running = Int32(0)

                if op_idx != Int32(TileInstruction.END_MARKER):
                    _config = Int64(0)
                    _compute_local = Int32(0)
                    _meta_base = _op_meta_base(op_idx)
                    _op_meta_changed = op_idx != _cached_op_idx
                    if _op_meta_changed:
                        _cached_handler = _op_meta_i32_base(
                            op_meta_ptr, _meta_base, Int32(_OP_META_HANDLER_IDX)
                        )
                        _cached_phase_mask = _op_meta_i32_base(
                            op_meta_ptr, _meta_base, Int32(_OP_META_PHASE_MASK)
                        )
                        _cached_op_idx = op_idx
                    _handler = _cached_handler
                    _barrier_meta_idx = ld_shared_i32(iq_base + Int32(4 * _INSTR_BARRIER_META_IDX))
                    if op_idx != _cached_config_idx:
                        _cached_config = ld_global_i64(op_configs_ptr, op_idx)
                        _cached_config_idx = op_idx
                    _config = _cached_config
                    if const_expr(dispatch_compute_uses_handler_local_idx):
                        _compute_local = _op_meta_i32_base(
                            op_meta_ptr, _meta_base, Int32(_OP_META_COMPUTE_LOCAL_IDX)
                        )

                    if warp_id == Int32(0) and lane_id == Int32(0):
                        if _op_meta_changed:
                            _cached_wait_count = _op_meta_i32_base(
                                op_meta_ptr, _meta_base, Int32(_OP_META_WAIT_COUNT)
                            )
                            _cached_signal_count = _op_meta_i32_base(
                                op_meta_ptr, _meta_base, Int32(_OP_META_SIGNAL_COUNT)
                            )
                            _cached_wait_acquire = _op_meta_i32_base(
                                op_meta_ptr, _meta_base, Int32(_OP_META_WAIT_ACQUIRE)
                            )

                        if _cached_wait_count > Int32(0):
                            _done_waits = Int32(0)
                            for _w in range_constexpr(max_waits):
                                if _done_waits == Int32(0):
                                    if _w < _cached_wait_count:
                                        _wi_off = (
                                            _barrier_meta_idx * Int32(max_waits * 2)
                                            + Int32(_w * 2)
                                        )
                                        _bar_idx = ld_global_i32(wait_info_ptr, _wi_off)
                                        if _bar_idx >= Int32(0):
                                            _bar_exp = ld_global_i32(wait_info_ptr, _wi_off + Int32(1))
                                            if (
                                                _bar_idx != _cached_wait_barrier
                                                or _bar_exp != _cached_wait_expected
                                            ):
                                                if const_expr(relaxed_global_barriers):
                                                    if _cached_wait_acquire != Int32(0):
                                                        global_barrier_wait(barriers_ptr, _bar_idx, _bar_exp)
                                                    else:
                                                        global_barrier_wait_relaxed(
                                                            barriers_ptr,
                                                            _bar_idx,
                                                            _bar_exp,
                                                            Int32(global_barrier_sleep_ns),
                                                        )
                                                else:
                                                    global_barrier_wait(barriers_ptr, _bar_idx, _bar_exp)
                                                _cached_wait_barrier = _bar_idx
                                                _cached_wait_expected = _bar_exp
                                        else:
                                            _done_waits = Int32(1)
                                    else:
                                        _done_waits = Int32(1)

                    named_barrier_sync(Int32(1), Int32(num_compute_threads))

                    _tile_01 = ld_shared_i32(iq_base + Int32(4 * _INSTR_TILE_01))
                    _tile_23 = ld_shared_i32(iq_base + Int32(4 * _INSTR_TILE_23))
                    tile_0 = _tile_01 & Int32(65535)
                    tile_1 = (_tile_01 >> Int32(16)) & Int32(65535)
                    tile_2 = _tile_23 & Int32(65535)
                    tile_3 = (_tile_23 >> Int32(16)) & Int32(65535)
                    page_ptr = _get_page_ptr(smem_base, Int32(0))
                    _phase_mask = _cached_phase_mask
                    _range_pos = Int32(0)
                    _range_end = Int32(0)
                    _range_stride = Int32(1)
                    _range_offset = Int32(0)
                    _range_axis = Int32(-1)
                    _range_meta = (_op_word0 >> Int32(16)) & Int32(65535)
                    _range_axis = (
                        _range_meta % Int32(16)
                    ) - Int32(1)
                    if _range_axis == Int32(0):
                        _range_pos = tile_0
                    if _range_axis == Int32(1):
                        _range_pos = tile_1
                    if _range_axis == Int32(2):
                        _range_pos = tile_2
                    if _range_axis == Int32(3):
                        _range_pos = tile_3
                    _range_end = ld_shared_i32(
                        iq_base + Int32(4 * _INSTR_RANGE_END)
                    ) & Int32(65535)
                    if _range_axis < Int32(0) or _range_end <= _range_pos:
                        _range_end = _range_pos + Int32(1)
                        _range_stride = Int32(1)

                    while _range_pos < _range_end:
                        _current_meta_idx = _barrier_meta_idx + _range_offset
                        if (
                            _range_axis >= Int32(0)
                            and warp_id == Int32(0)
                            and lane_id == Int32(0)
                        ):
                            if _cached_wait_count > Int32(0):
                                _done_waits = Int32(0)
                                for _w in range_constexpr(max_waits):
                                    if _done_waits == Int32(0):
                                        if _w < _cached_wait_count:
                                            _wi_off = (
                                                _current_meta_idx * Int32(max_waits * 2)
                                                + Int32(_w * 2)
                                            )
                                            _bar_idx = ld_global_i32(wait_info_ptr, _wi_off)
                                            if _bar_idx >= Int32(0):
                                                _bar_exp = ld_global_i32(wait_info_ptr, _wi_off + Int32(1))
                                                if (
                                                    _bar_idx != _cached_wait_barrier
                                                    or _bar_exp != _cached_wait_expected
                                                ):
                                                    if const_expr(relaxed_global_barriers):
                                                        if _cached_wait_acquire != Int32(0):
                                                            global_barrier_wait(barriers_ptr, _bar_idx, _bar_exp)
                                                        else:
                                                            global_barrier_wait_relaxed(
                                                                barriers_ptr,
                                                                _bar_idx,
                                                                _bar_exp,
                                                                Int32(global_barrier_sleep_ns),
                                                            )
                                                    else:
                                                        global_barrier_wait(barriers_ptr, _bar_idx, _bar_exp)
                                                    _cached_wait_barrier = _bar_idx
                                                    _cached_wait_expected = _bar_exp
                                            else:
                                                _done_waits = Int32(1)
                                        else:
                                            _done_waits = Int32(1)
                        if _range_axis >= Int32(0):
                            named_barrier_sync(Int32(1), Int32(num_compute_threads))

                        if _range_axis == Int32(0):
                            tile_0 = _range_pos
                        if _range_axis == Int32(1):
                            tile_1 = _range_pos
                        if _range_axis == Int32(2):
                            tile_2 = _range_pos
                        if _range_axis == Int32(3):
                            tile_3 = _range_pos
                        if const_expr(dispatch_compute_uses_handler_local_idx):
                            if const_expr(tracing):
                                _tc = trace_start()
                            dispatch_compute(
                                _handler,
                                _compute_local,
                                page_ptr,
                                tile_0,
                                tile_1,
                                tile_2,
                                tile_3,
                                _config,
                            )
                            if const_expr(tracing):
                                _mma_lane = end_event_dynamic_raw_1(
                                    _tc,
                                    _trace_buf,
                                    Int32(trace_row_stride),
                                    _mma_lane,
                                    ld_global_i32(trace_compute_fmt_ptr, op_idx),
                                    op_idx,
                                )
                        else:
                            if const_expr(tracing):
                                _tc = trace_start()
                            dispatch_compute(
                                _handler,
                                page_ptr,
                                tile_0,
                                tile_1,
                                tile_2,
                                tile_3,
                                _config,
                            )
                            if const_expr(tracing):
                                _mma_lane = end_event_dynamic_raw_1(
                                    _tc,
                                    _trace_buf,
                                    Int32(trace_row_stride),
                                    _mma_lane,
                                    ld_global_i32(trace_compute_fmt_ptr, op_idx),
                                    op_idx,
                                )
                        if (
                            _range_axis >= Int32(0)
                            and warp_id == Int32(0)
                            and lane_id == Int32(0)
                        ):
                            if _cached_signal_count > Int32(0):
                                if _cached_signal_count == Int32(1):
                                    _sig_barrier = ld_global_i32(
                                        signal_meta_ptr,
                                        _current_meta_idx * Int32(max_signal_formulas),
                                    )
                                    if _sig_barrier >= Int32(0):
                                        global_barrier_signal_gpu(barriers_ptr, _sig_barrier)
                                else:
                                    signal_barriers(
                                        signal_meta_ptr,
                                        _current_meta_idx,
                                        _cached_signal_count,
                                        barriers_ptr,
                                    )
                        _range_offset = _range_offset + Int32(1)
                        _range_pos = _range_pos + _range_stride

                    if _range_axis == Int32(0):
                        tile_0 = _tile_01 & Int32(65535)
                    if _range_axis == Int32(1):
                        tile_1 = (_tile_01 >> Int32(16)) & Int32(65535)
                    if _range_axis == Int32(2):
                        tile_2 = _tile_23 & Int32(65535)
                    if _range_axis == Int32(3):
                        tile_3 = (_tile_23 >> Int32(16)) & Int32(65535)

                    if const_expr(sync_compute_warps_after_tile):
                        named_barrier_sync(
                            Int32(1),
                            Int32(num_compute_threads),
                        )

                    if (
                        _range_axis < Int32(0)
                        and warp_id == Int32(0)
                        and lane_id == Int32(0)
                    ):
                        if _cached_signal_count > Int32(0):
                            if _cached_signal_count == Int32(1):
                                _sig_barrier = ld_global_i32(
                                    signal_meta_ptr,
                                    _barrier_meta_idx * Int32(max_signal_formulas),
                                )
                                if _sig_barrier >= Int32(0):
                                    global_barrier_signal_gpu(barriers_ptr, _sig_barrier)
                            else:
                                signal_barriers(
                                    signal_meta_ptr,
                                    _barrier_meta_idx,
                                    _cached_signal_count,
                                    barriers_ptr,
                                )
                named_barrier_sync(Int32(1), Int32(num_compute_threads))

            if const_expr(tracing):
                finish_lane_dynamic_raw(_trace_buf, _mma_lane)

        return _kernel_loop_compute_only

    def _build_kernel(
        self,
        kernel_loop_fn,
        dispatch_load,
        dispatch_compute,
        dispatch_store,
        signal_barriers,
        get_page_ptr_fn,
        num_sms,
        threads_per_block,
        smem_size,
        num_pages,
        num_slots,
        iq_offset,
        flags_offset,
        ring_state_offset,
        phase_tensor_names,
        phase_tma_names,
        phase_uses_handler_local_idx,
        phase_uses_runtime_transport_selector,
        phase_uses_desc_slot_selector,
        extra_exec_globals=None,
    ):
        """Build the PersistentKernel via source transformation."""
        all_canonical = self._backend.all_canonical(self)
        tensor_sig = self._signature_suffix(all_canonical)

        tma_registry = self._tma_registry
        all_tma_canonical = self._collect_phase_unique_names(phase_tma_names)
        tma_sig = self._signature_suffix(all_tma_canonical)

        peer_tma_registry = self._peer_tma_registry
        peer_tma_sig = ""

        dispatch_extra_params = self._dispatch_extra_params_by_phase(
            phase_tensor_names,
            phase_tma_names,
            phase_uses_runtime_transport_selector,
            phase_uses_desc_slot_selector,
        )
        fn_source = self._build_kernel_loop_source(
            kernel_loop_fn,
            tensor_sig=tensor_sig,
            tma_sig=tma_sig,
            peer_tma_sig=peer_tma_sig,
            has_communicate=bool(peer_tma_registry.has_peer_tma),
            tracing=bool(self.config.tracing),
            phase_uses_handler_local_idx=phase_uses_handler_local_idx,
            dispatch_extra_params=dispatch_extra_params,
        )
        if extra_exec_globals and extra_exec_globals.get("has_page_free_ops", False):
            fn_source = self._enable_page_free_ring_source(fn_source)

        exec_globals = self._build_kernel_exec_globals(
            dispatch_load=dispatch_load,
            dispatch_compute=dispatch_compute,
            dispatch_store=dispatch_store,
            signal_barriers=signal_barriers,
            get_page_ptr_fn=get_page_ptr_fn,
            num_pages=num_pages,
            num_slots=num_slots,
            iq_offset=iq_offset,
            flags_offset=flags_offset,
            ring_state_offset=ring_state_offset,
            extra_exec_globals=extra_exec_globals,
        )

        exec_generated_source(fn_source, "kernel_loop", exec_globals)
        kernel_loop = exec_globals["_kernel_loop"]
        tma_components = self._build_tma_kernel_components(tma_registry, peer_tma_registry)
        pk_source = self._build_persistent_kernel_source(
            num_sms=num_sms,
            threads_per_block=threads_per_block,
            smem_size=smem_size,
            tensor_sig=tensor_sig,
            kernel_tma_sig=tma_sig,
            tma_components=tma_components,
            has_communicate=bool(peer_tma_registry.has_peer_tma),
            tracing=bool(self.config.tracing),
            phase_uses_handler_local_idx=phase_uses_handler_local_idx,
        )
        pk_globals = self._build_persistent_kernel_globals(
            tma_registry,
            peer_tma_registry,
            kernel_loop,
        )
        exec_generated_source(pk_source, "persistent_kernel", pk_globals)
        return pk_globals["PersistentKernel"]()

    @staticmethod
    def _replace_required(source: str, old: str, new: str) -> str:
        if old not in source:
            def _shift_left(text: str) -> str:
                lines = text.splitlines(True)
                return "".join(
                    line[8:] if line.startswith("        ") else line
                    for line in lines
                )

            old_shifted = _shift_left(old)
            new_shifted = _shift_left(new)
            if old_shifted in source:
                return source.replace(old_shifted, new_shifted, 1)
            raise RuntimeError("Failed to patch generated page-free ring source")
        return source.replace(old, new, 1)

    def _enable_page_free_ring_source(self, source: str) -> str:
        """Specialize the generated ring loop for ops that do not own smem pages.

        CUTLASS DSL still lowers statically-false Python branches into SCF in
        some cases, so the page-free path is spliced into the generated source
        only for kernels that actually need it.
        """
        source = self._replace_required(
            source,
            "                _temp_instr = iq_base\n",
            "                data_release_idx_ptr = flags_ptr + FLAG_DATA_RELEASE_IDX\n"
            "                data_produce_idx_ptr = flags_ptr + FLAG_DATA_PRODUCE_IDX\n"
            "                _temp_instr = iq_base\n"
        )
        source = self._replace_required(
            source,
            "                            st_shared_i32(\n"
            "                                _p_ti + Int32(4 * _TILE_INFO_PAGE_ID),\n"
            "                                _p_slot % Int32(num_pages),\n"
            "                            )\n",
            "                            _ctrl_no_page = (\n"
            "                                (_ctrl_cached_phase_mask // Int32(1 << _INSTR_NO_SMEM_PAGE_BIT))\n"
            "                                % Int32(2)\n"
            "                            )\n"
            "                            st_shared_i32(\n"
            "                                _p_ti + Int32(4 * _TILE_INFO_PAGE_ID),\n"
            "                                Int32(-1),\n"
            "                            )\n"
            "                            if _ctrl_no_page == Int32(0):\n"
            "                                _dp = ld_shared_i32(data_produce_idx_ptr)\n"
            "                                _dr = ld_shared_acquire_cta_i32(data_release_idx_ptr)\n"
            "                                while (_dp - _dr) >= Int32(num_pages):\n"
            "                                    nanosleep(Int32(loader_idle_sleep_ns))\n"
            "                                    _dr = ld_shared_acquire_cta_i32(data_release_idx_ptr)\n"
            "                                st_shared_i32(\n"
            "                                    _p_ti + Int32(4 * _TILE_INFO_PAGE_ID),\n"
            "                                    _dp % Int32(num_pages),\n"
            "                                )\n"
            "                                st_shared_i32(data_produce_idx_ptr, _dp + Int32(1))\n",
        )
        source = self._replace_required(
            source,
            "                load_done_ptr = flags_ptr + FLAG_LOAD_DONE\n"
            "                while _sw_done == Int32(0):\n",
            "                load_done_ptr = flags_ptr + FLAG_LOAD_DONE\n"
            "                data_release_idx_ptr = flags_ptr + FLAG_DATA_RELEASE_IDX\n"
            "                while _sw_done == Int32(0):\n",
        )
        source = self._replace_required(
            source,
            "                            st_shared_i32(store_idx_ptr, _s_idx + Int32(1))\n",
            "                            if _ds_page >= Int32(0):\n"
            "                                _store_data_release_idx = ld_shared_i32(data_release_idx_ptr) + Int32(1)\n"
            "                                st_shared_release_cta_i32(\n"
            "                                    data_release_idx_ptr,\n"
            "                                    _store_data_release_idx,\n"
            "                                )\n"
            "                            st_shared_i32(store_idx_ptr, _s_idx + Int32(1))\n",
        )
        return source

    @property
    def num_peer_barriers(self) -> int:
        """Number of peer barriers needed for cross-GPU signaling."""
        total = 0
        for op in self.ops:
            if self._op_has_peer_barriers(op):
                total += op.total_tiles
        return total

    def _create_kernel(self):
        """Create the persistent warp-specialized kernel with ring buffer pages.

        Architecture:
        - DMA warp (last warp): non-blocking polling loop for fetch/load/store
        - MMA warps (all other warps): compute only, synced via mbarrier

        Ring buffer page management:
        - DMA polls in a three-phase loop: try-store, try-produce, check-done
        - All DMA barrier checks are non-blocking (no spin loops in inline asm)
        - Phase alternates 0/1 with each use; tracked via produce/consume counters

        Two mbarrier arrays:
        - work_notify[slot]:  DMA->MMA data ready (1 arrive per tile via TMA)
        - compute_done[slot]: MMA->DMA (num_mma_warps arrivals)
        """
        kernel_cfg = self._kernel_static_config()
        runtime = self._kernel_runtime_components(kernel_cfg)
        if self._use_compute_only_replay():
            kernel_loop_fn = self._build_compute_only_kernel_loop(kernel_cfg, runtime)
        else:
            kernel_loop_fn = self._build_ring_kernel_loop(kernel_cfg, runtime)

        return self._build_kernel(
            kernel_loop_fn,
            runtime["dispatch_load"],
            runtime["dispatch_compute"],
            runtime["dispatch_store"],
            runtime["signal_barriers"],
            runtime["_get_page_ptr"],
            kernel_cfg["num_sms"],
            kernel_cfg["actual_threads_per_block"],
            kernel_cfg["smem_size"],
            kernel_cfg["num_pages"],
            kernel_cfg["num_slots"],
            kernel_cfg["iq_offset"],
            kernel_cfg["flags_offset"],
            kernel_cfg["ring_state_offset"],
            runtime["phase_tensor_names"],
            runtime["phase_tma_names"],
            runtime["phase_uses_handler_local_idx"],
            runtime["phase_uses_runtime_transport_selector"],
            runtime["phase_uses_desc_slot_selector"],
            extra_exec_globals=self._kernel_extra_exec_globals(kernel_cfg, runtime),
        )

    def _make_cache_key(self) -> Tuple:
        """Create a cache key for the compiled kernel.

        The key includes all parameters that affect kernel compilation:
        - Op classes, their static dimensions, and tensor dtypes/strides
        - Config parameters (threads, pages, etc.)
        - Backward flag

        Tile counts, barrier formulas, and instruction streams now live in
        runtime tensors (`op_meta`, instructions, wait_info, signal metadata).
        They no longer affect the emitted kernel body directly, so
        batch-dynamic scheduling can reuse the same compiled kernel while
        rebuilding only runtime metadata.
        """
        handler_keys = tuple(
            spec.compile_key for spec in self._backend_ir.handler_specs
        )
        phase_runtime_keys = tuple(
            (
                phase,
                self._phase_local_transport_position_widths[phase],
                self._phase_local_desc_slot_widths[phase],
                self._phase_local_transport_position_tensors[phase] is not None,
                self._phase_local_desc_slot_tensors[phase] is not None,
            )
            for phase in PHASE_NAMES
        )
        signal_shape_key = (
            self._max_wait_formulas,
            self._max_compute_wait_formulas,
            self._max_signal_formulas,
        )

        config_key = (
            self.config.num_sms,
            self.config.threads_per_block,
            self.config.page_size,
            self.config.num_pages,
            self.config.tracing,
            self.config.dma_reg_count,
            self.config.mma_reg_count,
            self.config.num_devices,
            self._sync_compute_warps_after_tile(),
            self.config.loader_idle_sleep_ns,
            self.config.relaxed_global_barriers,
            self.config.global_barrier_sleep_ns,
            self.config.opt_level,
        )

        # TMA descriptors are created at launch time from runtime tensors, but
        # the generated kernel signature still has one argument per descriptor.
        # Keep descriptor order/count in the key, not only unique structures.
        #
        # Different allocations with the same shape/stride layout should share
        # the same compiled kernel. Runtime tensor handles still flow through
        # launch_state and TMA descriptors are rebuilt per launch.
        local_tma_key = tuple(
            (
                desc.direction,
                tuple(desc.tile_shape),
                getattr(desc.dtype, "__name__", str(desc.dtype)),
                desc.smem_layout_src,
                tuple(desc.dim_perm),
            )
            for desc in self._tma_registry.descriptors
        )
        peer_tma_key = tuple(
            (
                desc.direction,
                tuple(desc.tile_shape),
                getattr(desc.dtype, "__name__", str(desc.dtype)),
                desc.smem_layout_src,
            )
            for desc in self._peer_tma_registry.descriptors
        )

        return (
            handler_keys,
            phase_runtime_keys,
            signal_shape_key,
            config_key,
            local_tma_key,
            peer_tma_key,
        )

    def compile(self) -> None:
        """Compile the kernel without running it.

        Triggers JIT compilation so that subsequent run() calls have no
        compilation overhead. Safe to call multiple times (no-op after first).

        Uses a class-level cache to avoid recompilation when multiple Megakernel
        instances have the same configuration (same ops, static_dims, config).
        """
        # All are idempotent (check for None internally)
        self._prepare_tensors()
        self._prepare_cute_tensors()
        cache_key = self._make_cache_key()
        cache_key_has_tma = self._cache_key_has_tma(cache_key)
        if (
            self._compiled_kernel is None
            and cache_key_has_tma
            and cache_key not in Megakernel._compiled_kernel_cache
        ):
            Megakernel._unload_active_tma_modules()
            Megakernel._drop_cached_tma_kernels()
        self._prepare_tma_tensors()
        self._prepare_peer_tma_tensors()
        if self._compiled_kernel is not None and self._compiled_kernel_module_is_unloaded(
            self._compiled_kernel
        ):
            self._compiled_kernel = None
        if self._compiled_kernel is None:
            # Check class-level cache first
            if cache_key in Megakernel._compiled_kernel_cache:
                self._compiled_kernel = Megakernel._compiled_kernel_cache[cache_key]
                if self._eager_load_compiled_kernel_for_current_device():
                    torch.cuda.synchronize()
                return
            self._validate_requirements()
            from cutedsl_trace.config import set_tracing_enabled

            set_tracing_enabled(self.config.tracing)

            tracing_str = " [traced]" if self.config.tracing else ""
            tma_str = " [TMA]" if self._tma_registry.has_tma else ""
            peer_str = " [peer]" if self._peer_tma_registry.has_peer_tma else ""
            print(
                f"Compiling persistent kernel ({tracing_str}{tma_str}{peer_str} [op-phase-policy]) for "
                f"{len(self.ops)} ops, "
                f"{self.total_tiles} tiles, {self.num_sms} SMs, "
                f"{self.smem_size // 1024}KB smem..."
            )

            # Install noinline support; op phase policy decides which wrappers use it.
            _pipeline_patch = None
            from . import noinline as noinline_mod
            noinline_mod.install()
            if self.config.opt_level != 3:
                from cutlass.cutlass_dsl.cutlass import CutlassBaseDSL
                _orig_preprocess = CutlassBaseDSL.preprocess_pipeline
                target_level = self.config.opt_level

                def _patched_preprocess(self_dsl, pipeline, arch):
                    """Lower the generated pipeline with the requested LLVM opt level."""
                    pipeline = _orig_preprocess(self_dsl, pipeline, arch)
                    return pipeline.replace("opt-level=3", f"opt-level={target_level}")

                CutlassBaseDSL.preprocess_pipeline = _patched_preprocess
                _pipeline_patch = _orig_preprocess

            try:
                self._compiled_kernel = self._create_kernel()

                # Force upfront JIT compilation with cute.compile()
                # This avoids lazy compilation on first run()
                import cuda.bindings.driver as cuda

                torch_stream = torch.cuda.current_stream()
                cu_stream = cuda.CUstream(torch_stream.cuda_stream)

                launch_state = self._build_launch_state()
                compile_args = [
                    launch_state.instructions_ptr,
                    launch_state.barriers_ptr,
                    launch_state.op_configs_ptr,
                    launch_state.op_meta_ptr,
                    *self._phase_local_idx_launch_args(launch_state),
                    *self._phase_local_transport_position_launch_args(launch_state),
                    *self._phase_local_desc_slot_launch_args(launch_state),
                    launch_state.signal_meta_ptr,
                ]
                if self._peer_tma_registry.has_peer_tma:
                    compile_args.append(launch_state.peer_signal_ptr)
                compile_args.extend(
                    [
                        launch_state.wait_info_ptr,
                        launch_state.compute_wait_info_ptr,
                        self._num_instructions_i32,
                    ]
                )
                if self.config.tracing:
                    compile_args.append(launch_state.trace_buffer_ptr)
                compile_args.extend(
                    [
                        *launch_state.cute_tensors,
                        launch_state.local_tma_desc_pool_ptr,
                        launch_state.peer_tma_desc_pool_ptr,
                        *launch_state.tma_tensor_args,
                        *launch_state.peer_tma_tensor_args,
                        self._needs_tma_desc_pool_init,
                        cu_stream,
                    ]
                )
                self._compiled_kernel = cute.compile(self._compiled_kernel, *compile_args)
                if self._eager_load_compiled_kernel_for_current_device():
                    torch.cuda.synchronize()
            finally:
                noinline_mod.uninstall()
                if _pipeline_patch is not None:
                    CutlassBaseDSL.preprocess_pipeline = _pipeline_patch

            # Store in class-level cache
            Megakernel._compiled_kernel_cache[cache_key] = self._compiled_kernel
            if cache_key_has_tma:
                Megakernel._track_active_tma_module(self._compiled_kernel)
            print("Compilation complete.")

    def _eager_load_compiled_kernel_for_current_device(self) -> bool:
        """Resolve a compiled CUTLASS CUDA library on the active device.

        CUTLASS's default CUDA executor lazily loads device symbols on first
        launch and does so for every visible device. In this stack, fresh
        attention-style megakernels can fault when that lazy load happens after
        a prior executed fresh compile. Resolve the module once for the active
        CUDA device and cache the resulting executor on the compiled function.
        """
        compiled = self._compiled_kernel
        if compiled is None or not hasattr(compiled, "_get_cuda_init_and_load"):
            return False

        jit_module = getattr(compiled, "jit_module", None)
        if jit_module is not None and not jit_module.is_unloaded():
            if getattr(compiled, "_default_executor", None) is None:
                from cutlass.base_dsl.jit_executor import JitExecutor

                compiled._default_executor = JitExecutor(
                    jit_module, None, compiled.jit_time_profiling
                )
            return False

        from cuda.bindings import runtime as cuda_runtime
        from cutlass.base_dsl.jit_executor import JitExecutor
        from cutlass.cutlass_dsl.cuda_jit_executor import (
            CudaDialectJitModule,
            checkCudaErrors,
        )

        cuda_init, cuda_load_to_device = compiled._get_cuda_init_and_load()
        library = ctypes.c_void_p()
        pointer_to_library = ctypes.pointer(library)
        pointer_to_pointer_to_library = ctypes.pointer(pointer_to_library)
        err = ctypes.c_int32(0)
        pointer_to_err = ctypes.pointer(err)

        cuda_init_args = [pointer_to_pointer_to_library, pointer_to_err]
        packed_init_args = (ctypes.c_void_p * len(cuda_init_args))()
        for i, arg in enumerate(cuda_init_args):
            packed_init_args[i] = ctypes.cast(arg, ctypes.c_void_p)
        cuda_init(packed_init_args)
        checkCudaErrors((cuda_runtime.cudaError_t(err.value),))

        device_id = ctypes.c_int32(torch.cuda.current_device())
        pointer_to_device_id = ctypes.pointer(device_id)
        cuda_load_args = [
            pointer_to_pointer_to_library,
            pointer_to_device_id,
            pointer_to_err,
        ]
        packed_load_args = (ctypes.c_void_p * len(cuda_load_args))()
        for i, arg in enumerate(cuda_load_args):
            packed_load_args[i] = ctypes.cast(arg, ctypes.c_void_p)
        cuda_load_to_device(packed_load_args)
        checkCudaErrors((cuda_runtime.cudaError_t(err.value),))

        compiled.jit_module = CudaDialectJitModule(
            compiled.engine,
            compiled.capi_func,
            compiled.args_spec,
            [cuda_runtime.cudaLibrary_t(library.value)],
        )
        compiled._default_executor = JitExecutor(
            compiled.jit_module, None, compiled.jit_time_profiling
        )
        return True

    def _validate_tensors(self) -> None:
        """Validate tensors match op requirements before kernel launch.

        Checks that tensors haven't been resized or reallocated since
        schedule() was called.
        """
        for op in self.ops:
            if not op.tensor_metas:
                continue
            for name, meta in op.tensor_metas.items():
                ref = op.tensor_refs.get(name)
                if ref is None:
                    continue
                if ref.data_ptr() != meta.data_ptr:
                    raise RuntimeError(
                        f"Op {op.op_cls.__name__}: tensor '{name}' data_ptr changed "
                        f"since schedule() (was 0x{meta.data_ptr:x}, "
                        f"now 0x{ref.data_ptr():x}). Re-schedule the op."
                    )
                if tuple(ref.shape) != meta.shape:
                    raise RuntimeError(
                        f"Op {op.op_cls.__name__}: tensor '{name}' shape changed "
                        f"since schedule() (was {meta.shape}, now {tuple(ref.shape)})."
                    )

    def _build_launch_state(self) -> _LaunchState:
        """Build stable launch arguments shared by run() and bench_spec()."""
        trace_buffer_ptr = Int64(0)
        if self.config.tracing:
            from .tracing import ensure_device_trace_buffer

            ensure_device_trace_buffer(self._tracing_state)
            trace_buffer_ptr = Int64(self._tracing_state.builder._buffer.data_ptr())

        selected_tensor_names = self._backend.all_canonical(self)
        selected_cute_tensors = []
        if self._cute_tensors:
            cute_tensor_by_name = {
                canonical_name: cute_tensor
                for canonical_name, cute_tensor in zip(
                    self._tensor_registry.canonical_names, self._cute_tensors
                )
            }
            selected_cute_tensors = [
                cute_tensor_by_name[name] for name in selected_tensor_names
            ]

        phase_local_idx_ptrs = self._phase_table_ptrs(self._phase_local_idx_tensors)
        phase_local_transport_position_ptrs = self._phase_table_ptrs(
            self._phase_local_transport_position_tensors
        )
        phase_local_desc_slot_ptrs = self._phase_table_ptrs(
            self._phase_local_desc_slot_tensors
        )
        return _LaunchState(
            instructions_ptr=Int64(self._instructions_tensor.data_ptr()),
            barriers_ptr=Int64(self._barriers_tensor.data_ptr()),
            op_configs_ptr=Int64(self._op_configs_tensor.data_ptr()),
            wait_info_ptr=Int64(self._wait_info.data_ptr()),
            compute_wait_info_ptr=Int64(self._compute_wait_info.data_ptr()),
            op_meta_ptr=Int64(self._op_metadata_tensor.data_ptr()),
            **self._phase_launch_state_kwargs(
                phase_local_idx_ptrs,
                _PHASE_LOCAL_IDX_PTR_ATTRS,
            ),
            **self._phase_launch_state_kwargs(
                phase_local_transport_position_ptrs,
                _PHASE_TRANSPORT_PTR_ATTRS,
            ),
            **self._phase_launch_state_kwargs(
                phase_local_desc_slot_ptrs,
                _PHASE_DESC_SLOT_PTR_ATTRS,
            ),
            local_tma_desc_pool_ptr=Int64(self._local_tma_desc_pool.data_ptr()) if self._local_tma_desc_pool is not None else Int64(0),
            peer_tma_desc_pool_ptr=Int64(self._peer_tma_desc_pool.data_ptr()) if self._peer_tma_desc_pool is not None else Int64(0),
            signal_meta_ptr=Int64(self._signal_metadata_tensor.data_ptr()),
            peer_signal_ptr=Int64(self._peer_signal_tensor.data_ptr()) if self._peer_signal_tensor is not None else Int64(0),
            trace_buffer_ptr=trace_buffer_ptr,
            cute_tensors=selected_cute_tensors,
            tma_tensor_args=[ct for _, ct in self._tma_cute_tensors] if self._tma_cute_tensors else [],
            peer_tma_tensor_args=(
                [ct for _, _, ct in self._peer_tma_cute_tensors]
                if self._peer_tma_cute_tensors else []
            ),
        )

    def _phase_local_idx_launch_args(self, launch_state: _LaunchState) -> List[Int64]:
        """Return phase-local index table pointers in kernel ABI order."""
        return self._phase_launch_args(launch_state, self._phase_local_idx_tensors, _PHASE_LOCAL_IDX_PTR_ATTRS)

    def _phase_local_transport_position_launch_args(self, launch_state: _LaunchState) -> List[Int64]:
        """Return per-phase local-transport selector table pointers in ABI order."""
        return self._phase_launch_args(
            launch_state,
            self._phase_local_transport_position_tensors,
            _PHASE_TRANSPORT_PTR_ATTRS,
        )

    def _phase_local_desc_slot_launch_args(self, launch_state: _LaunchState) -> List[Int64]:
        """Return per-phase desc-slot selector table pointers in ABI order."""
        return self._phase_launch_args(
            launch_state,
            self._phase_local_desc_slot_tensors,
            _PHASE_DESC_SLOT_PTR_ATTRS,
        )

    @staticmethod
    def _phase_launch_args(
        launch_state: _LaunchState,
        phase_tensors: Dict[str, Optional[torch.Tensor]],
        phase_attr_names: Dict[str, str],
    ) -> List[Int64]:
        """Return per-phase pointer arguments in ABI order for present tables."""
        args: List[Int64] = []
        for phase in phase_tensors:
            if phase_tensors[phase] is not None:
                args.append(getattr(launch_state, phase_attr_names[phase]))
        return args

    @staticmethod
    def _phase_table_ptrs(phase_tensors: Dict[str, Optional[torch.Tensor]]) -> Dict[str, Int64]:
        """Return Int64 table pointers for each phase, defaulting to null."""
        return {
            phase: Int64(phase_tensors[phase].data_ptr())
            if phase_tensors[phase] is not None else Int64(0)
            for phase in phase_tensors
        }

    @staticmethod
    def _phase_launch_state_kwargs(
        phase_ptrs: Dict[str, Int64],
        phase_attr_names: Dict[str, str],
    ) -> Dict[str, Int64]:
        """Return `_LaunchState` kwargs for one per-phase pointer family."""
        return {
            phase_attr_names[phase]: phase_ptrs[phase]
            for phase in phase_ptrs
        }

    @staticmethod
    def _phase_local_indices_for_op(
        op_phase_local_indices: Dict[str, List[int]],
        op_idx: int,
    ) -> List[int]:
        """Return phase-local indices for one op in canonical phase order."""
        return [int(op_phase_local_indices[phase][op_idx]) for phase in PHASE_NAMES]

    @staticmethod
    def _encode_phase_selector_table(
        nested_rows: List[List[Tuple[int, ...] | List[int]]],
    ) -> Tuple[int, Optional[List[int]]]:
        """Flatten ragged per-handler selector rows into a padded table."""
        if not any(len(row) > 0 for handler_rows in nested_rows for row in handler_rows):
            return 0, None
        width = max(
            1,
            max((len(row) for handler_rows in nested_rows for row in handler_rows), default=0),
        )
        flat_values: List[int] = []
        for handler_rows in nested_rows:
            for row in handler_rows:
                encoded = list(row)
                encoded.extend([-1] * (width - len(encoded)))
                flat_values.extend(encoded)
        return width, flat_values

    def _set_phase_i32_table(
        self,
        phase: str,
        enabled: bool,
        flat_values: Optional[List[int]],
        width: int,
        widths: Dict[str, int],
        tensors: Dict[str, Optional[torch.Tensor]],
    ) -> None:
        """Materialize one optional int32 per-phase table on device."""
        if enabled and flat_values is not None:
            widths[phase] = width
            tensors[phase] = torch.tensor(flat_values, dtype=torch.int32, device=self.device)
        else:
            widths[phase] = 0
            tensors[phase] = None

    @staticmethod
    def _collect_phase_unique_names(phase_names: Dict[str, List[str]]) -> List[str]:
        """Return phase-ordered unique names across all phases."""
        names: List[str] = []
        seen = set()
        for phase in PHASE_NAMES:
            for name in phase_names.get(phase, []):
                if name in seen:
                    continue
                seen.add(name)
                names.append(name)
        return names

    @staticmethod
    def _dispatch_extra_params_by_phase(
        phase_tensor_names: Dict[str, List[str]],
        phase_tma_names: Dict[str, List[str]],
        phase_uses_runtime_transport_selector: Dict[str, bool],
        phase_uses_desc_slot_selector: Dict[str, bool],
    ) -> Dict[str, str]:
        """Return extra dispatch-call parameters for each phase."""
        extra_params: Dict[str, str] = {}
        for phase in PHASE_NAMES:
            selector_param = (
                f"{phase}_local_transport_positions_ptr"
                if phase_uses_runtime_transport_selector.get(phase, False)
                else ""
            )
            desc_slot_selector_param = (
                f"{phase}_local_desc_slots_ptr"
                if phase_uses_desc_slot_selector.get(phase, False)
                else ""
            )
            extra_params[phase] = ", ".join(
                filter(
                    None,
                    [
                        selector_param,
                        desc_slot_selector_param,
                        ", ".join(phase_tensor_names.get(phase, [])),
                        ", ".join(phase_tma_names.get(phase, [])),
                    ],
                )
            )
        return extra_params

    def _cache_launch_state(self) -> None:
        """Cache stable launch arguments to avoid per-run Python overhead.

        Int64() construction costs ~0.6us each, CUstream ~2us.
        These values are stable after compile() — cache them once.
        """
        if self._launch_state is not None:
            return
        import cuda.bindings.driver as cuda

        self._launch_state = self._build_launch_state()

        # Cache CUstream for the default stream
        torch_stream = torch.cuda.current_stream()
        self._cached_cu_stream = cuda.CUstream(torch_stream.cuda_stream)
        self._cached_torch_stream_id = torch_stream.cuda_stream

    def _resolve_launch_stream(self, stream=None):
        """Convert an optional torch stream choice into the CUstream launch arg."""
        if stream is not None:
            return stream

        import cuda.bindings.driver as cuda

        torch_stream_id = torch.cuda.current_stream().cuda_stream
        if torch_stream_id == self._cached_torch_stream_id:
            return self._cached_cu_stream
        return cuda.CUstream(torch_stream_id)

    def _launch_compiled_kernel(self, launch_state: _LaunchState, stream, trace_buffer_ptr: Optional[Int64] = None) -> None:
        """Invoke the compiled kernel with a cached launch state."""
        def _launch_args(num_instructions: Int32, desc_pool_init_needed: bool) -> List[Any]:
            launch_args = [
                launch_state.instructions_ptr,
                launch_state.barriers_ptr,
                launch_state.op_configs_ptr,
                launch_state.op_meta_ptr,
                *self._phase_local_idx_launch_args(launch_state),
                *self._phase_local_transport_position_launch_args(launch_state),
                *self._phase_local_desc_slot_launch_args(launch_state),
                launch_state.signal_meta_ptr,
            ]
            if self._peer_tma_registry.has_peer_tma:
                launch_args.append(launch_state.peer_signal_ptr)
            launch_args.extend(
                [
                    launch_state.wait_info_ptr,
                    launch_state.compute_wait_info_ptr,
                    num_instructions,
                ]
            )
            if self.config.tracing:
                launch_args.append(
                    trace_buffer_ptr if trace_buffer_ptr is not None else launch_state.trace_buffer_ptr
                )
            launch_args.extend(
                [
                    *launch_state.cute_tensors,
                    launch_state.local_tma_desc_pool_ptr,
                    launch_state.peer_tma_desc_pool_ptr,
                    *launch_state.tma_tensor_args,
                    *launch_state.peer_tma_tensor_args,
                    desc_pool_init_needed,
                ]
            )
            launch_args.append(stream)
            return launch_args

        needs_desc_pool_init = self._needs_tma_desc_pool_init and (
            self._tma_registry.has_tma or self._peer_tma_registry.has_peer_tma
        )
        if needs_desc_pool_init:
            # The generated wrapper launches descriptor initialization and the
            # persistent kernel back-to-back. Some CUDA/CuTe stacks can still
            # race the first runtime-TMA descriptor use in fully async mode, so
            # first run a zero-instruction launch that only initializes the
            # descriptor pool, drain it, then launch the real work hot path.
            self._compiled_kernel(*_launch_args(Int32(0), True))
            _sync_tma_desc_init_stream(stream)
            self._needs_tma_desc_pool_init = False

        self._compiled_kernel(*_launch_args(self._num_instructions_i32, False))
        self._needs_tma_desc_pool_init = False

    def wait(self) -> None:
        """Drain one outstanding async launch before reusing kernel state.

        One ``Megakernel`` instance reuses barrier tensors, runtime metadata,
        and cached launch arguments across invocations. On this stack, batched
        async relaunches of the same instance have proven unstable unless the
        previous launch is drained before host-side setup mutates visible
        tensors or before the kernel reuses its internal barrier state.
        """
        if self._has_pending_async_launch:
            torch.cuda.synchronize()
            self._has_pending_async_launch = False

    def prepare_run(self, stream=None, validate: bool = True, reset_barriers: bool = True):
        """Prepare one launch without actually invoking the kernel.

        This separates host/runtime setup from the launch itself so callers can
        measure only the hot launch path when needed.

        Args:
            stream: CUDA stream (optional). If None, uses current stream.
            validate: If True, validate tensors haven't been reallocated since
                schedule().
            reset_barriers: If True, zero the barrier tensor for this run.

        Returns:
            Tuple of ``(launch_state, resolved_stream, trace_buffer_ptr)``.
        """
        self.wait()

        if validate:
            self._validate_tensors()

        # Always call compile() so tensors/TMA state are prepared even when
        # the compiled kernel was injected externally.
        self.compile()
        self._cache_launch_state()
        launch_state = self._launch_state
        resolved_stream = self._resolve_launch_stream(stream)

        if reset_barriers:
            self._barriers_tensor.zero_()

        if self.config.tracing:
            from .tracing import ensure_device_trace_buffer

            ensure_device_trace_buffer(self._tracing_state)
            self._tracing_state.builder.reset()
            trace_buffer_ptr = Int64(self._tracing_state.builder._buffer.data_ptr())
        else:
            trace_buffer_ptr = launch_state.trace_buffer_ptr

        return launch_state, resolved_stream, trace_buffer_ptr

    def launch_prepared(
        self,
        launch_state: _LaunchState,
        stream,
        trace_buffer_ptr: Optional[Int64] = None,
        sync: bool = True,
    ) -> None:
        """Launch a kernel after ``prepare_run()`` has already been done."""
        self._launch_compiled_kernel(
            launch_state,
            stream,
            trace_buffer_ptr=trace_buffer_ptr,
        )

        if sync:
            torch.cuda.synchronize()
            self._has_pending_async_launch = False
        else:
            self._has_pending_async_launch = True

    def run(self, stream=None, sync: bool = True, validate: bool = True) -> None:
        """Run the persistent megakernel.

        Args:
            stream: CUDA stream (optional). If None, uses current stream.
            sync: If True (default), synchronize after launch. Set to False
                for benchmarking or when managing synchronization externally.
            validate: If True (default), validate tensors haven't been
                reallocated since schedule(). Set to False for production
                inner loops where tensors are known to be stable.
        """
        launch_state, resolved_stream, trace_buffer_ptr = self.prepare_run(
            stream=stream,
            validate=validate,
            reset_barriers=True,
        )
        self.launch_prepared(
            launch_state,
            resolved_stream,
            trace_buffer_ptr=trace_buffer_ptr,
            sync=sync,
        )

    def bench_spec(self, setup_fn=None, keep_alive=None):
        """Create a KernelBenchSpec for raw GPU kernel timing.

        Returns a spec that can be passed to the benchmark framework for
        precise kernel-only timing via per-iteration CUDA event timing.

        The persistent megakernel requires barrier resets between invocations,
        so CUDA graph replay cannot be used. Each timed iteration calls
        launch_fn() which resets barriers and launches the kernel.

        Args:
            setup_fn: Optional callable invoked before each timed iteration
                to reset input tensors or other state.
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
        from machete.utils.benchmark_utils import KernelBenchSpec

        self.compile()
        self._cache_launch_state()
        bench_stream = torch.cuda.Stream()
        cu_stream = cuda.CUstream(bench_stream.cuda_stream)
        barriers_tensor = self._barriers_tensor

        def _setup():
            """Reset benchmark-visible runtime state before each launch."""
            self.wait()
            with torch.cuda.stream(bench_stream):
                if setup_fn is not None:
                    setup_fn()
                barriers_tensor.zero_()

        def _launch():
            launch_state, _, trace_buffer_ptr = self.prepare_run(
                stream=cu_stream,
                validate=False,
                reset_barriers=False,
            )
            self.launch_prepared(
                launch_state,
                cu_stream,
                trace_buffer_ptr=trace_buffer_ptr,
                sync=False,
            )

        return KernelBenchSpec(
            launch_fn=_launch,
            setup_fn=_setup,
            stream=(bench_stream, cu_stream),
            _keep_alive=(self, keep_alive),  # prevent GC from freeing GPU memory
        )

    def __repr__(self) -> str:
        """Return a compact debug summary of the scheduled megakernel."""
        op_names = ", ".join(f"{op.op_cls.__name__}({op.total_tiles})" for op in self.ops)
        return (
            f"Megakernel(\n"
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
    **kwargs,
) -> Megakernel:
    """Create a megakernel for the given operations.

    Args:
        ops: List of scheduled operations
        num_sms: Number of SMs (default: all available)
        **kwargs: Additional arguments passed to MegakernelConfig

    Returns:
        Configured Megakernel instance
    """
    config = MegakernelConfig(num_sms=num_sms, **kwargs)
    return Megakernel(ops, config=config)


__all__ = [
    "MegakernelConfig",
    "Megakernel",
    "create_megakernel",
]
