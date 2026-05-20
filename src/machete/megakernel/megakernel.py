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
from .codegen_support import (
    build_kernel_exec_globals,
    build_persistent_kernel_globals,
    enable_page_free_ring_source,
)
from .source_fragments import build_kernel_loop_source, build_persistent_kernel_source
from .tma_codegen import build_tma_kernel_components
from .runtime_components import (
    build_kernel_extra_exec_globals,
    build_kernel_runtime_components,
    build_kernel_static_config,
)
from .replay_loops import build_compute_only_kernel_loop, build_ring_kernel_loop
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

            unit_range_cache: Dict[int, bool] = {}

            def _op_owns_unit_range(instr: TileInstruction) -> bool:
                if instr.op_idx == TileInstruction.END_MARKER or instr.range_axis >= 0:
                    return False
                op = self.ops[instr.op_idx]
                tile_rank = len(op.tile_counts)
                if tile_rank <= 0 or tile_rank >= MAX_TILE_DIMS:
                    return False
                if instr.op_idx not in unit_range_cache:
                    unit_range_cache[instr.op_idx] = not self._framework_expands_range(
                        instr.op_idx,
                        tile_rank,
                    )
                return unit_range_cache[instr.op_idx]

            if not any(op.static_dims.get("disable_instruction_coalescing", False) for op in self.ops):
                coalesced_instructions = self._builder.coalesce_pipeline_instructions(
                    instructions,
                    num_blocks=self.num_sms,
                    framework_expands_predicate=_framework_expands_instruction_range,
                )
                if len(coalesced_instructions) <= len(instructions):
                    instructions = coalesced_instructions
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
                    if _op_owns_unit_range(instr):
                        op = self.ops[instr.op_idx]
                        tile_rank = len(op.tile_counts)
                        range_axis = tile_rank - 1
                        tiles = list(instr.tiles) + [0] * (MAX_TILE_DIMS - len(instr.tiles))
                        tiles[tile_rank] = tiles[range_axis] + 1
                        return TileInstruction(instr.op_idx, tuple(tiles[:MAX_TILE_DIMS]))
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

    def _build_tma_kernel_components(self, tma_registry, peer_tma_registry) -> Dict[str, Any]:
        """Assemble TMA-specific signature fragments and descriptor setup code."""
        return build_tma_kernel_components(
            tma_registry,
            peer_tma_registry,
            signature_suffix=self._signature_suffix,
        )

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

    def _kernel_static_config(self) -> Dict[str, Any]:
        """Collect the compile-time constants used to build the persistent kernel."""
        return build_kernel_static_config(
            self,
            use_compute_only_replay=self._use_compute_only_replay(),
            num_dma_warps=NUM_DMA_WARPS,
            mbarrier_stride=NPageLayout._MBARRIER_SIZE,
            tile_info_bytes=NPageLayout._TILE_INFO_SIZE,
        )

    def _kernel_runtime_components(self, kernel_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Build reusable runtime helpers needed by `_create_kernel`."""
        return build_kernel_runtime_components(
            self,
            kernel_cfg,
            self._op_meta_exec_globals(),
        )

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
            "_OP_META_STRIDE": _OP_META_STRIDE,
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
        return build_kernel_extra_exec_globals(
            self,
            kernel_cfg,
            runtime,
            op_meta=self._op_meta_exec_globals(),
            tile_info=self._tile_info_exec_globals(),
            sync_compute_warps_after_tile=self._sync_compute_warps_after_tile(),
            min_idle_regs=MIN_IDLE_REGS,
            op_phase_load=_OP_PHASE_LOAD,
            op_phase_store=_OP_PHASE_STORE,
            op_phase_communicate=_OP_PHASE_COMMUNICATE,
        )

    def _build_ring_kernel_loop(self, kernel_cfg: Dict[str, Any], runtime: Dict[str, Any]):
        """Build the warp-specialized ring-buffer loop used by the persistent kernel."""
        return build_ring_kernel_loop(self, kernel_cfg, runtime)

    def _build_compute_only_kernel_loop(self, kernel_cfg, runtime):
        """Build a lighter replay loop for graphs with compute phases only."""
        return build_compute_only_kernel_loop(self, kernel_cfg, runtime)

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
        fn_source = build_kernel_loop_source(
            kernel_loop_fn,
            tensor_sig=tensor_sig,
            tma_sig=tma_sig,
            peer_tma_sig=peer_tma_sig,
            has_communicate=bool(peer_tma_registry.has_peer_tma),
            tracing=bool(self.config.tracing),
            local_idx_tensors=self._phase_local_idx_tensors,
            transport_position_tensors=self._phase_local_transport_position_tensors,
            desc_slot_tensors=self._phase_local_desc_slot_tensors,
            dispatch_extra_params=dispatch_extra_params,
        )
        if extra_exec_globals and extra_exec_globals.get("has_page_free_ops", False):
            fn_source = enable_page_free_ring_source(fn_source)

        exec_globals = build_kernel_exec_globals(
            tracing=bool(self.config.tracing),
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
        pk_source = build_persistent_kernel_source(
            num_sms=num_sms,
            threads_per_block=threads_per_block,
            smem_size=smem_size,
            tensor_sig=tensor_sig,
            kernel_tma_sig=tma_sig,
            tma_components=tma_components,
            has_communicate=bool(peer_tma_registry.has_peer_tma),
            tracing=bool(self.config.tracing),
            local_idx_tensors=self._phase_local_idx_tensors,
            transport_position_tensors=self._phase_local_transport_position_tensors,
            desc_slot_tensors=self._phase_local_desc_slot_tensors,
        )
        pk_globals = build_persistent_kernel_globals(
            tma_registry,
            peer_tma_registry,
            kernel_loop,
            _sync_tma_desc_init_stream,
        )
        exec_generated_source(pk_source, "persistent_kernel", pk_globals)
        return pk_globals["PersistentKernel"]()

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
            self.config.threads_per_block,
            self.config.num_sms,
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
        if hasattr(self._compiled_kernel, "num_sms"):
            self._compiled_kernel.num_sms = self.num_sms
        if hasattr(self._compiled_kernel, "threads_per_block"):
            self._compiled_kernel.threads_per_block = self.config.threads_per_block
        if hasattr(self._compiled_kernel, "smem_size"):
            self._compiled_kernel.smem_size = self.smem_size

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
