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

from dataclasses import dataclass
import ctypes
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple
import torch

import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr, range_constexpr

from .ops import (
    MAX_TILE_DIMS,
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
    BarrierFormula,
    InstructionStreamBuilder,
    TileInstruction,
    TileScheduler,
)
from .compile import compile_phase, exec_generated_source
from .backend import build_handler_backend_ir, HandlerBackend
from .interpreter import (
    global_barrier_signal,
    global_barrier_signal_gpu,
    global_barrier_wait,
    load_instruction_to_smem,
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
    NPageLayout,
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
)


# =============================================================================
# Megakernel Configuration
# =============================================================================

# Three DMA warps: controller (fetch + barrier wait), loader (TMA dispatch), store (S→G).
NUM_DMA_WARPS = 3

# PTX minimum register count for setmaxnreg (idle warp floor).
MIN_IDLE_REGS = 24

# Per-op metadata layout (int32 entries).
_OP_META_INNER_ITERS = 0
_OP_META_NUM_WARPS = 1
_OP_META_USES_CONFIG_MASK = 2
_OP_META_STRIDE_0 = 3
_OP_META_STRIDE_1 = 4
_OP_META_STRIDE_2 = 5
_OP_META_STRIDE_3 = 6
_OP_META_STRIDE_4 = 7
_OP_META_COUNT_0 = 8
_OP_META_COUNT_1 = 9
_OP_META_COUNT_2 = 10
_OP_META_COUNT_3 = 11
_OP_META_COUNT_4 = 12
_OP_META_HANDLER_IDX = 13
_OP_META_LOAD_LOCAL_IDX = 14
_OP_META_COMPUTE_LOCAL_IDX = 15
_OP_META_STORE_LOCAL_IDX = 16
_OP_META_COMM_LOCAL_IDX = 17
_OP_META_STRIDE = 18

# Per-signal-formula metadata layout (int32 entries).
_SIGNAL_META_BASE = 0
_SIGNAL_META_GUARD_MAX = 1
_SIGNAL_META_COEFF_0 = 2
_SIGNAL_META_DIV_0 = 7
_SIGNAL_META_STRIDE = 12

# Per-slot tile-info layout in shared memory (int32 entries).
_TILE_INFO_OP_IDX = 0
_TILE_INFO_LINEAR_IDX = 1
_TILE_INFO_HANDLER_IDX = 2
_TILE_INFO_TILE_0 = 3
_TILE_INFO_TILE_1 = 4
_TILE_INFO_TILE_2 = 5
_TILE_INFO_TILE_3 = 6
_TILE_INFO_TILE_4 = 7
_TILE_INFO_INNER_ITERS = 8
_TILE_INFO_NUM_WARPS = 9
_TILE_INFO_OP_CONFIG = 10


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
        noinline: Emit op phases as noinline device functions when legal
            (compute always, load/store/communicate when TMA can be recreated
            locally). Reduces register pressure and compile-time blowup from
            repeatedly inlined phase bodies.
        opt_level: LLVM optimization level 0-3 (default: 2). noinline requires <= 2.
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
    noinline: bool = True
    opt_level: int = 2
    backend: str = "handler"

    # Multi-GPU communication
    peer_buffers: Optional[Dict[str, List[Any]]] = None
    peer_barriers: Optional[Any] = None
    device_idx: int = 0
    num_devices: int = 1

    @property
    def warps_per_block(self) -> int:
        """Number of warps per block."""
        return self.threads_per_block // 32


@dataclass
class _LaunchState:
    """Stable kernel launch arguments cached after compile()."""

    instructions_ptr: Int64
    barriers_ptr: Int64
    op_configs_ptr: Int64
    wait_info_ptr: Int64
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
    _compiled_kernel_cache: Dict[Tuple, Any] = {}

    def __init__(
        self,
        ops: List[ScheduledOp],
        config: Optional[MegakernelConfig] = None,
        device: str = "cuda",
        scheduler: Optional["TileScheduler"] = None,
    ):
        """Construct a megakernel instance and build its host-side runtime state."""
        self.ops = ops
        self.config = config or MegakernelConfig()
        self.device = device
        self._scheduler = scheduler

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
                min_pages=1,
            )
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
        self._has_pending_async_launch = False
        self._needs_tma_desc_pool_init = True
        self._op_metadata_tensor: Optional[torch.Tensor] = None
        self._signal_metadata_tensor: Optional[torch.Tensor] = None
        self._phase_local_idx_tensors: Dict[str, Optional[torch.Tensor]] = {
            "load": None,
            "compute": None,
            "store": None,
            "communicate": None,
        }
        self._phase_local_transport_position_tensors: Dict[str, Optional[torch.Tensor]] = {
            "load": None,
            "compute": None,
            "store": None,
            "communicate": None,
        }
        self._phase_local_transport_position_widths: Dict[str, int] = {
            "load": 0,
            "compute": 0,
            "store": 0,
            "communicate": 0,
        }
        self._phase_local_desc_slot_tensors: Dict[str, Optional[torch.Tensor]] = {
            "load": None,
            "compute": None,
            "store": None,
            "communicate": None,
        }
        self._phase_local_desc_slot_widths: Dict[str, int] = {
            "load": 0,
            "compute": 0,
            "store": 0,
            "communicate": 0,
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
        self._backend_ir = build_handler_backend_ir(self)
        self._backend = HandlerBackend(self._backend_ir)
        self._launch_state: Optional[_LaunchState] = None
        self._cached_cu_stream = None
        self._cached_torch_stream_id = None

        # Validate barrier formulas eagerly (catches incompatible tile sizes early)
        _ = self._builder.num_barriers

        # Trace setup
        from .tracing import setup_tracing

        self._tracing_state = None
        if self.config.tracing:
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
            self._instructions_tensor = self._builder.build_tensor(
                self.device, scheduler=self._scheduler
            )
            self._num_instructions = self._instructions_tensor.shape[0]
            self._num_instructions_i32 = Int32(self._num_instructions)
            # Pre-computed barrier indices for controller warp
            self._wait_info = self._builder.build_wait_info_tensor(instructions, self.device)

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

        if self._op_metadata_tensor is None:
            self._prepare_op_metadata_tensors()

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
        self._max_signal_formulas = max(
            1, max((len(signal) for _wait, signal in formulas.values()), default=0)
        )
        threads_per_block = self.config.threads_per_block
        num_compute_threads = threads_per_block - NUM_DMA_WARPS * 32
        default_num_mma_warps = num_compute_threads // 32

        op_meta = []
        signal_meta = []
        has_communicate = self._peer_tma_registry.has_peer_tma
        peer_signal_offsets = [] if has_communicate else None
        peer_barrier_offset = 0
        op_handler_indices = self._backend.handler_indices()
        op_phase_local_indices = {
            phase: self._backend.phase_local_indices(phase)
            for phase in ("load", "compute", "store", "communicate")
        }
        from .backend_dispatch import (
            _build_tma_runtime_layout,
            group_uses_handler_local_idx,
            group_uses_handler_local_idx_from_transport,
        )
        (
            op_phase_tma_args,
            _phase_tma_names,
            _kernel_tma_arg_names,
            _op_phase_transport_indices,
            _phase_transport_records,
            phase_local_transport_positions,
            phase_local_desc_slots,
            _handler_rebind_specs,
        ) = _build_tma_runtime_layout(self._backend, self)
        by_handler: Dict[int, List[int]] = {}
        for op_idx, handler_idx in enumerate(op_handler_indices):
            by_handler.setdefault(handler_idx, []).append(op_idx)

        phase_uses_local_idx = {}
        phase_uses_transport_selector = {}
        phase_uses_desc_slot_selector = {}
        for phase in ("load", "compute", "store", "communicate"):
            uses_local_idx = False
            uses_transport_selector = False
            uses_desc_slot_selector = False
            for handler_idx, op_indices in by_handler.items():
                handler_local_ids = [
                    op_phase_local_indices[phase][op_idx] for op_idx in op_indices
                ]
                handler_uses_local_idx = group_uses_handler_local_idx(
                    handler_local_ids=handler_local_ids,
                    op_indices=op_indices,
                    op_phase_tensor_args=[[] for _ in self.ops],
                    op_phase_tma_args=op_phase_tma_args[phase],
                )
                unique_desc_slot_groups = {
                    tuple(phase_local_desc_slots[phase][handler_idx][local_id])
                    for local_id in set(handler_local_ids)
                }
                handler_uses_transport_selector = group_uses_handler_local_idx_from_transport(
                    handler_local_ids=handler_local_ids,
                    handler_local_transport_positions=phase_local_transport_positions[
                    phase
                    ][handler_idx],
                )
                if handler_uses_transport_selector:
                    uses_transport_selector = True
                handler_uses_desc_slot_selector = len(unique_desc_slot_groups) != 1
                if handler_uses_desc_slot_selector:
                    uses_desc_slot_selector = True
                if (
                    handler_uses_local_idx
                    or handler_uses_transport_selector
                    or handler_uses_desc_slot_selector
                ):
                    uses_local_idx = True
                    break

            phase_uses_local_idx[phase] = uses_local_idx
            phase_uses_transport_selector[phase] = uses_transport_selector
            phase_uses_desc_slot_selector[phase] = uses_desc_slot_selector

        for op_idx, op in enumerate(self.ops):
            kernel_config = {"threads_per_row": num_compute_threads}
            config = build_op_config(op, kernel_config=kernel_config)
            instance = op.op_cls(**config)
            tile_strides = self._tile_linear_strides(op.tile_counts)
            _wait_formulas, signal_formulas = formulas.get(op_idx, ([], []))
            # Keep op_config_ptr live for every phase.
            #
            # Phase wrappers may reconstruct tensors from op_config_ptr even
            # when the original op method signature does not mention it.
            # Inferring "uses config" from the raw method signature is
            # therefore not stable once wrapper generation changes. Always
            # materializing the config pointer keeps the runtime contract
            # simple and avoids null-config bugs in load/store wrappers.
            load_uses = 1
            compute_uses = 1
            store_uses = 1
            communicate_uses = 1 if has_communicate else 0
            uses_mask = (
                load_uses
                + compute_uses * 2
                + store_uses * 4
                + communicate_uses * 8
            )

            op_meta.extend(
                [
                    int(getattr(instance, "inner_iters", 1)),
                    int(getattr(instance, "num_mma_warps", default_num_mma_warps)),
                    uses_mask,
                    *tile_strides,
                    *tuple(op.tile_counts) + (1,) * (MAX_TILE_DIMS - len(op.tile_counts)),
                    int(op_handler_indices[op_idx]),
                    int(op_phase_local_indices["load"][op_idx]),
                    int(op_phase_local_indices["compute"][op_idx]),
                    int(op_phase_local_indices["store"][op_idx]),
                    int(op_phase_local_indices["communicate"][op_idx]),
                ]
            )

            for signal_idx in range(self._max_signal_formulas):
                if signal_idx < len(signal_formulas):
                    formula = signal_formulas[signal_idx]
                    signal_meta.extend(
                        [
                            formula.base,
                            formula.guard_max,
                            *formula.coeffs,
                            *formula.divs,
                        ]
                    )
                else:
                    signal_meta.extend([-1, BarrierFormula.NO_GUARD] + [0] * (MAX_TILE_DIMS * 2))

            if has_communicate:
                if self._op_has_peer_barriers(op):
                    peer_signal_offsets.append(peer_barrier_offset)
                    peer_barrier_offset += op.total_tiles
                else:
                    peer_signal_offsets.append(-1)

        self._op_metadata_tensor = torch.tensor(
            op_meta, dtype=torch.int32, device=self.device
        )
        self._signal_metadata_tensor = torch.tensor(
            signal_meta, dtype=torch.int32, device=self.device
        )
        for phase in ("load", "compute", "store", "communicate"):
            if phase_uses_local_idx.get(phase, False):
                self._phase_local_idx_tensors[phase] = torch.tensor(
                    op_phase_local_indices[phase], dtype=torch.int32, device=self.device
                )
            else:
                self._phase_local_idx_tensors[phase] = None

            local_transport_positions = phase_local_transport_positions[phase]
            if (
                phase_uses_transport_selector[phase]
                and any(
                    len(positions) > 0
                    for handler_positions in local_transport_positions
                    for positions in handler_positions
                )
            ):
                width = max(
                    1,
                    max(
                        (len(positions) for handler_positions in local_transport_positions for positions in handler_positions),
                        default=0,
                    ),
                )
                self._phase_local_transport_position_widths[phase] = width
                flat_positions: List[int] = []
                for handler_positions in local_transport_positions:
                    for positions in handler_positions:
                        encoded = list(positions)
                        encoded.extend([-1] * (width - len(encoded)))
                        flat_positions.extend(encoded)
                self._phase_local_transport_position_tensors[phase] = torch.tensor(
                    flat_positions, dtype=torch.int32, device=self.device
                )
            else:
                self._phase_local_transport_position_widths[phase] = 0
                self._phase_local_transport_position_tensors[phase] = None

            local_desc_slots = phase_local_desc_slots[phase]
            if (
                phase_uses_desc_slot_selector[phase]
                and any(
                    len(slots) > 0
                    for handler_slots in local_desc_slots
                    for slots in handler_slots
                )
            ):
                width = max(
                    1,
                    max(
                        (len(slots) for handler_slots in local_desc_slots for slots in handler_slots),
                        default=0,
                    ),
                )
                self._phase_local_desc_slot_widths[phase] = width
                flat_slots: List[int] = []
                for handler_slots in local_desc_slots:
                    for slots in handler_slots:
                        encoded = list(slots)
                        encoded.extend([-1] * (width - len(encoded)))
                        flat_slots.extend(encoded)
                self._phase_local_desc_slot_tensors[phase] = torch.tensor(
                    flat_slots, dtype=torch.int32, device=self.device
                )
            else:
                self._phase_local_desc_slot_widths[phase] = 0
                self._phase_local_desc_slot_tensors[phase] = None
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
        dispatch_inputs = self._backend.compile_phase_dispatch_inputs(self)
        return (
            dispatch_inputs["dispatch_load"],
            dispatch_inputs["dispatch_compute"],
            dispatch_inputs["dispatch_store"],
            dispatch_inputs["dispatch_communicate"],
            dispatch_inputs["phase_uses_handler_local_idx"],
            dispatch_inputs["phase_uses_runtime_transport_selector"],
            dispatch_inputs["phase_uses_desc_slot_selector"],
            dispatch_inputs["inner_iters_list"],
            dispatch_inputs["has_communicate"],
            dispatch_inputs["per_op_warps"],
            dispatch_inputs["phase_tensor_names"],
            dispatch_inputs["phase_tma_names"],
            dispatch_inputs["all_tma_canonical"],
        )

    def _collect_dispatch_signatures(self) -> Dict[str, Any]:
        """Collect per-op dispatch signatures and handler ids."""
        registry = self._tensor_registry
        tma_registry = self._tma_registry
        peer_tma_registry = self._peer_tma_registry
        all_canonical = registry.canonical_names
        all_tma_canonical = tma_registry.all_canonical_names + peer_tma_registry.all_canonical_names

        op_tensor_args = []
        op_tma_args = {"load": [], "compute": [], "store": [], "communicate": []}
        compile_keys = []
        op_weights = [max(1, op.total_tiles) for op in self.ops]

        for i, op in enumerate(self.ops):
            tensor_args = registry.get_op_tensor_args(i, op.op_cls)
            op_tensor_args.append(tensor_args)
            for phase in ("load", "compute", "store"):
                op_tma_args[phase].append(tma_registry.get_op_tma_args(i, phase))
            op_tma_args["communicate"].append(
                peer_tma_registry.get_op_peer_tma_args(i, "communicate")
            )

            static_dims_key = tuple(sorted(op.static_dims.items())) if op.static_dims else ()
            dtypes_key = (
                tuple(sorted((k, v.__name__) for k, v in op.tensor_dtypes.items()))
                if op.tensor_dtypes else ()
            )
            strides_key = (
                tuple(sorted((k, v) for k, v in op.tensor_strides.items()))
                if op.tensor_strides else ()
            )
            n_tensor = len(list(dict.fromkeys(tensor_args)))
            tma_counts = tuple(
                len(op_tma_args[ph][-1]) for ph in ("load", "compute", "store", "communicate")
            )
            compile_key = (op.op_cls, static_dims_key, dtypes_key, strides_key, n_tensor, tma_counts)
            compile_keys.append(compile_key)

        return {
            "all_canonical": all_canonical,
            "all_tma_canonical": all_tma_canonical,
            "op_tensor_args": op_tensor_args,
            "op_tma_args": op_tma_args,
            "compile_keys": compile_keys,
            "op_weights": op_weights,
        }

    def _build_exec_dispatch_fn(
        self,
        phase_fns,
        phase_name,
        op_tensor_args,
        all_canonical,
        op_tma_args=None,
        all_tma_canonical=None,
        op_weights=None,
    ):
        """Build a dispatch function via exec() with tensor and TMA parameters.

        For load phase (phase_name="load"), all compiled load functions
        receive work_mbar uniformly. Sync ops have mbarrier_arrive baked
        into their compiled body by compile_load; async ops signal it
        themselves via mbarrier_arrive_expect_tx + cute.copy.

        Generates source like:
            @cute.jit
            def dispatch_load(op_idx, page_ptr, tile_0, ..., tile_4,
                              op_config_ptr, work_mbar, t0, t1, tma0_atom, tma0_gmem):
                if op_idx == Int32(0):
                    _fn_0(page_ptr, tile_0, ..., tile_4, op_config_ptr, work_mbar, t0, t1, tma0_atom, tma0_gmem)
        """
        is_load = phase_name == "load"
        tensor_params = ", ".join(all_canonical)

        tile_params = ", ".join(f"tile_{i}" for i in range(MAX_TILE_DIMS))

        # TMA params for signature
        tma_params = ", ".join(all_tma_canonical) if all_tma_canonical else ""

        prefix_weights = [0]
        if op_weights:
            running = 0
            for weight in op_weights:
                running += max(1, int(weight))
                prefix_weights.append(running)

        def _range_weight(lo: int, hi: int) -> int:
            if not op_weights:
                return hi - lo + 1
            return prefix_weights[hi + 1] - prefix_weights[lo]

        def _weighted_mid(lo: int, hi: int) -> int:
            if lo >= hi:
                return lo
            if not op_weights:
                return (lo + hi) // 2

            best_mid = lo
            best_cost = None
            for mid in range(lo, hi):
                left = _range_weight(lo, mid)
                right = _range_weight(mid + 1, hi)
                cost = abs(left - right)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_mid = mid
            return best_mid

        def _emit_dispatch(lo: int, hi: int, indent: str) -> List[str]:
            if lo > hi:
                return [f"{indent}pass"]
            if lo == hi:
                args = op_tensor_args[lo]
                all_args = list(dict.fromkeys(args))
                if op_tma_args and lo < len(op_tma_args):
                    all_args.extend(op_tma_args[lo])
                args_str = ", ".join(all_args)
                if args_str:
                    args_str = ", " + args_str
                if is_load:
                    return [
                        f"{indent}_fn_{lo}(page_ptr, {tile_params}, op_config_ptr, work_mbar, "
                        f"inner_iter_idx{args_str})"
                    ]
                return [f"{indent}_fn_{lo}(page_ptr, {tile_params}, op_config_ptr{args_str})"]

            mid = _weighted_mid(lo, hi)
            lines = [f"{indent}if op_idx <= Int32({mid}):"]
            lines.extend(_emit_dispatch(lo, mid, indent + "    "))
            lines.append(f"{indent}else:")
            lines.extend(_emit_dispatch(mid + 1, hi, indent + "    "))
            return lines

        body = "\n".join(_emit_dispatch(0, len(op_tensor_args) - 1, "    ")) if op_tensor_args else "    pass"
        fn_name = f"dispatch_{phase_name}"
        tensor_sig = f", {tensor_params}" if tensor_params else ""
        tma_sig = f", {tma_params}" if tma_params else ""
        extra_sig = ""
        if is_load:
            extra_sig = ", work_mbar, inner_iter_idx"
        fn_source = (
            "@cute.jit\n"
            f"def {fn_name}(op_idx, page_ptr, {tile_params}, "
            f"op_config_ptr{extra_sig}{tensor_sig}{tma_sig}):\n"
            f"{body}\n"
        )

        exec_globals = {"cute": cute, "Int32": Int32, "Int64": Int64}
        for i, fn in enumerate(phase_fns):
            exec_globals[f"_fn_{i}"] = fn

        exec_generated_source(fn_source, fn_name, exec_globals)
        return exec_globals[fn_name]

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
        trace_sig = ", trace_buffer_ptr" if tracing else ""
        trace_init = "" if tracing else "    trace_buffer_ptr = Int64(0)\n"
        local_sig_parts = []
        selector_sig_parts = []
        desc_slot_sig_parts = []
        local_init = ""
        selector_init = ""
        desc_slot_init = ""
        for phase in ("load", "compute", "store", "communicate"):
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
            f"                  op_meta_ptr{local_sig}{selector_sig}{desc_slot_sig}, signal_meta_ptr{peer_signal_sig},\n"
            "                  num_instructions, tidx, block_id, num_blocks,\n"
            f"                  smem_base{trace_sig}, wait_info_ptr{tensor_sig}{tma_sig}{peer_tma_sig}):\n"
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
            "ld_global_i64": ld_global_i64,
            "mbarrier_init": mbarrier_init,
            "mbarrier_init_fence": mbarrier_init_fence,
            "mbarrier_arrive": mbarrier_arrive,
            "mbarrier_wait": mbarrier_wait,
            "nanosleep": nanosleep,
            "named_barrier_sync": named_barrier_sync,
            "num_pages": num_pages,
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
            helper_sources.append(
                "@cute.jit\n"
                f"def {helper_name}(tensor):\n"
                f"    atom, gmem = {self._render_tma_creation_expr(desc, 'tensor')}\n"
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
        for phase in ("load", "compute", "store", "communicate"):
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
            f"                 op_meta_ptr{local_sig}{selector_sig}{desc_slot_sig}, signal_meta_ptr{peer_signal_sig},\n"
            f"                 wait_info_ptr, num_instructions{trace_sig}"
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
            "        self.kernel(instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            f"                    op_meta_ptr{local_arg}{selector_arg}{desc_slot_arg}, signal_meta_ptr{peer_signal_arg},\n"
            f"                    wait_info_ptr{trace_arg},\n"
            f"                    num_instructions{tensor_sig}{kernel_tma_sig}).launch(\n"
            "            grid=[self.num_sms, 1, 1],\n"
            "            block=[self.threads_per_block, 1, 1],\n"
            "            smem=self.smem_size,\n"
            "            stream=stream,\n"
            "            min_blocks_per_mp=1,\n"
            "        )\n"
            "\n"
            "    @cute.kernel\n"
            "    def kernel(self, instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            f"               op_meta_ptr{local_sig}{selector_sig}{desc_slot_sig}, signal_meta_ptr{peer_signal_sig},\n"
            f"               wait_info_ptr, num_instructions{trace_sig}{tensor_sig}{kernel_tma_sig}):\n"
            f"{local_init}"
            f"{selector_init}"
            f"{desc_slot_init}"
            "        tidx = cute.arch.thread_idx()[0]\n"
            "        block_id = cute.arch.block_idx()[0]\n"
            "        num_blocks = cute.arch.grid_dim()[0]\n"
            "        smem_base = get_smem_base_ptr()\n"
            "        _kernel_loop(instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            f"                     op_meta_ptr{local_arg}{selector_arg}{desc_slot_arg}, signal_meta_ptr{peer_signal_arg},\n"
            "                     num_instructions, tidx, block_id, num_blocks,\n"
             f"                     smem_base{trace_arg}, wait_info_ptr{tensor_sig}{kernel_tma_sig})\n"
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
        }
        from .transport import copy_runtime_desc_to_pool, fence_runtime_desc_pool
        pk_globals["copy_runtime_desc_to_pool"] = copy_runtime_desc_to_pool
        pk_globals["fence_runtime_desc_pool"] = fence_runtime_desc_pool

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

    def _build_inner_iters_fn(self, inner_iters_list):
        """Build a small JIT function that maps op index to inner iteration count."""
        iters_lines = []
        for idx, n_iters in enumerate(inner_iters_list):
            keyword = "if" if idx == 0 else "elif"
            iters_lines.append(f"    {keyword} op_idx == Int32({idx}):\n        _r = Int32({n_iters})")
        iters_body = "\n".join(iters_lines) if iters_lines else "    _r = Int32(1)"
        iters_source = (
            "@cute.jit\n"
            "def _get_inner_iters(op_idx) -> Int32:\n"
            "    _r = Int32(1)\n"
            f"{iters_body}\n"
            "    return _r\n"
        )
        iters_globals = {"cute": cute, "Int32": Int32}
        exec_generated_source(iters_source, "_get_inner_iters", iters_globals)
        return iters_globals["_get_inner_iters"]

    def _kernel_static_config(self) -> Dict[str, Any]:
        """Collect the compile-time constants used to build the persistent kernel."""
        layout = self._layout
        threads_per_block = self.config.threads_per_block
        num_mma_warps = (threads_per_block // 32) - NUM_DMA_WARPS

        return {
            "num_sms": self.config.num_sms,
            "threads_per_block": threads_per_block,
            "smem_size": layout.total_size,
            "tracing": self.config.tracing,
            "num_pages": layout.num_pages,
            "iq_offset": layout.iq_offset,
            "flags_offset": layout.flags_offset,
            "ring_state_offset": layout.ring_state_offset,
            "pages_start": layout.pages_start,
            "aligned_page_size": layout.aligned_page_size,
            "work_notify_mbar_offset_0": layout.work_notify_mbar_offset(0),
            "compute_done_mbar_offset_0": layout.compute_done_mbar_offset(0),
            "num_mma_warps": num_mma_warps,
            "num_compute_threads": num_mma_warps * 32,
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
            inner_iters_list,
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
            t4 = Int32(0)
            c0 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_0))
            c1 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_1))
            c2 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_2))
            c3 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_3))
            c4 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_4))
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
            if c4 > Int32(1):
                t4 = rem
            return t0, t1, t2, t3, t4


        @cute.jit
        def _signal_barriers_from_meta(
            signal_meta_ptr: Int64,
            op_idx: Int32,
            tile_0: Int32,
            tile_1: Int32,
            tile_2: Int32,
            tile_3: Int32,
            tile_4: Int32,
            barriers_ptr: Int64,
        ):
            for sig_idx in range_constexpr(self._max_signal_formulas):
                base_offset = (
                    op_idx * Int32(self._max_signal_formulas * _SIGNAL_META_STRIDE)
                    + Int32(sig_idx * _SIGNAL_META_STRIDE)
                )
                barrier_base = ld_global_i32(
                    signal_meta_ptr, base_offset + Int32(_SIGNAL_META_BASE)
                )
                if barrier_base >= Int32(0):
                    guard_max = ld_global_i32(
                        signal_meta_ptr, base_offset + Int32(_SIGNAL_META_GUARD_MAX)
                    )
                    coeff0 = ld_global_i32(signal_meta_ptr, base_offset + Int32(_SIGNAL_META_COEFF_0 + 0))
                    coeff1 = ld_global_i32(signal_meta_ptr, base_offset + Int32(_SIGNAL_META_COEFF_0 + 1))
                    coeff2 = ld_global_i32(signal_meta_ptr, base_offset + Int32(_SIGNAL_META_COEFF_0 + 2))
                    coeff3 = ld_global_i32(signal_meta_ptr, base_offset + Int32(_SIGNAL_META_COEFF_0 + 3))
                    coeff4 = ld_global_i32(signal_meta_ptr, base_offset + Int32(_SIGNAL_META_COEFF_0 + 4))
                    linear = (
                        coeff0 * tile_0
                        + coeff1 * tile_1
                        + coeff2 * tile_2
                        + coeff3 * tile_3
                        + coeff4 * tile_4
                    )
                    if linear < guard_max:
                        div0 = ld_global_i32(signal_meta_ptr, base_offset + Int32(_SIGNAL_META_DIV_0 + 0))
                        div1 = ld_global_i32(signal_meta_ptr, base_offset + Int32(_SIGNAL_META_DIV_0 + 1))
                        div2 = ld_global_i32(signal_meta_ptr, base_offset + Int32(_SIGNAL_META_DIV_0 + 2))
                        div3 = ld_global_i32(signal_meta_ptr, base_offset + Int32(_SIGNAL_META_DIV_0 + 3))
                        div4 = ld_global_i32(signal_meta_ptr, base_offset + Int32(_SIGNAL_META_DIV_0 + 4))
                        barrier_idx = (
                            barrier_base
                            + coeff0 * (tile_0 // div0)
                            + coeff1 * (tile_1 // div1)
                            + coeff2 * (tile_2 // div2)
                            + coeff3 * (tile_3 // div3)
                            + coeff4 * (tile_4 // div4)
                        )
                        global_barrier_signal_gpu(barriers_ptr, barrier_idx)

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
            "max_waits": max(1, self._builder.max_wait_deps),
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
            "_op_meta_i32": runtime["_op_meta_i32"],
            "_op_meta_base": runtime["_op_meta_base"],
            "_op_meta_i32_base": runtime["_op_meta_i32_base"],
            "max_waits": runtime["max_waits"],
            "global_barrier_wait": global_barrier_wait,
            "ld_global_i32": ld_global_i32,
            "has_communicate": runtime["has_communicate"],
            "needs_warp_transition": runtime["needs_warp_transition"],
            "dispatch_load_uses_handler_local_idx": runtime["phase_uses_handler_local_idx"]["load"],
            "dispatch_compute_uses_handler_local_idx": runtime["phase_uses_handler_local_idx"]["compute"],
            "dispatch_store_uses_handler_local_idx": runtime["phase_uses_handler_local_idx"]["store"],
            "dispatch_communicate_uses_handler_local_idx": runtime["phase_uses_handler_local_idx"]["communicate"],
            "MIN_IDLE_REGS": MIN_IDLE_REGS,
            "_OP_META_INNER_ITERS": _OP_META_INNER_ITERS,
            "_OP_META_NUM_WARPS": _OP_META_NUM_WARPS,
            "_OP_META_USES_CONFIG_MASK": _OP_META_USES_CONFIG_MASK,
            "_OP_META_HANDLER_IDX": _OP_META_HANDLER_IDX,
            "_OP_META_LOAD_LOCAL_IDX": _OP_META_LOAD_LOCAL_IDX,
            "_OP_META_COMPUTE_LOCAL_IDX": _OP_META_COMPUTE_LOCAL_IDX,
            "_OP_META_STORE_LOCAL_IDX": _OP_META_STORE_LOCAL_IDX,
            "_OP_META_COMM_LOCAL_IDX": _OP_META_COMM_LOCAL_IDX,
            "_TILE_INFO_LINEAR_IDX": _TILE_INFO_LINEAR_IDX,
            "_TILE_INFO_HANDLER_IDX": _TILE_INFO_HANDLER_IDX,
            "_TILE_INFO_TILE_0": _TILE_INFO_TILE_0,
            "_TILE_INFO_TILE_1": _TILE_INFO_TILE_1,
            "_TILE_INFO_TILE_2": _TILE_INFO_TILE_2,
            "_TILE_INFO_TILE_3": _TILE_INFO_TILE_3,
            "_TILE_INFO_TILE_4": _TILE_INFO_TILE_4,
            "_TILE_INFO_INNER_ITERS": _TILE_INFO_INNER_ITERS,
            "_TILE_INFO_NUM_WARPS": _TILE_INFO_NUM_WARPS,
            "_TILE_INFO_OP_CONFIG": _TILE_INFO_OP_CONFIG,
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
                    for _ip in range(num_pages):
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
                produce_idx = Int32(0)
                _fetch_idx = block_id
                _fetch_idx_save = Int32(0)
                _ctrl_done = Int32(0)
                _ctrl_cached_op_idx = Int32(-1)
                _ctrl_cached_meta_base = Int32(0)
                _ctrl_cached_uses_mask = Int32(0)
                _ctrl_cached_config = Int64(0)

                dispatch_load_slot_ptr = flags_ptr + FLAG_DISPATCH_LOAD
                produce_idx_ptr = flags_ptr + FLAG_PRODUCE_IDX
                store_idx_ptr = flags_ptr + FLAG_STORE_IDX
                load_done_ptr = flags_ptr + FLAG_LOAD_DONE
                _temp_instr = iq_base

                while _ctrl_done == Int32(0):
                    if lane_id == Int32(0):
                        _instr_op = Int32(TileInstruction.END_MARKER)
                        _instr_lin = Int32(0)
                        if _fetch_idx < num_instructions:
                            load_instruction_to_smem(instructions_ptr, _fetch_idx, _temp_instr)
                            _instr_op, _instr_lin = ld_shared_v2_b32(_temp_instr)
                            if _instr_op == Int32(TileInstruction.END_MARKER):
                                _fetch_idx = num_instructions
                            if _instr_op != Int32(TileInstruction.END_MARKER):
                                _fetch_idx_save = _fetch_idx
                                _fetch_idx = _fetch_idx + num_blocks

                        _p_meta_base = Int32(0)
                        if _instr_op >= Int32(0):
                            for _w in range_constexpr(max_waits):
                                _wi_off = _fetch_idx_save * Int32(max_waits * 2) + Int32(_w * 2)
                                _bar_idx = ld_global_i32(wait_info_ptr, _wi_off)
                                if _bar_idx >= Int32(0):
                                    _bar_exp = ld_global_i32(wait_info_ptr, _wi_off + Int32(1))
                                    global_barrier_wait(barriers_ptr, _bar_idx, _bar_exp)

                            _si = ld_shared_i32(store_idx_ptr)
                            while (produce_idx - _si) >= Int32(num_pages):
                                _wait_slot = produce_idx % Int32(num_pages)
                                _wait_phase = ((produce_idx // Int32(num_pages)) + Int32(1)) % Int32(2)
                                mbarrier_wait(
                                    _compute_done_mbar(smem_base, _wait_slot), _wait_phase
                                )
                                _si = ld_shared_i32(store_idx_ptr)

                            _dl_prev = ld_shared_acquire_cta_i32(dispatch_load_slot_ptr)
                            while _dl_prev != Int32(-1):
                                _dl_prev = ld_shared_acquire_cta_i32(dispatch_load_slot_ptr)

                            _p_slot = produce_idx % Int32(num_pages)
                            _p_ti = smem_base + Int32(ring_state_offset) + _p_slot * Int32(tile_info_bytes)
                            if _instr_op != _ctrl_cached_op_idx:
                                _p_meta_base = _op_meta_base(_instr_op)
                                _ctrl_cached_meta_base = _p_meta_base
                                _ctrl_cached_uses_mask = _op_meta_i32_base(
                                    op_meta_ptr, _p_meta_base, Int32(_OP_META_USES_CONFIG_MASK)
                                )
                                if _ctrl_cached_uses_mask != Int32(0):
                                    _ctrl_cached_config = ld_global_i64(op_configs_ptr, _instr_op)
                                else:
                                    _ctrl_cached_config = Int64(0)
                                _ctrl_cached_op_idx = _instr_op
                            else:
                                _p_meta_base = _ctrl_cached_meta_base
                            _p_t0, _p_t1, _p_t2, _p_t3, _p_t4 = decompose_tile(
                                op_meta_ptr, _p_meta_base, _instr_lin
                            )
                            st_shared_i32(_p_ti, _instr_op)
                            if const_expr(has_communicate):
                                st_shared_i32(
                                    _p_ti + Int32(4 * _TILE_INFO_LINEAR_IDX),
                                    _instr_lin,
                                )
                            st_shared_i32(
                                _p_ti + Int32(4 * _TILE_INFO_HANDLER_IDX),
                                _op_meta_i32_base(
                                    op_meta_ptr, _p_meta_base, Int32(_OP_META_HANDLER_IDX)
                                ),
                            )
                            st_shared_i32(_p_ti + Int32(4 * _TILE_INFO_TILE_0), _p_t0)
                            st_shared_i32(_p_ti + Int32(4 * _TILE_INFO_TILE_1), _p_t1)
                            st_shared_i32(_p_ti + Int32(4 * _TILE_INFO_TILE_2), _p_t2)
                            st_shared_i32(_p_ti + Int32(4 * _TILE_INFO_TILE_3), _p_t3)
                            st_shared_i32(_p_ti + Int32(4 * _TILE_INFO_TILE_4), _p_t4)
                            st_shared_i32(
                                _p_ti + Int32(4 * _TILE_INFO_INNER_ITERS),
                                _op_meta_i32_base(
                                    op_meta_ptr, _p_meta_base, Int32(_OP_META_INNER_ITERS)
                                ),
                            )
                            st_shared_i32(
                                _p_ti + Int32(4 * _TILE_INFO_NUM_WARPS),
                                _op_meta_i32_base(
                                    op_meta_ptr, _p_meta_base, Int32(_OP_META_NUM_WARPS)
                                ),
                            )
                            st_shared_i64(
                                _p_ti + Int32(4 * _TILE_INFO_OP_CONFIG),
                                _ctrl_cached_config,
                            )
                            produce_idx = produce_idx + Int32(1)
                            st_shared_i32(produce_idx_ptr, produce_idx)
                            st_shared_release_cta_i32(dispatch_load_slot_ptr, _p_slot)

                        if _instr_op == Int32(TileInstruction.END_MARKER):
                            if _fetch_idx >= num_instructions:
                                _store_idx_done = ld_shared_i32(store_idx_ptr)
                                if (produce_idx - _store_idx_done) < Int32(num_pages):
                                    _dl_last = ld_shared_acquire_cta_i32(dispatch_load_slot_ptr)
                                    while _dl_last != Int32(-1):
                                        _dl_last = ld_shared_acquire_cta_i32(dispatch_load_slot_ptr)
                                    _sent = produce_idx % Int32(num_pages)
                                    st_shared_i32(
                                        smem_base + Int32(ring_state_offset) + _sent * Int32(tile_info_bytes),
                                        Int32(TileInstruction.END_MARKER),
                                    )
                                    mbarrier_arrive(_work_notify_mbar(smem_base, _sent))
                                    st_shared_i32(load_done_ptr, Int32(1))

                    _ctrl_done = ld_shared_i32(load_done_ptr)

            # ========== LOADER WARP (TMA dispatch) ==========
            if warp_id == Int32(num_mma_warps + 1):
                if const_expr(tracing):
                    _dma_lane = begin_lane_dynamic_raw(
                        Int32(3),
                        Int32(trace_row_stride),
                        block_id,
                        Int32(0),
                        lane_id == Int32(0),
                    )

                _ldr_done = Int32(0)
                _ldr_dispatch_ptr = flags_ptr + FLAG_DISPATCH_LOAD
                _ldr_load_done_ptr = flags_ptr + FLAG_LOAD_DONE

                while _ldr_done == Int32(0):
                    _dl_slot = ld_shared_acquire_cta_i32(_ldr_dispatch_ptr)
                    if _dl_slot != Int32(-1):
                        _dl_ti = smem_base + Int32(ring_state_offset) + _dl_slot * Int32(tile_info_bytes)
                        _dl_op = ld_shared_i32(_dl_ti)
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
                        _dl_4 = ld_shared_i32(_dl_ti + Int32(4 * _TILE_INFO_TILE_4))
                        _dl_config = ld_shared_i64(_dl_ti + Int32(4 * _TILE_INFO_OP_CONFIG))
                        _dl_pp = _get_page_ptr(smem_base, _dl_slot)
                        _dl_mbar = _work_notify_mbar(smem_base, _dl_slot)
                        _dl_iter = Int32(0)
                        if const_expr(tracing):
                            _tl = trace_start()
                        if const_expr(dispatch_load_uses_handler_local_idx):
                            dispatch_load(
                                _dl_handler,
                                _dl_handler_local,
                                _dl_pp,
                                _dl_0,
                                _dl_1,
                                _dl_2,
                                _dl_3,
                                _dl_4,
                                _dl_config,
                                _dl_mbar,
                                _dl_iter,
                            )
                        else:
                            dispatch_load(
                                _dl_handler,
                                _dl_pp,
                                _dl_0,
                                _dl_1,
                                _dl_2,
                                _dl_3,
                                _dl_4,
                                _dl_config,
                                _dl_mbar,
                                _dl_iter,
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
                        with cute.arch.elect_one():
                            st_shared_release_cta_i32(_ldr_dispatch_ptr, Int32(-1))

                    if _dl_slot == Int32(-1):
                        nanosleep(Int32(100))

                    _ldr_done = ld_shared_i32(_ldr_load_done_ptr)
                    if _ldr_done == Int32(1):
                        _dl_final = ld_shared_i32(_ldr_dispatch_ptr)
                        if _dl_final != Int32(-1):
                            _ldr_done = Int32(0)

                if const_expr(tracing):
                    finish_lane_dynamic_raw(_trace_buf, _dma_lane)

            # ========== STORE WARP LOOP ==========
            if is_store_warp:
                if const_expr(tracing):
                    _store_lane = begin_lane_dynamic_raw(
                        Int32(3),
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
                        _s_slot = _s_idx % Int32(num_pages)
                        _s_phase = (_s_idx // Int32(num_pages)) % Int32(2)

                        _ds_ti = smem_base + Int32(ring_state_offset) + _s_slot * Int32(tile_info_bytes)
                        _ds_op = ld_shared_i32(_ds_ti)
                        _ds_lin = Int32(0)
                        if const_expr(has_communicate):
                            _ds_lin = ld_shared_i32(
                                _ds_ti + Int32(4 * _TILE_INFO_LINEAR_IDX)
                            )
                        _ds_meta_base = _op_meta_base(_ds_op)
                        _ds_handler = ld_shared_i32(_ds_ti + Int32(4 * _TILE_INFO_HANDLER_IDX))
                        _dl_store_handler_local = Int32(0)
                        if const_expr(dispatch_load_uses_handler_local_idx):
                            _dl_store_handler_local = _op_meta_i32_base(
                                op_meta_ptr, _ds_meta_base, Int32(_OP_META_LOAD_LOCAL_IDX)
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
                        _ds_4 = ld_shared_i32(_ds_ti + Int32(4 * _TILE_INFO_TILE_4))
                        _ds_inner_iters = ld_shared_i32(_ds_ti + Int32(4 * _TILE_INFO_INNER_ITERS))
                        _ds_config = ld_shared_i64(_ds_ti + Int32(4 * _TILE_INFO_OP_CONFIG))
                        _ds_pp = _get_page_ptr(smem_base, _s_slot)
                        _ds_mbar = _work_notify_mbar(smem_base, _s_slot)

                        _n_iters = _ds_inner_iters
                        if _n_iters > Int32(1):
                            mbarrier_wait(_ds_mbar, _s_phase)
                        _iter_idx = Int32(1)
                        while _iter_idx < _n_iters:
                            if const_expr(dispatch_load_uses_handler_local_idx):
                                dispatch_load(
                                    _ds_handler,
                                    _dl_store_handler_local,
                                    _ds_pp,
                                    _ds_0,
                                    _ds_1,
                                    _ds_2,
                                    _ds_3,
                                    _ds_4,
                                    _ds_config,
                                    _ds_mbar,
                                    _iter_idx,
                                )
                            else:
                                dispatch_load(
                                    _ds_handler,
                                    _ds_pp,
                                    _ds_0,
                                    _ds_1,
                                    _ds_2,
                                    _ds_3,
                                    _ds_4,
                                    _ds_config,
                                    _ds_mbar,
                                    _iter_idx,
                                )
                            _iter_idx = _iter_idx + Int32(1)

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
                        if const_expr(dispatch_store_uses_handler_local_idx):
                            dispatch_store(
                                _ds_handler,
                                _ds_handler_local,
                                _ds_pp,
                                _ds_0,
                                _ds_1,
                                _ds_2,
                                _ds_3,
                                _ds_4,
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
                                _ds_4,
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
                                    _ds_4,
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
                                    _ds_4,
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
                            signal_barriers(
                                signal_meta_ptr,
                                _ds_op,
                                _ds_0,
                                _ds_1,
                                _ds_2,
                                _ds_3,
                                _ds_4,
                                barriers_ptr,
                            )
                            if const_expr(has_communicate):
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
                            _sw_next_slot = _s_idx % Int32(num_pages)
                            _sw_next_phase = (_s_idx // Int32(num_pages)) % Int32(2)
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
                        Int32(3),
                        Int32(trace_row_stride),
                        block_id,
                        Int32(1),
                        (warp_id == Int32(0)) & (lane_id == Int32(0)),
                    )

                consume_ptr = Int32(0)
                mma_running = Int32(1)
                _cached_op_idx = Int32(-1)

                while mma_running == Int32(1):
                    slot = consume_ptr % Int32(num_pages)

                    _wn_phase = (consume_ptr // Int32(num_pages)) % Int32(2)
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
                        tile_4 = ld_shared_i32(tile_info_ptr + Int32(4 * _TILE_INFO_TILE_4))
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
                        page_ptr = _get_page_ptr(smem_base, slot)
                        if op_idx != _cached_op_idx:
                            _cached_op_idx = op_idx

                            if const_expr(needs_warp_transition):
                                _op_warps = ld_shared_i32(
                                    tile_info_ptr + Int32(4 * _TILE_INFO_NUM_WARPS)
                                )
                                if warp_id >= _op_warps:
                                    setmaxregister_decrease(MIN_IDLE_REGS)
                                named_barrier_sync(
                                    Int32(1), Int32(num_compute_threads))
                                if warp_id < _op_warps:
                                    setmaxregister_increase(mma_reg_count)

                        if const_expr(tracing):
                            _tc = trace_start()

                        if const_expr(dispatch_compute_uses_handler_local_idx):
                            dispatch_compute(
                                _handler_idx,
                                _handler_local_idx,
                                page_ptr,
                                tile_0,
                                tile_1,
                                tile_2,
                                tile_3,
                                tile_4,
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
                                tile_4,
                                _op_config,
                            )

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
        """Build the PersistentKernel via source transformation.

        Extracts the kernel loop body, adds tensor and TMA params to dispatch
        call sites (if any), and exec-generates the PersistentKernel class
        with all params threaded through __call__ -> kernel -> _kernel_loop.
        """
        all_canonical = self._backend.all_canonical(self)
        tensor_params = ", ".join(all_canonical)
        tensor_sig = self._signature_suffix(all_canonical)

        # TMA params threaded into dispatch calls.
        tma_registry = self._tma_registry
        all_tma_canonical: List[str] = []
        seen_tma = set()
        for phase in ("load", "compute", "store", "communicate"):
            for name in phase_tma_names.get(phase, []):
                if name in seen_tma:
                    continue
                seen_tma.add(name)
                all_tma_canonical.append(name)
        tma_params = ", ".join(all_tma_canonical)
        tma_sig = self._signature_suffix(all_tma_canonical)

        # Peer TMA params for multi-GPU communication
        peer_tma_registry = self._peer_tma_registry
        peer_tma_sig = ""

        dispatch_extra_params = {}
        for phase in ("load", "compute", "store", "communicate"):
            phase_tensor_params = ", ".join(phase_tensor_names.get(phase, []))
            phase_tma_params = ", ".join(phase_tma_names.get(phase, []))
            selector_param = ""
            if phase_uses_runtime_transport_selector.get(phase, False):
                selector_param = f"{phase}_local_transport_positions_ptr"
            desc_slot_selector_param = ""
            if phase_uses_desc_slot_selector.get(phase, False):
                desc_slot_selector_param = f"{phase}_local_desc_slots_ptr"
            dispatch_extra_params[phase] = ", ".join(
                filter(None, [selector_param, desc_slot_selector_param, phase_tensor_params, phase_tma_params])
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

        exec_globals = self._build_kernel_exec_globals(
            dispatch_load=dispatch_load,
            dispatch_compute=dispatch_compute,
            dispatch_store=dispatch_store,
            signal_barriers=signal_barriers,
            get_page_ptr_fn=get_page_ptr_fn,
            num_pages=num_pages,
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

    def _build_signal_barriers(self):
        """Build function to signal barriers after tile completion.

        Returns a @cute.jit function that signals all barriers for the
        given tile after it completes. Uses if/elif chain so only the
        matching op's barriers are evaluated.
        """
        barrier_formulas = self._builder.get_op_barrier_formulas()
        tile_params = ", ".join(f"tile_{i}" for i in range(MAX_TILE_DIMS))

        fn_name = "signal_barriers"

        def _barrier_idx_expr(wf):
            """Generate barrier_idx expression from formula coefficients."""
            parts = [f"Int32({wf.base})"]
            for j in range(MAX_TILE_DIMS):
                if wf.coeffs[j] != 0:
                    parts.append(f"Int32({wf.coeffs[j]}) * (tile_{j} // Int32({wf.divs[j]}))")
            return " + ".join(parts)

        # Build if/elif branches for each op
        branches = []
        first = True
        for i, (wait_formulas, signal_formulas) in barrier_formulas.items():
            keyword = "if" if first else "elif"
            first = False

            if not signal_formulas:
                branches.append(f"    {keyword} op_idx == Int32({i}):\n        pass")
                continue

            lines = [f"    {keyword} op_idx == Int32({i}):"]
            for wf in signal_formulas:
                lines.append(f"        barrier_idx = ({_barrier_idx_expr(wf)})")
                lines.append("        global_barrier_signal(barriers_ptr, barrier_idx)")

            branches.append("\n".join(lines))

        body = "\n".join(branches) if branches else "    pass"

        fn_source = (
            f"@cute.jit\ndef {fn_name}(op_idx, {tile_params}, barriers_ptr):\n{body}\n"
        )

        # Use GPU-scoped barrier ops for local (intra-GPU) barriers.
        # Peer barriers use .sys scope (see _build_signal_peer_barriers).
        exec_globals = {
            "cute": cute,
            "Int32": Int32,
            "Int64": Int64,
            "global_barrier_signal": global_barrier_signal_gpu,
        }

        exec_generated_source(fn_source, fn_name, exec_globals)
        return exec_globals[fn_name]

    def _build_signal_peer_barriers(self):
        """Build function to signal peer barriers after communicate.

        Returns a @cute.jit function that signals a barrier in the
        peer_barriers array for ops with peer_stores. Uses .sys scope
        global_barrier_signal for cross-GPU NVLink visibility.

        Barrier index = per-op offset + linear_tile_idx within op.
        """
        lines = []
        barrier_offset = 0
        first = True
        for i, op in enumerate(self.ops):
            if not self._op_has_peer_barriers(op):
                continue
            keyword = "if" if first else "elif"
            first = False
            lines.append(
                f"    {keyword} op_idx == Int32({i}):\n"
                f"        barrier_idx = Int32({barrier_offset}) + linear_tile_idx\n"
                f"        global_barrier_signal(peer_barriers_ptr, barrier_idx)"
            )
            barrier_offset += op.total_tiles

        body = "\n".join(lines) if lines else "    pass"

        fn_source = f"@cute.jit\ndef signal_peer_barriers(op_idx, linear_tile_idx, peer_barriers_ptr):\n{body}\n"

        exec_globals = {
            "cute": cute,
            "Int32": Int32,
            "Int64": Int64,
            "global_barrier_signal": global_barrier_signal,
        }

        exec_generated_source(fn_source, "signal_peer_barriers", exec_globals)
        return exec_globals["signal_peer_barriers"]

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

        Tile counts, barrier formulas, and instruction streams are now
        materialized into runtime tensors (`op_meta`, instructions,
        wait_info, signal metadata). They no longer affect the emitted
        kernel body directly, so batch-dynamic scheduling can reuse the
        same compiled kernel while rebuilding only runtime metadata.
        """
        handler_keys = tuple(
            spec.compile_key for spec in self._backend_ir.handler_specs
        )
        phase_runtime_keys = tuple(
            (
                phase,
                self._backend_ir.phase_local_transport_positions[phase],
                self._phase_local_transport_position_widths[phase],
                self._phase_local_desc_slot_widths[phase],
                self._phase_local_transport_position_tensors[phase] is not None,
                self._phase_local_desc_slot_tensors[phase] is not None,
            )
            for phase in ("load", "compute", "store", "communicate")
        )
        signal_shape_key = (
            self._builder.max_wait_deps,
            self._max_signal_formulas,
        )

        config_key = (
            self.config.backend,
            self.config.num_sms,
            self.config.threads_per_block,
            self.config.page_size,
            self.config.num_pages,
            self.config.tracing,
            self.config.dma_reg_count,
            self.config.mma_reg_count,
            self.config.num_devices,
            self.config.noinline,
            self.config.opt_level,
        )

        # TMA descriptors are created at launch time from runtime tensors, so
        # the compiled kernel only depends on descriptor *structure*:
        # tensor/layout rank, tile shape, dtype, direction, and stride order.
        #
        # Different allocations with the same shape/stride layout should share
        # the same compiled kernel. Runtime tensor handles still flow through
        # launch_state and TMA descriptors are rebuilt per launch.
        local_tma_key = tuple(
            dict.fromkeys(
                (
                    desc.tensor_canonical,
                    desc.direction,
                    tuple(desc.tile_shape),
                    getattr(desc.dtype, "__name__", str(desc.dtype)),
                    desc.smem_layout_src,
                    tuple(desc.dim_perm),
                )
                for desc in self._tma_registry.descriptors
            )
        )
        peer_tma_key = tuple(
            dict.fromkeys(
                (
                    desc.tensor_canonical,
                    desc.peer_idx,
                    desc.direction,
                    tuple(desc.tile_shape),
                    getattr(desc.dtype, "__name__", str(desc.dtype)),
                    desc.smem_layout_src,
                )
                for desc in self._peer_tma_registry.descriptors
            )
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
        self._prepare_tma_tensors()
        self._prepare_peer_tma_tensors()
        if self._compiled_kernel is None:
            # Check class-level cache first
            cache_key = self._make_cache_key()
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
            noinline_str = " [noinline]" if self.config.noinline else ""
            print(
                f"Compiling persistent kernel ({tracing_str}{tma_str}{peer_str}{noinline_str}) for "
                f"{len(self.ops)} ops, "
                f"{self.total_tiles} tiles, {self.num_sms} SMs, "
                f"{self.smem_size // 1024}KB smem..."
            )

            # Install noinline + opt_level patches during compilation
            _pipeline_patch = None
            if self.config.noinline:
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
                if self.config.tracing:
                    compile_args.append(launch_state.trace_buffer_ptr)
                compile_args.extend(
                    [
                        launch_state.wait_info_ptr,
                        self._num_instructions_i32,
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
                if self.config.noinline:
                    noinline_mod.uninstall()
                if _pipeline_patch is not None:
                    CutlassBaseDSL.preprocess_pipeline = _pipeline_patch

            # Store in class-level cache
            Megakernel._compiled_kernel_cache[cache_key] = self._compiled_kernel
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

        phase_local_idx_ptrs = {
            phase: (
                Int64(self._phase_local_idx_tensors[phase].data_ptr())
                if self._phase_local_idx_tensors[phase] is not None else Int64(0)
            )
            for phase in ("load", "compute", "store", "communicate")
        }
        phase_local_transport_position_ptrs = {
            phase: (
                Int64(self._phase_local_transport_position_tensors[phase].data_ptr())
                if self._phase_local_transport_position_tensors[phase] is not None else Int64(0)
            )
            for phase in ("load", "compute", "store", "communicate")
        }
        phase_local_desc_slot_ptrs = {
            phase: (
                Int64(self._phase_local_desc_slot_tensors[phase].data_ptr())
                if self._phase_local_desc_slot_tensors[phase] is not None else Int64(0)
            )
            for phase in ("load", "compute", "store", "communicate")
        }
        return _LaunchState(
            instructions_ptr=Int64(self._instructions_tensor.data_ptr()),
            barriers_ptr=Int64(self._barriers_tensor.data_ptr()),
            op_configs_ptr=Int64(self._op_configs_tensor.data_ptr()),
            wait_info_ptr=Int64(self._wait_info.data_ptr()),
            op_meta_ptr=Int64(self._op_metadata_tensor.data_ptr()),
            load_local_idx_ptr=phase_local_idx_ptrs["load"],
            compute_local_idx_ptr=phase_local_idx_ptrs["compute"],
            store_local_idx_ptr=phase_local_idx_ptrs["store"],
            communicate_local_idx_ptr=phase_local_idx_ptrs["communicate"],
            load_local_transport_positions_ptr=phase_local_transport_position_ptrs["load"],
            compute_local_transport_positions_ptr=phase_local_transport_position_ptrs["compute"],
            store_local_transport_positions_ptr=phase_local_transport_position_ptrs["store"],
            communicate_local_transport_positions_ptr=phase_local_transport_position_ptrs["communicate"],
            load_local_desc_slots_ptr=phase_local_desc_slot_ptrs["load"],
            compute_local_desc_slots_ptr=phase_local_desc_slot_ptrs["compute"],
            store_local_desc_slots_ptr=phase_local_desc_slot_ptrs["store"],
            communicate_local_desc_slots_ptr=phase_local_desc_slot_ptrs["communicate"],
            local_tma_desc_pool_ptr=Int64(self._local_tma_desc_pool.data_ptr()) if self._local_tma_desc_pool is not None else Int64(0),
            peer_tma_desc_pool_ptr=Int64(self._peer_tma_desc_pool.data_ptr()) if self._peer_tma_desc_pool is not None else Int64(0),
            signal_meta_ptr=Int64(self._signal_metadata_tensor.data_ptr()),
            peer_signal_ptr=Int64(self._peer_signal_tensor.data_ptr()) if self._peer_signal_tensor is not None else Int64(0),
            trace_buffer_ptr=Int64(0),
            cute_tensors=selected_cute_tensors,
            tma_tensor_args=[ct for _, ct in self._tma_cute_tensors] if self._tma_cute_tensors else [],
            peer_tma_tensor_args=(
                [ct for _, _, ct in self._peer_tma_cute_tensors]
                if self._peer_tma_cute_tensors else []
            ),
        )

    def _phase_local_idx_launch_args(self, launch_state: _LaunchState) -> List[Int64]:
        """Return phase-local index table pointers in kernel ABI order."""
        args: List[Int64] = []
        if self._phase_local_idx_tensors["load"] is not None:
            args.append(launch_state.load_local_idx_ptr)
        if self._phase_local_idx_tensors["compute"] is not None:
            args.append(launch_state.compute_local_idx_ptr)
        if self._phase_local_idx_tensors["store"] is not None:
            args.append(launch_state.store_local_idx_ptr)
        if self._phase_local_idx_tensors["communicate"] is not None:
            args.append(launch_state.communicate_local_idx_ptr)
        return args

    def _phase_local_transport_position_launch_args(self, launch_state: _LaunchState) -> List[Int64]:
        """Return per-phase local-transport selector table pointers in ABI order."""
        args: List[Int64] = []
        if self._phase_local_transport_position_tensors["load"] is not None:
            args.append(launch_state.load_local_transport_positions_ptr)
        if self._phase_local_transport_position_tensors["compute"] is not None:
            args.append(launch_state.compute_local_transport_positions_ptr)
        if self._phase_local_transport_position_tensors["store"] is not None:
            args.append(launch_state.store_local_transport_positions_ptr)
        if self._phase_local_transport_position_tensors["communicate"] is not None:
            args.append(launch_state.communicate_local_transport_positions_ptr)
        return args

    def _phase_local_desc_slot_launch_args(self, launch_state: _LaunchState) -> List[Int64]:
        """Return per-phase desc-slot selector table pointers in ABI order."""
        args: List[Int64] = []
        if self._phase_local_desc_slot_tensors["load"] is not None:
            args.append(launch_state.load_local_desc_slots_ptr)
        if self._phase_local_desc_slot_tensors["compute"] is not None:
            args.append(launch_state.compute_local_desc_slots_ptr)
        if self._phase_local_desc_slot_tensors["store"] is not None:
            args.append(launch_state.store_local_desc_slots_ptr)
        if self._phase_local_desc_slot_tensors["communicate"] is not None:
            args.append(launch_state.communicate_local_desc_slots_ptr)
        return args

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
        if self.config.tracing:
            launch_args.append(
                trace_buffer_ptr if trace_buffer_ptr is not None else launch_state.trace_buffer_ptr
            )
        launch_args.extend(
            [
                launch_state.wait_info_ptr,
                self._num_instructions_i32,
                *launch_state.cute_tensors,
                launch_state.local_tma_desc_pool_ptr,
                launch_state.peer_tma_desc_pool_ptr,
                *launch_state.tma_tensor_args,
                *launch_state.peer_tma_tensor_args,
                self._needs_tma_desc_pool_init,
            ]
        )
        launch_args.append(stream)
        self._compiled_kernel(*launch_args)
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
        self.wait()

        # Validate tensors haven't changed since schedule()
        if validate:
            self._validate_tensors()

        # Compile on first call (no-op if already compiled).
        # Always call compile() to ensure tensors are prepared, even when
        # _compiled_kernel was injected externally (e.g., autograd cache).
        self.compile()

        # Cache stable launch args on first run
        self._cache_launch_state()
        launch_state = self._launch_state
        stream = self._resolve_launch_stream(stream)

        # Reset barriers for this run
        self._barriers_tensor.zero_()

        # Launch (trace_buffer_ptr is always passed; trace calls compile
        # to nothing when tracing is disabled via constexpr elimination)
        if self.config.tracing:
            self._tracing_state.builder.reset()
            trace_buffer_ptr = Int64(self._tracing_state.builder._buffer.data_ptr())
        else:
            trace_buffer_ptr = launch_state.trace_buffer_ptr

        self._launch_compiled_kernel(launch_state, stream, trace_buffer_ptr=trace_buffer_ptr)

        if sync:
            torch.cuda.synchronize()
            self._has_pending_async_launch = False
        else:
            self._has_pending_async_launch = True

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
        launch_state = self._launch_state
        barriers_tensor = self._barriers_tensor

        def _setup():
            """Reset benchmark-visible runtime state before each launch."""
            self.wait()
            with torch.cuda.stream(bench_stream):
                if setup_fn is not None:
                    setup_fn()
                barriers_tensor.zero_()

        def _launch():
            self._launch_compiled_kernel(launch_state, cu_stream)
            self._has_pending_async_launch = True

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
