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
    from machete.megakernel import Megakernel
    from machete.kernels.rms_norm import RMSNormOp

    ops = RMSNormOp.schedule(x=x, weight=w, y=y)
    kernel = Megakernel(ops)
    kernel.run()
"""

from dataclasses import dataclass
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
    InstructionStreamBuilder,
    TileInstruction,
    TileScheduler,
)
from .compile import compile_phase, exec_generated_source
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
        noinline: Emit each op's compute as noinline device function (default: True).
            Reduces register pressure by isolating op register usage.
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
    - All SMs launched as persistent blocks
    - Each block fetches instructions from global memory in a strided pattern
    - Fine-grained barriers enable tile-level dependencies between operations
    - Barrier wait/signal logic is baked into op handlers at compile time
    - Shared memory is divided into pages managed by a circular buffer
    - Ops are dispatched via compile-time inlined branches

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
                min_pages=1,
            )
            # Store computed num_pages back to config for cache key
            self.config.num_pages = self._layout.num_pages

        # Validate that config page_size is large enough for all ops
        for op in ops:
            op_page = op.static_dims.get('page_size')
            if op_page is not None and op_page > self.config.page_size:
                raise ValueError(
                    f"Op {op.op_class.__name__} was scheduled for "
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
        self._launch_state: Optional[_LaunchState] = None
        self._cached_cu_stream = None
        self._cached_torch_stream_id = None

        # Validate barrier formulas eagerly (catches incompatible tile sizes early)
        _ = self._builder.num_barriers

        # Trace setup
        from .tracing import setup_tracing

        self._tracing_state = None
        if self.config.tracing:
            self._tracing_state = setup_tracing(self.ops, self.num_sms, self.total_tiles)

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
        """Convert a tensor to a CuTe tensor with static layout metadata."""
        from cutlass.cute.runtime import from_dlpack

        return from_dlpack(tensor, assumed_align=16)

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
        ops = self.ops
        registry = self._tensor_registry
        tma_registry = self._tma_registry
        peer_tma_registry = self._peer_tma_registry
        all_canonical = registry.canonical_names  # ['t0', 't1', ...]
        # Combine local + peer TMA canonical names so all dispatch functions
        # have a unified signature (peer TMA params are only used by communicate).
        all_tma_canonical = tma_registry.all_canonical_names + peer_tma_registry.all_canonical_names

        load_fns = []
        compute_fns = []
        store_fns = []
        communicate_fns = []
        inner_iters_list = []  # Per-op inner iteration count
        per_op_warps = []  # Per-op num_mma_warps for setmaxnreg transitions
        per_op_compute_uses_config = []  # Per-op whether compute wrapper consumes op_config_ptr
        op_tensor_args = []  # Per-op list of canonical tensor arg names
        op_tma_args = {"load": [], "compute": [], "store": [], "communicate": []}

        # Warp-specialized mode: DMA warps are last warps, compute threads = rest
        threads_per_block = self.config.threads_per_block
        num_compute_threads = threads_per_block - NUM_DMA_WARPS * 32  # Exclude load + store warps
        num_mma_warps = num_compute_threads // 32

        for i, op in enumerate(ops):
            # Get canonical names in declaration order for this op
            tensor_args = registry.get_op_tensor_args(i, op.op_cls)
            op_tensor_args.append(tensor_args)

            # Get TMA canonical names and local mappings per phase
            for phase in ("load", "compute", "store"):
                op_tma_args[phase].append(tma_registry.get_op_tma_args(i, phase))
            op_tma_args["communicate"].append(
                peer_tma_registry.get_op_peer_tma_args(i, "communicate"))

            kernel_config = {"threads_per_row": num_compute_threads}

            # Create Op instance with compile-time config, wrap its methods.
            config = build_op_config(op, kernel_config=kernel_config)
            instance = op.op_cls(**config)
            inner_iters_list.append(getattr(instance, "inner_iters", 1))
            per_op_warps.append(getattr(instance, "num_mma_warps", num_mma_warps))

            # Compile all four phases
            phase_fn_lists = {
                "load": load_fns, "compute": compute_fns,
                "store": store_fns, "communicate": communicate_fns,
            }
            for phase_name, fn_list in phase_fn_lists.items():
                if phase_name in ("load", "compute", "store"):
                    tma_args = tma_registry.get_op_tma_args(i, phase_name)
                    tma_mapping = tma_registry.op_mappings.get((i, phase_name), {})
                else:
                    tma_args = op_tma_args["communicate"][-1]
                    tma_mapping = peer_tma_registry.op_mappings.get((i, "communicate"), {})
                fn_list.append(
                    compile_phase(
                        instance,
                        phase_name,
                        tensor_param_names=tensor_args,
                        tma_param_names=tma_args,
                        tma_local_mapping=tma_mapping,
                        noinline=(self.config.noinline if phase_name == "compute" else False),
                    )
                )
            per_op_compute_uses_config.append(bool(getattr(compute_fns[-1], "_uses_op_config_ptr", True)))

        # Generate dispatch functions via exec() — each accepts ALL canonical
        # tensor and TMA names and routes the correct subset to each phase fn.
        def _build_dispatch(phase_fns, phase_name):
            """Build the per-phase dispatch wrapper with canonical tensor/TMA params."""
            return self._build_exec_dispatch_fn(
                phase_fns,
                phase_name,
                op_tensor_args,
                all_canonical,
                op_tma_args=op_tma_args.get(phase_name),
                all_tma_canonical=all_tma_canonical,
            )

        dispatch_load = _build_dispatch(load_fns, "load")
        dispatch_compute = _build_dispatch(compute_fns, "compute")
        dispatch_store = _build_dispatch(store_fns, "store")
        dispatch_communicate = _build_dispatch(communicate_fns, "communicate")

        has_communicate = peer_tma_registry.has_peer_tma

        return (
            dispatch_load,
            dispatch_compute,
            dispatch_store,
            dispatch_communicate,
            inner_iters_list,
            has_communicate,
            per_op_warps,
            per_op_compute_uses_config,
        )

    def _build_exec_dispatch_fn(
        self, phase_fns, phase_name, op_tensor_args, all_canonical, op_tma_args=None, all_tma_canonical=None
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

            mid = (lo + hi) // 2
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
        extra_dispatch_params: str,
    ) -> str:
        """Render the generated `_kernel_loop` source."""
        from .compile import _extract_body

        body = _extract_body(kernel_loop_fn)
        if extra_dispatch_params:
            body = re.sub(
                r"(dispatch_(?:load|compute|store|communicate))\(([^)]*)\)",
                lambda m: (
                    m.group(1)
                    + "("
                    + m.group(2).rstrip().rstrip(",")
                    + ", "
                    + extra_dispatch_params
                    + ")"
                ),
                body,
            )

        return (
            "@cute.jit\n"
            "def _kernel_loop(instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            "                  num_instructions, tidx, block_id, num_blocks,\n"
            f"                  smem_base, trace_buffer_ptr, wait_info_ptr{tensor_sig}{tma_sig}{peer_tma_sig}):\n"
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

    def _append_tma_descriptor_code(self, tma_creation_lines, tma_kernel_args, desc, tensor_source: str) -> None:
        """Append source that constructs one TMA descriptor pair."""
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
        tma_creation_lines.append(
            f"        _smem_layout_{desc.canonical_atom} = {smem_layout_code}\n"
            f"        {desc.canonical_atom}, {desc.canonical_gmem} = "
            f"cute.nvgpu.cpasync.make_tiled_tma_atom(\n"
            f"            {copy_op},\n"
            f"            {tensor_source},\n"
            f"            _smem_layout_{desc.canonical_atom},\n"
            f"            ({shape_str},),\n"
            f"            num_multicast=1,\n"
            f"        )"
        )
        tma_kernel_args.extend((desc.canonical_atom, desc.canonical_gmem))

    def _build_tma_kernel_components(self, tma_registry, peer_tma_registry) -> Dict[str, Any]:
        """Assemble the TMA-specific signature fragments and descriptor setup code."""
        tma_tensor_names = self._collect_tma_tensor_names(tma_registry)
        peer_tma_tensor_names = self._collect_peer_tma_tensor_names(peer_tma_registry)

        tma_creation_lines: List[str] = []
        tma_kernel_args: List[str] = []

        for desc in tma_registry.descriptors:
            ndim = len(desc.tile_shape)
            self._append_tma_descriptor_code(
                tma_creation_lines,
                tma_kernel_args,
                desc,
                f"tma_{desc.tensor_canonical}_{ndim}d",
            )

        for desc in peer_tma_registry.descriptors:
            self._append_tma_descriptor_code(
                tma_creation_lines,
                tma_kernel_args,
                desc,
                f"ptma_{desc.tensor_canonical}_p{desc.peer_idx}",
            )

        tma_creation_code = "\n".join(tma_creation_lines)
        if tma_creation_code:
            tma_creation_code = "\n" + tma_creation_code + "\n"

        combined_tma_names = [
            *tma_registry.all_canonical_names,
            *peer_tma_registry.all_canonical_names,
        ]
        return {
            "tma_tensor_sig": self._signature_suffix(tma_tensor_names),
            "peer_tma_tensor_input_sig": self._signature_suffix(peer_tma_tensor_names),
            "tma_kernel_args_sig": self._signature_suffix(tma_kernel_args),
            "combined_tma_sig": self._signature_suffix(combined_tma_names),
            "tma_creation_code": tma_creation_code,
        }

    def _build_persistent_kernel_source(
        self,
        *,
        num_sms: int,
        threads_per_block: int,
        smem_size: int,
        tensor_sig: str,
        tma_components: Dict[str, Any],
    ) -> str:
        """Render the `PersistentKernel` class source."""
        return (
            "class PersistentKernel:\n"
            "    def __init__(self):\n"
            f"        self.num_sms = {num_sms}\n"
            f"        self.threads_per_block = {threads_per_block}\n"
            f"        self.smem_size = {smem_size}\n"
            "\n"
            "    @cute.jit\n"
            "    def __call__(self, instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            f"                 trace_buffer_ptr, wait_info_ptr, num_instructions"
            f"{tensor_sig}{tma_components['tma_tensor_sig']}{tma_components['peer_tma_tensor_input_sig']}, stream):\n"
            f"{tma_components['tma_creation_code']}"
            "        self.kernel(instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            f"                    trace_buffer_ptr, wait_info_ptr,\n"
            f"                    num_instructions{tensor_sig}{tma_components['tma_kernel_args_sig']}).launch(\n"
            "            grid=[self.num_sms, 1, 1],\n"
            "            block=[self.threads_per_block, 1, 1],\n"
            "            smem=self.smem_size,\n"
            "            stream=stream,\n"
            "            min_blocks_per_mp=1,\n"
            "        )\n"
            "\n"
            "    @cute.kernel\n"
            "    def kernel(self, instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            f"               trace_buffer_ptr, wait_info_ptr, num_instructions{tensor_sig}{tma_components['combined_tma_sig']}):\n"
            "        tidx = cute.arch.thread_idx()[0]\n"
            "        block_id = cute.arch.block_idx()[0]\n"
            "        num_blocks = cute.arch.grid_dim()[0]\n"
            "        smem_base = get_smem_base_ptr()\n"
            "        _kernel_loop(instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            "                     num_instructions, tidx, block_id, num_blocks,\n"
            f"                     smem_base, trace_buffer_ptr, wait_info_ptr{tensor_sig}{tma_components['combined_tma_sig']})\n"
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
            inner_iters_list,
            has_communicate,
            per_op_warps,
            per_op_compute_uses_config,
        ) = self._build_pipelined_dispatch_fns()

        from .tracing import get_trace_exec_globals

        trace_exec_globals = get_trace_exec_globals(self._tracing_state)
        signal_barriers = self._build_signal_barriers()
        signal_peer_barriers = self._build_signal_peer_barriers()

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

        return {
            "setmaxregister_increase": setmaxregister_increase,
            "setmaxregister_decrease": setmaxregister_decrease,
            "dispatch_load": dispatch_load,
            "dispatch_compute": dispatch_compute,
            "dispatch_store": dispatch_store,
            "dispatch_communicate": dispatch_communicate,
            "inner_iters_list": inner_iters_list,
            "has_communicate": has_communicate,
            "per_op_warps": per_op_warps,
            "per_op_compute_uses_config": per_op_compute_uses_config,
            "needs_op_config_load": any(per_op_compute_uses_config),
            "needs_warp_transition": any(w < kernel_cfg["num_mma_warps"] for w in per_op_warps),
            "max_waits": max(1, self._builder.max_wait_deps),
            "signal_barriers": signal_barriers,
            "signal_peer_barriers": signal_peer_barriers,
            "trace_exec_globals": trace_exec_globals,
            "decompose_tile": self._builder.build_decompose_tile_fn(),
            "_get_inner_iters": self._build_inner_iters_fn(inner_iters_list),
            "_op_uses_config": self._build_op_uses_config_fn(per_op_compute_uses_config),
            "_get_page_ptr": _get_page_ptr,
            "_work_notify_mbar": _work_notify_mbar,
            "_compute_done_mbar": _compute_done_mbar,
        }

    def _kernel_extra_exec_globals(
        self,
        kernel_cfg: Dict[str, Any],
        runtime: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the extra globals consumed by the generated kernel loop."""
        return {
            "_work_notify_mbar": runtime["_work_notify_mbar"],
            "_compute_done_mbar": runtime["_compute_done_mbar"],
            "decompose_tile": runtime["decompose_tile"],
            "ld_shared_v2_b32": ld_shared_v2_b32,
            "st_shared_v2_b32": st_shared_v2_b32,
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
            "_get_inner_iters": runtime["_get_inner_iters"],
            "_op_uses_config": runtime["_op_uses_config"],
            "max_waits": runtime["max_waits"],
            "global_barrier_wait": global_barrier_wait,
            "ld_global_i32": ld_global_i32,
            "dispatch_communicate": runtime["dispatch_communicate"],
            "has_communicate": runtime["has_communicate"],
            "signal_peer_barriers": runtime["signal_peer_barriers"],
            "_peer_barriers_data_ptr": kernel_cfg["peer_barriers_data_ptr"],
            "needs_warp_transition": runtime["needs_warp_transition"],
            "per_op_warps": runtime["per_op_warps"],
            "per_op_compute_uses_config": runtime["per_op_compute_uses_config"],
            "needs_op_config_load": runtime["needs_op_config_load"],
            "MIN_IDLE_REGS": MIN_IDLE_REGS,
            "num_ops": len(self.ops),
            **runtime["trace_exec_globals"],
        }

    def _build_op_uses_config_fn(self, uses_config_list):
        """Build a small JIT function mapping op index to whether compute needs op_config_ptr."""
        lines = []
        for idx, uses in enumerate(uses_config_list):
            keyword = "if" if idx == 0 else "elif"
            value = 1 if uses else 0
            lines.append(f"    {keyword} op_idx == Int32({idx}):\n        _r = Int32({value})")
        body = "\n".join(lines) if lines else "    _r = Int32(0)"
        source = (
            "@cute.jit\n"
            "def _op_uses_config(op_idx) -> Int32:\n"
            "    _r = Int32(0)\n"
            f"{body}\n"
            "    return _r\n"
        )
        globs = {"cute": cute, "Int32": Int32}
        exec_generated_source(source, "_op_uses_config", globs)
        return globs["_op_uses_config"]


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
                            st_shared_v2_b32(_p_ti, _instr_op, _instr_lin)
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

                _dl_cached_op_idx = Int32(-1)
                _dl_cached_config = Int64(0)
                _ldr_done = Int32(0)
                _ldr_dispatch_ptr = flags_ptr + FLAG_DISPATCH_LOAD
                _ldr_load_done_ptr = flags_ptr + FLAG_LOAD_DONE

                while _ldr_done == Int32(0):
                    _dl_slot = ld_shared_acquire_cta_i32(_ldr_dispatch_ptr)
                    if _dl_slot != Int32(-1):
                        _dl_ti = smem_base + Int32(ring_state_offset) + _dl_slot * Int32(tile_info_bytes)
                        _dl_op, _dl_lin = ld_shared_v2_b32(_dl_ti)
                        _dl_0, _dl_1, _dl_2, _dl_3, _dl_4 = decompose_tile(_dl_op, _dl_lin)
                        _dl_pp = _get_page_ptr(smem_base, _dl_slot)
                        if _dl_op != _dl_cached_op_idx:
                            _dl_cached_config = ld_global_i64(op_configs_ptr, _dl_op)
                            _dl_cached_op_idx = _dl_op
                        _dl_mbar = _work_notify_mbar(smem_base, _dl_slot)
                        _dl_iter = Int32(0)
                        if const_expr(tracing):
                            _tl = trace_start()
                        dispatch_load(
                            _dl_op,
                            _dl_pp,
                            _dl_0,
                            _dl_1,
                            _dl_2,
                            _dl_3,
                            _dl_4,
                            _dl_cached_config,
                            _dl_mbar,
                            _dl_iter,
                        )
                        if const_expr(tracing):
                            for _i in range_constexpr(num_ops):
                                if _dl_op == Int32(_i):
                                    _dma_lane = end_event_dynamic_raw_1(
                                        _tl,
                                        _trace_buf,
                                        Int32(trace_row_stride),
                                        _dma_lane,
                                        Int32(trace_load_fmts[_i]),
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
                _ds_cached_op_idx = Int32(-1)
                _ds_cached_config = Int64(0)

                while _sw_done == Int32(0):
                    _s_idx = ld_shared_i32(store_idx_ptr)
                    _p_idx = ld_shared_i32(produce_idx_ptr)

                    if _s_idx < _p_idx:
                        _s_slot = _s_idx % Int32(num_pages)
                        _s_phase = (_s_idx // Int32(num_pages)) % Int32(2)

                        _ds_ti = smem_base + Int32(ring_state_offset) + _s_slot * Int32(tile_info_bytes)
                        _ds_op, _ds_lin = ld_shared_v2_b32(_ds_ti)
                        _ds_0, _ds_1, _ds_2, _ds_3, _ds_4 = decompose_tile(_ds_op, _ds_lin)
                        _ds_pp = _get_page_ptr(smem_base, _s_slot)
                        if _ds_op != _ds_cached_op_idx:
                            _ds_cached_config = ld_global_i64(op_configs_ptr, _ds_op)
                            _ds_cached_op_idx = _ds_op
                        _ds_mbar = _work_notify_mbar(smem_base, _s_slot)

                        _n_iters = _get_inner_iters(_ds_op)
                        if _n_iters > Int32(1):
                            mbarrier_wait(_ds_mbar, _s_phase)
                        _iter_idx = Int32(1)
                        while _iter_idx < _n_iters:
                            dispatch_load(
                                _ds_op,
                                _ds_pp,
                                _ds_0,
                                _ds_1,
                                _ds_2,
                                _ds_3,
                                _ds_4,
                                _ds_cached_config,
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
                        dispatch_store(
                            _ds_op,
                            _ds_pp,
                            _ds_0,
                            _ds_1,
                            _ds_2,
                            _ds_3,
                            _ds_4,
                            _ds_cached_config,
                        )
                        if const_expr(has_communicate):
                            dispatch_communicate(
                                _ds_op,
                                _ds_pp,
                                _ds_0,
                                _ds_1,
                                _ds_2,
                                _ds_3,
                                _ds_4,
                                _ds_cached_config,
                            )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        if const_expr(tracing):
                            for _si in range_constexpr(num_ops):
                                if _ds_op == Int32(_si):
                                    _store_lane = end_event_dynamic_raw_1(
                                        _tss,
                                        _trace_buf,
                                        Int32(trace_row_stride),
                                        _store_lane,
                                        Int32(trace_store_fmts[_si]),
                                        _ds_op,
                                    )

                        with cute.arch.elect_one():
                            signal_barriers(
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
                _cached_op_config = Int64(0)

                while mma_running == Int32(1):
                    slot = consume_ptr % Int32(num_pages)

                    _wn_phase = (consume_ptr // Int32(num_pages)) % Int32(2)
                    if const_expr(tracing):
                        _tw = trace_start()
                    mbarrier_wait(_work_notify_mbar(smem_base, slot), _wn_phase)

                    tile_info_ptr = smem_base + Int32(ring_state_offset) + slot * Int32(tile_info_bytes)
                    op_idx, _mma_lin = ld_shared_v2_b32(tile_info_ptr)

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

                        tile_0, tile_1, tile_2, tile_3, tile_4 = decompose_tile(op_idx, _mma_lin)

                        page_ptr = _get_page_ptr(smem_base, slot)
                        if op_idx != _cached_op_idx:
                            if const_expr(needs_op_config_load):
                                if _op_uses_config(op_idx) != Int32(0):
                                    _cached_op_config = ld_global_i64(op_configs_ptr, op_idx)
                                else:
                                    _cached_op_config = Int64(0)
                            _cached_op_idx = op_idx

                            if const_expr(needs_warp_transition):
                                for _ow in range_constexpr(num_ops):
                                    if op_idx == Int32(_ow):
                                        if warp_id >= Int32(per_op_warps[_ow]):
                                            setmaxregister_decrease(
                                                MIN_IDLE_REGS)
                                named_barrier_sync(
                                    Int32(1), Int32(num_compute_threads))
                                for _ow in range_constexpr(num_ops):
                                    if op_idx == Int32(_ow):
                                        if warp_id < Int32(per_op_warps[_ow]):
                                            setmaxregister_increase(
                                                mma_reg_count)

                        if const_expr(tracing):
                            _tc = trace_start()

                        dispatch_compute(
                            op_idx,
                            page_ptr,
                            tile_0,
                            tile_1,
                            tile_2,
                            tile_3,
                            tile_4,
                            _cached_op_config,
                        )

                        named_barrier_sync(Int32(1), Int32(num_compute_threads))

                        if const_expr(tracing):
                            for _i in range_constexpr(num_ops):
                                if op_idx == Int32(_i):
                                    _mma_lane = end_event_dynamic_raw_1(
                                        _tc,
                                        _trace_buf,
                                        Int32(trace_row_stride),
                                        _mma_lane,
                                        Int32(trace_compute_fmts[_i]),
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
        extra_exec_globals=None,
    ):
        """Build the PersistentKernel via source transformation.

        Extracts the kernel loop body, adds tensor and TMA params to dispatch
        call sites (if any), and exec-generates the PersistentKernel class
        with all params threaded through __call__ -> kernel -> _kernel_loop.
        """
        all_canonical = self._tensor_registry.canonical_names
        tensor_params = ", ".join(all_canonical)
        tensor_sig = self._signature_suffix(all_canonical)

        # TMA params that flow through kernel -> loop -> dispatch
        tma_registry = self._tma_registry
        all_tma_canonical = tma_registry.all_canonical_names
        tma_params = ", ".join(all_tma_canonical)
        tma_sig = self._signature_suffix(all_tma_canonical)

        # Peer TMA params for multi-GPU communication
        peer_tma_registry = self._peer_tma_registry
        all_peer_tma_canonical = peer_tma_registry.all_canonical_names
        peer_tma_params = ", ".join(all_peer_tma_canonical)
        peer_tma_sig = self._signature_suffix(all_peer_tma_canonical)

        # Combined extra params for dispatch calls (tensor + TMA + peer TMA)
        extra_dispatch_params = ", ".join(filter(None, [tensor_params, tma_params, peer_tma_params]))
        fn_source = self._build_kernel_loop_source(
            kernel_loop_fn,
            tensor_sig=tensor_sig,
            tma_sig=tma_sig,
            peer_tma_sig=peer_tma_sig,
            extra_dispatch_params=extra_dispatch_params,
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
            tma_components=tma_components,
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
            extra_exec_globals=self._kernel_extra_exec_globals(kernel_cfg, runtime),
        )

    def _make_cache_key(self) -> Tuple:
        """Create a cache key for the compiled kernel.

        The key includes all parameters that affect kernel compilation:
        - Op classes, their static dimensions, tensor dtypes, and tile counts
        - Config parameters (threads, pages, etc.)
        - Backward flag

        Tile counts are included because barrier formulas are baked into
        the kernel at compile time. Different tile counts produce different
        barrier formulas and instruction streams.
        """
        op_keys = []
        for op in self.ops:
            static_dims_tuple = tuple(sorted(op.static_dims.items())) if op.static_dims else ()
            # Include tensor dtypes - different dtypes require different compiled code
            # Convert dtypes to their names for hashing (CUTLASS dtype objects aren't hashable directly)
            tensor_dtypes_tuple = (
                tuple(sorted((k, v.__name__) for k, v in op.tensor_dtypes.items())) if op.tensor_dtypes else ()
            )
            # Include strides - different stride patterns require different compiled code
            strides_tuple = tuple(sorted((k, v) for k, v in op.tensor_strides.items())) if op.tensor_strides else ()
            tma_loads = frozenset(getattr(op.op_cls, "_TMA_LOADS", set()))
            tma_stores = frozenset(getattr(op.op_cls, "_TMA_STORES", set()))
            peer_stores = frozenset(getattr(op.op_cls, "_PEER_STORES", set()))
            op_keys.append(
                (
                    op.op_cls,
                    static_dims_tuple,
                    tensor_dtypes_tuple,
                    op.tile_counts,
                    strides_tuple,
                    tma_loads,
                    tma_stores,
                    peer_stores,
                )
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
            self.config.noinline,
            self.config.opt_level,
            self._builder.max_wait_deps,
        )

        # TMA descriptors are baked into the compiled kernel by cute.compile()
        # (they contain tensor base addresses). Include data_ptrs of TMA tensors
        # so different tensor allocations don't share stale TMA descriptors.
        tma_data_ptrs = ()
        if self._tma_registry.has_tma:
            tma_names = {d.tensor_canonical for d in self._tma_registry.descriptors}
            ptrs = []
            for canonical_name, tensor, dtype in self._tensor_registry.tensors:
                if canonical_name in tma_names:
                    ptrs.append(tensor.data_ptr())
            tma_data_ptrs = tuple(ptrs)

        return (tuple(op_keys), config_key, tma_data_ptrs)

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
                self._compiled_kernel = cute.compile(
                    self._compiled_kernel,
                    launch_state.instructions_ptr,
                    launch_state.barriers_ptr,
                    launch_state.op_configs_ptr,
                    launch_state.trace_buffer_ptr,
                    launch_state.wait_info_ptr,
                    self._num_instructions_i32,
                    *launch_state.cute_tensors,
                    *launch_state.tma_tensor_args,
                    *launch_state.peer_tma_tensor_args,
                    cu_stream,
                )
            finally:
                if self.config.noinline:
                    noinline_mod.uninstall()
                if _pipeline_patch is not None:
                    CutlassBaseDSL.preprocess_pipeline = _pipeline_patch

            # Store in class-level cache
            Megakernel._compiled_kernel_cache[cache_key] = self._compiled_kernel
            print("Compilation complete.")

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
        return _LaunchState(
            instructions_ptr=Int64(self._instructions_tensor.data_ptr()),
            barriers_ptr=Int64(self._barriers_tensor.data_ptr()),
            op_configs_ptr=Int64(self._op_configs_tensor.data_ptr()),
            wait_info_ptr=Int64(self._wait_info.data_ptr()),
            trace_buffer_ptr=Int64(0),
            cute_tensors=list(self._cute_tensors) if self._cute_tensors else [],
            tma_tensor_args=[ct for _, ct in self._tma_cute_tensors] if self._tma_cute_tensors else [],
            peer_tma_tensor_args=(
                [ct for _, _, ct in self._peer_tma_cute_tensors]
                if self._peer_tma_cute_tensors else []
            ),
        )

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
        self._compiled_kernel(
            launch_state.instructions_ptr,
            launch_state.barriers_ptr,
            launch_state.op_configs_ptr,
            trace_buffer_ptr if trace_buffer_ptr is not None else launch_state.trace_buffer_ptr,
            launch_state.wait_info_ptr,
            self._num_instructions_i32,
            *launch_state.cute_tensors,
            *launch_state.tma_tensor_args,
            *launch_state.peer_tma_tensor_args,
            stream,
        )

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

        bench_stream = torch.cuda.Stream()
        cu_stream = cuda.CUstream(bench_stream.cuda_stream)

        # Capture references to internal state (stable after compile)
        launch_state = self._build_launch_state()
        barriers_tensor = self._barriers_tensor

        def _setup():
            """Reset benchmark-visible runtime state before each launch."""
            if setup_fn is not None:
                setup_fn()
            barriers_tensor.zero_()

        def _launch():
            """Launch the already-compiled kernel on the benchmark stream."""
            self._launch_compiled_kernel(launch_state, cu_stream)

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
