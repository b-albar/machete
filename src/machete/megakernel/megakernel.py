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

from .ops import (
    ScheduledOp,
    TensorRegistry,
    InstructionStreamBuilder,
    TileInstruction,
    validate_op_compatibility,
)
from .compile import (
    compile_load,
    compile_compute,
    compile_store,
    compile_backward_load,
    compile_backward_compute,
    compile_backward_store,
)
from .interpreter import (
    global_barrier_signal,
    check_barrier_ready,
    load_instruction_to_smem,
    ld_global_i64,
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_arrive,
    mbarrier_wait,
    mbarrier_try_wait,
    named_barrier_sync,
    get_smem_base_ptr,
)
from .paged_memory import (
    PAGE_SIZE,
    NPageLayout,
    st_shared_i32,
    ld_shared_i32,
)


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
    dma_reg_count: int = 40
    mma_reg_count: int = 232

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

        # Tensor parameter mode: build registry, validate compatibility, prepare tensors
        self._tensor_registry = TensorRegistry.from_ops(ops, backward=backward)
        validate_op_compatibility(ops, self._tensor_registry)
        self._cute_tensors: Optional[List] = None  # torch.Tensor objects for kernel params

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

    def _prepare_cute_tensors(self) -> None:
        """Prepare tensor objects for kernel parameters.

        Passes torch.Tensor objects directly — CuTe DSL's TensorAdapter
        (registered for torch.Tensor) auto-converts via from_dlpack +
        mark_layout_dynamic, preserving N-D shape with dynamic layout.

        Tensors are threaded as runtime parameters through:
        PersistentKernel -> kernel_loop -> dispatch -> phase_fn.
        """
        if self._cute_tensors is not None:
            return

        self._cute_tensors = []
        for canonical_name, tensor, dtype in self._tensor_registry.tensors:
            # detach() for gradient-tracking tensors, contiguous() for layout.
            # Pass N-D torch.Tensor directly — TensorAdapter auto-converts
            # via from_dlpack + mark_layout_dynamic, preserving shape for TMA.
            # Ops that need flat 1D access get a flat alias in init source.
            t = tensor.detach().contiguous()
            self._cute_tensors.append(t)

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

        Each phase function receives an op-specific subset of tensor params
        (e.g., t0, t1, t2 for RMSNorm). The dispatch functions accept ALL
        canonical tensor names and route the correct subset to each phase fn.

        Returns:
            (dispatch_load, dispatch_compute, dispatch_store)
        """
        ops = self.ops
        use_backward = self.backward
        registry = self._tensor_registry
        all_canonical = registry.canonical_names  # ['t0', 't1', ...]

        load_fns = []
        compute_fns = []
        store_fns = []
        op_tensor_args = []  # Per-op list of canonical tensor arg names

        # Warp-specialized mode: DMA warp is last warp, compute threads = rest
        threads_per_block = self.config.threads_per_block
        num_compute_threads = threads_per_block - 32  # Exclude DMA warp

        for i, op in enumerate(ops):
            # Build tensor param map: local name -> canonical name
            mapping = registry.op_mappings[i]
            # Get canonical names in declaration order for this op
            tensor_args = registry.get_op_tensor_args(i, op.op_cls, backward=use_backward)
            op_tensor_args.append(tensor_args)

            kernel_config = {"threads_per_row": num_compute_threads}
            has_init = hasattr(op.op_cls, "gen_init_source") and (op.static_dims or op.tensor_dtypes)

            # Separate init sources: compute (MMA warps) vs load/store (DMA warp)
            compute_init = None
            dma_init = None
            if has_init:
                compute_init = op.op_cls.gen_init_source(
                    op.static_dims,
                    tensor_param_map=mapping,
                    backward=use_backward,
                    kernel_config=kernel_config,
                    tensor_dtypes=op.tensor_dtypes,
                    tensor_strides=op.tensor_strides,
                    warp_specialized=True,
                )
                dma_init = op.op_cls.gen_init_source(
                    op.static_dims,
                    tensor_param_map=mapping,
                    backward=use_backward,
                    kernel_config=kernel_config,
                    tensor_dtypes=op.tensor_dtypes,
                    tensor_strides=op.tensor_strides,
                    dma_warp_mode=True,
                )

            # Compile each phase with tensor params in signature
            # Compute phases get barrier replacement (named barrier instead of __syncthreads)
            # Load/store phases get DMA warp init (tidx remapped to 0-31, num_threads=32)
            if use_backward:
                load_fns.append(compile_backward_load(
                    op.op_cls, dma_init, tensor_param_names=tensor_args))
                compute_fns.append(compile_backward_compute(
                    op.op_cls, compute_init, tensor_param_names=tensor_args,
                    replace_barrier=True, num_compute_threads=num_compute_threads))
                store_fns.append(compile_backward_store(
                    op.op_cls, dma_init, tensor_param_names=tensor_args))
            else:
                load_fns.append(compile_load(
                    op.op_cls, dma_init, tensor_param_names=tensor_args))
                compute_fns.append(compile_compute(
                    op.op_cls, compute_init, tensor_param_names=tensor_args,
                    replace_barrier=True, num_compute_threads=num_compute_threads))
                store_fns.append(compile_store(
                    op.op_cls, dma_init, tensor_param_names=tensor_args))

        # Generate dispatch functions via exec() — each accepts ALL canonical
        # tensor names and routes the op-specific subset to each phase fn.
        def _build_dispatch(phase_fns, phase_name):
            return self._build_exec_dispatch_fn(
                phase_fns, phase_name, op_tensor_args, all_canonical)

        dispatch_load = _build_dispatch(load_fns, "load")
        dispatch_compute = _build_dispatch(compute_fns, "compute")
        dispatch_store = _build_dispatch(store_fns, "store")

        return dispatch_load, dispatch_compute, dispatch_store

    def _build_exec_dispatch_fn(self, phase_fns, phase_name, op_tensor_args,
                                all_canonical):
        """Build a dispatch function via exec() with tensor parameters.

        For load phase (phase_name="load"), all compiled load functions
        receive work_mbar uniformly. Sync ops have mbarrier_arrive baked
        into their compiled body by compile_load; async ops signal it
        themselves via mbarrier_arrive_expect_tx + cute.copy.

        Generates source like:
            @cute.jit
            def dispatch_load(op_idx, page_ptr, tile_m, tile_n, tile_l,
                              op_config_ptr, work_mbar, t0, t1, t2):
                if op_idx == Int32(0):
                    _fn_0(page_ptr, tile_m, tile_n, tile_l, op_config_ptr, work_mbar, t0, t1)
                if op_idx == Int32(1):
                    _fn_1(page_ptr, tile_m, tile_n, tile_l, op_config_ptr, work_mbar, t2)
        """
        import linecache
        import machete.megakernel.compile as compile_mod

        is_load = phase_name == "load"
        tensor_params = ", ".join(all_canonical)

        # Build dispatch branches (if/elif chain so only the matching op executes)
        lines = []
        for i, args in enumerate(op_tensor_args):
            args_str = ", ".join(args)
            if args_str:
                args_str = ", " + args_str

            keyword = "if" if i == 0 else "elif"
            if is_load:
                lines.append(
                    f"    {keyword} op_idx == Int32({i}):\n"
                    f"        _fn_{i}(page_ptr, tile_m, tile_n, tile_l, op_config_ptr, work_mbar{args_str})"
                )
            else:
                lines.append(
                    f"    {keyword} op_idx == Int32({i}):\n"
                    f"        _fn_{i}(page_ptr, tile_m, tile_n, tile_l, op_config_ptr{args_str})"
                )

        body = "\n".join(lines) if lines else "    pass"
        fn_name = f"dispatch_{phase_name}"
        tensor_sig = f", {tensor_params}" if tensor_params else ""
        work_mbar_sig = ", work_mbar" if is_load else ""
        fn_source = (
            "@cute.jit\n"
            f"def {fn_name}(op_idx, page_ptr, tile_m, tile_n, tile_l, "
            f"op_config_ptr{work_mbar_sig}{tensor_sig}):\n"
            f"{body}\n"
        )

        exec_globals = {"cute": cute, "Int32": Int32, "Int64": Int64}
        for i, fn in enumerate(phase_fns):
            exec_globals[f"_fn_{i}"] = fn

        # Use compile module's counter for unique filenames
        compile_mod._compile_counter += 1
        unique_filename = f"<{fn_name}>_{compile_mod._compile_counter}"

        linecache.cache[unique_filename] = (
            len(fn_source), None, fn_source.splitlines(True), unique_filename,
        )
        compile_mod._linecache_entries.append(unique_filename)

        code = compile(fn_source, unique_filename, "exec")
        exec(code, exec_globals)
        return exec_globals[fn_name]

    def _build_kernel(
        self, kernel_loop_fn,
        dispatch_load, dispatch_compute, dispatch_store,
        signal_barriers, get_page_ptr_fn,
        num_sms, threads_per_block, smem_size,
        num_pages, scratch_offset, flags_offset, ring_state_offset,
        extra_exec_globals=None,
    ):
        """Build the PersistentKernel via source transformation.

        Extracts the kernel loop body, adds tensor params to dispatch call
        sites (if any tensors in registry), and exec-generates the
        PersistentKernel class with tensor params threaded through
        __call__ -> kernel -> _kernel_loop.
        """
        import re
        import textwrap
        import linecache
        import machete.megakernel.compile as compile_mod
        from .compile import _extract_body

        all_canonical = self._tensor_registry.canonical_names
        tensor_params = ", ".join(all_canonical)
        tensor_sig = f", {tensor_params}" if tensor_params else ""

        # Extract kernel loop body and add tensor args to dispatch calls
        body = _extract_body(kernel_loop_fn)
        if tensor_params:
            body = re.sub(
                r'(dispatch_(?:load|compute|store)\([^)]+)\)',
                lambda m: m.group(1) + ', ' + tensor_params + ')',
                body,
            )

        # Build kernel loop with tensor params in signature
        fn_source = (
            "@cute.jit\n"
            "def _kernel_loop(instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            "                  num_instructions, tidx, block_id, num_blocks,\n"
            f"                  smem_base{tensor_sig}):\n"
            + textwrap.indent(body, "    ")
            + "\n"
        )

        # Exec globals: all references used in the kernel loop body
        exec_globals = {
            "cute": cute, "Int32": Int32, "Int64": Int64,
            "TileInstruction": TileInstruction,
            "dispatch_load": dispatch_load,
            "dispatch_compute": dispatch_compute,
            "dispatch_store": dispatch_store,
            "signal_barriers": signal_barriers,
            "_get_page_ptr": get_page_ptr_fn,
            "ld_shared_i32": ld_shared_i32,
            "st_shared_i32": st_shared_i32,
            "load_instruction_to_smem": load_instruction_to_smem,
            "ld_global_i64": ld_global_i64,
            "mbarrier_init": mbarrier_init,
            "mbarrier_init_fence": mbarrier_init_fence,
            "mbarrier_arrive": mbarrier_arrive,
            "mbarrier_wait": mbarrier_wait,
            "named_barrier_sync": named_barrier_sync,
            "num_pages": num_pages,
            "scratch_offset": scratch_offset,
            "flags_offset": flags_offset,
            "ring_state_offset": ring_state_offset,
        }
        if extra_exec_globals:
            exec_globals.update(extra_exec_globals)

        compile_mod._compile_counter += 1
        kl_filename = f"<kernel_loop>_{compile_mod._compile_counter}"
        linecache.cache[kl_filename] = (
            len(fn_source), None, fn_source.splitlines(True), kl_filename,
        )
        compile_mod._linecache_entries.append(kl_filename)

        code = compile(fn_source, kl_filename, "exec")
        exec(code, exec_globals)
        kernel_loop = exec_globals["_kernel_loop"]

        # Build PersistentKernel via exec with tensor params threaded through
        pk_source = (
            "class PersistentKernel:\n"
            "    def __init__(self):\n"
            f"        self.num_sms = {num_sms}\n"
            f"        self.threads_per_block = {threads_per_block}\n"
            f"        self.smem_size = {smem_size}\n"
            "\n"
            "    @cute.jit\n"
            "    def __call__(self, instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            f"                 trace_buffer_ptr, num_instructions{tensor_sig}, stream):\n"
            "        self.kernel(instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            f"                    num_instructions{tensor_sig}).launch(\n"
            "            grid=[self.num_sms, 1, 1],\n"
            "            block=[self.threads_per_block, 1, 1],\n"
            "            smem=self.smem_size,\n"
            "            stream=stream,\n"
            "        )\n"
            "\n"
            "    @cute.kernel\n"
            "    def kernel(self, instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            f"               num_instructions{tensor_sig}):\n"
            "        tidx = cute.arch.thread_idx()[0]\n"
            "        block_id = cute.arch.block_idx()[0]\n"
            "        num_blocks = cute.arch.grid_dim()[0]\n"
            "        smem_base = get_smem_base_ptr()\n"
            "        _kernel_loop(instructions_ptr, barriers_ptr, op_configs_ptr,\n"
            "                     num_instructions, tidx, block_id, num_blocks,\n"
            f"                     smem_base{tensor_sig})\n"
        )

        pk_globals = {
            "cute": cute,
            "get_smem_base_ptr": get_smem_base_ptr,
            "_kernel_loop": kernel_loop,
        }

        compile_mod._compile_counter += 1
        pk_filename = f"<persistent_kernel>_{compile_mod._compile_counter}"
        linecache.cache[pk_filename] = (
            len(pk_source), None, pk_source.splitlines(True), pk_filename,
        )
        compile_mod._linecache_entries.append(pk_filename)

        code = compile(pk_source, pk_filename, "exec")
        exec(code, pk_globals)
        return pk_globals["PersistentKernel"]()

    def _build_check_barriers(self):
        """Build non-blocking function to check if all barriers are ready.

        Returns a @cute.jit function that checks (without blocking) whether
        all wait barriers for the given tile are satisfied.
        Uses if/elif chain so only the matching op's barriers are evaluated.

        Returns Int32(1) if all deps ready, Int32(0) otherwise.
        """
        return self._build_barrier_fn("check")

    def _build_signal_barriers(self):
        """Build function to signal barriers after tile completion.

        Returns a @cute.jit function that signals all barriers for the
        given tile after it completes. Uses if/elif chain so only the
        matching op's barriers are evaluated.
        """
        return self._build_barrier_fn("signal")

    def _build_barrier_fn(self, mode: str):
        """Generate a barrier function via exec() with if/elif dispatch.

        Args:
            mode: "check" (non-blocking) or "signal"
        """
        import linecache
        import machete.megakernel.compile as compile_mod

        barrier_formulas = self._builder.get_op_barrier_formulas()

        if mode == "check":
            fn_name = "check_barriers"
            ret_type = " -> Int32"
            preamble = "    all_ready = Int32(1)\n"
            epilogue = "    return all_ready\n"
        else:
            fn_name = "signal_barriers"
            ret_type = ""
            preamble = ""
            epilogue = ""

        # Build if/elif branches for each op
        branches = []
        first = True
        for i, (wait_formulas, signal_formulas) in barrier_formulas.items():
            keyword = "if" if first else "elif"
            first = False

            formulas = wait_formulas if mode == "check" else signal_formulas
            if not formulas:
                branches.append(f"    {keyword} op_idx == Int32({i}):\n        pass")
                continue

            lines = [f"    {keyword} op_idx == Int32({i}):"]
            for wf in formulas:
                if mode == "signal":
                    lines.append(
                        f"        barrier_idx = ("
                        f"Int32({wf.base})"
                        f" + (Int32({wf.coeff_m}) * tile_m) // Int32({wf.div_m})"
                        f" + (Int32({wf.coeff_n}) * tile_n) // Int32({wf.div_n})"
                        f" + (Int32({wf.coeff_l}) * tile_l) // Int32({wf.div_l})"
                        f")"
                    )
                    lines.append("        global_barrier_signal(barriers_ptr, barrier_idx)")
                else:  # check
                    lines.append(
                        f"        _linear = ("
                        f"Int32({wf.coeff_m}) * tile_m"
                        f" + Int32({wf.coeff_n}) * tile_n"
                        f" + Int32({wf.coeff_l}) * tile_l"
                        f")"
                    )
                    lines.append(f"        if _linear < Int32({wf.guard_max}):")
                    lines.append(
                        f"            barrier_idx = ("
                        f"Int32({wf.base})"
                        f" + (Int32({wf.coeff_m}) * tile_m) // Int32({wf.div_m})"
                        f" + (Int32({wf.coeff_n}) * tile_n) // Int32({wf.div_n})"
                        f" + (Int32({wf.coeff_l}) * tile_l) // Int32({wf.div_l})"
                        f")"
                    )
                    lines.append(
                        f"            r = check_barrier_ready("
                        f"barriers_ptr, barrier_idx, Int32({wf.expected}))"
                    )
                    lines.append("            if r == Int32(0):")
                    lines.append("                all_ready = Int32(0)")

            branches.append("\n".join(lines))

        body = "\n".join(branches) if branches else "    pass"

        fn_source = (
            "@cute.jit\n"
            f"def {fn_name}(op_idx, tile_m, tile_n, tile_l, barriers_ptr){ret_type}:\n"
            f"{preamble}"
            f"{body}\n"
            f"{epilogue}"
        )

        exec_globals = {
            "cute": cute, "Int32": Int32, "Int64": Int64,
            "global_barrier_signal": global_barrier_signal,
            "check_barrier_ready": check_barrier_ready,
        }

        compile_mod._compile_counter += 1
        unique_filename = f"<{fn_name}>_{compile_mod._compile_counter}"
        linecache.cache[unique_filename] = (
            len(fn_source), None, fn_source.splitlines(True), unique_filename,
        )
        compile_mod._linecache_entries.append(unique_filename)

        code = compile(fn_source, unique_filename, "exec")
        exec(code, exec_globals)
        return exec_globals[fn_name]

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
        - work_notify[slot]: DMA arrives (1) -> MMA warps wait
        - compute_done[slot]: MMA warps arrive (num_mma_warps) -> DMA polls
        """
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

        # Mbarrier sub-offsets (work_notify at position 0, compute_done at position N)
        work_notify_mbar_offset_0 = layout.work_notify_mbar_offset(0)
        compute_done_mbar_offset_0 = layout.compute_done_mbar_offset(0)

        # Warp specialization constants
        num_mma_warps = (threads_per_block // 32) - 1
        num_compute_threads = num_mma_warps * 32
        dma_reg_count = self.config.dma_reg_count
        mma_reg_count = self.config.mma_reg_count

        from cutlass.cute.arch import setmaxregister_increase, setmaxregister_decrease

        # Build pipelined dispatch functions (with barrier replacement for compute)
        dispatch_load, dispatch_compute, dispatch_store = self._build_pipelined_dispatch_fns()

        # Build barrier functions
        check_barriers = self._build_check_barriers()
        signal_barriers = self._build_signal_barriers()

        # Helper to get page pointer by index
        @cute.jit
        def _get_page_ptr(smem_base: Int32, page_idx: Int32) -> Int32:
            return smem_base + Int32(pages_start) + page_idx * Int32(aligned_page_size)

        # Helper to get work_notify mbarrier smem address (per slot)
        @cute.jit
        def _work_notify_mbar(smem_base: Int32, slot: Int32) -> Int32:
            return smem_base + Int32(work_notify_mbar_offset_0) + slot * Int32(8)

        # Helper to get compute_done mbarrier smem address (per slot)
        @cute.jit
        def _compute_done_mbar(smem_base: Int32, page_idx: Int32) -> Int32:
            return smem_base + Int32(compute_done_mbar_offset_0) + page_idx * Int32(8)

        # Ring buffer kernel loop
        def _kernel_loop_ring(
            instructions_ptr: Int64,
            barriers_ptr: Int64,
            op_configs_ptr: Int64,
            num_instructions: Int32,
            tidx: Int32,
            block_id: Int32,
            num_blocks: Int32,
            smem_base: Int32,
        ) -> None:
            """Warp-specialized ring buffer loop.

            DMA warp (last warp): in-order fetch, blocking dep wait, load/store.
            MMA warps: compute from ring buffer slots (page == slot).
            Mbarrier phases alternate 0/1 with each use (hardware auto-reset).
            """
            warp_id = tidx // Int32(32)
            lane_id = tidx % Int32(32)
            is_dma_warp = warp_id == Int32(num_mma_warps)

            # Register reallocation: DMA warp frees registers, MMA warps gain
            if is_dma_warp:
                setmaxregister_decrease(dma_reg_count)
            if warp_id < Int32(num_mma_warps):
                setmaxregister_increase(mma_reg_count)

            # Scratch pointer for instruction decode
            scratch_ptr = smem_base + Int32(scratch_offset)
            flags_ptr = smem_base + Int32(flags_offset)

            # ========== INIT (DMA warp thread 0) ==========
            if is_dma_warp:
                if lane_id == Int32(0):
                    # Clear done flag
                    st_shared_i32(flags_ptr, Int32(0))
                    for _ip in range(num_pages):
                        mbarrier_init(
                            _work_notify_mbar(smem_base, Int32(_ip)),
                            Int32(1),
                        )
                        mbarrier_init(
                            _compute_done_mbar(smem_base, Int32(_ip)),
                            Int32(num_mma_warps),
                        )
                    mbarrier_init_fence()

            # Sync all threads after init (named barrier 0 = full block)
            named_barrier_sync(Int32(0), Int32(threads_per_block))

            # ========== DMA WARP LOOP (non-blocking polling) ==========
            if is_dma_warp:
                produce_idx = Int32(0)
                store_idx = Int32(0)
                next_instr_idx = block_id
                done = Int32(0)
                has_pending_store = Int32(0)
                _ps_op = Int32(0)
                _ps_m = Int32(0)
                _ps_n = Int32(0)
                _ps_l = Int32(0)

                # Instruction cache: load once from gmem, reuse across
                # iterations when barriers aren't ready yet.
                _instr_cached = Int32(0)
                _ic_op = Int32(0)
                _ic_m = Int32(0)
                _ic_n = Int32(0)
                _ic_l = Int32(0)

                while done == Int32(0):
                    if lane_id == Int32(0):
                        iter_done = Int32(0)

                        # STEP 1: COMPLETE PREVIOUS STORE
                        # Wait for S2G copy issued last iteration to finish,
                        # then signal barriers and free the page.
                        if has_pending_store == Int32(1):
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                            signal_barriers(_ps_op, _ps_m, _ps_n, _ps_l, barriers_ptr)
                            store_idx = store_idx + Int32(1)
                            has_pending_store = Int32(0)

                        # STEP 2: TRY PRODUCE (G2S load) — runs before store
                        # so the async G2S copy starts earlier and has more
                        # time to complete before MMA warps need the data.
                        _can_produce = Int32(0)
                        if (produce_idx - store_idx) < Int32(num_pages):
                            if next_instr_idx < num_instructions:
                                _can_produce = Int32(1)
                        if _can_produce == Int32(1):
                            # Load instruction once; reuse cached copy on
                            # subsequent iterations while deps aren't ready.
                            if _instr_cached == Int32(0):
                                load_instruction_to_smem(instructions_ptr, next_instr_idx, scratch_ptr)
                                _ic_op = ld_shared_i32(scratch_ptr)
                                if _ic_op == Int32(TileInstruction.END_MARKER):
                                    next_instr_idx = num_instructions
                                if _ic_op != Int32(TileInstruction.END_MARKER):
                                    _ic_m = ld_shared_i32(scratch_ptr + 4)
                                    _ic_n = ld_shared_i32(scratch_ptr + 8)
                                    _ic_l = ld_shared_i32(scratch_ptr + 12)
                                    _instr_cached = Int32(1)
                            if _instr_cached == Int32(1):
                                if check_barriers(_ic_op, _ic_m, _ic_n, _ic_l, barriers_ptr) == Int32(1):
                                    _p_slot = produce_idx % Int32(num_pages)
                                    _p_ti = smem_base + Int32(ring_state_offset) + _p_slot * Int32(16)
                                    st_shared_i32(_p_ti, _ic_op)
                                    st_shared_i32(_p_ti + 4, _ic_m)
                                    st_shared_i32(_p_ti + 8, _ic_n)
                                    st_shared_i32(_p_ti + 12, _ic_l)
                                    _p_config_ptr = ld_global_i64(op_configs_ptr, _ic_op)
                                    _p_pp = _get_page_ptr(smem_base, _p_slot)
                                    _p_wn = _work_notify_mbar(smem_base, _p_slot)
                                    dispatch_load(_ic_op, _p_pp, _ic_m, _ic_n, _ic_l, _p_config_ptr, _p_wn)
                                    produce_idx = produce_idx + Int32(1)
                                    next_instr_idx = next_instr_idx + num_blocks

                                    # Prefetch next instruction while G2S copy
                                    # is in flight — hides gmem fetch latency.
                                    _instr_cached = Int32(0)
                                    if next_instr_idx < num_instructions:
                                        load_instruction_to_smem(
                                            instructions_ptr, next_instr_idx, scratch_ptr,
                                        )
                                        _ic_op = ld_shared_i32(scratch_ptr)
                                        if _ic_op == Int32(TileInstruction.END_MARKER):
                                            next_instr_idx = num_instructions
                                        if _ic_op != Int32(TileInstruction.END_MARKER):
                                            _ic_m = ld_shared_i32(scratch_ptr + 4)
                                            _ic_n = ld_shared_i32(scratch_ptr + 8)
                                            _ic_l = ld_shared_i32(scratch_ptr + 12)
                                            _instr_cached = Int32(1)

                        # STEP 3: TRY ISSUE STORE (S2G copy) — after produce
                        # so the G2S from step 2 starts before the S2G here.
                        if has_pending_store == Int32(0):
                            if store_idx < produce_idx:
                                _s_slot = store_idx % Int32(num_pages)
                                _s_phase = (store_idx // Int32(num_pages)) % Int32(2)
                                if mbarrier_try_wait(_compute_done_mbar(smem_base, _s_slot), _s_phase) == Int32(1):
                                    _s_ti = smem_base + Int32(ring_state_offset) + _s_slot * Int32(16)
                                    _s_op = ld_shared_i32(_s_ti)
                                    _s_m = ld_shared_i32(_s_ti + 4)
                                    _s_n = ld_shared_i32(_s_ti + 8)
                                    _s_l = ld_shared_i32(_s_ti + 12)
                                    _s_config_ptr = ld_global_i64(op_configs_ptr, _s_op)
                                    _s_pp = _get_page_ptr(smem_base, _s_slot)
                                    dispatch_store(_s_op, _s_pp, _s_m, _s_n, _s_l, _s_config_ptr)
                                    cute.arch.cp_async_bulk_commit_group()
                                    _ps_op = _s_op
                                    _ps_m = _s_m
                                    _ps_n = _s_n
                                    _ps_l = _s_l
                                    has_pending_store = Int32(1)

                        # STEP 4: DONE — all fetched, all stored, nothing pending
                        if next_instr_idx >= num_instructions:
                            if store_idx >= produce_idx:
                                if has_pending_store == Int32(0):
                                    iter_done = Int32(1)
                        st_shared_i32(flags_ptr, iter_done)

                    done = ld_shared_i32(flags_ptr)

                # SENTINEL: wake MMA warps to exit
                if lane_id == Int32(0):
                    _sent = produce_idx % Int32(num_pages)
                    st_shared_i32(
                        smem_base + Int32(ring_state_offset) + _sent * Int32(16),
                        Int32(TileInstruction.END_MARKER),
                    )
                    mbarrier_arrive(_work_notify_mbar(smem_base, _sent))

            # ========== MMA WARP LOOP ==========
            if warp_id < Int32(num_mma_warps):
                consume_ptr = Int32(0)
                mma_running = Int32(1)

                while mma_running == Int32(1):
                    slot = consume_ptr % Int32(num_pages)

                    # Wait for work (phase alternates with each ring buffer pass)
                    _wn_phase = (consume_ptr // Int32(num_pages)) % Int32(2)
                    mbarrier_wait(_work_notify_mbar(smem_base, slot), _wn_phase)

                    # Read tile info (page == slot, no work queue indirection)
                    tile_info_ptr = (
                        smem_base + Int32(ring_state_offset) + slot * Int32(16)
                    )
                    op_idx = ld_shared_i32(tile_info_ptr)

                    # Check for sentinel (END_MARKER = exit)
                    if op_idx == Int32(TileInstruction.END_MARKER):
                        mma_running = Int32(0)

                    if op_idx != Int32(TileInstruction.END_MARKER):
                        tile_m = ld_shared_i32(tile_info_ptr + 4)
                        tile_n = ld_shared_i32(tile_info_ptr + 8)
                        tile_l = ld_shared_i32(tile_info_ptr + 12)

                        page_ptr = _get_page_ptr(smem_base, slot)
                        op_config_ptr = ld_global_i64(op_configs_ptr, op_idx)

                        # Compute
                        dispatch_compute(op_idx, page_ptr, tile_m, tile_n, tile_l, op_config_ptr)

                        # Sync MMA warps post-compute
                        named_barrier_sync(Int32(1), Int32(num_compute_threads))

                        # Signal compute_done for this slot
                        if lane_id == Int32(0):
                            mbarrier_arrive(_compute_done_mbar(smem_base, slot))

                        consume_ptr = consume_ptr + Int32(1)

        if tracing:
            raise NotImplementedError("Tracing not yet supported with warp-specialized kernel")

        return self._build_kernel(
            _kernel_loop_ring,
            dispatch_load, dispatch_compute, dispatch_store,
            signal_barriers, _get_page_ptr,
            num_sms, threads_per_block, smem_size,
            num_pages, scratch_offset, flags_offset, ring_state_offset,
            extra_exec_globals={
                "_work_notify_mbar": _work_notify_mbar,
                "_compute_done_mbar": _compute_done_mbar,
                "check_barriers": check_barriers,
                "mbarrier_try_wait": mbarrier_try_wait,
                "num_mma_warps": num_mma_warps,
                "num_compute_threads": num_compute_threads,
                "threads_per_block": threads_per_block,
                "setmaxregister_increase": setmaxregister_increase,
                "setmaxregister_decrease": setmaxregister_decrease,
                "dma_reg_count": dma_reg_count,
                "mma_reg_count": mma_reg_count,
            },
        )

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
            # Include strides - different stride patterns require different compiled code
            strides_tuple = (
                tuple(sorted((k, v) for k, v in op.tensor_strides.items()))
                if op.tensor_strides else ()
            )
            op_keys.append((op.op_cls, static_dims_tuple, tensor_dtypes_tuple,
                            tile_counts, strides_tuple))

        config_key = (
            self.config.num_sms,
            self.config.threads_per_block,
            self.config.page_size,
            self.config.num_pages,
            self.config.tracing,
            self.config.dma_reg_count,
            self.config.mma_reg_count,
        )

        # Tensors are runtime parameters (not compile-time constants), so
        # the cache key does NOT include tensor addresses. Same shapes/dtypes
        # (captured in static_dims and tensor_dtypes above) share compiled kernels.
        return (tuple(op_keys), config_key, self.backward)

    def compile(self) -> None:
        """Compile the kernel without running it.

        Triggers JIT compilation so that subsequent run() calls have no
        compilation overhead. Safe to call multiple times (no-op after first).

        Uses a class-level cache to avoid recompilation when multiple Megakernel
        instances have the same configuration (same ops, static_dims, config).
        """
        # Both are idempotent (check for None internally)
        self._prepare_tensors()
        self._prepare_cute_tensors()

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
                *self._cute_tensors,
                cu_stream,
            )

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

    def run(self, stream=None, sync: bool = True) -> None:
        """Run the persistent megakernel.

        Args:
            stream: CUDA stream (optional). If None, uses current stream.
            sync: If True (default), synchronize after launch. Set to False
                for benchmarking or when managing synchronization externally.
        """
        # Validate tensors haven't changed since schedule()
        self._validate_tensors()

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
            *self._cute_tensors,
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

        cute_tensors = list(self._cute_tensors) if self._cute_tensors else []

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
                *cute_tensors,
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
]
