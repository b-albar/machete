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
    TensorRegistry,
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
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_arrive,
    mbarrier_wait,
    mbarrier_try_wait,
    named_barrier_sync,
    nanosleep,
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

        # Tensor parameter mode: build registry and prepare cute.Tensors
        self._tensor_registry = TensorRegistry.from_ops(ops, backward=backward)
        self._cute_tensors: Optional[List] = None  # cute.Tensor objects for kernel params

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
        """Prepare cute.Tensor objects for kernel parameters.

        Creates 1D flat cute.Tensor views from the torch.Tensors in the
        tensor registry. These are passed as runtime kernel parameters,
        threaded through PersistentKernel -> kernel_loop -> dispatch -> phase_fn.
        """
        if self._cute_tensors is not None:
            return

        from cutlass.cute.runtime import from_dlpack

        self._cute_tensors = []
        for canonical_name, tensor, dtype in self._tensor_registry.tensors:
            # Flatten to 1D for backward compat with flat indexing (x[row_offset + i])
            # detach() needed because from_dlpack can't export gradient-tracking tensors
            flat = tensor.detach().contiguous().view(-1)
            ct = from_dlpack(flat, assumed_align=16)
            self._cute_tensors.append(ct)

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
            (dispatch_load_async, dispatch_compute, dispatch_store)
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
                    warp_specialized=True,
                )
                dma_init = op.op_cls.gen_init_source(
                    op.static_dims,
                    tensor_param_map=mapping,
                    backward=use_backward,
                    kernel_config=kernel_config,
                    tensor_dtypes=op.tensor_dtypes,
                    dma_warp_mode=True,
                )

            # Compile each phase with tensor params in signature
            # Compute phases get barrier replacement (named barrier instead of __syncthreads)
            # Load/store phases get DMA warp init (tidx remapped to 0-31, num_threads=32)
            if use_backward:
                load_fns.append(compile_backward_load_async(
                    op.op_cls, dma_init, tensor_param_names=tensor_args))
                compute_fns.append(compile_backward_compute(
                    op.op_cls, compute_init, tensor_param_names=tensor_args,
                    replace_barrier=True, num_compute_threads=num_compute_threads))
                store_fns.append(compile_backward_store(
                    op.op_cls, dma_init, tensor_param_names=tensor_args))
            else:
                load_fns.append(compile_load_async(
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

        dispatch_load_async = _build_dispatch(load_fns, "load_async")
        dispatch_compute = _build_dispatch(compute_fns, "compute")
        dispatch_store = _build_dispatch(store_fns, "store")

        return dispatch_load_async, dispatch_compute, dispatch_store

    def _build_exec_dispatch_fn(self, phase_fns, phase_name, op_tensor_args, all_canonical):
        """Build a dispatch function via exec() with tensor parameters.

        Generates source like:
            @cute.jit
            def dispatch_compute(op_idx, page_ptr, tile_m, tile_n, tile_l,
                                 op_config_ptr, t0, t1, t2, t3, t4):
                if op_idx == Int32(0):
                    _fn_0(page_ptr, tile_m, tile_n, tile_l, op_config_ptr, t0, t1, t2)
                if op_idx == Int32(1):
                    _fn_1(page_ptr, tile_m, tile_n, tile_l, op_config_ptr, t2, t3, t4)
        """
        import linecache
        import machete.megakernel.compile as compile_mod

        tensor_params = ", ".join(all_canonical)

        # Build dispatch branches
        lines = []
        for i, args in enumerate(op_tensor_args):
            args_str = ", ".join(args)
            if args_str:
                args_str = ", " + args_str
            lines.append(
                f"    if op_idx == Int32({i}):\n"
                f"        _fn_{i}(page_ptr, tile_m, tile_n, tile_l, op_config_ptr{args_str})"
            )

        body = "\n".join(lines) if lines else "    pass"
        fn_name = f"dispatch_{phase_name}"
        tensor_sig = f", {tensor_params}" if tensor_params else ""
        fn_source = (
            "@cute.jit\n"
            f"def {fn_name}(op_idx, page_ptr, tile_m, tile_n, tile_l, "
            f"op_config_ptr{tensor_sig}):\n"
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
        dispatch_load_async, dispatch_compute, dispatch_store,
        check_deps, wait_barriers, signal_barriers, get_page_ptr_fn,
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
                r'(dispatch_(?:load_async|compute|store)\([^)]*?config_ptr)\s*\)',
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
            "dispatch_load_async": dispatch_load_async,
            "dispatch_compute": dispatch_compute,
            "dispatch_store": dispatch_store,
            "check_deps": check_deps,
            "wait_barriers": wait_barriers,
            "signal_barriers": signal_barriers,
            "_get_page_ptr": get_page_ptr_fn,
            "ld_shared_i32": ld_shared_i32,
            "st_shared_i32": st_shared_i32,
            "load_instruction_to_smem": load_instruction_to_smem,
            "ld_global_i64": ld_global_i64,
            "cp_async_commit": cp_async_commit,
            "cp_async_wait_all": cp_async_wait_all,
            "cp_async_wait_group": cp_async_wait_group,
            "mbarrier_init": mbarrier_init,
            "mbarrier_init_fence": mbarrier_init_fence,
            "mbarrier_arrive": mbarrier_arrive,
            "mbarrier_wait": mbarrier_wait,
            "mbarrier_try_wait": mbarrier_try_wait,
            "named_barrier_sync": named_barrier_sync,
            "nanosleep": nanosleep,
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
        """Create the persistent warp-specialized kernel with mbarrier sync.

        Architecture:
        - DMA warp (last warp): instruction fetch, dependency checks, load_async,
          store, global barrier signaling, mbarrier management
        - MMA warps (all other warps): compute only, synced via mbarrier

        Two mbarriers per page:
        - load_done[p]: DMA arrives (1) -> MMA warps wait
        - compute_done[p]: MMA warps arrive (num_mma_warps) -> DMA waits
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
        mbarrier_offset = layout.mbarrier_offset

        # Warp specialization constants
        num_mma_warps = (threads_per_block // 32) - 1
        num_compute_threads = num_mma_warps * 32

        # Build pipelined dispatch functions (with barrier replacement for compute)
        dispatch_load_async, dispatch_compute, dispatch_store = self._build_pipelined_dispatch_fns()

        # Build barrier functions
        check_deps = self._build_dependency_checker()
        wait_barriers = self._build_wait_barriers()
        signal_barriers = self._build_signal_barriers()

        # Helper to get page pointer by index
        @cute.jit
        def _get_page_ptr(smem_base: Int32, page_idx: Int32) -> Int32:
            return smem_base + Int32(pages_start) + page_idx * Int32(aligned_page_size)

        # Helper to get load_done mbarrier smem address
        @cute.jit
        def _load_done_mbar(smem_base: Int32, page_idx: Int32) -> Int32:
            return smem_base + Int32(mbarrier_offset) + page_idx * Int32(8)

        # Helper to get compute_done mbarrier smem address
        @cute.jit
        def _compute_done_mbar(smem_base: Int32, page_idx: Int32) -> Int32:
            return smem_base + Int32(mbarrier_offset) + Int32(num_pages * 8) + page_idx * Int32(8)

        # Warp-specialized kernel loop
        def _kernel_loop_warp_specialized(
            instructions_ptr: Int64,
            barriers_ptr: Int64,
            op_configs_ptr: Int64,
            num_instructions: Int32,
            tidx: Int32,
            block_id: Int32,
            num_blocks: Int32,
            smem_base: Int32,
        ) -> None:
            """Warp-specialized persistent loop with mbarrier synchronization.

            DMA warp (last warp): fetches instructions, loads data, stores results.
            MMA warps (all others): compute only.
            Synchronization via mbarrier (load_done, compute_done per page).
            """
            warp_id = tidx // Int32(32)
            lane_id = tidx % Int32(32)
            is_dma_warp = warp_id == Int32(num_mma_warps)

            # Scratch pointer for instruction decode
            scratch_ptr = smem_base + Int32(scratch_offset)

            # ========== INIT: Initialize mbarriers (DMA warp thread 0) ==========
            if is_dma_warp:
                if lane_id == Int32(0):
                    for _init_page in range(num_pages):
                        # load_done: 1 arrival (DMA warp signals after load)
                        mbarrier_init(
                            _load_done_mbar(smem_base, Int32(_init_page)),
                            Int32(1),
                        )
                        # compute_done: num_mma_warps arrivals (one per MMA warp)
                        mbarrier_init(
                            _compute_done_mbar(smem_base, Int32(_init_page)),
                            Int32(num_mma_warps),
                        )
                    mbarrier_init_fence()

            # Sync all threads after mbarrier init (use named barrier 0 = full block)
            named_barrier_sync(Int32(0), Int32(threads_per_block))

            # ========== DMA WARP LOOP (fully non-blocking) ==========
            if is_dma_warp:
                acquire_ptr = Int32(0)   # Next ring slot to load into
                store_ptr = Int32(0)     # Next ring slot to store from
                next_instr_idx = block_id
                done_loading = Int32(0)  # All instructions consumed
                done_all = Int32(0)      # All work complete

                # Pending instruction state (fetched but deps not ready)
                pending = Int32(0)
                pending_op_idx = Int32(0)
                pending_tile_m = Int32(0)
                pending_tile_n = Int32(0)
                pending_tile_l = Int32(0)

                flags_ptr = smem_base + Int32(flags_offset)

                while done_all == Int32(0):
                    did_work = Int32(0)

                    # === PRIORITY 1: Try to store completed tiles (non-blocking) ===
                    if store_ptr < acquire_ptr:
                        s_page = store_ptr % Int32(num_pages)
                        s_phase = (store_ptr // Int32(num_pages)) % Int32(2)
                        s_ready = mbarrier_try_wait(_compute_done_mbar(smem_base, s_page), s_phase)
                        if s_ready == Int32(1):
                            s_page_ptr = _get_page_ptr(smem_base, s_page)
                            s_info_ptr = smem_base + Int32(ring_state_offset) + s_page * Int32(16)
                            s_op = ld_shared_i32(s_info_ptr)
                            s_tm = ld_shared_i32(s_info_ptr + 4)
                            s_tn = ld_shared_i32(s_info_ptr + 8)
                            s_tl = ld_shared_i32(s_info_ptr + 12)
                            s_config_ptr = ld_global_i64(op_configs_ptr, s_op)

                            dispatch_store(s_op, s_page_ptr, s_tm, s_tn, s_tl, s_config_ptr)
                            if lane_id == Int32(0):
                                signal_barriers(s_op, s_tm, s_tn, s_tl, barriers_ptr)
                            store_ptr = store_ptr + Int32(1)
                            did_work = Int32(1)

                    # === PRIORITY 2: Try to fetch + load (non-blocking) ===
                    if done_loading == Int32(0):
                        # Only proceed if we have a free page
                        if acquire_ptr - store_ptr < Int32(num_pages):
                            # Fetch next instruction if nothing pending
                            if pending == Int32(0):
                                if next_instr_idx < num_instructions:
                                    if lane_id == Int32(0):
                                        load_instruction_to_smem(instructions_ptr, next_instr_idx, scratch_ptr)
                                    named_barrier_sync(Int32(2), Int32(32))

                                    pending_op_idx = ld_shared_i32(scratch_ptr)
                                    if pending_op_idx == Int32(TileInstruction.END_MARKER):
                                        done_loading = Int32(1)
                                    if pending_op_idx != Int32(TileInstruction.END_MARKER):
                                        pending_tile_m = ld_shared_i32(scratch_ptr + 4)
                                        pending_tile_n = ld_shared_i32(scratch_ptr + 8)
                                        pending_tile_l = ld_shared_i32(scratch_ptr + 12)
                                        pending = Int32(1)
                                        did_work = Int32(1)
                                else:
                                    done_loading = Int32(1)

                            # Try to issue load for pending instruction (non-blocking dep check)
                            if pending == Int32(1):
                                if lane_id == Int32(0):
                                    dep_ok = check_deps(pending_op_idx, pending_tile_m, pending_tile_n, pending_tile_l, barriers_ptr)
                                    st_shared_i32(flags_ptr, dep_ok)
                                named_barrier_sync(Int32(2), Int32(32))
                                deps_ready = ld_shared_i32(flags_ptr)

                                if deps_ready == Int32(1):
                                    l_page = acquire_ptr % Int32(num_pages)
                                    l_page_ptr = _get_page_ptr(smem_base, l_page)

                                    # Write tile info for MMA warps
                                    l_info_ptr = smem_base + Int32(ring_state_offset) + l_page * Int32(16)
                                    if lane_id == Int32(0):
                                        st_shared_i32(l_info_ptr, pending_op_idx)
                                        st_shared_i32(l_info_ptr + 4, pending_tile_m)
                                        st_shared_i32(l_info_ptr + 8, pending_tile_n)
                                        st_shared_i32(l_info_ptr + 12, pending_tile_l)

                                    l_config_ptr = ld_global_i64(op_configs_ptr, pending_op_idx)
                                    dispatch_load_async(
                                        pending_op_idx, l_page_ptr,
                                        pending_tile_m, pending_tile_n,
                                        pending_tile_l, l_config_ptr)
                                    cp_async_commit()
                                    cp_async_wait_all()

                                    if lane_id == Int32(0):
                                        mbarrier_arrive(_load_done_mbar(smem_base, l_page))

                                    acquire_ptr = acquire_ptr + Int32(1)
                                    next_instr_idx = next_instr_idx + num_blocks
                                    pending = Int32(0)
                                    did_work = Int32(1)

                    # === CHECK DONE ===
                    if done_loading == Int32(1):
                        if pending == Int32(0):
                            if store_ptr >= acquire_ptr:
                                done_all = Int32(1)

                    # === NANOSLEEP when idle to save energy ===
                    if did_work == Int32(0):
                        if done_all == Int32(0):
                            nanosleep(Int32(8))

                # ===== SENTINEL: Wake MMA warps to exit =====
                sentinel_page = acquire_ptr % Int32(num_pages)
                sentinel_tile_info = smem_base + Int32(ring_state_offset) + sentinel_page * Int32(16)
                if lane_id == Int32(0):
                    st_shared_i32(sentinel_tile_info, Int32(TileInstruction.END_MARKER))
                    mbarrier_arrive(_load_done_mbar(smem_base, sentinel_page))

            # ========== MMA WARP LOOP ==========
            if warp_id < Int32(num_mma_warps):
                consume_ptr = Int32(0)
                mma_running = Int32(1)

                while mma_running == Int32(1):
                    page = consume_ptr % Int32(num_pages)
                    phase = (consume_ptr // Int32(num_pages)) % Int32(2)

                    # Wait for load to complete on this page
                    mbarrier_wait(_load_done_mbar(smem_base, page), phase)

                    # Read tile info
                    tile_info_ptr = smem_base + Int32(ring_state_offset) + page * Int32(16)
                    op_idx = ld_shared_i32(tile_info_ptr)

                    # Check for sentinel (END_MARKER = exit)
                    if op_idx == Int32(TileInstruction.END_MARKER):
                        mma_running = Int32(0)

                    if op_idx != Int32(TileInstruction.END_MARKER):
                        tile_m = ld_shared_i32(tile_info_ptr + 4)
                        tile_n = ld_shared_i32(tile_info_ptr + 8)
                        tile_l = ld_shared_i32(tile_info_ptr + 12)

                        page_ptr = _get_page_ptr(smem_base, page)
                        op_config_ptr = ld_global_i64(op_configs_ptr, op_idx)

                        # Compute
                        dispatch_compute(op_idx, page_ptr, tile_m, tile_n, tile_l, op_config_ptr)

                        # Sync MMA warps post-compute (named barrier 1, compute threads only)
                        named_barrier_sync(Int32(1), Int32(num_compute_threads))

                        # Signal compute_done for this page (one arrive per MMA warp)
                        if lane_id == Int32(0):
                            mbarrier_arrive(_compute_done_mbar(smem_base, page))

                        consume_ptr = consume_ptr + Int32(1)

        if tracing:
            raise NotImplementedError("Tracing not yet supported with warp-specialized kernel")

        return self._build_kernel(
            _kernel_loop_warp_specialized,
            dispatch_load_async, dispatch_compute, dispatch_store,
            check_deps, wait_barriers, signal_barriers, _get_page_ptr,
            num_sms, threads_per_block, smem_size,
            num_pages, scratch_offset, flags_offset, ring_state_offset,
            extra_exec_globals={
                "_load_done_mbar": _load_done_mbar,
                "_compute_done_mbar": _compute_done_mbar,
                "num_mma_warps": num_mma_warps,
                "num_compute_threads": num_compute_threads,
                "threads_per_block": threads_per_block,
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
            op_keys.append((op.op_cls, static_dims_tuple, tensor_dtypes_tuple, tile_counts))

        config_key = (
            self.config.threads_per_block,
            self.config.page_size,
            self.config.num_pages,
            self.config.tracing,
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
    # Utilities
    "get_smem_base_ptr",
]
