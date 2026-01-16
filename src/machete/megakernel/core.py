# Copyright (c) 2025, Machete Authors
import torch
import cutlass.cute as cute
from cutlass import Int32, Float32
import os
import importlib.util
import hashlib
from typing import Callable, Union
from machete.megakernel.interface import FusableKernel, MegakernelOp, FusableOp, WarpSpecializedKernel
from machete.megakernel.scheduler import (
    NoBubblesConfig,
    NoBubblesScheduler,
    MicroOpType,
    OpDescriptor,
    build_op_descriptor_from_kernel,
    BarrierConfig,
    SequentialScheduler,
    ReductionBarrierConfig,
    DimensionMapping,
    InterOpDependency,
    DependencyGranularity,
    SchedulingMode,
    MixedModeScheduler,
    CodeGenContext,
)
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.compile_utils import make_fake_tensor as fake_tensor
import tvm_ffi.core
import logging
from cutedsl_trace import TraceType, BlockType, TraceWriter, DynamicTraceBuilder, TrackType, TrackLevel

logger = logging.getLogger(__name__)

# TRACING CONFIGURATION - L/C/S phases plus page management
TRACING_PHASES = {
    MicroOpType.LOAD: TraceType("LOAD", "LOAD-{0}", "Load phase for Op {0}", param_count=1),
    MicroOpType.COMPUTE: TraceType("COMPUTE", "COMP-{0}", "Compute phase for Op {0}", param_count=1),
    MicroOpType.STORE: TraceType("STORE", "STORE-{0}", "Store phase for Op {0}", param_count=1),
}

# Additional trace types for page management (No Bubbles pattern)
PAGE_TRACE_TYPES = {
    "ACQUIRE_WAIT": TraceType("ACQUIRE_WAIT", "WAIT_PAGE-{0}", "Waiting for pages (Op {0})", param_count=1),
    "ACQUIRE_DONE": TraceType("ACQUIRE_DONE", "ACQ_PAGE-{0},{1}", "Acquired pages {1} for Op {0}", param_count=2),
    "RELEASE": TraceType("RELEASE", "REL_PAGE-{0},{1}", "Released pages {1} for Op {0}", param_count=2),
}

# Trace types for fine-grained semaphore synchronization (debugging)
SYNC_TRACE_TYPES = {
    "SEM_WAIT_LOAD": TraceType("SEM_WAIT_LOAD", "WAIT_LOAD-{0}", "Wait load sem (Op {0})", param_count=1),
    "SEM_WAIT_COMPUTE": TraceType("SEM_WAIT_COMPUTE", "WAIT_COMP-{0}", "Wait compute sem (Op {0})", param_count=1),
    "SEM_WAIT_FINISHED": TraceType("SEM_WAIT_FINISHED", "WAIT_FIN-{0}", "Wait finished sem (Op {0})", param_count=1),
    "SEM_SIGNAL_LOAD": TraceType("SEM_SIGNAL_LOAD", "SIG_LOAD-{0}", "Signal load done (Op {0})", param_count=1),
    "SEM_SIGNAL_COMPUTE": TraceType(
        "SEM_SIGNAL_COMPUTE", "SIG_COMP-{0}", "Signal compute done (Op {0})", param_count=1
    ),
    "SEM_SIGNAL_FINISHED": TraceType("SEM_SIGNAL_FINISHED", "SIG_FIN-{0}", "Signal finished (Op {0})", param_count=1),
}

# Workaround for TVM-FFI bug (Python < 3.13) with positional arguments in __init__
if not hasattr(tvm_ffi.core.Function, "_monkey_patched"):

    def _safe_init(self, *args, **kwargs):
        mro = type(self).mro()
        try:
            idx = mro.index(tvm_ffi.core.Function)
            for i in range(idx + 1, len(mro)):
                if mro[i].__init__ != object.__init__:
                    mro[i].__init__(self, *args, **kwargs)
                    break
        except Exception:
            pass

    tvm_ffi.core.Function.__init__ = _safe_init
    tvm_ffi.core.Function._monkey_patched = True

# Global registry to pass symbols to generated kernels
MEGAKERNEL_REGISTRY = {}

# Global compilation cache to reuse compiled kernels across Megakernel instances
_GLOBAL_COMPILE_CACHE = {}


class Megakernel:
    def __init__(
        self,
        name: str = "megakernel",
        mode: str = "forward",
        paged_pool_bytes: int = 0,
        num_stages: int = 0,
        page_size: int = 16384,
    ):
        """
        Args:
            name: Kernel name for caching/debugging.
            mode: "forward" or "backward".
            paged_pool_bytes: Total size of the paged shared memory pool in bytes.
                              Set to 0 (default) for compute-only mode.
            num_stages: Number of stages (pages) for No Bubbles scheduling.
                        If 0, uses paged_pool_bytes / page_size.
            page_size: Size of each shared memory page in bytes (default 16KB).
        """
        self.name = name
        self.mode = mode
        self.instructions = []
        self.gen_dir = os.path.join(os.path.dirname(__file__), ".generated")
        os.makedirs(self.gen_dir, exist_ok=True)

        # No Bubbles configuration
        self.page_size = page_size
        if num_stages > 0:
            self.num_stages = num_stages
            self.paged_pool_bytes = num_stages * page_size
        elif paged_pool_bytes > 0:
            self.num_stages = paged_pool_bytes // page_size
            self.paged_pool_bytes = paged_pool_bytes
        else:
            self.num_stages = 0
            self.paged_pool_bytes = 0

        # Logical Blocks configuration
        self._use_logical_blocks = False
        self._total_logical_blocks = 1
        self._barrier_config: BarrierConfig = None

        # Reduction barrier configuration (for many-to-one dependencies)
        self._reduction_barrier_config: ReductionBarrierConfig = None
        self._reduction_barrier = None  # Torch tensor at runtime

    def add(self, op: Union[FusableKernel, MegakernelOp, Callable], *args):
        """Add an operation and its arguments to the megakernel."""

        # Handle FusableKernel - keep reference for L/C/S methods
        if isinstance(op, FusableKernel):
            kernel = op
            if self.mode == "forward":
                load_fn = kernel.load_forward
                compute_fn = kernel.compute_forward
                store_fn = kernel.store_forward
            else:
                load_fn = kernel.load_backward
                compute_fn = kernel.compute_backward
                store_fn = kernel.store_backward

            smem_dtype = getattr(kernel, "cute_dtype", cute.Uint8)
            smem_size = kernel.smem_size
            logger.debug(
                "Adding kernel %s with smem_size=%s",
                kernel.__class__.__name__,
                smem_size,
            )

            # Check for warp specialization
            uses_warp_spec = isinstance(kernel, WarpSpecializedKernel)
            warp_config = kernel.warp_config if uses_warp_spec else None

            self.instructions.append(
                {
                    "kernel": kernel,
                    "load": load_fn,
                    "compute": compute_fn,
                    "store": store_fn,
                    "args": list(args),
                    "smem_size": smem_size,
                    "smem_dtype": smem_dtype,
                    "op_obj": kernel,
                    "uses_warp_specialization": uses_warp_spec,
                    "warp_config": warp_config,
                }
            )
            return

        # Handle MegakernelOp directly
        if isinstance(op, MegakernelOp):
            actual_op = op
        elif hasattr(op, "_machete_is_op"):
            # Decorated function - wrap in FusableOp
            smem_size = getattr(op, "_machete_smem_size", 0)

            instance = getattr(op, "__self__", None)
            if instance and hasattr(instance, "smem_size"):
                smem_size = instance.smem_size

            actual_op = FusableOp(
                compute_func=op,
                num_tensors=op._machete_num_tensors,
                smem_size=smem_size,
            )
        else:
            raise TypeError(f"Unsupported operation type: {type(op)}")

        smem_dtype = cute.Uint8
        check_op = actual_op
        if isinstance(actual_op, FusableOp):
            check_op = actual_op._compute_func

        obj = getattr(check_op, "__self__", check_op)
        if hasattr(obj, "cute_dtype"):
            smem_dtype = obj.cute_dtype

        self.instructions.append(
            {
                "kernel": None,
                "load": getattr(actual_op, "load", None),
                "compute": actual_op.compute,
                "store": getattr(actual_op, "store", None),
                "args": list(args),
                "smem_size": actual_op.smem_size,
                "smem_dtype": smem_dtype,
                "op_obj": actual_op,
            }
        )

    def clear(self):
        self.instructions = []
        self._use_logical_blocks = False
        self._total_logical_blocks = 1
        self._barrier_config = None
        self._reduction_barrier_config = None
        self._reduction_barrier = None

    def add_reduction_dependency(
        self,
        producer_op_idx: int,
        consumer_op_idx: int,
        producer_dims: tuple,
        consumer_dims: tuple,
    ):
        """Register a reduction dependency between operations.

        This is used when a consumer operation has fewer logical blocks than
        its producer (many-to-one pattern). The consumer must wait for ALL
        producer blocks that map to it.

        Args:
            producer_op_idx: Index of the producer operation
            consumer_op_idx: Index of the consumer operation
            producer_dims: Shape of producer's logical grid (e.g., (batch, head, seq))
            consumer_dims: Shape of consumer's logical grid (e.g., (batch, head))

        Example:
            # Attention (batch=2, head=8, seq=512) -> Output proj (batch=2, head=8)
            megakernel.add_reduction_dependency(
                producer_op_idx=0,  # attention
                consumer_op_idx=1,  # output_proj
                producer_dims=(2, 8, 512),
                consumer_dims=(2, 8),
            )
            # Consumer block 0 (batch=0, head=0) waits for 512 producer blocks
        """
        if self._reduction_barrier_config is None:
            self._reduction_barrier_config = ReductionBarrierConfig()

        dim_mapping = DimensionMapping(
            producer_dims=producer_dims,
            consumer_dims=consumer_dims,
        )

        if not dim_mapping.is_reduction:
            logger.warning(
                "add_reduction_dependency called but dims don't form a reduction: "
                "producer=%s, consumer=%s",
                producer_dims,
                consumer_dims,
            )
            return

        self._reduction_barrier_config.add_reduction(
            producer_op_idx=producer_op_idx,
            consumer_op_idx=consumer_op_idx,
            dim_mapping=dim_mapping,
        )

        # Store reduction info in instructions for code generation
        if producer_op_idx < len(self.instructions):
            self.instructions[producer_op_idx].setdefault("reduction_signals", []).append({
                "consumer_op_idx": consumer_op_idx,
                "dim_mapping": dim_mapping,
            })

        if consumer_op_idx < len(self.instructions):
            self.instructions[consumer_op_idx]["reduction_wait"] = {
                "producer_op_idx": producer_op_idx,
                "dim_mapping": dim_mapping,
                "wait_count": dim_mapping.blocks_per_consumer,
            }

        logger.debug(
            "Added reduction dependency: op[%d] (%s) -> op[%d] (%s), wait_count=%d",
            producer_op_idx,
            producer_dims,
            consumer_op_idx,
            consumer_dims,
            dim_mapping.blocks_per_consumer,
        )

    def _calculate_logical_blocks(self) -> int:
        """Calculate TotalLogicalBlocks = max(op.get_logical_grid_size()) across all ops.

        This also configures the barrier tensor size for logical block synchronization.

        Returns:
            Total logical blocks for the grid dimension
        """
        max_logical = 1
        has_logical_api = False

        for inst in self.instructions:
            op = inst.get("op_obj")
            if op and hasattr(op, "get_logical_grid_size"):
                args = inst.get("args", [])
                try:
                    logical_size = op.get_logical_grid_size(*args)
                    if logical_size > 1:
                        has_logical_api = True
                        max_logical = max(max_logical, logical_size)
                        inst["logical_grid_size"] = logical_size
                except Exception as e:
                    logger.warning("Failed to compute logical grid size for %s: %s", type(op).__name__, e)
                    inst["logical_grid_size"] = 1
            else:
                inst["logical_grid_size"] = 1

        self._use_logical_blocks = has_logical_api
        self._total_logical_blocks = max_logical

        # Configure barrier tensor: (NumOps, TotalLogicalBlocks)
        if has_logical_api:
            self._barrier_config = BarrierConfig(
                num_ops=len(self.instructions),
                total_logical_blocks=max_logical,
            )
            logger.debug(
                "Logical Blocks enabled: %d total blocks, barrier tensor %s",
                max_logical,
                self._barrier_config.tensor_size,
            )

        return max_logical

    def _get_megakernel_class(self, mapping, num_flat_args, op_info_key, arg_info, traced=False):
        """Generate and load the megakernel class for the current configuration."""
        # Include arg types/shapes in sig_hash to distinguish different modules
        data_to_hash = (
            f"{self.name}_{num_flat_args}_{op_info_key}_{arg_info}_{self.num_stages}_{self.paged_pool_bytes}_{traced}"
        )
        sig_hash = hashlib.md5(data_to_hash.encode()).hexdigest()

        if sig_hash not in MEGAKERNEL_REGISTRY:
            MEGAKERNEL_REGISTRY[sig_hash] = self.instructions

        gen_path = os.path.join(self.gen_dir, f"kernel_{sig_hash}.py")
        arg_names = [f"arg_{i}" for i in range(num_flat_args)]
        all_args_str = ", ".join(arg_names)

        # Generate module bindings and kernel operations
        module_bindings = self._generate_module_bindings()
        unrolled_ops = self._generate_kernel_ops(mapping, traced)

        # Generate the kernel source code
        content = self._generate_kernel_source(sig_hash, all_args_str, module_bindings, unrolled_ops, traced)

        # Write and load the module
        with open(gen_path, "w") as f:
            f.write(content)

        spec = importlib.util.spec_from_file_location(f"gen_{sig_hash}", gen_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.GeneratedMegakernel(), sig_hash

    def _generate_module_bindings(self) -> list:
        """Generate module-level bindings for L/C/S functions."""
        bindings = []
        for i in range(len(self.instructions)):
            bindings.append(f"op_{i} = instructions[{i}]['op_obj']")
            bindings.append(f"op_{i}_load = instructions[{i}]['load']")
            bindings.append(f"op_{i}_compute = instructions[{i}]['compute']")
            bindings.append(f"op_{i}_store = instructions[{i}]['store']")
        return bindings

    def _generate_kernel_ops(self, mapping, traced: bool) -> list:
        """Generate the kernel operation code (allocations + scheduling).

        Uses MixedModeScheduler to handle heterogeneous kernel types,
        allowing each operation to use its optimal scheduling mode.
        """
        ops = ["        smem_alloc = cutlass.utils.SmemAllocator()"]

        # Paged pool allocation
        if self.paged_pool_bytes > 0:
            ops.append(f"        # Paged pool for No Bubbles: {self.paged_pool_bytes} bytes ({self.num_stages} pages)")
            ops.append(
                "        paged_pool = smem_alloc.allocate_tensor("
                f"cute.Uint8, cute.make_layout({self.paged_pool_bytes}))"
            )
        else:
            ops.append("        paged_pool = smem_alloc.allocate_tensor(cute.Uint8, cute.make_layout(1))")
            ops.append("        page_idx = Int32(0)")

        # Per-op smem tensor allocations
        ops.extend(self._generate_smem_allocations())

        # Generate schedule using MixedModeScheduler
        ops.append("        # Generate Schedule")
        ops.extend(self._generate_mixed_mode_schedule(mapping, traced))

        return ops

    def _generate_mixed_mode_schedule(self, mapping, traced: bool) -> list:
        """Generate schedule using MixedModeScheduler for heterogeneous kernel support.

        Each operation is scheduled according to its optimal mode:
        - SEQUENTIAL: Simple L->C->S, all threads participate
        - ASYNC: Async loads with overlapped phases
        - WARP_SPECIALIZED: Dedicated warps for L/C/S roles

        Mode transitions are automatically handled with proper synchronization.
        """
        # Build operation descriptors with scheduling modes
        op_descs = self._build_op_descriptors()

        # Check if we have mixed modes or should use legacy path
        modes = {desc.scheduling_mode for desc in op_descs}

        # For backward compatibility, use legacy generators for single-mode cases
        # This ensures existing tests pass while we transition
        if len(modes) == 1:
            mode = modes.pop()
            if mode == SchedulingMode.WARP_SPECIALIZED:
                return self._generate_warp_specialized_schedule(mapping, traced)
            elif self.paged_pool_bytes > 0 and self.num_stages >= 2:
                return self._generate_page_aware_schedule(mapping, traced)
            else:
                return self._generate_simple_schedule(mapping, traced)

        # Mixed modes: use MixedModeScheduler
        config = NoBubblesConfig(num_pages=max(1, self.num_stages))
        scheduler = MixedModeScheduler(config)

        # Create code generation context
        ctx = CodeGenContext(
            instructions=self.instructions,
            mapping=mapping,
            traced=traced,
            paged_pool_bytes=self.paged_pool_bytes,
            num_stages=self.num_stages,
            make_smem_arg_getter=self._make_smem_arg_getter,
            emit_trace_start=self._emit_trace_start if traced else None,
            emit_trace_end=self._emit_trace_end if traced else None,
            tracing_phases=TRACING_PHASES if traced else None,
            sync_trace_types=SYNC_TRACE_TYPES if traced else None,
            reduction_barrier_config=self._reduction_barrier_config,
        )

        return scheduler.generate_schedule(ctx, op_descs)

    def _generate_smem_allocations(self) -> list:
        """Generate per-op shared memory tensor allocations."""
        ops = []
        for i, inst in enumerate(self.instructions):
            if inst["smem_size"] > 0:
                element_width_bytes = inst["smem_dtype"].width // 8
                elements_per_page = inst["smem_size"] // element_width_bytes
                # Always use 1D layout - kernels expect 1D smem tensors
                layout_str = f"cute.make_layout({elements_per_page})"
                ops.append(
                    f"        smem_op_{i} = smem_alloc.allocate_tensor(instructions[{i}]['smem_dtype'], {layout_str})"
                )
        return ops

    def _make_smem_arg_getter(self, op_idx: int):
        """Create a function to generate smem argument strings."""
        inst = self.instructions[op_idx]
        smem_size = inst["smem_size"]

        def get_smem_arg(page_var: str) -> str:
            if smem_size > 0:
                # Always pass 1D smem tensor directly
                return f"smem_op_{op_idx}, "
            return ""

        return get_smem_arg

    def _generate_page_aware_schedule(self, mapping, traced: bool) -> list:
        """Generate page-aware scheduling code with dependencies."""
        ops = []

        # Dynamically determine pages_per_op:
        # We need enough pages to double buffer the operation with the largest smem requirement.
        max_op_smem = 0
        for inst in self.instructions:
            max_op_smem = max(max_op_smem, inst["smem_size"])

        # pages_per_buffer = ceil(max_op_smem / page_size)
        pages_per_buffer = (max_op_smem + self.page_size - 1) // self.page_size
        pages_per_buffer = max(1, pages_per_buffer)

        # For double buffering, we need 2 * pages_per_buffer
        pages_per_op = 2 * pages_per_buffer

        ops.append(f"        # Page-aware scheduling: {self.num_stages} pages, {pages_per_op} per op")
        ops.append("        page_idx = Int32(0)")
        ops.append("        next_page = Int32(0)")
        ops.append(f"        # Page semaphore initialized to {self.num_stages} (all pages free)")

        # Build schedule
        config = NoBubblesConfig(num_pages=self.num_stages)
        page_scheduler = NoBubblesScheduler(config)
        op_descs = self._build_op_descriptors()

        # Use async pipeline if any op supports it, otherwise fallback to standard page-aware
        # For now, we use generate_async_pipeline_schedule as it's the most optimized
        page_scheduler.generate_async_pipeline_schedule(op_descs, pages_per_op=pages_per_op)

        # Generate code for each micro-op
        for uop in page_scheduler.micro_ops:
            ops.extend(self._generate_page_aware_uop(uop, mapping, traced))

        return ops

    def _build_op_descriptors(self) -> list:
        """Build OpDescriptors from instructions for scheduling.

        Each descriptor includes a scheduling_mode that determines how
        the operation will be scheduled (sequential, async, warp-specialized).
        """
        op_descs = []
        for i, inst in enumerate(self.instructions):
            kernel = inst.get("kernel") or inst.get("op_obj")
            if kernel and hasattr(kernel, "load_forward"):
                desc = build_op_descriptor_from_kernel(kernel, i, self.mode)
            else:
                desc = OpDescriptor(
                    name=f"Op{i}",
                    op_idx=i,
                    needs_global_sync=inst.get("needs_sync", False),
                )

            # Override scheduling mode from instruction if explicitly set
            if inst.get("uses_warp_specialization", False):
                desc.uses_warp_specialization = True
                desc.scheduling_mode = SchedulingMode.WARP_SPECIALIZED

            # Store logical grid size if available
            if "logical_grid_size" in inst:
                desc.logical_grid_size = inst["logical_grid_size"]

            op_descs.append(desc)
        return op_descs

    def _generate_page_aware_uop(self, uop, mapping, traced: bool) -> list:
        """Generate code for a single page-aware micro-op."""
        ops = []
        i = uop.op_idx
        args_str = ", ".join([f"arg_{idx}" for idx in mapping[i]])
        get_smem_arg = self._make_smem_arg_getter(i)

        if uop.type in (MicroOpType.LOAD, MicroOpType.LOAD_ASYNC):
            ops.extend(self._gen_load_with_page_acquire(uop, i, args_str, get_smem_arg, traced))
        elif uop.type == MicroOpType.COMPUTE:
            ops.extend(self._gen_compute(i, args_str, get_smem_arg, traced))
        elif uop.type in (MicroOpType.STORE, MicroOpType.STORE_ASYNC):
            ops.extend(self._gen_store_with_page_release(uop, i, args_str, get_smem_arg, traced))
        elif uop.type == MicroOpType.WAIT_LOAD:
            ops.append("        # Wait for async loads (cp.async / TMA)")
            ops.append("        cute.arch.cp_async_wait_group(0)")
        elif uop.type == MicroOpType.COMMIT_GROUP:
            ops.append("        cute.arch.cp_async_commit_group()")
        elif uop.type == MicroOpType.SYNC_BLOCK:
            ops.append("        cute.arch.fence_view_async_shared()")
        elif uop.type == MicroOpType.SYNC_GLOBAL:
            ops.extend(self._gen_global_sync())

        return ops

    def _gen_load_with_page_acquire(self, uop, op_idx, args_str, get_smem_arg, traced) -> list:
        """Generate LOAD code with page acquisition."""
        ops = []

        # Wait for dependencies
        if uop.depends_on:
            if traced:
                ops.append("        t_acq = cutedsl_trace.device.start()")
            ops.append(f"        # Wait for pages (deps: {sorted(uop.depends_on)})")
            ops.append("        cute.arch.fence_view_async_shared()  # Ensure prior stores visible")
            if traced:
                fmt_id = PAGE_TRACE_TYPES["ACQUIRE_WAIT"].id
                ops.append(
                    "        lane = cutedsl_trace.device.end_event_dynamic_raw_1("
                    f"t_acq, trace_buffer, trace_row_stride, lane, Int32({fmt_id}), Int32({op_idx}))"
                )

        # Acquire pages
        acquired = uop.acquires_pages
        if acquired and traced:
            ops.append("        t_acq_done = cutedsl_trace.device.start()")

        page_idx_val = acquired[0] if acquired else 0
        ops.append(f"        page_idx = Int32({page_idx_val})")

        if acquired and traced:
            fmt_id = PAGE_TRACE_TYPES["ACQUIRE_DONE"].id
            pages_mask = sum(1 << p for p in acquired)
            ops.append(
                "        lane = cutedsl_trace.device.end_event_dynamic_raw_2("
                "t_acq_done, trace_buffer, trace_row_stride, lane, "
                f"Int32({fmt_id}), Int32({op_idx}), Int32({pages_mask}))"
            )

        # Execute load
        if traced:
            ops.append("        t_start = cutedsl_trace.device.start()")

        smem_str = get_smem_arg("page_idx")
        ops.append(f"        op_{op_idx}_load(paged_pool, page_idx, logical_idx, {smem_str}{args_str})")

        if traced:
            fmt_id = TRACING_PHASES[MicroOpType.LOAD].id
            ops.append(
                f"        lane = cutedsl_trace.device.end_event_dynamic_raw_1("
                f"t_start, trace_buffer, trace_row_stride, lane, Int32({fmt_id}), Int32({op_idx}))"
            )

        return ops

    def _gen_compute(self, op_idx, args_str, get_smem_arg, traced) -> list:
        """Generate COMPUTE code."""
        ops = []
        if traced:
            ops.append("        t_start = cutedsl_trace.device.start()")

        smem_str = get_smem_arg("page_idx")
        ops.append(f"        op_{op_idx}_compute(logical_idx, {smem_str}{args_str})")

        if traced:
            fmt_id = TRACING_PHASES[MicroOpType.COMPUTE].id
            ops.append(
                f"        lane = cutedsl_trace.device.end_event_dynamic_raw_1("
                f"t_start, trace_buffer, trace_row_stride, lane, Int32({fmt_id}), Int32({op_idx}))"
            )
        return ops

    def _gen_store_with_page_release(self, uop, op_idx, args_str, get_smem_arg, traced) -> list:
        """Generate STORE code with page release."""
        ops = []
        if traced:
            ops.append("        t_start = cutedsl_trace.device.start()")

        smem_str = get_smem_arg("page_idx")
        ops.append(f"        op_{op_idx}_store(paged_pool, page_idx, logical_idx, {smem_str}{args_str})")

        if traced:
            fmt_id = TRACING_PHASES[MicroOpType.STORE].id
            ops.append(
                f"        lane = cutedsl_trace.device.end_event_dynamic_raw_1("
                f"t_start, trace_buffer, trace_row_stride, lane, Int32({fmt_id}), Int32({op_idx}))"
            )

        # Release pages
        released = uop.releases_pages
        if released:
            if traced:
                ops.append("        t_rel = cutedsl_trace.device.start()")
            ops.append(f"        # Release pages {released}")
            ops.append("        cute.arch.fence_view_async_shared()  # Ensure store complete")
            if traced:
                fmt_id = PAGE_TRACE_TYPES["RELEASE"].id
                pages_mask = sum(1 << p for p in released)
                ops.append(
                    "        lane = cutedsl_trace.device.end_event_dynamic_raw_2("
                    "t_rel, trace_buffer, trace_row_stride, lane, "
                    f"Int32({fmt_id}), Int32({op_idx}), Int32({pages_mask}))"
                )

        return ops

    def _gen_global_sync(self) -> list:
        """Generate global barrier synchronization code."""
        return [
            "        # Global Barrier",
            "        if tidx == 0:",
            "            atomic_add_i32(1, barrier_tensor.iterator)",
            "            target = (sync_step + 1) * n_blocks",
            "            while atomic_add_i32(0, barrier_tensor.iterator) < target:",
            "                pass",
            "        cute.arch.fence_view_async_shared()",
            "        sync_step = sync_step + Int32(1)",
        ]

    def _generate_warp_specialized_schedule(self, mapping, traced: bool) -> list:
        """Generate warp-specialized scheduling code (Persistent No Bubbles pattern).

        This generates a persistent grid loop where warps cooperate via a rotating
        circular buffer of shared memory pages and granular semaphores.

        Synchronization Protocol (3-stage):
        1. LOADER: Wait for FINISHED[page] -> LOAD -> Signal LOAD_DONE[page]
        2. CONSUMER: Wait for LOAD_DONE[page] -> COMPUTE -> Signal COMPUTE_DONE[page]
        3. STORER: Wait for COMPUTE_DONE[page] -> STORE -> Signal FINISHED[page]
        """
        ops = []

        # Get warp config from first warp-specialized kernel
        warp_config = None
        for inst in self.instructions:
            if inst.get("uses_warp_specialization"):
                warp_config = inst.get("warp_config")
                break

        if warp_config is None:
            # Fallback to default config
            from machete.megakernel.scheduler import WarpConfig

            warp_config = WarpConfig()

        num_consumer = warp_config.num_consumer_warps
        loader_warp = num_consumer
        storer_warp = num_consumer + warp_config.num_loader_warps

        # Configuration
        num_ops = len(self.instructions)
        num_stages = max(1, self.num_stages)
        # Offsets for semaphore types
        # 0: LOAD_DONE, 1: COMPUTE_DONE, 2: FINISHED (ready to reuse)

        ops.append("        # Warp-Specialized Scheduling (Persistent No Bubbles)")
        ops.append(f"        # Consumer warps: 0-{num_consumer - 1}, Loader: {loader_warp}, Storer: {storer_warp}")
        ops.append("        warp_id = tidx // Int32(32)")
        ops.append("        lane_id = tidx % Int32(32)")
        if traced:
            ops.append("        t_wait = cutedsl_trace.device.start()")
            ops.append("        t_sig = cutedsl_trace.device.start()")
            ops.append("        t_start = cutedsl_trace.device.start()")
        ops.append("")
        ops.append("        # Page semaphores in shared memory for fine-grained coordination")
        ops.append(f"        # Array size: {num_stages} stages * {num_ops} ops * 3 types")
        num_sems = num_stages * num_ops * 3 + 1
        ops.append(f"        semaphores = smem_alloc.allocate_tensor(cute.Int32, cute.make_layout({num_sems}))")
        ops.append(f"        init_sem_idx = Int32({num_stages * num_ops * 3})")
        ops.append("")
        ops.append("        # Initialize semaphores (single thread)")
        ops.append("        if tidx == 0:")
        ops.append("            semaphores[init_sem_idx] = Int32(0)")
        ops.append(f"            for i in range({num_stages * num_ops * 3}):")
        ops.append("                # Initialize FINISHED (type 2) to 1 (page free), others to 0")
        ops.append("                if i % 3 == 2:")
        ops.append("                    semaphores[i] = Int32(1)")
        ops.append("                else:")
        ops.append("                    semaphores[i] = Int32(0)")
        ops.append("            # Signal initialization complete")
        ops.append("            cute.arch.fence_view_async_shared()")
        ops.append("            atomic_add_i32(1, semaphores.iterator + init_sem_idx)")
        ops.append("")
        ops.append("        # Spin-wait for initialization")
        ops.append("        if tidx != 0:")
        ops.append("            while atomic_add_i32(0, semaphores.iterator + init_sem_idx) < Int32(1):")
        ops.append("                nanosleep(Int32(20))")
        ops.append("        cute.arch.fence_view_async_shared()  # Ensure all semaphore values are visible")
        ops.append("")

        # Grid loop for persistent blocks
        ops.append("        local_block_idx = Int32(0)")
        ops.append("        persistent_block_idx = bidx")
        ops.append("        n_blocks_val = n_blocks")
        ops.append("        g_dim, _, _ = cute.arch.grid_dim()")
        ops.append("")
        ops.append("        while persistent_block_idx < n_blocks_val:")
        ops.append(f"            page_idx = local_block_idx % Int32({num_stages})")
        ops.append(f"            iter_target = local_block_idx // Int32({num_stages}) + Int32(1)")
        ops.append("")

        self._emit_warp_spec_loader_role(ops, mapping, loader_warp, num_ops, traced)
        self._emit_warp_spec_consumer_role(ops, mapping, num_consumer, num_ops, traced)
        self._emit_warp_spec_storer_role(ops, mapping, storer_warp, num_consumer, num_ops, traced)

        # Other warps (controller/launcher) - idle in current implementation
        ops.append("            else:")
        ops.append("                pass  # Controller/Launcher warps - idle")

        ops.append("")
        ops.append("            # Persistence update")
        ops.append("            local_block_idx += Int32(1)")
        ops.append("            persistent_block_idx += g_dim")
        ops.append("        # End persistent loop")

        return ops

    def _emit_warp_spec_initialization(self, ops, num_stages, num_ops, init_sem_idx):
        """Emit initialization code for warp-specialized schedule."""
        ops.append("        if tidx == 0:")
        ops.append("            semaphores[init_sem_idx] = Int32(0)")
        ops.append(f"            for i in range({num_stages * num_ops * 3}):")
        ops.append("                if i % 3 == 2:")
        ops.append("                    semaphores[i] = Int32(1)")
        ops.append("                else:")
        ops.append("                    semaphores[i] = Int32(0)")
        ops.append("            cute.arch.fence_view_async_shared()")
        ops.append("            atomic_add_i32(1, semaphores.iterator + init_sem_idx)")
        ops.append("        if tidx != 0:")
        ops.append("            while atomic_add_i32(0, semaphores.iterator + init_sem_idx) < Int32(1):")
        ops.append("                nanosleep(Int32(20))")

    def _emit_warp_spec_loader_role(self, ops, mapping, loader_warp, num_ops, traced):
        """Emit LOADER role code.

        If this operation has a reduction_wait dependency, the loader must wait
        for ALL producer blocks to complete before loading (many-to-one sync).
        """
        ops.append(f"            if warp_id == Int32({loader_warp}):")
        for i, inst in enumerate(self.instructions):
            args = ", ".join([f"arg_{idx}" for idx in mapping[i]])
            smem = self._make_smem_arg_getter(i)("page_idx")
            idx_fin = f"(page_idx * Int32({num_ops}) + Int32({i})) * Int32(3) + Int32(2)"
            idx_load = f"(page_idx * Int32({num_ops}) + Int32({i})) * Int32(3) + Int32(0)"

            if traced:
                ops.append("                t_wait = cutedsl_trace.device.start()")
                ops.append("                if lane_id == 0:")
                self._emit_trace_start(ops, "                    ", "t_wait")
            ops.append(f"                while atomic_add_i32(0, semaphores.iterator + {idx_fin}) < iter_target:")
            ops.append("                    nanosleep(Int32(20))")
            if traced:
                ops.append("                if lane_id == 0:")
                self._emit_trace_end(ops, "                    ", SYNC_TRACE_TYPES, "SEM_WAIT_FINISHED", i, "t_wait")

            # Check for reduction wait dependency (many-to-one)
            reduction_wait = inst.get("reduction_wait")
            if reduction_wait and self._reduction_barrier_config:
                wait_count = reduction_wait["wait_count"]
                producer_op_idx = reduction_wait["producer_op_idx"]
                barrier_offset = self._reduction_barrier_config._barrier_offsets.get(i, 0)

                ops.append(f"                # Wait for reduction from producer op {producer_op_idx}")
                ops.append(f"                # Consumer block waits for {wait_count} producer blocks")
                red_bar_idx = f"Int32({barrier_offset}) + persistent_block_idx"
                ops.append(
                    f"                while atomic_add_i32(0, reduction_barrier.iterator + {red_bar_idx}) "
                    f"< Int32({wait_count}):"
                )
                ops.append("                    nanosleep(Int32(20))")

            if traced:
                self._emit_trace_start(ops, "                ")

            size = inst.get("logical_grid_size", -1)
            pref = f"if persistent_block_idx < Int32({size}): " if size != -1 else ""
            ops.append(f"                {pref}op_{i}_load(paged_pool, page_idx, persistent_block_idx, {smem}{args})")

            if traced:
                self._emit_trace_end(ops, "                ", TRACING_PHASES, MicroOpType.LOAD, i)
            # Warp-level sync to ensure all lanes have completed their writes
            ops.append("                cute.arch.sync_warp()")
            ops.append("                cute.arch.fence_view_async_shared()")
            ops.append("                if lane_id == 0:")
            if traced:
                ops.append("                    t_sig = cutedsl_trace.device.start()")
            ops.append(f"                    atomic_add_i32(1, semaphores.iterator + {idx_load})")
            if traced:
                self._emit_trace_end(ops, "                    ", SYNC_TRACE_TYPES, "SEM_SIGNAL_LOAD", i, "t_sig")

    def _emit_warp_spec_consumer_role(self, ops, mapping, num_consumer, num_ops, traced):
        """Emit CONSUMER role code.

        All consumer warps wait for LOAD_DONE, then compute, then each signals COMPUTE_DONE.
        The storer waits for all consumer warps to signal (count = iter_target * num_consumer).
        """
        ops.append(f"            elif warp_id < Int32({num_consumer}):")
        for i, inst in enumerate(self.instructions):
            args = ", ".join([f"arg_{idx}" for idx in mapping[i]])
            smem = self._make_smem_arg_getter(i)("page_idx")
            idx_load = f"(page_idx * Int32({num_ops}) + Int32({i})) * Int32(3) + Int32(0)"
            idx_comp = f"(page_idx * Int32({num_ops}) + Int32({i})) * Int32(3) + Int32(1)"

            if traced:
                ops.append("                t_wait = cutedsl_trace.device.start()")
                ops.append("                if lane_id == 0:")
                self._emit_trace_start(ops, "                    ", "t_wait")
            ops.append(f"                while atomic_add_i32(0, semaphores.iterator + {idx_load}) < iter_target:")
            ops.append("                    nanosleep(Int32(20))")
            if traced:
                ops.append("                if lane_id == 0:")
                self._emit_trace_end(ops, "                    ", SYNC_TRACE_TYPES, "SEM_WAIT_LOAD", i, "t_wait")

            if traced:
                self._emit_trace_start(ops, "                ")
            size = inst.get("logical_grid_size", -1)
            pref = f"if persistent_block_idx < Int32({size}): " if size != -1 else ""
            ops.append(f"                {pref}op_{i}_compute(persistent_block_idx, {smem}{args})")
            if traced:
                self._emit_trace_end(ops, "                ", TRACING_PHASES, MicroOpType.COMPUTE, i)
            # Warp-level sync to ensure all lanes have completed their writes
            ops.append("                cute.arch.sync_warp()")
            ops.append("                cute.arch.fence_view_async_shared()")
            # Each consumer warp signals (lane 0 only to avoid intra-warp overcounting)
            ops.append("                if lane_id == 0:")
            if traced:
                ops.append("                    t_sig = cutedsl_trace.device.start()")
            ops.append(f"                    atomic_add_i32(1, semaphores.iterator + {idx_comp})")
            if traced:
                self._emit_trace_end(ops, "                    ", SYNC_TRACE_TYPES, "SEM_SIGNAL_COMPUTE", i, "t_sig")

    def _emit_warp_spec_storer_role(self, ops, mapping, storer_warp, num_con, num_ops, traced):
        """Emit STORER role code.

        The storer waits for ALL consumer warps to signal COMPUTE_DONE
        (count = iter_target * num_consumer_warps), then stores and signals FINISHED.

        If this operation has reduction_signals, also signals the reduction barrier
        so downstream consumers can track how many producer blocks have completed.
        """
        ops.append(f"            elif warp_id == Int32({storer_warp}):")
        for i, inst in enumerate(self.instructions):
            args = ", ".join([f"arg_{idx}" for idx in mapping[i]])
            smem = self._make_smem_arg_getter(i)("page_idx")
            idx_comp = f"(page_idx * Int32({num_ops}) + Int32({i})) * Int32(3) + Int32(1)"
            idx_fin = f"(page_idx * Int32({num_ops}) + Int32({i})) * Int32(3) + Int32(2)"
            # Wait for ALL consumer warps to signal (num_con signals per iteration)
            target = f"iter_target * Int32({num_con})"

            if traced:
                ops.append("                t_wait = cutedsl_trace.device.start()")
                ops.append("                if lane_id == 0:")
                self._emit_trace_start(ops, "                    ", "t_wait")
            ops.append(f"                while atomic_add_i32(0, semaphores.iterator + {idx_comp}) < {target}:")
            ops.append("                    nanosleep(Int32(20))")
            if traced:
                ops.append("                if lane_id == 0:")
                self._emit_trace_end(ops, "                    ", SYNC_TRACE_TYPES, "SEM_WAIT_COMPUTE", i, "t_wait")

            if traced:
                self._emit_trace_start(ops, "                ")

            size = inst.get("logical_grid_size", -1)
            prefix = f"if persistent_block_idx < Int32({size}): " if size != -1 else ""
            ops.append(
                f"                {prefix}op_{i}_store(paged_pool, page_idx, persistent_block_idx, {smem}{args})"
            )

            if traced:
                self._emit_trace_end(ops, "                ", TRACING_PHASES, MicroOpType.STORE, i)
            # Warp-level sync to ensure all lanes have completed their writes
            ops.append("                cute.arch.sync_warp()")
            ops.append("                cute.arch.fence_view_async_shared()")
            ops.append("                if lane_id == 0:")
            if traced:
                ops.append("                    t_sig = cutedsl_trace.device.start()")
            ops.append(f"                    atomic_add_i32(1, semaphores.iterator + {idx_fin})")
            if traced:
                self._emit_trace_end(ops, "                    ", SYNC_TRACE_TYPES, "SEM_SIGNAL_FINISHED", i, "t_sig")

            # Signal reduction barriers if this operation has downstream reduction consumers
            reduction_signals = inst.get("reduction_signals", [])
            for red_sig in reduction_signals:
                dim_mapping = red_sig["dim_mapping"]
                consumer_op_idx = red_sig["consumer_op_idx"]
                blocks_per_consumer = dim_mapping.blocks_per_consumer

                # Compute which consumer logical block this producer block maps to
                # For trailing-axis reduction: consumer_logical = producer_logical // blocks_per_consumer
                ops.append(f"                    # Reduction signal to consumer op {consumer_op_idx}")
                ops.append(
                    f"                    reduction_consumer_idx = persistent_block_idx // Int32({blocks_per_consumer})"
                )
                # Get barrier offset for this consumer
                if self._reduction_barrier_config:
                    barrier_offset = self._reduction_barrier_config._barrier_offsets.get(consumer_op_idx, 0)
                    red_bar_idx = f"Int32({barrier_offset}) + reduction_consumer_idx"
                    ops.append(
                        f"                    atomic_add_i32(1, reduction_barrier.iterator + {red_bar_idx})"
                    )

    def _emit_trace_start(self, ops: list, indent: str, var: str = "t_start") -> None:
        """Emit trace start timestamp capture."""
        ops.append(f"{indent}{var} = cutedsl_trace.device.start()")

    def _emit_trace_end(
        self, ops: list, indent: str, trace_type_dict, type_key: str, op_idx: int, var: str = "t_start"
    ) -> None:
        """Emit trace end event with single parameter."""
        fmt_id = trace_type_dict[type_key].id
        ops.append(
            f"{indent}lane = cutedsl_trace.device.end_event_dynamic_raw_1("
            f"{var}, trace_buffer, trace_row_stride, lane, Int32({fmt_id}), Int32({op_idx}))"
        )

    def _emit_simple_op_phases(self, ops, i, map_idx, inst, traced, sizes, has_het):
        """Emit L/C/S phases for a single operation in simple schedule."""
        args_str = ", ".join([f"arg_{idx}" for idx in map_idx])
        smem_arg = self._make_smem_arg_getter(i)
        logical_size = inst.get("logical_grid_size", 1)
        needs_bounds = has_het and logical_size < max(sizes)
        indent = "            " if needs_bounds else "        "

        if needs_bounds:
            ops.append(f"        if logical_idx < Int32({logical_size}):")

        if i > 0:
            ops.append(f"{indent}if tidx == 0:")
            ops.append(f"{indent}    semaphores[0] = Int32(0)\n{indent}    semaphores[1] = Int32(0)")
            ops.append(f"{indent}cute.arch.fence_view_async_shared()")

        p_idx = "page_idx"
        for phase in [MicroOpType.LOAD, MicroOpType.COMPUTE, MicroOpType.STORE]:
            if phase == MicroOpType.COMPUTE and "compute" not in inst:
                continue

            # Wait (non-zero threads)
            if phase != MicroOpType.LOAD:
                s_idx = 0 if phase == MicroOpType.COMPUTE else 1
                t_key = "SEM_WAIT_LOAD" if phase == MicroOpType.COMPUTE else "SEM_WAIT_COMPUTE"
                ops.append(f"{indent}if tidx != 0:")
                if traced:
                    self._emit_trace_start(ops, indent + "    ", "t_wait")
                ops.append(f"{indent}    while atomic_add_i32(0, semaphores.iterator + Int32({s_idx})) < Int32(1):")
                ops.append(f"{indent}        nanosleep(Int32(20))")
                if traced:
                    self._emit_trace_end(ops, indent + "    ", SYNC_TRACE_TYPES, t_key, i, "t_wait")

            if traced:
                self._emit_trace_start(ops, indent)
            fn = phase.name.lower()
            prefix = "paged_pool, page_idx, " if phase != MicroOpType.COMPUTE else ""
            ops.append(f"{indent}op_{i}_{fn}({prefix}logical_idx, {smem_arg(p_idx)}{args_str})")
            if traced:
                self._emit_trace_end(ops, indent, TRACING_PHASES, phase, i)

            # Signal (tidx 0 only)
            if phase != MicroOpType.STORE:
                s_idx = 0 if phase == MicroOpType.LOAD else 1
                ops.append(f"{indent}cute.arch.fence_view_async_shared()")
                ops.append(f"{indent}if tidx == 0:")
                if traced:
                    ops.append(f"{indent}    t_sig = cutedsl_trace.device.start()")
                ops.append(f"{indent}    atomic_add_i32(1, semaphores.iterator + Int32({s_idx}))")
                if traced:
                    self._emit_trace_end(ops, indent + "    ", SYNC_TRACE_TYPES, f"SEM_SIGNAL_{phase.name}", i, "t_sig")

    def _generate_simple_schedule(self, mapping, traced: bool) -> list:
        """Generate simple scheduling code with fine-grained semaphore synchronization.

        Delegates to SequentialScheduler for cleaner separation of concerns.

        Simple mode uses semaphore-based synchronization where all threads execute
        all phases sequentially (no warp specialization). This ensures consistent
        synchronization semantics:
        - sem_load_done: Signaled after load completes, waited on before compute
        - sem_compute_done: Signaled after compute completes, waited on before store
        """
        scheduler = SequentialScheduler()
        return scheduler.generate_schedule_ops(
            instructions=self.instructions,
            mapping=mapping,
            paged_pool_bytes=self.paged_pool_bytes,
            traced=traced,
            make_smem_arg_getter=self._make_smem_arg_getter,
            emit_trace_start=self._emit_trace_start,
            emit_trace_end=self._emit_trace_end,
            tracing_phases=TRACING_PHASES,
            sync_trace_types=SYNC_TRACE_TYPES,
        )

    def _generate_kernel_source(self, sig_hash: str, all_args_str: str, bindings: list, ops: list, traced: bool) -> str:
        """Generate the complete Python source code for the kernel module."""
        bindings_content = "\n".join(bindings)
        ops_content = "\n".join(ops)
        kernel_name = f"kernel_{sig_hash[:8]}"

        trace_args = ", trace_buffer, trace_row_stride, trace_num_lanes" if traced else ""

        # Reduction barrier argument (for many-to-one dependencies)
        has_reduction = self._reduction_barrier_config is not None
        reduction_arg = ", reduction_barrier" if has_reduction else ""

        # Always get block index for logical blocks support
        # bidx is the logical block index when using Logical Blocks API
        bidx_init = "bidx, _, _ = cute.arch.block_idx()"

        # Logical block index: each block gets a unique logical_idx
        logical_idx_init = "logical_idx = bidx  # Logical block index for coordinate mapping"

        trace_init = (
            "is_trace_thread = (tidx % 32 == 0)\n"
            "        lane = cutedsl_trace.device.begin_lane_dynamic_raw("
            "trace_num_lanes, trace_row_stride, bidx, tidx // 32)\n"
            "        lane.enabled = is_trace_thread"
            if traced
            else ""
        )
        trace_finish = (
            "if is_trace_thread: cutedsl_trace.device.finish_lane_dynamic_raw(trace_buffer, lane)" if traced else ""
        )

        return f'''
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import cutedsl_trace
from cutlass import Int32, const_expr
from quack.utils import atomic_add_i32
from machete.megakernel.core import MEGAKERNEL_REGISTRY
from machete.megakernel.utils import nanosleep

# Retrieve instructions from registry
instructions = MEGAKERNEL_REGISTRY["{sig_hash}"]

# Bind module-level symbols
{bindings_content}

class GeneratedMegakernel:
    @cute.jit
    def __call__(self, barrier_tensor{reduction_arg}, n_blocks, {all_args_str}{trace_args}):
        # We launch utilizing grid/block from the registry instructions
        # This allows the caller to update registry with current grid/block before call.
        self.{kernel_name}(barrier_tensor{reduction_arg}, n_blocks, {all_args_str}{trace_args}).launch(
            grid=instructions[0]['grid'],
            block=instructions[0]['block'],
            smem=instructions[0]['total_smem']
        )

    @cute.kernel
    def {kernel_name}(self, barrier_tensor{reduction_arg}, n_blocks, {all_args_str}{trace_args}):
        sync_step = Int32(0)
        tidx, _, _ = cute.arch.thread_idx()
        {bidx_init}
        {logical_idx_init}
        {trace_init}
{ops_content}
        {trace_finish}
'''

    @staticmethod
    def _get_tensor_divisibility(tensor: torch.Tensor) -> int:
        """Calculate the max element divisibility for a tensor based on its pointer alignment and strides."""
        ptr = tensor.data_ptr()
        elem_size = tensor.element_size()

        # Max alignment we care about is 16 bytes (128-bit)
        bytes_align = 1
        for b in [16, 8, 4, 2]:
            if ptr % b == 0:
                bytes_align = b
                break

        align_elements = bytes_align // elem_size
        if align_elements <= 0:
            return 1

        # Strides must also be divisible by this alignment to maintain it across dimensions
        # (Except the unit stride)
        for s in tensor.stride():
            if s == 1:
                continue
            while align_elements > 1 and s % align_elements != 0:
                align_elements //= 2

        return align_elements

    def launch_logical(self, block, grid=None, stream=None, trace_file=None):
        """Launch the megakernel using Logical Blocks for grid dimension.

        This method automatically calculates TotalLogicalBlocks from all registered
        operations and uses it as the grid dimension. Each block will receive a
        unique logical_idx that can be mapped to kernel-specific coordinates via
        get_logical_coord().

        Args:
            block: Block dimensions (threads per block)
            grid: Grid dimensions (optional). If not provided, uses (total_logical, 1, 1).
            stream: CUDA stream (optional)
            trace_file: Path to write trace file (optional)

        Example:
            mk = Megakernel()
            mk.add(rope_kernel, q, k, cos, sin, batch=2, seq_len=512, heads=8)
            mk.add(gated_linear, x, gate, out)
            mk.launch_logical(block=(256, 1, 1))  # Grid auto-calculated
        """
        # Calculate total logical blocks from all operations
        total_logical = self._calculate_logical_blocks()
        if grid is None:
            grid = (total_logical, 1, 1)

        logger.debug("Launching with Logical Blocks: grid=%s, block=%s", grid, block)
        return self.launch(n_blocks=total_logical, grid=grid, block=block, stream=stream, trace_file=trace_file)

    def launch(self, n_blocks: int, grid, block, stream=None, trace_file=None):
        """Launch the megakernel with the given configuration."""
        traced = trace_file is not None

        # Calculate logical blocks if not already done
        if not self._use_logical_blocks:
            self._calculate_logical_blocks()

        # Prepare arguments and mapping
        flat_args, mapping, device = self._prepare_launch_args(grid, block)
        num_flat_args = len(flat_args)

        # Setup tracing if enabled
        trace_builder = self._setup_trace(traced, block, grid) if traced else None

        # Calculate and validate shared memory
        total_smem = self._calculate_smem(device)

        for inst in self.instructions:
            inst["total_smem"] = total_smem

        # Build compilation key and hash
        arg_info = self._build_arg_info(flat_args)
        op_info, op_info_key = self._build_op_info()

        compile_key = (
            self.mode,
            tuple(op_info),
            tuple(arg_info),
            tuple(grid),
            tuple(block),
            self.num_stages,
            self.paged_pool_bytes,
            traced,
        )

        sig_hash = self._compute_sig_hash(num_flat_args, op_info_key, arg_info, traced)

        # Update registry with current launch params
        self._update_registry(sig_hash)

        # Compile if not cached
        if compile_key not in _GLOBAL_COMPILE_CACHE:
            self._compile_kernel(
                compile_key, mapping, num_flat_args, flat_args, n_blocks, op_info_key, arg_info, traced, block, grid
            )

        # Execute kernel
        self._execute_kernel(compile_key, n_blocks, flat_args, traced, trace_builder, device)

        if traced:
            self._write_trace(trace_file, trace_builder, TRACING_PHASES)

    def _prepare_launch_args(self, grid, block):
        """Flatten arguments and build mapping for each instruction."""
        flat_args = []
        mapping = []
        curr = 0
        device = None

        for inst in self.instructions:
            args = inst["args"]
            flat_args.extend(args)
            mapping.append(list(range(curr, curr + len(args))))
            curr += len(args)
            inst["grid"] = grid
            inst["block"] = block

            if device is None:
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        device = arg.device
                        break

        return flat_args, mapping, device or torch.device("cuda")

    def _setup_trace(self, traced: bool, block, grid):
        """Setup trace builder for kernel profiling."""
        if not traced:
            return None
        num_lanes = (block[0] * block[1] * block[2] + 31) // 32
        max_events = len(self.instructions) * 10  # L/C/S + 4 sync events + page events
        return DynamicTraceBuilder(
            num_lanes=num_lanes, max_events_per_lane=max_events, grid_dims=(grid[0], grid[1], grid[2])
        )

    def _calculate_smem(self, device) -> int:
        """Calculate shared memory requirement, reducing pages if needed.

        New model: paging is managed at the Megakernel level via self.num_stages.
        Per-operation smem_size is allocated separately (not part of paged pool).
        Total smem = paged_pool_bytes + sum(op.smem_size for all ops) + margin.
        """
        props = torch.cuda.get_device_properties(device)
        max_smem = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
        limit_smem = max_smem - 2048  # Safety margin

        def calc_smem():
            # Paged pool for No Bubbles (global pool, managed by Megakernel)
            total = self.paged_pool_bytes
            # Semaphore array (3 semaphores per stage per instruction)
            num_ops = len(self.instructions)
            stages = max(1, self.num_stages)
            total += stages * num_ops * 3 * 4
            # Per-operation shared memory (allocated separately, not paged)
            for inst in self.instructions:
                total += inst["smem_size"]
            return total + 1024  # Safety margin

        # Adaptive reduction of page count if smem exceeds limit
        while self.num_stages >= 2:
            candidate_smem = calc_smem()
            if candidate_smem <= limit_smem:
                return candidate_smem

            old_pages = self.num_stages
            self.num_stages -= 1
            if self.num_stages < 1:
                self.num_stages = 1
            if old_pages > 0:
                self.paged_pool_bytes = int(self.paged_pool_bytes * (self.num_stages / old_pages))
            if self.num_stages == old_pages:
                break

        total_smem = calc_smem()
        if total_smem > max_smem:
            logger.critical("Calculated shared memory %d exceeds device limit %d. This may fail.", total_smem, max_smem)
        return total_smem

    def _build_arg_info(self, flat_args) -> list:
        """Build argument info for compilation cache key."""
        arg_info = []
        for arg in flat_args:
            if isinstance(arg, torch.Tensor):
                arg_info.append((arg.dtype, tuple(arg.shape), tuple(arg.stride())))
            elif isinstance(arg, int):
                arg_info.append(("int", arg))
            elif isinstance(arg, float):
                arg_info.append(("float", arg))
            else:
                arg_info.append((type(arg), None))
        return arg_info

    def _build_op_info(self):
        """Build operation info for compilation cache key."""
        op_info = []
        for inst in self.instructions:
            op = inst["op_obj"]
            state = {}
            actual_op = op
            if hasattr(op, "_compute_func") and hasattr(op._compute_func, "__self__"):
                actual_op = op._compute_func.__self__
            for attr in ["act_type", "head_dim", "backward"]:
                if hasattr(actual_op, attr):
                    state[attr] = getattr(actual_op, attr)
            op_info.append((type(op), type(actual_op), tuple(state.items())))
        return op_info, str(op_info)

    def _compute_sig_hash(self, num_flat_args, op_info_key, arg_info, traced) -> str:
        """Compute signature hash for kernel identification."""
        data_to_hash = (
            f"{self.name}_{num_flat_args}_{op_info_key}_{arg_info}_{self.num_stages}_{self.paged_pool_bytes}_{traced}"
        )
        return hashlib.md5(data_to_hash.encode()).hexdigest()

    def _update_registry(self, sig_hash: str):
        """Update registry with current instruction values."""
        if sig_hash in MEGAKERNEL_REGISTRY:
            cached_instrs = MEGAKERNEL_REGISTRY[sig_hash]
            for i, inst in enumerate(self.instructions):
                cached_instrs[i].update(inst)

    def _compile_kernel(
        self, compile_key, mapping, num_flat_args, flat_args, n_blocks, op_info_key, arg_info, traced, block, grid
    ):
        """Compile the megakernel and cache it."""
        fake_args = self._create_fake_args(flat_args)
        if self._barrier_config is not None:
            barrier_shape = self._barrier_config.tensor_size
        else:
            barrier_shape = (1,)
        barrier_fake = fake_tensor(Int32, barrier_shape)

        megakernel_obj, _ = self._get_megakernel_class(mapping, num_flat_args, op_info_key, arg_info, traced=traced)

        compile_args = [megakernel_obj, barrier_fake]

        # Add reduction barrier fake arg if we have reduction dependencies
        if self._reduction_barrier_config is not None:
            reduction_barrier_shape = self._reduction_barrier_config.shape
            reduction_barrier_fake = fake_tensor(Int32, reduction_barrier_shape)
            compile_args.append(reduction_barrier_fake)

        compile_args.extend([Int32(n_blocks), *fake_args])

        if traced:
            num_lanes = (block[0] * block[1] * block[2] + 31) // 32
            max_events = len(self.instructions) * 10  # L/C/S + 4 sync events + page events
            row_stride_bytes = (max_events + 1) * 32
            total_blocks = grid[0] * grid[1] * grid[2]
            total_bytes = total_blocks * num_lanes * row_stride_bytes
            trace_buffer_fake = fake_tensor(cute.Uint8, (total_bytes,))
            compile_args.extend([trace_buffer_fake, Int32(row_stride_bytes), Int32(num_lanes)])

        _GLOBAL_COMPILE_CACHE[compile_key] = cute.compile(*compile_args, options="--enable-tvm-ffi")

    def _create_fake_args(self, flat_args) -> list:
        """Create fake tensor arguments for compilation."""
        fake_args = []
        for arg in flat_args:
            if isinstance(arg, torch.Tensor):
                div = self._get_tensor_divisibility(arg)
                fake_args.append(fake_tensor(torch2cute_dtype_map[arg.dtype], arg.shape, divisibility=div))
            elif isinstance(arg, int):
                fake_args.append(Int32(arg))
            elif isinstance(arg, float):
                fake_args.append(Float32(arg))
            else:
                fake_args.append(arg)
        return fake_args

    def _wrap_args(self, flat_args) -> list:
        """Wrap scalar arguments for kernel launch."""
        wrapped = []
        for arg in flat_args:
            if isinstance(arg, int):
                wrapped.append(Int32(arg))
            elif isinstance(arg, float):
                wrapped.append(Float32(arg))
            else:
                wrapped.append(arg)
        return wrapped

    def _execute_kernel(self, compile_key, n_blocks, flat_args, traced, trace_builder, device):
        """Execute the compiled kernel."""
        # Allocate barrier tensor based on configuration
        # If using logical blocks: (NumOps, TotalLogicalBlocks) for fine-grained sync
        # Otherwise: single counter for global sync
        if self._barrier_config is not None:
            barrier_shape = self._barrier_config.tensor_size
            self._barrier = torch.zeros(barrier_shape, dtype=torch.int32, device=device)
            logger.debug("Allocated logical barrier tensor: %s", barrier_shape)
        else:
            self._barrier = torch.zeros(1, dtype=torch.int32, device=device)

        # Allocate reduction barrier tensor if we have reduction dependencies
        if self._reduction_barrier_config is not None:
            reduction_barrier_shape = self._reduction_barrier_config.shape
            self._reduction_barrier = torch.zeros(reduction_barrier_shape, dtype=torch.int32, device=device)
            logger.debug("Allocated reduction barrier tensor: %s", reduction_barrier_shape)

        wrapped_args = self._wrap_args(flat_args)

        launch_args = [self._barrier]

        # Add reduction barrier if present
        if self._reduction_barrier_config is not None:
            launch_args.append(self._reduction_barrier)

        launch_args.extend([Int32(n_blocks), *wrapped_args])

        if traced:
            launch_args.extend(
                [trace_builder._buffer, Int32(trace_builder.row_stride_bytes), Int32(trace_builder.num_lanes)]
            )

        _GLOBAL_COMPILE_CACHE[compile_key](*launch_args)
        torch.cuda.synchronize()

    def _write_trace(self, trace_file, trace_builder, tracing_phases):
        """Helper to write trace data to file without bloating launch()."""
        torch.cuda.synchronize()
        writer = TraceWriter(self.name)
        writer.set_block_type(BlockType("Megakernel", "{blockLinear}", "Block {blockX}"))

        # Setup warp tracks
        warp_track = TrackType("WorkWarp", "Warp {lane}", "Warp {lane}", level=TrackLevel.WARP)
        trace_builder.set_track_type(warp_track)

        # Register L/C/S trace types
        for pt in tracing_phases.values():
            writer.register_trace_type(pt)

        # Register page management trace types
        for pt in PAGE_TRACE_TYPES.values():
            writer.register_trace_type(pt)

        # Register synchronization trace types
        for pt in SYNC_TRACE_TYPES.values():
            writer.register_trace_type(pt)

        writer.add_tensor(trace_builder)

        # Ensure proper .nanotrace extension
        if not trace_file.endswith(".nanotrace"):
            if trace_file.endswith(".json"):
                trace_file = trace_file[:-5] + ".nanotrace"
            else:
                trace_file = trace_file + ".nanotrace"

        writer.write(trace_file)
