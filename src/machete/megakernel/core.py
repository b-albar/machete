# Copyright (c) 2025, Machete Authors
import torch
import cutlass.cute as cute
from cutlass import Int32, Float32
import os
import importlib.util
import hashlib
from typing import Callable, Union
from machete.megakernel.interface import FusableKernel, MegakernelOp, FusableOp
from machete.megakernel.scheduler import (
    NoBubblesScheduler,
    NoBubblesConfig,
    PageAwareScheduler,
    MicroOpType,
    OpDescriptor,
    build_op_descriptor_from_kernel,
)
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.compile_utils import make_fake_tensor as fake_tensor
import tvm_ffi.core
import logging
from cutedsl_trace import TraceType, BlockType, TraceWriter, DynamicTraceBuilder, TrackType, TrackLevel

logger = logging.getLogger(__name__)

# TRACING CONFIGURATION - L/C/S phases plus page management
TRACING_PHASES = {
    MicroOpType.LOAD: TraceType("LOAD", "L-{0}", "Load phase for Op {0}", param_count=1),
    MicroOpType.COMPUTE: TraceType("COMPUTE", "C-{0}", "Compute phase for Op {0}", param_count=1),
    MicroOpType.STORE: TraceType("STORE", "S-{0}", "Store phase for Op {0}", param_count=1),
}

# Additional trace types for page management (No Bubbles pattern)
PAGE_TRACE_TYPES = {
    "ACQUIRE_WAIT": TraceType("ACQUIRE_WAIT", "AW-{0}", "Waiting for pages (Op {0})", param_count=1),
    "ACQUIRE_DONE": TraceType("ACQUIRE_DONE", "AD-{0},{1}", "Acquired pages {1} for Op {0}", param_count=2),
    "RELEASE": TraceType("RELEASE", "RL-{0},{1}", "Released pages {1} for Op {0}", param_count=2),
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
        num_pages: int = 0,
        page_size: int = 16384,
    ):
        """
        Args:
            name: Kernel name for caching/debugging.
            mode: "forward" or "backward".
            paged_pool_bytes: Total size of the paged shared memory pool in bytes.
                              Set to 0 (default) for compute-only mode.
            num_pages: Number of pages for No Bubbles scheduling.
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
        if num_pages > 0:
            self.num_stages = num_pages
            self.paged_pool_bytes = num_pages * page_size
        elif paged_pool_bytes > 0:
            self.num_stages = paged_pool_bytes // page_size
            self.paged_pool_bytes = paged_pool_bytes
        else:
            self.num_stages = 0
            self.paged_pool_bytes = 0

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
            smem_size = kernel.smem_per_stage
            logger.debug(
                "Adding kernel %s with smem_size=%s num_stages=%s",
                kernel.__class__.__name__,
                smem_size,
                kernel.num_stages,
            )

            self.instructions.append(
                {
                    "kernel": kernel,
                    "load": load_fn,
                    "compute": compute_fn,
                    "store": store_fn,
                    "args": list(args),
                    "needs_sync": False,
                    "needs_block_sync": True,
                    "smem_size": smem_size,
                    "num_stages": kernel.num_stages,
                    "smem_dtype": smem_dtype,
                    "op_obj": kernel,
                }
            )
            return

        # Handle MegakernelOp directly
        if isinstance(op, MegakernelOp):
            actual_op = op
        elif hasattr(op, "_machete_is_op"):
            # Decorated function - wrap in FusableOp
            smem_per_page = getattr(op, "_machete_smem_per_page", 0)
            num_pages = getattr(op, "_machete_num_pages", 1)

            instance = getattr(op, "__self__", None)
            if instance:
                if hasattr(instance, "smem_per_page"):
                    smem_per_page = instance.smem_per_stage
                if hasattr(instance, "num_stages"):
                    num_pages = instance.num_stages

            actual_op = FusableOp(
                compute_func=op,
                num_tensors=op._machete_num_tensors,
                needs_sync=op._machete_needs_sync,
                smem_per_page=smem_per_page,
                num_pages=num_pages,
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
                "needs_sync": actual_op.needs_global_sync,
                "needs_block_sync": getattr(actual_op, "needs_block_sync", True),
                "smem_size": actual_op.smem_per_stage,
                "num_stages": actual_op.num_stages,
                "smem_dtype": smem_dtype,
                "op_obj": actual_op,
            }
        )

    def clear(self):
        self.instructions = []

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
        """Generate the kernel operation code (allocations + scheduling)."""
        ops = ["        smem_alloc = cutlass.utils.SmemAllocator()"]

        # Paged pool allocation
        if self.paged_pool_bytes > 0:
            ops.append(f"        # Paged pool for No Bubbles: {self.paged_pool_bytes} bytes ({self.num_stages} pages)")
            ops.append(
                f"        paged_pool = smem_alloc.allocate_tensor("
                f"cute.Uint8, cute.make_layout({self.paged_pool_bytes}))"
            )
        else:
            ops.append("        paged_pool = smem_alloc.allocate_tensor(cute.Uint8, cute.make_layout(1))")
            ops.append("        page_idx = Int32(0)")

        # Per-op smem tensor allocations
        ops.extend(self._generate_smem_allocations())

        # Generate schedule
        ops.append("        # Generate Schedule")
        use_page_aware = self.paged_pool_bytes > 0 and self.num_stages >= 2

        if use_page_aware:
            ops.extend(self._generate_page_aware_schedule(mapping, traced))
        else:
            ops.extend(self._generate_simple_schedule(mapping, traced))

        return ops

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
        max_smem_per_stage = 0
        for inst in self.instructions:
            max_smem_per_stage = max(max_smem_per_stage, inst["smem_size"])

        # pages_per_stage = ceil(smem_per_stage / page_size)
        pages_per_stage = (max_smem_per_stage + self.page_size - 1) // self.page_size
        pages_per_stage = max(1, pages_per_stage)

        # For double buffering, we need 2 * pages_per_stage
        pages_per_op = 2 * pages_per_stage

        ops.append(f"        # Page-aware scheduling: {self.num_stages} pages, {pages_per_op} per op")
        ops.append("        page_idx = Int32(0)")
        ops.append("        next_page = Int32(0)")
        ops.append(f"        # Page semaphore initialized to {self.num_stages} (all pages free)")

        # Build schedule
        config = NoBubblesConfig(num_pages=self.num_stages)
        page_scheduler = PageAwareScheduler(config)
        op_descs = self._build_op_descriptors()

        # Use async pipeline if any op supports it, otherwise fallback to standard page-aware
        # For now, we use generate_async_pipeline_schedule as it's the most optimized
        page_scheduler.generate_async_pipeline_schedule(op_descs, pages_per_op=pages_per_op)

        # Generate code for each micro-op
        for uop in page_scheduler.micro_ops:
            ops.extend(self._generate_page_aware_uop(uop, mapping, traced))

        return ops

    def _build_op_descriptors(self) -> list:
        """Build OpDescriptors from instructions for scheduling."""
        op_descs = []
        for i, inst in enumerate(self.instructions):
            kernel = inst.get("kernel") or inst.get("op_obj")
            if kernel and hasattr(kernel, "load_forward"):
                desc = build_op_descriptor_from_kernel(kernel, i, self.mode)
            else:
                desc = OpDescriptor(
                    name=f"Op{i}",
                    op_idx=i,
                    needs_block_sync=inst.get("needs_block_sync", True),
                    needs_global_sync=inst.get("needs_sync", False),
                )
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
            ops.append("        cute.arch.sync_threads()")
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
            ops.append("        cute.arch.sync_threads()  # Ensure prior stores visible")
            if traced:
                fmt_id = PAGE_TRACE_TYPES["ACQUIRE_WAIT"].id
                ops.append(
                    f"        lane = cutedsl_trace.device.end_event_dynamic_raw_1("
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
                f"        lane = cutedsl_trace.device.end_event_dynamic_raw_2("
                f"t_acq_done, trace_buffer, trace_row_stride, lane, "
                f"Int32({fmt_id}), Int32({op_idx}), Int32({pages_mask}))"
            )

        # Execute load
        if traced:
            ops.append("        t_start = cutedsl_trace.device.start()")

        smem_str = get_smem_arg("page_idx")
        ops.append(f"        op_{op_idx}_load(paged_pool, page_idx, {smem_str}{args_str})")

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
        ops.append(f"        op_{op_idx}_compute({smem_str}{args_str})")

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
        ops.append(f"        op_{op_idx}_store(paged_pool, page_idx, {smem_str}{args_str})")

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
            ops.append("        cute.arch.sync_threads()  # Ensure store complete")
            if traced:
                fmt_id = PAGE_TRACE_TYPES["RELEASE"].id
                pages_mask = sum(1 << p for p in released)
                ops.append(
                    f"        lane = cutedsl_trace.device.end_event_dynamic_raw_2("
                    f"t_rel, trace_buffer, trace_row_stride, lane, "
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
            "        cute.arch.sync_threads()",
            "        sync_step = sync_step + Int32(1)",
        ]

    def _generate_simple_schedule(self, mapping, traced: bool) -> list:
        """Generate simple pipeline scheduling code (legacy mode)."""
        ops = []
        scheduler = NoBubblesScheduler()
        scheduler.generate_pipeline_schedule(self.instructions, use_pipeline=(self.paged_pool_bytes > 0))

        if self.paged_pool_bytes > 0:
            ops.append("        page_idx = Int32(0)")
            ops.append("        next_page = Int32(0)")

        for uop in scheduler.micro_ops:
            i = uop.op_idx
            args_str = ", ".join([f"arg_{idx}" for idx in mapping[i]])
            get_smem_arg = self._make_smem_arg_getter(i)

            if traced and uop.type in TRACING_PHASES:
                ops.append("        t_start = cutedsl_trace.device.start()")

            if uop.type == MicroOpType.LOAD:
                target_page = "page_idx"
                if uop.desc == "Prefetch":
                    ops.append("        next_page = (page_idx + Int32(1)) % Int32(2)")
                    target_page = "next_page"
                smem_str = get_smem_arg(target_page)
                ops.append(f"        op_{i}_load(paged_pool, {target_page}, {smem_str}{args_str})")

            elif uop.type == MicroOpType.COMPUTE:
                smem_str = get_smem_arg("page_idx")
                ops.append(f"        op_{i}_compute({smem_str}{args_str})")

            elif uop.type == MicroOpType.STORE:
                smem_str = get_smem_arg("page_idx")
                ops.append(f"        op_{i}_store(paged_pool, page_idx, {smem_str}{args_str})")

            elif uop.type == MicroOpType.SYNC_BLOCK:
                ops.append("        cute.arch.sync_threads()")

            elif uop.type == MicroOpType.SYNC_GLOBAL:
                ops.extend(self._gen_global_sync())

            elif uop.type == MicroOpType.ADVANCE_PAGE:
                ops.append("        page_idx = next_page")

            if traced and uop.type in TRACING_PHASES:
                fmt_id = TRACING_PHASES[uop.type].id
                ops.append(
                    f"        lane = cutedsl_trace.device.end_event_dynamic_raw_1("
                    f"t_start, trace_buffer, trace_row_stride, lane, Int32({fmt_id}), Int32({i}))"
                )

        return ops

    def _generate_kernel_source(self, sig_hash: str, all_args_str: str, bindings: list, ops: list, traced: bool) -> str:
        """Generate the complete Python source code for the kernel module."""
        bindings_content = "\n".join(bindings)
        ops_content = "\n".join(ops)
        kernel_name = f"kernel_{sig_hash[:8]}"

        trace_args = ", trace_buffer, trace_row_stride, trace_num_lanes" if traced else ""
        bidx_init = "bidx, _, _ = cute.arch.block_idx()" if traced else ""
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

# Retrieve instructions from registry
instructions = MEGAKERNEL_REGISTRY["{sig_hash}"]

# Bind module-level symbols
{bindings_content}

class GeneratedMegakernel:
    @cute.jit
    def __call__(self, barrier_tensor, n_blocks, {all_args_str}{trace_args}):
        # We launch utilizing grid/block from the registry instructions
        # This allows the caller to update registry with current grid/block before call.
        self.{kernel_name}(barrier_tensor, n_blocks, {all_args_str}{trace_args}).launch(
            grid=instructions[0]['grid'],
            block=instructions[0]['block'],
            smem=instructions[0]['total_smem']
        )

    @cute.kernel
    def {kernel_name}(self, barrier_tensor, n_blocks, {all_args_str}{trace_args}):
        sync_step = Int32(0)
        tidx, _, _ = cute.arch.thread_idx()
        {bidx_init}
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

    def launch(self, n_blocks: int, grid, block, stream=None, trace_file=None):
        """Launch the megakernel with the given configuration."""
        traced = trace_file is not None

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
        max_events = len(self.instructions) * 6
        return DynamicTraceBuilder(
            num_lanes=num_lanes, max_events_per_lane=max_events, grid_dims=(grid[0], grid[1], grid[2])
        )

    def _calculate_smem(self, device) -> int:
        """Calculate shared memory requirement, reducing pages if needed."""
        props = torch.cuda.get_device_properties(device)
        max_smem = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
        limit_smem = max_smem - 2048  # Safety margin

        def calc_smem():
            total = self.paged_pool_bytes
            for inst in self.instructions:
                n_pages = inst["num_stages"]
                if self.paged_pool_bytes > 0 and inst["smem_size"] > 0:
                    n_pages = max(inst["num_stages"], self.num_stages)
                total += inst["smem_size"] * n_pages
            return total + 1024

        # Adaptive reduction
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
        barrier_fake = fake_tensor(Int32, (1,))

        megakernel_obj, _ = self._get_megakernel_class(mapping, num_flat_args, op_info_key, arg_info, traced=traced)

        compile_args = [megakernel_obj, barrier_fake, Int32(n_blocks), *fake_args]

        if traced:
            num_lanes = (block[0] * block[1] * block[2] + 31) // 32
            max_events = len(self.instructions) * 6
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
        self._barrier = torch.zeros(1, dtype=torch.int32, device=device)
        wrapped_args = self._wrap_args(flat_args)

        launch_args = [self._barrier, Int32(n_blocks), *wrapped_args]
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

        writer.add_tensor(trace_builder)
        writer.write(trace_file)
