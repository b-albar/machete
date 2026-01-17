# Copyright (c) 2025, Machete Authors
"""
Megakernel Code Generation and Execution.

This module implements:
1. Unified code generator for mixed kernel types (sequential + warp-specialized)
2. Kernel template and compilation
3. Megakernel class for building and launching fused kernels
"""

import os
import hashlib
import importlib.util
import logging
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

from machete.megakernel.scheduler import (
    OperationGraph,
    SmemPlanner,
    ScheduleOptimizer,
    ScheduleEntry,
    WarpConfig,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Global Registry and Cache
# ============================================================================

MEGAKERNEL_REGISTRY: Dict[str, List[Dict]] = {}
_GLOBAL_COMPILE_CACHE: Dict[Tuple, Any] = {}


# ============================================================================
# Emitter Context
# ============================================================================


@dataclass
class EmitterContext:
    """Context passed to code emitters."""

    graph: OperationGraph
    schedule: List[ScheduleEntry]
    smem_planner: SmemPlanner
    instructions: List[Dict]
    arg_mapping: Dict[int, List[int]]
    num_flat_args: int = 0

    traced: bool = False
    num_stages: int = 1
    total_smem_bytes: int = 0

    def get_args_str(self, op_idx: int, with_leading_comma: bool = False) -> str:
        """Get argument string for an operation.

        Args:
            op_idx: Operation index
            with_leading_comma: If True, prepend ", " when there are args
        """
        indices = self.arg_mapping.get(op_idx, [])
        if not indices:
            return ""
        args = ", ".join(f"arg_{i}" for i in indices)
        if with_leading_comma:
            return ", " + args
        return args

    def get_smem_var(self, op_idx: int) -> str:
        """Get smem variable name for an operation."""
        return f"smem_op_{op_idx}"

    def op_has_smem(self, op_idx: int) -> bool:
        """Check if operation uses shared memory."""
        return self.graph.nodes[op_idx].smem_bytes > 0


# ============================================================================
# Shared Memory Allocation Emitter
# ============================================================================


class SmemAllocEmitter:
    """Generates shared memory allocation code."""

    def __init__(self, smem_planner: SmemPlanner):
        self.planner = smem_planner

    def emit(self, ctx: EmitterContext) -> List[str]:
        """Emit smem allocation code for all operations.

        Handles two cases:
        1. Ops that can share smem (sequential, no overlap) -> single buffer
        2. Ops with early load optimization -> separate buffers (for overlap)
        """
        code = []
        code.append("        # Shared memory allocation")
        code.append("        smem_alloc = cutlass.utils.SmemAllocator()")

        # Separate ops into shared vs separate buffer groups
        shared_ops = []  # Ops that can share one buffer
        separate_ops = []  # Ops that need their own buffer (early load overlap)

        for op_idx in ctx.graph.get_topological_order():
            alloc = self.planner.get_allocation(op_idx)
            if alloc and alloc.size > 0:
                if self.planner.needs_separate_buffer(op_idx):
                    separate_ops.append(op_idx)
                else:
                    shared_ops.append(op_idx)

        # Allocate shared buffer (for ops without early load overlap)
        if shared_ops:
            max_size = max(self.planner.get_allocation(idx).size for idx in shared_ops)
            max_op_idx = shared_ops[0]  # Use first op's dtype
            inst = ctx.instructions[max_op_idx]
            dtype = inst.get("smem_dtype")
            dtype_str = "cute.Float16" if dtype is None else f"instructions[{max_op_idx}]['smem_dtype']"
            max_elements = max_size // 2

            code.append(
                f"        smem_shared = smem_alloc.allocate_tensor({dtype_str}, cute.make_layout({max_elements}))"
            )

            # Alias shared ops to the shared buffer
            for op_idx in shared_ops:
                code.append(f"        smem_op_{op_idx} = smem_shared")

        # Allocate separate buffers for early-load ops
        for op_idx in separate_ops:
            alloc = self.planner.get_allocation(op_idx)
            inst = ctx.instructions[op_idx]
            dtype = inst.get("smem_dtype")
            dtype_str = "cute.Float16" if dtype is None else f"instructions[{op_idx}]['smem_dtype']"
            elements = alloc.size // 2

            code.append(
                f"        smem_op_{op_idx} = smem_alloc.allocate_tensor({dtype_str}, cute.make_layout({elements}))"
            )

        return code


# ============================================================================
# Unified Code Generator
# ============================================================================


class UnifiedCodeGenerator:
    """Generates code for any combination of kernel types.

    Supports:
    - Sequential (LCS) kernels
    - Warp-specialized kernels
    - Any mix of both in the same megakernel
    - Early load optimization: next op's load overlaps current op's compute
    - Fine-grained synchronization: per-logical-block counters in global memory

    Synchronization strategy:
    - sync_threads() only for intra-block smem visibility (load->compute, compute->store)
    - Global memory counters for inter-op dependencies per logical block
    - An op can start processing block N as soon as its dependencies finish block N
    """

    def generate(self, ctx: EmitterContext) -> List[str]:
        """Generate schedule code with per-op mode dispatch and fine-grained sync."""
        code = []

        # Get warp config (needed if any op uses warp specialization)
        warp_config = self._get_warp_config(ctx)
        has_warp_spec = any(n.uses_warp_specialization for n in ctx.graph.nodes.values())
        topo_order = ctx.graph.get_topological_order()
        num_ops = len(topo_order)

        # Check if we need inter-op synchronization (multiple ops with dependencies)
        needs_inter_op_sync = self._needs_inter_op_sync(ctx, topo_order)

        # Header
        code.append("")
        if has_warp_spec:
            num_consumer = warp_config.num_consumer_warps
            loader_warp = num_consumer
            storer_warp = num_consumer + warp_config.num_loader_warps
            code.append("        # Mixed execution mode (per-op dispatch)")
            code.append(
                f"        # Warp roles - Consumer: 0-{num_consumer - 1}, Loader: {loader_warp}, Storer: {storer_warp}"
            )
            code.append("        warp_id = tidx // Int32(32)")
        else:
            code.append("        # Sequential execution mode")
            num_consumer = loader_warp = storer_warp = 0
        code.append("")

        # Generate per-op code with early load optimization
        early_loaded = set()  # Ops whose load was already issued

        for i, op_idx in enumerate(topo_order):
            node = ctx.graph.nodes[op_idx]

            # Check for early load opportunity
            next_op_idx = topo_order[i + 1] if i + 1 < len(topo_order) else None
            can_early_load = self._can_early_load(ctx, op_idx, next_op_idx, early_loaded)

            # Emit code for this op
            self._emit_op(
                code,
                ctx,
                op_idx,
                node,
                warp_config,
                num_consumer,
                loader_warp,
                storer_warp,
                skip_load=(op_idx in early_loaded),
                early_load_next=next_op_idx if can_early_load else None,
                needs_inter_op_sync=needs_inter_op_sync,
                is_last_op=(i == num_ops - 1),
            )

            # Track early-loaded ops
            if can_early_load:
                early_loaded.add(next_op_idx)

            code.append("")

        return code

    def _needs_inter_op_sync(self, ctx: EmitterContext, topo_order: List[int]) -> bool:
        """Check if we need fine-grained inter-op synchronization.

        We need it when:
        - There are multiple ops
        - At least one op has dependencies on a previous op
        """
        if len(topo_order) <= 1:
            return False

        for op_idx in topo_order[1:]:  # Skip first op
            node = ctx.graph.nodes[op_idx]
            if node.depends_on:
                return True
        return False

    def _get_warp_config(self, ctx: EmitterContext) -> WarpConfig:
        """Get warp config from first warp-specialized op, or default."""
        for node in ctx.graph.nodes.values():
            if node.warp_config:
                return node.warp_config
        return WarpConfig()

    def _get_op_args(self, ctx: EmitterContext, op_idx: int):
        """Build argument strings for an operation."""
        args = ctx.get_args_str(op_idx)
        smem = ctx.get_smem_var(op_idx)
        has_smem = ctx.op_has_smem(op_idx)

        if has_smem and args:
            all_args = f"logical_idx, {smem}, {args}"
        elif has_smem:
            all_args = f"logical_idx, {smem}"
        elif args:
            all_args = f"logical_idx, {args}"
        else:
            all_args = "logical_idx"

        return all_args, all_args, has_smem

    def _can_early_load(self, ctx: EmitterContext, curr_op: int, next_op: Optional[int], early_loaded: set) -> bool:
        """Check if next_op's load can overlap with curr_op's compute."""
        if next_op is None:
            return False
        if next_op in early_loaded:
            return False
        return ctx.graph.can_move_load_early(next_op)

    def _emit_op(
        self,
        code: List[str],
        ctx: EmitterContext,
        op_idx: int,
        node,
        warp_config: WarpConfig,
        num_consumer: int,
        loader_warp: int,
        storer_warp: int,
        skip_load: bool,
        early_load_next: Optional[int],
        needs_inter_op_sync: bool,
        is_last_op: bool,
    ):
        """Emit code for a single operation (either sequential or warp-specialized).

        Fine-grained synchronization:
        - sync_threads() only for smem visibility within a block
        - completion_counters[op_idx] tracks completed logical blocks
        - Before load: wait for dependencies' counters[logical_idx] to be set
        - After store: signal this op's counter[logical_idx]
        """
        load_store_args, compute_args, has_smem = self._get_op_args(ctx, op_idx)
        is_warp_spec = node.uses_warp_specialization

        mode_str = "warp-specialized" if is_warp_spec else "sequential"
        code.append(f"        # Op {op_idx}: {node.name} ({mode_str})")

        # === WAIT FOR DEPENDENCIES (fine-grained per-block sync) ===
        if needs_inter_op_sync and node.depends_on:
            code.append("        # Wait for dependencies to complete this logical block")
            code.append("        if tidx == Int32(0):")
            for dep_idx in sorted(node.depends_on):
                # Index into flat counters: dep_idx * n_blocks + logical_idx
                code.append(f"            dep_counter_idx = Int32({dep_idx}) * n_blocks + logical_idx")
                code.append("            while counters[dep_counter_idx] == Int32(0):")
                code.append("                nanosleep(100)")
            code.append("        cute.arch.sync_threads()")

        # === LOAD PHASE ===
        if not skip_load:
            if is_warp_spec:
                code.append(f"        if warp_id == Int32({loader_warp}):")
                code.append(f"            op_{op_idx}_load({load_store_args})")
            else:
                code.append(f"        op_{op_idx}_load({load_store_args})")

            if has_smem:
                code.append("        cute.arch.sync_threads()")
        else:
            code.append("        # (Load already issued via early-load optimization)")

        # === COMPUTE PHASE (with optional early load for next op) ===
        if early_load_next is not None:
            next_node = ctx.graph.nodes[early_load_next]
            next_load_args, _, next_has_smem = self._get_op_args(ctx, early_load_next)
            next_is_warp_spec = next_node.uses_warp_specialization

            code.append(f"        # Early load for Op {early_load_next}: {next_node.name}")

            # Issue next op's load (respecting its mode)
            if next_is_warp_spec:
                code.append(f"        if warp_id == Int32({loader_warp}):")
                code.append(f"            op_{early_load_next}_load({next_load_args})")
            else:
                code.append(f"        op_{early_load_next}_load({next_load_args})")

        # Issue current op's compute (respecting its mode)
        if is_warp_spec:
            code.append(f"        if warp_id < Int32({num_consumer}):")
            code.append(f"            op_{op_idx}_compute({compute_args})")
        else:
            code.append(f"        op_{op_idx}_compute({compute_args})")

        # Sync after compute if needed for smem visibility before store
        if has_smem or (early_load_next is not None):
            code.append("        cute.arch.sync_threads()")

        # === STORE PHASE ===
        if is_warp_spec:
            code.append(f"        if warp_id == Int32({storer_warp}):")
            code.append(f"            op_{op_idx}_store({load_store_args})")
            # Sync to ensure store completes before signaling
            code.append("        cute.arch.sync_threads()")
        else:
            code.append(f"        op_{op_idx}_store({load_store_args})")
            # Sync only if needed: inter-op sync or not the last op
            if needs_inter_op_sync or not is_last_op:
                code.append("        cute.arch.sync_threads()")

        # === SIGNAL COMPLETION (fine-grained per-block sync) ===
        if needs_inter_op_sync and not is_last_op:
            code.append("        # Signal this logical block is complete")
            code.append("        if tidx == Int32(0):")
            # Index into flat counters: op_idx * n_blocks + logical_idx
            code.append(f"            counter_idx = Int32({op_idx}) * n_blocks + logical_idx")
            code.append("            counters[counter_idx] = Int32(1)")


# ============================================================================
# Kernel Template
# ============================================================================


class KernelTemplate:
    """Template for generating complete megakernel code."""

    # Template without completion counters (single op or no dependencies)
    TEMPLATE_SIMPLE = """# Auto-generated megakernel - DO NOT EDIT
import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.runtime import make_ptr
from cutlass._mlir.dialects._cute_nvgpu_enum_gen import AddressSpace
from machete.megakernel.core import MEGAKERNEL_REGISTRY
from machete.megakernel.utils import nanosleep, atomic_add_i32

# Retrieve instructions from registry
instructions = MEGAKERNEL_REGISTRY["{sig_hash}"]

# Bind module-level symbols
{bindings}

# Pre-compiled kernel (populated on first use)
_compiled_kernel = None

class GeneratedMegakernel:
    @cute.jit
    def __call__(self, n_blocks, {args}):
        self.{kernel_name}(n_blocks, {args}).launch(
            grid={grid},
            block={block},
            smem={smem}
        )

    @cute.kernel
    def {kernel_name}(self, n_blocks, {args}):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        logical_idx = bidx
{smem_alloc}
{schedule_code}

def get_compiled_kernel(n_blocks, {args}):
    \"\"\"Get or compile the kernel with proper type hints.\"\"\"
    global _compiled_kernel
    if _compiled_kernel is None:
        _compiled_kernel = cute.compile(
            GeneratedMegakernel(),
            n_blocks,
            {args}
        )
    return _compiled_kernel
"""

    # Template with completion counters for fine-grained sync
    TEMPLATE_WITH_COUNTERS = """# Auto-generated megakernel - DO NOT EDIT
import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.runtime import make_ptr
from cutlass._mlir.dialects._cute_nvgpu_enum_gen import AddressSpace
from machete.megakernel.core import MEGAKERNEL_REGISTRY
from machete.megakernel.utils import nanosleep, atomic_add_i32

# Retrieve instructions from registry
instructions = MEGAKERNEL_REGISTRY["{sig_hash}"]

# Bind module-level symbols
{bindings}

NUM_OPS = {num_ops}

# Pre-compiled kernel (populated on first use)
_compiled_kernel = None

class GeneratedMegakernel:
    @cute.jit
    def __call__(self, n_blocks, completion_counters, {args}):
        self.{kernel_name}(n_blocks, completion_counters, {args}).launch(
            grid={grid},
            block={block},
            smem={smem}
        )

    @cute.kernel
    def {kernel_name}(self, n_blocks, completion_counters, {args}):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        logical_idx = bidx
{smem_alloc}
        # completion_counters is a flat tensor of size NUM_OPS * n_blocks
        # Access pattern: counters[op_idx * n_blocks + logical_idx]
        counters = completion_counters
{schedule_code}

def get_compiled_kernel(n_blocks, completion_counters, {args}):
    \"\"\"Get or compile the kernel with proper type hints.\"\"\"
    global _compiled_kernel
    if _compiled_kernel is None:
        _compiled_kernel = cute.compile(
            GeneratedMegakernel(),
            n_blocks,
            completion_counters,
            {args}
        )
    return _compiled_kernel
"""

    def __init__(
        self,
        ctx: EmitterContext,
        code_gen: UnifiedCodeGenerator,
        smem_emitter: SmemAllocEmitter,
        needs_completion_counters: bool = False,
    ):
        self.ctx = ctx
        self.code_gen = code_gen
        self.smem_emitter = smem_emitter
        self.needs_completion_counters = needs_completion_counters

    def generate(
        self,
        sig_hash: str,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
    ) -> str:
        """Generate complete kernel module source."""
        # Generate bindings
        bindings = self._generate_bindings()

        # Generate smem allocation
        smem_code = "\n".join(self.smem_emitter.emit(self.ctx))

        # Generate schedule code
        schedule_code = "\n".join(self.code_gen.generate(self.ctx))

        # Generate args list
        args = ", ".join(f"arg_{i}" for i in range(self.ctx.num_flat_args))

        # Choose template based on whether we need completion counters
        template = self.TEMPLATE_WITH_COUNTERS if self.needs_completion_counters else self.TEMPLATE_SIMPLE

        num_ops = len(self.ctx.graph.nodes)

        return template.format(
            sig_hash=sig_hash,
            bindings=bindings,
            kernel_name=f"kernel_{sig_hash[:8]}",
            args=args,
            grid=grid,
            block=block,
            smem=self.ctx.total_smem_bytes,
            smem_alloc=smem_code,
            schedule_code=schedule_code,
            num_ops=num_ops,
        )

    def _generate_bindings(self) -> str:
        """Generate module-level bindings for L/C/S functions."""
        lines = []
        for op_idx in self.ctx.graph.get_topological_order():
            lines.append(f"op_{op_idx}_load = instructions[{op_idx}]['load']")
            lines.append(f"op_{op_idx}_compute = instructions[{op_idx}]['compute']")
            lines.append(f"op_{op_idx}_store = instructions[{op_idx}]['store']")
        return "\n".join(lines)


# ============================================================================
# Megakernel Class
# ============================================================================


class Megakernel:
    """Simplified megakernel that uses operation graph for scheduling.

    Key features:
    1. No paged memory - smem is computed from graph
    2. Static scheduling - computed at compile time
    3. Modular code generation
    4. Unified mixed kernel type support
    """

    def __init__(
        self,
        name: str = "megakernel",
        mode: str = "forward",
        num_stages: int = 1,
    ):
        self.name = name
        self.mode = mode
        self.num_stages = num_stages

        self.instructions: List[Dict] = []
        self._graph: Optional[OperationGraph] = None

        self.gen_dir = os.path.join(os.path.dirname(__file__), ".generated")
        os.makedirs(self.gen_dir, exist_ok=True)

    def add(self, op, *args):
        """Add an operation to the megakernel.

        Args:
            op: A FusableKernel, WarpSpecializedKernel, or similar object
            *args: Arguments to pass to the kernel's L/C/S methods
        """
        kernel = op

        if self.mode == "forward":
            load_fn = getattr(kernel, "load_forward", None)
            compute_fn = getattr(kernel, "compute_forward", None)
            store_fn = getattr(kernel, "store_forward", None)
        else:
            load_fn = getattr(kernel, "load_backward", None)
            compute_fn = getattr(kernel, "compute_backward", None)
            store_fn = getattr(kernel, "store_backward", None)

        # Get smem info (separate for forward/backward, fallback to smem_size)
        smem_size_fwd = getattr(kernel, "smem_size_fwd", 0) or getattr(kernel, "smem_size", 0)
        smem_size_bwd = getattr(kernel, "smem_size_bwd", 0) or getattr(kernel, "smem_size", 0)
        smem_dtype = getattr(kernel, "cute_dtype", None)

        # Get logical grid size
        logical_grid_size = 1
        if hasattr(kernel, "get_logical_grid_size"):
            try:
                logical_grid_size = kernel.get_logical_grid_size(*args)
            except Exception:
                logical_grid_size = 1

        # Get warp config for warp-specialized kernels
        uses_warp_spec = getattr(kernel, "uses_warp_specialization", False)
        warp_config = getattr(kernel, "warp_config", None) if uses_warp_spec else None

        # Get grid/block functions if available
        grid_fn = getattr(kernel, "grid_fn", None)
        block_fn = getattr(kernel, "block_fn", None)

        self.instructions.append(
            {
                "kernel": kernel,
                "load": load_fn,
                "compute": compute_fn,
                "store": store_fn,
                "args": list(args),
                "smem_size_fwd": smem_size_fwd,
                "smem_size_bwd": smem_size_bwd,
                "smem_dtype": smem_dtype,
                "logical_grid_size": logical_grid_size,
                "uses_warp_specialization": uses_warp_spec,
                "warp_config": warp_config,
                "grid_fn": grid_fn,
                "block_fn": block_fn,
            }
        )

        # Invalidate cached graph
        self._graph = None

    def clear(self):
        """Clear all instructions."""
        self.instructions = []
        self._graph = None

    def _build_graph(self) -> OperationGraph:
        """Build operation graph from instructions."""
        if self._graph is not None:
            return self._graph

        self._graph = OperationGraph.from_instructions(self.instructions, self.mode)
        return self._graph

    def _calculate_logical_blocks(self) -> int:
        """Calculate total logical blocks across all operations."""
        graph = self._build_graph()
        return graph.compute_logical_grid_size()

    def launch(
        self,
        n_blocks: int,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        stream=None,
        trace_file: Optional[str] = None,
    ):
        """Launch the megakernel (compatibility wrapper for SingleKernel).

        Args:
            n_blocks: Number of logical blocks (used for grid if grid[0] matches)
            grid: Grid dimensions (list or tuple)
            block: Block dimensions (threads per block, list or tuple)
            stream: CUDA stream (optional)
            trace_file: Path to trace file for debugging (optional)
        """
        # Convert lists to tuples for hashability in compile cache
        grid = tuple(grid) if isinstance(grid, list) else grid
        block = tuple(block) if isinstance(block, list) else block
        self.launch_logical(block=block, grid=grid, stream=stream, trace_file=trace_file)

    def launch_logical(
        self,
        block: Tuple[int, int, int],
        grid: Optional[Tuple[int, int, int]] = None,
        stream=None,
        trace_file: Optional[str] = None,
    ):
        """Launch the megakernel with logical block scheduling.

        Args:
            block: Block dimensions (threads per block)
            grid: Grid dimensions (optional, computed from logical blocks if not provided)
            stream: CUDA stream (optional)
            trace_file: Path to trace file for debugging (optional)
        """
        if not self.instructions:
            raise ValueError("No operations added to megakernel")

        # Build graph
        graph = self._build_graph()

        # Calculate logical blocks
        n_blocks = graph.compute_logical_grid_size()

        # Determine grid if not provided
        if grid is None:
            grid = (n_blocks, 1, 1)

        # Plan shared memory
        smem_planner = SmemPlanner(graph)
        total_smem = smem_planner.plan(self.num_stages, enable_early_load=True)

        # Add space for semaphores
        num_ops = len(self.instructions)
        has_warp_spec = any(n.uses_warp_specialization for n in graph.nodes.values())
        if has_warp_spec:
            num_sems = self.num_stages * num_ops * 3 + 1
        else:
            num_sems = 2
        total_smem += num_sems * 4  # 4 bytes per semaphore

        # Build argument mapping
        arg_mapping: Dict[int, List[int]] = {}
        curr = 0
        for i, inst in enumerate(self.instructions):
            num_args = len(inst["args"])
            arg_mapping[i] = list(range(curr, curr + num_args))
            curr += num_args

        # Generate optimized schedule
        optimizer = ScheduleOptimizer(graph, smem_planner)
        schedule = optimizer.optimize()

        # Create emitter context
        ctx = EmitterContext(
            graph=graph,
            schedule=schedule,
            smem_planner=smem_planner,
            instructions=self.instructions,
            arg_mapping=arg_mapping,
            num_flat_args=curr,
            traced=trace_file is not None,
            num_stages=self.num_stages,
            total_smem_bytes=total_smem,
        )

        # Compile and run
        self._compile_and_run(ctx, n_blocks, grid, block, stream, trace_file)

    def _compute_sig_hash(self, ctx: EmitterContext) -> str:
        """Compute unique hash for this kernel configuration."""
        data = f"{self.name}_{self.mode}_{ctx.num_flat_args}_{self.num_stages}"
        for i, node in ctx.graph.nodes.items():
            kernel = self.instructions[i]["kernel"]
            # Capture identifying state from the kernel instance
            state = []
            if hasattr(kernel, "__dict__"):
                for k, v in kernel.__dict__.items():
                    # Include simple types that likely define the kernel's behavior
                    if isinstance(v, (str, int, float, bool, tuple, torch.dtype)):
                        state.append(f"{k}={v}")

            kernel_state = "_".join(sorted(state))
            data += f"_{node.name}_{node.smem_bytes}_{node.kernel_type.name}_{kernel_state}"

        sig = hashlib.md5(data.encode()).hexdigest()
        return sig

    def _compile_and_run(
        self,
        ctx: EmitterContext,
        n_blocks: int,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        stream=None,
        trace_file: Optional[str] = None,
    ):
        """Compile and execute the kernel."""
        sig_hash = self._compute_sig_hash(ctx)

        # Store instructions in registry
        MEGAKERNEL_REGISTRY[sig_hash] = self.instructions

        # Check if we need completion counters for fine-grained sync
        needs_counters = self._needs_completion_counters(ctx)

        # Check compile cache
        compile_key = (self.name, self.mode, sig_hash, grid, block, needs_counters)

        if compile_key not in _GLOBAL_COMPILE_CACHE:
            # Generate code
            code_gen = UnifiedCodeGenerator()
            smem_emitter = SmemAllocEmitter(ctx.smem_planner)
            template = KernelTemplate(ctx, code_gen, smem_emitter, needs_counters)
            source = template.generate(sig_hash, grid, block)

            # Write module
            gen_path = os.path.join(self.gen_dir, f"kernel_{sig_hash}.py")
            with open(gen_path, "w") as f:
                f.write(source)

            # Load module
            spec = importlib.util.spec_from_file_location(f"gen_{sig_hash}", gen_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            _GLOBAL_COMPILE_CACHE[compile_key] = module

        # Get module
        module = _GLOBAL_COMPILE_CACHE[compile_key]

        # Flatten arguments and convert tensors to CuTe pointers
        # Note: We pass Pointer objects which can be created outside JIT context
        # cute.compile() will compile once based on types, then reuse for same types
        import torch
        from cutlass.cute.runtime import make_ptr
        from cutlass._mlir.dialects._cute_nvgpu_enum_gen import AddressSpace
        from quack.cute_dsl_utils import torch2cute_dtype_map

        flat_args = []
        for inst in self.instructions:
            for arg in inst["args"]:
                if isinstance(arg, torch.Tensor):
                    # Convert tensor to CuTe pointer with 16-byte alignment for async copy
                    tensor = arg.detach() if arg.requires_grad else arg
                    cute_dtype = torch2cute_dtype_map[tensor.dtype]
                    ptr = make_ptr(cute_dtype, tensor.data_ptr(), AddressSpace.gmem, assumed_align=16)
                    flat_args.append(ptr)
                else:
                    flat_args.append(arg)

        # Get or compile the kernel, then execute
        if needs_counters:
            import cutlass
            num_ops = len(self.instructions)
            # Flat tensor: num_ops * n_blocks, indexed as op_idx * n_blocks + logical_idx
            completion_counters = torch.zeros(num_ops * n_blocks, dtype=torch.int32, device="cuda")
            # Convert to CuTe pointer
            counters_ptr = make_ptr(cutlass.Int32, completion_counters.data_ptr(), AddressSpace.gmem, assumed_align=16)
            compiled = module.get_compiled_kernel(n_blocks, counters_ptr, *flat_args)
            compiled(n_blocks, counters_ptr, *flat_args)
        else:
            compiled = module.get_compiled_kernel(n_blocks, *flat_args)
            compiled(n_blocks, *flat_args)

        # Write trace file if requested
        if trace_file:
            self._write_trace(trace_file, ctx, n_blocks, grid, block)

    def _needs_completion_counters(self, ctx: EmitterContext) -> bool:
        """Check if we need fine-grained inter-op synchronization."""
        topo_order = ctx.graph.get_topological_order()
        if len(topo_order) <= 1:
            return False

        for op_idx in topo_order[1:]:
            node = ctx.graph.nodes[op_idx]
            if node.depends_on:
                return True
        return False

    def _write_trace(
        self,
        trace_file: str,
        ctx: EmitterContext,
        n_blocks: int,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
    ):
        """Write a trace file for debugging.

        Creates a nanotrace-format JSON file with kernel execution info.
        """
        import json

        # Ensure proper .nanotrace extension
        if not trace_file.endswith(".nanotrace"):
            if trace_file.endswith(".json"):
                trace_file = trace_file[:-5] + ".nanotrace"
            else:
                trace_file = trace_file + ".nanotrace"

        # Build trace data
        operations = []
        for op_idx in ctx.graph.get_topological_order():
            node = ctx.graph.nodes[op_idx]
            operations.append(
                {
                    "idx": op_idx,
                    "name": node.name,
                    "kernel_type": node.kernel_type.name,
                    "smem_bytes": node.smem_bytes,
                    "smem_bytes_fwd": node.smem_bytes_fwd,
                    "smem_bytes_bwd": node.smem_bytes_bwd,
                    "uses_warp_specialization": node.uses_warp_specialization,
                }
            )

        trace_data = {
            "version": "1.0",
            "kernel_name": self.name,
            "mode": self.mode,
            "n_blocks": n_blocks,
            "grid": list(grid),
            "block": list(block),
            "num_stages": self.num_stages,
            "total_smem_bytes": ctx.total_smem_bytes,
            "operations": operations,
        }

        # Write trace file
        with open(trace_file, "w") as f:
            json.dump(trace_data, f, indent=2)
