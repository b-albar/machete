# Copyright (c) 2025, Machete Authors
"""Persistent megakernel with warp specialization and paged memory.

This module implements the "GPU OS" pattern:
- Warp specialization: Controller, Loader, Consumer, Storer, Launcher run decoupled loops
- Instruction pipeline: Controller fetches → Ring buffer → Warps execute
- Paged shared memory: Physical pages rotate, sem_loaded/sem_consumed for sync
- Global scoreboard: g_bar[op][chunk] for fine-grained inter-block dependencies

Architecture Overview:
┌─────────────────────────────────────────────────────────────────┐
│  Controller Warp (Warp 3)                                       │
│    The "CPU" of the block. Orchestrates instruction flow.       │
│    while has_work:                                              │
│      fetch instruction[inst_ptr++] from global memory           │
│      decode and put in smem ring buffer                         │
│      signal(inst_arrived) via semaphore                         │
└─────────────────────────────────────────────────────────────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     ▼                       ▼                       ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────┐
│Loader (W0)  │    │Consumer (W0-11) │    │Storer (W1)  │
│             │    │                 │    │             │
│wait(inst)   │    │wait(sem_loaded) │    │wait(compute)│
│wait(g_bar)  │    │compute on page  │    │store→HBM    │
│HBM→SMEM     │    │signal(computed) │    │atomic(g_bar)│
│signal(load) │    │                 │    │signal(free) │
└─────────────┘    └─────────────────┘    └─────────────┘
       │                                         │
       └──────── Launcher (W2): TMA prefetch ────┘

Synchronization Mechanisms:
1. Global Scoreboard (g_bar): Bar[op_id][chunk_id] for cross-block dependencies
   - Producer: atomicAdd(&Bar[op][chunk], 1) after store completes
   - Consumer: spin-wait until Bar[dep_op][chunk] >= target

2. Paged SMEM Semaphores: Per-page synchronization
   - sem_loaded[page]: Loader → Consumer handoff
   - sem_consumed[page]: Consumer → Loader (page recycling)

3. Instruction Ring Buffer: Controller → All warps coordination
   - SMEM ring buffer with instruction_arrived semaphore
   - Allows instruction pipelining (fetch N+1 while executing N)
"""

import torch
import builtins
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from typing import List, Dict, Tuple, Any, Literal
import inspect
import textwrap
import tempfile
import importlib.util
import os

from .scheduler import WarpRole, WarpConfig, PageConfig, BarrierConfig, Instruction, InstructionScheduler
from .utils import atomic_add_i32, atomic_load_i32, atomic_store_i32, nanosleep

# Counter for unique wrapper module names
_wrapper_counter = 0


def _has_tensor_defs(op):
    """Check if an op uses the TensorDef API."""
    return hasattr(op, "tensor_defs") and op.tensor_defs


def _build_tensor_metadata(op, args):
    """Build metadata from TensorDef definitions and actual tensor arguments."""
    metadata = []
    for i, tdef in enumerate(op.tensor_defs):
        if tdef.is_scalar:
            metadata.append({"name": tdef.name, "is_scalar": True, "slot": i})
        else:
            tensor = args[i]
            shape_vals = tuple(tensor.shape[idx] for idx in tdef.shape)
            stride_vals = tuple(tensor.stride()[idx] for idx in tdef.stride)
            metadata.append({
                "name": tdef.name,
                "is_scalar": False,
                "dtype": tdef.dtype,
                "shape_values": shape_vals,
                "stride_values": stride_vals,
                "slot": i,
            })
    return metadata


def _generate_tensordef_wrapper(op, metadata, slot_mapping=None):
    """Generate a JIT-compatible wrapper class for TensorDef-based kernels.

    This creates a wrapper that:
    1. Accepts the standard 32-slot signature (logical_idx, smem, t0..t31)
    2. Converts Int64 pointers to CuTe tensors based on metadata
    3. Calls the user's simple compute_forward with proper tensor arguments
    """
    global _wrapper_counter
    _wrapper_counter += 1

    # Extract the compute_forward body
    src = textwrap.dedent(inspect.getsource(op.compute_forward))
    lines = src.split("\n")
    def_idx = 0
    for idx, line in enumerate(lines):
        if line.strip().startswith("def "):
            def_idx = idx
            break
    body = textwrap.dedent("\n".join(lines[def_idx + 1:]))
    compute_body = textwrap.indent(body, "        ")

    # Generate tensor creation code
    tensor_creations = []
    for meta in metadata:
        name = meta["name"]
        gs = slot_mapping[meta["slot"]] if slot_mapping else meta["slot"]
        if meta.get("is_scalar"):
            tensor_creations.append(f"        {name} = t{gs}")
        else:
            dt = meta["dtype"].__name__ if hasattr(meta["dtype"], "__name__") else str(meta["dtype"])
            tensor_creations.append(f"        {name}_ptr = cute.make_ptr(cute.{dt}, t{gs}, cute.AddressSpace.gmem)")
            tensor_creations.append(
                f"        {name} = cute.make_tensor({name}_ptr, cute.make_layout(({meta['shape_values']}, {meta['stride_values']})))"
            )

    t_creations_str = "\n".join(tensor_creations)
    sig = ", ".join([f"t{i}" for i in range(32)])

    wrapper_code = f'''
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr

class TensorDefWrapper:
    def __init__(wself, inner_op):
        wself._inner = inner_op

    def __extract_mlir_values__(wself):
        return []

    def __new_from_mlir_values__(wself, values):
        return wself

    def __c_pointers__(wself):
        return []

    @property
    def smem_size_fwd(wself):
        return wself._inner.smem_size_fwd

    @property
    def smem_size_bwd(wself):
        return wself._inner.smem_size_bwd

    def get_logical_grid_size(wself, *args):
        return wself._inner.get_logical_grid_size(*args)

    @cute.jit
    def load_forward(wself, logical_idx, smem, {sig}):
        pass

    @cute.jit
    def compute_forward(wself, logical_idx, smem, {sig}):
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = const_expr(256)
        idx = logical_idx * num_threads + tidx
{t_creations_str}
{compute_body}

    @cute.jit
    def store_forward(wself, logical_idx, smem, {sig}):
        pass
'''

    # Write to temp file and import
    temp_path = os.path.join(tempfile.gettempdir(), f"machete_wrapper_{_wrapper_counter}.py")
    with open(temp_path, "w") as f:
        f.write(wrapper_code)

    spec = importlib.util.spec_from_file_location(f"machete_wrapper_{_wrapper_counter}", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.TensorDefWrapper(op)


class Megakernel:
    """Persistent megakernel with selectable execution strategy.

    Strategies:
    - "warp_specialized": "GPU OS" model. Pipelined warps (Loader, Consumer, Storer).
      Best for large independent tiles where pipelining hides latency.
    - "sequential": Traditional loop. Load → Sync → Compute → Sync → Store.
      Best for small tiles or when warp specialization overhead is too high.
      In this mode, ALL warps participate in each stage (Load/Compute/Store).

    Example:
        mk = Megakernel(name="my_kernel", strategy="warp_specialized")
        mk.add(rope_op, q, cos, sin, seq_len, n_tokens)
        mk.launch(n_blocks, grid, block)
    """

    def __init__(
        self,
        name: str = "megakernel",
        mode: str = "forward",
        strategy: Literal["warp_specialized", "sequential"] = "warp_specialized",
        warp_config: WarpConfig = None,
        page_config: PageConfig = None,
    ):
        self.name = name
        self.mode = mode  # "forward" or "backward"
        self.strategy = strategy
        self.warp_config = warp_config or WarpConfig()
        self.page_config = page_config or PageConfig()
        self.instructions: List[Dict] = []
        self._barrier_tensor = None
        self._instruction_tensor = None
        self._num_pages = None
        # Semaphore tensors for page synchronization
        self._sem_loaded_tensor = None
        self._sem_consumed_tensor = None

        # Ring buffer config
        self.max_instructions_in_ring = 8
        self.instruction_size_ints = 16  # Estimated max size of encoded instruction

        # Cached runner for performance (avoid recreating class on every launch)
        self._cached_runner = None
        self._cached_runner_key = None  # (op_ids, is_backward) to detect when to invalidate

    def add(self, op, *args) -> None:
        """Add operation to megakernel."""
        # Check if TensorDef-based kernel - wrapper will be generated later with correct slot mapping
        tensor_def_metadata = None
        if _has_tensor_defs(op):
            tensor_def_metadata = _build_tensor_metadata(op, args)
            # Don't wrap yet - will be done in launch() after slot mapping is determined

        smem = op.smem_size_fwd if self.mode == "forward" else op.smem_size_bwd

        # Extract dependencies from decorators
        reads_list = []
        writes_list = []
        if self.mode == "forward":
            if hasattr(op, "load_forward") and hasattr(op.load_forward, "_machete_deps"):
                for dep in op.load_forward._machete_deps:
                    if dep.is_read:
                        reads_list.append(dep.name)
            if hasattr(op, "store_forward") and hasattr(op.store_forward, "_machete_deps"):
                for dep in op.store_forward._machete_deps:
                    if not dep.is_read:
                        writes_list.append(dep.name)
        else:
            if hasattr(op, "load_backward") and hasattr(op.load_backward, "_machete_deps"):
                for dep in op.load_backward._machete_deps:
                    if dep.is_read:
                        reads_list.append(dep.name)
            if hasattr(op, "store_backward") and hasattr(op.store_backward, "_machete_deps"):
                for dep in op.store_backward._machete_deps:
                    if not dep.is_read:
                        writes_list.append(dep.name)

        # Get execution mode from kernel if available
        execution_mode = getattr(op, "execution_mode", None)

        self.instructions.append(
            {
                "op": op,  # Will be wrapped later if TensorDef-based
                "args": list(args),
                "smem_size": smem,
                "reads": reads_list,
                "writes": writes_list,
                "tensor_def_metadata": tensor_def_metadata,
                "execution_mode": execution_mode,
            }
        )

    def launch(self, n_blocks: int, grid: Tuple, block: Tuple) -> None:
        """Launch persistent megakernel."""
        if not self.instructions:
            return

        # Compute total logical work
        total_work = 0
        for inst in self.instructions:
            op = inst["op"]
            logical_size = op.get_logical_grid_size(*inst["args"])
            total_work += logical_size

        # Setup barrier config
        num_ops = len(self.instructions)
        # Use a safe heuristic for max_chunks if args are dynamic
        max_chunks = max(inst["op"].get_logical_grid_size(*inst["args"]) for inst in self.instructions)
        # Barrier: [NumOps, MaxChunks]
        barrier_config = BarrierConfig(num_ops=num_ops, num_chunks=max_chunks)

        # Allocate global scoreboard
        device = self._get_device()
        self._barrier_tensor = torch.zeros(barrier_config.total_size, dtype=torch.int32, device=device)

        # Compute smem size and num_pages
        smem_size = self._compute_smem_size()

        # Allocate page semaphores (global memory for simplicity in prototype)
        # sem_loaded: Signaled by Loader (or All in seq) -> Waited by Consumer
        # sem_consumed: Signaled by Storer (or All in seq) -> Waited by Loader
        self._sem_loaded_tensor = torch.zeros(self._num_pages, dtype=torch.int32, device=device)
        self._sem_consumed_tensor = torch.ones(self._num_pages, dtype=torch.int32, device=device)

        # Build instruction stream
        scheduler = InstructionScheduler(barrier_config, self.page_config, self._num_pages)
        for op_id, inst in enumerate(self.instructions):
            op = inst["op"]
            # Pass raw args for slot mapping in scheduler
            op_info = {
                "op_id": op_id,
                "op": op,
                "logical_grid_size": op.get_logical_grid_size(*inst["args"]),
                "reads": inst["reads"],
                "writes": inst["writes"],
                "raw_args": inst["args"],
            }
            scheduler.add_op(op_id, op, op_info["logical_grid_size"], inst["reads"], inst["writes"])
            scheduler.ops[-1]["raw_args"] = inst["args"]

        # NOTE: Scheduler handles dependencies, page allocation and slot mapping
        scheduled_instructions = scheduler.schedule()
        unique_tensors = scheduler.unique_tensors
        if len(unique_tensors) > 32:
            raise ValueError(f"Too many unique tensors in megakernel fusion ({len(unique_tensors)} > 32)")

        # Now wrap TensorDef ops with correct slot mapping from scheduler
        wrapped_ops = []
        for op_id, inst in enumerate(self.instructions):
            op = inst["op"]
            if inst.get("tensor_def_metadata"):
                # Build slot mapping: local_slot -> global_slot
                slot_mapping = {}
                for local_slot, arg in enumerate(inst["args"]):
                    global_slot = scheduler.tensor_to_slot.get(id(arg), local_slot)
                    slot_mapping[local_slot] = global_slot
                wrapped_op = _generate_tensordef_wrapper(op, inst["tensor_def_metadata"], slot_mapping)
                wrapped_ops.append(wrapped_op)
            else:
                wrapped_ops.append(op)

        # Update instructions with wrapped ops
        for i, wrapped_op in enumerate(wrapped_ops):
            self.instructions[i]["op"] = wrapped_op

        # Pad unique_tensors to 32 for fixed signature
        # Use first tensor as padding if available, else a dummy
        padding_tensor = unique_tensors[0] if unique_tensors else torch.zeros(1, device=device)
        padded_tensors = unique_tensors + [padding_tensor] * (32 - len(unique_tensors))

        # Encode instructions
        encoded = scheduler.get_encoded_instructions()
        if encoded:
            self._instruction_tensor = torch.tensor(encoded, dtype=torch.int32, device=device)
        else:
            self._instruction_tensor = torch.zeros(1, dtype=torch.int32, device=device)

        num_instructions = len(scheduled_instructions)

        # Launch appropriate kernel strategy
        # Fall back to sequential if block size doesn't support warp specialization
        num_warps_in_block = block[0] // 32
        can_use_warp_spec = num_warps_in_block >= self.warp_config.total_warps

        if self.strategy == "warp_specialized" and can_use_warp_spec:
            self._launch_warp_specialized(grid, block, smem_size, num_instructions, padded_tensors)
        else:
            self._launch_sequential(grid, block, smem_size, num_instructions, padded_tensors)

    def _get_device(self):
        for inst in self.instructions:
            for arg in inst["args"]:
                if isinstance(arg, torch.Tensor):
                    return arg.device
        return torch.device("cuda")

    def _get_device_smem_size(self) -> int:
        device = self._get_device()
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            return props.shared_memory_per_block_optin
        return 48 * 1024

    def _compute_smem_size(self) -> int:
        device_smem = self._get_device_smem_size()
        self._num_pages = self.page_config.get_num_pages(device_smem)
        op_smem = max(inst["smem_size"] for inst in self.instructions) if self.instructions else 0
        pages_size = self.page_config.get_total_smem_for_pages(self._num_pages)
        ring_size = self.max_instructions_in_ring * self.instruction_size_ints * 4
        return max(op_smem, pages_size) + ring_size + 1024

    def _launch_warp_specialized(self, grid, block, smem_size, num_instructions, padded_tensors):
        """Launch 'GPU OS' Kernel: Warp Specialized, Pipelined Execution.

        This implements pipelined warp specialization where each role runs
        its own independent grid-stride loop, allowing overlap:

        - Loader Warp: Waits for page free (sem_consumed), loads HBM→SMEM, signals sem_loaded
        - Consumer Warps: Wait for sem_loaded, compute on page, signal sem_computed
        - Storer Warp: Waits for sem_computed, stores SMEM→HBM, signals sem_consumed (page free)

        Semaphore Protocol (per page, per block):
        - sem_loaded[block][page]: incremented by Loader after load completes
        - sem_computed[block][page]: incremented by Consumer (lane 0) after compute completes
        - sem_consumed[block][page]: incremented by Storer after store completes

        Each role tracks its own instruction index, enabling pipelining:
        - Loader can be loading instruction N+1 while Consumer processes N
        - Storer can be storing instruction N-1 while Consumer processes N

        Memory Layout for Semaphores (global memory, per-block arrays):
        - g_sem_loaded: [num_blocks * num_pages] int32
        - g_sem_computed: [num_blocks * num_pages] int32  (reuses sem_consumed tensor for now)
        - g_sem_consumed: [num_blocks * num_pages] int32
        """
        ops = [inst["op"] for inst in self.instructions]
        num_ops_py = len(ops)
        num_pages_py = self._num_pages
        page_size_py = self.page_config.page_size

        # Warp config values to pass as kernel args
        num_consumer_py = self.warp_config.num_consumer_warps
        loader_warp_py = self.warp_config.loader_warp_start
        storer_warp_py = self.warp_config.storer_warp_start
        launcher_warp_py = self.warp_config.launcher_warp_start
        controller_warp_py = self.warp_config.controller_warp_start

        # For pipelining, we need separate sem_computed array
        device = self._get_device()
        num_blocks = grid[0]
        # Allocate per-block semaphore arrays: [num_blocks, num_pages]
        # sem_loaded: starts at 0, Loader increments after load
        # sem_computed: starts at 0, Consumer increments after compute
        # sem_consumed: starts at 1 (page initially free), Storer increments after store
        self._sem_loaded_tensor = torch.zeros(num_blocks * num_pages_py, dtype=torch.int32, device=device)
        sem_computed_tensor = torch.zeros(num_blocks * num_pages_py, dtype=torch.int32, device=device)
        self._sem_consumed_tensor = torch.ones(num_blocks * num_pages_py, dtype=torch.int32, device=device)

        # Create a runner class with the kernel as a method
        class MegakernelRunner:
            def __init__(rself, g, b, smem, all_ops, n_pages, pg_size, n_cons, ld_w, st_w, ln_w, ct_w):
                rself._grid = list(g)
                rself._block = list(b)
                rself._smem = smem
                rself._ops = all_ops
                rself._num_pages = n_pages
                rself._page_size = pg_size
                rself._num_consumer = n_cons
                rself._loader_warp = ld_w
                rself._storer_warp = st_w
                rself._launcher_warp = ln_w
                rself._controller_warp = ct_w

            def __extract_mlir_values__(rself):
                return []

            def __new_from_mlir_values__(rself, values):
                return rself

            def __c_pointers__(rself):
                return []

            @cute.jit
            def run(
                rself,
                p_bar: Int64, p_instr: Int64, p_sem_loaded: Int64, p_sem_computed: Int64, p_sem_consumed: Int64,
                p_n_instr: Int32, p_n_ops: Int32,
                p_n_pages: Int32, p_pg_size: Int32,
                p_n_cons: Int32, p_ld_w: Int32, p_st_w: Int32, p_ln_w: Int32, p_ct_w: Int32,
                p0: Int64, p1: Int64, p2: Int64, p3: Int64,
                p4: Int64, p5: Int64, p6: Int64, p7: Int64,
                p8: Int64, p9: Int64, p10: Int64, p11: Int64,
                p12: Int64, p13: Int64, p14: Int64, p15: Int64,
                p16: Int64, p17: Int64, p18: Int64, p19: Int64,
                p20: Int64, p21: Int64, p22: Int64, p23: Int64,
                p24: Int64, p25: Int64, p26: Int64, p27: Int64,
                p28: Int64, p29: Int64, p30: Int64, p31: Int64,
            ):
                rself.kernel(
                    p_bar, p_instr, p_sem_loaded, p_sem_computed, p_sem_consumed,
                    p_n_instr, p_n_ops, p_n_pages, p_pg_size,
                    p_n_cons, p_ld_w, p_st_w, p_ln_w, p_ct_w,
                    p0, p1, p2, p3, p4, p5, p6, p7,
                    p8, p9, p10, p11, p12, p13, p14, p15,
                    p16, p17, p18, p19, p20, p21, p22, p23,
                    p24, p25, p26, p27, p28, p29, p30, p31,
                ).launch(grid=rself._grid, block=rself._block, smem=rself._smem)

            @cute.kernel
            def kernel(
                rself,
                g_bar: Int64,
                g_instructions: Int64,
                g_sem_loaded: Int64,
                g_sem_computed: Int64,
                g_sem_consumed: Int64,
                n_instructions: Int32,
                n_ops: Int32,
                num_pages: Int32,
                page_size: Int32,
                num_consumer: Int32,
                loader_warp: Int32,
                storer_warp: Int32,
                launcher_warp: Int32,
                controller_warp: Int32,
                t0: Int64, t1: Int64, t2: Int64, t3: Int64,
                t4: Int64, t5: Int64, t6: Int64, t7: Int64,
                t8: Int64, t9: Int64, t10: Int64, t11: Int64,
                t12: Int64, t13: Int64, t14: Int64, t15: Int64,
                t16: Int64, t17: Int64, t18: Int64, t19: Int64,
                t20: Int64, t21: Int64, t22: Int64, t23: Int64,
                t24: Int64, t25: Int64, t26: Int64, t27: Int64,
                t28: Int64, t29: Int64, t30: Int64, t31: Int64,
            ):
                tidx, _, _ = cute.arch.thread_idx()
                bidx, _, _ = cute.arch.block_idx()
                grid_dim, _, _ = cute.arch.grid_dim()
                warp_id = tidx // 32
                lane_id = tidx % 32

                # Warp role identification
                is_consumer = warp_id < num_consumer
                is_loader = warp_id == loader_warp
                is_storer = warp_id == storer_warp

                # Internal SMEM pointer
                smem = cute.runtime.make_ptr(cute.Uint8, 0, cute.AddressSpace.smem)

                # Base offset for this block's semaphores
                sem_base = bidx * num_pages

                # ========== LOADER WARP ==========
                # Independent loop: load instructions as pages become available
                if is_loader:
                    loader_instr = Int32(bidx)  # Start at block's first instruction
                    loader_expected = Int32(1)   # Expected sem_consumed value (starts at 1)

                    while loader_instr < n_instructions:
                        page_id = loader_instr % num_pages
                        page_ptr = smem + page_id * page_size

                        # Wait for page to be free (sem_consumed >= expected)
                        sem_consumed_addr = g_sem_consumed + (sem_base + page_id) * 4
                        if lane_id == 0:
                            val = atomic_load_i32(sem_consumed_addr)
                            while val < loader_expected:
                                nanosleep(100)
                                val = atomic_load_i32(sem_consumed_addr)

                        # Sync within warp before loading
                        cute.arch.sync_warp()

                        # Execute load for all fused ops
                        logical_idx = loader_instr
                        for op in rself._ops:
                            op.load_forward(
                                logical_idx, page_ptr,
                                t0, t1, t2, t3, t4, t5, t6, t7,
                                t8, t9, t10, t11, t12, t13, t14, t15,
                                t16, t17, t18, t19, t20, t21, t22, t23,
                                t24, t25, t26, t27, t28, t29, t30, t31,
                            )

                        # Signal load complete
                        cute.arch.sync_warp()
                        if lane_id == 0:
                            sem_loaded_addr = g_sem_loaded + (sem_base + page_id) * 4
                            atomic_add_i32(Int32(1), sem_loaded_addr)

                        # Next instruction for this block
                        loader_instr += grid_dim
                        # Update expected consumed count for next use of this page
                        if loader_instr < n_instructions:
                            next_page = loader_instr % num_pages
                            if next_page == page_id:
                                loader_expected += Int32(1)

                # ========== CONSUMER WARPS ==========
                # Independent loop: compute on pages as they become loaded
                # Each consumer warp signals completion; storer waits for all
                if is_consumer:
                    consumer_instr = Int32(bidx)
                    consumer_expected = Int32(1)  # Expected sem_loaded value

                    while consumer_instr < n_instructions:
                        page_id = consumer_instr % num_pages
                        page_ptr = smem + page_id * page_size

                        # Wait for load complete (sem_loaded >= expected)
                        sem_loaded_addr = g_sem_loaded + (sem_base + page_id) * 4
                        if lane_id == 0:
                            val = atomic_load_i32(sem_loaded_addr)
                            while val < consumer_expected:
                                nanosleep(100)
                                val = atomic_load_i32(sem_loaded_addr)

                        # Sync within consumer warps before compute
                        cute.arch.sync_warp()

                        # Execute compute for all fused ops
                        logical_idx = consumer_instr
                        for op in rself._ops:
                            op.compute_forward(
                                logical_idx, page_ptr,
                                t0, t1, t2, t3, t4, t5, t6, t7,
                                t8, t9, t10, t11, t12, t13, t14, t15,
                                t16, t17, t18, t19, t20, t21, t22, t23,
                                t24, t25, t26, t27, t28, t29, t30, t31,
                            )

                        # Each consumer warp signals completion (lane 0)
                        # Storer will wait for num_consumer signals
                        cute.arch.sync_warp()
                        if lane_id == 0:
                            sem_computed_addr = g_sem_computed + (sem_base + page_id) * 4
                            atomic_add_i32(Int32(1), sem_computed_addr)

                        # Next instruction
                        consumer_instr += grid_dim
                        if consumer_instr < n_instructions:
                            next_page = consumer_instr % num_pages
                            if next_page == page_id:
                                consumer_expected += Int32(1)

                # ========== STORER WARP ==========
                # Independent loop: store results as compute completes
                # Wait for ALL consumer warps to finish (num_consumer signals per instruction)
                if is_storer:
                    storer_instr = Int32(bidx)
                    # Expected = instruction_count * num_consumer_warps
                    # For first instruction: num_consumer, then 2*num_consumer, etc.
                    storer_expected = num_consumer  # Start expecting num_consumer signals

                    while storer_instr < n_instructions:
                        page_id = storer_instr % num_pages
                        page_ptr = smem + page_id * page_size

                        # Wait for ALL consumer warps to complete
                        sem_computed_addr = g_sem_computed + (sem_base + page_id) * 4
                        if lane_id == 0:
                            val = atomic_load_i32(sem_computed_addr)
                            while val < storer_expected:
                                nanosleep(100)
                                val = atomic_load_i32(sem_computed_addr)

                        # Sync within warp before storing
                        cute.arch.sync_warp()

                        # Execute store for all fused ops
                        logical_idx = storer_instr
                        for op in rself._ops:
                            op.store_forward(
                                logical_idx, page_ptr,
                                t0, t1, t2, t3, t4, t5, t6, t7,
                                t8, t9, t10, t11, t12, t13, t14, t15,
                                t16, t17, t18, t19, t20, t21, t22, t23,
                                t24, t25, t26, t27, t28, t29, t30, t31,
                            )

                        # Release page (signal consumed)
                        cute.arch.sync_warp()
                        if lane_id == 0:
                            sem_consumed_addr = g_sem_consumed + (sem_base + page_id) * 4
                            atomic_add_i32(Int32(1), sem_consumed_addr)

                        # Next instruction
                        storer_instr += grid_dim
                        if storer_instr < n_instructions:
                            next_page = storer_instr % num_pages
                            if next_page == page_id:
                                # Next time we use this page, expect num_consumer more signals
                                storer_expected += num_consumer

        # Create runner with all ops
        runner = MegakernelRunner(
            grid, block, smem_size, ops,
            num_pages_py, page_size_py,
            num_consumer_py, loader_warp_py, storer_warp_py, launcher_warp_py, controller_warp_py
        )

        # Launch with the additional sem_computed tensor
        self._launch_runner_with_instance_warp_pipelined(
            runner, num_instructions, num_ops_py,
            num_pages_py, page_size_py,
            num_consumer_py, loader_warp_py, storer_warp_py, launcher_warp_py, controller_warp_py,
            sem_computed_tensor, padded_tensors
        )

    def _launch_sequential(self, grid, block, smem_size, num_instructions, padded_tensors):
        """Launch Sequential Kernel: Vertical Fusion, All Warps Participate."""
        ops = [inst["op"] for inst in self.instructions]
        num_ops_py = len(ops)
        num_pages_py = self._num_pages
        page_size_py = self.page_config.page_size
        is_backward = self.mode == "backward"

        # Pre-select methods at Python time (before JIT compilation)
        # This avoids conditional branching in the kernel that references non-existent methods
        op_methods = []
        for op in ops:
            if is_backward and hasattr(op, 'load_backward'):
                load_m = op.load_backward
            else:
                load_m = op.load_forward
            if is_backward and hasattr(op, 'compute_backward'):
                compute_m = op.compute_backward
            else:
                compute_m = op.compute_forward
            if is_backward and hasattr(op, 'store_backward'):
                store_m = op.store_backward
            else:
                store_m = op.store_forward
            op_methods.append((load_m, compute_m, store_m))

        # Cache key: (tuple of op ids, is_backward, shapes tuple) - invalidate if ops or shapes change
        # Include shapes in key to ensure kernel is regenerated when shapes change
        shape_keys = []
        for op in ops:
            if hasattr(op, 'shapes') and op.shapes:
                # Create a hashable tuple from shapes dict
                shape_keys.append(tuple(sorted((k, v if not isinstance(v, (list, tuple)) else tuple(v))
                                               for k, v in op.shapes.items())))
            else:
                shape_keys.append(None)
        cache_key = (tuple(id(op) for op in ops), is_backward, tuple(shape_keys))

        # Reuse cached runner if available and valid
        if self._cached_runner is not None and self._cached_runner_key == cache_key:
            runner = self._cached_runner
            # Update grid/block/smem for this launch
            runner._grid = list(grid)
            runner._block = list(block)
            runner._smem = smem_size
        else:
            # Create a runner class with the kernel as a method
            # Store all ops for multi-op fusion support
            class MegakernelRunner:
                def __init__(runner_self, g, b, smem, all_ops, n_pages, pg_size, methods):
                    runner_self._grid = list(g)
                    runner_self._block = list(b)
                    runner_self._smem = smem
                    runner_self._ops = all_ops  # List of all ops for multi-op fusion
                    runner_self._num_pages = n_pages
                    runner_self._page_size = pg_size
                    runner_self._op_methods = methods  # Pre-selected L/C/S methods

                def __extract_mlir_values__(runner_self):
                    return []

                def __new_from_mlir_values__(runner_self, values):
                    return runner_self

                def __c_pointers__(runner_self):
                    return []

                @cute.jit
                def run(
                    runner_self,
                    p_bar: Int64, p_instr: Int64, p_sem_loaded: Int64, p_sem_consumed: Int64,
                    p_n_instr: Int32, p_n_ops: Int32, p_n_pages: Int32, p_pg_size: Int32,
                    p0: Int64, p1: Int64, p2: Int64, p3: Int64,
                    p4: Int64, p5: Int64, p6: Int64, p7: Int64,
                    p8: Int64, p9: Int64, p10: Int64, p11: Int64,
                    p12: Int64, p13: Int64, p14: Int64, p15: Int64,
                    p16: Int64, p17: Int64, p18: Int64, p19: Int64,
                    p20: Int64, p21: Int64, p22: Int64, p23: Int64,
                    p24: Int64, p25: Int64, p26: Int64, p27: Int64,
                    p28: Int64, p29: Int64, p30: Int64, p31: Int64,
                ):
                    runner_self.kernel(
                        p_bar, p_instr, p_sem_loaded, p_sem_consumed,
                        p_n_instr, p_n_ops, p_n_pages, p_pg_size,
                        p0, p1, p2, p3, p4, p5, p6, p7,
                        p8, p9, p10, p11, p12, p13, p14, p15,
                        p16, p17, p18, p19, p20, p21, p22, p23,
                        p24, p25, p26, p27, p28, p29, p30, p31,
                    ).launch(grid=runner_self._grid, block=runner_self._block, smem=runner_self._smem)

                @cute.kernel
                def kernel(
                    runner_self,
                    g_bar: Int64,
                    g_instructions: Int64,
                    g_sem_loaded: Int64,
                    g_sem_consumed: Int64,
                    n_instructions: Int32,
                    n_ops: Int32,
                    num_pages: Int32,
                    page_size: Int32,
                    t0: Int64, t1: Int64, t2: Int64, t3: Int64,
                    t4: Int64, t5: Int64, t6: Int64, t7: Int64,
                    t8: Int64, t9: Int64, t10: Int64, t11: Int64,
                    t12: Int64, t13: Int64, t14: Int64, t15: Int64,
                    t16: Int64, t17: Int64, t18: Int64, t19: Int64,
                    t20: Int64, t21: Int64, t22: Int64, t23: Int64,
                    t24: Int64, t25: Int64, t26: Int64, t27: Int64,
                    t28: Int64, t29: Int64, t30: Int64, t31: Int64,
                ):
                    tidx, _, _ = cute.arch.thread_idx()
                    bidx, _, _ = cute.arch.block_idx()
                    grid_dim, _, _ = cute.arch.grid_dim()

                    # Internal SMEM pointer
                    smem = cute.runtime.make_ptr(cute.Uint8, 0, cute.AddressSpace.smem)

                    instr_id = Int32(bidx)
                    while instr_id < n_instructions:
                        logical_idx = instr_id

                        page_id = instr_id % num_pages
                        page_ptr = smem + page_id * page_size

                        # Execute all ops in sequence (Python loop unrolled at JIT time)
                        # Methods are pre-selected at Python time (forward or backward)
                        for load_m, compute_m, store_m in runner_self._op_methods:
                            cute.arch.sync_threads()
                            load_m(
                                logical_idx,
                                page_ptr,
                                t0, t1, t2, t3, t4, t5, t6, t7,
                                t8, t9, t10, t11, t12, t13, t14, t15,
                                t16, t17, t18, t19, t20, t21, t22, t23,
                                t24, t25, t26, t27, t28, t29, t30, t31,
                            )
                            cute.arch.sync_threads()
                            compute_m(
                                logical_idx,
                                page_ptr,
                                t0, t1, t2, t3, t4, t5, t6, t7,
                                t8, t9, t10, t11, t12, t13, t14, t15,
                                t16, t17, t18, t19, t20, t21, t22, t23,
                                t24, t25, t26, t27, t28, t29, t30, t31,
                            )
                            cute.arch.sync_threads()
                            store_m(
                                logical_idx,
                                page_ptr,
                                t0, t1, t2, t3, t4, t5, t6, t7,
                                t8, t9, t10, t11, t12, t13, t14, t15,
                                t16, t17, t18, t19, t20, t21, t22, t23,
                                t24, t25, t26, t27, t28, t29, t30, t31,
                            )
                            cute.arch.sync_threads()

                        instr_id += grid_dim

            # Create runner with all ops for multi-op fusion
            runner = MegakernelRunner(grid, block, smem_size, ops, num_pages_py, page_size_py, op_methods)
            # Cache for reuse
            self._cached_runner = runner
            self._cached_runner_key = cache_key

        self._launch_runner_with_instance_seq(
            runner, num_instructions, num_ops_py, num_pages_py, page_size_py, padded_tensors
        )

    def _launch_runner(self, runner_cls, grid, block, smem_size, num_intr, num_ops, padded_tensors):
        """Launch using the @cute.jit runner pattern.

        This creates a runner instance and calls its run() method, which properly
        compiles and executes the kernel.
        """
        # Convert tensors to Int64 data pointers for the kernel
        int_args = []
        for t in padded_tensors:
            if isinstance(t, torch.Tensor):
                int_args.append(Int64(t.data_ptr()))
            elif isinstance(t, (int, float)):
                int_args.append(Int64(int(t)))
            elif hasattr(t, "ir_value"):  # DSL Type
                int_args.append(t)
            else:
                int_args.append(Int64(0))

        # Fixed args for the kernel
        g_bar = Int64(self._barrier_tensor.data_ptr())
        g_instr = Int64(self._instruction_tensor.data_ptr())
        g_sem_loaded = Int64(self._sem_loaded_tensor.data_ptr())
        g_sem_consumed = Int64(self._sem_consumed_tensor.data_ptr())
        n_instr = Int32(num_intr)
        n_ops = Int32(num_ops)

        # Grid/block as lists for launch
        grid_list = list(grid) if isinstance(grid, tuple) else list(grid)
        block_list = list(block) if isinstance(block, tuple) else list(block)

        # Create runner and execute
        runner = runner_cls(None, grid_list, block_list, smem_size)
        runner.run(
            g_bar, g_instr, g_sem_loaded, g_sem_consumed,
            n_instr, n_ops,
            *int_args,
        )

    def _launch_runner_direct(self, runner_cls, grid, block, smem_size, num_intr, num_ops, padded_tensors):
        """Launch using the @cute.jit runner pattern with kernel as class method.

        Similar to _launch_runner but for runners where kernel is defined inline.
        """
        # Convert tensors to Int64 data pointers for the kernel
        int_args = []
        for t in padded_tensors:
            if isinstance(t, torch.Tensor):
                int_args.append(Int64(t.data_ptr()))
            elif isinstance(t, (int, float)):
                int_args.append(Int64(int(t)))
            elif hasattr(t, "ir_value"):  # DSL Type
                int_args.append(t)
            else:
                int_args.append(Int64(0))

        # Fixed args for the kernel
        g_bar = Int64(self._barrier_tensor.data_ptr())
        g_instr = Int64(self._instruction_tensor.data_ptr())
        g_sem_loaded = Int64(self._sem_loaded_tensor.data_ptr())
        g_sem_consumed = Int64(self._sem_consumed_tensor.data_ptr())
        n_instr = Int32(num_intr)
        n_ops = Int32(num_ops)

        # Grid/block as lists for launch
        grid_list = list(grid) if isinstance(grid, tuple) else list(grid)
        block_list = list(block) if isinstance(block, tuple) else list(block)

        # Create runner and execute
        runner = runner_cls(grid_list, block_list, smem_size)
        runner.run(
            g_bar, g_instr, g_sem_loaded, g_sem_consumed,
            n_instr, n_ops,
            *int_args,
        )

    def _launch_runner_with_instance(self, runner, num_intr, num_ops, padded_tensors):
        """Launch using a pre-created runner instance.

        Used when the runner needs to capture closures via its constructor.
        """
        # Convert tensors to Int64 data pointers for the kernel
        int_args = []
        for t in padded_tensors:
            if isinstance(t, torch.Tensor):
                int_args.append(Int64(t.data_ptr()))
            elif isinstance(t, (int, float)):
                int_args.append(Int64(int(t)))
            elif hasattr(t, "ir_value"):  # DSL Type
                int_args.append(t)
            else:
                int_args.append(Int64(0))

        # Fixed args for the kernel
        g_bar = Int64(self._barrier_tensor.data_ptr())
        g_instr = Int64(self._instruction_tensor.data_ptr())
        g_sem_loaded = Int64(self._sem_loaded_tensor.data_ptr())
        g_sem_consumed = Int64(self._sem_consumed_tensor.data_ptr())
        n_instr = Int32(num_intr)
        n_ops = Int32(num_ops)

        # Execute
        runner.run(
            g_bar, g_instr, g_sem_loaded, g_sem_consumed,
            n_instr, n_ops,
            *int_args,
        )

    def _launch_runner_with_instance_seq(
        self, runner, num_intr, num_ops, num_pages, page_size, padded_tensors
    ):
        """Launch using a pre-created runner instance for sequential kernel.

        Adds num_pages and page_size as explicit kernel arguments.
        """
        # Convert tensors to Int64 data pointers for the kernel
        # All tensor slots use Int64 to maintain a uniform signature
        int_args = []
        for t in padded_tensors:
            if isinstance(t, torch.Tensor):
                int_args.append(Int64(t.data_ptr()))
            elif isinstance(t, (int, float)):
                int_args.append(Int64(int(t)))
            elif isinstance(t, Int32):
                # Convert Int32 scalars to Int64 to match kernel signature
                int_args.append(Int64(t.value))
            elif isinstance(t, Int64):
                int_args.append(t)
            elif hasattr(t, "ir_value") and hasattr(t, "value"):
                # Other DSL types with value attribute
                int_args.append(Int64(t.value))
            else:
                int_args.append(Int64(0))

        # Fixed args for the kernel
        g_bar = Int64(self._barrier_tensor.data_ptr())
        g_instr = Int64(self._instruction_tensor.data_ptr())
        g_sem_loaded = Int64(self._sem_loaded_tensor.data_ptr())
        g_sem_consumed = Int64(self._sem_consumed_tensor.data_ptr())
        n_instr = Int32(num_intr)
        n_ops = Int32(num_ops)
        n_pages = Int32(num_pages)
        pg_size = Int32(page_size)

        # Execute
        runner.run(
            g_bar, g_instr, g_sem_loaded, g_sem_consumed,
            n_instr, n_ops, n_pages, pg_size,
            *int_args,
        )

    def _launch_runner_with_instance_warp(
        self, runner, num_intr, num_ops, num_pages, page_size,
        num_consumer, loader_warp, storer_warp, launcher_warp, controller_warp,
        padded_tensors
    ):
        """Launch using a pre-created runner instance for warp-specialized kernel.

        Adds all warp config values as explicit kernel arguments.
        """
        # Convert tensors to Int64 data pointers for the kernel
        # All tensor slots use Int64 to maintain a uniform signature
        int_args = []
        for t in padded_tensors:
            if isinstance(t, torch.Tensor):
                int_args.append(Int64(t.data_ptr()))
            elif isinstance(t, (int, float)):
                int_args.append(Int64(int(t)))
            elif isinstance(t, Int32):
                # Convert Int32 scalars to Int64 to match kernel signature
                int_args.append(Int64(t.value))
            elif isinstance(t, Int64):
                int_args.append(t)
            elif hasattr(t, "ir_value") and hasattr(t, "value"):
                # Other DSL types with value attribute
                int_args.append(Int64(t.value))
            else:
                int_args.append(Int64(0))

        # Fixed args for the kernel
        g_bar = Int64(self._barrier_tensor.data_ptr())
        g_instr = Int64(self._instruction_tensor.data_ptr())
        g_sem_loaded = Int64(self._sem_loaded_tensor.data_ptr())
        g_sem_consumed = Int64(self._sem_consumed_tensor.data_ptr())
        n_instr = Int32(num_intr)
        n_ops = Int32(num_ops)
        n_pages = Int32(num_pages)
        pg_size = Int32(page_size)
        n_cons = Int32(num_consumer)
        ld_w = Int32(loader_warp)
        st_w = Int32(storer_warp)
        ln_w = Int32(launcher_warp)
        ct_w = Int32(controller_warp)

        # Execute
        runner.run(
            g_bar, g_instr, g_sem_loaded, g_sem_consumed,
            n_instr, n_ops, n_pages, pg_size,
            n_cons, ld_w, st_w, ln_w, ct_w,
            *int_args,
        )

    def _launch_runner_with_instance_warp_pipelined(
        self, runner, num_intr, num_ops, num_pages, page_size,
        num_consumer, loader_warp, storer_warp, launcher_warp, controller_warp,
        sem_computed_tensor, padded_tensors
    ):
        """Launch using a pre-created runner instance for pipelined warp-specialized kernel.

        This version includes the additional sem_computed tensor for 3-stage pipelining.
        """
        # Convert tensors to Int64 data pointers for the kernel
        int_args = []
        for t in padded_tensors:
            if isinstance(t, torch.Tensor):
                int_args.append(Int64(t.data_ptr()))
            elif isinstance(t, (int, float)):
                int_args.append(Int64(int(t)))
            elif isinstance(t, Int32):
                int_args.append(Int64(t.value))
            elif isinstance(t, Int64):
                int_args.append(t)
            elif hasattr(t, "ir_value") and hasattr(t, "value"):
                int_args.append(Int64(t.value))
            else:
                int_args.append(Int64(0))

        # Fixed args for the kernel
        g_bar = Int64(self._barrier_tensor.data_ptr())
        g_instr = Int64(self._instruction_tensor.data_ptr())
        g_sem_loaded = Int64(self._sem_loaded_tensor.data_ptr())
        g_sem_computed = Int64(sem_computed_tensor.data_ptr())
        g_sem_consumed = Int64(self._sem_consumed_tensor.data_ptr())
        n_instr = Int32(num_intr)
        n_ops = Int32(num_ops)
        n_pages = Int32(num_pages)
        pg_size = Int32(page_size)
        n_cons = Int32(num_consumer)
        ld_w = Int32(loader_warp)
        st_w = Int32(storer_warp)
        ln_w = Int32(launcher_warp)
        ct_w = Int32(controller_warp)

        # Execute with 5 semaphore args instead of 4
        runner.run(
            g_bar, g_instr, g_sem_loaded, g_sem_computed, g_sem_consumed,
            n_instr, n_ops, n_pages, pg_size,
            n_cons, ld_w, st_w, ln_w, ct_w,
            *int_args,
        )
