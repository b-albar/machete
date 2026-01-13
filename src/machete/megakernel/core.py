# Copyright (c) 2025, Machete Authors
import torch
import cutlass.cute as cute
from cutlass import Int32
import os
import importlib.util
import hashlib
from typing import Callable, Union
from .interface import MegakernelOp, FusableOp, FusableKernel
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.compile_utils import make_fake_tensor as fake_tensor
import tvm_ffi.core

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
    def __init__(self, name: str = "megakernel", mode: str = "forward", paged_pool_bytes: int = 0):
        """
        Args:
            name: Kernel name for caching/debugging.
            mode: "forward" or "backward".
            paged_pool_bytes: Size of the paged shared memory pool in bytes.
                              Set to 0 (default) for no pool allocation.
                              For No Bubbles pipelining, set to the total size needed
                              for double-buffering across operations (e.g., 2 * max_op_smem).
        """
        self.name = name
        self.mode = mode
        self.instructions = []
        self.gen_dir = os.path.join(os.path.dirname(__file__), ".generated")
        os.makedirs(self.gen_dir, exist_ok=True)
        self.paged_pool_bytes = paged_pool_bytes

    def add(self, op: Union[FusableKernel, MegakernelOp, Callable], *args):
        """Add an operation and its arguments to the megakernel."""
        if isinstance(op, FusableKernel):
            if self.mode == "forward":
                op = op.compute_forward
            elif self.mode == "backward":
                op = op.compute_backward
            else:
                raise ValueError(f"Unsupported megakernel mode: {self.mode}")

        if isinstance(op, MegakernelOp):
            actual_op = op
        elif hasattr(op, "_machete_is_op"):
            # Determine shared memory requirements
            # Prefer dynamic properties from the instance if available
            smem_per_page = getattr(op, "_machete_smem_per_page", 0)
            num_pages = getattr(op, "_machete_num_pages", 1)

            instance = getattr(op, "__self__", None)
            if instance:
                if hasattr(instance, "smem_per_page"):
                    smem_per_page = instance.smem_per_page
                if hasattr(instance, "num_pages"):
                    num_pages = instance.num_pages

            actual_op = FusableOp(
                compute_func=op,
                num_tensors=op._machete_num_tensors,
                needs_sync=op._machete_needs_sync,
                smem_per_page=smem_per_page,
                num_pages=num_pages,
            )
        smem_dtype = cute.Uint8
        check_op = actual_op
        if isinstance(actual_op, FusableOp):
            check_op = actual_op._compute_func

        # Try to find the object holding the operation state
        obj = getattr(check_op, "__self__", check_op)
        if hasattr(obj, "cute_dtype"):
            smem_dtype = obj.cute_dtype

        self.instructions.append(
            {
                "compute": actual_op.compute,
                "load": getattr(actual_op, "load", None),
                "store": getattr(actual_op, "store", None),
                "args": list(args),
                "needs_sync": actual_op.needs_global_sync,
                "smem_per_page": actual_op.smem_per_page,
                "num_pages": actual_op.num_pages,
                "smem_dtype": smem_dtype,
                "op_obj": actual_op,
            }
        )

    def clear(self):
        self.instructions = []

    def _get_megakernel_class(self, mapping, num_flat_args, op_info_key):
        # We rely on op_info_key and structure of instructions, NOT object identity.
        # This allows different Megakernel instances (and different Op instances)
        # to collide in the cache if they are structurally identical.
        data_to_hash = f"{self.name}_{num_flat_args}_{op_info_key}"
        sig_hash = hashlib.md5(data_to_hash.encode()).hexdigest()

        # Check if already registered?
        # If we just overwrite, we might replace 'instructions' list with a new list
        # which points to new 'compute' method instances.
        # This is actually GOOD, because if we are compiling a new kernel, we want it to bind
        # to the current valid instances (if they are not kept alive elsewhere).
        # However, for global caching, we assume the first-registered instances are representative.

        if sig_hash not in MEGAKERNEL_REGISTRY:
            MEGAKERNEL_REGISTRY[sig_hash] = self.instructions

        gen_filename = f"kernel_{sig_hash}.py"
        gen_path = os.path.join(self.gen_dir, gen_filename)

        arg_names = [f"arg_{i}" for i in range(num_flat_args)]
        all_args_str = ", ".join(arg_names)

        module_bindings = []
        unrolled_ops = []
        sync_needed = any(inst["needs_sync"] for inst in self.instructions)
        if sync_needed:
            pass  # Keep it for future logic if we want to use sync_needed

        unrolled_ops.append("        smem_alloc = cutlass.utils.SmemAllocator()")

        # Allocate paged pool for No Bubbles pipelining if requested
        if self.paged_pool_bytes > 0:
            unrolled_ops.append(f"        # Paged pool for No Bubbles: {self.paged_pool_bytes} bytes")
            unrolled_ops.append(
                f"        paged_pool = smem_alloc.allocate_tensor(cute.Uint8, cute.make_layout({self.paged_pool_bytes}))"
            )

        # L/C/S execution model
        n_ops = len(self.instructions)

        # Bind all ops
        for i in range(n_ops):
            module_bindings.append(f"op_{i} = instructions[{i}]['op_obj']")

        # Prologue: Load first operation (if paged pool is available)
        if self.paged_pool_bytes > 0 and n_ops > 0:
            indices = mapping[0]
            args_str = ", ".join([f"arg_{idx}" for idx in indices])
            unrolled_ops.append("        # Prologue: Load Op 0")
            unrolled_ops.append("        page_idx = Int32(0)")
            unrolled_ops.append(f"        op_0.load(paged_pool, page_idx, {args_str})")
            unrolled_ops.append("        cute.arch.sync_threads()")

        # Main execution loop
        for i, inst in enumerate(self.instructions):
            indices = mapping[i]
            args_str = ", ".join([f"arg_{idx}" for idx in indices])

            # Per-op shared memory allocation (if needed)
            smem_str = ""
            if inst["smem_per_page"] > 0:
                element_width_bytes = inst["smem_dtype"].width // 8
                total_bytes = inst["smem_per_page"] * inst["num_pages"]
                num_elements = total_bytes // element_width_bytes
                unrolled_ops.append(
                    f"        smem_op_{i} = smem_alloc.allocate_tensor("
                    f"instructions[{i}]['smem_dtype'], cute.make_layout({num_elements}))"
                )
                smem_str = f"smem_op_{i}, "

            unrolled_ops.append(f"        # Op {i}: Compute")
            unrolled_ops.append(f"        op_{i}.compute({smem_str}{args_str})")

            # Pipelined load of next operation (if paged pool available)
            if self.paged_pool_bytes > 0 and i + 1 < n_ops:
                next_indices = mapping[i + 1]
                next_args_str = ", ".join([f"arg_{idx}" for idx in next_indices])
                unrolled_ops.append(f"        # Overlap: Load Op {i + 1}")
                unrolled_ops.append("        next_page = (page_idx + Int32(1)) % Int32(2)")
                unrolled_ops.append(f"        op_{i + 1}.load(paged_pool, next_page, {next_args_str})")

            # Store current operation (if paged pool available)
            if self.paged_pool_bytes > 0:
                unrolled_ops.append(f"        # Op {i}: Store")
                unrolled_ops.append(f"        op_{i}.store(paged_pool, page_idx, {args_str})")
                if i + 1 < n_ops:
                    unrolled_ops.append("        page_idx = next_page")

            unrolled_ops.append("        cute.arch.sync_threads()")

            if inst["needs_sync"]:
                unrolled_ops.append("        # Global Barrier")
                unrolled_ops.append("        if tidx == 0:")
                unrolled_ops.append("            atomic_add_i32(1, barrier_tensor.iterator)")
                unrolled_ops.append("            target = (sync_step + 1) * n_blocks")
                unrolled_ops.append("            while atomic_add_i32(0, barrier_tensor.iterator) < target:")
                unrolled_ops.append("                pass")
                unrolled_ops.append("        cute.arch.sync_threads()")
                unrolled_ops.append("        sync_step = sync_step + Int32(1)")

        bindings_content = "\n".join(module_bindings)
        ops_content = "\n".join(unrolled_ops)

        # Note: We must fetch instructions dynamically in __call__ if we want to support
        # grid/block updates from the registry, BUT module bindings above (impl_i) are static!
        # This implies that `impl_i` (the Kernel object) must be consistent / interchangeable.
        # We assume that creating a new GatedLinearSM80 is fine as long as logic is same.
        # The grid/block are read from `instructions[0]['grid']` in launch.
        # We need to make sure `instructions` used in launch is the one in registry.

        content = f"""
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Int32, const_expr
from quack.utils import atomic_add_i32
from machete.megakernel.core import MEGAKERNEL_REGISTRY

# Retrieve instructions from registry
instructions = MEGAKERNEL_REGISTRY["{sig_hash}"]

# Bind module-level symbols
{bindings_content}

class GeneratedMegakernel:
    @cute.jit
    def __call__(self, barrier_tensor, n_blocks, {all_args_str}, stream: cuda.CUstream):
        # We launch utilizing grid/block from the registry instructions
        # This allows the caller to update registry with current grid/block before call.
        self.kernel(barrier_tensor, n_blocks, {all_args_str}).launch(
            grid=instructions[0]['grid'],
            block=instructions[0]['block'],
            smem=instructions[0]['total_smem'],
            stream=stream
        )

    @cute.kernel
    def kernel(self, barrier_tensor, n_blocks, {all_args_str}):
        sync_step = Int32(0)
        tidx, _, _ = cute.arch.thread_idx()
{ops_content}
"""
        with open(gen_path, "w") as f:
            f.write(content)

        spec = importlib.util.spec_from_file_location(f"gen_{sig_hash}", gen_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module.GeneratedMegakernel(), sig_hash

    def launch(self, barrier_tensor: torch.Tensor, n_blocks: int, grid, block, stream=None):
        flat_args = []
        mapping = []
        curr = 0
        for inst in self.instructions:
            args = inst["args"]
            flat_args.extend(args)
            mapping.append(list(range(curr, curr + len(args))))
            curr += len(args)
            inst["grid"] = grid
            inst["block"] = block

        num_flat_args = len(flat_args)

        # Total shared memory = paged pool + per-op allocations
        total_smem = self.paged_pool_bytes
        for inst in self.instructions:
            total_smem += inst["smem_per_page"] * inst["num_pages"]

        for inst in self.instructions:
            inst["total_smem"] = total_smem

        # Build compile key
        arg_info = []
        for arg in flat_args:
            if isinstance(arg, torch.Tensor):
                arg_info.append((arg.dtype, tuple(arg.shape), tuple(arg.stride())))
            elif isinstance(arg, int):
                arg_info.append(("int", arg))
            else:
                arg_info.append((type(arg), None))

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

        op_info_key = str(op_info)
        # Inclusion of self.mode is critical to distinguish fwd/bwd paths
        compile_key = (self.mode, tuple(op_info), tuple(arg_info), tuple(grid), tuple(block))

        # Calculate hash to find registry entry
        data_to_hash = f"{self.name}_{num_flat_args}_{op_info_key}"
        sig_hash = hashlib.md5(data_to_hash.encode()).hexdigest()

        # Ensure registry has the current grid/block/smem values AND current arguments
        # The generated code reads from MEGAKERNEL_REGISTRY[sig_hash]
        # We must update that list's dictionaries with our current launch params.
        if sig_hash in MEGAKERNEL_REGISTRY:
            # Registry exists, update it with our current environment
            cached_instrs = MEGAKERNEL_REGISTRY[sig_hash]
            for i, inst in enumerate(self.instructions):
                # Update the cached instruction dict in place
                # Tensors must be updated too because their addresses change!
                cached_instrs[i].update(inst)
        else:
            # Logic inside _get_megakernel_class will populate registry if missing.
            pass

        if compile_key not in _GLOBAL_COMPILE_CACHE:
            fake_args = []
            for arg in flat_args:
                if isinstance(arg, torch.Tensor):
                    fake_args.append(fake_tensor(torch2cute_dtype_map[arg.dtype], arg.shape))
                elif isinstance(arg, int):
                    fake_args.append(Int32(arg))
                else:
                    fake_args.append(arg)

            barrier_fake = fake_tensor(Int32, (1,))
            megakernel_obj, _ = self._get_megakernel_class(mapping, num_flat_args, op_info_key)

            _GLOBAL_COMPILE_CACHE[compile_key] = cute.compile(
                megakernel_obj,
                barrier_fake,
                Int32(n_blocks),
                *fake_args,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )

        _GLOBAL_COMPILE_CACHE[compile_key](barrier_tensor, Int32(n_blocks), *flat_args)
