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

# Global registry to pass symbols to generated kernels
MEGAKERNEL_REGISTRY = {}

# Global compilation cache to reuse compiled kernels across Megakernel instances
_GLOBAL_COMPILE_CACHE = {}


class Megakernel:
    def __init__(self, name: str = "megakernel", mode: str = "forward"):
        self.name = name
        self.mode = mode
        self.instructions = []
        self.gen_dir = os.path.join(os.path.dirname(__file__), ".generated")
        os.makedirs(self.gen_dir, exist_ok=True)

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
                # If the method belongs to an object, check if that object requests specific shared memory
                # But be careful not to pick up base class defaults if they just return 0/1 and decorator specified something else?
                # Actually, FusableKernel defaults are 0 and 1.
                # If the class overrides them, we want them.
                # Since decorator is static, class properties are "truer".
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
        unrolled_ops.append("        # Semaphores for paged coordination (if needed)")

        for i, inst in enumerate(self.instructions):
            indices = mapping[i]
            args_str = ", ".join([f"arg_{idx}" for idx in indices])

            smem_str = ""
            if inst["smem_per_page"] > 0:
                # Calculate number of elements
                # Assuming cute types have .width in bits
                element_width_bytes = inst["smem_dtype"].width // 8
                total_bytes = inst["smem_per_page"] * inst["num_pages"]
                num_elements = total_bytes // element_width_bytes

                unrolled_ops.append(
                    f"        # Allocate {inst['num_pages']} pages of {inst['smem_per_page']} bytes for Op {i}"
                )
                unrolled_ops.append(
                    f"        smem_op_{i} = smem_alloc.allocate_tensor(instructions[{i}]['smem_dtype'], cute.make_layout({num_elements}))"
                )
                smem_str = f"smem_op_{i}, "

            compute_fn = inst["compute"]
            if hasattr(compute_fn, "__self__"):
                module_bindings.append(f"impl_{i} = instructions[{i}]['compute'].__self__")
                module_bindings.append(f"func_{i} = instructions[{i}]['compute'].__func__")
                call_str = f"func_{i}(impl_{i}, {smem_str}{args_str})"
            else:
                module_bindings.append(f"func_{i} = instructions[{i}]['compute']")
                call_str = f"func_{i}({smem_str}{args_str})"

            unrolled_ops.append(f"        # Operation {i}")
            unrolled_ops.append(f"        {call_str}")
            # Ensure memory consistency between operations in the fused sequence.
            # Operations are executed sequentially by threads, but if they interact via
            # shared or global memory (as they often do in fusion), we need full visibility.
            # While __syncthreads() is heavy, it provides the required safety for fusion.
            unrolled_ops.append(f"        cute.arch.sync_threads()")

            if inst["needs_sync"]:
                unrolled_ops.append(f"        # Global Barrier")
                unrolled_ops.append(f"        if tidx == 0:")
                unrolled_ops.append(f"            atomic_add_i32(1, barrier_tensor.iterator)")
                unrolled_ops.append(f"            target = (sync_step + 1) * n_blocks")
                unrolled_ops.append(f"            while atomic_add_i32(0, barrier_tensor.iterator) < target:")
                unrolled_ops.append(f"                pass")
                unrolled_ops.append(f"        cute.arch.sync_threads()")
                unrolled_ops.append(f"        sync_step = sync_step + Int32(1)")

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

        total_smem = 0
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
