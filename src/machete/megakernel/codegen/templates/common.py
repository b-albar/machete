# Copyright (c) 2025, Machete Authors
"""
Common Base Classes and Utilities for Kernel Templates.

This module provides the base KernelTemplate class and TemplateContext
that all template types inherit from.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..inliner import CodeInliner, InlinedKernel


@dataclass
class TemplateContext:
    """Context passed to template generators containing all kernel information.

    Attributes:
        kernel_instances: List of kernel instances to fuse
        tensor_specs: Dict of all tensor specs from all kernels
        scalar_names: List of scalar parameter names
        smem_size: Total shared memory size in bytes
        grid: Grid dimensions tuple (x, y, z)
        block: Block dimensions tuple (x, y, z)
        num_stages: Number of pipeline stages
        mode: "forward" or "backward"
        sig_hash: Unique hash for this kernel configuration
        inlined_kernels: Pre-inlined kernel code (optional, computed if not provided)
    """
    kernel_instances: List[Any]
    tensor_specs: Dict[str, Any]  # TensorSpec objects
    scalar_names: List[str]
    smem_size: int
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    num_stages: int = 1
    mode: str = "forward"
    sig_hash: str = ""
    inlined_kernels: Optional[List[InlinedKernel]] = None
    captured_values: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_ops(self) -> int:
        """Number of operations being fused."""
        return len(self.kernel_instances)

    @property
    def num_threads(self) -> int:
        """Total threads per block."""
        return self.block[0] * self.block[1] * self.block[2]

    def get_kernel(self, idx: int) -> Any:
        """Get kernel instance by index."""
        return self.kernel_instances[idx]

    def get_tensor_names(self) -> List[str]:
        """Get all tensor names."""
        return list(self.tensor_specs.keys())

    def get_input_tensors(self) -> Dict[str, Any]:
        """Get tensors marked as inputs."""
        return {k: v for k, v in self.tensor_specs.items() if v.is_input}

    def get_output_tensors(self) -> Dict[str, Any]:
        """Get tensors marked as outputs."""
        return {k: v for k, v in self.tensor_specs.items() if v.is_output}


class KernelTemplate(ABC):
    """Base class for kernel code generation templates.

    Subclasses implement specific code generation strategies for different
    kernel types (single op, sequential fusing, warp-specialized).

    The template system generates complete Python modules that can be
    compiled with cute.compile().
    """

    # Standard imports for all generated kernels
    IMPORTS = '''# Auto-generated megakernel - DO NOT EDIT
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr
from cutlass.cute.runtime import make_ptr
from cutlass._mlir.dialects._cute_nvgpu_enum_gen import AddressSpace
'''

    def __init__(self):
        """Initialize the template."""
        self.inliner = CodeInliner()

    @abstractmethod
    def generate(self, ctx: TemplateContext) -> str:
        """Generate complete kernel module source code.

        Args:
            ctx: Template context with all kernel information

        Returns:
            Complete Python module source code
        """
        pass

    def generate_imports(self, ctx: TemplateContext) -> str:
        """Generate import statements.

        Override to add custom imports for specific template types.
        """
        return self.IMPORTS

    def generate_tensor_creation(self, ctx: TemplateContext) -> str:
        """Generate cute.make_tensor calls from TensorSpecs.

        This creates tensors from pointers at the start of the kernel.
        """
        lines = []
        lines.append("        # Create tensors from pointers")

        for name, spec in ctx.tensor_specs.items():
            # Generate symbolic layout code
            layout_code = spec.generate_symbolic_layout_code()
            lines.append(f"        {name} = cute.make_tensor({name}_ptr, {layout_code})")

        return '\n'.join(lines)

    def generate_smem_allocation(self, ctx: TemplateContext) -> str:
        """Generate shared memory allocation code."""
        lines = []
        lines.append("        # Shared memory allocation")
        lines.append("        smem_alloc = cutlass.utils.SmemAllocator()")

        # Allocate a shared buffer of the required size
        if ctx.smem_size > 0:
            # For simplicity, allocate as bytes and let kernels partition
            lines.append(f"        smem = smem_alloc.allocate_tensor(cute.Int8, cute.make_layout({ctx.smem_size}))")
        else:
            lines.append("        smem = None")

        return '\n'.join(lines)

    def generate_captured_bindings(self, ctx: TemplateContext) -> str:
        """Generate bindings for captured complex values."""
        if not ctx.captured_values:
            return ""

        lines = ["        # Captured values from kernel instances"]
        for name, value in ctx.captured_values.items():
            # The actual values are stored in a registry and looked up at runtime
            lines.append(f"        _k_{name} = _captured['{name}']")

        return '\n'.join(lines)

    def generate_args_list(self, ctx: TemplateContext) -> str:
        """Generate the function argument list.

        Args:
            ctx: Template context

        Returns:
            Comma-separated argument string like "n_blocks, q_ptr, cos_ptr, sin_ptr, seq_len"
        """
        args = ["n_blocks"]

        # Add tensor pointer arguments
        for name in ctx.tensor_specs:
            args.append(f"{name}_ptr")

        # Add scalar arguments
        for scalar in ctx.scalar_names:
            args.append(scalar)

        return ", ".join(args)

    def generate_kernel_class(
        self,
        ctx: TemplateContext,
        kernel_body: str,
    ) -> str:
        """Generate the kernel class with __call__ and kernel methods.

        Args:
            ctx: Template context
            kernel_body: The generated kernel body code

        Returns:
            Class definition string
        """
        args = self.generate_args_list(ctx)
        kernel_name = f"kernel_{ctx.sig_hash[:8]}" if ctx.sig_hash else "kernel"

        return f'''
class GeneratedMegakernel:
    @cute.jit
    def __call__(self, {args}):
        self.{kernel_name}({args}).launch(
            grid={ctx.grid},
            block={ctx.block},
            smem={ctx.smem_size}
        )

    @cute.kernel
    def {kernel_name}(self, {args}):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        logical_idx = bidx

{kernel_body}
'''

    def generate_compile_function(self, ctx: TemplateContext) -> str:
        """Generate the get_compiled_kernel function.

        Args:
            ctx: Template context

        Returns:
            Function definition string
        """
        args = self.generate_args_list(ctx)

        return f'''
# Pre-compiled kernel (populated on first use)
_compiled_kernel = None

def get_compiled_kernel({args}):
    """Get or compile the kernel with proper type hints."""
    global _compiled_kernel
    if _compiled_kernel is None:
        _compiled_kernel = cute.compile(
            GeneratedMegakernel(),
            {args}
        )
    return _compiled_kernel
'''

    def inline_kernel_methods(
        self,
        ctx: TemplateContext,
        kernel_idx: int,
        mode: str = "forward",
    ) -> InlinedKernel:
        """Inline all methods from a kernel.

        Args:
            ctx: Template context
            kernel_idx: Index of kernel in ctx.kernel_instances
            mode: "forward" or "backward"

        Returns:
            InlinedKernel with all inlined methods
        """
        kernel = ctx.kernel_instances[kernel_idx]

        # Build argument mapping if kernel has tensor specs
        arg_mapping = None
        if hasattr(kernel, 'declare_tensors'):
            specs = kernel.declare_tensors()
            arg_mapping = {f"{name}_ptr": name for name in specs}

        return self.inliner.inline_kernel(
            kernel,
            op_idx=kernel_idx,
            mode=mode,
            arg_mapping=arg_mapping,
        )

    def format_inlined_code(
        self,
        code: str,
        comment: str = "",
        indent: int = 8,
    ) -> str:
        """Format inlined code with optional comment and indentation.

        Args:
            code: The inlined code
            comment: Optional comment to add before the code
            indent: Number of spaces to indent

        Returns:
            Formatted code string
        """
        if not code or not code.strip():
            indent_str = ' ' * indent
            if comment:
                return f"{indent_str}# {comment} (no-op)"
            return f"{indent_str}pass"

        lines = []
        indent_str = ' ' * indent

        if comment:
            lines.append(f"{indent_str}# {comment}")

        # Add indentation to each line
        for line in code.split('\n'):
            if line.strip():
                lines.append(f"{indent_str}{line}")
            else:
                lines.append("")

        return '\n'.join(lines)
