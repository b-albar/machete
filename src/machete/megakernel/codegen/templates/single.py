# Copyright (c) 2025, Machete Authors
"""
Single Operation Kernel Template.

This template generates code for a single kernel operation (no fusing).
The L/C/S methods are inlined directly into a single @cute.jit kernel.
"""

from typing import Dict, Any, Optional
from .common import KernelTemplate, TemplateContext


class SingleKernelTemplate(KernelTemplate):
    """Template for single-operation kernels.

    This is the simplest template - it takes a single kernel's L/C/S methods
    and inlines them into a straightforward Load -> sync -> Compute -> sync -> Store
    sequence.

    Generated kernel structure:
    ```
    @cute.kernel
    def kernel(...):
        # Setup
        tidx, bidx = ...
        logical_idx = bidx

        # Create tensors from pointers
        q = cute.make_tensor(q_ptr, ...)

        # Shared memory allocation
        smem = ...

        # setup_kernel (per logical_idx) - INLINED
        ...

        # load_forward - INLINED
        ...
        cute.arch.sync_threads()

        # compute_forward - INLINED
        ...
        cute.arch.sync_threads()

        # store_forward - INLINED
        ...
    ```
    """

    def generate(self, ctx: TemplateContext) -> str:
        """Generate complete kernel module for a single operation.

        Args:
            ctx: Template context with kernel information

        Returns:
            Complete Python module source code
        """
        if ctx.num_ops != 1:
            raise ValueError(f"SingleKernelTemplate expects 1 op, got {ctx.num_ops}")

        kernel = ctx.kernel_instances[0]

        # Inline all methods
        inlined = self.inline_kernel_methods(ctx, 0, ctx.mode)

        # Merge captured values into context
        ctx.captured_values.update(inlined.all_captured)

        # Build kernel body
        kernel_body = self._build_kernel_body(ctx, inlined)

        # Assemble complete module
        parts = [
            self.generate_imports(ctx),
            self._generate_registry_lookup(ctx),
            self._generate_captured_dict(ctx),
            self.generate_kernel_class(ctx, kernel_body),
            self.generate_compile_function(ctx),
        ]

        return '\n'.join(parts)

    def _generate_registry_lookup(self, ctx: TemplateContext) -> str:
        """Generate code to look up instructions from registry."""
        if not ctx.sig_hash:
            return ""

        return f'''
from machete.megakernel.core import MEGAKERNEL_REGISTRY

# Retrieve instructions from registry
instructions = MEGAKERNEL_REGISTRY["{ctx.sig_hash}"]
'''

    def _generate_captured_dict(self, ctx: TemplateContext) -> str:
        """Generate the _captured dict for complex captured values."""
        if not ctx.captured_values:
            return "\n_captured = {}\n"

        # The actual values will be bound at module load time
        # from the kernel instances stored in the registry
        lines = ["\n# Captured complex values from kernel instance"]
        lines.append("_captured = {}")

        # These get populated when the module is loaded
        for name in ctx.captured_values:
            lines.append(f"# _captured['{name}'] = ... (bound at load time)")

        return '\n'.join(lines)

    def _build_kernel_body(self, ctx: TemplateContext, inlined) -> str:
        """Build the kernel body with all inlined code."""
        parts = []

        # Tensor creation
        parts.append(self.generate_tensor_creation(ctx))
        parts.append("")

        # Shared memory allocation
        parts.append(self.generate_smem_allocation(ctx))
        parts.append("")

        # Captured value bindings (if any)
        if ctx.captured_values:
            parts.append(self.generate_captured_bindings(ctx))
            parts.append("")

        # setup_kernel (if exists and not empty)
        if inlined.setup_kernel and not inlined.setup_kernel.is_empty:
            parts.append(self.format_inlined_code(
                inlined.setup_kernel.source_code,
                comment="setup_kernel (per logical_idx)",
                indent=8
            ))
            parts.append("")

        # Load phase
        if ctx.mode == "forward":
            load_method = inlined.load_forward
            compute_method = inlined.compute_forward
            store_method = inlined.store_forward
        else:
            load_method = inlined.load_backward
            compute_method = inlined.compute_backward
            store_method = inlined.store_backward

        # Load
        parts.append(self.format_inlined_code(
            load_method.source_code if load_method else "",
            comment=f"load_{ctx.mode}",
            indent=8
        ))

        # Sync after load
        parts.append("        cute.arch.sync_threads()")
        parts.append("")

        # Compute
        parts.append(self.format_inlined_code(
            compute_method.source_code if compute_method else "",
            comment=f"compute_{ctx.mode}",
            indent=8
        ))

        # Sync after compute
        parts.append("        cute.arch.sync_threads()")
        parts.append("")

        # Store
        parts.append(self.format_inlined_code(
            store_method.source_code if store_method else "",
            comment=f"store_{ctx.mode}",
            indent=8
        ))

        return '\n'.join(parts)

    def generate_tensor_creation(self, ctx: TemplateContext) -> str:
        """Generate tensor creation code for single op.

        Override to handle the case where kernel has declare_tensors().
        """
        kernel = ctx.kernel_instances[0]

        lines = []
        lines.append("        # Create tensors from pointers")

        # Check if kernel uses the new interface
        if hasattr(kernel, 'declare_tensors'):
            specs = kernel.declare_tensors()
            for name, spec in specs.items():
                layout_code = spec.generate_symbolic_layout_code()
                lines.append(f"        {name} = cute.make_tensor({name}_ptr, {layout_code})")
        else:
            # Fallback for legacy kernels - use ctx.tensor_specs
            for name, spec in ctx.tensor_specs.items():
                layout_code = spec.generate_symbolic_layout_code()
                lines.append(f"        {name} = cute.make_tensor({name}_ptr, {layout_code})")

        return '\n'.join(lines)
