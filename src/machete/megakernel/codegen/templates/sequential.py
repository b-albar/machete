# Copyright (c) 2025, Machete Authors
"""
Sequential Fusing Kernel Template.

This template generates code for fusing multiple operations sequentially.
Each operation's L/C/S methods are inlined in topological order with
sync_threads() between operations.
"""

from typing import Dict, Any, Optional, List
from .common import KernelTemplate, TemplateContext
from ..inliner import InlinedKernel


class SequentialTemplate(KernelTemplate):
    """Template for sequential fusing of multiple operations.

    This template fuses multiple kernels by executing their L/C/S phases
    in sequence. It supports:
    - Shared memory reuse between operations
    - Early load optimization (next op's load during current op's compute)
    - Per-operation sync_threads() for correctness

    Generated kernel structure:
    ```
    @cute.kernel
    def kernel(...):
        # Setup
        tidx, bidx = ...
        logical_idx = bidx

        # Create tensors from pointers
        ...

        # Shared memory allocation
        smem = ...

        # Op 0: KernelA
        # load_forward
        ...
        sync_threads()
        # compute_forward
        ...
        sync_threads()
        # store_forward
        ...

        # Op 1: KernelB
        # load_forward
        ...
        sync_threads()
        # compute_forward
        ...
        sync_threads()
        # store_forward
        ...
    ```
    """

    def generate(self, ctx: TemplateContext) -> str:
        """Generate complete kernel module for sequential fusing.

        Args:
            ctx: Template context with kernel information

        Returns:
            Complete Python module source code
        """
        if ctx.num_ops < 1:
            raise ValueError("SequentialTemplate requires at least 1 operation")

        # Inline all kernels
        inlined_kernels = []
        for i in range(ctx.num_ops):
            inlined = self.inline_kernel_methods(ctx, i, ctx.mode)
            inlined_kernels.append(inlined)
            ctx.captured_values.update(inlined.all_captured)

        # Build kernel body
        kernel_body = self._build_kernel_body(ctx, inlined_kernels)

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
        """Generate the _captured dict."""
        if not ctx.captured_values:
            return "\n_captured = {}\n"

        lines = ["\n# Captured complex values from kernel instances"]
        lines.append("_captured = {}")
        return '\n'.join(lines)

    def _build_kernel_body(
        self,
        ctx: TemplateContext,
        inlined_kernels: List[InlinedKernel]
    ) -> str:
        """Build the kernel body with all inlined operations."""
        parts = []

        # Tensor creation (union of all tensors)
        parts.append(self._generate_all_tensor_creation(ctx))
        parts.append("")

        # Shared memory allocation
        parts.append(self.generate_smem_allocation(ctx))
        parts.append("")

        # Captured value bindings
        if ctx.captured_values:
            parts.append(self.generate_captured_bindings(ctx))
            parts.append("")

        # Generate code for each operation
        for i, inlined in enumerate(inlined_kernels):
            kernel = ctx.kernel_instances[i]
            kernel_name = type(kernel).__name__

            parts.append(f"        # ========== Op {i}: {kernel_name} ==========")
            parts.append("")

            # Check for early load optimization
            can_early_load_next = self._can_early_load(ctx, i, i + 1)

            # setup_kernel (if exists)
            if inlined.setup_kernel and not inlined.setup_kernel.is_empty:
                parts.append(self.format_inlined_code(
                    inlined.setup_kernel.source_code,
                    comment=f"Op {i} setup_kernel",
                    indent=8
                ))
                parts.append("")

            # Get the right methods based on mode
            if ctx.mode == "forward":
                load_method = inlined.load_forward
                compute_method = inlined.compute_forward
                store_method = inlined.store_forward
            else:
                load_method = inlined.load_backward
                compute_method = inlined.compute_backward
                store_method = inlined.store_backward

            # Load phase
            parts.append(self.format_inlined_code(
                load_method.source_code if load_method else "",
                comment=f"Op {i} load_{ctx.mode}",
                indent=8
            ))

            # Sync after load (only if there was actual load code)
            if load_method and not load_method.is_empty:
                parts.append("        cute.arch.sync_threads()")
            parts.append("")

            # Early load for next operation (overlap with compute)
            if can_early_load_next:
                next_inlined = inlined_kernels[i + 1]
                next_load = next_inlined.load_forward if ctx.mode == "forward" else next_inlined.load_backward
                if next_load and not next_load.is_empty:
                    parts.append(self.format_inlined_code(
                        next_load.source_code,
                        comment=f"Early load for Op {i + 1} (overlapped)",
                        indent=8
                    ))
                    parts.append("")

            # Compute phase
            parts.append(self.format_inlined_code(
                compute_method.source_code if compute_method else "",
                comment=f"Op {i} compute_{ctx.mode}",
                indent=8
            ))

            # Sync after compute
            parts.append("        cute.arch.sync_threads()")
            parts.append("")

            # Store phase
            parts.append(self.format_inlined_code(
                store_method.source_code if store_method else "",
                comment=f"Op {i} store_{ctx.mode}",
                indent=8
            ))

            # Sync after store (except for last op)
            if i < ctx.num_ops - 1:
                parts.append("        cute.arch.sync_threads()")
            parts.append("")

        return '\n'.join(parts)

    def _generate_all_tensor_creation(self, ctx: TemplateContext) -> str:
        """Generate tensor creation for all operations' tensors."""
        lines = []
        lines.append("        # Create tensors from pointers (all operations)")

        # Collect all unique tensors across operations
        seen_tensors = set()

        for i, kernel in enumerate(ctx.kernel_instances):
            if hasattr(kernel, 'declare_tensors'):
                specs = kernel.declare_tensors()
                for name, spec in specs.items():
                    if name not in seen_tensors:
                        seen_tensors.add(name)
                        layout_code = spec.generate_symbolic_layout_code()
                        lines.append(f"        {name} = cute.make_tensor({name}_ptr, {layout_code})")

        # Fallback to ctx.tensor_specs if no kernels have declare_tensors
        if not seen_tensors:
            for name, spec in ctx.tensor_specs.items():
                layout_code = spec.generate_symbolic_layout_code()
                lines.append(f"        {name} = cute.make_tensor({name}_ptr, {layout_code})")

        return '\n'.join(lines)

    def _can_early_load(self, ctx: TemplateContext, current_idx: int, next_idx: int) -> bool:
        """Check if next operation's load can be issued early.

        Early load optimization: issue next op's load during current op's compute
        to overlap memory latency with computation.

        This is only safe when:
        1. Next op exists
        2. Next op doesn't depend on current op's output
        3. Next op has sufficient separate smem allocation

        For now, we use a conservative check: only allow early load if
        the kernels explicitly support it.
        """
        if next_idx >= ctx.num_ops:
            return False

        # Check if both kernels exist
        current_kernel = ctx.kernel_instances[current_idx]
        next_kernel = ctx.kernel_instances[next_idx]

        # Check if next kernel has can_early_load attribute
        can_early = getattr(next_kernel, 'can_early_load', False)

        return can_early
