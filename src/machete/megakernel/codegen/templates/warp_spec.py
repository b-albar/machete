# Copyright (c) 2025, Machete Authors
"""
Warp-Specialized Kernel Template.

This template generates code for warp-specialized producer-consumer patterns
following the "No Bubbles" approach from HazyResearch.

Key Features:
1. Single @cute.jit function containing the full pipeline
2. Shared barriers between load/compute/store phases
3. Inter-op pipelining: load of next op overlaps with compute of previous
4. Semaphore-based synchronization for fine-grained handoff

Note: The same MacheteKernel can be executed in either sequential or warp-specialized
mode. This template is selected when kernel.uses_warp_specialization returns True.
The kernel's load_forward/compute_forward/store_forward methods are used regardless
of execution mode.
"""

from typing import Dict, Any, Optional, List
from .common import KernelTemplate, TemplateContext
from ..inliner import InlinedKernel


class WarpSpecializedTemplate(KernelTemplate):
    """Template for warp-specialized producer-consumer kernels.

    This template implements the "No Bubbles" pattern where different warps
    execute different roles concurrently:
    - Loader warps: Execute load_forward/load_backward methods
    - Consumer warps: Execute compute_forward/compute_backward methods
    - Storer warps: Execute store_forward/store_backward methods

    The same MacheteKernel class can be executed in either sequential or
    warp-specialized mode - this template is selected when the kernel's
    uses_warp_specialization property returns True.

    Inter-Op Pipelining:
    When fusing multiple warp-specialized kernels (Op A -> Op B -> Op C),
    this template enables overlapped execution:
    - While consumer warps compute Op A, loader warps can prefetch Op B's data
    - This requires sufficient shared memory for both operations
    - Control via kernel.supports_inter_op_pipelining() and get_inter_op_smem_requirement()

    Synchronization:
    - Intra-op: semaphores between load->compute->store for same operation
    - Inter-op: semaphores between Op N store -> Op N+1 load (for dependent ops)
    - Independent ops with inter-op pipelining can overlap without waiting

    Generated kernel structure:
    ```
    @cute.kernel
    def kernel(...):
        tidx, bidx = ...
        warp_id = tidx // 32
        lane_id = tidx % 32

        # Initialize semaphores
        if tidx == 0:
            init_semaphores()
        sync_threads()

        # For each operation (with inter-op overlap where possible):
        for op_idx in operations:
            # Loader warp: load op_idx data (may overlap with prev compute)
            # Consumer warp: wait for load, compute, signal done
            # Storer warp: wait for compute, store, signal inter-op done
    ```
    """

    def generate(self, ctx: TemplateContext) -> str:
        """Generate complete kernel module for warp-specialized execution.

        Args:
            ctx: Template context with kernel information

        Returns:
            Complete Python module source code
        """
        # Inline all kernels
        inlined_kernels = []
        for i in range(ctx.num_ops):
            inlined = self.inline_kernel_methods(ctx, i, ctx.mode)
            inlined_kernels.append(inlined)
            ctx.captured_values.update(inlined.all_captured)

        # Get warp configuration from first kernel that has it
        warp_config = self._get_warp_config(ctx)

        # Determine inter-op pipelining opportunities
        inter_op_pipeline = self._analyze_inter_op_pipelining(ctx)

        # Build kernel body
        kernel_body = self._build_kernel_body(ctx, inlined_kernels, warp_config, inter_op_pipeline)

        # Assemble complete module
        parts = [
            self.generate_imports(ctx),
            self._generate_extra_imports(),
            self._generate_registry_lookup(ctx),
            self._generate_captured_dict(ctx),
            self.generate_kernel_class(ctx, kernel_body),
            self.generate_compile_function(ctx),
        ]

        return '\n'.join(parts)

    def _generate_extra_imports(self) -> str:
        """Generate additional imports for warp-specialized execution."""
        return '''
from machete.megakernel.utils import (
    semaphore_init, semaphore_signal, semaphore_wait, semaphore_try_wait, nanosleep
)
'''

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

    def _get_warp_config(self, ctx: TemplateContext):
        """Get warp configuration from kernels."""
        for kernel in ctx.kernel_instances:
            if hasattr(kernel, 'warp_config'):
                return kernel.warp_config
            if hasattr(kernel, 'WARP_CONFIG'):
                return kernel.WARP_CONFIG

        # Default warp config
        from machete.megakernel.scheduler import WarpConfig
        return WarpConfig()

    def _analyze_inter_op_pipelining(self, ctx: TemplateContext) -> Dict[int, int]:
        """Analyze which operations can have their loads pipelined with previous computes.

        Returns:
            Dict mapping op_idx to the op whose compute it can overlap with.
            If op B can load during op A's compute, result[B] = A.
        """
        inter_op_pipeline = {}

        if ctx.num_ops < 2:
            return inter_op_pipeline

        # Calculate available smem
        total_smem = ctx.smem_size

        for i in range(1, ctx.num_ops):
            prev_kernel = ctx.kernel_instances[i - 1]
            curr_kernel = ctx.kernel_instances[i]

            # Check if both kernels support inter-op pipelining
            if not self._supports_inter_op_pipelining(prev_kernel):
                continue
            if not self._supports_inter_op_pipelining(curr_kernel):
                continue

            # Check smem requirements
            prev_smem = self._get_smem_size(prev_kernel, ctx.mode)
            curr_prefetch_smem = self._get_inter_op_smem_requirement(curr_kernel)

            # Check if we have enough smem for overlap
            if prev_smem + curr_prefetch_smem <= total_smem:
                # Can overlap curr's load with prev's compute
                inter_op_pipeline[i] = i - 1

        return inter_op_pipeline

    def _supports_inter_op_pipelining(self, kernel) -> bool:
        """Check if kernel supports inter-op pipelining."""
        # Must use warp specialization to support inter-op pipelining
        if not getattr(kernel, 'uses_warp_specialization', False):
            return False
        if hasattr(kernel, 'supports_inter_op_pipelining'):
            return kernel.supports_inter_op_pipelining()
        return True

    def _get_smem_size(self, kernel, mode: str) -> int:
        """Get smem size for a kernel."""
        if mode == "forward":
            return getattr(kernel, 'smem_size_fwd', 0) or getattr(kernel, 'smem_size', 0)
        else:
            return getattr(kernel, 'smem_size_bwd', 0) or getattr(kernel, 'smem_size', 0)

    def _get_inter_op_smem_requirement(self, kernel) -> int:
        """Get smem requirement for inter-op pipelining."""
        if hasattr(kernel, 'get_inter_op_smem_requirement'):
            return kernel.get_inter_op_smem_requirement()
        # Default: one stage's worth
        smem = self._get_smem_size(kernel, "forward")
        num_stages = getattr(kernel, 'NUM_STAGES', 2)
        return smem // num_stages

    def _build_kernel_body(
        self,
        ctx: TemplateContext,
        inlined_kernels: List[InlinedKernel],
        warp_config,
        inter_op_pipeline: Dict[int, int],
    ) -> str:
        """Build kernel body with warp-specialized execution and inter-op pipelining."""
        parts = []

        num_consumer = warp_config.num_consumer_warps
        loader_warp = num_consumer
        storer_warp = num_consumer + warp_config.num_loader_warps
        num_ops = ctx.num_ops

        # Warp role calculation
        parts.append("        # Warp role calculation")
        parts.append("        warp_id = tidx // Int32(32)")
        parts.append("        lane_id = tidx % Int32(32)")
        parts.append(f"        is_loader = warp_id == Int32({loader_warp})")
        parts.append(f"        is_consumer = warp_id < Int32({num_consumer})")
        parts.append(f"        is_storer = warp_id == Int32({storer_warp})")
        parts.append("")

        # Tensor creation
        parts.append(self._generate_all_tensor_creation(ctx))
        parts.append("")

        # Shared memory allocation (includes semaphore space)
        parts.append(self.generate_smem_allocation(ctx))
        parts.append("")

        # Calculate semaphore count:
        # - 2 per op for intra-op (load_done, compute_done)
        # - 1 per op for inter-op (store_done for dependent ops)
        total_sems = num_ops * 3
        parts.append(f"        # Semaphores: 3 per op (load_done, compute_done, store_done)")
        parts.append(f"        NUM_OPS = const_expr({num_ops})")
        parts.append(f"        TOTAL_SEMS = const_expr({total_sems})")
        parts.append("")

        # Initialize semaphores
        parts.append("        # Initialize semaphores (thread 0 only)")
        parts.append("        if tidx == Int32(0):")
        for i in range(num_ops):
            load_done = i * 3
            compute_done = i * 3 + 1
            store_done = i * 3 + 2
            parts.append(f"            semaphore_init(smem, Int32({load_done}), Int32(0))  # op{i}_load_done")
            parts.append(f"            semaphore_init(smem, Int32({compute_done}), Int32(0))  # op{i}_compute_done")
            parts.append(f"            semaphore_init(smem, Int32({store_done}), Int32(0))  # op{i}_store_done")
        parts.append("        cute.arch.sync_threads()")
        parts.append("")

        # Captured value bindings
        if ctx.captured_values:
            parts.append(self.generate_captured_bindings(ctx))
            parts.append("")

        # Generate code for each operation with warp dispatch and inter-op pipelining
        for i, inlined in enumerate(inlined_kernels):
            kernel = ctx.kernel_instances[i]
            kernel_name = type(kernel).__name__

            # Check if this op can have its load pipelined with previous compute
            can_overlap_load = i in inter_op_pipeline
            overlap_with = inter_op_pipeline.get(i)

            parts.append(f"        # ========== Op {i}: {kernel_name} ==========")
            if can_overlap_load:
                parts.append(f"        # NOTE: Load can overlap with Op {overlap_with}'s compute")
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

            load_done_sem = i * 3
            compute_done_sem = i * 3 + 1
            store_done_sem = i * 3 + 2

            # ============ LOADER WARP ============
            parts.append(f"        # --- Loader warp for Op {i} ---")
            parts.append("        if is_loader:")

            # Wait for previous op's store if there's a data dependency AND no inter-op pipeline
            if i > 0 and not can_overlap_load:
                prev_store_done = (i - 1) * 3 + 2
                parts.append(f"            # Wait for Op {i-1} store (data dependency)")
                parts.append(f"            if lane_id == Int32(0):")
                parts.append(f"                semaphore_wait(smem, Int32({prev_store_done}), Int32(1))")
                parts.append(f"            cute.arch.sync_warp()")
            elif can_overlap_load:
                # Inter-op pipelining: can start loading while previous computes
                # But still need to wait if previous hasn't even started loading
                # We wait for previous load to be done (to ensure smem layout is stable)
                prev_load_done = (i - 1) * 3
                parts.append(f"            # Inter-op pipeline: overlap with Op {overlap_with} compute")
                parts.append(f"            # Wait for Op {overlap_with} load (smem layout stable)")
                parts.append(f"            if lane_id == Int32(0):")
                parts.append(f"                semaphore_wait(smem, Int32({prev_load_done}), Int32(1))")
                parts.append(f"            cute.arch.sync_warp()")

            # Execute load
            if load_method and not load_method.is_empty:
                load_code = self._indent_code(load_method.source_code, 12)
                parts.append(load_code)
            else:
                parts.append("            pass  # No load work")

            # Signal load done
            parts.append(f"            if lane_id == Int32(0):")
            parts.append(f"                semaphore_signal(smem, Int32({load_done_sem}))")
            parts.append("")

            # ============ CONSUMER WARPS ============
            parts.append(f"        # --- Consumer warps for Op {i} ---")
            parts.append("        if is_consumer:")

            # Wait for load
            parts.append(f"            if lane_id == Int32(0):")
            parts.append(f"                semaphore_wait(smem, Int32({load_done_sem}), Int32(1))")
            parts.append(f"            cute.arch.sync_warp()")

            # Execute compute
            if compute_method and not compute_method.is_empty:
                compute_code = self._indent_code(compute_method.source_code, 12)
                parts.append(compute_code)
            else:
                parts.append("            pass  # No compute work")

            # Signal compute done
            parts.append(f"            if lane_id == Int32(0):")
            parts.append(f"                semaphore_signal(smem, Int32({compute_done_sem}))")
            parts.append("")

            # ============ STORER WARP ============
            parts.append(f"        # --- Storer warp for Op {i} ---")
            parts.append("        if is_storer:")

            # Wait for compute
            parts.append(f"            if lane_id == Int32(0):")
            parts.append(f"                semaphore_wait(smem, Int32({compute_done_sem}), Int32(1))")
            parts.append(f"            cute.arch.sync_warp()")

            # Execute store
            if store_method and not store_method.is_empty:
                store_code = self._indent_code(store_method.source_code, 12)
                parts.append(store_code)
            else:
                parts.append("            pass  # No store work")

            # Signal store done (for inter-op dependencies)
            parts.append(f"            if lane_id == Int32(0):")
            parts.append(f"                semaphore_signal(smem, Int32({store_done_sem}))")
            parts.append("")

        return '\n'.join(parts)

    def _generate_all_tensor_creation(self, ctx: TemplateContext) -> str:
        """Generate tensor creation for all operations' tensors."""
        lines = []
        lines.append("        # Create tensors from pointers (all operations)")

        seen_tensors = set()

        for i, kernel in enumerate(ctx.kernel_instances):
            if hasattr(kernel, 'declare_tensors'):
                specs = kernel.declare_tensors()
                for name, spec in specs.items():
                    if name not in seen_tensors:
                        seen_tensors.add(name)
                        layout_code = spec.generate_symbolic_layout_code()
                        lines.append(f"        {name} = cute.make_tensor({name}_ptr, {layout_code})")

        if not seen_tensors:
            for name, spec in ctx.tensor_specs.items():
                layout_code = spec.generate_symbolic_layout_code()
                lines.append(f"        {name} = cute.make_tensor({name}_ptr, {layout_code})")

        return '\n'.join(lines)

    def _indent_code(self, code: str, indent: int) -> str:
        """Indent code by a given amount."""
        if not code or not code.strip():
            indent_str = ' ' * indent
            return f"{indent_str}pass"

        indent_str = ' ' * indent
        lines = []
        for line in code.split('\n'):
            if line.strip():
                lines.append(f"{indent_str}{line}")
            else:
                lines.append("")
        return '\n'.join(lines)
