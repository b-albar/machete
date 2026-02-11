# Copyright (c) 2025, Machete Authors
"""
PyTorch autograd bridge for megakernel ops.

Wraps one or more AutogradOps into a single differentiable operation via
``torch.autograd.Function``. Handles config packing, kernel caching,
in-place mutation via ``mark_dirty()``, and gradient routing.

Usage:
    output = MegakernelFunction.apply(
        [RopeAutogradOp()],   # autograd_ops
        MegakernelConfig(),    # config
        q, cos, sin,           # flat input tensors
    )
"""

import io
import contextlib
import torch
from torch.autograd import Function

from .megakernel import Megakernel, MegakernelConfig
from .autograd_op import AutogradOp
from .kernel_cache import KernelCache


def _run_cached(scheduled_ops, mk_config, backward):
    """Run a megakernel with kernel caching. Executes exactly once.

    On cache hit: injects the cached compiled kernel and launches via run().
    On cache miss: compiles + executes via mk.run(), then caches the result.
    """
    cache = KernelCache.get()
    compiled_kernel = cache.lookup(scheduled_ops, mk_config, backward=backward)

    mk = Megakernel(scheduled_ops, config=mk_config, backward=backward)

    if compiled_kernel is not None:
        # Cache hit: inject cached kernel, then use public run()
        mk._compiled_kernel = compiled_kernel
        mk.run()
    else:
        # Cache miss: compile + execute once
        with contextlib.redirect_stdout(io.StringIO()):
            mk.run()
        cache.store(scheduled_ops, mk_config, backward, mk._compiled_kernel)


class MegakernelFunction(Function):
    """PyTorch autograd bridge for megakernel ops.

    Wraps one or more AutogradOps into a single differentiable node.
    Uses true in-place semantics via ``mark_dirty()`` — no tensor cloning.

    The first two arguments to ``apply()`` are non-tensor:
    - ``autograd_ops``: List[AutogradOp]
    - ``mk_config``: MegakernelConfig

    Followed by all input tensors flattened across ops in tensor_specs order.
    """

    @staticmethod
    def forward(ctx, autograd_ops, mk_config, *flat_inputs):
        # ---- Step 1: Unflatten inputs into per-op named dicts ----
        tensor_offset = 0
        per_op_tensors = []
        for aop in autograd_ops:
            input_specs = aop.input_specs()
            op_tensors = {}
            for spec in input_specs:
                op_tensors[spec.name] = flat_inputs[tensor_offset]
                tensor_offset += 1
            per_op_tensors.append(op_tensors)

        # ---- Step 2: Identify mutated tensors for mark_dirty ----
        mutated_tensors = []
        output_tensors = []
        for i, aop in enumerate(autograd_ops):
            for spec in aop.output_specs():
                if spec.mutated_from:
                    tensor = per_op_tensors[i][spec.mutated_from]
                    mutated_tensors.append(tensor)
                    output_tensors.append(tensor)

        # ---- Step 3: Build ScheduledOps via schedule() ----
        scheduled_ops = []
        for i, aop in enumerate(autograd_ops):
            prepared = aop.prepare_tensors(**per_op_tensors[i])
            scheduled_ops.append(aop.op_cls.schedule(
                tile_sizes=aop.get_tile_sizes(), **prepared
            ))

        # ---- Step 4: Run forward megakernel (in-place, cached) ----
        if mk_config is None:
            mk_config = MegakernelConfig()
        _run_cached(scheduled_ops, mk_config, backward=False)

        # ---- Step 5: mark_dirty for in-place mutation ----
        ctx.mark_dirty(*mutated_tensors)

        # ---- Step 6: Save tensors for backward ----
        save_tensors_list = []
        ctx._per_op_saved_names = []
        for i, aop in enumerate(autograd_ops):
            saved = aop.save_for_backward(**per_op_tensors[i])
            ctx._per_op_saved_names.append(list(saved.keys()))
            save_tensors_list.extend(saved.values())

        ctx.save_for_backward(*save_tensors_list)
        ctx._autograd_ops = autograd_ops
        ctx._mk_config = mk_config

        # ---- Step 7: Return outputs ----
        if len(output_tensors) == 1:
            return output_tensors[0]
        return tuple(output_tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        autograd_ops = ctx._autograd_ops
        mk_config = ctx._mk_config
        saved = ctx.saved_tensors

        # ---- Step 1: Reconstruct saved tensor dicts ----
        tensor_offset = 0
        per_op_saved = []
        for i, aop in enumerate(autograd_ops):
            names = ctx._per_op_saved_names[i]
            saved_dict = {}
            for name in names:
                saved_dict[name] = saved[tensor_offset]
                tensor_offset += 1
            per_op_saved.append(saved_dict)

        # ---- Step 2: Map grad_outputs to per-op dicts ----
        grad_offset = 0
        per_op_grads = []
        for i, aop in enumerate(autograd_ops):
            grad_dict = {}
            for spec in aop.output_specs():
                grad_dict[spec.name] = grad_outputs[grad_offset]
                grad_offset += 1
            per_op_grads.append(grad_dict)

        # ---- Step 3: Build backward ScheduledOps via schedule() ----
        scheduled_ops = []
        for i, aop in enumerate(autograd_ops):
            # Map grad_outputs to Op input names via mutated_from
            backward_tensors = dict(per_op_saved[i])
            for spec in aop.output_specs():
                if spec.mutated_from and spec.name in per_op_grads[i]:
                    backward_tensors[spec.mutated_from] = per_op_grads[i][spec.name]

            prepared = aop.prepare_tensors(**backward_tensors)
            scheduled_ops.append(aop.op_cls.schedule(
                tile_sizes=aop.get_tile_sizes(), **prepared
            ))

        # ---- Step 4: Run backward megakernel (in-place on grad, cached) ----
        _run_cached(scheduled_ops, mk_config, backward=True)

        # ---- Step 5: Build gradient tuple ----
        # Return None for autograd_ops, mk_config, then per-input grads
        grad_inputs = [None, None]  # autograd_ops, mk_config

        for i, aop in enumerate(autograd_ops):
            for spec in aop.input_specs():
                if spec.needs_grad:
                    # Find the output that mutated_from this input
                    for out_spec in aop.output_specs():
                        if out_spec.mutated_from == spec.name:
                            grad_inputs.append(per_op_grads[i][out_spec.name])
                            break
                    else:
                        grad_inputs.append(None)
                else:
                    grad_inputs.append(None)

        return tuple(grad_inputs)


__all__ = ["MegakernelFunction"]
