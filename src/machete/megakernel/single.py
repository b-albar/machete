# Copyright (c) 2025, Machete Authors
import torch
from typing import Callable
from .interface import FusableKernel
from .core import Megakernel


class SingleKernel(torch.autograd.Function):
    def __init__(self, op: FusableKernel, grid_fn: Callable, block_fn: Callable):
        self.op = op
        self.grid_fn = grid_fn
        self.block_fn = block_fn
        self.mk_fwd = Megakernel(name=f"{type(op).__name__}_fwd", mode="forward")
        self.mk_bwd = Megakernel(name=f"{type(op).__name__}_bwd", mode="backward")

    @staticmethod
    def forward(ctx, runner, *args):
        ctx.runner = runner
        return runner.run_forward(ctx, *args)

    @staticmethod
    def backward(ctx, *grad_args):
        grads = ctx.runner.run_backward(ctx, *grad_args)
        # return None for the runner argument, followed by grads for *args
        return (None,) + grads

    def _update_or_add(self, mk: Megakernel, args: tuple):
        """
        Updates the arguments of the existing operation in the megakernel,
        or adds the operation if it hasn't been added yet.
        This avoids clearing and rebuilding the instruction list, which is crucial
        for correct caching and performance.
        """
        if not mk.instructions:
            mk.add(self.op, *args)
        else:
            # Update arguments for the existing operation
            # SingleKernel has exactly 1 op, so we update the 0-th instruction.
            mk.instructions[0]["args"] = list(args)

    def run_forward(self, ctx, *args):
        # Separate tensors and others used for rebuilding args in backward
        tensors = []
        others = []
        layout = []  # 't' for tensor, 'o' for other
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensors.append(arg)
                layout.append("t")
            else:
                others.append(arg)
                layout.append("o")

        ctx.save_for_backward(*tensors)
        ctx.others = others
        ctx.layout = layout

        grid = self.grid_fn(*args)
        block = self.block_fn(*args)
        n_blocks = grid[0] * grid[1] * grid[2]

        self._update_or_add(self.mk_fwd, args)
        self.mk_fwd.launch(n_blocks, grid, block)

        # Assume first argument is the in-place modified tensor
        return args[0]

    def run_backward(self, ctx, *grad_args):
        tensors = ctx.saved_tensors
        others = ctx.others
        layout = ctx.layout

        args = []
        t_ptr = 0
        o_ptr = 0
        for type_char in layout:
            if type_char == "t":
                args.append(tensors[t_ptr])
                t_ptr += 1
            else:
                args.append(others[o_ptr])
                o_ptr += 1
        args = tuple(args)

        main_grad = grad_args[0]

        if main_grad is None:
            return (None,) * len(args)

        bwd_args = (main_grad,) + args[1:]

        grid = self.grid_fn(*args)
        block = self.block_fn(*args)
        n_blocks = grid[0] * grid[1] * grid[2]

        self._update_or_add(self.mk_bwd, bwd_args)
        self.mk_bwd.launch(n_blocks, grid, block)

        # Return gradients
        # We assume only the first argument (mq) effectively receives a gradient (which is main_grad modified in place).
        return (main_grad,) + (None,) * (len(args) - 1)

    def apply_autograd(self, *args):
        # We must call the classmethod 'apply' from SingleKernel or its subclass.
        # Since SingleKernel inherits from Function, type(self).apply is the autograd entry point.
        return self.__class__.apply(self, *args)
