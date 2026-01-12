import torch


class MegakernelAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runner, *args):
        ctx.runner = runner
        return runner.run_forward(ctx, *args)

    @staticmethod
    def backward(ctx, *grad_args):
        grads = ctx.runner.run_backward(ctx, *grad_args)
        # return None for the runner argument, followed by grads for *args
        return (None,) + grads
