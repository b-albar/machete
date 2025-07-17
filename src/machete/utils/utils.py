import torch


def maybe_contiguous(x: torch.Tensor):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x
