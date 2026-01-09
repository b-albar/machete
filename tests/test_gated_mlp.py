import sys
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from machete.utils.testing import verify_kernel
from machete.kernels.gated_mlp import gated_mlp_func


def test_gated_mlp():
    device = "cuda"
    dtype = torch.float16
    d_model = 128
    d_intermediate = 256

    w_gate_up = torch.randn(d_model, 2 * d_intermediate, device=device, dtype=dtype)
    gate, up = w_gate_up.chunk(2, dim=-1)
    w_interleaved = torch.stack((gate, up), dim=-1).flatten(-2)

    def ref_func_interleaved(x, w):
        gu = x @ w
        gu = gu.reshape(*gu.shape[:-1], -1, 2)
        gate = gu[..., 0]
        up = gu[..., 1]
        return torch.nn.functional.silu(gate) * up

    batch_sizes = [(2, 128, d_model)]

    for shape in batch_sizes:
        print(f"Testing shape {shape}")
        x = torch.randn(*shape, device=device, dtype=dtype)

        # Rely on gated_mlp_func to dispatch to the correct kernel for the local GPU
        print(f"Testing GatedMLP on {torch.cuda.get_device_name(0)}")

        verify_kernel(
            "GatedMLP",
            gated_mlp_func,
            ref_func_interleaved,
            (x, w_interleaved),
            dtype,
            check_grad=True,
        )


if __name__ == "__main__":
    test_gated_mlp()
