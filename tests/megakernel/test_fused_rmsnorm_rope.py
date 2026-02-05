# Copyright (c) 2025, Machete Authors
"""Tests for fused RMSNorm + RoPE megakernel.

Verifies correctness of chaining RMSNormOp -> RopeOp in a single megakernel
against PyTorch reference implementations.

The framework automatically detects dependencies when the same tensor is used
as output of one op and input of another, even if buffer names differ
(e.g., RMSNormOp outputs 'y' and RopeOp reads 'q', but they share storage).
"""

import pytest
import torch

from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.rms_norm import RMSNormOp
from machete.kernels.rms_norm.ref import rmsnorm_pytorch
from machete.kernels.rope import RopeOp
from machete.kernels.rope.ref import rope_pytorch


def is_hopper_available():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def get_tolerances(dtype):
    """Get appropriate rtol/atol for dtype.

    bfloat16 has less precision than float32, so we use relaxed tolerances.
    """
    if dtype == torch.bfloat16:
        return {"rtol": 5e-2, "atol": 5e-2}
    return {"rtol": 1e-2, "atol": 1e-2}


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestFusedRMSNormRoPE:
    """Tests for fused RMSNorm + RoPE in a single megakernel."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_fused_rmsnorm_rope(self, dtype):
        """Test fused RMSNorm -> RoPE in single megakernel.

        This tests the core use case: normalizing query vectors then applying
        rotary position embedding, all in one kernel launch.

        Data flow:
            x (M, hidden_dim) -> RMSNorm -> y (M, hidden_dim)
            y viewed as (M, H, D) = q -> RoPE -> q (modified in-place)

        The dependency is automatically detected because y and q share the
        same tensor storage (y.data_ptr() == q.data_ptr()).

        Also verifies barrier reset by running the kernel multiple times.
        """
        batch, seq_len, n_heads, head_dim = 2, 16, 4, 64
        M = batch * seq_len
        hidden_dim = n_heads * head_dim

        # Input for RMSNorm
        x = torch.randn(M, hidden_dim, dtype=dtype, device="cuda")
        weight = torch.randn(hidden_dim, dtype=dtype, device="cuda")

        # Output of RMSNorm / Input for RoPE
        y = torch.empty_like(x)
        q = y.view(M, n_heads, head_dim)  # Alias sharing storage

        # RoPE tables
        cos = torch.randn(seq_len, head_dim // 2, dtype=dtype, device="cuda")
        sin = torch.randn(seq_len, head_dim // 2, dtype=dtype, device="cuda")

        # Fused megakernel: RMSNorm -> RoPE
        # Dependency is auto-detected via tensor pointer matching
        ops = [
            RMSNormOp.schedule(x=x, weight=weight, y=y),
            RopeOp.schedule(q=q, cos=cos, sin=sin),
        ]
        config = MegakernelConfig(threads_per_block=128)
        kernel = Megakernel(ops, config=config)

        # Run multiple times to verify barrier reset
        for _ in range(3):
            y.zero_()
            x.copy_(torch.randn_like(x))

            # PyTorch reference
            y_ref = rmsnorm_pytorch(x, weight)
            q_ref_4d = rope_pytorch(y_ref.view(batch, seq_len, n_heads, head_dim), cos, sin)
            q_ref = q_ref_4d.view(M, n_heads, head_dim)

            kernel.run()

            torch.testing.assert_close(q, q_ref, **get_tolerances(dtype), msg="Fused RMSNorm+RoPE output mismatch")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
