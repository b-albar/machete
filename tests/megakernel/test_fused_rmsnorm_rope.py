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
from machete.utils.testing import is_hopper_available


PAGE_SIZE = 16384  # 16KB


def get_tolerances(dtype):
    """Get appropriate rtol/atol for dtype.

    bfloat16 has less precision than float32, so we use relaxed tolerances.
    """
    if dtype == torch.bfloat16:
        return {"rtol": 5e-2, "atol": 5e-2}
    return {"rtol": 1e-2, "atol": 1e-2}


def _compute_tile_sizes(n_heads, head_dim, dtype):
    """Compute tile sizes for both RMSNorm and RoPE that fit in PAGE_SIZE."""
    elem_bytes = 4 if dtype == torch.float32 else 2
    hidden_dim = n_heads * head_dim

    # RMSNorm: tile_m_rms * hidden_dim * elem_bytes <= PAGE_SIZE
    tile_m_rms = min(4, max(1, PAGE_SIZE // (hidden_dim * elem_bytes)))

    # RoPE tile_size_H: largest <= 8 that divides n_heads
    tile_h = min(n_heads, 8)
    while n_heads % tile_h != 0:
        tile_h -= 1

    # RoPE smem: q (tile_m * tile_h * D) + cos (tile_m * D/2) + sin (tile_m * D/2)
    rope_row_bytes = (tile_h * head_dim + head_dim) * elem_bytes
    tile_m_rope = min(4, max(1, PAGE_SIZE // rope_row_bytes))

    return tile_m_rms, tile_h, tile_m_rope


# (batch, seq_len, n_heads, head_dim)
SHAPE_PARAMS = [
    (1, 8, 1, 64),      # minimal: single head
    (2, 16, 4, 64),      # original test shape
    (1, 32, 8, 64),      # many heads
    (4, 16, 2, 128),     # larger head_dim
    (2, 64, 4, 64),      # longer sequence
    (1, 16, 16, 64),     # 16 heads
    (2, 8, 4, 128),      # 4 heads x 128 head_dim
    (1, 32, 2, 256),     # large head_dim (hidden=512)
    (4, 32, 8, 64),      # larger batch
    (1, 128, 4, 64),     # long sequence
    (2, 16, 32, 64),     # many heads (hidden=2048)
    (1, 8, 8, 128),      # 8 heads x 128 head_dim (hidden=1024)
    (1, 16, 20, 64),     # non-divisible tile_m: rms=3, rope=4 (fp32)
]


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestFusedRMSNormRoPE:
    """Tests for fused RMSNorm + RoPE in a single megakernel."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("batch,seq_len,n_heads,head_dim", SHAPE_PARAMS)
    def test_fused_rmsnorm_rope(self, batch, seq_len, n_heads, head_dim, dtype):
        """Test fused RMSNorm -> RoPE in single megakernel.

        Data flow:
            x (M, hidden_dim) -> RMSNorm -> y (M, hidden_dim)
            y viewed as (M, H, D) = q -> RoPE -> q (modified in-place)

        The dependency is automatically detected because y and q share the
        same tensor storage (y.data_ptr() == q.data_ptr()).

        Also verifies barrier reset by running the kernel multiple times.
        """
        M = batch * seq_len
        hidden_dim = n_heads * head_dim
        tile_m_rms, tile_h, tile_m_rope = _compute_tile_sizes(n_heads, head_dim, dtype)

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
            RMSNormOp.schedule(x=x, weight=weight, y=y, tile_sizes={"M": tile_m_rms}),
            RopeOp.schedule(q=q, cos=cos, sin=sin, tile_sizes={"M": tile_m_rope, "H": tile_h}),
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
