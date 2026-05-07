# Copyright (c) 2025, Machete Authors
"""
Tests for Template-Based Megakernel.

Tests the megakernel that uses instruction stream and fine-grained barriers.
"""

import pytest
import torch

from machete.kernels.attention import FlashAttentionSm120Op
from machete.kernels.gemm import GemmOp, GemmSm100Op
from machete.kernels.glu import GLUBwdOp, GLUOp
from machete.kernels.qknorm_rope import QKNormRopeOp
from machete.kernels.rms_norm import RMSNormOp
from machete.megakernel.backend import _phase_should_noinline
from tests.megakernel.support import get_nop_op
from tests.megakernel.support_tma import (
    SYNTHETIC_TMA_N,
    SYNTHETIC_TMA_TILE_M,
    SyntheticTMAAddOneOp,
)
from machete.megakernel.backend_dispatch import TMARuntimeLayout, _build_tma_runtime_layout
from machete.megakernel.ops import Op


class TestMegakernel:
    """Test the template megakernel (now uses instruction stream)."""

    def test_megakernel_creation(self):
        """Test creating a megakernel instance."""
        from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp
        NopOp = get_nop_op()

        # Define some operations
        ops = [
            ScheduledOp(NopOp, tile_counts=(32,)),
            ScheduledOp(NopOp, tile_counts=(16,)),
        ]

        config = MegakernelConfig(num_sms=8)
        kernel = Megakernel(ops, config=config, device="cpu")

        # New API: total_tiles instead of total_blocks, grid is based on num_sms
        assert kernel.total_tiles == 48
        assert kernel.grid == (8, 1, 1)  # Now based on num_sms (persistent blocks)
        assert kernel.block == (256, 1, 1)

    def test_backend_does_not_duplicate_identical_handlers(self):
        """Repeated identical ops should share one emitted handler body."""
        from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp
        NopOp = get_nop_op()

        ops = [
            ScheduledOp(NopOp, tile_counts=(32,)),
            ScheduledOp(NopOp, tile_counts=(16,)),
            ScheduledOp(NopOp, tile_counts=(8,)),
        ]

        kernel = Megakernel(
            ops,
            config=MegakernelConfig(num_sms=1),
            device="cpu",
        )

        assert len(kernel._backend_ir.handler_specs) == 1
        assert list(kernel._backend_ir.op_handler_indices) == [0, 0, 0]

    def test_identical_tma_layers_share_handlers_but_keep_selector_metadata(self):
        """Repeated TMA layers should share handlers while preserving per-op bindings."""
        from machete.megakernel import Megakernel, MegakernelConfig

        def _make_kernel(num_layers: int):
            ops = []
            for _ in range(num_layers):
                x = torch.zeros(SYNTHETIC_TMA_TILE_M, SYNTHETIC_TMA_N, dtype=torch.float16)
                y = torch.zeros(SYNTHETIC_TMA_TILE_M, SYNTHETIC_TMA_N, dtype=torch.float16)
                ops.extend(
                    SyntheticTMAAddOneOp.schedule(
                        x=x,
                        y=y,
                        tile_sizes={"M": SYNTHETIC_TMA_TILE_M},
                    )
                )
            kernel = Megakernel(
                ops,
                config=MegakernelConfig(num_sms=1),
                device="cpu",
            )
            kernel._prepare_tensors()
            return kernel

        k1 = _make_kernel(1)
        k32 = _make_kernel(32)

        assert len(k1._backend_ir.handler_specs) == 1
        assert len(k32._backend_ir.handler_specs) == 1
        assert k1._phase_local_transport_position_widths["load"] <= k32._phase_local_transport_position_widths["load"]
        assert k1._phase_local_transport_position_widths["store"] <= k32._phase_local_transport_position_widths["store"]
        assert k1._phase_local_desc_slot_widths["load"] <= k32._phase_local_desc_slot_widths["load"]
        assert k1._phase_local_desc_slot_widths["store"] <= k32._phase_local_desc_slot_widths["store"]

        def _numel_or_zero(t):
            return 0 if t is None else t.numel()

        assert _numel_or_zero(k1._phase_local_transport_position_tensors["load"]) < _numel_or_zero(
            k32._phase_local_transport_position_tensors["load"]
        )
        assert _numel_or_zero(k1._phase_local_transport_position_tensors["store"]) < _numel_or_zero(
            k32._phase_local_transport_position_tensors["store"]
        )
        assert _numel_or_zero(k1._phase_local_desc_slot_tensors["load"]) < _numel_or_zero(
            k32._phase_local_desc_slot_tensors["load"]
        )
        assert _numel_or_zero(k1._phase_local_desc_slot_tensors["store"]) < _numel_or_zero(
            k32._phase_local_desc_slot_tensors["store"]
        )

    def test_tma_runtime_layout_is_cached_per_backend(self):
        """Structural TMA runtime layout should be built once per backend instance."""
        from machete.megakernel import Megakernel, MegakernelConfig

        x = torch.zeros(SYNTHETIC_TMA_TILE_M, SYNTHETIC_TMA_N, dtype=torch.float16)
        y = torch.zeros(SYNTHETIC_TMA_TILE_M, SYNTHETIC_TMA_N, dtype=torch.float16)
        ops = SyntheticTMAAddOneOp.schedule(
            x=x,
            y=y,
            tile_sizes={"M": SYNTHETIC_TMA_TILE_M},
        )
        kernel = Megakernel(
            ops,
            config=MegakernelConfig(num_sms=1),
            device="cpu",
        )

        layout0 = _build_tma_runtime_layout(kernel._backend, kernel)
        layout1 = _build_tma_runtime_layout(kernel._backend, kernel)

        assert layout0 is layout1
        assert isinstance(layout0, TMARuntimeLayout)
        assert layout0.op_phase_tma_args["load"]
        assert len(kernel._tma_runtime_layout_cache) == 1

    def test_qwen_forward_ops_expose_sequence_as_dynamic_dim(self):
        """Sequence dimensions should be runtime-packed, not compile-time baked."""
        assert RMSNormOp.dynamic_dims == ("B", "S")
        assert GemmOp.dynamic_dims == ("B", "S")
        assert GLUOp.dynamic_dims == ("B", "S")
        assert QKNormRopeOp.dynamic_dims == ("M",)
        assert FlashAttentionSm120Op.dynamic_dims == ("B", "M", "N")

    def test_phase_inline_policy_is_op_owned_and_defaults_to_noinline(self):
        """Phase outlining defaults to noinline unless an op opts in."""
        class DefaultPolicyOp(Op):
            pass

        class InlineComputeOp(Op):
            inline_phases = ("compute",)

        assert _phase_should_noinline(DefaultPolicyOp(), "compute")
        assert not _phase_should_noinline(InlineComputeOp(), "compute")
        assert _phase_should_noinline(InlineComputeOp(), "load")
        assert _phase_should_noinline(
            InlineComputeOp(),
            "compute",
            inline_thin_phases=False,
        )

    def test_migrated_thin_phase_policies(self):
        """Ops that used backend allowlists now declare their own inline phases."""
        all_thin = ("load", "compute", "store")
        assert GemmOp.inline_phases == all_thin
        assert GemmSm100Op.inline_phases == all_thin
        assert GLUOp.inline_phases == all_thin
        assert FlashAttentionSm120Op.inline_phases == all_thin
        assert QKNormRopeOp.inline_phases == all_thin
        assert RMSNormOp.inline_phases == all_thin
        assert GLUBwdOp.inline_phases == ("compute",)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_validation_check(self):
        """Test that validation checks for Hopper+ Architecture."""
        from machete.megakernel import Megakernel, ScheduledOp
        NopOp = get_nop_op()

        ops = [ScheduledOp(NopOp, tile_counts=(1,))]
        kernel = Megakernel(ops)

        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            with pytest.raises(RuntimeError, match="requires Hopper"):
                kernel.run()
        else:
            # Should run (or fail with launch error if env is unstable, but NOT validation error)
            try:
                kernel.run()
            except Exception as e:
                # If it's a launch error, we passed the validation check (it tried to run)
                if "requires Hopper" in str(e):
                    raise e
                print(f"Kernel tried to run but failed (expected on some envs): {e}")
