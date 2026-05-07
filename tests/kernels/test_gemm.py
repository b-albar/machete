# Copyright (c) 2025, Machete Authors
"""Tests for GemmOp — matmul correctness for fp16 and bf16.

Tests run on GPU (SM_90+) and compare the megakernel GemmOp against
torch.matmul with fp32 accumulation as reference.

GemmOp tiles on (M, N) with an inner K loop in compute. K reduction
uses fp32 accumulation across all K blocks, followed by a single
regular TMA store of the final C tile. No output pre-zeroing needed.
"""

import contextlib
import io

import torch
import pytest

from tests.kernels.support import requires_sm90_cutlass


# =============================================================================
# Helpers
# =============================================================================


def _gemm_reference(a, b_t):
    """Reference GEMM: C = A @ B_T^T = A @ B, computed in fp32."""
    # a: (M, K), b_t: (N, K) → C = a @ b_t.T → (M, N)
    return (a.float() @ b_t.float().t()).to(a.dtype)


def _run_gemm(a, b_t, tile_m=64, tile_n=32, tile_k=32):
    """Run GemmOp and return output tensor C.

    Inputs are 2D (M, K)/(N, K), auto-wrapped to 3D (1, M, K) for GemmOp.
    """
    from machete.megakernel import Megakernel
    from machete.kernels.gemm import GemmOp

    M, K = a.shape
    N = b_t.shape[0]
    c = torch.zeros(1, M, N, dtype=a.dtype, device=a.device)

    ops = GemmOp.schedule(
        a=a.unsqueeze(0), b=b_t, c=c,
        tile_sizes={"S": tile_m, "N": tile_n, "K": tile_k},
    )
    config = GemmOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    return c.squeeze(0)


def _run_gemm_batched(a_3d, b_t, tile_sizes=None, page_size=32768):
    """Run GemmOp on a batched 3D input and return output tensor C."""
    from machete.megakernel import Megakernel
    from machete.kernels.gemm import GemmOp

    B, M, K = a_3d.shape
    N = b_t.shape[0]
    c = torch.zeros(B, M, N, dtype=a_3d.dtype, device=a_3d.device)

    ops = GemmOp.schedule(
        a=a_3d, b=b_t, c=c,
        tile_sizes=tile_sizes,
        page_size=page_size,
    )
    config = GemmOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    return c


# =============================================================================
# Tests
# =============================================================================


def _gemm_case(M, K, N, dtype, tile_m=64, tile_n=32, tile_k=32,
               atol=1e-1, rtol=1e-2):
    """Run a single GEMM test case and assert correctness."""
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(K, N, dtype=dtype, device="cuda")
    b_t = b.t().contiguous()

    c = _run_gemm(a, b_t, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
    ref = _gemm_reference(a, b_t)

    # fp32 accumulation across all K blocks — precision is constant
    # regardless of number of K tiles.
    torch.testing.assert_close(c, ref, atol=atol, rtol=rtol)


FORWARD_CASES = [
    pytest.param(
        dict(M=64, K=32, N=32),
        id="single_k_tile",
    ),
    pytest.param(
        dict(M=128, K=128, N=64),
        id="multi_k_tiles",
    ),
    pytest.param(
        dict(M=64, K=512, N=32),
        id="many_k_tiles",
    ),
    pytest.param(
        dict(M=100, K=80, N=48),
        id="non_divisible_all_dims",
    ),
    pytest.param(
        dict(M=256, K=128, N=128),
        id="multi_mn_tiles",
    ),
    pytest.param(
        dict(M=64, K=16, N=32, tile_k=16),
        id="minimum_tile_k",
    ),
    pytest.param(
        dict(M=16, K=64, N=32),
        id="small_m_boundary",
    ),
    pytest.param(
        dict(M=64, K=64, N=16),
        id="small_n_boundary",
    ),
    pytest.param(
        dict(M=64, K=128, N=32, tile_k=64),
        id="large_tile_k",
    ),
    pytest.param(
        dict(M=64, K=64, N=32, tile_k=16),
        id="small_tile_k",
    ),
    pytest.param(
        dict(M=128, K=4096, N=16384),
        id="large_gemm",
    ),
]


def test_shape_aware_auto_tiles_long_rows_use_wide_spatial_tile():
    """Long-row BF16 GEMMs on 32KB pages prefer fewer, wider spatial tiles."""
    from machete.kernels.gemm import GemmOp

    assert GemmOp._shape_aware_auto_tiles(
        32768,
        input_k=1024,
        output_n=151936,
        rows=1024,
        elem_bytes=2,
        has_a_scale=False,
    ) == (128, 128, 16)


@requires_sm90_cutlass
class TestGemmForward:
    """GEMM forward pass correctness tests."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("case", FORWARD_CASES)
    def test_forward_matrix(self, dtype, case):
        """Representative forward coverage without redundant shape permutations."""
        _gemm_case(dtype=dtype, **case)

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_auto_tiles_large_batch_compile_and_run(self, dtype):
        """Auto-tiled large batched GEMM compiles and matches PyTorch."""
        torch.manual_seed(42)
        B, M, K, N = 8, 128, 4096, 4096

        a = torch.randn(B, M, K, dtype=dtype, device="cuda")
        b = torch.randn(K, N, dtype=dtype, device="cuda")
        b_t = b.t().contiguous()

        c = _run_gemm_batched(a, b_t, tile_sizes=None, page_size=32768)
        ref = torch.matmul(a.float(), b.float()).to(dtype)

        torch.testing.assert_close(c, ref, atol=2e-1, rtol=2e-2)



# =============================================================================
# Backward Tests
# =============================================================================


def _run_gemm_backward(dout, a, b, da=None, db=None,
                       tile_m=64, tile_n=32, tile_k=32):
    """Run GemmOp backward and return (da, db).

    Inputs are 2D (M, K)/(N, K), auto-wrapped to 3D (1, M, K) for GemmOp.
    Both dA and dB ops go into the same megakernel. They both produce
    buffer 'c' but with different data_ptr, so the framework treats
    them as independent.
    """
    from machete.megakernel import Megakernel
    from machete.kernels.gemm import GemmOp

    ts = {"S": tile_m, "N": tile_n, "K": tile_k}

    # Wrap 2D → 3D with B=1
    dout_3d = dout.unsqueeze(0)
    a_3d = a.unsqueeze(0)
    da_3d = da.unsqueeze(0) if da is not None else None
    db_3d = db.unsqueeze(0) if db is not None else None

    ops = GemmOp.schedule_backward(dout=dout_3d, a=a_3d, b=b, da=da_3d, db=db_3d,
                                   tile_sizes=ts)
    if ops:
        config = GemmOp.kernel_config(ops)
        with contextlib.redirect_stdout(io.StringIO()):
            Megakernel(ops, config=config).run()

    return da, db


def _gemm_backward_case(M, K, N, dtype, tile_m=64, tile_n=32, tile_k=32,
                        compute_da=True, compute_db=True,
                        atol=1e-1, rtol=1e-2):
    """Run a single GEMM backward test case and assert correctness."""
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(N, K, dtype=dtype, device="cuda")  # (N, K) layout
    dout = torch.randn(M, N, dtype=dtype, device="cuda")

    da = torch.zeros(M, K, dtype=dtype, device="cuda") if compute_da else None
    db = torch.zeros(N, K, dtype=dtype, device="cuda") if compute_db else None

    _run_gemm_backward(dout, a, b, da=da, db=db,
                       tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)

    # fp32 accumulation — precision is constant regardless of K tiles.
    if da is not None:
        da_ref = (dout.float() @ b.float()).to(dtype)
        torch.testing.assert_close(da, da_ref, atol=atol, rtol=rtol)

    if db is not None:
        db_ref = (dout.float().t() @ a.float()).to(dtype)
        torch.testing.assert_close(db, db_ref, atol=atol, rtol=rtol)


@requires_sm90_cutlass
class TestGemmBackward:
    """GEMM backward pass correctness tests."""

    # ----- dA only -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward_da_basic(self, dtype):
        """dA only, single K tile (N=32, tile_N=32)."""
        _gemm_backward_case(64, 32, 32, dtype, compute_db=False)

    # ----- dB only -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward_db_basic(self, dtype):
        """dB only, single K tile (M=64, tile_M=64)."""
        _gemm_backward_case(64, 32, 32, dtype, compute_da=False)

    # ----- Both dA + dB -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward_both(self, dtype):
        """Both dA and dB (separate megakernels, same buffer name 'c')."""
        _gemm_backward_case(64, 32, 32, dtype)

    # ----- Multiple tiles -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward_multi_tiles(self, dtype):
        """Larger shapes with multiple tiles along all dims."""
        _gemm_backward_case(128, 128, 64, dtype)

    # ----- Non-divisible -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward_non_divisible(self, dtype):
        """M, K, N all non-divisible by tile sizes.

        All dims must be multiples of 8 (TMA alignment for fp16/bf16),
        since backward remaps M→K and N→K for the two gradient GEMMs.
        """
        _gemm_backward_case(104, 80, 48, dtype)

    # ----- Large K (many K tiles) -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward_large_k(self, dtype):
        """Large K with many K tiles."""
        _gemm_backward_case(64, 256, 32, dtype)


# =============================================================================
# Fused GEMM + Activation
# =============================================================================


def _run_gemm_fused_act(a, b_t, activation, tile_m=64, tile_n=32, tile_k=32):
    """Run GemmOp with fused activation and return output tensor C.

    Inputs are 2D (M, K)/(N, K), auto-wrapped to 3D (1, M, K) for GemmOp.
    """
    from machete.megakernel import Megakernel
    from machete.kernels.gemm import GemmOp

    M, K = a.shape
    N = b_t.shape[0]
    c = torch.zeros(1, M, N, dtype=a.dtype, device=a.device)

    ops = GemmOp.schedule(
        a=a.unsqueeze(0), b=b_t, c=c, activation=activation,
        tile_sizes={"S": tile_m, "N": tile_n, "K": tile_k},
    )
    config = GemmOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    return c.squeeze(0)


@requires_sm90_cutlass
class TestGemmFusedActivation:
    """GEMM with fused epilogue activation."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fused_relu(self, dtype):
        """Fused GEMM + ReLU matches torch reference."""
        torch.manual_seed(42)
        M, K, N = 128, 128, 64
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b_t = torch.randn(N, K, dtype=dtype, device="cuda")

        c = _run_gemm_fused_act(a, b_t, activation='relu')
        ref = torch.relu((a.float() @ b_t.float().t())).to(dtype)
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fused_silu(self, dtype):
        """Fused GEMM + SiLU matches torch reference."""
        torch.manual_seed(42)
        M, K, N = 128, 128, 64
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b_t = torch.randn(N, K, dtype=dtype, device="cuda")

        c = _run_gemm_fused_act(a, b_t, activation='silu')
        ref = torch.nn.functional.silu((a.float() @ b_t.float().t())).to(dtype)
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


# =============================================================================
# Fused GEMM + Activation Backward
# =============================================================================


def _run_gemm_fused_act_backward(dout, a, b, activation, c=None, pre_act=None,
                                  da=None, db=None,
                                  tile_m=64, tile_n=32, tile_k=32):
    """Run GemmOp backward with fused activation and return (da, db).

    Inputs are 2D, auto-wrapped to 3D (1, M, K) for GemmOp.
    """
    from machete.megakernel import Megakernel
    from machete.kernels.gemm import GemmOp

    ts = {"S": tile_m, "N": tile_n, "K": tile_k}

    # Wrap 2D → 3D with B=1
    dout_3d = dout.unsqueeze(0)
    a_3d = a.unsqueeze(0)
    da_3d = da.unsqueeze(0) if da is not None else None
    db_3d = db.unsqueeze(0) if db is not None else None
    c_3d = c.unsqueeze(0) if c is not None else None
    pre_act_3d = pre_act.unsqueeze(0) if pre_act is not None else None

    ops = GemmOp.schedule_backward(dout=dout_3d, a=a_3d, b=b, da=da_3d, db=db_3d,
                                   activation=activation, c=c_3d, pre_act=pre_act_3d,
                                   tile_sizes=ts)
    if ops:
        config = GemmOp.kernel_config(ops)
        with contextlib.redirect_stdout(io.StringIO()):
            Megakernel(ops, config=config).run()

    return da, db


@requires_sm90_cutlass
class TestGemmFusedActivationBackward:
    """GEMM backward with fused activation gradient via A-operand scaling."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fused_relu_backward_da(self, dtype):
        """dA with fused ReLU backward: dA = (dout * relu'(pre_act)) @ B."""
        torch.manual_seed(42)
        M, K, N = 128, 128, 64
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b_t = torch.randn(N, K, dtype=dtype, device="cuda")
        dout = torch.randn(M, N, dtype=dtype, device="cuda")

        # Forward to get c = relu(A @ B^T)
        pre_act_f32 = a.float() @ b_t.float().t()
        c = torch.relu(pre_act_f32).to(dtype)

        da = torch.zeros(M, K, dtype=dtype, device="cuda")
        _run_gemm_fused_act_backward(dout, a, b_t, activation='relu',
                                      c=c, da=da)

        # Reference: dA = (dout * (c > 0)) @ B
        act_grad = (c > 0).float()
        da_ref = ((dout.float() * act_grad) @ b_t.float()).to(dtype)
        torch.testing.assert_close(da, da_ref, atol=1e-1, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fused_relu_backward_both(self, dtype):
        """Both dA and dB with fused ReLU backward."""
        torch.manual_seed(42)
        M, K, N = 128, 128, 64
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b_t = torch.randn(N, K, dtype=dtype, device="cuda")
        dout = torch.randn(M, N, dtype=dtype, device="cuda")

        pre_act_f32 = a.float() @ b_t.float().t()
        c = torch.relu(pre_act_f32).to(dtype)

        da = torch.zeros(M, K, dtype=dtype, device="cuda")
        db = torch.zeros(N, K, dtype=dtype, device="cuda")
        _run_gemm_fused_act_backward(dout, a, b_t, activation='relu',
                                      c=c, da=da, db=db)

        act_grad = (c > 0).float()
        dout_adj = dout.float() * act_grad
        da_ref = (dout_adj @ b_t.float()).to(dtype)
        db_ref = (dout_adj.t() @ a.float()).to(dtype)
        torch.testing.assert_close(da, da_ref, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(db, db_ref, atol=1e-1, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fused_silu_backward_da(self, dtype):
        """dA with fused SiLU backward: dA = (dout * silu'(pre_act)) @ B."""
        torch.manual_seed(42)
        M, K, N = 128, 128, 64
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b_t = torch.randn(N, K, dtype=dtype, device="cuda")
        dout = torch.randn(M, N, dtype=dtype, device="cuda")

        pre_act_f32 = a.float() @ b_t.float().t()
        pre_act = pre_act_f32.to(dtype)

        da = torch.zeros(M, K, dtype=dtype, device="cuda")
        _run_gemm_fused_act_backward(dout, a, b_t, activation='silu',
                                      pre_act=pre_act, da=da)

        # Reference: silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sig = torch.sigmoid(pre_act_f32)
        act_grad = sig * (1.0 + pre_act_f32 * (1.0 - sig))
        da_ref = ((dout.float() * act_grad) @ b_t.float()).to(dtype)
        torch.testing.assert_close(da, da_ref, atol=1e-1, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fused_silu_backward_both(self, dtype):
        """Both dA and dB with fused SiLU backward."""
        torch.manual_seed(42)
        M, K, N = 128, 128, 64
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b_t = torch.randn(N, K, dtype=dtype, device="cuda")
        dout = torch.randn(M, N, dtype=dtype, device="cuda")

        pre_act_f32 = a.float() @ b_t.float().t()
        pre_act = pre_act_f32.to(dtype)

        da = torch.zeros(M, K, dtype=dtype, device="cuda")
        db = torch.zeros(N, K, dtype=dtype, device="cuda")
        _run_gemm_fused_act_backward(dout, a, b_t, activation='silu',
                                      pre_act=pre_act, da=da, db=db)

        sig = torch.sigmoid(pre_act_f32)
        act_grad = sig * (1.0 + pre_act_f32 * (1.0 - sig))
        dout_adj = dout.float() * act_grad
        da_ref = (dout_adj @ b_t.float()).to(dtype)
        db_ref = (dout_adj.t() @ a.float()).to(dtype)
        torch.testing.assert_close(da, da_ref, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(db, db_ref, atol=1e-1, rtol=1e-2)
