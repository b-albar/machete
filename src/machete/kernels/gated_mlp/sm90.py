# Copyright (c) 2025, Machete Authors
"""
Gated MLP kernel for SM90+ using Quack GEMM with fused activation.

This kernel computes: output = activation(x @ W_gate) * (x @ W_up)
Where W is (K, 2*N) with gate/up columns interleaved: [g0, u0, g1, u1, ...]

Uses Quack's high-performance GEMM implementation with fused epilogue activation.
"""

import torch
from torch import Tensor
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr, Int32
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from typing import Tuple, Optional, Dict, Any
from functools import partial

import quack.activation as qact
import quack.gemm_act
import quack.gemm_sm90
import quack.gemm_sm100
import quack.gemm_default_epi
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from quack.gemm_wrapper_utils import GemmWrapperBase


# =============================================================================
# Activation Function Adapters for Quack GEMM
# =============================================================================

@dsl_user_op
def swiglu_tuple_dupe(args, *, loc=None, ip=None):
    """Adapter for SwiGLU activation in Quack GEMM epilogue."""
    res = qact.swiglu(args[0], args[1])
    return res, res


@dsl_user_op
def geglu_tuple_dupe(args, *, loc=None, ip=None):
    """Adapter for GeGLU activation in Quack GEMM epilogue."""
    res = qact.geglu(args[0], args[1])
    return res, res


@dsl_user_op
def swiglu_bwd_tuple(h_pair, dout_pair, *, loc=None, ip=None):
    """Backward adapter for SwiGLU: takes (gate, up) and (dout, dout) pairs."""
    res = qact.dswiglu(h_pair[0], h_pair[1], dout_pair[0])
    return res[0], res[1]


@dsl_user_op
def geglu_bwd_tuple(h_pair, dout_pair, *, loc=None, ip=None):
    """Backward adapter for GeGLU: takes (gate, up) and (dout, dout) pairs."""
    res = qact.dgeglu(h_pair[0], h_pair[1], dout_pair[0])
    return res[0], res[1]


# =============================================================================
# Quack GEMM with Gated Activation Mixin
# =============================================================================

class MacheteGemmActMixin(quack.gemm_act.GemmActMixin):
    """Custom GEMM epilogue mixin for gated activations."""

    @cute.jit
    def epi_visit_subtile(
        self,
        params: quack.gemm_act.GemmActMixin.EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        trs_rd: cute.Tensor,
        trs_rc: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        # Call GemmDefaultEpiMixin only if not doing a fused backward
        if const_expr(trs_rc is None):
            quack.gemm_default_epi.GemmDefaultEpiMixin.epi_visit_subtile(
                self, params, epi_loop_tensors, trs_rd, trs_rc
            )

        # Apply activation function if provided
        if const_expr(params.act_fn is not None):
            trs_rpostact = cute.make_fragment(trs_rd.layout.shape, self.acc_dtype)
            for i in cutlass.range(cute.size(trs_rpostact) // 2, unroll_full=True):
                if const_expr(trs_rc is not None):
                    # Fused backward pass
                    trs_rpostact[2 * i], trs_rpostact[2 * i + 1] = params.act_fn(
                        (trs_rd[2 * i], trs_rd[2 * i + 1]),
                        (trs_rc[2 * i], trs_rc[2 * i + 1])
                    )
                else:
                    # Normal forward pass
                    trs_rpostact[2 * i], trs_rpostact[2 * i + 1] = params.act_fn(
                        (trs_rd[2 * i], trs_rd[2 * i + 1])
                    )
        else:
            trs_rpostact = trs_rd

        # Type conversion
        trs_rpostact_out = cute.make_fragment_like(trs_rpostact, self.postact_dtype)
        trs_rpostact_out.store(trs_rpostact.load().to(self.postact_dtype))
        return trs_rpostact_out


class MacheteGemmActSm90(MacheteGemmActMixin, quack.gemm_sm90.GemmSm90):
    """SM90 GEMM with gated activation."""
    pass


class MacheteGemmActSm100(MacheteGemmActMixin, quack.gemm_sm100.GemmSm100):
    """SM100 GEMM with gated activation."""
    pass


# Activation function map
ACT_FN_MAP = dict(quack.gemm_act.act_fn_map)
ACT_FN_MAP["swiglu"] = swiglu_tuple_dupe
ACT_FN_MAP["geglu"] = geglu_tuple_dupe
ACT_FN_MAP["swiglu_bwd"] = swiglu_bwd_tuple
ACT_FN_MAP["geglu_bwd"] = geglu_bwd_tuple


# =============================================================================
# Low-level GEMM Launch Function
# =============================================================================

# Global compile cache for machete_gemm_act
_gemm_compile_cache: Dict[Any, Any] = {}


def machete_gemm_act(
    A: torch.Tensor,
    B: torch.Tensor,
    D: Optional[torch.Tensor],
    C: Optional[torch.Tensor],
    PostAct: torch.Tensor,
    tile_count_semaphore: Optional[torch.Tensor],
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = False,
    persistent: bool = True,
    max_swizzle_size: int = 8,
    rowvec_bias: Optional[torch.Tensor] = None,
    colvec_bias: Optional[torch.Tensor] = None,
    cu_seqlens_m: Optional[torch.Tensor] = None,
    A_idx: Optional[torch.Tensor] = None,
) -> None:
    """
    Launch fused GEMM + gated activation kernel.

    Args:
        A: Input tensor (l, m, k)
        B: Weight tensor (l, n, k)
        D: Optional D tensor for residual
        C: Optional C tensor for backward
        PostAct: Output tensor (l, m, n)
        tile_count_semaphore: Optional semaphore for persistent kernels
        activation: Activation type ("swiglu" or "geglu")
        tile_M, tile_N: Tile sizes
        cluster_M, cluster_N: Cluster sizes
        pingpong: Enable pingpong scheduling
        persistent: Enable persistent kernel
        max_swizzle_size: Maximum swizzle size
        rowvec_bias, colvec_bias: Optional bias tensors
        cu_seqlens_m: Cumulative sequence lengths for varlen
        A_idx: Index tensor for gather
    """
    if cu_seqlens_m is not None:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        if D is not None:
            assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"

    gather_A = A_idx is not None
    if gather_A:
        assert cu_seqlens_m is not None, "gather_A requires varlen"
        assert cluster_N == 1, "gather_A requires cluster_N=1"

    assert activation in ACT_FN_MAP, f"Unsupported activation {activation}"

    L, M, K, N, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A, B, D, C, additional_tensors={"PostAct": PostAct},
        cu_seqlens_m=cu_seqlens_m, A_idx=A_idx
    )
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_m=cu_seqlens_m is not None)
    GemmWrapperBase.extract_dtypes(tensor_infos)
    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
        "PostAct": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)

    device_capacity = get_device_capacity(A.device)
    GemmCls = MacheteGemmActSm100 if device_capacity[0] > 9 else MacheteGemmActSm90

    acc_dtype = Float32
    tile_shape_mn = (tile_M, tile_N)
    cluster_shape_mnk = (cluster_M, cluster_N, 1)

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    GemmWrapperBase.create_cute_tensors(tensor_infos, major_configs)
    act_fn = ACT_FN_MAP[activation]

    epi_args = GemmCls.EpilogueArguments(
        tensor_infos["PostAct"].cute_tensor,
        act_fn,
        mRowVecBroadcast=from_dlpack(rowvec_bias.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=1)
        if rowvec_bias is not None else None,
        mColVecBroadcast=from_dlpack(colvec_bias.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=1 if cu_seqlens_m is None else 0
        ) if colvec_bias is not None else None,
    )

    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters, tile_count_semaphore, max_swizzle_size=max_swizzle_size
    )

    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m,
        None,  # cu_seqlens_k
        A_idx,
        max_active_clusters,
        cluster_shape_mnk,
        tensor_infos,
        GemmCls.num_epi_tensormaps,
        pingpong,
    )

    current_stream = cutlass_torch.current_stream()
    compile_key = GemmWrapperBase.get_compile_key(
        tensor_infos,
        activation,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        tile_count_semaphore is not None,
        device_capacity,
        max_swizzle_size,
        rowvec_bias.dtype if rowvec_bias is not None else None,
        colvec_bias.dtype if colvec_bias is not None else None,
        cu_seqlens_m is not None,
        A_idx is not None,
        key_tensor_names=("A", "B", "D", "PostAct", "C"),
    )

    if compile_key not in _gemm_compile_cache:
        if device_capacity[0] == 9:
            GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent)
        gemm_obj = GemmCls(
            acc_dtype,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            gather_A=gather_A,
        )
        _gemm_compile_cache[compile_key] = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
            current_stream,
        )

    _gemm_compile_cache[compile_key](
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,
        tensor_infos["C"].cute_tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
    )


# =============================================================================
# Backward Kernel Classes
# =============================================================================

class SwigluBwdSM90:
    """SwiGLU backward pass kernel for SM90."""
    _instances: Dict[torch.dtype, "SwigluBwdSM90"] = {}

    def __init__(self, dtype: torch.dtype):
        self.torch_dtype = dtype
        self.cute_dtype = torch2cute_dtype_map[dtype]
        self._compile_cache: Dict[Tuple, Any] = {}

    @staticmethod
    def get_instance(dtype: torch.dtype) -> "SwigluBwdSM90":
        if dtype not in SwigluBwdSM90._instances:
            SwigluBwdSM90._instances[dtype] = SwigluBwdSM90(dtype)
        return SwigluBwdSM90._instances[dtype]

    def __call__(self, h_gate, h_up, dout):
        M, N = h_gate.shape
        h_gate_stride = tuple(h_gate.stride())
        h_up_stride = tuple(h_up.stride())
        dout_stride = tuple(dout.stride())

        key = (N, h_gate_stride, h_up_stride, dout_stride)

        if key not in self._compile_cache:
            m_sym = cute.sym_int()
            t_g = cute.runtime.make_fake_tensor(self.cute_dtype, (m_sym, N), stride=h_gate_stride)
            t_u = cute.runtime.make_fake_tensor(self.cute_dtype, (m_sym, N), stride=h_up_stride)
            t_d = cute.runtime.make_fake_tensor(self.cute_dtype, (m_sym, N), stride=dout_stride)
            out_stride = (N, 1)
            t_dg = cute.runtime.make_fake_tensor(self.cute_dtype, (m_sym, N), stride=out_stride)
            t_du = cute.runtime.make_fake_tensor(self.cute_dtype, (m_sym, N), stride=out_stride)

            self._compile_cache[key] = cute.compile(
                self.run,
                t_g, t_u, t_d, t_dg, t_du,
                Int32(N),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )

        dg = torch.empty((M, N), dtype=self.torch_dtype, device=h_gate.device)
        du = torch.empty((M, N), dtype=self.torch_dtype, device=h_gate.device)

        self._compile_cache[key](h_gate, h_up, dout, dg, du, Int32(N))
        return dg, du

    @cute.jit
    def run(self, h_gate, h_up, dout, dg, du, n_cols, stream):
        grid = [h_gate.shape[0], 1, 1]
        block = [128, 1, 1]
        self.kernel(h_gate, h_up, dout, dg, du, n_cols).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(self, h_gate, h_up, dout, dg, du, n_cols):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()

        row = bidx
        for col in range(tidx, n_cols, num_threads):
            g = h_gate[row, col].to(Float32)
            u = h_up[row, col].to(Float32)
            d = dout[row, col].to(Float32)
            dg_val, du_val, _ = qact.dswiglu(g, u, d)
            dg[row, col] = dg_val.to(dg.element_type)
            du[row, col] = du_val.to(du.element_type)


class GegluBwdSM90:
    """GeGLU backward pass kernel for SM90."""
    _instances: Dict[torch.dtype, "GegluBwdSM90"] = {}

    def __init__(self, dtype: torch.dtype):
        self.torch_dtype = dtype
        self.cute_dtype = torch2cute_dtype_map[dtype]
        self._compile_cache: Dict[Tuple, Any] = {}

    @staticmethod
    def get_instance(dtype: torch.dtype) -> "GegluBwdSM90":
        if dtype not in GegluBwdSM90._instances:
            GegluBwdSM90._instances[dtype] = GegluBwdSM90(dtype)
        return GegluBwdSM90._instances[dtype]

    def __call__(self, h_gate, h_up, dout):
        M, N = h_gate.shape
        h_gate_stride = tuple(h_gate.stride())
        h_up_stride = tuple(h_up.stride())
        dout_stride = tuple(dout.stride())

        key = (N, h_gate_stride, h_up_stride, dout_stride)

        if key not in self._compile_cache:
            m_sym = cute.sym_int()
            t_g = cute.runtime.make_fake_tensor(self.cute_dtype, (m_sym, N), stride=h_gate_stride)
            t_u = cute.runtime.make_fake_tensor(self.cute_dtype, (m_sym, N), stride=h_up_stride)
            t_d = cute.runtime.make_fake_tensor(self.cute_dtype, (m_sym, N), stride=dout_stride)
            out_stride = (N, 1)
            t_dg = cute.runtime.make_fake_tensor(self.cute_dtype, (m_sym, N), stride=out_stride)
            t_du = cute.runtime.make_fake_tensor(self.cute_dtype, (m_sym, N), stride=out_stride)

            self._compile_cache[key] = cute.compile(
                self.run,
                t_g, t_u, t_d, t_dg, t_du,
                Int32(N),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )

        dg = torch.empty((M, N), dtype=self.torch_dtype, device=h_gate.device)
        du = torch.empty((M, N), dtype=self.torch_dtype, device=h_gate.device)

        self._compile_cache[key](h_gate, h_up, dout, dg, du, Int32(N))
        return dg, du

    @cute.jit
    def run(self, h_gate, h_up, dout, dg, du, n_cols, stream):
        grid = [h_gate.shape[0], 1, 1]
        block = [128, 1, 1]
        self.kernel(h_gate, h_up, dout, dg, du, n_cols).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(self, h_gate, h_up, dout, dg, du, n_cols):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()

        row = bidx
        for col in range(tidx, n_cols, num_threads):
            g = h_gate[row, col].to(Float32)
            u = h_up[row, col].to(Float32)
            d = dout[row, col].to(Float32)
            dg_val, du_val, _ = qact.dgeglu(g, u, d)
            dg[row, col] = dg_val.to(dg.element_type)
            du[row, col] = du_val.to(du.element_type)


# =============================================================================
# GatedMLP SM90 Wrapper Class
# =============================================================================

class GatedMLPSM90:
    """
    Gated MLP kernel for SM90+ with Quack GEMM backend.

    Computes: output = activation(x @ W_gate) * (x @ W_up)
    Where W is (K, 2*N) with interleaved columns [g0, u0, g1, u1, ...].
    """

    def __init__(self, dtype: torch.dtype, act_type: str = "silu"):
        """
        Initialize the GatedMLP SM90 kernel.

        Args:
            dtype: Input/output data type
            act_type: Activation type ("silu" for SwiGLU, "gelu" for GeGLU)
        """
        self.torch_dtype = dtype
        self.act_type = act_type
        # Map external act_type to internal naming
        self._act_type_map = {"silu": "swiglu", "gelu": "geglu"}
        self._internal_act = self._act_type_map.get(act_type, act_type)

    def forward(self, x: Tensor, weight: Tensor, chunk_size: int = 1024) -> Tensor:
        """
        Forward pass for Gated MLP.

        Args:
            x: Input tensor of shape (..., K)
            weight: Weight tensor of shape (K, 2*N) with interleaved gate/up columns
            chunk_size: Chunk size for memory-efficient processing

        Returns:
            Output tensor of shape (..., N)
        """
        x_orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        m_dim, k_dim = x_2d.shape
        k2_dim, n2_dim = weight.shape
        n_dim = n2_dim // 2

        assert k_dim == k2_dim, f"Input dim {k_dim} != weight dim {k2_dim}"
        assert n2_dim % 2 == 0, "Weight N dimension must be even"

        if not x.is_cuda:
            raise ValueError("Input must be on CUDA")

        # Final output buffer (M, N)
        res = torch.empty((*x_orig_shape[:-1], n_dim), dtype=x.dtype, device=x.device)
        res_2d = res.reshape(m_dim, n_dim)

        # Prepare weight: (l, n, k) view from weight (K, 2N)
        weight_B = weight.unsqueeze(0).permute(0, 2, 1)

        # Pre-allocate workspace
        eff_chunk_size = min(chunk_size, m_dim)
        workspace = torch.empty((eff_chunk_size, n2_dim), dtype=x.dtype, device=x.device)

        # Launch kernel in chunks
        for i in range(0, m_dim, eff_chunk_size):
            actual_chunk_size = min(eff_chunk_size, m_dim - i)
            curr_workspace = workspace[:actual_chunk_size]

            machete_gemm_act(
                x_2d[i:i + actual_chunk_size].reshape(1, actual_chunk_size, k_dim),
                weight_B,
                None,  # D
                None,  # C
                curr_workspace.reshape(1, actual_chunk_size, n2_dim),
                None,  # tile_count_semaphore
                self._internal_act,
                tile_M=128,
                tile_N=128,
                cluster_M=1,
                cluster_N=1,
            )

            # Extract result (every other column due to interleaving)
            res_2d[i:i + actual_chunk_size].copy_(curr_workspace[:, ::2])

        return res

    def backward(self, dout: Tensor, x: Tensor, weight: Tensor, chunk_size: int = 4096) -> Tuple[Tensor, Tensor]:
        """
        Backward pass for Gated MLP.

        Args:
            dout: Gradient of output (..., N)
            x: Original input (..., K)
            weight: Weight tensor (K, 2*N)
            chunk_size: Chunk size for memory-efficient processing

        Returns:
            Tuple of (dx, dweight)
        """
        x_orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        m_dim, k_dim = x_2d.shape
        n2_dim = weight.shape[1]
        n_dim = n2_dim // 2
        dout_2d = dout.reshape(-1, n_dim)

        dx = torch.empty_like(x_2d)
        dweight = torch.zeros_like(weight)

        # Pre-slice weight for efficiency
        w_gate = weight[:, ::2]
        w_up = weight[:, 1::2]
        w_gate_t = w_gate.t()
        w_up_t = w_up.t()

        # Get backward kernel instance
        if self._internal_act == "swiglu":
            bwd_kernel = SwigluBwdSM90.get_instance(self.torch_dtype)
        else:
            bwd_kernel = GegluBwdSM90.get_instance(self.torch_dtype)

        for i in range(0, m_dim, chunk_size):
            actual_chunk_size = min(chunk_size, m_dim - i)
            x_chunk = x_2d[i:i + actual_chunk_size]
            dout_chunk = dout_2d[i:i + actual_chunk_size]

            # Recompute hidden states: h = x @ weight
            h_chunk = x_chunk @ weight
            h_gate = h_chunk[:, ::2]
            h_up = h_chunk[:, 1::2]

            # Compute gradients for gate and up
            dg, du = bwd_kernel(h_gate, h_up, dout_chunk)

            # dx = dg @ w_gate.T + du @ w_up.T
            dx_chunk = dg @ w_gate_t
            dx_chunk.addmm_(du, w_up_t)
            dx[i:i + actual_chunk_size] = dx_chunk

            # Accumulate dweight
            dweight[:, ::2].addmm_(x_chunk.t(), dg)
            dweight[:, 1::2].addmm_(x_chunk.t(), du)

        return dx.reshape(x_orig_shape), dweight

    def __call__(self, x: Tensor, weight: Tensor) -> Tensor:
        """Convenience call for forward pass."""
        return self.forward(x, weight)


# =============================================================================
# Autograd Function
# =============================================================================

class GatedMLPSM90Func(torch.autograd.Function):
    """Autograd function for GatedMLP SM90."""

    @staticmethod
    def forward(ctx, x, weight, act_type="swiglu"):
        ctx.save_for_backward(x, weight)
        ctx.act_type = act_type
        kernel = GatedMLPSM90(x.dtype, act_type)
        return kernel.forward(x, weight)

    @staticmethod
    def backward(ctx, dout):
        x, weight = ctx.saved_tensors
        act_type = ctx.act_type
        kernel = GatedMLPSM90(x.dtype, act_type)
        dx, dweight = kernel.backward(dout, x, weight)
        return dx, dweight, None


# =============================================================================
# Convenience Functions
# =============================================================================

def gated_mlp_sm90(x: Tensor, weight: Tensor, act_type: str = "swiglu") -> Tensor:
    """
    Gated MLP forward pass for SM90+.

    Args:
        x: Input tensor of shape (..., K)
        weight: Weight tensor of shape (K, 2*N)
        act_type: Activation type ("swiglu" or "geglu")

    Returns:
        Output tensor of shape (..., N)
    """
    return GatedMLPSM90Func.apply(x, weight, act_type)


def clear_kernel_cache():
    """Clear all SM90 kernel caches."""
    _gemm_compile_cache.clear()
    SwigluBwdSM90._instances.clear()
    GegluBwdSM90._instances.clear()
