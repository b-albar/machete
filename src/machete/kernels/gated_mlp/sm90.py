# Copyright (c) 2025, Machete Authors
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from typing import Tuple, Optional
from functools import partial

import quack.activation as qact
import quack.gemm_act
import quack.gemm_sm90
import quack.gemm_sm100
import quack.gemm_default_epi
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
from quack.gemm_wrapper_utils import GemmWrapperBase


# Helper to adapter tuple input for SwiGLU
# The GemmActMixin in Quack passes `(tRS_rD[2 * i], tRS_rD[2 * i + 1])`.
# We need to return a tuple to match the assignment interactions in Quack's gemm_act.
@dsl_user_op
def swiglu_tuple_dupe(args, *, loc=None, ip=None):
    res = qact.swiglu(args[0], args[1])
    return res, res


@dsl_user_op
def geglu_tuple_dupe(args, *, loc=None, ip=None):
    res = qact.geglu(args[0], args[1])
    return res, res


# Custom GemmAct implementation that fixes SM90 activation loop for gated activations
class MacheteGemmActMixin(quack.gemm_act.GemmActMixin):
    @cute.jit
    def epi_visit_subtile(
        self,
        params: quack.gemm_act.GemmActMixin.EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        trs_rd: cute.Tensor,
        trs_rc: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        # Call GemmDefaultEpiMixin from quack
        quack.gemm_default_epi.GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, trs_rd, trs_rc)

        # Apply activation function if provided
        if const_expr(params.act_fn is not None):
            trs_rpostact = cute.make_fragment(trs_rd.layout.shape, self.acc_dtype)
            # Unify epilogue activation loop to use pairwise calls for BOTH SM90 and SM100.
            # This enables gated activations (like SwiGLU) on SM90 by passing pairs of accumulators.
            for i in cutlass.range(cute.size(trs_rpostact) // 2, unroll_full=True):
                trs_rpostact[2 * i], trs_rpostact[2 * i + 1] = params.act_fn((trs_rd[2 * i], trs_rd[2 * i + 1]))
        else:
            trs_rpostact = trs_rd

        # Type conversion
        trs_rpostact_out = cute.make_fragment_like(trs_rpostact, self.postact_dtype)
        trs_rpostact_out.store(trs_rpostact.load().to(self.postact_dtype))
        return trs_rpostact_out


class MacheteGemmActSm90(MacheteGemmActMixin, quack.gemm_sm90.GemmSm90):
    pass


class MacheteGemmActSm100(MacheteGemmActMixin, quack.gemm_sm100.GemmSm100):
    pass


# Local act_fn_map including our gated ops
act_fn_map = dict(quack.gemm_act.act_fn_map)
act_fn_map["swiglu"] = swiglu_tuple_dupe
act_fn_map["geglu"] = geglu_tuple_dupe


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
    if cu_seqlens_m is not None:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        if D is not None:
            assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    gather_A = A_idx is not None
    if gather_A:
        assert cu_seqlens_m is not None, "gather_A requires varlen (cu_seqlens_m must be specified)"
        assert cluster_N == 1, "gather_A requires cluster_N=1"
    assert activation in act_fn_map, f"Unsupported activation {activation}"

    L, M, K, N, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A, B, D, C, additional_tensors={"PostAct": PostAct}, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx
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
    act_fn = act_fn_map[activation]
    epi_args = GemmCls.EpilogueArguments(
        tensor_infos["PostAct"].cute_tensor,
        act_fn,
        mRowVecBroadcast=from_dlpack(rowvec_bias.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=1)
        if rowvec_bias is not None
        else None,
        mColVecBroadcast=from_dlpack(colvec_bias.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=1 if cu_seqlens_m is None else 0
        )
        if colvec_bias is not None
        else None,
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
    cache = machete_gemm_act.compile_cache
    if compile_key not in cache:
        if device_capacity[0] == 9:
            GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent)
        gemm_obj = GemmCls(
            acc_dtype,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            gather_A=gather_A,
        )
        cache[compile_key] = cute.compile(
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
    cache[compile_key](
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,
        tensor_infos["C"].cute_tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
    )


machete_gemm_act.compile_cache = {}


def gated_mlp_sm90_forward(x, weight, bias=None, act_type="swiglu"):
    """
    Forward pass for Gated MLP on SM90 using local MacheteGemmAct kernel.
    """
    # x shape handling
    x_orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    m_dim, k_dim = x_2d.shape
    k2_dim, n2_dim = weight.shape

    assert k_dim == k2_dim
    assert n2_dim % 2 == 0

    # Check device capability
    if not x.is_cuda:
        raise ValueError("Input must be on CUDA")

    # We output to a buffer of size (M, 2N) because the kernel expects to write back
    # the same number of elements it read for the activation (2 inputs -> 2 outputs).
    # We will slice this buffer later.
    post_act = torch.empty((m_dim, n2_dim), dtype=x.dtype, device=x.device)

    # Launch local machete_gemm_act kernel
    machete_gemm_act(
        x_2d.reshape(1, m_dim, k_dim),  # A: (l, m, k)
        weight.reshape(1, n2_dim, k_dim),  # B: (l, n, k)
        None,  # D
        None,  # C
        post_act.reshape(1, m_dim, n2_dim),  # PostAct
        None,  # tile_count_semaphore
        act_type,
        tile_M=128,
        tile_N=128,
        cluster_M=1,
        cluster_N=1,
    )

    # Result is in post_act (M, 2N) where columns 2i and 2i+1 are identical.
    # We return the first of the pairs and restore the original shape.
    res = post_act[:, ::2]
    return res.reshape(*x_orig_shape[:-1], -1)


class GatedMLPSM90Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, act_type="swiglu"):
        ctx.save_for_backward(x, weight)
        ctx.act_type = act_type
        return gated_mlp_sm90_forward(x, weight, act_type=act_type)

    @staticmethod
    def backward(ctx, dout):
        x, weight = ctx.saved_tensors
        return None, None, None


def gated_mlp_sm90(x, weight, act_type="swiglu"):
    return GatedMLPSM90Func.apply(x, weight, act_type)
