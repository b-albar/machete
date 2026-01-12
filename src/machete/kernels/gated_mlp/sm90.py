# Copyright (c) 2025, Machete Authors
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
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
from cutlass import Int32
from machete.kernels.gated_linear.sm80 import GatedLinearSM80


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


@dsl_user_op
def swiglu_bwd_tuple(h_pair, dout_pair, *, loc=None, ip=None):
    # dswiglu takes (gate, up, dout) and returns (dg, du, out)
    res = qact.dswiglu(h_pair[0], h_pair[1], dout_pair[0])
    return res[0], res[1]


@dsl_user_op
def geglu_bwd_tuple(h_pair, dout_pair, *, loc=None, ip=None):
    res = qact.dgeglu(h_pair[0], h_pair[1], dout_pair[0])
    return res[0], res[1]


# We use the standard GemmActMixin from Quack but with our Unified activation loop
class MacheteGemmActMixin(quack.gemm_act.GemmActMixin):
    @cute.jit
    def epi_visit_subtile(
        self,
        params: quack.gemm_act.GemmActMixin.EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        trs_rd: cute.Tensor,
        trs_rc: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        # Call GemmDefaultEpiMixin only if not doing a fused backward (where C is used for act_fn)
        if const_expr(trs_rc is None):
            quack.gemm_default_epi.GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, trs_rd, trs_rc)

        # Apply activation function if provided
        if const_expr(params.act_fn is not None):
            trs_rpostact = cute.make_fragment(trs_rd.layout.shape, self.acc_dtype)
            for i in cutlass.range(cute.size(trs_rpostact) // 2, unroll_full=True):
                if const_expr(trs_rc is not None):
                    # Fused backward pass: pass both (gate, up) and (dout, dout) pairs
                    trs_rpostact[2 * i], trs_rpostact[2 * i + 1] = params.act_fn(
                        (trs_rd[2 * i], trs_rd[2 * i + 1]), (trs_rc[2 * i], trs_rc[2 * i + 1])
                    )
                else:
                    # Normal forward pass
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
act_fn_map["swiglu_bwd"] = swiglu_bwd_tuple
act_fn_map["geglu_bwd"] = geglu_bwd_tuple


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


def gated_mlp_sm90_forward(x, weight, bias=None, act_type="swiglu", chunk_size=1024):
    """
    Forward pass for Gated MLP on SM90 using local MacheteGemmAct kernel.
    Chunked to avoid large intermediate buffers (M, 2N).
    Uses reuse of workspace and zero-copy weight view to keep memory footprint minimal.
    """
    # x shape handling
    x_orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    m_dim, k_dim = x_2d.shape
    k2_dim, n2_dim = weight.shape  # n2_dim is 2*N
    n_dim = n2_dim // 2

    assert k_dim == k2_dim
    assert n2_dim % 2 == 0

    # Ensure CUDA
    if not x.is_cuda:
        raise ValueError("Input must be on CUDA")

    # Final output buffer (M, N)
    res = torch.empty((*x_orig_shape[:-1], n_dim), dtype=x.dtype, device=x.device)
    res_2d = res.reshape(m_dim, n_dim)

    # Prepare weight: (l, n, k). Use a zero-copy view (1, 2N, K) from weight (K, 2N).
    # weight.unsqueeze(0).permute(0, 2, 1) is a view that Quack will detect as N-major.
    weight_B = weight.unsqueeze(0).permute(0, 2, 1)

    # Pre-allocate one workspace for the chunks to minimize allocation overhead
    # and ensure peak memory is predictable.
    eff_chunk_size = min(chunk_size, m_dim)
    workspace = torch.empty((eff_chunk_size, n2_dim), dtype=x.dtype, device=x.device)

    # Launch local machete_gemm_act kernel in chunks
    for i in range(0, m_dim, eff_chunk_size):
        actual_chunk_size = min(eff_chunk_size, m_dim - i)
        curr_workspace = workspace[:actual_chunk_size]

        machete_gemm_act(
            x_2d[i : i + actual_chunk_size].reshape(1, actual_chunk_size, k_dim),  # A: (l, m, k)
            weight_B,  # B: (l, n, k)
            None,  # D
            None,  # C
            curr_workspace.reshape(1, actual_chunk_size, n2_dim),  # PostAct
            None,  # tile_count_semaphore
            act_type,
            tile_M=128,
            tile_N=128,
            cluster_M=1,
            cluster_N=1,
        )

        # Result is in workspace. Copy to final output.
        res_2d[i : i + actual_chunk_size].copy_(curr_workspace[:, ::2])

    return res


class SwigluBwdSM90:
    _instances: Dict[Tuple, "SwigluBwdSM90"] = {}

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
                t_g,
                t_u,
                t_d,
                t_dg,
                t_du,
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
    _instances: Dict[Tuple, "GegluBwdSM90"] = {}

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
                t_g,
                t_u,
                t_d,
                t_dg,
                t_du,
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


class GatedMLPSM90Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, act_type="swiglu"):
        ctx.save_for_backward(x, weight)
        ctx.act_type = act_type
        return gated_mlp_sm90_forward(x, weight, act_type=act_type)

    @staticmethod
    def backward(ctx, dout):
        x, weight = ctx.saved_tensors
        act_type = ctx.act_type

        x_orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        m_dim, k_dim = x_2d.shape
        n2_dim = weight.shape[1]
        n_dim = n2_dim // 2
        dout_2d = dout.reshape(-1, n_dim)

        dx = torch.empty_like(x_2d)
        dweight = torch.zeros_like(weight)

        # Chunk size for backward to keep memory footprint low
        chunk_size = 4096

        # Pre-slice weight for efficiency
        w_gate = weight[:, ::2]
        w_up = weight[:, 1::2]
        w_gate_t = w_gate.t()
        w_up_t = w_up.t()
        x_t = x_2d.t()

        for i in range(0, m_dim, chunk_size):
            actual_chunk_size = min(chunk_size, m_dim - i)
            x_chunk = x_2d[i : i + actual_chunk_size]
            dout_chunk = dout_2d[i : i + actual_chunk_size]

            # h = x @ weight (chunk_size, 2N)
            h_chunk = x_chunk @ weight
            h_gate = h_chunk[:, ::2]
            h_up = h_chunk[:, 1::2]

            linear_act_type = "silu" if act_type == "swiglu" else "gelu"
            if not hasattr(ctx, "gated_linear_op"):
                ctx.gated_linear_op = GatedLinearSM80(x_chunk.dtype, linear_act_type)

            dg, du = ctx.gated_linear_op.backward(dout_chunk, h_gate, h_up)

            # Update dx for this chunk
            # dx_chunk = dg @ w_gate.T + du @ w_up.T
            dx_chunk = dg @ w_gate_t
            dx_chunk.addmm_(du, w_up_t)
            dx[i : i + actual_chunk_size] = dx_chunk

            # Update dweight incrementally
            # dweight_gate += x_chunk.T @ dg
            # dweight_up += x_chunk.T @ du
            dweight[:, ::2].addmm_(x_chunk.t(), dg)
            dweight[:, 1::2].addmm_(x_chunk.t(), du)

        return dx.reshape(x_orig_shape), dweight, None


def gated_mlp_sm90(x, weight, act_type="swiglu"):
    return GatedMLPSM90Func.apply(x, weight, act_type)
