import torch
import os
from machete.jit.jit import load_cuda_ops
from typing import Optional, Callable
import textwrap

from machete.jit.jit import get_cuda_arch, get_gpu_device
from machete.jit.jit_env import ROOT_DIR
from machete.utils.utils import maybe_contiguous
from machete.utils.templates import render_template


# Global variable to store the flash attention ops
_flash_attention_ops: Optional[Callable] = None

def _variant_instantiations(
        variant_name: str,
        variant_code_fwd: str,
        variant_code_bwd: str,
        head_dim: int,
        qo_size: int,
        kv_size: int,
        stages: int) -> dict[str, str]:

    parameters: dict[str, str] = {}

    indentation_level = "    "

    constants = textwrap.indent(textwrap.dedent(f"""
        constexpr unsigned int HEAD_DIM = {head_dim};
        constexpr unsigned int QO_SIZE = {qo_size};
        constexpr unsigned int KV_SIZE = {kv_size};
        constexpr unsigned int STAGES = {stages};
    """), indentation_level)

    variant_name_fwd = f"{variant_name}Fwd"
    variant_name_bwd = f"{variant_name}Bwd"

    parameters["variant_fwd_initialization"] = constants + textwrap.indent(textwrap.dedent(f"""
        auto variant = {variant_name_fwd}();
    """), indentation_level)
    parameters["variant_bwd_initialization"] = constants + textwrap.indent(textwrap.dedent(f"""
        auto variant = {variant_name_bwd}();
    """), indentation_level)

    parameters["variant_interface_fwd"] = textwrap.dedent(variant_code_fwd)
    parameters["variant_interface_bwd"] = textwrap.dedent(variant_code_bwd)
    parameters["variant_fwd_name"] = variant_name_fwd
    parameters["variant_bwd_name"] = variant_name_bwd

    parameters["fwd_explicit_instantiations"] = textwrap.dedent(f"""
        template __global__ void fwd_attend_ker<64, 1, 2, 2, true>(const __grid_constant__ fwd_globals<64, 1, 2, 2> g, AttentionDefaultFwd& variant);
        template __global__ void fwd_attend_ker<64, 1, 2, 2, false>(const __grid_constant__ fwd_globals<64, 1, 2, 2> g, AttentionDefaultFwd& variant);
    """)
    parameters["bwd_explicit_instantiations"] = textwrap.dedent(f"""
        template __global__ void bwd_prep_ker<{head_dim}, {qo_size}, {kv_size}>(const __grid_constant__ bwd_prep_globals<{head_dim}, {qo_size}, {kv_size}> g);

        template __global__ void bwd_attend_ker<{head_dim}, {qo_size}, {kv_size}, {stages}, false, {variant_name_bwd}>(const __grid_constant__ bwd_globals<{head_dim}, {qo_size}, {kv_size}, {stages}> g, {variant_name_bwd}& variant);
        template __global__ void bwd_attend_ker<{head_dim}, {qo_size}, {kv_size}, {stages}, true, {variant_name_bwd}>(const __grid_constant__ bwd_globals<{head_dim}, {qo_size}, {kv_size}, {stages}> g, {variant_name_bwd}& variant);
    """)

    return parameters

def _get_default_variant(device: str, head_dim: int) -> str:
    fa_dir = ROOT_DIR / "csrc/kernels/flash-attention"

    if "H100" in device:
        variant_path = fa_dir / "h100/variants/h100_default.cuh"
        pass
    elif "A100" in device:
        variant_fwd = fa_dir / "a100/variants/default/a100_default_fwd.cuh"
        variant_bwd = fa_dir / "a100/variants/default/a100_default_bwd.cuh"
    elif "5070" in device:
        variant_fwd = fa_dir / "a100/variants/default/a100_default_fwd.cuh"
        variant_bwd = fa_dir / "a100/variants/default/a100_default_bwd.cuh"
        f_fwd = open(variant_fwd, "r")
        f_bwd = open(variant_bwd, "r")
        variant_code_fwd = f_fwd.read()
        variant_code_bwd = f_bwd.read()
        f_fwd.close()
        f_bwd.close()
        template_kwargs = _variant_instantiations(
            "AttentionDefault", variant_code_fwd, variant_code_bwd, head_dim, 1, 2, 2
        )
    else:
        raise ValueError(f"Unsupported device type: {device}")

    return render_template("flash_attention_default",fa_dir / "a100/", **template_kwargs)

def _get_flash_attention_ops(head_dim: int):
    global _flash_attention_ops

    if _flash_attention_ops is None:
        device = get_gpu_device(0)
        variant_dir = _get_default_variant(device, head_dim)
        print(f"variant_dir: {variant_dir}")

        src_files = [os.path.join(variant_dir, f) for f in os.listdir(variant_dir) if f.endswith(".cu")]

        _flash_attention_ops = load_cuda_ops(
            "flash_attention",
            gpu_target=device,
            sources=src_files
        )

    return _flash_attention_ops

@torch.library.custom_op("machete::flash_attention_fwd", mutates_args=(), device_types="cuda")
def _flash_attn_forward(q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        causal: bool,
                        sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    func_op = _get_flash_attention_ops(q.size(-1))
    o, l_vec = func_op.fwd(q, k, v, causal, sm_scale)

    return o, l_vec

@torch.library.register_fake("machete::flash_attention_fwd")
def _flash_attn_forward_fake(q: torch.Tensor,
                            k: torch.Tensor,
                            v: torch.Tensor,
                            causal: bool,
                            sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    batch_size, seqlen_q, num_heads, head_size = q.shape
    o = torch.empty_like(q)
    l_vec = torch.empty(batch_size, seqlen_q, num_heads, 1, device=q.device)

    return o, l_vec


@torch.library.custom_op("machete::flash_attention_bwd", mutates_args=(), device_types="cuda")
def _flash_attn_backward(do:torch.Tensor,
                         o: torch.Tensor,
                         q: torch.Tensor,
                         k: torch.Tensor,
                         v: torch.Tensor,
                         l_vec: torch.Tensor,
                         causal: bool,
                         sm_scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    func_op = _get_flash_attention_ops()

    return func_op.bwd(q, k, v, o, l_vec, do, causal, sm_scale)

@torch.library.register_fake("machete::flash_attention_bwd")
def _flash_attn_backward_fake(do: torch.Tensor,
                              o: torch.Tensor,
                              q: torch.Tensor,
                              k: torch.Tensor,
                              v: torch.Tensor,
                              l_vec: torch.Tensor,
                              causal: bool,
                              sm_scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    return dq, dk, dv


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        if sm_scale is None:
            sm_scale = q.size(-1) ** (-0.5)

        o, l_vec = torch.ops.machete.flash_attention_fwd(q, k, v, causal, sm_scale)

        ctx.save_for_backward(q, k, v, o, l_vec)
        ctx.causal = causal
        ctx.sm_scale = sm_scale

        return o, l_vec

    @staticmethod
    def backward(ctx, do, dl_vec=None):
        q, k, v, o, l_vec = ctx.saved_tensors
        causal = ctx.causal
        sm_scale = ctx.sm_scale

        # Placeholder for backward implementation
        dq, dk, dv = torch.ops.machete.flash_attention_bwd(do.clone(), o.clone(), q.clone(), k.clone(), v.clone(), l_vec.clone(), causal, sm_scale)

        return dq, dk, dv, None, None

flash_attention = FlashAttention.apply
