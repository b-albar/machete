import torch
from machete.jit.jit import load_cuda_ops
from typing import Optional, Callable

from machete.jit.jit import get_cuda_arch, get_gpu_device
from machete.jit.jit_env import ROOT_DIR
from machete.utils.utils import maybe_contiguous


# Global variable to store the flash attention ops
_flash_attention_ops: Optional[Callable] = None

def _get_flash_attention_ops():
    global _flash_attention_ops

    if _flash_attention_ops is None:
        device = get_gpu_device(0)

        if "H100" in device:
            _flash_attention_ops = load_cuda_ops(
                "flash_attention",
                gpu_target="h100",
                sources=[
                    ROOT_DIR / "csrc/kernels/flash-attention/h100/h100_fwd.cu",
                    ROOT_DIR / "csrc/kernels/flash-attention/h100/h100_bwd.cu",
                    ROOT_DIR / "csrc/kernels/flash-attention/h100/h100_interface.cu",
                ],
            )
        elif "A100" in device:
            _flash_attention_ops = load_cuda_ops(
                "flash_attention",
                gpu_target="a100",
                sources=[
                    ROOT_DIR / "csrc/kernels/flash-attention/a100/a100_fwd.cu",
                    ROOT_DIR / "csrc/kernels/flash-attention/a100/a100_bwd.cu",
                    ROOT_DIR / "csrc/kernels/flash-attention/a100/a100_interface.cu",
                ],
            )
        elif "5070" in device:
            _flash_attention_ops = load_cuda_ops(
                "flash_attention",
                gpu_target="5070",
                sources=[
                    ROOT_DIR / "csrc/kernels/flash-attention/a100/a100_fwd.cu",
                    ROOT_DIR / "csrc/kernels/flash-attention/a100/a100_bwd.cu",
                    ROOT_DIR / "csrc/kernels/flash-attention/a100/a100_interface.cu",
                ],
            )
        else:
            raise ValueError(f"Unsupported device type: {device}")

    return _flash_attention_ops

@torch.library.custom_op("machete::flash_attention_fwd", mutates_args=(), device_types="cuda")
def _flash_attn_forward(q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        causal: bool,
                        sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    func_op = _get_flash_attention_ops()
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
