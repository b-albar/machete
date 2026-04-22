# Copyright (c) 2025, Machete Authors
"""BSHD interface wrappers for SM120 FlashAttention backward."""

import torch

from .sm_120_bwd import FlashAttentionSm120BwdOp as FlashAttentionSm120BHSDBwdOp


def _max_attention_page_size(device=None):
    if device is None:
        device = torch.cuda.current_device()
    max_smem = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    return ((max_smem - 512) // 128) * 128


def _bshd_to_bhsd(x):
    if x is None:
        return None
    assert x.ndim == 4, f"Expected 4D BSHD tensor, got shape={tuple(x.shape)}"
    return x.permute(0, 2, 1, 3)


def _effective_page_size(q, page_size):
    if page_size is None:
        return _max_attention_page_size(q.device)
    return page_size


class FlashAttentionSm120BSHDBwdOp:
    """BSHD-facing wrapper over :class:`FlashAttentionSm120BwdOp`."""

    @classmethod
    def schedule(cls, *, k, v, q, dout, dq, dk, dv, lse, dpsum, **kwargs):
        if q.ndim == 3:
            return FlashAttentionSm120BHSDBwdOp.schedule(
                k=k,
                v=v,
                q=q,
                dout=dout,
                dq=dq,
                dk=dk,
                dv=dv,
                lse=lse,
                dpsum=dpsum,
                **kwargs,
            )
        kwargs["page_size"] = _effective_page_size(q, kwargs.get("page_size"))
        return FlashAttentionSm120BHSDBwdOp.schedule(
            k=_bshd_to_bhsd(k),
            v=_bshd_to_bhsd(v),
            q=_bshd_to_bhsd(q),
            dout=_bshd_to_bhsd(dout),
            dq=_bshd_to_bhsd(dq),
            dk=_bshd_to_bhsd(dk),
            dv=_bshd_to_bhsd(dv),
            lse=lse,
            dpsum=dpsum,
            **kwargs,
        )

    @classmethod
    def kernel_config(cls, ops):
        return FlashAttentionSm120BHSDBwdOp.kernel_config(ops)


__all__ = ["FlashAttentionSm120BSHDBwdOp"]
