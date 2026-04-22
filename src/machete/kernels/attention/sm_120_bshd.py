# Copyright (c) 2025, Machete Authors
"""BSHD interface wrappers for SM120 FlashAttention.

These wrappers keep the existing SM120 kernel implementation unchanged while
presenting a sequence-major `(B, S, H, D)` interface to the rest of the model
graph. Internally they pass zero-copy `permute(...)` views to the existing
`BHSD` kernels.
"""

import torch

from .sm_120 import FlashAttentionSm120Op as FlashAttentionSm120BHSDOp


def _max_attention_page_size(device=None):
    if device is None:
        device = torch.cuda.current_device()
    max_smem = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    return ((max_smem - 512) // 128) * 128


def _effective_page_size(q, page_size):
    """Prefer full attention smem budget for long-sequence BSHD prefill.

    The BHSD kernel already benefits materially from larger page sizes at long
    sequence lengths because attention does its own KV pipelining internally.
    The BSHD wrapper should not require every caller to rediscover that.
    """
    if page_size is None:
        return _max_attention_page_size(q.device)
    return page_size


def _bshd_to_bhsd(x):
    if x is None:
        return None
    assert x.ndim == 4, f"Expected 4D BSHD tensor, got shape={tuple(x.shape)}"
    return x.permute(0, 2, 1, 3)


class FlashAttentionSm120BSHDOp:
    """BSHD-facing wrapper over :class:`FlashAttentionSm120Op`."""

    @classmethod
    def schedule(cls, *, q, k, v, o, lse=None, **kwargs):
        if q.ndim == 3:
            tensors = dict(q=q, k=k, v=v, o=o)
            if lse is not None:
                tensors["lse"] = lse
            return FlashAttentionSm120BHSDOp.schedule(**tensors, **kwargs)
        kwargs["page_size"] = _effective_page_size(q, kwargs.get("page_size"))
        q_bhsd = _bshd_to_bhsd(q)
        k_bhsd = _bshd_to_bhsd(k)
        v_bhsd = _bshd_to_bhsd(v)
        o_bhsd = _bshd_to_bhsd(o)
        tensors = dict(q=q_bhsd, k=k_bhsd, v=v_bhsd, o=o_bhsd)
        if lse is not None:
            tensors["lse"] = lse
        return FlashAttentionSm120BHSDOp.schedule(**tensors, **kwargs)

    @classmethod
    def kernel_config(cls, ops):
        return FlashAttentionSm120BHSDOp.kernel_config(ops)


__all__ = ["FlashAttentionSm120BSHDOp"]
