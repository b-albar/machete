# Copyright (c) 2025, Machete Authors
"""Flash Attention kernels for the megakernel framework."""

from .sm_100 import FlashAttentionSm100Op
from .sm_120 import FlashAttentionSm120Op

import torch


def _get_flash_attention_op():
    if not torch.cuda.is_available():
        return FlashAttentionSm100Op
    major, _ = torch.cuda.get_device_capability()
    match major:
        case m if m == 12:
            return FlashAttentionSm120Op
        case m if m == 10:
            return FlashAttentionSm100Op
        case _:
            print(f"FlashAttentionOp: unsupported GPU (SM {major}x), requires Hopper (SM90+)")
            return FlashAttentionSm100Op


FlashAttentionOp = _get_flash_attention_op()

__all__ = ["FlashAttentionOp", "FlashAttentionSm100Op", "FlashAttentionSm120Op"]
