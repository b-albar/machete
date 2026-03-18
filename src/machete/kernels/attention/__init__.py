# Copyright (c) 2025, Machete Authors
"""Flash Attention kernels for the megakernel framework."""

from .sm_100 import FlashAttentionSm100Op
from .sm_120 import FlashAttentionSm120Op
from .sm_120_bwd import FlashAttentionSm120BwdOp
from .flash_decoding import FlashDecodingSplitOp, FlashDecodingCombineOp, flash_decoding_schedule
from machete.megakernel.ops import DEFAULT_PAGE_SIZE

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


def flash_attention_schedule(q, k, v, o, causal=False, page_size=DEFAULT_PAGE_SIZE,
                             kv_group_size=1, lse=None):
    """Schedule attention with auto-dispatch between FA and FlashDecoding.

    Uses FlashDecoding for decode-like workloads where there aren't enough
    tiles to saturate the GPU (small BH × ceil(M/tile_M) vs SM count).
    Falls back to regular FlashAttention for prefill workloads.

    Returns:
        (ops, config): ScheduledOps and MegakernelConfig.
    """
    BH, M, D = q.shape
    elem = q.element_size()

    num_SMs = torch.cuda.get_device_properties(q.device).multi_processor_count

    # Estimate FA tile_M to compute tile count
    min_kv_bytes = 2 * 16 * D * elem  # 2 KV buffers × 16 rows minimum
    max_tile_M_page = (page_size - min_kv_bytes) // (D * elem)
    nw = 1
    max_nw = min(8, max_tile_M_page // 16, max(1, M // 16))
    while nw * 2 <= max_nw:
        nw *= 2
    tile_M = max(16, nw * 16)

    total_tiles = BH * ((M + tile_M - 1) // tile_M)

    # Flash decoding constraints: Q must fit in a single page, M must be MMA-aligned
    fd_possible = (M * D * elem <= page_size and M >= 16 and M % 16 == 0)

    # Dispatch: use FD when too few tiles to saturate SMs.
    # Exception: BH=1 with large M (>64) — combine overhead dominates.
    use_fd = (fd_possible
              and total_tiles < num_SMs // 2
              and (total_tiles >= 2 or M <= 64))

    tensors = dict(q=q, k=k, v=v, o=o)
    if lse is not None:
        tensors["lse"] = lse

    if use_fd:
        ops = FlashDecodingSplitOp.schedule_forward(
            causal=causal, page_size=page_size, kv_group_size=kv_group_size,
            **tensors,
        )
        config = FlashDecodingSplitOp.kernel_config(ops)
    else:
        ops = FlashAttentionOp.schedule_forward(
            causal=causal, page_size=page_size, kv_group_size=kv_group_size,
            **tensors,
        )
        config = FlashAttentionOp.kernel_config(ops)

    return ops, config


__all__ = [
    "FlashAttentionOp", "FlashAttentionSm100Op", "FlashAttentionSm120Op", "FlashAttentionSm120BwdOp",
    "FlashDecodingSplitOp", "FlashDecodingCombineOp", "flash_decoding_schedule",
    "flash_attention_schedule",
]
