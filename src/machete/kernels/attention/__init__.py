# Copyright (c) 2025, Machete Authors
"""Flash Attention kernels for the megakernel framework."""

from .sm_100 import FlashAttentionSm100Op
from .sm_120 import FlashAttentionSm120Op
from .sm_120_bwd import FlashAttentionSm120BwdOp
from .dpsum import AttentionDPSumOp
from .flash_decoding import (
    FlashDecodingSplitOp,
    flash_decoding_schedule,
)

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


def _max_attention_page_size(device=None):
    """Max page_size for attention with num_pages=1 (full smem budget).

    Attention ops do KV pipelining internally via cpasync, so the framework's
    ring-buffer page pipeline provides no benefit. Using the full smem budget
    as a single page maximizes the KV buffer size.
    """
    if device is None:
        device = torch.cuda.current_device()
    max_smem = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    # Reserve 512 bytes for framework scratch (ring_state, flags, IQ, mbarriers).
    # With num_pages=1, overhead is minimal.
    return ((max_smem - 512) // 128) * 128


def flash_attention_schedule(q, k, v, o, causal=False, page_size=None,
                             kv_group_size=1, lse=None):
    """Schedule attention with auto-dispatch between FA and FlashDecoding.

    Uses FlashDecoding for decode-like workloads where there aren't enough
    tiles to saturate the GPU (small BH × ceil(M/tile_M) vs SM count).
    Falls back to regular FlashAttention for prefill workloads.

    Args:
        page_size: Size of each smem page in bytes. None = auto-detect max
            available smem (recommended for attention).

    Returns:
        (ops, config): ScheduledOps and MegakernelConfig.
    """
    # 3D backward compat
    if q.ndim == 3:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        o = o.unsqueeze(0)
        if lse is not None:
            lse = lse.unsqueeze(0)

    if FlashAttentionOp is FlashAttentionSm120Op and q.ndim == 4:
        B, M, H, D = q.shape
    else:
        B, H, M, D = q.shape
    elem = q.element_size()

    # Auto-detect max page_size for attention
    if page_size is None:
        page_size = _max_attention_page_size(q.device)

    num_SMs = torch.cuda.get_device_properties(q.device).multi_processor_count

    # Estimate FA tile_M to compute tile count
    min_kv_bytes = 2 * 16 * D * elem  # 2 KV buffers × 16 rows minimum
    max_tile_M_page = (page_size - min_kv_bytes) // (D * elem)
    nw = 1
    max_nw = min(8, max_tile_M_page // 16, max(1, M // 16))
    while nw * 2 <= max_nw:
        nw *= 2
    tile_M = max(16, nw * 16)

    total_tiles = B * H * ((M + tile_M - 1) // tile_M)

    # Flash decoding constraints: Q must fit in page with room for KV, M must be MMA-aligned
    fd_q_bytes = M * D * elem
    fd_kv_min = 2 * 16 * D * elem  # 2 KV buffers × 16 rows minimum
    fd_possible = (fd_q_bytes + fd_kv_min <= page_size and M >= 16 and M % 16 == 0)

    # Dispatch: use FD only for decode-like short-query workloads.
    # At moderate prefill lengths like M=128, FA2-style attention is still the
    # better path even when the tile count looks low, because FlashDecoding's
    # split/combine overhead dominates.
    use_fd = (fd_possible
              and total_tiles < num_SMs // 2
              and M <= 64
              and (total_tiles >= 2 or M <= 64))

    tensors = dict(q=q, k=k, v=v, o=o)
    if lse is not None:
        tensors["lse"] = lse

    if use_fd:
        # flash_decoding_schedule creates its own lse internally
        fd_tensors = {k: v for k, v in tensors.items() if k != "lse"}
        ops, config = flash_decoding_schedule(
            page_size=page_size, causal=causal, kv_group_size=kv_group_size,
            **fd_tensors,
        )
    else:
        ops = FlashAttentionOp.schedule(
            causal=causal, page_size=page_size, kv_group_size=kv_group_size,
            **tensors,
        )
        config = FlashAttentionOp.kernel_config(ops)

    return ops, config


__all__ = [
    "FlashAttentionOp", "FlashAttentionSm100Op", "FlashAttentionSm120Op", "FlashAttentionSm120BwdOp",
    "AttentionDPSumOp",
    "FlashDecodingSplitOp", "flash_decoding_schedule",
    "flash_attention_schedule",
]
