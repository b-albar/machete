# Copyright (c) 2025, Machete Authors
"""Shared chunk size (BT) selection for GDN 5-op pipeline.

BT is the temporal chunk size shared across all 5 ops. Larger BT means
fewer chunks but more smem per tile. The bottleneck is GDNSolveOp which
needs a [BT, BT] fp32 scratch matrix.

Supported values: 32 (fits in 16KB pages) and 64 (needs ≥32KB pages).
"""

_DEFAULT_BT = 64
_VALID_BTS = (32, 64)


def auto_bt(page_size: int) -> int:
    """Select the largest BT that fits in page_size.

    BT=64 needs ≥32KB (SolveOp's [64,64] fp32 = 16KB + K buffers).
    BT=32 needs ≥8KB  (SolveOp's [32,32] fp32 = 4KB + K buffers).
    """
    if page_size >= 32768:
        return 64
    return 32
