# Copyright (c) 2025, Machete Authors
"""
Gated Linear Unit (GLU) Ops for the Megakernel.

Computes y = act(gate) * up where:
    gate = x[:, :, :D]
    up   = x[:, :, D:]
    x has shape (B, S, 2D), y has shape (B, S, D)

Supported activations (via static_dims['activation']):
    0 = Identity (plain gating): y = gate * up
    1 = ReLU:  y = relu(gate) * up
    2 = SiLU/Swish:  y = silu(gate) * up  (default)

Direct global access (no TMA) — same pattern as RMSNormOp.
One row per block for maximum GPU occupancy.
All compute warps cooperate on each row (split D elements).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32

from machete.megakernel.ops import Op


# =============================================================================
# Constants
# =============================================================================

ACT_IDENTITY = 0
ACT_RELU = 1
ACT_SILU = 2

ACT_MAP = {
    'identity': ACT_IDENTITY,
    'relu': ACT_RELU,
    'silu': ACT_SILU,
    'swish': ACT_SILU,
}

# Scratch-only page size: no smem staging needed for element-wise ops.
_DIRECT_PAGE_SIZE = 128


# =============================================================================
# Shared helpers
# =============================================================================


def _glu_kernel_config(ops):
    """Kernel config for direct-mode GLU (shared by forward and backward).

    threads_per_block includes DMA warps (framework always reserves them).
    Cap at 4 compute warps (128 threads) — GLU is memory-bound.
    """
    from machete.megakernel import MegakernelConfig
    from machete.megakernel.megakernel import NUM_DMA_WARPS
    D = ops[0].static_dims.get('D', 4096)
    compute_threads = 64
    for ct in [128, 64]:
        if D % ct == 0:
            compute_threads = ct
            break
    return MegakernelConfig(
        threads_per_block=compute_threads + NUM_DMA_WARPS * 32,
        page_size=_DIRECT_PAGE_SIZE,
    )


def _glu_init(self):
    """Common __init__ logic for GLUOp and GLUBwdOp."""
    self.activation = getattr(self, 'activation', ACT_SILU)
    self.page_size = getattr(self, 'page_size', _DIRECT_PAGE_SIZE)
    if self.x_dtype in (cutlass.Float16, cutlass.BFloat16):
        self.elem_bytes = 2
    else:
        self.elem_bytes = 4

    assert self.D >= 32 and self.D % 32 == 0, (
        f"GLU requires D >= 32 and D % 32 == 0, got D={self.D}")

    max_warps = min(4, self.threads_per_row // 32)
    max_et = max_warps * 32
    self.effective_threads = 32
    for t in range(32, max_et + 1, 32):
        if self.D % t == 0:
            self.effective_threads = t
    self.effective_warps = self.effective_threads // 32


# =============================================================================
# Forward Op
# =============================================================================


class GLUOp(Op):
    """GLU forward — direct global access (no TMA staging).

    x (B, S, N=2D) → y (B, S, D) = act(gate) * up
    where gate = x[:,:,:D], up = x[:,:,D:].
    """

    reads = {
        "x": (None, ("B", "S", "N")),
    }
    writes = {
        "y": (None, ("B", "S", "D")),
    }
    tile = ("B", "S", "D")

    tma_loads = set()
    tma_stores = set()

    def __init__(self, **config):
        super().__init__(**config)
        _glu_init(self)

    kernel_config = staticmethod(_glu_kernel_config)

    @classmethod
    def schedule_forward(cls, tile_sizes=None, activation='silu',
                         page_size=_DIRECT_PAGE_SIZE, **tensors):
        """Schedule GLU forward.

        Args:
            activation: 'silu' (default), 'relu', or 'identity'.
            page_size: Shared memory page size in bytes.
            **tensors: x (B,S,2D) and y (B,S,D) required.
        """
        x = tensors['x']
        y = tensors['y']
        assert x.shape[-1] == 2 * y.shape[-1], (
            f"GLU: x last dim ({x.shape[-1]}) must be 2 * y last dim ({y.shape[-1]})")

        tile_sizes = dict(tile_sizes or {})
        tile_sizes["S"] = 1
        tile_sizes.setdefault("B", 1)

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims['activation'] = ACT_MAP.get(activation, ACT_SILU)
        ops[0].static_dims['page_size'] = page_size
        return ops

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D, x, y):
        """GLU forward: read gate+up from global x, write y to global."""
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        tidx = warp_idx * Int32(32) + lane_idx

        if tidx < Int32(self.effective_threads):
            row_idx = tile_S
            x_offset = tile_B * Int32(self.S * self.N) + row_idx * Int32(self.N)
            y_offset = tile_B * Int32(self.S * self.D) + row_idx * Int32(self.D)
            thr_layout = cute.make_layout(self.effective_threads)

            # Load gate (first D elements of row)
            gate_row = cute.make_tensor(
                x.iterator + x_offset,
                cute.make_layout(self.D),
            )
            gate_part = cute.local_partition(gate_row, thr_layout, tidx)
            gate_reg = cute.make_fragment_like(gate_part)
            cute.autovec_copy(gate_part, gate_reg)

            # Load up (last D elements of row)
            up_row = cute.make_tensor(
                x.iterator + x_offset + Int32(self.D),
                cute.make_layout(self.D),
            )
            up_part = cute.local_partition(up_row, thr_layout, tidx)
            up_reg = cute.make_fragment_like(up_part)
            cute.autovec_copy(up_part, up_reg)

            # Compute y = act(gate) * up in float32
            y_reg = cute.make_fragment_like(gate_reg)
            for i in range(cute.size(gate_reg)):
                g = gate_reg[i].to(Float32)
                u = up_reg[i].to(Float32)

                # Must init before branches (CuTe DSL traces all paths)
                act_val = g
                if self.activation == ACT_IDENTITY:
                    act_val = g
                elif self.activation == ACT_RELU:
                    act_val = g
                    if g < Float32(0.0):
                        act_val = Float32(0.0)
                elif self.activation == ACT_SILU:
                    neg_g = Float32(0.0) - g
                    exp_neg = cute.math.exp(neg_g, fastmath=True)
                    act_val = g / (Float32(1.0) + exp_neg)

                y_reg[i] = (act_val * u).to(self.x_dtype)

            # Store y to global
            y_row = cute.make_tensor(
                y.iterator + y_offset,
                cute.make_layout(self.D),
            )
            y_part = cute.local_partition(y_row, thr_layout, tidx)
            cute.autovec_copy(y_reg, y_part)


# =============================================================================
# Backward Op
# =============================================================================


class GLUBwdOp(Op):
    """GLU backward — direct global access (no TMA staging).

    dy (B,S,D), x (B,S,N=2D) → dx (B,S,N=2D)
    where dx[:,:,:D] = d_gate, dx[:,:,D:] = d_up.

    d_gate = dy * up * act'(gate)
    d_up   = dy * act(gate)
    """

    reads = {
        "dy": (None, ("B", "S", "D")),
        "x":  (None, ("B", "S", "N")),
    }
    writes = {
        "dx": (None, ("B", "S", "N")),
    }
    tile = ("B", "S", "D")

    tma_loads = set()
    tma_stores = set()

    def __init__(self, **config):
        super().__init__(**config)
        # x_dtype comes from the first read tensor (dy). Override to get x's dtype.
        self.x_dtype = getattr(self, 'x_dtype', self.dy_dtype)
        _glu_init(self)

    kernel_config = staticmethod(_glu_kernel_config)

    @classmethod
    def schedule_forward(cls, tile_sizes=None, activation='silu',
                         page_size=_DIRECT_PAGE_SIZE, **tensors):
        """Schedule GLU backward.

        Args:
            activation: 'silu' (default), 'relu', or 'identity'.
            **tensors: dy (B,S,D), x (B,S,2D), dx (B,S,2D) required.
        """
        dy = tensors['dy']
        x = tensors['x']
        assert x.shape[-1] == 2 * dy.shape[-1], (
            f"GLU bwd: x last dim ({x.shape[-1]}) must be 2 * dy last dim ({dy.shape[-1]})")

        tile_sizes = dict(tile_sizes or {})
        tile_sizes["S"] = 1
        tile_sizes.setdefault("B", 1)

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims['activation'] = ACT_MAP.get(activation, ACT_SILU)
        ops[0].static_dims['page_size'] = page_size
        return ops

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D, dy, x, dx):
        """GLU backward: compute d_gate and d_up, write dx to global."""
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        tidx = warp_idx * Int32(32) + lane_idx

        if tidx < Int32(self.effective_threads):
            row_idx = tile_S
            x_offset = tile_B * Int32(self.S * self.N) + row_idx * Int32(self.N)
            dy_offset = tile_B * Int32(self.S * self.D) + row_idx * Int32(self.D)
            thr_layout = cute.make_layout(self.effective_threads)

            # Load dy
            dy_row = cute.make_tensor(
                dy.iterator + dy_offset,
                cute.make_layout(self.D),
            )
            dy_part = cute.local_partition(dy_row, thr_layout, tidx)
            dy_reg = cute.make_fragment_like(dy_part)
            cute.autovec_copy(dy_part, dy_reg)

            # Load gate (first D elements)
            gate_row = cute.make_tensor(
                x.iterator + x_offset,
                cute.make_layout(self.D),
            )
            gate_part = cute.local_partition(gate_row, thr_layout, tidx)
            gate_reg = cute.make_fragment_like(gate_part)
            cute.autovec_copy(gate_part, gate_reg)

            # Load up (last D elements)
            up_row = cute.make_tensor(
                x.iterator + x_offset + Int32(self.D),
                cute.make_layout(self.D),
            )
            up_part = cute.local_partition(up_row, thr_layout, tidx)
            up_reg = cute.make_fragment_like(up_part)
            cute.autovec_copy(up_part, up_reg)

            # Compute d_gate and d_up in float32
            dgate_reg = cute.make_fragment_like(gate_reg)
            dup_reg = cute.make_fragment_like(up_reg)

            for i in range(cute.size(gate_reg)):
                g = gate_reg[i].to(Float32)
                u = up_reg[i].to(Float32)
                d = dy_reg[i].to(Float32)

                if self.activation == ACT_IDENTITY:
                    # y = gate * up → d_gate = dy * up, d_up = dy * gate
                    dgate_reg[i] = (d * u).to(self.x_dtype)
                    dup_reg[i] = (d * g).to(self.x_dtype)
                elif self.activation == ACT_RELU:
                    # y = relu(gate) * up
                    if g > Float32(0.0):
                        dgate_reg[i] = (d * u).to(self.x_dtype)
                        dup_reg[i] = (d * g).to(self.x_dtype)
                    else:
                        dgate_reg[i] = Float32(0.0).to(self.x_dtype)
                        dup_reg[i] = Float32(0.0).to(self.x_dtype)
                elif self.activation == ACT_SILU:
                    # y = silu(gate) * up
                    neg_g = Float32(0.0) - g
                    exp_neg = cute.math.exp(neg_g, fastmath=True)
                    sig = Float32(1.0) / (Float32(1.0) + exp_neg)
                    silu_val = g * sig
                    silu_grad = sig * (Float32(1.0) + g * (Float32(1.0) - sig))
                    dup_reg[i] = (d * silu_val).to(self.x_dtype)
                    dgate_reg[i] = (d * u * silu_grad).to(self.x_dtype)

            # Store d_gate to dx[:,:,:D]
            dgate_row = cute.make_tensor(
                dx.iterator + x_offset,
                cute.make_layout(self.D),
            )
            dgate_part = cute.local_partition(dgate_row, thr_layout, tidx)
            cute.autovec_copy(dgate_reg, dgate_part)

            # Store d_up to dx[:,:,D:]
            dup_row = cute.make_tensor(
                dx.iterator + x_offset + Int32(self.D),
                cute.make_layout(self.D),
            )
            dup_part = cute.local_partition(dup_row, thr_layout, tidx)
            cute.autovec_copy(dup_reg, dup_part)
