# Copyright (c) 2025, Machete Authors
"""
Causal Depthwise Conv1d Op for the Megakernel.

Implements Tri Dao's causal-conv1d semantics in channels-last layout:
    y[b, l, d] = sum(w[d, K-1-j] * x[b, l-j, d]  for j in 0..K-1)
    where x[b, l-j, d] = 0 when l-j < 0.

Optionally applies SiLU activation after convolution.

Tensor layout (channels-last):
    x: (B, L, D) — input
    w: (D, K)   — depthwise convolution kernel (user API)
    y: (B, L, D) — output

Internally, w is transposed to (K, D) for contiguous row access per tap.

Pipelined load/compute/store:
    load:    TMA G->S (x tile)
    compute: read x from smem (in-tile) or global (halo), convolve, write y to smem
    store:   TMA S->G (y tile)

Smem layout within a page:
    [0, x_tile_bytes)              — x input (TMA loaded)
    [x_tile_bytes, 2*x_tile_bytes) — y output (compute writes, TMA stored)

Halo handling:
    For the first K-1 output positions in each tile, some input positions
    fall before the tile start (halo). These are read from global memory
    via the x tensor parameter. All other positions read from smem.

Tensor parallelism:
    Depthwise conv is trivially TP-parallel along D.
    Declare peer_stores={"y"} and override communicate() to send
    sharded results to peer GPUs via TMA S2G.

Usage:
    from machete.kernels.conv1d import Conv1dOp
    from machete.megakernel import Megakernel

    ops = Conv1dOp.schedule(x=x, w=w, activation='silu')
    kernel = Megakernel(ops, config=Conv1dOp.kernel_config(ops))
    kernel.run()
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Float32

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import mbarrier_arrive_expect_tx


# Activation type constants
ACT_NONE = 0
ACT_SILU = 1

ACT_MAP = {None: ACT_NONE, 'none': ACT_NONE, 'silu': ACT_SILU, 'swish': ACT_SILU}


class Conv1dOp(Op):
    """Causal depthwise Conv1d for the megakernel framework.

    TMA loads x into the first half of the page, compute reads from smem
    (with global fallback for K-1 halo rows), writes y into the second half.
    TMA stores y from the second half.

    Tensor declarations:
        x: (B, L, D) — input tensor
        w: (K, D)    — depthwise conv weight (transposed internally)
        y: (B, L, D) — output tensor

    Tiling:
        tile_B=1, tile_D=D (full extent), tile_L auto-computed.
        2 * tile_L * D * elem_bytes <= page_size.
    """

    reads = {
        "x": (None, ("B", "L", "D")),
        "w": (None, ("K", "D")),
    }
    writes = {
        "y": (None, ("B", "L", "D")),
    }
    tile = ("B", "L", "D")

    tma_loads = {"x"}
    tma_stores = {"y"}

    def __init__(self, **config):
        super().__init__(**config)
        self.activation = getattr(self, 'activation', ACT_NONE)
        self.page_size = getattr(self, 'page_size', DEFAULT_PAGE_SIZE)

        if self.x_dtype == cutlass.Float32:
            self.elem_bytes = 4
        elif self.x_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        self.x_tile_bytes = self.tile_size_L * self.D * self.elem_bytes
        self.y_smem_offset = self.x_tile_bytes
        self.total_smem = 2 * self.x_tile_bytes

        assert self.total_smem <= self.page_size, (
            f"Conv1dOp: tile smem ({self.total_smem}B) exceeds "
            f"page_size ({self.page_size}B). Reduce tile_size_L={self.tile_size_L}."
        )

        self.num_warps = self.threads_per_row // 32

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule_forward(cls, tile_sizes=None, activation=None,
                         page_size=DEFAULT_PAGE_SIZE, **tensors):
        """Schedule causal conv1d forward.

        Args:
            tile_sizes: optional {"L": int} override.
            activation: None, 'silu', or 'swish'.
            page_size: shared memory page size in bytes.
            **tensors: x (B,L,D) required, w (D,K) required,
                       y (B,L,D) optional (allocated if omitted).
        """
        tile_sizes = dict(tile_sizes or {})

        # Transpose w from user layout (D, K) to internal layout (K, D)
        w = tensors.get('w')
        if w is not None and w.ndim == 2:
            D, K = w.shape
            tensors['w'] = w.t().contiguous()  # (K, D)

        x = tensors.get('x')
        if x is not None:
            D = x.shape[2]
            elem_bytes = x.element_size()
            if "L" not in tile_sizes:
                # 2x smem: x tile + y tile
                tile_sizes["L"] = max(1, page_size // (2 * D * elem_bytes))
            # One batch per tile; D is full extent (one tile along D)
            tile_sizes.setdefault("B", 1)
            tile_sizes["D"] = D

        if 'y' not in tensors:
            tensors['y'] = torch.empty_like(x)

        act_id = ACT_MAP.get(activation, ACT_NONE)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims['activation'] = act_id
        ops[0].static_dims['page_size'] = page_size
        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for Conv1dOp."""
        from machete.megakernel import MegakernelConfig
        page_size = ops[0].static_dims.get('page_size', DEFAULT_PAGE_SIZE)
        return MegakernelConfig(page_size=page_size)

    # =========================================================================
    # Forward Load (TMA G->S)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_L, tile_D, x_tma, x_tma_gmem, work_mbar):
        """TMA load of x tile from global to shared memory."""
        sX = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_L, 1)),
        )
        gX = cute.local_tile(
            x_tma_gmem, (self.D, self.tile_size_L, 1), (None, None, None),
        )
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            x_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sX, 0, 3),
            cute.group_modes(gX, 0, 3),
        )

        nbytes = Int32(self.x_tile_bytes)
        mbar_ptr = cute.make_ptr(
            cutlass.Int64, work_mbar, cute.AddressSpace.smem
        )
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(x_tma, tXgX[(None, tile_D, tile_L, tile_B)], tXsX,
                  tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Forward Compute
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_L, tile_D, x, w, y):
        """Causal depthwise conv1d: read x from smem/global, write y to smem.

        For each output position l in [tile_start, tile_start + tile_L):
            y[b, l, d] = sum(w[K-1-j, d] * x[b, l-j, d] for j in 0..K-1)
        with optional SiLU activation.

        x is read from smem for in-tile positions (src_l in [l_start, l_start+tile_L)).
        For halo positions (src_l < l_start), x is read from global memory.
        y is written to smem at y_smem_offset for TMA store.
        """
        x_smem = cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem)
        y_smem = cute.make_ptr(
            self.x_dtype, page_ptr + self.y_smem_offset, cute.AddressSpace.smem
        )

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        thr_layout = cute.make_layout(32)

        b = tile_B
        l_start = tile_L * self.tile_size_L

        # Pre-load weight taps into registers: w_regs[j] = w[K-1-j, :]
        w_regs = []
        for j in cutlass.range_constexpr(self.K):
            w_row = cute.make_tensor(
                w.iterator + (self.K - 1 - j) * self.D,
                cute.make_layout(self.D),
            )
            w_part = cute.local_partition(w_row, thr_layout, lane_idx)
            w_reg = cute.make_fragment_like(w_part)
            cute.autovec_copy(w_part, w_reg)
            w_regs.append(w_reg)

        # Each warp processes different L positions in round-robin
        for local_l in range(warp_idx, self.tile_size_L, num_warps):
            l_idx = l_start + local_l

            if l_idx < self.L:
                # Output row in y smem region
                out_row = cute.make_tensor(
                    y_smem + local_l * self.D,
                    cute.make_layout(self.D),
                )
                out_part = cute.local_partition(out_row, thr_layout, lane_idx)

                # Float32 accumulator
                acc_reg = cute.make_fragment_like(out_part, Float32)
                for i in range(cute.size(acc_reg)):
                    acc_reg[i] = Float32(0.0)

                # Convolve: accumulate K taps
                for j in cutlass.range_constexpr(self.K):
                    src_l = l_idx - j
                    src_local = local_l - j  # Python int (constexpr j, dynamic local_l)

                    # Case 1: in-tile position — read x from smem
                    if src_local >= Int32(0):
                        xs_row = cute.make_tensor(
                            x_smem + src_local * self.D,
                            cute.make_layout(self.D),
                        )
                        xs_part = cute.local_partition(xs_row, thr_layout, lane_idx)
                        xs_reg = cute.make_fragment_like(xs_part)
                        cute.autovec_copy(xs_part, xs_reg)

                        for i in range(cute.size(acc_reg)):
                            acc_reg[i] = acc_reg[i] + w_regs[j][i].to(Float32) * xs_reg[i].to(Float32)

                    # Case 2: halo position — read x from global
                    if src_local < Int32(0):
                        if src_l >= Int32(0):
                            xg_row = cute.make_tensor(
                                x.iterator + b * self.L * self.D + src_l * self.D,
                                cute.make_layout(self.D),
                            )
                            xg_part = cute.local_partition(xg_row, thr_layout, lane_idx)
                            xg_reg = cute.make_fragment_like(xg_part)
                            cute.autovec_copy(xg_part, xg_reg)

                            for i in range(cute.size(acc_reg)):
                                acc_reg[i] = acc_reg[i] + w_regs[j][i].to(Float32) * xg_reg[i].to(Float32)

                # Optional SiLU activation
                if self.activation == ACT_SILU:
                    for i in range(cute.size(acc_reg)):
                        val = acc_reg[i]
                        neg_val = Float32(0.0) - val
                        exp_neg = cute.math.exp(neg_val, fastmath=True)
                        acc_reg[i] = val / (Float32(1.0) + exp_neg)

                # Cast back to x_dtype and write to y smem region
                out_reg = cute.make_fragment_like(out_part)
                for i in range(cute.size(out_reg)):
                    out_reg[i] = acc_reg[i].to(self.x_dtype)
                cute.autovec_copy(out_reg, out_part)

    # =========================================================================
    # Forward Store (3D TMA S->G)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_B, tile_L, tile_D, y_tma, y_tma_gmem):
        """TMA store of conv1d result from shared to global memory."""
        sY = cute.make_tensor(
            cute.make_ptr(
                self.x_dtype, page_ptr + self.y_smem_offset,
                cute.AddressSpace.smem,
            ),
            cute.make_layout((self.D, self.tile_size_L, 1)),
        )
        gY = cute.local_tile(
            y_tma_gmem, (self.D, self.tile_size_L, 1), (None, None, None),
        )
        tYsY, tYgY = cute.nvgpu.cpasync.tma_partition(
            y_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sY, 0, 3),
            cute.group_modes(gY, 0, 3),
        )
        with cute.arch.elect_one():
            cute.copy(y_tma, tYsY, tYgY[(None, tile_D, tile_L, tile_B)])


__all__ = ["Conv1dOp"]
