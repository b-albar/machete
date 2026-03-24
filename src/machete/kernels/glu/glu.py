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

TMA pipelined load/compute/store:
    Forward:  TMA load x (N=2D per row) → compute y in registers → write y to
              smem (D per row) → TMA store y.
    Backward: TMA load x (N=2D per row) → read dy from global → compute dx →
              write dx to smem (N per row, overwrites x) → TMA store dx.

Multi-row tiling: tile_size_S = page_size / (N * elem_bytes).
Forward uses two-phase compute with named_barrier_sync (y has different stride
from x). Backward overwrites in-place (dx has same stride as x, no barrier).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import (
    mbarrier_arrive_expect_tx,
    named_barrier_sync,
)


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


# =============================================================================
# Forward Op
# =============================================================================


class GLUOp(Op):
    """GLU forward — TMA pipelined load/compute/store.

    x (B, S, N=2D) → y (B, S, D) = act(gate) * up
    where gate = x[:,:,:D], up = x[:,:,D:].

    Smem reuse: x loaded via TMA (tile_S × N), y overwrites at offset 0
    (tile_S × D) after compute barrier.
    """

    reads = {
        "x": (None, ("B", "S", "N")),
    }
    writes = {
        "y": (None, ("B", "S", "D")),
    }
    tile = ("B", "S", "D")

    tma_loads = {"x"}
    tma_stores = {"y"}

    def __init__(self, **config):
        super().__init__(**config)
        self.activation = getattr(self, 'activation', ACT_SILU)
        self.page_size = getattr(self, 'page_size', DEFAULT_PAGE_SIZE)

        if self.x_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        assert self.D >= 32 and self.D % 32 == 0, (
            f"GLU requires D >= 32 and D % 32 == 0, got D={self.D}")
        assert self.N == 2 * self.D, (
            f"GLU: N ({self.N}) must be 2*D ({2 * self.D})")

        # Smem: x occupies tile_S * N * elem_bytes (TMA loaded).
        # After barrier, y overwrites [0, tile_S * D * elem_bytes).
        self.x_tile_bytes = self.tile_size_S * self.N * self.elem_bytes

        assert self.x_tile_bytes <= self.page_size, (
            f"GLU: x tile ({self.x_tile_bytes}B) > page ({self.page_size}B). "
            f"Reduce tile_size_S={self.tile_size_S}.")

        self.num_warps = self.threads_per_row // 32
        self.max_rows_per_warp = (
            (self.tile_size_S + self.num_warps - 1) // self.num_warps
        )

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule(cls, tile_sizes=None, activation='silu',
                         page_size=DEFAULT_PAGE_SIZE, **tensors):
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
        if "S" not in tile_sizes:
            N = x.shape[-1]
            elem_bytes = x.element_size()
            tile_sizes["S"] = max(1, page_size // (N * elem_bytes))
        tile_sizes.setdefault("B", 1)

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims['activation'] = ACT_MAP.get(activation, ACT_SILU)
        ops[0].static_dims['page_size'] = page_size
        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig."""
        from machete.megakernel import MegakernelConfig
        page_size = ops[0].static_dims.get('page_size', DEFAULT_PAGE_SIZE)
        return MegakernelConfig(page_size=page_size)

    # =========================================================================
    # Forward Load (TMA G->S)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_D,
             x_tma, x_tma_gmem, work_mbar):
        """TMA load x tile (N × tile_S × 1) from global to smem."""
        sX = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_S, 1)),
        )
        gX = cute.local_tile(
            x_tma_gmem, (self.N, self.tile_size_S, 1), (None, None, None),
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
        cute.copy(x_tma, tXgX[(None, Int32(0), tile_S, tile_B)], tXsX,
                  tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Forward Compute (two-phase with barrier)
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D):
        """GLU forward: read gate+up from smem x, compute y, write y to smem.

        Phase 1: All warps read from smem x (N-stride per row), compute y in
                 registers. Uses range_constexpr to build list of reg fragments.
        Barrier: named_barrier_sync ensures all warps done reading x.
        Phase 2: Write y to smem at offset 0 with D-stride per row.
        """
        x_smem = cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem)

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        thr_layout = cute.make_layout(32)

        row_start = tile_S * Int32(self.tile_size_S)

        # Phase 1: compute y into registers for all rows assigned to this warp
        y_regs = []
        for ri in cutlass.range_constexpr(self.max_rows_per_warp):
            local_row = warp_idx + Int32(ri * self.num_warps)

            # Allocate register fragment (even for OOB rows — keeps list regular)
            dummy_row = cute.make_tensor(x_smem, cute.make_layout(self.D))
            dummy_part = cute.local_partition(dummy_row, thr_layout, lane_idx)
            y_reg = cute.make_fragment_like(dummy_part)
            for i in range(cute.size(y_reg)):
                y_reg[i] = Float32(0.0).to(self.x_dtype)

            if local_row < Int32(self.tile_size_S):
                row_idx = row_start + local_row
                if row_idx < Int32(self.S):
                    # Read gate (first D elements of smem row, N-stride)
                    gate_row = cute.make_tensor(
                        x_smem + local_row * Int32(self.N),
                        cute.make_layout(self.D),
                    )
                    gate_part = cute.local_partition(
                        gate_row, thr_layout, lane_idx)
                    gate_reg = cute.make_fragment_like(gate_part)
                    cute.autovec_copy(gate_part, gate_reg)

                    # Read up (last D elements of smem row, N-stride)
                    up_row = cute.make_tensor(
                        x_smem + local_row * Int32(self.N) + Int32(self.D),
                        cute.make_layout(self.D),
                    )
                    up_part = cute.local_partition(
                        up_row, thr_layout, lane_idx)
                    up_reg = cute.make_fragment_like(up_part)
                    cute.autovec_copy(up_part, up_reg)

                    # y = act(gate) * up in float32
                    for i in range(cute.size(gate_reg)):
                        g = gate_reg[i].to(Float32)
                        u = up_reg[i].to(Float32)

                        act_val = g  # CuTe DSL: must init before dynamic if
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

            y_regs.append(y_reg)

        # Barrier: all warps done reading x from smem
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        # Phase 2: write y to smem at offset 0, D-stride per row
        y_smem = cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem)
        for ri in cutlass.range_constexpr(self.max_rows_per_warp):
            local_row = warp_idx + Int32(ri * self.num_warps)
            if local_row < Int32(self.tile_size_S):
                y_row = cute.make_tensor(
                    y_smem + local_row * Int32(self.D),
                    cute.make_layout(self.D),
                )
                y_part = cute.local_partition(y_row, thr_layout, lane_idx)
                cute.autovec_copy(y_regs[ri], y_part)

    # =========================================================================
    # Forward Store (TMA S->G)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_B, tile_S, tile_D,
              y_tma, y_tma_gmem):
        """TMA store y (D × tile_S × 1) from smem[0] to global."""
        sY = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_S, 1)),
        )
        gY = cute.local_tile(
            y_tma_gmem, (self.D, self.tile_size_S, 1), (None, None, None),
        )
        tYsY, tYgY = cute.nvgpu.cpasync.tma_partition(
            y_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sY, 0, 3),
            cute.group_modes(gY, 0, 3),
        )
        with cute.arch.elect_one():
            cute.copy(y_tma, tYsY, tYgY[(None, tile_D, tile_S, tile_B)])


# =============================================================================
# Backward Op
# =============================================================================


class GLUBwdOp(Op):
    """GLU backward — TMA pipelined.

    dy (B,S,D), x (B,S,N=2D) → dx (B,S,N=2D)
    where dx[:,:,:D] = d_gate, dx[:,:,D:] = d_up.

    d_gate = dy * up * act'(gate)
    d_up   = dy * act(gate)

    TMA loads x, dy read from global, dx written to smem (overwrites x,
    same N-stride — no barrier needed), TMA stores dx.
    """

    reads = {
        "dy": (None, ("B", "S", "D")),
        "x":  (None, ("B", "S", "N")),
    }
    writes = {
        "dx": (None, ("B", "S", "N")),
    }
    tile = ("B", "S", "D")

    tma_loads = {"x"}
    tma_stores = {"dx"}

    def __init__(self, **config):
        super().__init__(**config)
        # x_dtype comes from the first read tensor (dy). Override to get x's.
        self.x_dtype = getattr(self, 'x_dtype', self.dy_dtype)
        self.activation = getattr(self, 'activation', ACT_SILU)
        self.page_size = getattr(self, 'page_size', DEFAULT_PAGE_SIZE)

        if self.x_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        assert self.D >= 32 and self.D % 32 == 0, (
            f"GLU requires D >= 32 and D % 32 == 0, got D={self.D}")

        # x and dx have same shape (N per row) → smem reuse, no barrier.
        self.x_tile_bytes = self.tile_size_S * self.N * self.elem_bytes
        assert self.x_tile_bytes <= self.page_size, (
            f"GLU bwd: x tile ({self.x_tile_bytes}B) > page ({self.page_size}B)")

        self.num_warps = self.threads_per_row // 32

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule(cls, tile_sizes=None, activation='silu',
                         page_size=DEFAULT_PAGE_SIZE, **tensors):
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
        if "S" not in tile_sizes:
            N = x.shape[-1]
            elem_bytes = x.element_size()
            tile_sizes["S"] = max(1, page_size // (N * elem_bytes))
        tile_sizes.setdefault("B", 1)

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims['activation'] = ACT_MAP.get(activation, ACT_SILU)
        ops[0].static_dims['page_size'] = page_size
        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig."""
        from machete.megakernel import MegakernelConfig
        page_size = ops[0].static_dims.get('page_size', DEFAULT_PAGE_SIZE)
        return MegakernelConfig(page_size=page_size)

    # =========================================================================
    # Backward Load (TMA G->S)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_D,
             x_tma, x_tma_gmem, work_mbar):
        """TMA load x tile (N × tile_S × 1) from global to smem."""
        sX = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_S, 1)),
        )
        gX = cute.local_tile(
            x_tma_gmem, (self.N, self.tile_size_S, 1), (None, None, None),
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
        cute.copy(x_tma, tXgX[(None, Int32(0), tile_S, tile_B)], tXsX,
                  tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Backward Compute
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D, dy, x, dx):
        """GLU backward: read x from smem, dy from global, write dx to smem.

        dx overwrites x in smem (same N-stride per row, no barrier needed).
        Each warp processes disjoint rows.
        """
        x_smem = cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem)

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        thr_layout = cute.make_layout(32)

        row_start = tile_S * Int32(self.tile_size_S)

        for local_row in range(warp_idx, self.tile_size_S, self.num_warps):
            row_idx = row_start + Int32(local_row)

            if row_idx < Int32(self.S):
                # Read dy from global
                dy_offset = (tile_B * Int32(self.S * self.D)
                             + row_idx * Int32(self.D))
                dy_row = cute.make_tensor(
                    dy.iterator + dy_offset,
                    cute.make_layout(self.D),
                )
                dy_part = cute.local_partition(dy_row, thr_layout, lane_idx)
                dy_reg = cute.make_fragment_like(dy_part)
                cute.autovec_copy(dy_part, dy_reg)

                # Read gate from smem (first D of row)
                gate_row = cute.make_tensor(
                    x_smem + Int32(local_row) * Int32(self.N),
                    cute.make_layout(self.D),
                )
                gate_part = cute.local_partition(
                    gate_row, thr_layout, lane_idx)
                gate_reg = cute.make_fragment_like(gate_part)
                cute.autovec_copy(gate_part, gate_reg)

                # Read up from smem (last D of row)
                up_row = cute.make_tensor(
                    x_smem + Int32(local_row) * Int32(self.N) + Int32(self.D),
                    cute.make_layout(self.D),
                )
                up_part = cute.local_partition(
                    up_row, thr_layout, lane_idx)
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
                        dgate_reg[i] = (d * u).to(self.x_dtype)
                        dup_reg[i] = (d * g).to(self.x_dtype)
                    elif self.activation == ACT_RELU:
                        if g > Float32(0.0):
                            dgate_reg[i] = (d * u).to(self.x_dtype)
                            dup_reg[i] = (d * g).to(self.x_dtype)
                        else:
                            dgate_reg[i] = Float32(0.0).to(self.x_dtype)
                            dup_reg[i] = Float32(0.0).to(self.x_dtype)
                    elif self.activation == ACT_SILU:
                        neg_g = Float32(0.0) - g
                        exp_neg = cute.math.exp(neg_g, fastmath=True)
                        sig = Float32(1.0) / (Float32(1.0) + exp_neg)
                        silu_val = g * sig
                        silu_grad = sig * (Float32(1.0) + g * (Float32(1.0) - sig))
                        dup_reg[i] = (d * silu_val).to(self.x_dtype)
                        dgate_reg[i] = (d * u * silu_grad).to(self.x_dtype)

                # Write dx to smem (overwrites x, same N-stride)
                # d_gate at [0..D), d_up at [D..N)
                dgate_row = cute.make_tensor(
                    x_smem + Int32(local_row) * Int32(self.N),
                    cute.make_layout(self.D),
                )
                dgate_part = cute.local_partition(
                    dgate_row, thr_layout, lane_idx)
                cute.autovec_copy(dgate_reg, dgate_part)

                dup_row = cute.make_tensor(
                    x_smem + Int32(local_row) * Int32(self.N) + Int32(self.D),
                    cute.make_layout(self.D),
                )
                dup_part = cute.local_partition(
                    dup_row, thr_layout, lane_idx)
                cute.autovec_copy(dup_reg, dup_part)

    # =========================================================================
    # Backward Store (TMA S->G)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_B, tile_S, tile_D,
              dx_tma, dx_tma_gmem):
        """TMA store dx (N × tile_S × 1) from smem to global."""
        sDX = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_S, 1)),
        )
        gDX = cute.local_tile(
            dx_tma_gmem, (self.N, self.tile_size_S, 1), (None, None, None),
        )
        tDXsDX, tDXgDX = cute.nvgpu.cpasync.tma_partition(
            dx_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sDX, 0, 3),
            cute.group_modes(gDX, 0, 3),
        )
        with cute.arch.elect_one():
            cute.copy(dx_tma, tDXsDX, tDXgDX[(None, Int32(0), tile_S, tile_B)])
