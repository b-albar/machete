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

Pipelined load/compute/store with smem reuse:
    load:    DMA warp loads K-1 halo rows from global + TMA G->S (x tile)
    compute: Phase 1 — read x from smem, convolve into register accumulators.
             Barrier sync (all warps done reading x).
             Phase 2 — write y from registers to smem (overwrites input region).
    store:   TMA S->G (y tile from smem offset 0)

Smem layout within a page:
    [0, halo_bytes)                            — backward halo (K-1 rows, DMA loaded)
    [halo_bytes, halo_bytes + x_tile_bytes)    — x input tile (TMA loaded)

    After barrier, y overwrites [0, y_tile_bytes) of the same region.
    tile_S = page_size/(D*eb) - (K-1), nearly doubling capacity vs separate y.

Halo handling:
    The DMA warp pre-loads K-1 halo rows into smem before issuing the TMA load.
    For tile 0, halo positions before the sequence start are zero-filled.
    mbarrier_arrive_expect_tx release semantics order halo writes before compute.

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
from machete.megakernel.interpreter import (
    mbarrier_arrive_expect_tx,
    named_barrier_sync,
)


# Activation type constants
ACT_NONE = 0
ACT_SILU = 1

ACT_MAP = {None: ACT_NONE, 'none': ACT_NONE, 'silu': ACT_SILU, 'swish': ACT_SILU}


class Conv1dOp(Op):
    """Causal depthwise Conv1d for the megakernel framework.

    DMA warp loads K-1 backward halo rows from global memory, then TMA loads
    the main x tile. Compute reads from contiguous halo+tile smem (no global
    fallback). TMA stores y.

    Tensor declarations:
        x: (B, L, D) — input tensor
        w: (K, D)    — depthwise conv weight (transposed internally)
        y: (B, L, D) — output tensor

    Tiling:
        tile_B=1, tile_D=D (full extent), tile_S auto-computed.
        (2 * tile_S + K - 1) * D * elem_bytes <= page_size.
    """

    reads = {
        "x": (None, ("B", "S", "D")),
        "w": (None, ("K", "D")),
    }
    writes = {
        "y": (None, ("B", "S", "D")),
    }
    tile = ("B", "S", "D")

    tma_loads = {"x"}
    tma_stores = {"y"}
    peer_stores = {"y"}

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

        self.halo_size = self.K - 1
        self.halo_bytes = self.halo_size * self.D * self.elem_bytes
        self.x_tile_bytes = self.tile_size_S * self.D * self.elem_bytes
        # Smem reuse: y overwrites input region at offset 0 after barrier
        self.y_smem_offset = 0
        self.total_smem = self.halo_bytes + self.x_tile_bytes  # No separate y

        assert self.total_smem <= self.page_size, (
            f"Conv1dOp: tile smem ({self.total_smem}B) exceeds "
            f"page_size ({self.page_size}B). Reduce tile_size_S={self.tile_size_S}."
        )

        self.num_warps = self.threads_per_row // 32
        self.max_rows_per_warp = (
            (self.tile_size_S + self.num_warps - 1) // self.num_warps
        )

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule_forward(cls, tile_sizes=None, activation=None,
                         page_size=DEFAULT_PAGE_SIZE, **tensors):
        """Schedule causal conv1d forward.

        Args:
            tile_sizes: optional {"S": int} override.
            activation: None, 'silu', or 'swish'.
            page_size: shared memory page size in bytes.
            **tensors: x (B,S,D) required, w (D,K) required,
                       y (B,S,D) optional (allocated if omitted).
        """
        tile_sizes = dict(tile_sizes or {})

        # Transpose w from user layout (D, K) to internal layout (K, D)
        w = tensors.get('w')
        K = None
        if w is not None and w.ndim == 2:
            D_w, K = w.shape
            tensors['w'] = w.t().contiguous()  # (K, D)

        x = tensors.get('x')
        if x is not None:
            D = x.shape[2]
            elem_bytes = x.element_size()
            if "S" not in tile_sizes:
                # Smem reuse: halo (K-1 rows) + x tile (tile_S rows), y overwrites
                # (tile_S + K - 1) * D * elem_bytes <= page_size
                halo = (K - 1) if K is not None else 0
                total_rows = page_size // (D * elem_bytes)
                tile_sizes["S"] = max(1, total_rows - halo)
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
    # Forward Load (halo from global + TMA G->S)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_D,
             x, w, y, x_tma, x_tma_gmem, work_mbar):
        """Load K-1 backward halo rows from global, then TMA load x tile.

        All 32 DMA warp threads collaborate on halo loading for bandwidth.
        mbarrier_arrive_expect_tx release semantics order halo writes before
        compute's mbarrier_wait acquire.
        """
        lane_idx = cute.arch.lane_idx()
        thr_layout = cute.make_layout(32)

        b = tile_B
        l_start = tile_S * Int32(self.tile_size_S)
        halo_smem = cute.make_ptr(
            self.x_dtype, page_ptr, cute.AddressSpace.smem
        )

        # --- Load K-1 backward halo rows ---
        for j in cutlass.range_constexpr(self.K - 1):
            # Global position for halo row j: l_start - (K-1) + j
            halo_l = l_start - Int32(self.K - 1 - j)

            halo_row = cute.make_tensor(
                halo_smem + j * self.D,
                cute.make_layout(self.D),
            )
            halo_part = cute.local_partition(halo_row, thr_layout, lane_idx)

            # Valid halo position — read from global
            if halo_l >= Int32(0):
                xg_row = cute.make_tensor(
                    x.iterator + b * self.S * self.D + halo_l * self.D,
                    cute.make_layout(self.D),
                )
                xg_part = cute.local_partition(xg_row, thr_layout, lane_idx)
                xg_reg = cute.make_fragment_like(xg_part)
                cute.autovec_copy(xg_part, xg_reg)
                cute.autovec_copy(xg_reg, halo_part)

            # OOB (before sequence start) — zero fill
            if halo_l < Int32(0):
                zero_reg = cute.make_fragment_like(halo_part)
                for i in range(cute.size(zero_reg)):
                    zero_reg[i] = Float32(0.0).to(self.x_dtype)
                cute.autovec_copy(zero_reg, halo_part)

        # --- TMA load main x tile at halo_bytes offset ---
        sX = cute.make_tensor(
            cute.make_ptr(
                self.x_dtype, page_ptr + self.halo_bytes,
                cute.AddressSpace.smem,
            ),
            cute.make_layout((self.D, self.tile_size_S, 1)),
        )
        gX = cute.local_tile(
            x_tma_gmem, (self.D, self.tile_size_S, 1), (None, None, None),
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
        # Release semantics on arrive orders prior halo smem writes
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(x_tma, tXgX[(None, tile_D, tile_S, tile_B)], tXsX,
                  tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Forward Compute
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D, x, w, y):
        """Causal depthwise conv1d with smem reuse.

        Two-phase compute:
          Phase 1: Read x from smem [halo + x_tile], convolve into register accumulators.
          Barrier: named_barrier_sync ensures all warps done reading x.
          Phase 2: Write y from registers to smem[0..tile_S] (overwrites input region).
        """
        x_smem = cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem)

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        thr_layout = cute.make_layout(32)

        l_start = tile_S * self.tile_size_S

        # Phase 1: Compute y into register arrays (range_constexpr for list building)
        y_regs = []
        for ri in cutlass.range_constexpr(self.max_rows_per_warp):
            local_l = warp_idx + Int32(ri * self.num_warps)

            # Create accumulator fragment (always, even for OOB — keeps list regular)
            dummy_row = cute.make_tensor(x_smem, cute.make_layout(self.D))
            dummy_part = cute.local_partition(dummy_row, thr_layout, lane_idx)
            acc_reg = cute.make_fragment_like(dummy_part, Float32)
            for i in range(cute.size(acc_reg)):
                acc_reg[i] = Float32(0.0)

            l_idx = l_start + local_l
            if local_l < Int32(self.tile_size_S):
                if l_idx < Int32(self.S):
                    # Convolve: load weight per-tap (L1-cached), read x from smem
                    for j in cutlass.range_constexpr(self.K):
                        w_row = cute.make_tensor(
                            w.iterator + (self.K - 1 - j) * self.D,
                            cute.make_layout(self.D),
                        )
                        w_part = cute.local_partition(w_row, thr_layout, lane_idx)
                        w_reg = cute.make_fragment_like(w_part)
                        cute.autovec_copy(w_part, w_reg)

                        smem_row = local_l + Int32(self.K - 1 - j)
                        xs_row = cute.make_tensor(
                            x_smem + smem_row * self.D,
                            cute.make_layout(self.D),
                        )
                        xs_part = cute.local_partition(xs_row, thr_layout, lane_idx)
                        xs_reg = cute.make_fragment_like(xs_part)
                        cute.autovec_copy(xs_part, xs_reg)

                        for i in range(cute.size(acc_reg)):
                            acc_reg[i] = acc_reg[i] + w_reg[i].to(Float32) * xs_reg[i].to(Float32)

                    # Optional SiLU activation
                    if self.activation == ACT_SILU:
                        for i in range(cute.size(acc_reg)):
                            val = acc_reg[i]
                            neg_val = Float32(0.0) - val
                            exp_neg = cute.math.exp(neg_val, fastmath=True)
                            acc_reg[i] = val / (Float32(1.0) + exp_neg)

            y_regs.append(acc_reg)

        # Barrier: all compute warps done reading x from smem
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        # Phase 2: Write y from registers to smem (overwriting input region)
        y_smem = cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem)
        for ri in cutlass.range_constexpr(self.max_rows_per_warp):
            local_l = warp_idx + Int32(ri * self.num_warps)
            if local_l < Int32(self.tile_size_S):
                out_row = cute.make_tensor(
                    y_smem + local_l * self.D,
                    cute.make_layout(self.D),
                )
                out_part = cute.local_partition(out_row, thr_layout, lane_idx)
                out_reg = cute.make_fragment_like(out_part)
                for i in range(cute.size(out_reg)):
                    out_reg[i] = y_regs[ri][i].to(self.x_dtype)
                cute.autovec_copy(out_reg, out_part)

    # =========================================================================
    # Forward Store (3D TMA S->G)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_B, tile_S, tile_D, y_tma, y_tma_gmem):
        """TMA store y from smem offset 0 (reused input region) to global."""
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

    # =========================================================================
    # Communicate (TMA S->G to peer GPU)
    # =========================================================================

    @cute.jit
    def communicate(self, page_ptr, tile_B, tile_S, tile_D,
                    y_p0_tma, y_p0_tma_gmem):
        """Send y tile to peer GPU 0 via TMA S2G."""
        sY = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_S, 1)),
        )
        gY = cute.local_tile(
            y_p0_tma_gmem, (self.D, self.tile_size_S, 1), (None, None, None),
        )
        tYsY, tYgY = cute.nvgpu.cpasync.tma_partition(
            y_p0_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sY, 0, 3),
            cute.group_modes(gY, 0, 3),
        )
        with cute.arch.elect_one():
            cute.copy(y_p0_tma, tYsY, tYgY[(None, tile_D, tile_S, tile_B)])


class Conv1dBwdOp(Op):
    """Causal depthwise Conv1d backward (dx gradient) for the megakernel.

    Computes dx[b,l,d] = sum_{j=0}^{K-1} w[K-1-j,d] * dy[b, l+j, d]
    where dy[b, l+j, d] = 0 when l+j >= L.

    DMA warp loads K-1 forward halo rows from global memory after TMA loads
    the main dy tile. Compute reads from contiguous dy+halo smem.

    Smem layout within a page:
        [0, dy_tile_bytes)                     — dy input (TMA loaded)
        [dy_tile_bytes, dy_full_bytes)         — forward halo (K-1 rows, DMA loaded)
        [dy_full_bytes, dy_full_bytes + dx_bytes) — dx output (TMA stored)
    """

    reads = {
        "dy": (None, ("B", "S", "D")),
        "w":  (None, ("K", "D")),
    }
    writes = {
        "dx": (None, ("B", "S", "D")),
    }
    tile = ("B", "S", "D")

    tma_loads = {"dy"}
    tma_stores = {"dx"}
    peer_stores = {"dx"}

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, 'page_size', DEFAULT_PAGE_SIZE)

        if self.dy_dtype == cutlass.Float32:
            self.elem_bytes = 4
        elif self.dy_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        self.halo_size = self.K - 1
        self.halo_bytes = self.halo_size * self.D * self.elem_bytes
        self.dy_tile_bytes = self.tile_size_S * self.D * self.elem_bytes
        self.dy_full_bytes = self.dy_tile_bytes + self.halo_bytes
        # Smem reuse: dx overwrites input region at offset 0 after barrier
        self.dx_smem_offset = 0
        self.dx_tile_bytes = self.tile_size_S * self.D * self.elem_bytes
        self.total_smem = self.dy_full_bytes  # No separate dx region

        assert self.total_smem <= self.page_size, (
            f"Conv1dBwdOp: tile smem ({self.total_smem}B) exceeds "
            f"page_size ({self.page_size}B). Reduce tile_size_S={self.tile_size_S}."
        )

        self.num_warps = self.threads_per_row // 32
        self.max_rows_per_warp = (
            (self.tile_size_S + self.num_warps - 1) // self.num_warps
        )

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule_forward(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE,
                         **tensors):
        """Schedule causal conv1d backward (dx).

        Args:
            tile_sizes: optional {"S": int} override.
            page_size: shared memory page size in bytes.
            **tensors: dy (B,S,D) required, w (D,K) required,
                       dx (B,S,D) optional (allocated if omitted).
        """
        tile_sizes = dict(tile_sizes or {})

        # Transpose w from user layout (D, K) to internal layout (K, D)
        w = tensors.get('w')
        K = None
        if w is not None and w.ndim == 2:
            D, K = w.shape
            tensors['w'] = w.t().contiguous()  # (K, D)

        dy = tensors.get('dy')
        if dy is not None:
            D = dy.shape[2]
            elem_bytes = dy.element_size()
            if "S" not in tile_sizes:
                # Smem reuse: dy tile (tile_S) + halo (K-1), dx overwrites
                # (tile_S + K - 1) * D * elem_bytes <= page_size
                halo = (K - 1) if K is not None else 0
                total_rows = page_size // (D * elem_bytes)
                tile_sizes["S"] = max(1, total_rows - halo)
            tile_sizes.setdefault("B", 1)
            tile_sizes["D"] = D

        if 'dx' not in tensors:
            tensors['dx'] = torch.empty_like(dy)

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims['page_size'] = page_size
        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for Conv1dBwdOp."""
        from machete.megakernel import MegakernelConfig
        page_size = ops[0].static_dims.get('page_size', DEFAULT_PAGE_SIZE)
        return MegakernelConfig(page_size=page_size)

    # =========================================================================
    # Backward Load (TMA G->S for dy + forward halo from global)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_D,
             dy, w, dx, dy_tma, dy_tma_gmem, work_mbar):
        """Load K-1 forward halo rows from global, then TMA load dy tile.

        Halo rows go into smem after the main tile for contiguous access.
        """
        lane_idx = cute.arch.lane_idx()
        thr_layout = cute.make_layout(32)

        b = tile_B
        l_start = tile_S * Int32(self.tile_size_S)
        halo_smem = cute.make_ptr(
            self.dy_dtype,
            page_ptr + self.dy_tile_bytes,
            cute.AddressSpace.smem,
        )

        # --- Load K-1 forward halo rows ---
        for j in cutlass.range_constexpr(self.K - 1):
            # Global position: l_start + tile_size_S + j
            halo_l = l_start + Int32(self.tile_size_S + j)

            halo_row = cute.make_tensor(
                halo_smem + j * self.D,
                cute.make_layout(self.D),
            )
            halo_part = cute.local_partition(halo_row, thr_layout, lane_idx)

            # Valid position — read from global
            if halo_l < Int32(self.S):
                dyg_row = cute.make_tensor(
                    dy.iterator + b * self.S * self.D + halo_l * self.D,
                    cute.make_layout(self.D),
                )
                dyg_part = cute.local_partition(dyg_row, thr_layout, lane_idx)
                dyg_reg = cute.make_fragment_like(dyg_part)
                cute.autovec_copy(dyg_part, dyg_reg)
                cute.autovec_copy(dyg_reg, halo_part)

            # OOB (past sequence end) — zero fill
            if halo_l >= Int32(self.S):
                zero_reg = cute.make_fragment_like(halo_part)
                for i in range(cute.size(zero_reg)):
                    zero_reg[i] = Float32(0.0).to(self.dy_dtype)
                cute.autovec_copy(zero_reg, halo_part)

        # --- TMA load main dy tile ---
        sDY = cute.make_tensor(
            cute.make_ptr(self.dy_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_S, 1)),
        )
        gDY = cute.local_tile(
            dy_tma_gmem, (self.D, self.tile_size_S, 1), (None, None, None),
        )
        tDYsDY, tDYgDY = cute.nvgpu.cpasync.tma_partition(
            dy_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sDY, 0, 3),
            cute.group_modes(gDY, 0, 3),
        )

        nbytes = Int32(self.dy_tile_bytes)
        mbar_ptr = cute.make_ptr(
            cutlass.Int64, work_mbar, cute.AddressSpace.smem
        )
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(dy_tma, tDYgDY[(None, tile_D, tile_S, tile_B)], tDYsDY,
                  tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Backward Compute
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D, dy, w, dx):
        """Backward conv1d with smem reuse.

        Two-phase compute:
          Phase 1: Read dy from smem [dy_tile + halo], convolve into register accumulators.
          Barrier: named_barrier_sync ensures all warps done reading dy.
          Phase 2: Write dx from registers to smem[0..tile_S] (overwrites dy region).
        """
        dy_smem = cute.make_ptr(self.dy_dtype, page_ptr, cute.AddressSpace.smem)

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        thr_layout = cute.make_layout(32)

        l_start = tile_S * self.tile_size_S

        # Phase 1: Compute dx into register arrays
        dx_regs = []
        for ri in cutlass.range_constexpr(self.max_rows_per_warp):
            local_l = warp_idx + Int32(ri * self.num_warps)

            dummy_row = cute.make_tensor(dy_smem, cute.make_layout(self.D))
            dummy_part = cute.local_partition(dummy_row, thr_layout, lane_idx)
            acc_reg = cute.make_fragment_like(dummy_part, Float32)
            for i in range(cute.size(acc_reg)):
                acc_reg[i] = Float32(0.0)

            l_idx = l_start + local_l
            if local_l < Int32(self.tile_size_S):
                if l_idx < Int32(self.S):
                    # Convolve: load weight per-tap (L1-cached), read dy from smem
                    for j in cutlass.range_constexpr(self.K):
                        w_row = cute.make_tensor(
                            w.iterator + (self.K - 1 - j) * self.D,
                            cute.make_layout(self.D),
                        )
                        w_part = cute.local_partition(w_row, thr_layout, lane_idx)
                        w_reg = cute.make_fragment_like(w_part)
                        cute.autovec_copy(w_part, w_reg)

                        smem_row = local_l + Int32(j)
                        dys_row = cute.make_tensor(
                            dy_smem + smem_row * self.D,
                            cute.make_layout(self.D),
                        )
                        dys_part = cute.local_partition(dys_row, thr_layout, lane_idx)
                        dys_reg = cute.make_fragment_like(dys_part)
                        cute.autovec_copy(dys_part, dys_reg)

                        for i in range(cute.size(acc_reg)):
                            acc_reg[i] = acc_reg[i] + w_reg[i].to(Float32) * dys_reg[i].to(Float32)

            dx_regs.append(acc_reg)

        # Barrier: all compute warps done reading dy from smem
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        # Phase 2: Write dx from registers to smem (overwriting dy region)
        dx_smem = cute.make_ptr(self.dy_dtype, page_ptr, cute.AddressSpace.smem)
        for ri in cutlass.range_constexpr(self.max_rows_per_warp):
            local_l = warp_idx + Int32(ri * self.num_warps)
            if local_l < Int32(self.tile_size_S):
                out_row = cute.make_tensor(
                    dx_smem + local_l * self.D,
                    cute.make_layout(self.D),
                )
                out_part = cute.local_partition(out_row, thr_layout, lane_idx)
                out_reg = cute.make_fragment_like(out_part)
                for i in range(cute.size(out_reg)):
                    out_reg[i] = dx_regs[ri][i].to(self.dy_dtype)
                cute.autovec_copy(out_reg, out_part)

    # =========================================================================
    # Backward Store (3D TMA S->G)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_B, tile_S, tile_D, dx_tma, dx_tma_gmem):
        """TMA store dx from smem offset 0 (reused input region) to global."""
        sDX = cute.make_tensor(
            cute.make_ptr(self.dy_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_S, 1)),
        )
        gDX = cute.local_tile(
            dx_tma_gmem, (self.D, self.tile_size_S, 1), (None, None, None),
        )
        tDXsDX, tDXgDX = cute.nvgpu.cpasync.tma_partition(
            dx_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sDX, 0, 3),
            cute.group_modes(gDX, 0, 3),
        )
        with cute.arch.elect_one():
            cute.copy(dx_tma, tDXsDX, tDXgDX[(None, tile_D, tile_S, tile_B)])

    # =========================================================================
    # Communicate (TMA S->G to peer GPU)
    # =========================================================================

    @cute.jit
    def communicate(self, page_ptr, tile_B, tile_S, tile_D,
                    dx_p0_tma, dx_p0_tma_gmem):
        """Send dx tile to peer GPU 0 via TMA S2G."""
        sDX = cute.make_tensor(
            cute.make_ptr(self.dy_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_S, 1)),
        )
        gDX = cute.local_tile(
            dx_p0_tma_gmem, (self.D, self.tile_size_S, 1), (None, None, None),
        )
        tDXsDX, tDXgDX = cute.nvgpu.cpasync.tma_partition(
            dx_p0_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sDX, 0, 3),
            cute.group_modes(gDX, 0, 3),
        )
        with cute.arch.elect_one():
            cute.copy(dx_p0_tma, tDXsDX, tDXgDX[(None, tile_D, tile_S, tile_B)])


__all__ = ["Conv1dOp", "Conv1dBwdOp"]
