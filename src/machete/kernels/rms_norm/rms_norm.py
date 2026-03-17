# Copyright (c) 2025, Machete Authors
"""
RMSNorm Ops for the Megakernel.

Supports all RMSNorm variants via flags:
    - Standard:    y = rmsnorm(x, weight)
    - Residual:    y = rmsnorm(x, w) + x
    - Gemma:       y = rmsnorm(x, (1+w))
    - Fused add:   residual_out = x + residual_in; y = rmsnorm(residual_out, w)
    - Gated:       y = rmsnorm(x, w) * silu(gate)
    - Per-row weight: weight is (B, S, D) instead of shared (D,)

TMA pipelined load/compute/store with multi-row tiling:
    tile_size_S = (page_size - SCRATCH_BYTES) / (D * elem_bytes)

Forward:  TMA load x → compute (cross-warp sum_sq reduction + normalize) →
          write y to smem (overwrite x, same D-stride) → TMA store y.
          Weight/residual_in/gate read from global.
Backward: TMA load x → compute (two cross-warp reductions) →
          write dx to smem (overwrite x) → TMA store dx.
          dout/weight/gate read from global.

All compute warps cooperate on each row via cross-warp reduction:
    - warp_reduction → scratch smem → named_barrier_sync → sum scratch → sync
    - Forward: 1 barrier per row
    - Backward: 2 barriers per row (double-buffered scratch for 2 reductions)
"""

import operator

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import (
    mbarrier_arrive_expect_tx,
    named_barrier_sync,
)


# =============================================================================
# Constants
# =============================================================================

RMSNORM_EPS = 1e-6
# Scratch for cross-warp reduction: 2 × max_warps × 4 bytes.
# Max 8 warps → 2 × 8 × 4 = 64 bytes.
SCRATCH_BYTES = 64


# =============================================================================
# Helpers
# =============================================================================


def _expand_weight(tensors):
    """Auto-expand 1D weight (D,) to 3D (B, S, D) for uniform handling."""
    w = tensors.get('weight')
    if w is not None and w.ndim == 1:
        B, S = tensors['x'].shape[0], tensors['x'].shape[1]
        tensors['weight'] = w.reshape(1, 1, -1).expand(B, S, -1).contiguous()


def _auto_tile_S(D, elem_bytes, page_size):
    """Compute tile_size_S from page budget minus scratch."""
    usable = page_size - SCRATCH_BYTES
    return max(1, usable // (D * elem_bytes))


def _tma_kernel_config(ops):
    """Return config for TMA-mode RMSNorm.

    threads_per_block includes DMA warps (framework always reserves them).
    Cap at 4 compute warps (128 threads) — RMSNorm is memory-bound,
    more warps don't improve throughput and >5 MMA warps can hang.
    """
    from machete.megakernel import MegakernelConfig
    from machete.megakernel.megakernel import NUM_DMA_WARPS
    D = ops[0].static_dims.get('D', 4096)
    page_size = ops[0].static_dims.get('page_size', DEFAULT_PAGE_SIZE)
    compute_threads = 64
    for ct in [128, 64]:
        if D % ct == 0:
            compute_threads = ct
            break
    return MegakernelConfig(
        threads_per_block=compute_threads + NUM_DMA_WARPS * 32,
        page_size=page_size,
    )


# =============================================================================
# RMSNorm Forward Op (TMA pipelined)
# =============================================================================


class RMSNormOp(Op):
    """RMSNorm forward — TMA pipelined load/compute/store.

    TMA loads x tile (D × tile_S × 1) into smem. Compute reads from smem,
    does cross-warp reduction for sum_sq, normalizes, writes y to smem
    (overwrite x at offset 0, same D-stride). TMA stores y.

    Weight, residual_in, gate are read from global memory (small or
    non-tiled data).

    Supports all variants via flags: standard, residual, gemma, fused add,
    gated, per-row weight.
    """

    reads = {
        "x": (None, ("B", "S", "D")),
        "weight": (None, ("B", "S", "D")),
        "residual_in": (None, ("B", "S", "D")),
        "gate": (None, ("B", "S", "D")),
    }
    writes = {
        "y": (None, ("B", "S", "D")),
        "residual_out": (None, ("B", "S", "D")),
    }
    tile = ("B", "S", "D")

    tma_loads = {"x"}
    tma_stores = {"y"}

    def __init__(self, **config):
        super().__init__(**config)
        self.residual = getattr(self, 'residual', 0)
        self.gemma = getattr(self, 'gemma', 0)
        self.has_residual = getattr(self, 'has_residual', 0)
        self.has_gate = getattr(self, 'has_gate', 0)
        self.per_row_weight = getattr(self, 'per_row_weight', 0)
        self.page_size = getattr(self, 'page_size', DEFAULT_PAGE_SIZE)
        if self.x_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        assert self.D >= 32 and self.D % 32 == 0

        # Smem layout: x tile at [0, x_tile_bytes), scratch after it.
        self.x_tile_bytes = self.tile_size_S * self.D * self.elem_bytes
        self.scratch_offset = self.x_tile_bytes

        max_warps = min(8, self.threads_per_row // 32)
        max_et = max_warps * 32
        self.effective_threads = 32
        for t in range(32, max_et + 1, 32):
            if self.D % t == 0:
                self.effective_threads = t
        self.effective_warps = self.effective_threads // 32

    kernel_config = staticmethod(_tma_kernel_config)

    @classmethod
    def _fill_dummies(cls, tensors):
        """Fill dummy tensors for optional forward inputs."""
        x, y = tensors['x'], tensors['y']
        has_residual = 'residual_in' in tensors
        has_gate = 'gate' in tensors
        if not has_residual:
            tensors['residual_in'] = x
        if 'residual_out' not in tensors:
            tensors['residual_out'] = y
        if not has_gate:
            tensors['gate'] = x
        return has_residual, has_gate

    @classmethod
    def schedule_forward(cls, tile_sizes=None, residual=False, gemma=False,
                         per_row_weight=False, page_size=DEFAULT_PAGE_SIZE,
                         **tensors):
        tensors = dict(tensors)
        _expand_weight(tensors)
        has_residual, has_gate = cls._fill_dummies(tensors)
        tile_sizes = dict(tile_sizes or {})
        if "S" not in tile_sizes:
            D = tensors['x'].shape[-1]
            elem_bytes = tensors['x'].element_size()
            tile_sizes["S"] = _auto_tile_S(D, elem_bytes, page_size)
        tile_sizes.setdefault("B", 1)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        if residual:
            ops[0].static_dims['residual'] = 1
        if gemma:
            ops[0].static_dims['gemma'] = 1
        if has_residual:
            ops[0].static_dims['has_residual'] = 1
        if has_gate:
            ops[0].static_dims['has_gate'] = 1
        if per_row_weight:
            ops[0].static_dims['per_row_weight'] = 1
        ops[0].static_dims['page_size'] = page_size
        return ops

    # =========================================================================
    # TMA Load (G->S)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_D,
             x_tma, x_tma_gmem, work_mbar):
        """TMA load x tile (D × tile_S × 1) from global to smem."""
        sX = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
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
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(x_tma, tXgX[(None, tile_D, tile_S, tile_B)], tXsX,
                  tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Forward Compute
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D,
                x, weight, residual_in, gate, y, residual_out):
        """RMSNorm forward: read x from smem, write y to smem (overwrite x).

        Phase 1: Read x from smem, apply fused-add if needed, cross-warp
                 reduce for sum_sq, compute rstd, compute y into registers.
        Barrier: named_barrier_sync ensures all warps done reading x.
        Phase 2: Write y to smem at offset 0 (overwrites x, same D-stride).
        """
        x_smem = cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem)
        scratch = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float32,
                page_ptr + Int32(self.scratch_offset),
                cute.AddressSpace.smem,
            ),
            cute.make_layout(self.effective_warps),
        )

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        tidx = warp_idx * 32 + lane_idx
        thr_layout = cute.make_layout(self.effective_threads)

        if tidx < self.effective_threads:
            row_start = tile_S * Int32(self.tile_size_S)

            # Load weight from global (shared across all rows)
            w_row_0 = cute.make_tensor(
                weight.iterator, cute.make_layout(self.D),
            )
            w_part_0 = cute.local_partition(w_row_0, thr_layout, tidx)
            w_reg = cute.make_fragment_like(w_part_0)

            if const_expr(not self.per_row_weight):
                cute.autovec_copy(w_part_0, w_reg)
                if const_expr(self.gemma):
                    for i in range(cute.size(w_reg)):
                        w_reg[i] = (w_reg[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

            # Process rows sequentially (all warps cooperate per row)
            for local_row in range(self.tile_size_S):
                row_idx = row_start + Int32(local_row)

                if row_idx < Int32(self.S):
                    global_offset = tile_B * Int32(self.S * self.D) + row_idx * Int32(self.D)

                    # Per-row weight if needed
                    if const_expr(self.per_row_weight):
                        pw_row = cute.make_tensor(
                            weight.iterator + global_offset,
                            cute.make_layout(self.D),
                        )
                        pw_part = cute.local_partition(pw_row, thr_layout, tidx)
                        cute.autovec_copy(pw_part, w_reg)
                        if const_expr(self.gemma):
                            for i in range(cute.size(w_reg)):
                                w_reg[i] = (w_reg[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

                    # Read x from smem
                    x_row = cute.make_tensor(
                        x_smem + Int32(local_row * self.D),
                        cute.make_layout(self.D),
                    )
                    x_part = cute.local_partition(x_row, thr_layout, tidx)
                    x_reg = cute.make_fragment_like(x_part)
                    cute.autovec_copy(x_part, x_reg)

                    # Fused add: x_reg += residual_in
                    if const_expr(self.has_residual):
                        res_row = cute.make_tensor(
                            residual_in.iterator + global_offset,
                            cute.make_layout(self.D),
                        )
                        res_part = cute.local_partition(res_row, thr_layout, tidx)
                        res_reg = cute.make_fragment_like(res_part)
                        cute.autovec_copy(res_part, res_reg)

                        for i in range(cute.size(x_reg)):
                            x_reg[i] = (x_reg[i].to(Float32) + res_reg[i].to(Float32)).to(self.x_dtype)

                        # Write residual_out to global
                        res_out_row = cute.make_tensor(
                            residual_out.iterator + global_offset,
                            cute.make_layout(self.D),
                        )
                        res_out_part = cute.local_partition(res_out_row, thr_layout, tidx)
                        cute.autovec_copy(x_reg, res_out_part)

                    # Cross-warp reduction for sum_sq
                    partial_sq = Float32(0.0)
                    for i in range(cute.size(x_reg)):
                        val = x_reg[i].to(Float32)
                        partial_sq = partial_sq + val * val

                    warp_sum = cute.arch.warp_reduction(partial_sq, operator.add)
                    if lane_idx == 0:
                        scratch[warp_idx] = warp_sum
                    named_barrier_sync(Int32(2), Int32(self.effective_threads))

                    sum_sq = Float32(0.0)
                    for wi in range(self.effective_warps):
                        sum_sq = sum_sq + scratch[wi]

                    rstd = cute.math.rsqrt(
                        sum_sq / self.D + RMSNORM_EPS, fastmath=True
                    )

                    # Compute y
                    y_reg = cute.make_fragment_like(x_reg)

                    if const_expr(self.has_gate):
                        gate_row = cute.make_tensor(
                            gate.iterator + global_offset,
                            cute.make_layout(self.D),
                        )
                        gate_part = cute.local_partition(gate_row, thr_layout, tidx)
                        gate_reg = cute.make_fragment_like(gate_part)
                        cute.autovec_copy(gate_part, gate_reg)

                        for i in range(cute.size(x_reg)):
                            normed = x_reg[i].to(Float32) * rstd * w_reg[i].to(Float32)
                            g = gate_reg[i].to(Float32)
                            sig = Float32(1.0) / (Float32(1.0) + cute.math.exp(-g, fastmath=True))
                            silu_g = g * sig
                            y_reg[i] = (normed * silu_g).to(self.x_dtype)
                    else:
                        for i in range(cute.size(x_reg)):
                            val = x_reg[i].to(Float32) * rstd * w_reg[i].to(Float32)
                            if const_expr(self.residual):
                                val = val + x_reg[i].to(Float32)
                            y_reg[i] = val.to(self.x_dtype)

                    # Write y to smem (overwrite x row)
                    y_smem_row = cute.make_tensor(
                        x_smem + Int32(local_row * self.D),
                        cute.make_layout(self.D),
                    )
                    y_smem_part = cute.local_partition(y_smem_row, thr_layout, tidx)
                    cute.autovec_copy(y_reg, y_smem_part)

    # =========================================================================
    # TMA Store (S->G)
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
# RMSNorm Backward Op (TMA pipelined)
# =============================================================================


class RMSNormBwdOp(Op):
    """RMSNorm backward — TMA pipelined load/compute/store.

    TMA loads x into smem. Compute reads x from smem, dout/weight/gate
    from global. Two cross-warp reductions (sum_sq + sum_grad). Writes dx
    to smem (overwrites x, same D-stride). TMA stores dx.
    """

    reads = {
        "dout": (None, ("B", "S", "D")),
        "x": (None, ("B", "S", "D")),
        "weight": (None, ("B", "S", "D")),
        "gate": (None, ("B", "S", "D")),
    }
    writes = {
        "dx": (None, ("B", "S", "D")),
        "d_residual": (None, ("B", "S", "D")),
        "dgate": (None, ("B", "S", "D")),
    }
    tile = ("B", "S", "D")

    tma_loads = {"x"}
    tma_stores = {"dx"}

    def __init__(self, **config):
        super().__init__(**config)
        self.residual = getattr(self, 'residual', 0)
        self.gemma = getattr(self, 'gemma', 0)
        self.has_residual = getattr(self, 'has_residual', 0)
        self.has_gate = getattr(self, 'has_gate', 0)
        self.per_row_weight = getattr(self, 'per_row_weight', 0)
        self.page_size = getattr(self, 'page_size', DEFAULT_PAGE_SIZE)
        if self.x_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        assert self.D >= 32 and self.D % 32 == 0

        self.x_tile_bytes = self.tile_size_S * self.D * self.elem_bytes
        self.scratch_offset = self.x_tile_bytes

        max_warps = min(8, self.threads_per_row // 32)
        max_et = max_warps * 32
        self.effective_threads = 32
        for t in range(32, max_et + 1, 32):
            if self.D % t == 0:
                self.effective_threads = t
        self.effective_warps = self.effective_threads // 32

    kernel_config = staticmethod(_tma_kernel_config)

    @classmethod
    def _fill_dummies(cls, tensors):
        """Fill dummy tensors for optional backward inputs."""
        x, dx = tensors['x'], tensors['dx']
        has_residual = 'd_residual' in tensors
        has_gate = 'gate' in tensors
        if not has_gate:
            tensors['gate'] = x
        if not has_residual:
            tensors['d_residual'] = dx
        if 'dgate' not in tensors:
            tensors['dgate'] = dx
        return has_residual, has_gate

    @classmethod
    def schedule_forward(cls, tile_sizes=None, residual=False, gemma=False,
                         per_row_weight=False, page_size=DEFAULT_PAGE_SIZE,
                         **tensors):
        tensors = dict(tensors)
        _expand_weight(tensors)
        has_residual, has_gate = cls._fill_dummies(tensors)
        tile_sizes = dict(tile_sizes or {})
        if "S" not in tile_sizes:
            D = tensors['x'].shape[-1]
            elem_bytes = tensors['x'].element_size()
            tile_sizes["S"] = _auto_tile_S(D, elem_bytes, page_size)
        tile_sizes.setdefault("B", 1)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        if residual:
            ops[0].static_dims['residual'] = 1
        if gemma:
            ops[0].static_dims['gemma'] = 1
        if has_residual:
            ops[0].static_dims['has_residual'] = 1
        if has_gate:
            ops[0].static_dims['has_gate'] = 1
        if per_row_weight:
            ops[0].static_dims['per_row_weight'] = 1
        ops[0].static_dims['page_size'] = page_size
        return ops

    # =========================================================================
    # TMA Load (G->S)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_D,
             x_tma, x_tma_gmem, work_mbar):
        """TMA load x tile (D × tile_S × 1) from global to smem."""
        sX = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
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
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(x_tma, tXgX[(None, tile_D, tile_S, tile_B)], tXsX,
                  tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Backward Compute
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D,
                dout, x, weight, gate, dx, d_residual, dgate):
        """RMSNorm backward: read x from smem, dout/weight/gate from global.

        Two cross-warp reductions per row (sum_sq, sum_grad).
        Write dx to smem (overwrite x, same D-stride).
        """
        x_smem = cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem)
        scratch = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float32,
                page_ptr + Int32(self.scratch_offset),
                cute.AddressSpace.smem,
            ),
            cute.make_layout(2 * self.effective_warps),
        )

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        tidx = warp_idx * 32 + lane_idx
        thr_layout = cute.make_layout(self.effective_threads)

        if tidx < self.effective_threads:
            row_start = tile_S * Int32(self.tile_size_S)

            # Load weight from global (shared across all rows)
            w_row_0 = cute.make_tensor(
                weight.iterator, cute.make_layout(self.D),
            )
            w_part_0 = cute.local_partition(w_row_0, thr_layout, tidx)
            w_reg = cute.make_fragment_like(w_part_0)

            if const_expr(not self.per_row_weight):
                cute.autovec_copy(w_part_0, w_reg)
                if const_expr(self.gemma):
                    for i in range(cute.size(w_reg)):
                        w_reg[i] = (w_reg[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

            # Process rows sequentially (all warps cooperate per row)
            for local_row in range(self.tile_size_S):
                row_idx = row_start + Int32(local_row)

                if row_idx < Int32(self.S):
                    global_offset = tile_B * Int32(self.S * self.D) + row_idx * Int32(self.D)

                    # Per-row weight if needed
                    if const_expr(self.per_row_weight):
                        pw_row = cute.make_tensor(
                            weight.iterator + global_offset,
                            cute.make_layout(self.D),
                        )
                        pw_part = cute.local_partition(pw_row, thr_layout, tidx)
                        cute.autovec_copy(pw_part, w_reg)
                        if const_expr(self.gemma):
                            for i in range(cute.size(w_reg)):
                                w_reg[i] = (w_reg[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

                    # Read x from smem
                    x_row = cute.make_tensor(
                        x_smem + Int32(local_row * self.D),
                        cute.make_layout(self.D),
                    )
                    x_part = cute.local_partition(x_row, thr_layout, tidx)
                    x_reg = cute.make_fragment_like(x_part)
                    cute.autovec_copy(x_part, x_reg)

                    # Pass 1: sum_sq via cross-warp reduction
                    partial_sq = Float32(0.0)
                    for i in range(cute.size(x_reg)):
                        val = x_reg[i].to(Float32)
                        partial_sq = partial_sq + val * val

                    warp_sum = cute.arch.warp_reduction(partial_sq, operator.add)
                    buf_off = Int32(0)
                    if lane_idx == 0:
                        scratch[buf_off + warp_idx] = warp_sum
                    named_barrier_sync(Int32(2), Int32(self.effective_threads))

                    sum_sq = Float32(0.0)
                    for wi in range(self.effective_warps):
                        sum_sq = sum_sq + scratch[buf_off + wi]

                    rstd = cute.math.rsqrt(
                        sum_sq / self.D + RMSNORM_EPS, fastmath=True
                    )

                    # Load dout from global
                    dout_row = cute.make_tensor(
                        dout.iterator + global_offset, cute.make_layout(self.D),
                    )
                    dout_part = cute.local_partition(dout_row, thr_layout, tidx)
                    dout_reg = cute.make_fragment_like(dout_part)
                    cute.autovec_copy(dout_part, dout_reg)

                    # Pre-allocate gate fragment
                    gate_reg = cute.make_fragment_like(x_part)

                    if const_expr(self.has_gate):
                        gate_row = cute.make_tensor(
                            gate.iterator + global_offset,
                            cute.make_layout(self.D),
                        )
                        gate_part = cute.local_partition(gate_row, thr_layout, tidx)
                        cute.autovec_copy(gate_part, gate_reg)

                    # Pass 2: sum_grad via cross-warp reduction
                    partial_grad = Float32(0.0)
                    buf2_off = Int32(self.effective_warps)

                    if const_expr(self.has_gate):
                        for i in range(cute.size(x_reg)):
                            g = gate_reg[i].to(Float32)
                            sig = Float32(1.0) / (Float32(1.0) + cute.math.exp(-g, fastmath=True))
                            silu_g = g * sig
                            dy_norm = dout_reg[i].to(Float32) * silu_g
                            partial_grad = partial_grad + dy_norm * w_reg[i].to(Float32) * x_reg[i].to(Float32)
                    else:
                        for i in range(cute.size(x_reg)):
                            d = dout_reg[i].to(Float32)
                            partial_grad = partial_grad + d * w_reg[i].to(Float32) * x_reg[i].to(Float32)

                    warp_grad = cute.arch.warp_reduction(partial_grad, operator.add)
                    if lane_idx == 0:
                        scratch[buf2_off + warp_idx] = warp_grad
                    named_barrier_sync(Int32(2), Int32(self.effective_threads))

                    sum_grad = Float32(0.0)
                    for wi in range(self.effective_warps):
                        sum_grad = sum_grad + scratch[buf2_off + wi]

                    mean_grad = sum_grad / self.D

                    # Pass 3: dx [and dgate]
                    dx_reg = cute.make_fragment_like(x_reg)

                    if const_expr(self.has_gate):
                        dgate_reg = cute.make_fragment_like(x_reg)
                        for i in range(cute.size(x_reg)):
                            g = gate_reg[i].to(Float32)
                            sig = Float32(1.0) / (Float32(1.0) + cute.math.exp(-g, fastmath=True))
                            silu_g = g * sig
                            silu_grad = sig * (Float32(1.0) + g * (Float32(1.0) - sig))

                            d = dout_reg[i].to(Float32)
                            x_val = x_reg[i].to(Float32)
                            wi = w_reg[i].to(Float32)

                            dy_norm = d * silu_g
                            dw_x = dy_norm * wi
                            dx_val = (dw_x - x_val * rstd * rstd * mean_grad) * rstd

                            normed = x_val * rstd * wi
                            dgate_val = d * normed * silu_grad

                            dx_reg[i] = dx_val.to(self.x_dtype)
                            dgate_reg[i] = dgate_val.to(self.x_dtype)

                        # Write dgate to global
                        dgate_row = cute.make_tensor(
                            dgate.iterator + global_offset,
                            cute.make_layout(self.D),
                        )
                        dgate_part = cute.local_partition(dgate_row, thr_layout, tidx)
                        cute.autovec_copy(dgate_reg, dgate_part)
                    else:
                        for i in range(cute.size(x_reg)):
                            d = dout_reg[i].to(Float32)
                            wi = w_reg[i].to(Float32)
                            x_val = x_reg[i].to(Float32)
                            dw_x = d * wi
                            result = (dw_x - x_val * rstd * rstd * mean_grad) * rstd
                            if const_expr(self.residual):
                                result = result + d
                            dx_reg[i] = result.to(self.x_dtype)

                    # Write dx to smem (overwrite x row)
                    dx_smem_row = cute.make_tensor(
                        x_smem + Int32(local_row * self.D),
                        cute.make_layout(self.D),
                    )
                    dx_smem_part = cute.local_partition(dx_smem_row, thr_layout, tidx)
                    cute.autovec_copy(dx_reg, dx_smem_part)

                    # Write d_residual to global (same as dx, for fused-add backward)
                    if const_expr(self.has_residual):
                        dres_row = cute.make_tensor(
                            d_residual.iterator + global_offset,
                            cute.make_layout(self.D),
                        )
                        dres_part = cute.local_partition(dres_row, thr_layout, tidx)
                        cute.autovec_copy(dx_reg, dres_part)

    # =========================================================================
    # TMA Store (S->G)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_B, tile_S, tile_D,
              dx_tma, dx_tma_gmem):
        """TMA store dx (D × tile_S × 1) from smem[0] to global."""
        sDX = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
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


__all__ = [
    "RMSNormOp", "RMSNormBwdOp", "RMSNORM_EPS",
]
