# Copyright (c) 2025, Machete Authors
"""
RMSNorm Ops for the Megakernel.

Supports all RMSNorm variants via flags:
    - Standard:    y = rmsnorm(x, weight)
    - Residual:    y = rmsnorm(x, w) + x
    - Gemma:       y = rmsnorm(x, (1+w))
    - Fused add:   residual_out = x + residual_in; y = rmsnorm(residual_out, w)
    - Gated:       y = rmsnorm(x, w) * silu(gate)
    - Per-row weight: weight is (M, D) instead of shared (D,)

Pipelined load/compute/store with shared memory staging:
    load:    TMA async G->S (x tile)
    compute: read x from smem, other tensors from global, write result to smem
    store:   TMA S->G (y forward / dx backward)

All compute warps cooperate on each row via cross-warp reduction:
    - warp_reduction → scratch smem → named_barrier_sync → sum scratch → sync
    - Forward: 2 barriers per row (1 reduction for sum_sq)
    - Backward: 4 barriers per row (2 reductions for sum_sq + sum_grad)
"""

import operator

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import mbarrier_arrive_expect_tx, named_barrier_sync


# =============================================================================
# Constants
# =============================================================================

RMSNORM_EPS = 1e-6
SCRATCH_BYTES = 128  # Cross-warp reduction scratch (up to 32 warps × 4B Float32)


# =============================================================================
# Shared helpers
# =============================================================================

def _rmsnorm_init(self, **config):
    """Common __init__ logic for RMSNormOp and RMSNormBwdOp."""
    self.residual = getattr(self, 'residual', 0)
    self.gemma = getattr(self, 'gemma', 0)
    self.has_residual = getattr(self, 'has_residual', 0)
    self.has_gate = getattr(self, 'has_gate', 0)
    self.per_row_weight = getattr(self, 'per_row_weight', 0)
    self.page_size = getattr(self, 'page_size', DEFAULT_PAGE_SIZE)

    if self.x_dtype == cutlass.Float32:
        self.elem_bytes = 4
    elif self.x_dtype in (cutlass.Float16, cutlass.BFloat16):
        self.elem_bytes = 2
    else:
        self.elem_bytes = 4

    self.x_tile_bytes = self.tile_size_M * self.D * self.elem_bytes
    self.scratch_offset = self.x_tile_bytes

    assert self.D >= 32 and self.D % 32 == 0, (
        f"RMSNorm requires D >= 32 and D % 32 == 0, got D={self.D}"
    )
    assert self.x_tile_bytes + SCRATCH_BYTES <= self.page_size, (
        f"RMSNorm: tile smem ({self.x_tile_bytes}B) + scratch ({SCRATCH_BYTES}B) "
        f"exceeds page_size ({self.page_size}B). Reduce tile_size_M={self.tile_size_M}."
    )

    max_et = min(self.D, self.threads_per_row)
    self.effective_threads = 32
    for t in range(32, max_et + 1, 32):
        if self.D % t == 0:
            self.effective_threads = t
    self.effective_warps = self.effective_threads // 32


def _auto_tile_M(page_size, **tensors):
    """Compute largest tile_size_M that fits in page_size minus scratch."""
    x = tensors.get('x')
    if x is None:
        return None
    D = x.shape[1]
    elem_bytes = x.element_size()
    usable = page_size - SCRATCH_BYTES
    return max(1, usable // (D * elem_bytes))


def _expand_weight(tensors):
    """Auto-expand 1D weight (D,) to 2D (M, D) for uniform handling."""
    w = tensors.get('weight')
    if w is not None and w.ndim == 1:
        M = tensors['x'].shape[0]
        tensors['weight'] = w.unsqueeze(0).expand(M, -1).contiguous()


def _kernel_config(ops):
    """Return recommended MegakernelConfig for RMSNorm ops."""
    from machete.megakernel import MegakernelConfig
    page_size = ops[0].static_dims.get('page_size', DEFAULT_PAGE_SIZE)
    D = ops[0].static_dims.get('D', 4096)
    compute_threads = 128
    for ct in [256, 128, 64]:
        if D % ct == 0:
            compute_threads = ct
            break
    from machete.megakernel.megakernel import NUM_DMA_WARPS
    return MegakernelConfig(
        threads_per_block=compute_threads + NUM_DMA_WARPS * 32,
        page_size=page_size,
    )


# =============================================================================
# RMSNormOp (Forward)
# =============================================================================


class RMSNormOp(Op):
    """RMSNorm forward operation.

    All variants handled via flags: residual, gemma, has_residual, has_gate,
    per_row_weight. See module docstring for details.
    """

    reads = {
        "x": (None, ("M", "D")),
        "weight": (None, ("M", "D")),
        "residual_in": (None, ("M", "D")),
        "gate": (None, ("M", "D")),
    }
    writes = {
        "y": (None, ("M", "D")),
        "residual_out": (None, ("M", "D")),
    }
    tile = ("M", "D")

    tma_loads = {"x"}
    tma_stores = {"y"}

    def __init__(self, **config):
        super().__init__(**config)
        _rmsnorm_init(self, **config)

    # =========================================================================
    # Scheduling
    # =========================================================================

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
                 per_row_weight=False, page_size=DEFAULT_PAGE_SIZE, **tensors):
        """Schedule RMSNorm forward with optional fused-add and/or gating."""
        tensors = dict(tensors)
        _expand_weight(tensors)
        has_residual, has_gate = cls._fill_dummies(tensors)

        tile_sizes = dict(tile_sizes or {})
        if "M" not in tile_sizes:
            auto_m = _auto_tile_M(page_size, **tensors)
            if auto_m is not None:
                tile_sizes["M"] = auto_m

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

    kernel_config = staticmethod(_kernel_config)

    # =========================================================================
    # Load (G->S)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_M, tile_D, x_tma, x_tma_gmem, work_mbar):
        """TMA load of x tile from global to shared memory."""
        sX = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_M)),
        )
        gX = cute.local_tile(
            x_tma_gmem, (self.D, self.tile_size_M), (None, None),
        )
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            x_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(gX, 0, 2),
        )

        nbytes = Int32(self.x_tile_bytes)
        mbar_ptr = cute.make_ptr(
            cutlass.Int64, work_mbar, cute.AddressSpace.smem
        )
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(x_tma, tXgX[(None, tile_D, tile_M)], tXsX,
                  tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Forward Compute (cooperative cross-warp)
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_M, tile_D,
                x, weight, residual_in, gate, y, residual_out):
        """RMSNorm forward: read x from smem, write y to same smem region."""
        x_smem = cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem)
        scratch = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, page_ptr + self.scratch_offset,
                          cute.AddressSpace.smem),
            cute.make_layout(self.effective_warps),
        )

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        tidx = warp_idx * 32 + lane_idx

        thr_layout = cute.make_layout(self.effective_threads)

        if tidx < self.effective_threads:
            row_start = tile_M * self.tile_size_M

            # Pre-allocate weight registers; load shared weight ONCE
            w_row_0 = cute.make_tensor(
                weight.iterator,
                cute.make_layout(self.D),
            )
            w_part_0 = cute.local_partition(w_row_0, thr_layout, tidx)
            w_reg = cute.make_fragment_like(w_part_0)

            if const_expr(not self.per_row_weight):
                cute.autovec_copy(w_part_0, w_reg)
                if const_expr(self.gemma):
                    for i in range(cute.size(w_reg)):
                        w_reg[i] = (w_reg[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

            for local_row in range(self.tile_size_M):
                row_idx = row_start + local_row

                if row_idx < self.M:
                    # Per-row weight: reload from the correct row
                    if const_expr(self.per_row_weight):
                        w_row = cute.make_tensor(
                            weight.iterator + row_idx * self.D,
                            cute.make_layout(self.D),
                        )
                        w_part = cute.local_partition(w_row, thr_layout, tidx)
                        cute.autovec_copy(w_part, w_reg)
                        if const_expr(self.gemma):
                            for i in range(cute.size(w_reg)):
                                w_reg[i] = (w_reg[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

                    # --- Read x from smem ---
                    x_row = cute.make_tensor(
                        x_smem + local_row * self.D,
                        cute.make_layout(self.D),
                    )
                    x_part = cute.local_partition(x_row, thr_layout, tidx)
                    x_reg = cute.make_fragment_like(x_part)
                    cute.autovec_copy(x_part, x_reg)

                    # --- Fused add: x_reg += residual_in ---
                    if const_expr(self.has_residual):
                        res_row = cute.make_tensor(
                            residual_in.iterator + row_idx * self.D,
                            cute.make_layout(self.D),
                        )
                        res_part = cute.local_partition(res_row, thr_layout, tidx)
                        res_reg = cute.make_fragment_like(res_part)
                        cute.autovec_copy(res_part, res_reg)

                        for i in range(cute.size(x_reg)):
                            x_reg[i] = (x_reg[i].to(Float32) + res_reg[i].to(Float32)).to(self.x_dtype)

                        # Write residual_out to global
                        res_out_row = cute.make_tensor(
                            residual_out.iterator + row_idx * self.D,
                            cute.make_layout(self.D),
                        )
                        res_out_part = cute.local_partition(res_out_row, thr_layout, tidx)
                        cute.autovec_copy(x_reg, res_out_part)

                    # --- Cross-warp reduction for sum_sq ---
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
                    named_barrier_sync(Int32(2), Int32(self.effective_threads))

                    rstd = cute.math.rsqrt(sum_sq / self.D + RMSNORM_EPS, fastmath=True)

                    # --- Compute output y ---
                    y_reg = cute.make_fragment_like(x_reg)

                    if const_expr(self.has_gate):
                        gate_row = cute.make_tensor(
                            gate.iterator + row_idx * self.D,
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

                    # Write y to smem for TMA store
                    y_row = cute.make_tensor(
                        x_smem + local_row * self.D,
                        cute.make_layout(self.D),
                    )
                    y_part = cute.local_partition(y_row, thr_layout, tidx)
                    cute.autovec_copy(y_reg, y_part)

    # =========================================================================
    # Forward Store (S->G)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_M, tile_D, y_tma, y_tma_gmem):
        """TMA store of y from shared to global memory."""
        sY = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_M)),
        )
        gY = cute.local_tile(
            y_tma_gmem, (self.D, self.tile_size_M), (None, None),
        )
        tYsY, tYgY = cute.nvgpu.cpasync.tma_partition(
            y_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sY, 0, 2),
            cute.group_modes(gY, 0, 2),
        )
        with cute.arch.elect_one():
            cute.copy(y_tma, tYsY, tYgY[(None, tile_D, tile_M)])


# =============================================================================
# RMSNormBwdOp (Backward)
# =============================================================================


class RMSNormBwdOp(Op):
    """RMSNorm backward operation.

    Computes dx (gradient w.r.t. x), optionally d_residual and dgate.
    Reuses RMSNormOp's TMA load for x. Same flag-based variants as forward.
    """

    reads = {
        "dout": (None, ("M", "D")),
        "x": (None, ("M", "D")),
        "weight": (None, ("M", "D")),
        "gate": (None, ("M", "D")),
    }
    writes = {
        "dx": (None, ("M", "D")),
        "d_residual": (None, ("M", "D")),
        "dgate": (None, ("M", "D")),
    }
    tile = ("M", "D")

    tma_loads = {"x"}
    tma_stores = {"dx"}

    def __init__(self, **config):
        super().__init__(**config)
        _rmsnorm_init(self, **config)

    # Reuse RMSNormOp's TMA load for x
    load = RMSNormOp.load

    kernel_config = staticmethod(_kernel_config)

    # =========================================================================
    # Scheduling
    # =========================================================================

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
                 per_row_weight=False, page_size=DEFAULT_PAGE_SIZE, **tensors):
        """Schedule RMSNorm backward with optional fused-add and/or gating."""
        tensors = dict(tensors)
        _expand_weight(tensors)
        has_residual, has_gate = cls._fill_dummies(tensors)

        tile_sizes = dict(tile_sizes or {})
        if "M" not in tile_sizes:
            auto_m = _auto_tile_M(page_size, **tensors)
            if auto_m is not None:
                tile_sizes["M"] = auto_m

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
    # Backward Compute (cooperative cross-warp)
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_M, tile_D,
                dout, x, weight, gate, dx, d_residual, dgate):
        """RMSNorm backward: read x from smem, dout+weight+gate from global.

        dx written to smem for TMA store. d_residual, dgate to global.
        """
        x_smem = cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem)
        scratch = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, page_ptr + self.scratch_offset,
                          cute.AddressSpace.smem),
            cute.make_layout(self.effective_warps),
        )

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        tidx = warp_idx * 32 + lane_idx

        thr_layout = cute.make_layout(self.effective_threads)

        if tidx < self.effective_threads:
            row_start = tile_M * self.tile_size_M

            # Pre-allocate weight registers; load shared weight ONCE
            w_row_0 = cute.make_tensor(
                weight.iterator,
                cute.make_layout(self.D),
            )
            w_part_0 = cute.local_partition(w_row_0, thr_layout, tidx)
            w_reg = cute.make_fragment_like(w_part_0)

            if const_expr(not self.per_row_weight):
                cute.autovec_copy(w_part_0, w_reg)
                if const_expr(self.gemma):
                    for i in range(cute.size(w_reg)):
                        w_reg[i] = (w_reg[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

            for local_row in range(self.tile_size_M):
                row_idx = row_start + local_row

                if row_idx < self.M:
                    # Per-row weight: reload from the correct row
                    if const_expr(self.per_row_weight):
                        w_row = cute.make_tensor(
                            weight.iterator + row_idx * self.D,
                            cute.make_layout(self.D),
                        )
                        w_part = cute.local_partition(w_row, thr_layout, tidx)
                        cute.autovec_copy(w_part, w_reg)
                        if const_expr(self.gemma):
                            for i in range(cute.size(w_reg)):
                                w_reg[i] = (w_reg[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

                    # --- Read x from smem (TMA loaded) ---
                    x_row = cute.make_tensor(
                        x_smem + local_row * self.D,
                        cute.make_layout(self.D),
                    )
                    x_part = cute.local_partition(x_row, thr_layout, tidx)
                    x_reg = cute.make_fragment_like(x_part)
                    cute.autovec_copy(x_part, x_reg)

                    # --- Pass 1: sum_sq via cross-warp reduction ---
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
                    named_barrier_sync(Int32(2), Int32(self.effective_threads))

                    rstd = cute.math.rsqrt(sum_sq / self.D + RMSNORM_EPS, fastmath=True)

                    # --- Load dout from global ---
                    dout_row = cute.make_tensor(
                        dout.iterator + row_idx * self.D,
                        cute.make_layout(self.D),
                    )
                    dout_part = cute.local_partition(dout_row, thr_layout, tidx)
                    dout_reg = cute.make_fragment_like(dout_part)
                    cute.autovec_copy(dout_part, dout_reg)

                    # --- Pre-allocate gate fragment ---
                    gate_reg = cute.make_fragment_like(x_part)

                    if const_expr(self.has_gate):
                        gate_row = cute.make_tensor(
                            gate.iterator + row_idx * self.D,
                            cute.make_layout(self.D),
                        )
                        gate_part = cute.local_partition(gate_row, thr_layout, tidx)
                        cute.autovec_copy(gate_part, gate_reg)

                    # --- Pass 2: sum_grad via cross-warp reduction ---
                    partial_grad = Float32(0.0)

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
                        scratch[warp_idx] = warp_grad
                    named_barrier_sync(Int32(2), Int32(self.effective_threads))

                    sum_grad = Float32(0.0)
                    for wi in range(self.effective_warps):
                        sum_grad = sum_grad + scratch[wi]
                    named_barrier_sync(Int32(2), Int32(self.effective_threads))

                    mean_grad = sum_grad / self.D

                    # --- Pass 3: dx [and dgate] ---
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
                            dgate.iterator + row_idx * self.D,
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

                    # Write dx to smem for TMA store
                    dx_row_out = cute.make_tensor(
                        x_smem + local_row * self.D,
                        cute.make_layout(self.D),
                    )
                    dx_part_out = cute.local_partition(dx_row_out, thr_layout, tidx)
                    cute.autovec_copy(dx_reg, dx_part_out)

                    # Write d_residual to global (same as dx, for fused-add backward)
                    if const_expr(self.has_residual):
                        dres_row = cute.make_tensor(
                            d_residual.iterator + row_idx * self.D,
                            cute.make_layout(self.D),
                        )
                        dres_part = cute.local_partition(dres_row, thr_layout, tidx)
                        cute.autovec_copy(dx_reg, dres_part)

    # =========================================================================
    # Backward Store (S->G)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_M, tile_D, dx_tma, dx_tma_gmem):
        """TMA store of dx from shared to global memory."""
        sDX = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_M)),
        )
        gDX = cute.local_tile(
            dx_tma_gmem, (self.D, self.tile_size_M), (None, None),
        )
        tDXsDX, tDXgDX = cute.nvgpu.cpasync.tma_partition(
            dx_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sDX, 0, 2),
            cute.group_modes(gDX, 0, 2),
        )
        with cute.arch.elect_one():
            cute.copy(dx_tma, tDXsDX, tDXgDX[(None, tile_D, tile_M)])


# Backward-compatible aliases
FusedAddRMSNormOp = RMSNormOp
RMSNormGatedOp = RMSNormOp

__all__ = ["RMSNormOp", "RMSNormBwdOp", "FusedAddRMSNormOp", "RMSNormGatedOp", "RMSNORM_EPS"]
