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

Direct global access with vectorized loads (LDG.128) and stores.
One row per block for maximum GPU occupancy.

All compute warps cooperate on each row via cross-warp reduction:
    - warp_reduction → scratch smem → named_barrier_sync → sum scratch → sync
    - Forward: 1 barrier per row
    - Backward: 2 barriers per row (double-buffered scratch for 2 reductions)
"""

import operator

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

from machete.megakernel.ops import Op
from machete.megakernel.interpreter import named_barrier_sync


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


# =============================================================================
# RMSNorm Ops (direct global access, vectorized loads/stores)
# =============================================================================

# Scratch-only page size: enough for double-buffered cross-warp reduction.
_DIRECT_PAGE_SIZE = 128  # Aligned; actual scratch usage <= SCRATCH_BYTES


def _direct_kernel_config(ops):
    """Return config for direct-mode RMSNorm.

    threads_per_block includes DMA warps (framework always reserves them).
    Cap at 4 compute warps (128 threads) — RMSNorm is memory-bound,
    more warps don't improve throughput and >5 MMA warps can hang.
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


class RMSNormOp(Op):
    """RMSNorm forward — direct global access (no TMA staging).

    Uses vectorized global loads (LDG.128) and stores instead of
    TMA pipelining.  One row per block for maximum GPU occupancy.

    Supports all RMSNorm variants via flags:
        - Standard:    y = rmsnorm(x, weight)
        - Residual:    y = rmsnorm(x, w) + x
        - Gemma:       y = rmsnorm(x, (1+w))
        - Fused add:   residual_out = x + residual_in; y = rmsnorm(residual_out, w)
        - Gated:       y = rmsnorm(x, w) * silu(gate)
        - Per-row weight: weight is (B, S, D) instead of shared (D,)
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

    tma_loads = set()
    tma_stores = set()

    def __init__(self, **config):
        super().__init__(**config)
        self.residual = getattr(self, 'residual', 0)
        self.gemma = getattr(self, 'gemma', 0)
        self.has_residual = getattr(self, 'has_residual', 0)
        self.has_gate = getattr(self, 'has_gate', 0)
        self.per_row_weight = getattr(self, 'per_row_weight', 0)
        self.page_size = getattr(self, 'page_size', _DIRECT_PAGE_SIZE)
        if self.x_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        assert self.D >= 32 and self.D % 32 == 0

        max_warps = min(8, self.threads_per_row // 32)
        max_et = max_warps * 32
        self.effective_threads = 32
        for t in range(32, max_et + 1, 32):
            if self.D % t == 0:
                self.effective_threads = t
        self.effective_warps = self.effective_threads // 32

    kernel_config = staticmethod(_direct_kernel_config)

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
                         per_row_weight=False, page_size=_DIRECT_PAGE_SIZE,
                         **tensors):
        tensors = dict(tensors)
        _expand_weight(tensors)
        has_residual, has_gate = cls._fill_dummies(tensors)
        tile_sizes = dict(tile_sizes or {})
        tile_sizes["S"] = 1  # One row per block
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

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D,
                x, weight, residual_in, gate, y, residual_out):
        """RMSNorm forward: read x from global, write y to global."""
        scratch = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(self.effective_warps),
        )

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        tidx = warp_idx * 32 + lane_idx
        thr_layout = cute.make_layout(self.effective_threads)

        if tidx < self.effective_threads:
            row_idx = tile_S
            global_offset = tile_B * self.S * self.D + row_idx * self.D

            # Load weight from global
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
            else:
                w_row = cute.make_tensor(
                    weight.iterator + global_offset,
                    cute.make_layout(self.D),
                )
                w_part = cute.local_partition(w_row, thr_layout, tidx)
                cute.autovec_copy(w_part, w_reg)
                if const_expr(self.gemma):
                    for i in range(cute.size(w_reg)):
                        w_reg[i] = (w_reg[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

            # Load x from global (vectorized LDG.128)
            x_row = cute.make_tensor(
                x.iterator + global_offset, cute.make_layout(self.D),
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

            # Compute y and write to global (vectorized STG.128)
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

            y_row = cute.make_tensor(
                y.iterator + global_offset, cute.make_layout(self.D),
            )
            y_part = cute.local_partition(y_row, thr_layout, tidx)
            cute.autovec_copy(y_reg, y_part)


class RMSNormBwdOp(Op):
    """RMSNorm backward — direct global access (no TMA staging).

    Computes dx (gradient w.r.t. x), optionally d_residual and dgate.
    Same flag-based variants as forward.
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

    tma_loads = set()
    tma_stores = set()

    def __init__(self, **config):
        super().__init__(**config)
        self.residual = getattr(self, 'residual', 0)
        self.gemma = getattr(self, 'gemma', 0)
        self.has_residual = getattr(self, 'has_residual', 0)
        self.has_gate = getattr(self, 'has_gate', 0)
        self.per_row_weight = getattr(self, 'per_row_weight', 0)
        self.page_size = getattr(self, 'page_size', _DIRECT_PAGE_SIZE)
        if self.x_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        assert self.D >= 32 and self.D % 32 == 0

        max_warps = min(8, self.threads_per_row // 32)
        max_et = max_warps * 32
        self.effective_threads = 32
        for t in range(32, max_et + 1, 32):
            if self.D % t == 0:
                self.effective_threads = t
        self.effective_warps = self.effective_threads // 32

    kernel_config = staticmethod(_direct_kernel_config)

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
                         per_row_weight=False, page_size=_DIRECT_PAGE_SIZE,
                         **tensors):
        tensors = dict(tensors)
        _expand_weight(tensors)
        has_residual, has_gate = cls._fill_dummies(tensors)
        tile_sizes = dict(tile_sizes or {})
        tile_sizes["S"] = 1
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

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D,
                dout, x, weight, gate, dx, d_residual, dgate):
        """RMSNorm backward: read dout/x/weight from global, write dx to global."""
        scratch = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(2 * self.effective_warps),
        )

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        tidx = warp_idx * 32 + lane_idx
        thr_layout = cute.make_layout(self.effective_threads)

        if tidx < self.effective_threads:
            row_idx = tile_S
            global_offset = tile_B * self.S * self.D + row_idx * self.D

            # Load weight from global
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
            else:
                w_row = cute.make_tensor(
                    weight.iterator + global_offset,
                    cute.make_layout(self.D),
                )
                w_part = cute.local_partition(w_row, thr_layout, tidx)
                cute.autovec_copy(w_part, w_reg)
                if const_expr(self.gemma):
                    for i in range(cute.size(w_reg)):
                        w_reg[i] = (w_reg[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

            # Load x from global
            x_row = cute.make_tensor(
                x.iterator + global_offset, cute.make_layout(self.D),
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

            # Write dx to global
            dx_row = cute.make_tensor(
                dx.iterator + global_offset, cute.make_layout(self.D),
            )
            dx_part = cute.local_partition(dx_row, thr_layout, tidx)
            cute.autovec_copy(dx_reg, dx_part)

            # Write d_residual to global (same as dx, for fused-add backward)
            if const_expr(self.has_residual):
                dres_row = cute.make_tensor(
                    d_residual.iterator + global_offset,
                    cute.make_layout(self.D),
                )
                dres_part = cute.local_partition(dres_row, thr_layout, tidx)
                cute.autovec_copy(dx_reg, dres_part)


__all__ = [
    "RMSNormOp", "RMSNormBwdOp", "RMSNORM_EPS",
]
