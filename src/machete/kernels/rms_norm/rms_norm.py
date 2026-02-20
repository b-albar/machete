# Copyright (c) 2025, Machete Authors
"""
RMSNorm Op for the Megakernel.

Applies Root Mean Square Layer Normalization:
    y = x / sqrt(mean(x²) + eps) * weight

Pipelined load/compute/store with shared memory staging:
    load:    TMA async G->S (x tile)
    compute: read x from smem, weight from global->regs, write output to smem
    store:   TMA S->G (y forward / dx backward)

Weight is loaded to registers in compute (small, shared across tiles).
All compute warps cooperate on each row via cross-warp reduction:
    - Forward: load x, compute y = x * rstd * w, store y
    - Backward: load x, compute dx from x + dout (global) + w (global), store dx

Configuration:
    - threads_per_row: all compute threads (threads_per_block - 32)
    - tile_size_M: from tile_sizes at schedule time, constrained by page_size

Forward:
    rstd = 1 / sqrt(mean(x²) + eps)
    y = x * rstd * weight

Usage:
    from machete.kernels.rms_norm import RMSNormOp
    from machete.megakernel import Megakernel

    x_2d = x.view(-1, D).contiguous()
    ops = RMSNormOp.schedule(x=x_2d, weight=w, y=y, tile_sizes={"M": 4})
    kernel = Megakernel(ops)
    kernel.run()
"""

import operator

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import mbarrier_arrive_expect_tx, named_barrier_sync


# =============================================================================
# Constants
# =============================================================================

RMSNORM_EPS = 1e-6
SCRATCH_BYTES = 128  # Cross-warp reduction scratch (up to 32 warps × 4B Float32)


# =============================================================================
# RMSNorm Op
# =============================================================================


class RMSNormOp(Op):
    """RMSNorm operation for the megakernel framework.

    Applies Root Mean Square Layer Normalization:
        y = x / sqrt(mean(x²) + eps) * weight       (residual=False)
        y = x / sqrt(mean(x²) + eps) * weight + x   (residual=True)

    Both forward and backward use pipelined load/compute/store:
    - load: TMA G->S for x (DMA warp, elect_one)
    - compute: x from smem, weight+dout from global, output to smem
    - store: TMA S->G for y (forward) or dx (backward)

    Shared memory layout per page:
        [tile_size_M * D elements]  -- x on load, y/dx after compute

    Tensor declarations:
        x:      (M, D)  -- input tensor (bf16/fp16/fp32)
        weight: (D,)    -- per-element scale (bf16/fp16/fp32)
        y:      (M, D)  -- output tensor (bf16/fp16/fp32)

    Tiling:
        tile_M indexes row groups (ceil(M / tile_size_M) tiles).
        Each tile processes tile_size_M rows (matching warp count).

    Requirements:
        D >= 32 (warp-parallel vectorized access)
        tile_size_M * D * elem_bytes <= page_size
    """

    # dtype=None means infer from tensor at schedule time (supports bf16/fp16/fp32)
    reads = {
        "x": (None, ("M", "D")),
        "weight": (None, ("D",)),
    }
    writes = {"y": (None, ("M", "D"))}
    tile = ("M", "D")

    tma_loads = {"x"}
    tma_stores = {"y", "dx"}

    backward_reads = {
        "dout": (None, ("M", "D")),
        "x": (None, ("M", "D")),
        "weight": (None, ("D",)),
    }
    backward_writes = {"dx": (None, ("M", "D"))}

    def __init__(self, **config):
        super().__init__(**config)
        self.residual = getattr(self, 'residual', 0)
        self.page_size = getattr(self, 'page_size', DEFAULT_PAGE_SIZE)

        if self.x_dtype == cutlass.Float32:
            self.elem_bytes = 4
        elif self.x_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        self.x_tile_bytes = self.tile_size_M * self.D * self.elem_bytes
        self.scratch_offset = self.x_tile_bytes  # Cross-warp reduction scratch

        assert self.D >= 32, f"RMSNormOp requires D >= 32, got D={self.D}"
        assert self.x_tile_bytes + SCRATCH_BYTES <= self.page_size, (
            f"RMSNormOp: tile smem ({self.x_tile_bytes}B) + scratch ({SCRATCH_BYTES}B) "
            f"exceeds page_size ({self.page_size}B). Reduce tile_size_M={self.tile_size_M}."
        )

        self.num_warps = self.threads_per_row // 32

        # Effective threads must DIVIDE D (else local_partition's strided access
        # reads beyond row boundary). Find largest multiple of 32 that divides D
        # and fits within threads_per_row. Threads >= effective_threads skip compute.
        max_et = min(self.D, self.threads_per_row)
        self.effective_threads = 32  # At least 1 warp
        for t in range(32, max_et + 1, 32):
            if self.D % t == 0:
                self.effective_threads = t
        self.effective_warps = self.effective_threads // 32

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def _auto_tile_M(cls, page_size, **tensors):
        """Compute largest tile_size_M that fits in page_size minus scratch."""
        x = tensors.get('x')
        if x is None:
            return None
        D = x.shape[1]
        elem_bytes = x.element_size()
        usable = page_size - SCRATCH_BYTES
        return max(1, usable // (D * elem_bytes))

    @classmethod
    def schedule_forward(cls, tile_sizes=None, residual=False, page_size=DEFAULT_PAGE_SIZE, **tensors):
        """Schedule RMSNorm forward, optionally with residual connection."""
        tile_sizes = dict(tile_sizes or {})
        if "M" not in tile_sizes:
            auto_m = cls._auto_tile_M(page_size, **tensors)
            if auto_m is not None:
                tile_sizes["M"] = auto_m
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        if residual:
            ops[0].static_dims['residual'] = 1
        ops[0].static_dims['page_size'] = page_size
        return ops

    @classmethod
    def schedule_backward(cls, tile_sizes=None, residual=False, page_size=DEFAULT_PAGE_SIZE, **tensors):
        """Schedule RMSNorm backward, optionally with residual connection."""
        tile_sizes = dict(tile_sizes or {})
        if "M" not in tile_sizes:
            auto_m = cls._auto_tile_M(page_size, **tensors)
            if auto_m is not None:
                tile_sizes["M"] = auto_m
        ops = [cls._schedule_single(backward=True, tile_sizes=tile_sizes, **tensors)]
        if residual:
            ops[0].static_dims['residual'] = 1
        ops[0].static_dims['page_size'] = page_size
        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for scheduled RMSNormOps.

        Chooses threads_per_block so compute threads (= threads_per_block - 32)
        divide D evenly for vectorized loads. Prefers 128 compute threads.
        """
        from machete.megakernel import MegakernelConfig
        page_size = ops[0].static_dims.get('page_size', DEFAULT_PAGE_SIZE)
        D = ops[0].static_dims.get('D', 4096)
        # Pick compute_threads as power-of-2 that divides D for clean partitioning.
        # Prefer more threads (256 > 128 > 64) for better multi-warp parallelism.
        compute_threads = 128
        for ct in [256, 128, 64]:
            if D % ct == 0:
                compute_threads = ct
                break
        return MegakernelConfig(
            threads_per_block=compute_threads + 32,
            page_size=page_size,
        )

    # =========================================================================
    # Forward Load (G->S)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_M, tile_D, x_tma, x_tma_gmem, work_mbar):
        """TMA load of x tile from global to shared memory.

        TMA transposes gmem: x(M,D) -> smem (D,M). Handles boundary
        conditions (partial last M tile) automatically.
        """
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
    # Forward Compute
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_M, tile_D, x, weight, y):
        """RMSNorm forward: read x from smem, write y to same smem region.

        All effective warps cooperate on each row via cross-warp reduction.
        Weight is loaded from global memory to registers (small, reused
        across all rows in the tile). OOB rows compute harmlessly on stale
        smem data — TMA store handles boundary conditions.

        When D < threads_per_row, threads with tidx >= effective_threads
        skip all computation (avoids local_partition OOB reads).
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

        # Effective threads capped to D (avoid OOB partition reads)
        thr_layout = cute.make_layout(self.effective_threads)

        if tidx < self.effective_threads:
            # Load weight into registers (partitioned across effective threads)
            w_row = cute.make_tensor(
                weight.iterator,
                cute.make_layout(self.D),
            )
            w_part = cute.local_partition(w_row, thr_layout, tidx)
            w_reg = cute.make_fragment_like(w_part)
            cute.autovec_copy(w_part, w_reg)

            # All warps process each row cooperatively (no bounds check needed —
            # TMA zero-fills OOB smem regions, TMA store skips OOB rows)
            for local_row in range(self.tile_size_M):
                # Read x from smem
                x_row = cute.make_tensor(
                    x_smem + local_row * self.D,
                    cute.make_layout(self.D),
                )
                x_part = cute.local_partition(x_row, thr_layout, tidx)
                x_reg = cute.make_fragment_like(x_part)
                cute.autovec_copy(x_part, x_reg)

                # Per-thread partial sum of squares (in fp32)
                partial_sq = Float32(0.0)
                for i in range(cute.size(x_reg)):
                    val = x_reg[i].to(Float32)
                    partial_sq = partial_sq + val * val

                # Cross-warp reduction: warp reduce → scratch → sync → sum
                warp_sum = cute.arch.warp_reduction(partial_sq, operator.add)
                if lane_idx == 0:
                    scratch[warp_idx] = warp_sum
                named_barrier_sync(Int32(2), Int32(self.effective_threads))

                sum_sq = Float32(0.0)
                for w in range(self.effective_warps):
                    sum_sq = sum_sq + scratch[w]
                named_barrier_sync(Int32(2), Int32(self.effective_threads))

                # rstd = 1 / sqrt(mean(x²) + eps)
                rstd = cute.math.rsqrt(sum_sq / self.D + RMSNORM_EPS, fastmath=True)

                # y = x * rstd * weight [+ x] (compute in fp32, store in input dtype)
                y_reg = cute.make_fragment_like(x_reg)
                for i in range(cute.size(x_reg)):
                    val = x_reg[i].to(Float32) * rstd * w_reg[i].to(Float32)
                    if self.residual:
                        val = val + x_reg[i].to(Float32)
                    y_reg[i] = val.to(self.x_dtype)

                # Write y back to same smem location
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
        """TMA store of y from shared to global memory.

        Compute writes y as (tile_M, D) row-major stride (D, 1).
        TMA sees smem as (D, tile_M) col-major — same physical layout.
        TMA handles boundary conditions for partial last M tile.
        """
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

    # =========================================================================
    # Backward Load (G->S) — reuse forward TMA load for x
    # =========================================================================

    backward_load = load

    # =========================================================================
    # Backward Compute
    # =========================================================================

    @cute.jit
    def backward_compute(self, page_ptr, tile_M, tile_D, dout, x, weight, dx):
        """RMSNorm backward: read x from smem, dout+weight from global.

        dx = (dout * weight - x * rstd² * mean(dout * weight * x)) * rstd

        All effective warps cooperate on each row via cross-warp reduction.
        x is loaded to smem by backward_load (TMA). dout and weight are
        read from global memory. dx is written to smem for TMA store.

        Bounds check kept because dout is read from global memory.
        row_idx is uniform (same for all threads) so all threads take
        the same branch — named_barrier_sync inside the if is safe.
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

        # Effective threads capped to D (avoid OOB partition reads)
        thr_layout = cute.make_layout(self.effective_threads)

        if tidx < self.effective_threads:
            # Load weight into registers (partitioned across effective threads)
            w_row = cute.make_tensor(
                weight.iterator,
                cute.make_layout(self.D),
            )
            w_part = cute.local_partition(w_row, thr_layout, tidx)
            w_reg = cute.make_fragment_like(w_part)
            cute.autovec_copy(w_part, w_reg)

            row_start = tile_M * self.tile_size_M

            for local_row in range(self.tile_size_M):
                row_idx = row_start + local_row

                if row_idx < self.M:
                    # Read x from smem (loaded by backward_load)
                    x_row = cute.make_tensor(
                        x_smem + local_row * self.D,
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
                    if lane_idx == 0:
                        scratch[warp_idx] = warp_sum
                    named_barrier_sync(Int32(2), Int32(self.effective_threads))

                    sum_sq = Float32(0.0)
                    for w in range(self.effective_warps):
                        sum_sq = sum_sq + scratch[w]
                    named_barrier_sync(Int32(2), Int32(self.effective_threads))

                    rstd = cute.math.rsqrt(sum_sq / self.D + RMSNORM_EPS, fastmath=True)

                    # Load dout from global (partitioned across effective threads)
                    dout_row = cute.make_tensor(
                        dout.iterator + row_idx * self.D,
                        cute.make_layout(self.D),
                    )
                    dout_part = cute.local_partition(dout_row, thr_layout, tidx)
                    dout_reg = cute.make_fragment_like(dout_part)
                    cute.autovec_copy(dout_part, dout_reg)

                    # Pass 2: sum_grad via cross-warp reduction
                    partial_grad = Float32(0.0)
                    for i in range(cute.size(x_reg)):
                        d = dout_reg[i].to(Float32)
                        w = w_reg[i].to(Float32)
                        x_val = x_reg[i].to(Float32)
                        partial_grad = partial_grad + d * w * x_val

                    warp_grad = cute.arch.warp_reduction(partial_grad, operator.add)
                    if lane_idx == 0:
                        scratch[warp_idx] = warp_grad
                    named_barrier_sync(Int32(2), Int32(self.effective_threads))

                    sum_grad = Float32(0.0)
                    for w in range(self.effective_warps):
                        sum_grad = sum_grad + scratch[w]
                    named_barrier_sync(Int32(2), Int32(self.effective_threads))

                    mean_grad = sum_grad / self.D

                    # Pass 3: dx = (dout * w - x * rstd² * mean_grad) * rstd [+ dout]
                    dx_reg = cute.make_fragment_like(x_reg)
                    for i in range(cute.size(x_reg)):
                        d = dout_reg[i].to(Float32)
                        w = w_reg[i].to(Float32)
                        x_val = x_reg[i].to(Float32)
                        dw_x = d * w
                        result = (dw_x - x_val * rstd * rstd * mean_grad) * rstd
                        if self.residual:
                            result = result + d
                        dx_reg[i] = result.to(self.x_dtype)

                    # Write dx to smem for TMA store
                    dx_row_out = cute.make_tensor(
                        x_smem + local_row * self.D,
                        cute.make_layout(self.D),
                    )
                    dx_part_out = cute.local_partition(dx_row_out, thr_layout, tidx)
                    cute.autovec_copy(dx_reg, dx_part_out)

    # =========================================================================
    # Backward Store (S->G)
    # =========================================================================

    @cute.jit
    def backward_store(self, page_ptr, tile_M, tile_D, dx_tma, dx_tma_gmem):
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


__all__ = ["RMSNormOp"]
