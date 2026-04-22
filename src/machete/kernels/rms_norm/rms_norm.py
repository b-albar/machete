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

from machete.megakernel.ops import (
    Op,
    DEFAULT_PAGE_SIZE,
    config_dim_i32,
    config_flat_tensor,
)
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


def _align_up(x, align):
    return ((x + align - 1) // align) * align


def _pick_rowwise_tma_tile_n(width, tile_s, elem_bytes):
    upper = min(width, 256)
    for tile_n in range(upper, 15, -1):
        if width % tile_n == 0 and (tile_n * tile_s * elem_bytes) % 128 == 0:
            return tile_n
    for tile_n in range(upper, 15, -1):
        if width % tile_n == 0:
            return tile_n
    return 16


def _rowwise_chunked_bytes(width, tile_s, elem_bytes):
    tile_n = _pick_rowwise_tma_tile_n(width, tile_s, elem_bytes)
    num_tiles = width // tile_n
    chunk_bytes = tile_n * tile_s * elem_bytes
    return num_tiles * _align_up(chunk_bytes, 128)


def _auto_chunked_tile_S(width, elem_bytes, page_size, scratch_bytes=0):
    tile_s = max(1, (page_size - scratch_bytes) // (width * elem_bytes))
    while tile_s > 1 and _rowwise_chunked_bytes(width, tile_s, elem_bytes) + scratch_bytes > page_size:
        tile_s -= 1
    return tile_s


def _tma_kernel_config(cls, ops):
    """Return config for TMA-mode RMSNorm.

    threads_per_block includes DMA warps (framework always reserves them).
    Picks the largest compute thread count (multiple of 32) that divides D,
    up to 256 (8 warps).
    """
    from machete.megakernel import MegakernelConfig
    from machete.megakernel.megakernel import NUM_DMA_WARPS
    D = ops[0].static_dims.get('D', 4096)
    page_size = ops[0].static_dims.get('page_size', DEFAULT_PAGE_SIZE)
    compute_threads = 64
    # 256 compute threads overprovision RMSNorm on this backend more often
    # than they help: the extra warps increase cross-warp/barrier overhead
    # and persistent-shell pressure, especially on small and medium sequence
    # lengths. Capping at 128 keeps enough parallelism for D up to 4096 while
    # improving the common decode/inference shapes.
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
    dynamic_dims = ("B",)

    tma_loads = {"x"}
    tma_stores = {"y"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name in {"x", "y"}:
            return (1, tile_sizes["S"], static_dims["tma_tile_D"])
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name in {"x", "y"}:
            tile_d, tile_s, tile_b = tma_tile_shape
            return f"cute.make_layout(({tile_d}, {tile_s}, {tile_b}))"
        return None

    def __init__(self, **config):
        super().__init__(**config)
        self.residual = getattr(self, 'residual', 0)
        self.gemma = getattr(self, 'gemma', 0)
        self.has_residual = getattr(self, 'has_residual', 0)
        self.has_gate = getattr(self, 'has_gate', 0)
        self.per_row_weight = getattr(self, 'per_row_weight', 0)
        if self.x_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        assert self.D >= 32 and self.D % 32 == 0

        self.tma_tile_D = getattr(
            self,
            'tma_tile_D',
            _pick_rowwise_tma_tile_n(self.D, self.tile_size_S, self.elem_bytes),
        )
        self.num_tma_tiles = self.D // self.tma_tile_D
        self.chunk_tile_elems = self.tma_tile_D * self.tile_size_S
        self.chunk_tile_bytes = self.chunk_tile_elems * self.elem_bytes
        self.chunk_stride_bytes = _align_up(self.chunk_tile_bytes, 128)
        self.chunk_stride_elems = self.chunk_stride_bytes // self.elem_bytes
        # Smem layout: chunked x tile at [0, x_tile_bytes), scratch after it.
        self.x_tile_bytes = self.num_tma_tiles * self.chunk_stride_bytes
        self.scratch_offset = self.x_tile_bytes

        max_warps = min(8, self.threads_per_row // 32)
        max_et = max_warps * 32
        self.effective_threads = 32
        for t in range(32, max_et + 1, 32):
            if self.D % t == 0:
                self.effective_threads = t
        self.effective_warps = self.effective_threads // 32

    kernel_config = classmethod(_tma_kernel_config)

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
    def schedule(cls, tile_sizes=None, residual=False, gemma=False,
                         per_row_weight=False, page_size=DEFAULT_PAGE_SIZE,
                         **tensors):
        tensors = dict(tensors)
        _expand_weight(tensors)
        has_residual, has_gate = cls._fill_dummies(tensors)
        tile_sizes = dict(tile_sizes or {})
        D = tensors['x'].shape[-1]
        elem_bytes = tensors['x'].element_size()
        max_tile_S = _auto_chunked_tile_S(D, elem_bytes, page_size, SCRATCH_BYTES)
        if "S" not in tile_sizes:
            tile_sizes["S"] = max_tile_S
        else:
            # Clamp caller-specified tile_S so scratch fits in page
            tile_sizes["S"] = min(tile_sizes["S"], max_tile_S)
            while (
                tile_sizes["S"] > 1
                and _rowwise_chunked_bytes(D, tile_sizes["S"], elem_bytes) + SCRATCH_BYTES > page_size
            ):
                tile_sizes["S"] -= 1
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
        ops[0].static_dims['tma_tile_D'] = _pick_rowwise_tma_tile_n(
            D,
            tile_sizes["S"],
            elem_bytes,
        )
        return ops

    # =========================================================================
    # TMA Load (G->S)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_D,
             x_tma, x_tma_gmem, work_mbar):
        """TMA load x tile (D × tile_S × 1) from global to smem."""
        nbytes = Int32(self.x_tile_bytes)
        mbar_ptr = cute.make_ptr(
            cutlass.Int64, work_mbar, cute.AddressSpace.smem
        )
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        for wi in range(self.num_tma_tiles):
            gXi = cute.local_tile(
                x_tma_gmem,
                (self.tma_tile_D, self.tile_size_S, 1),
                (Int32(wi), tile_S, tile_B),
            )
            sXi = cute.make_tensor(
                cute.make_ptr(
                    self.x_dtype,
                    page_ptr + Int32(wi * self.chunk_stride_bytes),
                    cute.AddressSpace.smem,
                ),
                cute.make_layout((self.tma_tile_D, self.tile_size_S, 1)),
            )
            tXsXi, tXgXi = cute.nvgpu.cpasync.tma_partition(
                x_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sXi, 0, 3),
                cute.group_modes(gXi, 0, 3),
            )
            cute.copy(x_tma, tXgXi, tXsXi, tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Forward Compute
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D, op_config_ptr):
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
        batch_size = config_dim_i32(op_config_ptr, "B", type(self))
        flat_size = batch_size * Int32(self.S * self.D)
        weight = config_flat_tensor(
            op_config_ptr,
            "weight",
            self.weight_dtype,
            flat_size,
            type(self),
        )
        if const_expr(self.has_residual):
            residual_in = config_flat_tensor(
                op_config_ptr,
                "residual_in",
                self.residual_in_dtype,
                flat_size,
                type(self),
            )
            residual_out = config_flat_tensor(
                op_config_ptr,
                "residual_out",
                self.residual_out_dtype,
                flat_size,
                type(self),
            )
        if const_expr(self.has_gate):
            gate = config_flat_tensor(
                op_config_ptr,
                "gate",
                self.gate_dtype,
                flat_size,
                type(self),
            )

        if tidx < self.effective_threads:
            row_start = tile_S * Int32(self.tile_size_S)

            for local_row in range(self.tile_size_S):
                row_idx = row_start + Int32(local_row)

                if row_idx < Int32(self.S):
                    global_offset = tile_B * Int32(self.S * self.D) + row_idx * Int32(self.D)

                    partial_sq = Float32(0.0)
                    for wi in range(self.num_tma_tiles):
                        chunk_offset = Int32(wi * self.tma_tile_D)
                        chunk_ptr = (
                            x_smem
                            + Int32(wi * self.chunk_stride_elems)
                            + Int32(local_row * self.tma_tile_D)
                        )
                        x_row = cute.make_tensor(
                            chunk_ptr,
                            cute.make_layout(self.tma_tile_D),
                        )
                        x_part = cute.local_partition(x_row, thr_layout, tidx)
                        x_reg = cute.make_fragment_like(x_part)
                        cute.autovec_copy(x_part, x_reg)

                        if const_expr(self.has_residual):
                            res_row = cute.make_tensor(
                                residual_in.iterator + global_offset + chunk_offset,
                                cute.make_layout(self.tma_tile_D),
                            )
                            res_part = cute.local_partition(res_row, thr_layout, tidx)
                            res_reg = cute.make_fragment_like(res_part)
                            cute.autovec_copy(res_part, res_reg)

                            for i in range(cute.size(x_reg)):
                                x_reg[i] = (x_reg[i].to(Float32) + res_reg[i].to(Float32)).to(self.x_dtype)

                            res_out_row = cute.make_tensor(
                                residual_out.iterator + global_offset + chunk_offset,
                                cute.make_layout(self.tma_tile_D),
                            )
                            res_out_part = cute.local_partition(res_out_row, thr_layout, tidx)
                            cute.autovec_copy(x_reg, res_out_part)
                            cute.autovec_copy(x_reg, x_part)

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
                    for wi in range(self.num_tma_tiles):
                        chunk_offset = Int32(wi * self.tma_tile_D)
                        chunk_ptr = (
                            x_smem
                            + Int32(wi * self.chunk_stride_elems)
                            + Int32(local_row * self.tma_tile_D)
                        )
                        x_row = cute.make_tensor(
                            chunk_ptr,
                            cute.make_layout(self.tma_tile_D),
                        )
                        x_part = cute.local_partition(x_row, thr_layout, tidx)
                        x_reg = cute.make_fragment_like(x_part)
                        cute.autovec_copy(x_part, x_reg)

                        w_row = cute.make_tensor(
                            weight.iterator + global_offset + chunk_offset,
                            cute.make_layout(self.tma_tile_D),
                        )
                        w_part = cute.local_partition(w_row, thr_layout, tidx)
                        w_reg = cute.make_fragment_like(w_part)
                        cute.autovec_copy(w_part, w_reg)
                        if const_expr(self.gemma):
                            for i in range(cute.size(w_reg)):
                                w_reg[i] = (w_reg[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

                        y_reg = cute.make_fragment_like(x_reg)
                        if const_expr(self.has_gate):
                            gate_row = cute.make_tensor(
                                gate.iterator + global_offset + chunk_offset,
                                cute.make_layout(self.tma_tile_D),
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

                        cute.autovec_copy(y_reg, x_part)

    # =========================================================================
    # TMA Store (S->G)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_B, tile_S, tile_D,
              y_tma, y_tma_gmem):
        """TMA store y (D × tile_S × 1) from smem[0] to global."""
        with cute.arch.elect_one():
            for wi in range(self.num_tma_tiles):
                gYi = cute.local_tile(
                    y_tma_gmem,
                    (self.tma_tile_D, self.tile_size_S, 1),
                    (Int32(wi), tile_S, tile_B),
                )
                sYi = cute.make_tensor(
                    cute.make_ptr(
                        self.x_dtype,
                        page_ptr + Int32(wi * self.chunk_stride_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.tma_tile_D, self.tile_size_S, 1)),
                )
                tYsYi, tYgYi = cute.nvgpu.cpasync.tma_partition(
                    y_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sYi, 0, 3),
                    cute.group_modes(gYi, 0, 3),
                )
                cute.copy(y_tma, tYsYi, tYgYi)


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
        "add": (None, ("B", "S", "D")),
    }
    writes = {
        "dx": (None, ("B", "S", "D")),
        "d_residual": (None, ("B", "S", "D")),
        "dgate": (None, ("B", "S", "D")),
    }
    tile = ("B", "S", "D")
    dynamic_dims = ("B",)

    tma_loads = {"x"}
    tma_stores = {"dx"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name in {"x", "dx"}:
            return (1, tile_sizes["S"], static_dims["tma_tile_D"])
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name in {"x", "dx"}:
            tile_d, tile_s, tile_b = tma_tile_shape
            return f"cute.make_layout(({tile_d}, {tile_s}, {tile_b}))"
        return None

    def __init__(self, **config):
        super().__init__(**config)
        self.residual = getattr(self, 'residual', 0)
        self.gemma = getattr(self, 'gemma', 0)
        self.has_residual = getattr(self, 'has_residual', 0)
        self.has_gate = getattr(self, 'has_gate', 0)
        self.has_add = getattr(self, 'has_add', 0)
        self.per_row_weight = getattr(self, 'per_row_weight', 0)
        if self.x_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        assert self.D >= 32 and self.D % 32 == 0

        self.tma_tile_D = getattr(
            self,
            'tma_tile_D',
            _pick_rowwise_tma_tile_n(self.D, self.tile_size_S, self.elem_bytes),
        )
        self.num_tma_tiles = self.D // self.tma_tile_D
        self.chunk_tile_elems = self.tma_tile_D * self.tile_size_S
        self.chunk_tile_bytes = self.chunk_tile_elems * self.elem_bytes
        self.chunk_stride_bytes = _align_up(self.chunk_tile_bytes, 128)
        self.chunk_stride_elems = self.chunk_stride_bytes // self.elem_bytes
        self.x_tile_bytes = self.num_tma_tiles * self.chunk_stride_bytes
        self.scratch_offset = self.x_tile_bytes

        max_warps = min(8, self.threads_per_row // 32)
        max_et = max_warps * 32
        self.effective_threads = 32
        for t in range(32, max_et + 1, 32):
            if self.D % t == 0:
                self.effective_threads = t
        self.effective_warps = self.effective_threads // 32

    kernel_config = classmethod(_tma_kernel_config)

    @classmethod
    def _fill_dummies(cls, tensors):
        """Fill dummy tensors for optional backward inputs."""
        x, dx = tensors['x'], tensors['dx']
        has_residual = 'd_residual' in tensors
        has_gate = 'gate' in tensors
        has_add = 'add' in tensors
        if not has_gate:
            tensors['gate'] = x
        if not has_add:
            tensors['add'] = x
        if not has_residual:
            tensors['d_residual'] = dx
        if 'dgate' not in tensors:
            tensors['dgate'] = dx
        return has_residual, has_gate, has_add

    @classmethod
    def schedule(cls, tile_sizes=None, residual=False, gemma=False,
                         per_row_weight=False, page_size=DEFAULT_PAGE_SIZE,
                         **tensors):
        tensors = dict(tensors)
        _expand_weight(tensors)
        has_residual, has_gate, has_add = cls._fill_dummies(tensors)
        tile_sizes = dict(tile_sizes or {})
        D = tensors['x'].shape[-1]
        elem_bytes = tensors['x'].element_size()
        max_tile_S = _auto_chunked_tile_S(D, elem_bytes, page_size, SCRATCH_BYTES)
        if "S" not in tile_sizes:
            tile_sizes["S"] = max_tile_S
        else:
            # Clamp caller-specified tile_S so scratch fits in page
            tile_sizes["S"] = min(tile_sizes["S"], max_tile_S)
            while (
                tile_sizes["S"] > 1
                and _rowwise_chunked_bytes(D, tile_sizes["S"], elem_bytes) + SCRATCH_BYTES > page_size
            ):
                tile_sizes["S"] -= 1
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
        if has_add:
            ops[0].static_dims['has_add'] = 1
        if per_row_weight:
            ops[0].static_dims['per_row_weight'] = 1
        ops[0].static_dims['page_size'] = page_size
        ops[0].static_dims['tma_tile_D'] = _pick_rowwise_tma_tile_n(
            D,
            tile_sizes["S"],
            elem_bytes,
        )
        return ops

    # =========================================================================
    # TMA Load (G->S)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_D,
             x_tma, x_tma_gmem, work_mbar):
        """TMA load x tile (D × tile_S × 1) from global to smem."""
        nbytes = Int32(self.x_tile_bytes)
        mbar_ptr = cute.make_ptr(
            cutlass.Int64, work_mbar, cute.AddressSpace.smem
        )
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        for wi in range(self.num_tma_tiles):
            gXi = cute.local_tile(
                x_tma_gmem,
                (self.tma_tile_D, self.tile_size_S, 1),
                (Int32(wi), tile_S, tile_B),
            )
            sXi = cute.make_tensor(
                cute.make_ptr(
                    self.x_dtype,
                    page_ptr + Int32(wi * self.chunk_stride_bytes),
                    cute.AddressSpace.smem,
                ),
                cute.make_layout((self.tma_tile_D, self.tile_size_S, 1)),
            )
            tXsXi, tXgXi = cute.nvgpu.cpasync.tma_partition(
                x_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sXi, 0, 3),
                cute.group_modes(gXi, 0, 3),
            )
            cute.copy(x_tma, tXgXi, tXsXi, tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Backward Compute
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D, op_config_ptr):
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
        batch_size = config_dim_i32(op_config_ptr, "B", type(self))
        flat_size = batch_size * Int32(self.S * self.D)
        dout = config_flat_tensor(
            op_config_ptr,
            "dout",
            self.dout_dtype,
            flat_size,
            type(self),
        )
        weight = config_flat_tensor(
            op_config_ptr,
            "weight",
            self.weight_dtype,
            flat_size,
            type(self),
        )
        if const_expr(self.has_gate):
            gate = config_flat_tensor(
                op_config_ptr,
                "gate",
                self.gate_dtype,
                flat_size,
                type(self),
            )
            dgate = config_flat_tensor(
                op_config_ptr,
                "dgate",
                self.dgate_dtype,
                flat_size,
                type(self),
            )
        if const_expr(self.has_add):
            add = config_flat_tensor(
                op_config_ptr,
                "add",
                self.add_dtype,
                flat_size,
                type(self),
            )
        if const_expr(self.has_residual):
            d_residual = config_flat_tensor(
                op_config_ptr,
                "d_residual",
                self.d_residual_dtype,
                flat_size,
                type(self),
            )

        if tidx < self.effective_threads:
            row_start = tile_S * Int32(self.tile_size_S)

            # Process rows sequentially (all warps cooperate per row)
            for local_row in range(self.tile_size_S):
                row_idx = row_start + Int32(local_row)

                if row_idx < Int32(self.S):
                    global_offset = tile_B * Int32(self.S * self.D) + row_idx * Int32(self.D)

                    # Fused reduction: compute both sum_sq and sum_grad
                    # in a single pass with one barrier sync
                    partial_sq = Float32(0.0)
                    partial_grad = Float32(0.0)
                    buf2_off = Int32(self.effective_warps)
                    for chunk_idx in range(self.num_tma_tiles):
                        chunk_offset = Int32(chunk_idx * self.tma_tile_D)
                        chunk_ptr = (
                            x_smem
                            + Int32(chunk_idx * self.chunk_stride_elems)
                            + Int32(local_row * self.tma_tile_D)
                        )
                        x_row = cute.make_tensor(chunk_ptr, cute.make_layout(self.tma_tile_D))
                        x_part = cute.local_partition(x_row, thr_layout, tidx)
                        x_reg = cute.make_fragment_like(x_part)
                        cute.autovec_copy(x_part, x_reg)

                        dout_row = cute.make_tensor(
                            dout.iterator + global_offset + chunk_offset,
                            cute.make_layout(self.tma_tile_D),
                        )
                        dout_part = cute.local_partition(dout_row, thr_layout, tidx)
                        dout_reg = cute.make_fragment_like(dout_part)
                        cute.autovec_copy(dout_part, dout_reg)

                        w_row = cute.make_tensor(
                            weight.iterator + global_offset + chunk_offset,
                            cute.make_layout(self.tma_tile_D),
                        )
                        w_part = cute.local_partition(w_row, thr_layout, tidx)
                        w_reg = cute.make_fragment_like(w_part)
                        cute.autovec_copy(w_part, w_reg)
                        if const_expr(self.gemma):
                            for i in range(cute.size(w_reg)):
                                w_reg[i] = (w_reg[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

                        gate_reg = cute.make_fragment_like(x_part)
                        if const_expr(self.has_gate):
                            gate_row = cute.make_tensor(
                                gate.iterator + global_offset + chunk_offset,
                                cute.make_layout(self.tma_tile_D),
                            )
                            gate_part = cute.local_partition(gate_row, thr_layout, tidx)
                            cute.autovec_copy(gate_part, gate_reg)

                        if const_expr(self.has_gate):
                            for i in range(cute.size(x_reg)):
                                val = x_reg[i].to(Float32)
                                partial_sq = partial_sq + val * val
                                g = gate_reg[i].to(Float32)
                                sig = Float32(1.0) / (Float32(1.0) + cute.math.exp(-g, fastmath=True))
                                silu_g = g * sig
                                dy_norm = dout_reg[i].to(Float32) * silu_g
                                partial_grad = partial_grad + dy_norm * w_reg[i].to(Float32) * val
                        else:
                            for i in range(cute.size(x_reg)):
                                val = x_reg[i].to(Float32)
                                partial_sq = partial_sq + val * val
                                partial_grad = partial_grad + dout_reg[i].to(Float32) * w_reg[i].to(Float32) * val

                    warp_sum = cute.arch.warp_reduction(partial_sq, operator.add)
                    warp_grad = cute.arch.warp_reduction(partial_grad, operator.add)
                    if lane_idx == 0:
                        scratch[warp_idx] = warp_sum
                        scratch[buf2_off + warp_idx] = warp_grad
                    named_barrier_sync(Int32(2), Int32(self.effective_threads))

                    sum_sq = Float32(0.0)
                    sum_grad = Float32(0.0)
                    for wi in range(self.effective_warps):
                        sum_sq = sum_sq + scratch[wi]
                        sum_grad = sum_grad + scratch[buf2_off + wi]

                    rstd = cute.math.rsqrt(
                        sum_sq / self.D + RMSNORM_EPS, fastmath=True
                    )
                    mean_grad = sum_grad / self.D

                    # Pass 3: dx [and dgate]
                    for chunk_idx in range(self.num_tma_tiles):
                        chunk_offset = Int32(chunk_idx * self.tma_tile_D)
                        chunk_ptr = (
                            x_smem
                            + Int32(chunk_idx * self.chunk_stride_elems)
                            + Int32(local_row * self.tma_tile_D)
                        )
                        x_row = cute.make_tensor(chunk_ptr, cute.make_layout(self.tma_tile_D))
                        x_part = cute.local_partition(x_row, thr_layout, tidx)
                        x_reg = cute.make_fragment_like(x_part)
                        cute.autovec_copy(x_part, x_reg)

                        dout_row = cute.make_tensor(
                            dout.iterator + global_offset + chunk_offset,
                            cute.make_layout(self.tma_tile_D),
                        )
                        dout_part = cute.local_partition(dout_row, thr_layout, tidx)
                        dout_reg = cute.make_fragment_like(dout_part)
                        cute.autovec_copy(dout_part, dout_reg)

                        w_row = cute.make_tensor(
                            weight.iterator + global_offset + chunk_offset,
                            cute.make_layout(self.tma_tile_D),
                        )
                        w_part = cute.local_partition(w_row, thr_layout, tidx)
                        w_reg = cute.make_fragment_like(w_part)
                        cute.autovec_copy(w_part, w_reg)
                        if const_expr(self.gemma):
                            for i in range(cute.size(w_reg)):
                                w_reg[i] = (w_reg[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

                        dx_reg = cute.make_fragment_like(x_reg)

                        if const_expr(self.has_gate):
                            gate_row = cute.make_tensor(
                                gate.iterator + global_offset + chunk_offset,
                                cute.make_layout(self.tma_tile_D),
                            )
                            gate_part = cute.local_partition(gate_row, thr_layout, tidx)
                            gate_reg = cute.make_fragment_like(gate_part)
                            cute.autovec_copy(gate_part, gate_reg)

                            add_reg = cute.make_fragment_like(x_part)
                            if const_expr(self.has_add):
                                add_row = cute.make_tensor(
                                    add.iterator + global_offset + chunk_offset,
                                    cute.make_layout(self.tma_tile_D),
                                )
                                add_part = cute.local_partition(add_row, thr_layout, tidx)
                                cute.autovec_copy(add_part, add_reg)
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
                                if const_expr(self.has_add):
                                    dx_val = dx_val + add_reg[i].to(Float32)

                                normed = x_val * rstd * wi
                                dgate_val = d * normed * silu_grad

                                dx_reg[i] = dx_val.to(self.x_dtype)
                                dgate_reg[i] = dgate_val.to(self.x_dtype)

                            dgate_row = cute.make_tensor(
                                dgate.iterator + global_offset + chunk_offset,
                                cute.make_layout(self.tma_tile_D),
                            )
                            dgate_part = cute.local_partition(dgate_row, thr_layout, tidx)
                            cute.autovec_copy(dgate_reg, dgate_part)
                        else:
                            add_reg = cute.make_fragment_like(x_part)
                            if const_expr(self.has_add):
                                add_row = cute.make_tensor(
                                    add.iterator + global_offset + chunk_offset,
                                    cute.make_layout(self.tma_tile_D),
                                )
                                add_part = cute.local_partition(add_row, thr_layout, tidx)
                                cute.autovec_copy(add_part, add_reg)
                            for i in range(cute.size(x_reg)):
                                d = dout_reg[i].to(Float32)
                                wi = w_reg[i].to(Float32)
                                x_val = x_reg[i].to(Float32)
                                dw_x = d * wi
                                result = (dw_x - x_val * rstd * rstd * mean_grad) * rstd
                                if const_expr(self.residual):
                                    result = result + d
                                if const_expr(self.has_add):
                                    result = result + add_reg[i].to(Float32)
                                dx_reg[i] = result.to(self.x_dtype)

                        cute.autovec_copy(dx_reg, x_part)

                        if const_expr(self.has_residual):
                            dres_row = cute.make_tensor(
                                d_residual.iterator + global_offset + chunk_offset,
                                cute.make_layout(self.tma_tile_D),
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
        with cute.arch.elect_one():
            for chunk_idx in range(self.num_tma_tiles):
                gDXi = cute.local_tile(
                    dx_tma_gmem,
                    (self.tma_tile_D, self.tile_size_S, 1),
                    (Int32(chunk_idx), tile_S, tile_B),
                )
                sDXi = cute.make_tensor(
                    cute.make_ptr(
                        self.x_dtype,
                        page_ptr + Int32(chunk_idx * self.chunk_stride_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.tma_tile_D, self.tile_size_S, 1)),
                )
                tDXsDXi, tDXgDXi = cute.nvgpu.cpasync.tma_partition(
                    dx_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sDXi, 0, 3),
                    cute.group_modes(gDXi, 0, 3),
                )
                cute.copy(dx_tma, tDXsDXi, tDXgDXi)


__all__ = [
    "RMSNormOp", "RMSNormBwdOp", "RMSNORM_EPS",
]
