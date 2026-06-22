# Copyright (c) 2025, Machete Authors
"""Qwen 3.5 full-attention forward op schedule.

This module owns the Qwen-specific forward graph shape, tiling policy, and
scratch tensor layout. Benchmarks and trace scripts should build the megakernel
around this schedule instead of assembling the forward ops inline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32

from machete.megakernel.interpreter import (
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_init,
    mbarrier_init_fence_async_proxy,
    mbarrier_inval,
    mbarrier_wait,
    named_barrier_sync,
)
from machete.kernels.attention import (
    AttentionDPSumOp,
    FlashAttentionSm120BwdOp,
    FlashAttentionSm120Op as _BaseFlashAttentionSm120Op,
)
from machete.kernels.attention.flash_decoding import (
    FlashDecodingCombineBSHDOp,
    FlashDecodingSplitBSHDOp,
    FlashPrefillCombineBSHDOp,
    FlashPrefillDirectBSHDOp,
    FlashPrefillSplitBSHDOp,
)
from machete.kernels.decode_matvec import ResidualAddSm120Op as _BaseResidualAddSm120Op
from machete.kernels.gemm import GemmOp as _BaseGemmOp
from machete.kernels.gemm.gemm import (
    ProjectionDaReduceGemmOp,
    _gemm_epilogue_store_no_mbar_inval_helper,
)
from machete.kernels.glu import DirectGLUOp, GLUBwdOp, GLUOp as _BaseGLUOp
from machete.kernels.qknorm_rope import (
    PackedQKNormRopeOp as _BasePackedQKNormRopeOp,
    QKNormRopeBwdOp,
    QKNormRopeOp as _BaseQKNormRopeOp,
)
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkG2SOp,
    CopyBulkS2GOp,
)
from machete.kernels.qwen_3_5.sm120 import (
    QWEN3_5_EPS,
    Qwen3_5PackedQkvChunkProjectSm120Op,
)
from machete.kernels.rms_norm.rms_norm import (
    RMSNormBwdOp,
    RMSNormOp,
    SCRATCH_BYTES,
    _auto_chunked_tile_S,
    _expand_weight,
    _pick_rmsnorm_tma_tile_d,
    _rowwise_chunked_bytes,
)
from machete.megakernel.dim_windows import iter_dim_windows
from machete.megakernel.scheduling import OverlapTileScheduler
from machete.megakernel.ops import (
    AccessRegions,
    Op,
    RegionAxis,
    TensorAccessRegion,
    config_dim_i32,
    iter_tensor_access_regions,
)


DEFAULT_PAGE_SIZE = 32768
HIDDEN = 1024
INTERMEDIATE = 3584
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 256
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM
KV_GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS


def _region_axis_from_tile(op, canonical_name: str, tile_dim: str) -> RegionAxis:
    return RegionAxis(
        name=canonical_name,
        extent=int(op.static_dims.get(tile_dim, 1)),
        tile_dim=tile_dim,
        tile_size=int(op.tile_sizes.get(tile_dim, 1)),
        tile_origin=int(op.tile_origins.get(tile_dim, 0)),
    )


def _rename_region_tile_dim(region: TensorAccessRegion, tile_dim: str, canonical_name: str) -> TensorAccessRegion:
    return replace(
        region,
        axes=tuple(
            replace(axis, name=canonical_name)
            if axis.tile_dim == tile_dim
            else axis
            for axis in region.axes
        ),
    )


def _region_with_causal_sequence_prefix(
    region: TensorAccessRegion,
    op,
    *,
    sequence_axis_name: str = "N",
) -> TensorAccessRegion:
    """Map an attention K/V sequence axis to a causal prefix of producer S tiles."""
    return replace(
        region,
        axes=tuple(
            replace(
                axis,
                name="S",
                tile_dim="M",
                tile_size=int(op.tile_sizes.get("M", 1)),
                tile_origin=int(op.tile_origins.get("M", 0)),
            )
            if axis.name == sequence_axis_name
            else axis
            for axis in region.axes
        ),
        prefix_dim="S",
        prefix_index_dim="S",
    )


def _region_with_group(
    region: TensorAccessRegion,
    *,
    group_dim: str | None = None,
    group_tiles: int = 1,
    group_count: int | None = None,
    group_index: int | None = None,
    group_index_offset: int = 0,
    group_index_dim: str | None = None,
    group_index_group_tiles: int = 1,
    group_index_mode: str = "exact",
    group_index_all: bool = False,
    prefix_dim: str | None = None,
    prefix_index_dim: str | None = None,
    prefix_group_tiles: int = 1,
    prefix_group_count: int | None = None,
) -> TensorAccessRegion:
    return replace(
        region,
        group_dim=group_dim,
        group_tiles=int(group_tiles),
        group_count=None if group_count is None else int(group_count),
        group_index=None if group_index is None else int(group_index),
        group_index_offset=int(group_index_offset),
        group_index_dim=group_index_dim,
        group_index_group_tiles=int(group_index_group_tiles),
        group_index_mode=group_index_mode,
        group_index_all=bool(group_index_all),
        prefix_dim=prefix_dim,
        prefix_index_dim=prefix_index_dim,
        prefix_group_tiles=int(prefix_group_tiles),
        prefix_group_count=None if prefix_group_count is None else int(prefix_group_count),
    )


def _rename_sequence_regions(regions: AccessRegions) -> AccessRegions:
    def _rename_access(access):
        renamed = tuple(
            _rename_region_tile_dim(region, "M", "S")
            for region in iter_tensor_access_regions(access)
        )
        if not renamed:
            return access
        return renamed[0] if len(renamed) == 1 else renamed

    return AccessRegions(
        reads={
            name: _rename_access(region)
            for name, region in regions.reads.items()
        },
        writes={
            name: _rename_access(region)
            for name, region in regions.writes.items()
        },
    )


def _slice_region_axis(
    region: TensorAccessRegion,
    axis_name: str,
    *,
    tensor: str,
    start: int,
    stop: int,
) -> TensorAccessRegion:
    return replace(
        region,
        tensor=tensor,
        axes=tuple(
            replace(axis, start=int(start), stop=int(stop))
            if axis.name == axis_name or axis.tile_dim == axis_name
            else axis
            for axis in region.axes
        ),
        group_dim=None,
        group_tiles=1,
        group_count=None,
        group_index=None,
        group_index_offset=0,
        group_index_dim=None,
        group_index_group_tiles=1,
        group_index_mode="exact",
        group_index_all=False,
        prefix_dim=None,
        prefix_index_dim=None,
        prefix_group_tiles=1,
        prefix_group_count=None,
    )


def _split_packed_qkv_write_regions(regions: AccessRegions, *, buffer_name: str = "c") -> dict | None:
    """Expose packed QKV projection output as logical QK and V producer slices."""
    c_region = regions.writes.get(buffer_name)
    if not isinstance(c_region, TensorAccessRegion):
        return None
    n_axis = c_region.axis("N")
    if n_axis is None or n_axis.tile_dim != "N":
        return None
    n_extent = int(n_axis.extent)
    writes = dict(regions.writes)
    if n_extent == Q_DIM + 2 * KV_DIM:
        writes[buffer_name] = (
            _slice_region_axis(
                c_region,
                "N",
                tensor="qk",
                start=0,
                stop=Q_DIM + KV_DIM,
            ),
            _slice_region_axis(
                c_region,
                "N",
                tensor="v",
                start=Q_DIM + KV_DIM,
                stop=Q_DIM + 2 * KV_DIM,
            ),
        )
        return writes
    if n_extent == Q_DIM + KV_DIM:
        writes[buffer_name] = _slice_region_axis(
            c_region,
            "N",
            tensor="qk",
            start=0,
            stop=Q_DIM + KV_DIM,
        )
        return writes
    return None


def _qwen_attention_regions(op, base_regions: AccessRegions) -> AccessRegions:
    regions = _rename_sequence_regions(base_regions)
    reads = dict(regions.reads)
    writes = dict(regions.writes)
    causal_prefix = bool(int(op.static_dims.get("causal", 0)))

    k_region = reads.get("k")
    k_meta = op.tensor_metas.get("k")
    if k_region is not None and k_meta is not None:
        # Packed Q/K stores K immediately after Q. The producer declares one
        # scheduler-visible H group per packed head tile; for Qwen's packed
        # qknorm tile that means Q is group 0 and K is group 1. Do not derive
        # this from the consumer attention H tile, which is usually different.
        k_group_index = 1 if int(k_meta.storage_offset) >= Q_DIM else 0
        if causal_prefix:
            k_region = _region_with_causal_sequence_prefix(k_region, op)
        reads["k"] = _region_with_group(
            k_region,
            group_index=k_group_index,
        )

    v_region = reads.get("v")
    v_meta = op.tensor_metas.get("v")
    if v_region is not None and v_meta is not None:
        if causal_prefix:
            v_region = _region_with_causal_sequence_prefix(v_region, op)
        reads["v"] = _region_with_group(
            v_region,
            group_index=int(v_meta.storage_offset // KV_DIM),
        )

    o_region = writes.get("o")
    h_axis = o_region.axis("H") if o_region is not None else None
    if o_region is not None and h_axis is not None and h_axis.tile_dim == "H":
        writes["o"] = _region_with_group(
            o_region,
            group_dim="H",
            group_tiles=1,
            group_count=op.tile_counts[op.dim_names["H"]],
        )

    return AccessRegions(reads=reads, writes=writes)


@dataclass
class Qwen3_5ForwardSchedule:
    ops: list
    output: torch.Tensor
    residual: torch.Tensor
    keep_alive: list


@dataclass
class Qwen3_5BackwardSchedule:
    ops: list
    output: torch.Tensor
    keep_alive: list


class Qwen3_5ForwardRMSNormOp(RMSNormOp):
    """RMSNorm schedule variant for Qwen forward overlap.

    The generic RMSNorm schedule caps large hidden-size tiles aggressively.
    For Qwen forward, the producer tile must be compatible with the following
    projection GEMM row tile so the GEMM can start after only its row slice is
    normalized. This schedule uses the largest row tile that fits the shared
    memory page and opts into generic piecewise row dependencies.
    """

    @classmethod
    def access_regions(cls, op) -> AccessRegions:
        regions = super().access_regions(op)
        writes = dict(regions.writes)
        for name in ("y", "residual_out"):
            region = writes.get(name)
            d_axis = region.axis("D") if region is not None else None
            if region is not None and d_axis is not None and d_axis.tile_dim == "D":
                d_tiles = op.tile_counts[op.dim_names["D"]]
                writes[name] = _region_with_group(
                    region,
                    group_dim="D",
                    group_tiles=d_tiles,
                    group_count=1,
                )
        return AccessRegions(reads=regions.reads, writes=writes)

    @classmethod
    def _fill_dummies(cls, tensors):
        x, y = tensors["x"], tensors["y"]
        has_residual = "residual_in" in tensors
        has_gate = "gate" in tensors
        if not has_residual:
            tensors["residual_in"] = torch.empty_like(x)
        if "residual_out" not in tensors:
            tensors["residual_out"] = torch.empty_like(y)
        if not has_gate:
            tensors["gate"] = torch.empty_like(x)
        return has_residual, has_gate

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        residual=False,
        gemma=False,
        per_row_weight=False,
        page_size=DEFAULT_PAGE_SIZE,
        **tensors,
    ):
        tensors = dict(tensors)
        _expand_weight(tensors)
        has_residual, has_gate = cls._fill_dummies(tensors)
        tile_sizes = dict(tile_sizes or {})
        D = tensors["x"].shape[-1]
        elem_bytes = tensors["x"].element_size()
        fit_tile_s = _auto_chunked_tile_S(D, elem_bytes, page_size, SCRATCH_BYTES)
        if "S" not in tile_sizes:
            tile_sizes["S"] = fit_tile_s
        else:
            tile_sizes["S"] = min(tile_sizes["S"], fit_tile_s)
            while (
                tile_sizes["S"] > 1
                and _rowwise_chunked_bytes(D, tile_sizes["S"], elem_bytes) + SCRATCH_BYTES > page_size
            ):
                tile_sizes["S"] -= 1
        tile_sizes.setdefault("B", 1)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        if residual:
            ops[0].static_dims["residual"] = 1
        if gemma:
            ops[0].static_dims["gemma"] = 1
        if has_residual:
            ops[0].static_dims["has_residual"] = 1
        if has_gate:
            ops[0].static_dims["has_gate"] = 1
        if per_row_weight:
            ops[0].static_dims["per_row_weight"] = 1
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["tma_tile_D"] = _pick_rmsnorm_tma_tile_d(
            D,
            tile_sizes["S"],
            elem_bytes,
            page_size,
        )
        ops[0].static_dims["barrier_allow_piecewise_overlap"] = 1
        return ops


class Qwen3_5RmsProjOp(_BaseGemmOp):
    """Fused RMSNorm + projection GEMM, written for the Qwen forward layer.

    Computes ``C = rstd * (x @ B_fused^T)`` in ONE op, where
    ``B_fused[n,k] = w[n,k] * rmsnorm_weight[k]`` (baked at schedule time) and
    ``rstd = rsqrt(mean(x^2) + eps)`` per row.

    Unlike the generic ``RMSNormGemmOp`` (which re-reads A from global in the
    epilogue to form ``sum(x^2)`` — a net loss because the projection already
    re-reads A once per N-tile), this op accumulates ``sum(x^2)`` per row *from
    the A register fragments that are already loaded for the MMA* during the
    K-loop. That adds only cheap ALU and ZERO extra memory traffic, so the fused
    op removes the separate RMSNorm pass and its ``x0`` global round-trip for
    free. The persistent megakernel keeps ``x`` resident / prefetched, an
    advantage torch.compile cannot match (it must materialize ``x0`` to global
    between its rmsnorm and matmul kernels).

    A-fragment row mapping (m16n8k16): each warp owns 16 rows; lane ``l`` owns
    rows ``g=l//4`` (lo) and ``g+8`` (hi). The 4 lanes sharing ``g`` cover
    disjoint K columns, so a butterfly reduction over the group-of-4 sums the
    full row. This matches the C-fragment scaling (``ci%4<2`` -> lo).
    """

    dynamic_dims = ("B",)

    @classmethod
    def access_regions(cls, op) -> AccessRegions:
        regions = super().access_regions(op)
        writes = _split_packed_qkv_write_regions(regions)
        if writes is None:
            return regions
        return AccessRegions(reads=regions.reads, writes=writes)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_N, a, a_scale, b, c):
        tidx = cute.arch.thread_idx()[0]

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(self.a_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)
        smem_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.a_dtype)
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        smem_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.b_dtype)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        def _mk_A(off):
            return cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.a_dtype, page_ptr + Int32(off),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.a_dtype),
                cute.make_layout((self.tile_size_S, self.tile_K), stride=(self.tile_K, 1)))

        def _mk_B(off):
            return cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.b_dtype, page_ptr + Int32(off),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.b_dtype),
                cute.make_layout((self.tile_size_N, self.tile_K), stride=(self.tile_K, 1)))

        sA_0 = _mk_A(0)
        sB_0 = _mk_B(self.b_offset)
        sA_1 = _mk_A(self.buf_stride)
        sB_1 = _mk_B(self.buf_stride + self.b_offset)

        tCsA = thr_mma.partition_A(sA_0)
        tCsB = thr_mma.partition_B(sB_0)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrA_ld = smem_thr_copy_A.retile(tCrA)
        tCrB_ld = smem_thr_copy_B.retile(tCrB)
        tAsA_ld_0 = smem_thr_copy_A.partition_S(sA_0)
        tBsB_ld_0 = smem_thr_copy_B.partition_S(sB_0)
        tAsA_ld_1 = smem_thr_copy_A.partition_S(sA_1)
        tBsB_ld_1 = smem_thr_copy_B.partition_S(sB_1)

        _bf_0 = page_ptr + Int32(self.mbar_offset)
        _bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        _kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        _kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        acc = cute.make_fragment(
            tiled_mma.partition_shape_C((self.tile_size_S, self.tile_size_N)), Float32)
        acc.fill(0.0)
        sq_lo = Float32(0.0)
        sq_hi = Float32(0.0)

        # Accumulate sum(x^2) for this thread's two rows from the A fragment of
        # one 16-wide k-block. lo rows = fragment elems where (i//2)%2==0.
        def _accum_sq(frag, sq_lo, sq_hi):
            for i in cutlass.range_constexpr(cute.size(frag)):
                v = frag[i].to(Float32)
                if (i // 2) % 2 == 0:
                    sq_lo = sq_lo + v * v
                else:
                    sq_hi = sq_hi + v * v
            return sq_lo, sq_hi

        for kb in cutlass.range_constexpr(self.tile_K // 16):
            cute.copy(smem_tiled_copy_A, tAsA_ld_0[None, None, kb], tCrA_ld[None, None, kb])
            cute.copy(smem_tiled_copy_B, tBsB_ld_0[None, None, kb], tCrB_ld[None, None, kb])
            sq_lo, sq_hi = _accum_sq(tCrA[None, None, kb], sq_lo, sq_hi)
            cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
        if tidx % Int32(32) == Int32(0):
            mbarrier_arrive(_bf_0)

        if self.num_k_blocks >= 2:
            for kb in cutlass.range_constexpr(self.tile_K // 16):
                cute.copy(smem_tiled_copy_A, tAsA_ld_1[None, None, kb], tCrA_ld[None, None, kb])
                cute.copy(smem_tiled_copy_B, tBsB_ld_1[None, None, kb], tCrB_ld[None, None, kb])
                sq_lo, sq_hi = _accum_sq(tCrA[None, None, kb], sq_lo, sq_hi)
                cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
            if tidx % Int32(32) == Int32(0):
                mbarrier_arrive(_bf_1)

        _kr_phase_0 = Int32(0)
        _kr_phase_1 = Int32(0)
        k_idx = Int32(2)
        while k_idx < Int32(self.num_k_blocks):
            if k_idx % Int32(2) == Int32(0):
                mbarrier_wait(_kr_0, _kr_phase_0)
                _kr_phase_0 = _kr_phase_0 ^ Int32(1)
                for kb in cutlass.range_constexpr(self.tile_K // 16):
                    cute.copy(smem_tiled_copy_A, tAsA_ld_0[None, None, kb], tCrA_ld[None, None, kb])
                    cute.copy(smem_tiled_copy_B, tBsB_ld_0[None, None, kb], tCrB_ld[None, None, kb])
                    sq_lo, sq_hi = _accum_sq(tCrA[None, None, kb], sq_lo, sq_hi)
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
                if tidx % Int32(32) == Int32(0):
                    mbarrier_arrive(_bf_0)
            else:
                mbarrier_wait(_kr_1, _kr_phase_1)
                _kr_phase_1 = _kr_phase_1 ^ Int32(1)
                for kb in cutlass.range_constexpr(self.tile_K // 16):
                    cute.copy(smem_tiled_copy_A, tAsA_ld_1[None, None, kb], tCrA_ld[None, None, kb])
                    cute.copy(smem_tiled_copy_B, tBsB_ld_1[None, None, kb], tCrB_ld[None, None, kb])
                    sq_lo, sq_hi = _accum_sq(tCrA[None, None, kb], sq_lo, sq_hi)
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
                if tidx % Int32(32) == Int32(0):
                    mbarrier_arrive(_bf_1)
            k_idx = k_idx + Int32(1)

        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
        if tidx == Int32(0):
            mbarrier_inval(_bf_0)
            mbarrier_inval(_bf_1)
            mbarrier_inval(_kr_0)
            mbarrier_inval(_kr_1)

        # Reduce sum(x^2) across the 4 lanes that share each row, then rstd.
        sq_lo = sq_lo + cute.arch.shuffle_sync_bfly(sq_lo, offset=1)
        sq_lo = sq_lo + cute.arch.shuffle_sync_bfly(sq_lo, offset=2)
        sq_hi = sq_hi + cute.arch.shuffle_sync_bfly(sq_hi, offset=1)
        sq_hi = sq_hi + cute.arch.shuffle_sync_bfly(sq_hi, offset=2)
        _inv_K = Float32(1.0 / self.K)
        rstd_lo = cute.math.rsqrt(sq_lo * _inv_K + Float32(QWEN3_5_EPS), fastmath=True)
        rstd_hi = cute.math.rsqrt(sq_hi * _inv_K + Float32(QWEN3_5_EPS), fastmath=True)

        for ci in cutlass.range_constexpr(cute.size(acc)):
            if ci % 4 < 2:
                acc[ci] = acc[ci] * rstd_lo
            else:
                acc[ci] = acc[ci] * rstd_hi

        swz_c = cute.make_swizzle(self.swz_B_c, 4, 3)
        sC = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.c_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                swz_c, dtype=self.c_dtype),
            cute.make_layout((self.tile_size_S, self.tile_size_N),
                             stride=(self.tile_size_N, 1)))
        acc_out = cute.make_fragment_like(acc, self.c_dtype)
        for ci in cutlass.range_constexpr(cute.size(acc)):
            acc_out[ci] = acc[ci].to(self.c_dtype)
        r2s_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        r2s_copy = cute.make_tiled_copy_C(r2s_atom, tiled_mma)
        r2s_thr = r2s_copy.get_slice(tidx)
        cute.copy(r2s_copy, r2s_thr.retile(acc_out), r2s_thr.partition_D(sC))

    @classmethod
    def schedule(cls, rmsnorm_weight=None, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        b = tensors["b"]
        if rmsnorm_weight is not None:
            w = rmsnorm_weight.float()
            tensors["b"] = (b.float() * w.unsqueeze(0)).to(b.dtype).contiguous()
        tile_sizes = dict(tile_sizes or {})
        a = tensors.get("a")
        if a is not None:
            _complete_qwen_forward_gemm_tile_sizes(tile_sizes, a, b, page_size)
        ops = super().schedule(tile_sizes=tile_sizes, page_size=page_size, **tensors)
        for op in ops:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        return ops


class Qwen3_5DownResidualOp(_BaseGemmOp):
    """Fused down-projection GEMM + residual add: ``y = mlp @ w_down^T + residual``.

    Folds the final residual add into the down GEMM epilogue, removing the
    standalone residual-add op (a serial tail in the layer) and the ``mlp_out``
    global round-trip. ``residual`` is a regular (non-TMA) read, so the framework
    hands it to ``compute`` as a global CuTe tensor; the epilogue adds it through
    ``partition_C`` (identical layout to the MMA accumulator), needing no manual
    C-fragment row/col mapping.
    """

    reads = {**_BaseGemmOp.reads, "residual": (None, ("B", "S", "N"))}
    dynamic_dims = ("B",)

    @classmethod
    def access_regions(cls, op) -> AccessRegions:
        regions = super().access_regions(op)
        reads = dict(regions.reads)
        a_region = reads.get("a")
        if a_region is not None:
            reads["a"] = _region_with_group(a_region, group_index_all=True)
        return AccessRegions(reads=reads, writes=regions.writes)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_N, a, a_scale, b, residual, c):
        # NOTE: param order MUST match the framework's canonical tensor order
        # (reads then writes => a, a_scale, b, residual, c). The wrapper passes
        # tensors positionally, so listing c before residual silently swaps them.
        tidx = cute.arch.thread_idx()[0]

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(self.a_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)
        smem_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.a_dtype)
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        smem_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.b_dtype)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        def _mk_A(off):
            return cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.a_dtype, page_ptr + Int32(off),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.a_dtype),
                cute.make_layout((self.tile_size_S, self.tile_K), stride=(self.tile_K, 1)))

        def _mk_B(off):
            return cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.b_dtype, page_ptr + Int32(off),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.b_dtype),
                cute.make_layout((self.tile_size_N, self.tile_K), stride=(self.tile_K, 1)))

        sA_0 = _mk_A(0)
        sB_0 = _mk_B(self.b_offset)
        sA_1 = _mk_A(self.buf_stride)
        sB_1 = _mk_B(self.buf_stride + self.b_offset)

        tCsA = thr_mma.partition_A(sA_0)
        tCsB = thr_mma.partition_B(sB_0)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrA_ld = smem_thr_copy_A.retile(tCrA)
        tCrB_ld = smem_thr_copy_B.retile(tCrB)
        tAsA_ld_0 = smem_thr_copy_A.partition_S(sA_0)
        tBsB_ld_0 = smem_thr_copy_B.partition_S(sB_0)
        tAsA_ld_1 = smem_thr_copy_A.partition_S(sA_1)
        tBsB_ld_1 = smem_thr_copy_B.partition_S(sB_1)

        _bf_0 = page_ptr + Int32(self.mbar_offset)
        _bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        _kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        _kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        acc = cute.make_fragment(
            tiled_mma.partition_shape_C((self.tile_size_S, self.tile_size_N)), Float32)
        acc.fill(0.0)

        for kb in cutlass.range_constexpr(self.tile_K // 16):
            cute.copy(smem_tiled_copy_A, tAsA_ld_0[None, None, kb], tCrA_ld[None, None, kb])
            cute.copy(smem_tiled_copy_B, tBsB_ld_0[None, None, kb], tCrB_ld[None, None, kb])
            cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
        if tidx % Int32(32) == Int32(0):
            mbarrier_arrive(_bf_0)

        if self.num_k_blocks >= 2:
            for kb in cutlass.range_constexpr(self.tile_K // 16):
                cute.copy(smem_tiled_copy_A, tAsA_ld_1[None, None, kb], tCrA_ld[None, None, kb])
                cute.copy(smem_tiled_copy_B, tBsB_ld_1[None, None, kb], tCrB_ld[None, None, kb])
                cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
            if tidx % Int32(32) == Int32(0):
                mbarrier_arrive(_bf_1)

        _kr_phase_0 = Int32(0)
        _kr_phase_1 = Int32(0)
        k_idx = Int32(2)
        while k_idx < Int32(self.num_k_blocks):
            if k_idx % Int32(2) == Int32(0):
                mbarrier_wait(_kr_0, _kr_phase_0)
                _kr_phase_0 = _kr_phase_0 ^ Int32(1)
                for kb in cutlass.range_constexpr(self.tile_K // 16):
                    cute.copy(smem_tiled_copy_A, tAsA_ld_0[None, None, kb], tCrA_ld[None, None, kb])
                    cute.copy(smem_tiled_copy_B, tBsB_ld_0[None, None, kb], tCrB_ld[None, None, kb])
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
                if tidx % Int32(32) == Int32(0):
                    mbarrier_arrive(_bf_0)
            else:
                mbarrier_wait(_kr_1, _kr_phase_1)
                _kr_phase_1 = _kr_phase_1 ^ Int32(1)
                for kb in cutlass.range_constexpr(self.tile_K // 16):
                    cute.copy(smem_tiled_copy_A, tAsA_ld_1[None, None, kb], tCrA_ld[None, None, kb])
                    cute.copy(smem_tiled_copy_B, tBsB_ld_1[None, None, kb], tCrB_ld[None, None, kb])
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
                if tidx % Int32(32) == Int32(0):
                    mbarrier_arrive(_bf_1)
            k_idx = k_idx + Int32(1)

        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
        if tidx == Int32(0):
            mbarrier_inval(_bf_0)
            mbarrier_inval(_bf_1)
            mbarrier_inval(_kr_0)
            mbarrier_inval(_kr_1)

        # Fuse the residual add: y = acc + residual. Build a global view of this
        # output tile [tile_size_S, tile_size_N] (contiguous [B,S,N], row stride
        # = N) and let the tiled MMA partition it exactly like the accumulator,
        # so acc[ci] and tCres[ci] are the same (row,col) — no manual mapping.
        _res_base = (
            tile_B * Int32(self.S * self.N)
            + tile_S * Int32(self.tile_size_S) * Int32(self.N)
            + tile_N * Int32(self.tile_size_N)
        )
        g_res = cute.make_tensor(
            residual.iterator + _res_base,
            cute.make_layout((self.tile_size_S, self.tile_size_N), stride=(self.N, 1)))
        tCres = thr_mma.partition_C(g_res)
        for ci in cutlass.range_constexpr(cute.size(acc)):
            acc[ci] = acc[ci] + tCres[ci].to(Float32)

        swz_c = cute.make_swizzle(self.swz_B_c, 4, 3)
        sC = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.c_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                swz_c, dtype=self.c_dtype),
            cute.make_layout((self.tile_size_S, self.tile_size_N),
                             stride=(self.tile_size_N, 1)))
        acc_out = cute.make_fragment_like(acc, self.c_dtype)
        for ci in cutlass.range_constexpr(cute.size(acc)):
            acc_out[ci] = acc[ci].to(self.c_dtype)
        r2s_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        r2s_copy = cute.make_tiled_copy_C(r2s_atom, tiled_mma)
        r2s_thr = r2s_copy.get_slice(tidx)
        cute.copy(r2s_copy, r2s_thr.retile(acc_out), r2s_thr.partition_D(sC))

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        if "a_scale" not in tensors:
            a = tensors.get("a")
            if a is not None:
                tensors["a_scale"] = torch.empty_like(a)
        tile_sizes = dict(tile_sizes or {})
        a = tensors.get("a")
        b = tensors.get("b")
        if a is not None and b is not None:
            _complete_qwen_forward_gemm_tile_sizes(tile_sizes, a, b, page_size)
        ops = super().schedule(tile_sizes=tile_sizes, page_size=page_size, **tensors)
        for op in ops:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        return ops


class Qwen3_5ForwardProjectionOp(_BaseGemmOp):
    """Qwen forward projection GEMM.

    This keeps Qwen forward scheduling policy out of the generic GEMM op.  The
    current implementation still uses the proven GEMM phase bodies, but all
    tile/dependency choices are owned here so the op can diverge without
    changing other models.
    """

    @classmethod
    def access_regions(cls, op) -> AccessRegions:
        regions = super().access_regions(op)
        reads = dict(regions.reads)
        a_region = reads.get("a")
        if a_region is not None:
            reads["a"] = _region_with_group(a_region, group_index_all=True)
        c_region = regions.writes.get("c")
        if c_region is None:
            return regions
        split_writes = _split_packed_qkv_write_regions(regions)
        if split_writes is not None:
            return AccessRegions(reads=reads, writes=split_writes)
        n_axis = c_region.axis("N")
        if n_axis is None or n_axis.tile_dim != "N":
            return AccessRegions(reads=reads, writes=regions.writes)
        n_extent = int(n_axis.extent)
        if n_extent in (KV_DIM, Q_DIM + KV_DIM, Q_DIM + 2 * KV_DIM):
            group_elems = KV_DIM
        elif n_extent == 2 * INTERMEDIATE:
            group_elems = int(n_axis.tile_size)
        else:
            return AccessRegions(reads=reads, writes=regions.writes)
        group_tiles = max(1, group_elems // int(n_axis.tile_size))
        group_count = max(1, (n_extent + group_elems - 1) // group_elems)
        writes = dict(regions.writes)
        writes["c"] = _region_with_group(
            c_region,
            group_dim="N",
            group_tiles=group_tiles,
            group_count=group_count,
        )
        return AccessRegions(reads=reads, writes=writes)

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        **tensors,
    ):
        if "a_scale" not in tensors:
            a = tensors.get("a")
            if a is not None:
                # Generic GEMM uses a_scale=a as a no-op placeholder when
                # has_a_scale=0. In the Qwen forward schedule that creates a
                # second, false producer dependency on every projection input.
                # Keep a same-shaped independent placeholder instead; the
                # kernel never loads it while has_a_scale=0.
                tensors["a_scale"] = torch.empty_like(a)
        tile_sizes = dict(tile_sizes or {})
        a = tensors.get("a")
        b = tensors.get("b")
        if a is not None and b is not None:
            _complete_qwen_forward_gemm_tile_sizes(tile_sizes, a, b, page_size)
        ops = super().schedule(tile_sizes=tile_sizes, page_size=page_size, **tensors)
        for op in ops:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        return ops


class Qwen3_5ForwardCpAsyncProjectionOp(_BaseGemmOp):
    """Hybrid Qwen projection GEMM.

    The load warp TMA-prefetches the first two K-blocks so dependency/scheduler
    overlap stays identical to the regular GEMM prologue.  MMA warps then load
    later K-blocks with cp.async inside compute, leaving the framework load warp
    available for other ops instead of driving the long inner-K tail.
    """

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_N,
             a_tma, a_tma_gmem, a_scale_tma, a_scale_tma_gmem,
             b_tma, b_tma_gmem,
             work_mbar):
        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)
        _mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, Int32(self.tma_k_blocks * self.ab_tma_bytes))

        _k_block = Int32(0)
        while _k_block < Int32(self.tma_k_blocks):
            _buf_base = (_k_block % Int32(2)) * Int32(self.buf_stride) + page_ptr

            sA_ptr = cute.recast_ptr(
                cute.make_ptr(self.a_dtype, _buf_base, cute.AddressSpace.smem),
                swz, dtype=self.a_dtype)
            sA = cute.make_tensor(
                sA_ptr,
                cute.make_layout((self.tile_K, self.tile_size_S, 1),
                                 stride=(1, self.tile_K, self.tile_K * self.tile_size_S)))
            gA = cute.local_tile(a_tma_gmem, (self.tile_K, self.tile_size_S, 1), (None, None, None))
            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                a_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sA, 0, 3), cute.group_modes(gA, 0, 3))

            sB_ptr = cute.recast_ptr(
                cute.make_ptr(self.b_dtype, _buf_base + Int32(self.b_offset), cute.AddressSpace.smem),
                swz, dtype=self.b_dtype)
            sB = cute.make_tensor(
                sB_ptr,
                cute.make_layout((self.tile_K, self.tile_size_N), stride=(1, self.tile_K)))
            gB = cute.local_tile(b_tma_gmem, (self.tile_K, self.tile_size_N), (None, None))
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                b_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))

            cute.copy(a_tma, tAgA[(None, _k_block, tile_S, tile_B)], tAsA, tma_bar_ptr=_mbar_ptr)
            cute.copy(b_tma, tBgB[(None, _k_block, tile_N)], tBsB, tma_bar_ptr=_mbar_ptr)
            _k_block = _k_block + Int32(1)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_N, a, a_scale, b):
        tidx = cute.arch.thread_idx()[0]

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(self.a_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps_m, self.num_mma_warps_n, 1)),
            permutation_mnk=(self.num_mma_warps_m * 16, self.num_mma_warps_n * 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)
        smem_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.a_dtype)
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        smem_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.b_dtype)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        sA_0 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.a_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.a_dtype),
            cute.make_layout((self.tile_size_S, self.tile_K), stride=(self.tile_K, 1)))
        sB_0 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.b_dtype, page_ptr + Int32(self.b_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.b_dtype),
            cute.make_layout((self.tile_size_N, self.tile_K), stride=(self.tile_K, 1)))
        sA_1 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.a_dtype, page_ptr + Int32(self.buf_stride),
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.a_dtype),
            cute.make_layout((self.tile_size_S, self.tile_K), stride=(self.tile_K, 1)))
        sB_1 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.b_dtype, page_ptr + Int32(self.buf_stride + self.b_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.b_dtype),
            cute.make_layout((self.tile_size_N, self.tile_K), stride=(self.tile_K, 1)))

        tCsA = thr_mma.partition_A(sA_0)
        tCsB = thr_mma.partition_B(sB_0)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrA_ld = smem_thr_copy_A.retile(tCrA)
        tCrB_ld = smem_thr_copy_B.retile(tCrB)
        tAsA_ld_0 = smem_thr_copy_A.partition_S(sA_0)
        tBsB_ld_0 = smem_thr_copy_B.partition_S(sB_0)
        tAsA_ld_1 = smem_thr_copy_A.partition_S(sA_1)
        tBsB_ld_1 = smem_thr_copy_B.partition_S(sB_1)

        async_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), self.a_dtype, num_bits_per_copy=128)
        async_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), self.b_dtype, num_bits_per_copy=128)
        copy_thread_layout = cute.make_layout(
            (self.cpasync_copy_dim0, self.cpasync_copy_dim1),
            stride=(self.cpasync_copy_dim1, 1))
        copy_value_layout = cute.make_layout((1, self.cpasync_copy_elems))
        g2s_copy_A = cute.make_tiled_copy_tv(async_copy_atom_A, copy_thread_layout, copy_value_layout)
        g2s_copy_B = cute.make_tiled_copy_tv(async_copy_atom_B, copy_thread_layout, copy_value_layout)
        thr_copy_A = g2s_copy_A.get_slice(tidx)
        thr_copy_B = g2s_copy_B.get_slice(tidx)
        tAsA_cp_0 = thr_copy_A.partition_D(sA_0)
        tBsB_cp_0 = thr_copy_B.partition_D(sB_0)
        tAsA_cp_1 = thr_copy_A.partition_D(sA_1)
        tBsB_cp_1 = thr_copy_B.partition_D(sB_1)

        a_base = (
            tile_B * Int32(self.a_stride_B)
            + tile_S * Int32(self.tile_size_S) * Int32(self.a_stride_S)
        )
        b_base = tile_N * Int32(self.tile_size_N) * Int32(self.b_stride_N)
        gA_full = cute.make_tensor(
            (a.iterator + a_base).align(16),
            cute.make_layout((self.tile_size_S, self.K), stride=(self.a_stride_S, self.a_stride_K)))
        gB_full = cute.make_tensor(
            (b.iterator + b_base).align(16),
            cute.make_layout((self.tile_size_N, self.K), stride=(self.b_stride_N, self.b_stride_K)))

        acc = cute.make_fragment(
            tiled_mma.partition_shape_C((self.tile_size_S, self.tile_size_N)), Float32)
        acc.fill(0.0)

        k_idx = Int32(0)
        while k_idx < Int32(self.num_k_blocks):
            if k_idx >= Int32(2):
                if k_idx + Int32(1) < Int32(self.num_k_blocks):
                    cute.arch.cp_async_wait_group(1)
                if k_idx + Int32(1) >= Int32(self.num_k_blocks):
                    cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            if k_idx % Int32(2) == Int32(0):
                for k_block in cutlass.range_constexpr(self.tile_K // 16):
                    cute.copy(smem_tiled_copy_A, tAsA_ld_0[None, None, k_block], tCrA_ld[None, None, k_block])
                    cute.copy(smem_tiled_copy_B, tBsB_ld_0[None, None, k_block], tCrB_ld[None, None, k_block])
                    cute.gemm(tiled_mma, acc, tCrA[None, None, k_block], tCrB[None, None, k_block], acc)
            if k_idx % Int32(2) == Int32(1):
                for k_block in cutlass.range_constexpr(self.tile_K // 16):
                    cute.copy(smem_tiled_copy_A, tAsA_ld_1[None, None, k_block], tCrA_ld[None, None, k_block])
                    cute.copy(smem_tiled_copy_B, tBsB_ld_1[None, None, k_block], tCrB_ld[None, None, k_block])
                    cute.gemm(tiled_mma, acc, tCrA[None, None, k_block], tCrB[None, None, k_block], acc)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            prefetch_idx = k_idx + Int32(2)
            if prefetch_idx < Int32(self.num_k_blocks):
                gA_block = cute.local_tile(gA_full, (self.tile_size_S, self.tile_K), (Int32(0), prefetch_idx))
                gB_block = cute.local_tile(gB_full, (self.tile_size_N, self.tile_K), (Int32(0), prefetch_idx))
                tAgA = thr_copy_A.partition_S(gA_block)
                tBgB = thr_copy_B.partition_S(gB_block)
                if k_idx % Int32(2) == Int32(0):
                    for ci in cutlass.range_constexpr(cute.size(tAsA_cp_0.shape[2])):
                        cute.copy(g2s_copy_A, tAgA[None, None, ci], tAsA_cp_0[None, None, ci])
                    for ci in cutlass.range_constexpr(cute.size(tBsB_cp_0.shape[2])):
                        cute.copy(g2s_copy_B, tBgB[None, None, ci], tBsB_cp_0[None, None, ci])
                if k_idx % Int32(2) == Int32(1):
                    for ci in cutlass.range_constexpr(cute.size(tAsA_cp_1.shape[2])):
                        cute.copy(g2s_copy_A, tAgA[None, None, ci], tAsA_cp_1[None, None, ci])
                    for ci in cutlass.range_constexpr(cute.size(tBsB_cp_1.shape[2])):
                        cute.copy(g2s_copy_B, tBgB[None, None, ci], tBsB_cp_1[None, None, ci])
                cute.arch.cp_async_commit_group()
            k_idx = k_idx + Int32(1)

        _gemm_epilogue_store_no_mbar_inval_helper(
            page_ptr, tidx, tiled_mma, acc,
            self.num_mma_threads, self.swz_B_c, self.c_dtype,
            self.tile_size_S, self.tile_size_N, self.activation,
        )

    def __init__(self, **config):
        super().__init__(**config)
        if self.has_a_scale:
            raise ValueError("Qwen3_5ForwardCpAsyncProjectionOp only supports has_a_scale=0")
        self.cpasync_copy_elems = 128 // (self.elem_bytes * 8)
        self.cpasync_copy_dim1 = self.tile_K // self.cpasync_copy_elems
        self.cpasync_copy_dim0 = self.num_mma_threads // self.cpasync_copy_dim1
        assert self.cpasync_copy_dim0 > 0, (
            f"Qwen3_5ForwardCpAsyncProjectionOp: not enough MMA threads "
            f"({self.num_mma_threads}) for tile_K={self.tile_K} cp.async"
        )

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        if "a_scale" not in tensors:
            a = tensors.get("a")
            if a is not None:
                tensors["a_scale"] = torch.empty_like(a)
        tile_sizes = dict(tile_sizes or {})
        a = tensors.get("a")
        b = tensors.get("b")
        if a is not None and b is not None:
            _complete_qwen_forward_gemm_tile_sizes(tile_sizes, a, b, page_size)
        ops = super().schedule(tile_sizes=tile_sizes, page_size=page_size, **tensors)
        for op in ops:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        return ops


class Qwen3_5ForwardPackedQKProjectionOp(Qwen3_5PackedQkvChunkProjectSm120Op):
    """Qwen forward packed Q/K projection with Q/K norm partial statistics."""

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("S", 128)
        tile_sizes.setdefault("N", 64)
        tile_sizes.setdefault("K", 32)
        ops = super().schedule(tile_sizes=tile_sizes, page_size=page_size, **tensors)
        for op in ops:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        return ops


class Qwen3_5ForwardPackedQKNormRopeOp(_BasePackedQKNormRopeOp):
    """Qwen forward packed Q/K norm+RoPE boundary op.

    Unlike the generic packed op, this local variant honors the captured
    ``qk`` strides. That lets the forward schedule pass Q/K as a last-dim slice
    of a larger QKV projection buffer without changing generic qknorm behavior.
    """

    @cute.jit
    def load(self, page_ptr, tile_M, tile_H, qk, q_norm_weight, k_norm_weight, cos, sin,
             op_config_ptr, work_mbar):
        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        pos_start = tile_M * self.tile_size_M
        head_start = tile_H * self.tile_size_H

        qk_per_pos_bytes = Int32(self.qk_row_elems * self.elem_bytes)
        cs_per_pos_bytes = Int32(2 * self.D2 * self.elem_bytes)
        actual_rows = Int32(self.tile_size_M)
        remaining = runtime_M - pos_start
        if remaining < Int32(self.tile_size_M):
            actual_rows = remaining
        total_bytes = actual_rows * (qk_per_pos_bytes + cs_per_pos_bytes)

        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        mbarrier_arrive_expect_tx(work_mbar, total_bytes)

        g2s_qk = cute.make_copy_atom(
            CopyBulkG2SOp(),
            self.qk_dtype,
            num_bits_per_copy=self.qk_nbits_per_row,
        )
        g2s_cs = cute.make_copy_atom(
            CopyBulkG2SOp(),
            self.qk_dtype,
            num_bits_per_copy=self.cs_nbits_per_row,
        )

        cos_smem_start = page_ptr + Int32(self.qk_tile_bytes)
        sin_smem_start = page_ptr + Int32(self.qk_tile_bytes + self.cs_tile_bytes)

        for local_pos in range(self.tile_size_M):
            pos = pos_start + local_pos
            if pos < runtime_M:
                s = pos % self.S
                g_qk = cute.make_tensor(
                    qk.iterator + pos * Int32(self.qk_stride_M) + head_start * Int32(self.qk_stride_H),
                    cute.make_layout((self.qk_row_elems,)),
                )
                s_qk = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        page_ptr + Int32(local_pos * self.qk_row_elems * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.qk_row_elems,)),
                )
                gsrc = cute.group_modes(g_qk, 0, 1)
                sdst = cute.group_modes(s_qk, 0, 1)
                cute.copy(g2s_qk, gsrc, sdst, mbar_ptr=mbar_ptr)

                g_cos = cute.make_tensor(cos.iterator + s * self.D2, cute.make_layout((self.D2,)))
                s_cos = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        cos_smem_start + Int32(local_pos * self.D2 * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.D2,)),
                )
                gc_src = cute.group_modes(g_cos, 0, 1)
                sc_dst = cute.group_modes(s_cos, 0, 1)
                cute.copy(g2s_cs, gc_src, sc_dst, mbar_ptr=mbar_ptr)

                g_sin = cute.make_tensor(sin.iterator + s * self.D2, cute.make_layout((self.D2,)))
                s_sin = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        sin_smem_start + Int32(local_pos * self.D2 * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.D2,)),
                )
                gs_src = cute.group_modes(g_sin, 0, 1)
                ss_dst = cute.group_modes(s_sin, 0, 1)
                cute.copy(g2s_cs, gs_src, ss_dst, mbar_ptr=mbar_ptr)

    @cute.jit
    def store(self, page_ptr, tile_M, tile_H, qk, q_norm_weight, k_norm_weight, cos, sin,
              op_config_ptr):
        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        s2g = cute.make_copy_atom(
            CopyBulkS2GOp(),
            self.qk_dtype,
            num_bits_per_copy=self.qk_nbits_per_row,
        )
        pos_start = tile_M * self.tile_size_M
        head_start = tile_H * self.tile_size_H

        for local_pos in range(self.tile_size_M):
            pos = pos_start + local_pos
            if pos < runtime_M:
                s_tile = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        page_ptr + Int32(local_pos * self.qk_row_elems * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.qk_row_elems,)),
                )
                g_tile = cute.make_tensor(
                    qk.iterator + pos * Int32(self.qk_stride_M) + head_start * Int32(self.qk_stride_H),
                    cute.make_layout((self.qk_row_elems,)),
                )
                ssrc = cute.group_modes(s_tile, 0, 1)
                gdst = cute.group_modes(g_tile, 0, 1)
                cute.copy(s2g, ssrc, gdst)


class Qwen3_5ForwardPackedQKNormRope4DOp(Qwen3_5ForwardPackedQKNormRopeOp):
    """Qwen-local packed Q/K norm+RoPE over a BMHD packed-QK view.

    This keeps the same shared-memory compute body as the 3D packed op, but
    exposes B/M/H tile axes to the dependency scheduler. That lets attention
    consume normalized row blocks instead of waiting for the whole flattened
    QK tensor.
    """

    reads = {
        "qk": (None, ("B", "M", "H", "D")),
        "q_norm_weight": (None, ("D",)),
        "k_norm_weight": (None, ("D",)),
        "cos": (None, ("S", "D2")),
        "sin": (None, ("S", "D2")),
    }
    writes = {"qk": (None, ("B", "M", "H", "D"))}
    tile = ("B", "M", "H")

    @classmethod
    def access_regions(cls, op) -> AccessRegions:
        regions = _rename_sequence_regions(super().access_regions(op))
        qk_write = regions.writes.get("qk")
        if qk_write is None:
            return regions
        writes = dict(regions.writes)
        writes["qk"] = _region_with_group(
            qk_write,
            group_dim="H",
            group_tiles=1,
            group_count=op.tile_counts[op.dim_names["H"]],
        )
        return AccessRegions(reads=regions.reads, writes=writes)

    @cute.jit
    def load(self, page_ptr, tile_B, tile_M, tile_H, qk, q_norm_weight, k_norm_weight, cos, sin,
             op_config_ptr, work_mbar):
        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        pos_start = tile_M * self.tile_size_M
        head_start = tile_H * self.tile_size_H

        qk_per_pos_bytes = Int32(self.qk_row_elems * self.elem_bytes)
        cs_per_pos_bytes = Int32(2 * self.D2 * self.elem_bytes)
        actual_rows = Int32(self.tile_size_M)
        remaining = runtime_M - pos_start
        if remaining < Int32(self.tile_size_M):
            actual_rows = remaining
        total_bytes = actual_rows * (qk_per_pos_bytes + cs_per_pos_bytes)

        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        mbarrier_arrive_expect_tx(work_mbar, total_bytes)

        g2s_qk = cute.make_copy_atom(
            CopyBulkG2SOp(),
            self.qk_dtype,
            num_bits_per_copy=self.qk_nbits_per_row,
        )
        g2s_cs = cute.make_copy_atom(
            CopyBulkG2SOp(),
            self.qk_dtype,
            num_bits_per_copy=self.cs_nbits_per_row,
        )

        cos_smem_start = page_ptr + Int32(self.qk_tile_bytes)
        sin_smem_start = page_ptr + Int32(self.qk_tile_bytes + self.cs_tile_bytes)

        for local_pos in range(self.tile_size_M):
            pos = pos_start + local_pos
            if pos < runtime_M:
                s = pos % self.S
                g_qk = cute.make_tensor(
                    qk.iterator
                    + tile_B * Int32(self.qk_stride_B)
                    + pos * Int32(self.qk_stride_M)
                    + head_start * Int32(self.qk_stride_H),
                    cute.make_layout((self.qk_row_elems,)),
                )
                s_qk = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        page_ptr + Int32(local_pos * self.qk_row_elems * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.qk_row_elems,)),
                )
                gsrc = cute.group_modes(g_qk, 0, 1)
                sdst = cute.group_modes(s_qk, 0, 1)
                cute.copy(g2s_qk, gsrc, sdst, mbar_ptr=mbar_ptr)

                g_cos = cute.make_tensor(cos.iterator + s * self.D2, cute.make_layout((self.D2,)))
                s_cos = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        cos_smem_start + Int32(local_pos * self.D2 * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.D2,)),
                )
                gc_src = cute.group_modes(g_cos, 0, 1)
                sc_dst = cute.group_modes(s_cos, 0, 1)
                cute.copy(g2s_cs, gc_src, sc_dst, mbar_ptr=mbar_ptr)

                g_sin = cute.make_tensor(sin.iterator + s * self.D2, cute.make_layout((self.D2,)))
                s_sin = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        sin_smem_start + Int32(local_pos * self.D2 * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.D2,)),
                )
                gs_src = cute.group_modes(g_sin, 0, 1)
                ss_dst = cute.group_modes(s_sin, 0, 1)
                cute.copy(g2s_cs, gs_src, ss_dst, mbar_ptr=mbar_ptr)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_M, tile_H, op_config_ptr):
        _BasePackedQKNormRopeOp.compute(self, page_ptr, tile_M, tile_H, op_config_ptr)

    @cute.jit
    def store(self, page_ptr, tile_B, tile_M, tile_H, qk, q_norm_weight, k_norm_weight, cos, sin,
              op_config_ptr):
        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        s2g = cute.make_copy_atom(
            CopyBulkS2GOp(),
            self.qk_dtype,
            num_bits_per_copy=self.qk_nbits_per_row,
        )
        pos_start = tile_M * self.tile_size_M
        head_start = tile_H * self.tile_size_H

        for local_pos in range(self.tile_size_M):
            pos = pos_start + local_pos
            if pos < runtime_M:
                s_tile = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        page_ptr + Int32(local_pos * self.qk_row_elems * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.qk_row_elems,)),
                )
                g_tile = cute.make_tensor(
                    qk.iterator
                    + tile_B * Int32(self.qk_stride_B)
                    + pos * Int32(self.qk_stride_M)
                    + head_start * Int32(self.qk_stride_H),
                    cute.make_layout((self.qk_row_elems,)),
                )
                ssrc = cute.group_modes(s_tile, 0, 1)
                gdst = cute.group_modes(g_tile, 0, 1)
                cute.copy(s2g, ssrc, gdst)

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        eps=1e-6,
        num_q_heads=None,
        num_k_heads=None,
        **tensors,
    ):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("M", 8)
        tile_sizes.setdefault("H", 5)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["eps"] = eps
        if num_q_heads is not None:
            ops[0].static_dims["num_q_heads"] = int(num_q_heads)
        if num_k_heads is not None:
            ops[0].static_dims["num_k_heads"] = int(num_k_heads)
        return ops


class Qwen3_5ForwardQKNormRopeOp(_BaseQKNormRopeOp):
    """Qwen forward single Q/K norm+RoPE boundary op."""


class _Qwen3_5ForwardAttentionBase(_BaseFlashAttentionSm120Op):
    """Shared Qwen forward full-attention scheduling base.

    The plain non-TMA FlashAttention path is intentionally not exposed for
    Qwen 3.5 forward: HEAD_DIM is 256 and the generic op only supports D <= 64
    in this schedule. Use one of the TMA attention variants instead.
    """

    # Q is the only DMA-warp TMA load. K/V are loaded by the attention MMA
    # warps through cp.async inside compute, so their producer dependencies can
    # be delayed until compute instead of blocking Q prefetch in the controller.
    compute_wait_inputs = {"k", "v"}


class Qwen3_5ForwardTmaAttentionOp(_Qwen3_5ForwardAttentionBase):
    """Qwen forward attention with compute-issued TMA K/V loads.

    This variant keeps Q as the framework load-phase TMA, then preloads Q into
    registers and reuses the page for compact K/V buffers driven by TMA from
    compute warp 0. It targets the Qwen forward q-preload shape used at S=512.
    """

    tma_compute_loads = {"k", "v"}
    max_supported_D = HEAD_DIM
    d_block = 64

    @classmethod
    def access_regions(cls, op) -> AccessRegions:
        return _qwen_attention_regions(op, super().access_regions(op))

    @classmethod
    def schedule(cls, tile_sizes=None, **kwargs):
        ops = super().schedule(tile_sizes=tile_sizes, **kwargs)
        q = kwargs.get("q")
        k = kwargs.get("k")
        page_size = int(kwargs.get("page_size", DEFAULT_PAGE_SIZE))
        for op in ops:
            tile_m = int((tile_sizes or {}).get("M", op.tile_sizes.get("M", 16)))
            op.static_dims.setdefault("num_mma_warps", max(1, tile_m // 16))
            if q is None:
                op.static_dims["tma_n_block"] = 16
                op.static_dims["tma_copy_d_block"] = HEAD_DIM
                continue
            D = int(q.shape[-1])
            elem_bytes = int(q.element_size())
            k_d = min(D, int(getattr(cls, "d_block", 64))) if D > 64 else D
            copy_d = D
            if D > 64 and tile_m * D * elem_bytes >= page_size:
                raise ValueError(
                    f"{cls.__name__} D={D} tile_M={tile_m} needs page_size >= "
                    f"{tile_m * D * elem_bytes + 1} bytes for a valid single-step S2G TMA O tile; "
                    "use tile_M<=32 with 32KB pages, or use one-page attention with "
                    "page_size>=64KB/96KB for tile_M=64."
                )
            scratch_q_bytes = tile_m * k_d * elem_bytes if D > 64 else 0
            capacity = max(
                16,
                (page_size - 256 - scratch_q_bytes) // ((2 * copy_d + (k_d if D > 64 else 0)) * elem_bytes),
            )
            n_block = max(16, min(64, (capacity // 16) * 16))
            op.static_dims["tma_n_block"] = n_block
            op.static_dims["tma_d_block"] = k_d
            op.static_dims["tma_copy_d_block"] = copy_d
            op.static_dims["tma_even_n_block"] = int(int(k.shape[1]) % n_block == 0) if kwargs.get("k") is not None else 0
        return ops

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name in ("k", "v"):
            d_block = int(static_dims.get("tma_load_d_block", static_dims.get("tma_copy_d_block", static_dims["D"])))
            if int(static_dims.get("tma_flash_kv_layout", 0)):
                # Flash-style K/V TMA is a 2D copy tile (S, D) from a 4D
                # BSHD backing tensor viewed as (S, D, H, B).  H/B are sliced
                # by the op before local_tile(), matching FlashAttention's
                # tma_partition contract and avoiding a rejected 4D basis with
                # unit H/B modes.
                return (static_dims["tma_n_block"], d_block)
            return (1, static_dims["tma_n_block"], 1, d_block)
        return None

    @classmethod
    def get_tma_dim_permutation(cls, tensor_name, tile_sizes, static_dims, tensor_shape, tensor_stride):
        if tensor_name in ("k", "v") and int(static_dims.get("tma_flash_kv_layout", 0)):
            # Public K/V tensors are BSHD.  Flash-style TMA wants the first two
            # descriptor modes to be the copied tile (S, D), with H and B as
            # outer coordinates.  This keeps the op-facing shape BSHD while the
            # descriptor view matches the shared-memory atom.
            return (1, 3, 2, 0)
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name == "o" and int(static_dims["D"]) > 64:
            tile_d, one_h, tile_m, one_b = tma_tile_shape
            return (
                f"cute.make_layout(({tile_d}, {one_h}, {tile_m}, {one_b}), "
                f"stride=(1, {tile_d}, {tile_d}, {tile_d * tile_m}))"
            )
        if tensor_name in ("k", "v"):
            if int(static_dims.get("tma_flash_kv_layout", 0)):
                # 2D TMA basis (S, D) into a Flash-style physical shared
                # memory atom.  The tensor view still has rank 4 internally;
                # only the copy tile is 2D.
                tile_n, tile_d = tma_tile_shape
                swizzle_b = 3 if int(tile_d) >= 64 else 2 if int(tile_d) >= 32 else 1
                return (
                    f"cute.slice_(cute.tile_to_shape("
                    f"cute.make_composed_layout("
                    f"cute.make_swizzle({swizzle_b}, 4, 3), 0, "
                    f"cute.make_layout((8, {tile_d}), stride=({tile_d}, 1))), "
                    f"({tile_n}, {tile_d}, 2), (0, 1, 2)), "
                    f"(None, None, 0))"
                )
            # tma_tile_shape is stride-sorted to (D, H_kv=1, N, B=1).
            tile_d, one_h, tile_n, one_b = tma_tile_shape
            if int(static_dims.get("tma_chunked_kv_swizzle", 0)):
                swizzle_b = 3 if int(tile_d) >= 64 else 2 if int(tile_d) >= 32 else 1
                return (
                    f"cute.make_composed_layout("
                    f"cute.make_swizzle({swizzle_b}, 4, 3), 0, "
                    f"cute.make_layout(({tile_d}, {one_h}, {tile_n}, {one_b}), "
                    f"stride=(1, {tile_d * tile_n}, {tile_d}, {tile_d * tile_n})))"
                )
            return (
                f"cute.make_layout(({tile_d}, {one_h}, {tile_n}, {one_b}), "
                f"stride=(1, {tile_d * tile_n}, {tile_d}, {tile_d * tile_n}))"
            )
        return super().get_tma_smem_layout_src(tensor_name, tma_tile_shape, tile_sizes, static_dims)

    def __init__(self, **config):
        super().__init__(**config)
        if self.q_in_smem:
            raise ValueError(f"{type(self).__name__} currently supports q-preload mode only")
        self.n_block = int(getattr(self, "tma_n_block", 16))
        self.tma_d_block = int(getattr(self, "tma_d_block", min(self.D, type(self).d_block)))
        self.tma_copy_d_block = int(getattr(self, "tma_copy_d_block", self.D))
        self.tma_load_d_block = int(getattr(self, "tma_load_d_block", self.tma_copy_d_block))
        self.tma_chunked_kv_swizzle = bool(getattr(self, "tma_chunked_kv_swizzle", 0))
        self.tma_flash_kv_layout = bool(getattr(self, "tma_flash_kv_layout", 0))
        if self.D > type(self).max_supported_D or self.D % type(self).d_block != 0:
            raise ValueError(
                f"{type(self).__name__} supports D multiples of {type(self).d_block} "
                f"up to {type(self).max_supported_D}, got D={self.D}"
            )
        if self.tma_chunked_kv_swizzle:
            if self.tma_load_d_block % self.tma_d_block != 0:
                raise ValueError(
                    f"{type(self).__name__}: tma_load_d_block={self.tma_load_d_block} "
                    f"must be a multiple of tma_d_block={self.tma_d_block}"
                )
            if self.D % self.tma_load_d_block != 0:
                raise ValueError(
                    f"{type(self).__name__}: D={self.D} must be a multiple of "
                    f"tma_load_d_block={self.tma_load_d_block}"
                )
        if self.tma_flash_kv_layout and self.tma_load_d_block != self.tma_d_block:
            raise ValueError(
                f"{type(self).__name__}: Flash K/V smem layout requires "
                f"tma_load_d_block == tma_d_block, got "
                f"{self.tma_load_d_block} and {self.tma_d_block}"
            )
        self.k_smem_stride = self.tma_load_d_block if self.tma_chunked_kv_swizzle else self.tma_copy_d_block
        self.v_smem_stride = self.tma_load_d_block if self.tma_chunked_kv_swizzle else self.tma_copy_d_block
        self.kv_load_chunk_byte_stride = (
            self.n_block * self.tma_load_d_block * self.elem_bytes
            if self.tma_chunked_kv_swizzle
            else self.tma_d_block * self.elem_bytes
        )
        self.num_kv_tma_load_chunks = (
            (self.D + self.tma_load_d_block - 1) // self.tma_load_d_block
            if self.tma_chunked_kv_swizzle
            else 1
        )
        self.kv_compute_chunks_per_tma_load = (
            self.tma_load_d_block // self.tma_d_block
            if self.tma_chunked_kv_swizzle
            else 1
        )
        self.kv_compute_chunk_byte_stride = (
            self.n_block * self.tma_d_block * self.elem_bytes
            if self.tma_chunked_kv_swizzle
            else self.tma_d_block * self.elem_bytes
        )
        self.kv_compute_chunk1_byte_offset = self._kv_compute_chunk_byte_offset(1)
        self.kv_compute_chunk2_byte_offset = self._kv_compute_chunk_byte_offset(2)
        self.kv_compute_chunk3_byte_offset = self._kv_compute_chunk_byte_offset(3)
        self.k_tile_bytes = self.n_block * self.tma_copy_d_block * self.elem_bytes
        self.v_tile_bytes = self.n_block * self.tma_copy_d_block * self.elem_bytes
        self.kv_tile_bytes = self.k_tile_bytes + self.v_tile_bytes
        self.tma_reverse_kv = bool(
            getattr(
                self,
                "tma_reverse_kv",
                int(bool(self.causal) and self.tile_size_M <= self.n_block),
            )
        )
        self.tma_separate_kv_pages = bool(getattr(self, "tma_separate_kv_pages", 0))
        if self.tma_separate_kv_pages:
            self.tma_k_base = 0
            self.tma_v_base = 0
            self.tma_mbar_offset = self.k_tile_bytes if bool(getattr(self, "tma_scratch_after_k", 0)) else 0
            assert self.k_tile_bytes <= self.page_size, (
                f"{type(self).__name__}: K TMA tile exceeds page_size"
            )
            assert self.v_tile_bytes <= self.page_size, (
                f"{type(self).__name__}: V TMA tile exceeds page_size"
            )
        else:
            self.tma_k_base = 0
            self.tma_v_base = self.k_tile_bytes
            self.tma_mbar_offset = self.kv_tile_bytes
        self.q_chunk_tile_bytes = self.tile_size_M * self.tma_d_block * self.elem_bytes if self.D > 64 else 0
        self.k_chunk_tile_bytes = self.n_block * self.tma_d_block * self.elem_bytes if self.D > 64 else 0
        self.q_chunk_base = ((self.tma_mbar_offset + 16 + 127) // 128) * 128
        self.k_chunk_base = ((self.q_chunk_base + self.q_chunk_tile_bytes + 127) // 128) * 128
        self.scratch_end = self.k_chunk_base + self.k_chunk_tile_bytes if self.D > 64 else self.tma_mbar_offset + 16
        assert self.scratch_end <= self.page_size, (
            f"{type(self).__name__}: compact K/V buffers plus mbarriers exceed page_size"
        )
        self.rescale_threshold = getattr(self, "rescale_threshold", 8.0)
        self._bind_phase("compute", "compute_mma_tma")

    @cute.jit
    def _flash_kv_layout(self):
        return cute.tile_to_shape(
            cute.make_composed_layout(
                cute.make_swizzle(self.swizzle_B, 4, 3),
                0,
                cute.make_layout((8, self.tma_d_block), stride=(self.tma_d_block, 1)),
            ),
            (self.n_block, self.tma_d_block),
            (0, 1),
        )

    @cute.jit
    def _flash_kv_tma_layout(self):
        return cute.composition(
            self._flash_kv_layout(),
            cute.make_layout(
                (self.n_block, self.tma_load_d_block, 1, 1),
                stride=(
                    self.tma_load_d_block,
                    1,
                    self.tma_load_d_block * self.n_block,
                    self.tma_load_d_block * self.n_block,
                ),
            ),
        )

    @cute.jit
    def _flash_vt_layout(self):
        return cute.composition(
            self._flash_kv_layout(),
            cute.make_layout(
                (self.tma_d_block, self.n_block),
                stride=(self.n_block, 1),
            ),
        )

    def _kv_compute_chunk_byte_offset(self, chunk_idx: int) -> int:
        if not self.tma_chunked_kv_swizzle:
            return chunk_idx * self.tma_d_block * self.elem_bytes
        tma_chunk_idx = chunk_idx // self.kv_compute_chunks_per_tma_load
        in_tma_chunk_idx = chunk_idx % self.kv_compute_chunks_per_tma_load
        return (
            tma_chunk_idx * self.kv_load_chunk_byte_stride
            + in_tma_chunk_idx * self.tma_d_block * self.elem_bytes
        )

    @cute.jit
    def _q_page(self, page_ptr):
        return page_ptr

    @cute.jit
    def _kv_page(self, page_ptr):
        return page_ptr

    @cute.jit
    def _k_page(self, page_ptr):
        return self._kv_page(page_ptr)

    @cute.jit
    def _v_page(self, page_ptr):
        return self._kv_page(page_ptr)

    @cute.jit
    def _scratch_page(self, page_ptr):
        return self._kv_page(page_ptr)

    @cute.jit
    def _o_page(self, page_ptr):
        return self._kv_page(page_ptr)

    @cute.jit
    def compute_mma_tma(self, page_ptr, tile_B, tile_M, tile_H, tile_D,
                        q, k, v, o, lse,
                        k_tma, k_tma_gmem, v_tma, v_tma_gmem,
                        op_config_ptr):
        self._compute_mma_tma_impl(
            page_ptr,
            tile_B,
            tile_M,
            tile_H,
            tile_D,
            q,
            k,
            v,
            o,
            lse,
            k_tma,
            k_tma_gmem,
            v_tma,
            v_tma_gmem,
            op_config_ptr,
            False,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
        )

    @cute.jit
    def _compute_mma_tma_impl(
        self,
        page_ptr,
        tile_B,
        tile_M,
        tile_H,
        tile_D,
        q,
        k,
        v,
        o,
        lse,
        k_tma,
        k_tma_gmem,
        v_tma,
        v_tma_gmem,
        op_config_ptr,
        release_q_page,
        page_release_table_ptr,
        page_release_page_base,
        page_release_mbar_base,
        page_release_mbar_stride,
        page_release_page_size,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        runtime_N = config_dim_i32(op_config_ptr, "N", type(self))
        runtime_num_kv_blocks = (runtime_N + Int32(self.n_block - 1)) // Int32(self.n_block)

        q_page = self._q_page(page_ptr)
        k_page = self._k_page(page_ptr)
        v_page = self._v_page(page_ptr)
        scratch_page = self._scratch_page(page_ptr)
        o_page = self._o_page(page_ptr)
        k_base = k_page + Int32(self.tma_k_base)
        v_base = v_page + Int32(self.tma_v_base)
        k_ready = scratch_page + Int32(self.tma_mbar_offset)
        v_ready = scratch_page + Int32(self.tma_mbar_offset + 8)
        k_ready_ptr = cute.make_ptr(cutlass.Int64, k_ready, cute.AddressSpace.smem)
        v_ready_ptr = cute.make_ptr(cutlass.Int64, v_ready, cute.AddressSpace.smem)
        k_bytes = Int32(self.n_block * self.tma_copy_d_block * self.elem_bytes)
        v_bytes = Int32(self.n_block * self.tma_copy_d_block * self.elem_bytes)
        kv_swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)

        # TMA atoms/partitions must be materialized outside dynamic warp
        # control flow. Creating them inside the MMA-warp branch can lower to
        # invalid NVGPU code on the D-blocked attention path.
        if cutlass.const_expr(self.tma_flash_kv_layout):
            k_tma_ptr = cute.make_ptr(self.q_dtype, k_base, cute.AddressSpace.smem, assumed_align=128)
            v_tma_ptr = cute.make_ptr(self.q_dtype, v_base, cute.AddressSpace.smem, assumed_align=128)
        elif cutlass.const_expr(self.tma_chunked_kv_swizzle):
            k_tma_ptr = cute.recast_ptr(
                cute.make_ptr(self.q_dtype, k_base, cute.AddressSpace.smem, assumed_align=128),
                kv_swz,
                dtype=self.q_dtype,
            )
            v_tma_ptr = cute.recast_ptr(
                cute.make_ptr(self.q_dtype, v_base, cute.AddressSpace.smem, assumed_align=128),
                kv_swz,
                dtype=self.q_dtype,
            )
        else:
            k_tma_ptr = cute.make_ptr(self.q_dtype, k_base, cute.AddressSpace.smem, assumed_align=128)
            v_tma_ptr = cute.make_ptr(self.q_dtype, v_base, cute.AddressSpace.smem, assumed_align=128)
        if cutlass.const_expr(self.tma_flash_kv_layout):
            tma_k_layout = cute.tile_to_shape(
                cute.make_composed_layout(
                    cute.make_swizzle(self.swizzle_B, 4, 3),
                    0,
                    cute.make_layout((8, self.tma_load_d_block), stride=(self.tma_load_d_block, 1)),
                ),
                (self.n_block, self.tma_load_d_block, 2),
                (0, 1, 2),
            )
            tma_v_layout = tma_k_layout
        else:
            tma_k_layout = cute.make_layout(
                (self.tma_load_d_block, 1, self.n_block, 1),
                stride=(
                    1,
                    self.tma_load_d_block * self.n_block,
                    self.tma_load_d_block,
                    self.tma_load_d_block * self.n_block,
                ),
            )
            tma_v_layout = tma_k_layout
        if cutlass.const_expr(self.tma_flash_kv_layout):
            sK_tma = cute.make_tensor(
                cute.recast_ptr(k_tma_ptr, tma_k_layout.inner, dtype=self.q_dtype),
                tma_k_layout.outer,
            )
        else:
            sK_tma = cute.make_tensor(
                k_tma_ptr,
                tma_k_layout,
            )
        kv_h = self._kv_head_index(tile_H)
        if cutlass.const_expr(self.tma_flash_kv_layout):
            gK_tma = cute.local_tile(
                k_tma_gmem[None, None, kv_h, tile_B],
                (self.n_block, self.tma_load_d_block),
                (None, Int32(0)),
            )
        else:
            gK_tma = cute.local_tile(
                k_tma_gmem,
                (self.tma_load_d_block, 1, self.n_block, 1),
                (None, None, None, None),
            )
        if cutlass.const_expr(self.tma_flash_kv_layout):
            tKsK_tma, tKgK_tma = cute.nvgpu.cpasync.tma_partition(
                k_tma,
                0,
                cute.make_layout(1),
                cute.group_modes(sK_tma, 0, 2),
                cute.group_modes(gK_tma, 0, 2),
            )
        else:
            tKsK_tma, tKgK_tma = cute.nvgpu.cpasync.tma_partition(
                k_tma,
                0,
                cute.make_layout(1),
                cute.group_modes(sK_tma, 0, 4),
                cute.group_modes(gK_tma, 0, 4),
            )
        if cutlass.const_expr(self.tma_flash_kv_layout):
            sV_tma = cute.make_tensor(
                cute.recast_ptr(v_tma_ptr, tma_v_layout.inner, dtype=self.q_dtype),
                tma_v_layout.outer,
            )
        else:
            sV_tma = cute.make_tensor(
                v_tma_ptr,
                tma_v_layout,
            )
        if cutlass.const_expr(self.tma_flash_kv_layout):
            gV_tma = cute.local_tile(
                v_tma_gmem[None, None, kv_h, tile_B],
                (self.n_block, self.tma_load_d_block),
                (None, Int32(0)),
            )
        else:
            gV_tma = cute.local_tile(
                v_tma_gmem,
                (self.tma_load_d_block, 1, self.n_block, 1),
                (None, None, None, None),
            )
        if cutlass.const_expr(self.tma_flash_kv_layout):
            tVsV_tma, tVgV_tma = cute.nvgpu.cpasync.tma_partition(
                v_tma,
                0,
                cute.make_layout(1),
                cute.group_modes(sV_tma, 0, 2),
                cute.group_modes(gV_tma, 0, 2),
            )
        else:
            tVsV_tma, tVgV_tma = cute.nvgpu.cpasync.tma_partition(
                v_tma,
                0,
                cute.make_layout(1),
                cute.group_modes(sV_tma, 0, 4),
                cute.group_modes(gV_tma, 0, 4),
            )
        if cutlass.const_expr(self.tma_chunked_kv_swizzle and self.num_kv_tma_load_chunks > 1):
            if cutlass.const_expr(self.tma_flash_kv_layout):
                sK_tma1_ptr = cute.make_ptr(
                        self.q_dtype,
                        k_base + Int32(1 * self.kv_load_chunk_byte_stride),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                sK_tma1 = cute.make_tensor(
                    cute.recast_ptr(sK_tma1_ptr, tma_k_layout.inner, dtype=self.q_dtype),
                    tma_k_layout.outer,
                )
                gK_tma1 = cute.local_tile(
                    k_tma_gmem[None, None, kv_h, tile_B],
                    (self.n_block, self.tma_load_d_block),
                    (None, Int32(1)),
                )
            else:
                sK_tma1 = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            k_base + Int32(1 * self.kv_load_chunk_byte_stride),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        kv_swz,
                        dtype=self.q_dtype,
                    ),
                    tma_k_layout,
                )
                gK_tma1 = gK_tma
            if cutlass.const_expr(self.tma_flash_kv_layout):
                tKsK_tma1, _tKgK_tma1 = cute.nvgpu.cpasync.tma_partition(
                    k_tma,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK_tma1, 0, 2),
                    cute.group_modes(gK_tma1, 0, 2),
                )
            else:
                tKsK_tma1, _tKgK_tma1 = cute.nvgpu.cpasync.tma_partition(
                    k_tma,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK_tma1, 0, 4),
                    cute.group_modes(gK_tma1, 0, 4),
                )
            if cutlass.const_expr(self.tma_flash_kv_layout):
                sV_tma1_ptr = cute.make_ptr(
                        self.q_dtype,
                        v_base + Int32(1 * self.kv_load_chunk_byte_stride),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                sV_tma1 = cute.make_tensor(
                    cute.recast_ptr(sV_tma1_ptr, tma_v_layout.inner, dtype=self.q_dtype),
                    tma_v_layout.outer,
                )
                gV_tma1 = cute.local_tile(
                    v_tma_gmem[None, None, kv_h, tile_B],
                    (self.n_block, self.tma_load_d_block),
                    (None, Int32(1)),
                )
            else:
                sV_tma1 = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            v_base + Int32(1 * self.kv_load_chunk_byte_stride),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        kv_swz,
                        dtype=self.q_dtype,
                    ),
                    tma_v_layout,
                )
                gV_tma1 = gV_tma
            if cutlass.const_expr(self.tma_flash_kv_layout):
                tVsV_tma1, _tVgV_tma1 = cute.nvgpu.cpasync.tma_partition(
                    v_tma,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sV_tma1, 0, 2),
                    cute.group_modes(gV_tma1, 0, 2),
                )
            else:
                tVsV_tma1, _tVgV_tma1 = cute.nvgpu.cpasync.tma_partition(
                    v_tma,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sV_tma1, 0, 4),
                    cute.group_modes(gV_tma1, 0, 4),
                )
        if cutlass.const_expr(self.tma_chunked_kv_swizzle and self.num_kv_tma_load_chunks > 2):
            if cutlass.const_expr(self.tma_flash_kv_layout):
                sK_tma2_ptr = cute.make_ptr(
                        self.q_dtype,
                        k_base + Int32(2 * self.kv_load_chunk_byte_stride),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                sK_tma2 = cute.make_tensor(
                    cute.recast_ptr(sK_tma2_ptr, tma_k_layout.inner, dtype=self.q_dtype),
                    tma_k_layout.outer,
                )
                gK_tma2 = cute.local_tile(
                    k_tma_gmem[None, None, kv_h, tile_B],
                    (self.n_block, self.tma_load_d_block),
                    (None, Int32(2)),
                )
            else:
                sK_tma2 = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            k_base + Int32(2 * self.kv_load_chunk_byte_stride),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        kv_swz,
                        dtype=self.q_dtype,
                    ),
                    tma_k_layout,
                )
                gK_tma2 = gK_tma
            if cutlass.const_expr(self.tma_flash_kv_layout):
                tKsK_tma2, _tKgK_tma2 = cute.nvgpu.cpasync.tma_partition(
                    k_tma,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK_tma2, 0, 2),
                    cute.group_modes(gK_tma2, 0, 2),
                )
            else:
                tKsK_tma2, _tKgK_tma2 = cute.nvgpu.cpasync.tma_partition(
                    k_tma,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK_tma2, 0, 4),
                    cute.group_modes(gK_tma2, 0, 4),
                )
            if cutlass.const_expr(self.tma_flash_kv_layout):
                sV_tma2_ptr = cute.make_ptr(
                        self.q_dtype,
                        v_base + Int32(2 * self.kv_load_chunk_byte_stride),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                sV_tma2 = cute.make_tensor(
                    cute.recast_ptr(sV_tma2_ptr, tma_v_layout.inner, dtype=self.q_dtype),
                    tma_v_layout.outer,
                )
                gV_tma2 = cute.local_tile(
                    v_tma_gmem[None, None, kv_h, tile_B],
                    (self.n_block, self.tma_load_d_block),
                    (None, Int32(2)),
                )
            else:
                sV_tma2 = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            v_base + Int32(2 * self.kv_load_chunk_byte_stride),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        kv_swz,
                        dtype=self.q_dtype,
                    ),
                    tma_v_layout,
                )
                gV_tma2 = gV_tma
            if cutlass.const_expr(self.tma_flash_kv_layout):
                tVsV_tma2, _tVgV_tma2 = cute.nvgpu.cpasync.tma_partition(
                    v_tma,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sV_tma2, 0, 2),
                    cute.group_modes(gV_tma2, 0, 2),
                )
            else:
                tVsV_tma2, _tVgV_tma2 = cute.nvgpu.cpasync.tma_partition(
                    v_tma,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sV_tma2, 0, 4),
                    cute.group_modes(gV_tma2, 0, 4),
                )
        if cutlass.const_expr(self.tma_chunked_kv_swizzle and self.num_kv_tma_load_chunks > 3):
            if cutlass.const_expr(self.tma_flash_kv_layout):
                sK_tma3_ptr = cute.make_ptr(
                        self.q_dtype,
                        k_base + Int32(3 * self.kv_load_chunk_byte_stride),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                sK_tma3 = cute.make_tensor(
                    cute.recast_ptr(sK_tma3_ptr, tma_k_layout.inner, dtype=self.q_dtype),
                    tma_k_layout.outer,
                )
                gK_tma3 = cute.local_tile(
                    k_tma_gmem[None, None, kv_h, tile_B],
                    (self.n_block, self.tma_load_d_block),
                    (None, Int32(3)),
                )
            else:
                sK_tma3 = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            k_base + Int32(3 * self.kv_load_chunk_byte_stride),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        kv_swz,
                        dtype=self.q_dtype,
                    ),
                    tma_k_layout,
                )
                gK_tma3 = gK_tma
            if cutlass.const_expr(self.tma_flash_kv_layout):
                tKsK_tma3, _tKgK_tma3 = cute.nvgpu.cpasync.tma_partition(
                    k_tma,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK_tma3, 0, 2),
                    cute.group_modes(gK_tma3, 0, 2),
                )
            else:
                tKsK_tma3, _tKgK_tma3 = cute.nvgpu.cpasync.tma_partition(
                    k_tma,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK_tma3, 0, 4),
                    cute.group_modes(gK_tma3, 0, 4),
                )
            if cutlass.const_expr(self.tma_flash_kv_layout):
                sV_tma3_ptr = cute.make_ptr(
                        self.q_dtype,
                        v_base + Int32(3 * self.kv_load_chunk_byte_stride),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                sV_tma3 = cute.make_tensor(
                    cute.recast_ptr(sV_tma3_ptr, tma_v_layout.inner, dtype=self.q_dtype),
                    tma_v_layout.outer,
                )
                gV_tma3 = cute.local_tile(
                    v_tma_gmem[None, None, kv_h, tile_B],
                    (self.n_block, self.tma_load_d_block),
                    (None, Int32(3)),
                )
            else:
                sV_tma3 = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            v_base + Int32(3 * self.kv_load_chunk_byte_stride),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        kv_swz,
                        dtype=self.q_dtype,
                    ),
                    tma_v_layout,
                )
                gV_tma3 = gV_tma
            if cutlass.const_expr(self.tma_flash_kv_layout):
                tVsV_tma3, _tVgV_tma3 = cute.nvgpu.cpasync.tma_partition(
                    v_tma,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sV_tma3, 0, 2),
                    cute.group_modes(gV_tma3, 0, 2),
                )
            else:
                tVsV_tma3, _tVgV_tma3 = cute.nvgpu.cpasync.tma_partition(
                    v_tma,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sV_tma3, 0, 4),
                    cute.group_modes(gV_tma3, 0, 4),
                )

        if warp_idx < Int32(self.num_mma_warps):
            mma_op = cute.nvgpu.warp.MmaF16BF16Op(self.q_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)
            sQ = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, q_page, cute.AddressSpace.smem, assumed_align=128),
                    swz,
                    dtype=self.q_dtype,
                ),
                cute.make_layout((self.tile_size_M, self.D), stride=(self.D, 1)),
            )
            _tCsQ = thr_mma.partition_A(sQ)
            tCrQ = tiled_mma.make_fragment_A(_tCsQ)
            smem_copy_atom_Q = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.q_dtype
            )
            smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
            smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
            tQrQ_view = smem_thr_copy_Q.retile(tCrQ)
            tQsQ = smem_thr_copy_Q.partition_S(sQ)
            self._compute_load_q(q_page, tile_B, tile_H, tile_M, tile_D, q, runtime_M)

            for _qkb in cutlass.range_constexpr(self.D // 16):
                cute.copy(smem_tiled_copy_Q, tQsQ[None, None, _qkb], tQrQ_view[None, None, _qkb])
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
            if cutlass.const_expr(release_q_page):
                self.release_compute_page_relaxed(
                    0,
                    page_release_table_ptr,
                    page_release_page_base,
                    page_release_mbar_base,
                    page_release_mbar_stride,
                    page_release_page_size,
                )

            if warp_idx == Int32(0):
                with cute.arch.elect_one():
                    mbarrier_init(k_ready, Int32(1))
                    mbarrier_init(v_ready, Int32(1))
                mbarrier_init_fence_async_proxy()
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            smem_copy_atom_K = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.q_dtype
            )
            smem_copy_atom_Vt = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.q_dtype
            )
            smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
            smem_tiled_copy_Vt = cute.make_tiled_copy_B(smem_copy_atom_Vt, tiled_mma)
            smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
            smem_thr_copy_Vt = smem_tiled_copy_Vt.get_slice(tidx)

            if cutlass.const_expr(self.tma_flash_kv_layout):
                k_compute_layout_full = self._flash_kv_layout()
                vt_compute_layout_full = self._flash_vt_layout()
                k_smem_ptr = cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, k_base, cute.AddressSpace.smem, assumed_align=128),
                    k_compute_layout_full.inner,
                    dtype=self.q_dtype,
                )
                v_smem_ptr = cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, v_base, cute.AddressSpace.smem, assumed_align=128),
                    vt_compute_layout_full.inner,
                    dtype=self.q_dtype,
                )
                k_compute_layout = k_compute_layout_full.outer
                vt_compute_layout = vt_compute_layout_full.outer
            elif cutlass.const_expr(self.tma_chunked_kv_swizzle):
                k_smem_ptr = cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, k_base, cute.AddressSpace.smem, assumed_align=128),
                    kv_swz,
                    dtype=self.q_dtype,
                )
                v_smem_ptr = cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, v_base, cute.AddressSpace.smem, assumed_align=128),
                    kv_swz,
                    dtype=self.q_dtype,
                )
                k_compute_layout = cute.make_layout((self.n_block, self.tma_d_block), stride=(self.k_smem_stride, 1))
                vt_compute_layout = cute.make_layout((self.tma_d_block, self.n_block), stride=(1, self.v_smem_stride))
            else:
                k_smem_ptr = cute.make_ptr(self.q_dtype, k_base, cute.AddressSpace.smem, assumed_align=128)
                v_smem_ptr = cute.make_ptr(self.q_dtype, v_base, cute.AddressSpace.smem, assumed_align=128)
                k_compute_layout = cute.make_layout((self.n_block, self.tma_d_block), stride=(self.k_smem_stride, 1))
                vt_compute_layout = cute.make_layout((self.tma_d_block, self.n_block), stride=(1, self.v_smem_stride))
            sK = cute.make_tensor(
                k_smem_ptr,
                k_compute_layout,
            )
            _tCsK = thr_mma.partition_B(sK)
            tCrK = tiled_mma.make_fragment_B(_tCsK)
            tKrK_view = smem_thr_copy_K.retile(tCrK)
            tKsK = smem_thr_copy_K.partition_S(sK)
            sVt = cute.make_tensor(
                v_smem_ptr,
                vt_compute_layout,
            )
            _tBsVt = thr_mma.partition_B(sVt)
            tBrVt = tiled_mma.make_fragment_B(_tBsVt)
            tVrVt_view = smem_thr_copy_Vt.retile(tBrVt)
            tVsVt = smem_thr_copy_Vt.partition_S(sVt)
            if cutlass.const_expr(self.D > 64):
                if cutlass.const_expr(self.tma_flash_kv_layout):
                    k_smem_ptr1 = cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            k_base + Int32(self.kv_compute_chunk1_byte_offset),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        k_compute_layout_full.inner,
                        dtype=self.q_dtype,
                    )
                    v_smem_ptr1 = cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            v_base + Int32(self.kv_compute_chunk1_byte_offset),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        vt_compute_layout_full.inner,
                        dtype=self.q_dtype,
                    )
                else:
                    k_smem_ptr1 = cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            k_base + Int32(self.kv_compute_chunk1_byte_offset),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        kv_swz,
                        dtype=self.q_dtype,
                    ) if cutlass.const_expr(self.tma_chunked_kv_swizzle) else cute.make_ptr(
                        self.q_dtype,
                        k_base + Int32(self.kv_compute_chunk1_byte_offset),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                    v_smem_ptr1 = cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            v_base + Int32(self.kv_compute_chunk1_byte_offset),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        kv_swz,
                        dtype=self.q_dtype,
                    ) if cutlass.const_expr(self.tma_chunked_kv_swizzle) else cute.make_ptr(
                        self.q_dtype,
                        v_base + Int32(self.kv_compute_chunk1_byte_offset),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                sK1 = cute.make_tensor(
                    k_smem_ptr1,
                    k_compute_layout,
                )
                _tCsK1 = thr_mma.partition_B(sK1)
                tCrK1 = tiled_mma.make_fragment_B(_tCsK1)
                tKrK1_view = smem_thr_copy_K.retile(tCrK1)
                tKsK1 = smem_thr_copy_K.partition_S(sK1)
                sVt1 = cute.make_tensor(
                    v_smem_ptr1,
                    vt_compute_layout,
                )
                _tBsVt1 = thr_mma.partition_B(sVt1)
                tBrVt1 = tiled_mma.make_fragment_B(_tBsVt1)
                tVrVt1_view = smem_thr_copy_Vt.retile(tBrVt1)
                tVsVt1 = smem_thr_copy_Vt.partition_S(sVt1)
            if cutlass.const_expr(self.D > 128):
                if cutlass.const_expr(self.tma_flash_kv_layout):
                    k_smem_ptr2 = cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            k_base + Int32(self.kv_compute_chunk2_byte_offset),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        k_compute_layout_full.inner,
                        dtype=self.q_dtype,
                    )
                    v_smem_ptr2 = cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            v_base + Int32(self.kv_compute_chunk2_byte_offset),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        vt_compute_layout_full.inner,
                        dtype=self.q_dtype,
                    )
                else:
                    k_smem_ptr2 = cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            k_base + Int32(self.kv_compute_chunk2_byte_offset),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        kv_swz,
                        dtype=self.q_dtype,
                    ) if cutlass.const_expr(self.tma_chunked_kv_swizzle) else cute.make_ptr(
                        self.q_dtype,
                        k_base + Int32(self.kv_compute_chunk2_byte_offset),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                    v_smem_ptr2 = cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            v_base + Int32(self.kv_compute_chunk2_byte_offset),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        kv_swz,
                        dtype=self.q_dtype,
                    ) if cutlass.const_expr(self.tma_chunked_kv_swizzle) else cute.make_ptr(
                        self.q_dtype,
                        v_base + Int32(self.kv_compute_chunk2_byte_offset),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                sK2 = cute.make_tensor(
                    k_smem_ptr2,
                    k_compute_layout,
                )
                _tCsK2 = thr_mma.partition_B(sK2)
                tCrK2 = tiled_mma.make_fragment_B(_tCsK2)
                tKrK2_view = smem_thr_copy_K.retile(tCrK2)
                tKsK2 = smem_thr_copy_K.partition_S(sK2)
                sVt2 = cute.make_tensor(
                    v_smem_ptr2,
                    vt_compute_layout,
                )
                _tBsVt2 = thr_mma.partition_B(sVt2)
                tBrVt2 = tiled_mma.make_fragment_B(_tBsVt2)
                tVrVt2_view = smem_thr_copy_Vt.retile(tBrVt2)
                tVsVt2 = smem_thr_copy_Vt.partition_S(sVt2)
            if cutlass.const_expr(self.D > 192):
                if cutlass.const_expr(self.tma_flash_kv_layout):
                    k_smem_ptr3 = cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            k_base + Int32(self.kv_compute_chunk3_byte_offset),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        k_compute_layout_full.inner,
                        dtype=self.q_dtype,
                    )
                    v_smem_ptr3 = cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            v_base + Int32(self.kv_compute_chunk3_byte_offset),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        vt_compute_layout_full.inner,
                        dtype=self.q_dtype,
                    )
                else:
                    k_smem_ptr3 = cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            k_base + Int32(self.kv_compute_chunk3_byte_offset),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        kv_swz,
                        dtype=self.q_dtype,
                    ) if cutlass.const_expr(self.tma_chunked_kv_swizzle) else cute.make_ptr(
                        self.q_dtype,
                        k_base + Int32(self.kv_compute_chunk3_byte_offset),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                    v_smem_ptr3 = cute.recast_ptr(
                        cute.make_ptr(
                            self.q_dtype,
                            v_base + Int32(self.kv_compute_chunk3_byte_offset),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        kv_swz,
                        dtype=self.q_dtype,
                    ) if cutlass.const_expr(self.tma_chunked_kv_swizzle) else cute.make_ptr(
                        self.q_dtype,
                        v_base + Int32(self.kv_compute_chunk3_byte_offset),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                sK3 = cute.make_tensor(
                    k_smem_ptr3,
                    k_compute_layout,
                )
                _tCsK3 = thr_mma.partition_B(sK3)
                tCrK3 = tiled_mma.make_fragment_B(_tCsK3)
                tKrK3_view = smem_thr_copy_K.retile(tCrK3)
                tKsK3 = smem_thr_copy_K.partition_S(sK3)
                sVt3 = cute.make_tensor(
                    v_smem_ptr3,
                    vt_compute_layout,
                )
                _tBsVt3 = thr_mma.partition_B(sVt3)
                tBrVt3 = tiled_mma.make_fragment_B(_tBsVt3)
                tVrVt3_view = smem_thr_copy_Vt.retile(tBrVt3)
                tVsVt3 = smem_thr_copy_Vt.partition_S(sVt3)
            smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.q_dtype)
            smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
            smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)

            acc_S = cute.make_fragment(tiled_mma.partition_shape_C((self.tile_size_M, self.n_block)), Float32)
            rP = cute.make_fragment_like(acc_S, self.q_dtype)
            rP_ld = cute.logical_divide(rP.layout, (None, None, 2))
            rP_mma_view = cute.make_layout(
                ((rP_ld.shape[0], rP_ld.shape[2][0]), rP_ld.shape[1], rP_ld.shape[2][1]),
                stride=((rP_ld.stride[0], rP_ld.stride[2][0]), rP_ld.stride[1], rP_ld.stride[2][1]),
            )
            tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

            acc_O = cute.make_fragment(tiled_mma.partition_shape_C((self.tile_size_M, self.tma_d_block)), Float32)
            acc_O.fill(0.0)
            acc_O_shape = tiled_mma.partition_shape_C((self.tile_size_M, self.tma_d_block))
            acc_O1 = cute.make_fragment(tiled_mma.partition_shape_C((self.tile_size_M, self.tma_d_block)), Float32)
            acc_O2 = cute.make_fragment(tiled_mma.partition_shape_C((self.tile_size_M, self.tma_d_block)), Float32)
            acc_O3 = cute.make_fragment(tiled_mma.partition_shape_C((self.tile_size_M, self.tma_d_block)), Float32)
            acc_O1.fill(0.0)
            acc_O2.fill(0.0)
            acc_O3.fill(0.0)
            num_rows = acc_O_shape[0][1] * acc_O_shape[1]
            row_max = cute.make_fragment(cute.make_layout(num_rows), Float32)
            row_sum = cute.make_fragment(cute.make_layout(num_rows), Float32)
            for r in cutlass.range_constexpr(num_rows):
                row_max[r] = Float32(-1e20)
                row_sum[r] = Float32(0.0)

            mcS = cute.make_identity_tensor((self.tile_size_M, self.n_block))
            tScS = thr_mma.partition_C(mcS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)
            acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
            if cutlass.const_expr(self.D > 64):
                acc_O1_mn = self._make_acc_tensor_mn_view(acc_O1)
                acc_O2_mn = self._make_acc_tensor_mn_view(acc_O2)
                acc_O3_mn = self._make_acc_tensor_mn_view(acc_O3)

            num_kv_blocks_eff = runtime_num_kv_blocks
            first_packed_row = tile_M * Int32(self.tile_size_M)
            first_row = self._seq_row_index(first_packed_row)
            causal_first_limit = first_row + (runtime_N - runtime_M)
            if cutlass.const_expr(self.causal):
                last_packed_row = tile_M * Int32(self.tile_size_M) + Int32(self.tile_size_M - 1)
                last_row = self._seq_row_index(last_packed_row)
                max_col = last_row + (runtime_N - runtime_M)
                num_kv_blocks_eff = (max_col + Int32(self.n_block)) // Int32(self.n_block)
                if num_kv_blocks_eff > runtime_num_kv_blocks:
                    num_kv_blocks_eff = runtime_num_kv_blocks

            k_phase = Int32(0)
            v_phase = Int32(0)
            # Reverse causal traversal only helps when every row in the M tile
            # has valid columns in the rightmost visited K/V block.  For wider
            # M tiles than N blocks, the first rightmost block can be fully
            # masked for some rows and weakens the online-softmax numerics, so
            # those shapes keep the normal left-to-right traversal.
            kv_iter = Int32(0)
            if cutlass.const_expr(self.tma_reverse_kv):
                kv_idx = num_kv_blocks_eff - Int32(1)
            else:
                kv_idx = Int32(0)
            if warp_idx == Int32(0):
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(k_ready, k_bytes)
                    cute.copy(
                        k_tma,
                        tKgK_tma[(None, kv_idx)]
                        if cutlass.const_expr(self.tma_flash_kv_layout)
                        else tKgK_tma[(None, Int32(0), kv_h, kv_idx, tile_B)],
                        tKsK_tma[(None, Int32(0))]
                        if cutlass.const_expr(self.tma_flash_kv_layout)
                        else tKsK_tma,
                        tma_bar_ptr=k_ready_ptr,
                    )
                    if cutlass.const_expr(self.tma_chunked_kv_swizzle and self.num_kv_tma_load_chunks > 1):
                        cute.copy(
                            k_tma,
                            _tKgK_tma1[(None, kv_idx)]
                            if cutlass.const_expr(self.tma_flash_kv_layout)
                            else tKgK_tma[(None, Int32(1), kv_h, kv_idx, tile_B)],
                            tKsK_tma1[(None, Int32(0))]
                            if cutlass.const_expr(self.tma_flash_kv_layout)
                            else tKsK_tma1,
                            tma_bar_ptr=k_ready_ptr,
                        )
                    if cutlass.const_expr(self.tma_chunked_kv_swizzle and self.num_kv_tma_load_chunks > 2):
                        cute.copy(
                            k_tma,
                            _tKgK_tma2[(None, kv_idx)]
                            if cutlass.const_expr(self.tma_flash_kv_layout)
                            else tKgK_tma[(None, Int32(2), kv_h, kv_idx, tile_B)],
                            tKsK_tma2[(None, Int32(0))]
                            if cutlass.const_expr(self.tma_flash_kv_layout)
                            else tKsK_tma2,
                            tma_bar_ptr=k_ready_ptr,
                        )
                    if cutlass.const_expr(self.tma_chunked_kv_swizzle and self.num_kv_tma_load_chunks > 3):
                        cute.copy(
                            k_tma,
                            _tKgK_tma3[(None, kv_idx)]
                            if cutlass.const_expr(self.tma_flash_kv_layout)
                            else tKgK_tma[(None, Int32(3), kv_h, kv_idx, tile_B)],
                            tKsK_tma3[(None, Int32(0))]
                            if cutlass.const_expr(self.tma_flash_kv_layout)
                            else tKsK_tma3,
                            tma_bar_ptr=k_ready_ptr,
                        )

            while kv_iter < num_kv_blocks_eff:
                kv_start = kv_idx * Int32(self.n_block)

                mbarrier_wait(k_ready, k_phase)
                k_phase = k_phase ^ Int32(1)

                if warp_idx == Int32(0):
                        with cute.arch.elect_one():
                            mbarrier_arrive_expect_tx(v_ready, v_bytes)
                        cute.copy(
                            v_tma,
                            tVgV_tma[(None, kv_idx)]
                            if cutlass.const_expr(self.tma_flash_kv_layout)
                            else tVgV_tma[(None, Int32(0), kv_h, kv_idx, tile_B)],
                            tVsV_tma[(None, Int32(0))]
                            if cutlass.const_expr(self.tma_flash_kv_layout)
                            else tVsV_tma,
                            tma_bar_ptr=v_ready_ptr,
                        )
                        if cutlass.const_expr(self.tma_chunked_kv_swizzle and self.num_kv_tma_load_chunks > 1):
                            cute.copy(
                                v_tma,
                                _tVgV_tma1[(None, kv_idx)]
                                if cutlass.const_expr(self.tma_flash_kv_layout)
                                else tVgV_tma[(None, Int32(1), kv_h, kv_idx, tile_B)],
                                tVsV_tma1[(None, Int32(0))]
                                if cutlass.const_expr(self.tma_flash_kv_layout)
                                else tVsV_tma1,
                                tma_bar_ptr=v_ready_ptr,
                            )
                        if cutlass.const_expr(self.tma_chunked_kv_swizzle and self.num_kv_tma_load_chunks > 2):
                            cute.copy(
                                v_tma,
                                _tVgV_tma2[(None, kv_idx)]
                                if cutlass.const_expr(self.tma_flash_kv_layout)
                                else tVgV_tma[(None, Int32(2), kv_h, kv_idx, tile_B)],
                                tVsV_tma2[(None, Int32(0))]
                                if cutlass.const_expr(self.tma_flash_kv_layout)
                                else tVsV_tma2,
                                tma_bar_ptr=v_ready_ptr,
                            )
                        if cutlass.const_expr(self.tma_chunked_kv_swizzle and self.num_kv_tma_load_chunks > 3):
                            cute.copy(
                                v_tma,
                                _tVgV_tma3[(None, kv_idx)]
                                if cutlass.const_expr(self.tma_flash_kv_layout)
                                else tVgV_tma[(None, Int32(3), kv_h, kv_idx, tile_B)],
                                tVsV_tma3[(None, Int32(0))]
                                if cutlass.const_expr(self.tma_flash_kv_layout)
                                else tVsV_tma3,
                                tma_bar_ptr=v_ready_ptr,
                            )

                acc_S.fill(0.0)
                if cutlass.const_expr(self.D <= 64):
                    cute.copy(smem_tiled_copy_K, tKsK[None, None, 0], tKrK_view[None, None, 0])
                    for kb in cutlass.range_constexpr(self.D // 16 - 1):
                        kb_next = kb + 1
                        cute.copy(smem_tiled_copy_K, tKsK[None, None, kb_next], tKrK_view[None, None, kb_next])
                        cute.gemm(tiled_mma, acc_S, tCrQ[None, None, kb], tCrK[None, None, kb], acc_S)
                    cute.gemm(
                        tiled_mma,
                        acc_S,
                        tCrQ[None, None, self.D // 16 - 1],
                        tCrK[None, None, self.D // 16 - 1],
                        acc_S,
                    )
                else:
                    cute.copy(smem_tiled_copy_K, tKsK[None, None, 0], tKrK_view[None, None, 0])
                    for kb in cutlass.range_constexpr(self.tma_d_block // 16 - 1):
                        kb_next = kb + 1
                        cute.copy(smem_tiled_copy_K, tKsK[None, None, kb_next], tKrK_view[None, None, kb_next])
                        cute.gemm(tiled_mma, acc_S, tCrQ[None, None, kb], tCrK[None, None, kb], acc_S)
                    cute.gemm(
                        tiled_mma,
                        acc_S,
                        tCrQ[None, None, self.tma_d_block // 16 - 1],
                        tCrK[None, None, self.tma_d_block // 16 - 1],
                        acc_S,
                    )

                    if cutlass.const_expr(self.D > 64):
                        cute.copy(smem_tiled_copy_K, tKsK1[None, None, 0], tKrK1_view[None, None, 0])
                        for kb in cutlass.range_constexpr(self.tma_d_block // 16 - 1):
                            kb_next = kb + 1
                            cute.copy(
                                smem_tiled_copy_K,
                                tKsK1[None, None, kb_next],
                                tKrK1_view[None, None, kb_next],
                            )
                            cute.gemm(
                                tiled_mma,
                                acc_S,
                                tCrQ[None, None, 1 * (self.tma_d_block // 16) + kb],
                                tCrK1[None, None, kb],
                                acc_S,
                            )
                        cute.gemm(
                            tiled_mma,
                            acc_S,
                            tCrQ[None, None, 1 * (self.tma_d_block // 16) + self.tma_d_block // 16 - 1],
                            tCrK1[None, None, self.tma_d_block // 16 - 1],
                            acc_S,
                        )
                    if cutlass.const_expr(self.D > 128):
                        cute.copy(smem_tiled_copy_K, tKsK2[None, None, 0], tKrK2_view[None, None, 0])
                        for kb in cutlass.range_constexpr(self.tma_d_block // 16 - 1):
                            kb_next = kb + 1
                            cute.copy(
                                smem_tiled_copy_K,
                                tKsK2[None, None, kb_next],
                                tKrK2_view[None, None, kb_next],
                            )
                            cute.gemm(
                                tiled_mma,
                                acc_S,
                                tCrQ[None, None, 2 * (self.tma_d_block // 16) + kb],
                                tCrK2[None, None, kb],
                                acc_S,
                            )
                        cute.gemm(
                            tiled_mma,
                            acc_S,
                            tCrQ[None, None, 2 * (self.tma_d_block // 16) + self.tma_d_block // 16 - 1],
                            tCrK2[None, None, self.tma_d_block // 16 - 1],
                            acc_S,
                        )
                    if cutlass.const_expr(self.D > 192):
                        cute.copy(smem_tiled_copy_K, tKsK3[None, None, 0], tKrK3_view[None, None, 0])
                        for kb in cutlass.range_constexpr(self.tma_d_block // 16 - 1):
                            kb_next = kb + 1
                            cute.copy(
                                smem_tiled_copy_K,
                                tKsK3[None, None, kb_next],
                                tKrK3_view[None, None, kb_next],
                            )
                            cute.gemm(
                                tiled_mma,
                                acc_S,
                                tCrQ[None, None, 3 * (self.tma_d_block // 16) + kb],
                                tCrK3[None, None, kb],
                                acc_S,
                            )
                        cute.gemm(
                            tiled_mma,
                            acc_S,
                            tCrQ[None, None, 3 * (self.tma_d_block // 16) + self.tma_d_block // 16 - 1],
                            tCrK3[None, None, self.tma_d_block // 16 - 1],
                            acc_S,
                        )

                if kv_iter + Int32(1) < num_kv_blocks_eff:
                        if cutlass.const_expr(self.tma_reverse_kv):
                            next_kv_idx = kv_idx - Int32(1)
                        else:
                            next_kv_idx = kv_idx + Int32(1)
                        if warp_idx == Int32(0):
                            with cute.arch.elect_one():
                                mbarrier_arrive_expect_tx(k_ready, k_bytes)
                            cute.copy(
                                k_tma,
                                tKgK_tma[(None, next_kv_idx)]
                                if cutlass.const_expr(self.tma_flash_kv_layout)
                                else tKgK_tma[(None, Int32(0), kv_h, next_kv_idx, tile_B)],
                                tKsK_tma[(None, Int32(0))]
                                if cutlass.const_expr(self.tma_flash_kv_layout)
                                else tKsK_tma,
                                tma_bar_ptr=k_ready_ptr,
                            )
                            if cutlass.const_expr(self.tma_chunked_kv_swizzle and self.num_kv_tma_load_chunks > 1):
                                cute.copy(
                                    k_tma,
                                    _tKgK_tma1[(None, next_kv_idx)]
                                    if cutlass.const_expr(self.tma_flash_kv_layout)
                                    else tKgK_tma[(None, Int32(1), kv_h, next_kv_idx, tile_B)],
                                    tKsK_tma1[(None, Int32(0))]
                                    if cutlass.const_expr(self.tma_flash_kv_layout)
                                    else tKsK_tma1,
                                    tma_bar_ptr=k_ready_ptr,
                                )
                            if cutlass.const_expr(self.tma_chunked_kv_swizzle and self.num_kv_tma_load_chunks > 2):
                                cute.copy(
                                    k_tma,
                                    _tKgK_tma2[(None, next_kv_idx)]
                                    if cutlass.const_expr(self.tma_flash_kv_layout)
                                    else tKgK_tma[(None, Int32(2), kv_h, next_kv_idx, tile_B)],
                                    tKsK_tma2[(None, Int32(0))]
                                    if cutlass.const_expr(self.tma_flash_kv_layout)
                                    else tKsK_tma2,
                                    tma_bar_ptr=k_ready_ptr,
                                )
                            if cutlass.const_expr(self.tma_chunked_kv_swizzle and self.num_kv_tma_load_chunks > 3):
                                cute.copy(
                                    k_tma,
                                    _tKgK_tma3[(None, next_kv_idx)]
                                    if cutlass.const_expr(self.tma_flash_kv_layout)
                                    else tKgK_tma[(None, Int32(3), kv_h, next_kv_idx, tile_B)],
                                    tKsK_tma3[(None, Int32(0))]
                                    if cutlass.const_expr(self.tma_flash_kv_layout)
                                    else tKsK_tma3,
                                    tma_bar_ptr=k_ready_ptr,
                                )

                mbarrier_wait(v_ready, v_phase)
                v_phase = v_phase ^ Int32(1)

                if cutlass.const_expr(not bool(getattr(self, "tma_even_n_block", 0))):
                    if kv_start + Int32(self.n_block) > runtime_N:
                        for r in cutlass.range_constexpr(num_rows):
                            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                                col_idx = tScS_mn[0, c][1]
                                global_col = kv_start + Int32(col_idx)
                                if global_col >= runtime_N:
                                    acc_S_mn[r, c] = Float32(-1e20)

                if cutlass.const_expr(self.causal):
                    last_blk_col = kv_start + Int32(self.n_block - 1)
                    if last_blk_col > causal_first_limit:
                        for r in cutlass.range_constexpr(num_rows):
                            row_idx = tScS_mn[r, 0][0]
                            packed_row = first_packed_row + Int32(row_idx)
                            global_row = self._seq_row_index(packed_row)
                            causal_row_limit = global_row + (runtime_N - runtime_M)
                            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                                col_idx = tScS_mn[0, c][1]
                                global_col = kv_start + Int32(col_idx)
                                if global_col > causal_row_limit:
                                    acc_S_mn[r, c] = Float32(-1e20)

                _any_correction = Int32(0)
                corrections = cute.make_fragment(cute.make_layout(num_rows), Float32)
                for r in cutlass.range_constexpr(num_rows):
                    acc_S_row = acc_S_mn[r, None].load()
                    row_max_cur = acc_S_row.reduce(cute.ReductionOp.MAX, Float32(-1e20), 0)
                    row_max_cur = self._threadquad_reduce_max(row_max_cur)
                    m_old = row_max[r]
                    m_new = cute.arch.fmax(m_old, row_max_cur)
                    acc_scale_ = (m_old - m_new) * Float32(self.scale_log2e)
                    correction = cute.math.exp2(cute.arch.fmax(acc_scale_, Float32(-126.0)), fastmath=True)
                    if acc_scale_ >= Float32(-self.rescale_threshold):
                        m_new = m_old
                        correction = Float32(1.0)
                    row_sum[r] = row_sum[r] * correction
                    corrections[r] = correction
                    if m_new > m_old:
                        _any_correction = Int32(1)
                    acc_S_row_exp = cute.math.exp2(
                        acc_S_row * Float32(self.scale_log2e) - m_new * Float32(self.scale_log2e),
                        fastmath=True,
                    )
                    acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                    row_sum[r] = row_sum[r] + acc_S_row_sum
                    row_max[r] = m_new
                    acc_S_mn[r, None] = acc_S_row_exp

                _skip_rescale = cute.arch.vote_all_sync(_any_correction == Int32(0))
                if not _skip_rescale:
                    for r in cutlass.range_constexpr(num_rows):
                        acc_O_mn[r, None] = acc_O_mn[r, None].load() * corrections[r]
                    if cutlass.const_expr(self.D > 64):
                        for r in cutlass.range_constexpr(num_rows):
                            acc_O1_mn[r, None] = acc_O1_mn[r, None].load() * corrections[r]
                            acc_O2_mn[r, None] = acc_O2_mn[r, None].load() * corrections[r]
                            acc_O3_mn[r, None] = acc_O3_mn[r, None].load() * corrections[r]

                rP.store(acc_S.load().to(self.q_dtype))
                cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, 0], tVrVt_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.n_block // 16 - 1):
                    kb_next = kb + 1
                    cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, kb_next], tVrVt_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_O, tOrS[None, None, kb], tBrVt[None, None, kb], acc_O)
                cute.gemm(
                    tiled_mma,
                    acc_O,
                    tOrS[None, None, self.n_block // 16 - 1],
                    tBrVt[None, None, self.n_block // 16 - 1],
                    acc_O,
                )
                if cutlass.const_expr(self.D > 64):
                    cute.copy(smem_tiled_copy_Vt, tVsVt1[None, None, 0], tVrVt1_view[None, None, 0])
                    for kb in cutlass.range_constexpr(self.n_block // 16 - 1):
                        kb_next = kb + 1
                        cute.copy(smem_tiled_copy_Vt, tVsVt1[None, None, kb_next], tVrVt1_view[None, None, kb_next])
                        cute.gemm(tiled_mma, acc_O1, tOrS[None, None, kb], tBrVt1[None, None, kb], acc_O1)
                    cute.gemm(
                        tiled_mma,
                        acc_O1,
                        tOrS[None, None, self.n_block // 16 - 1],
                        tBrVt1[None, None, self.n_block // 16 - 1],
                        acc_O1,
                    )
                if cutlass.const_expr(self.D > 128):
                    cute.copy(smem_tiled_copy_Vt, tVsVt2[None, None, 0], tVrVt2_view[None, None, 0])
                    for kb in cutlass.range_constexpr(self.n_block // 16 - 1):
                        kb_next = kb + 1
                        cute.copy(smem_tiled_copy_Vt, tVsVt2[None, None, kb_next], tVrVt2_view[None, None, kb_next])
                        cute.gemm(tiled_mma, acc_O2, tOrS[None, None, kb], tBrVt2[None, None, kb], acc_O2)
                    cute.gemm(
                        tiled_mma,
                        acc_O2,
                        tOrS[None, None, self.n_block // 16 - 1],
                        tBrVt2[None, None, self.n_block // 16 - 1],
                        acc_O2,
                    )
                if cutlass.const_expr(self.D > 192):
                    cute.copy(smem_tiled_copy_Vt, tVsVt3[None, None, 0], tVrVt3_view[None, None, 0])
                    for kb in cutlass.range_constexpr(self.n_block // 16 - 1):
                        kb_next = kb + 1
                        cute.copy(smem_tiled_copy_Vt, tVsVt3[None, None, kb_next], tVrVt3_view[None, None, kb_next])
                        cute.gemm(tiled_mma, acc_O3, tOrS[None, None, kb], tBrVt3[None, None, kb], acc_O3)
                    cute.gemm(
                        tiled_mma,
                        acc_O3,
                        tOrS[None, None, self.n_block // 16 - 1],
                        tBrVt3[None, None, self.n_block // 16 - 1],
                        acc_O3,
                    )
                kv_iter = kv_iter + Int32(1)
                if cutlass.const_expr(self.tma_reverse_kv):
                    kv_idx = kv_idx - Int32(1)
                else:
                    kv_idx = kv_idx + Int32(1)

            if cutlass.const_expr(not bool(getattr(self, "tma_scratch_after_k", 0))):
                # N=64 on 32 KiB pages uses page 0 for Q, op-local TMA
                # mbarriers/scratch, and finally O. The mbarrier storage must
                # be invalidated before regular smem writes reuse those bytes.
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
                if warp_idx == Int32(0):
                    with cute.arch.elect_one():
                        mbarrier_inval(k_ready)
                        mbarrier_inval(v_ready)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            for r in cutlass.range_constexpr(num_rows):
                row_sum[r] = self._threadquad_reduce_sum(row_sum[r])

            if cutlass.const_expr(self.write_lse):
                lane_in_quad = tidx % Int32(4)
                for r in cutlass.range_constexpr(num_rows):
                    if lane_in_quad == Int32(0):
                        row_idx = tScS_mn[r, 0][0]
                        packed_row = tile_M * Int32(self.tile_size_M) + Int32(row_idx)
                        global_row = self._seq_row_index(packed_row)
                        if global_row < runtime_M:
                            q_head = self._q_head_index(tile_H, packed_row)
                            lse_head_ptr = lse.iterator + tile_B * runtime_M * Int32(self.H) + q_head
                            g_lse = cute.make_tensor(lse_head_ptr, cute.make_layout(runtime_M, stride=Int32(self.H)))
                            lse_val = row_max[r] * Float32(self.scale_val) + cute.math.log(row_sum[r])
                            g_lse[global_row] = lse_val

            for r in cutlass.range_constexpr(num_rows):
                inv_sum = cute.arch.rcp_approx(row_sum[r])
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv_sum
            if cutlass.const_expr(self.D > 64):
                for r in cutlass.range_constexpr(num_rows):
                    inv_sum = cute.arch.rcp_approx(row_sum[r])
                    acc_O1_mn[r, None] = acc_O1_mn[r, None].load() * inv_sum
            if cutlass.const_expr(self.D > 128):
                for r in cutlass.range_constexpr(num_rows):
                    inv_sum = cute.arch.rcp_approx(row_sum[r])
                    acc_O2_mn[r, None] = acc_O2_mn[r, None].load() * inv_sum
            if cutlass.const_expr(self.D > 192):
                for r in cutlass.range_constexpr(num_rows):
                    inv_sum = cute.arch.rcp_approx(row_sum[r])
                    acc_O3_mn[r, None] = acc_O3_mn[r, None].load() * inv_sum

            tCrO_q = cute.make_fragment_like(acc_O, self.q_dtype)
            for i in cutlass.range_constexpr(cute.size(acc_O)):
                tCrO_q[i] = acc_O[i].to(self.q_dtype)
            o_swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)
            sO = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, o_page, cute.AddressSpace.smem),
                    o_swz,
                    dtype=self.q_dtype,
                ),
                cute.make_layout((self.tile_size_M, self.D), stride=(self.D, 1)),
            )
            if cutlass.const_expr(self.D <= 64):
                tOrO = smem_thr_copy_O.retile(tCrO_q)
                tOsO = smem_thr_copy_O.partition_D(sO)
                cute.copy(smem_tiled_copy_O, tOrO, tOsO)
            else:
                tCrO1_q = cute.make_fragment_like(acc_O1, self.q_dtype)
                tCrO2_q = cute.make_fragment_like(acc_O2, self.q_dtype)
                tCrO3_q = cute.make_fragment_like(acc_O3, self.q_dtype)
                for i in cutlass.range_constexpr(cute.size(acc_O1)):
                    tCrO1_q[i] = acc_O1[i].to(self.q_dtype)
                    tCrO2_q[i] = acc_O2[i].to(self.q_dtype)
                    tCrO3_q[i] = acc_O3[i].to(self.q_dtype)

                sO0 = cute.make_tensor(
                    cute.make_ptr(self.q_dtype, o_page, cute.AddressSpace.smem, assumed_align=128),
                    cute.make_layout((self.tile_size_M, self.tma_d_block), stride=(self.D, 1)),
                )
                sO1 = cute.make_tensor(
                    cute.make_ptr(
                        self.q_dtype,
                        o_page + Int32(self.tma_d_block * self.elem_bytes),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    ),
                    cute.make_layout((self.tile_size_M, self.tma_d_block), stride=(self.D, 1)),
                )
                sO2 = cute.make_tensor(
                    cute.make_ptr(
                        self.q_dtype,
                        o_page + Int32(2 * self.tma_d_block * self.elem_bytes),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    ),
                    cute.make_layout((self.tile_size_M, self.tma_d_block), stride=(self.D, 1)),
                )
                sO3 = cute.make_tensor(
                    cute.make_ptr(
                        self.q_dtype,
                        o_page + Int32(3 * self.tma_d_block * self.elem_bytes),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    ),
                    cute.make_layout((self.tile_size_M, self.tma_d_block), stride=(self.D, 1)),
                )
                tOrO = smem_thr_copy_O.retile(tCrO_q)
                tOrO1 = smem_thr_copy_O.retile(tCrO1_q)
                tOrO2 = smem_thr_copy_O.retile(tCrO2_q)
                tOrO3 = smem_thr_copy_O.retile(tCrO3_q)
                tOsO0 = smem_thr_copy_O.partition_D(sO0)
                tOsO1 = smem_thr_copy_O.partition_D(sO1)
                tOsO2 = smem_thr_copy_O.partition_D(sO2)
                tOsO3 = smem_thr_copy_O.partition_D(sO3)

                cute.copy(smem_tiled_copy_O, tOrO, tOsO0)
                cute.copy(smem_tiled_copy_O, tOrO1, tOsO1)
                if cutlass.const_expr(self.D > 128):
                    cute.copy(smem_tiled_copy_O, tOrO2, tOsO2)
                if cutlass.const_expr(self.D > 192):
                    cute.copy(smem_tiled_copy_O, tOrO3, tOsO3)
            self._compute_store_o(o_page, tile_B, tile_H, tile_M, tile_D, o, runtime_M)

    @cute.jit
    def store(self, page_ptr, tile_B, tile_M, tile_H, tile_D, o_tma, o_tma_gmem):
        if cutlass.const_expr(self.D <= 64):
            _BaseFlashAttentionSm120Op.store(
                self,
                self._o_page(page_ptr),
                tile_B,
                tile_M,
                tile_H,
                tile_D,
                o_tma,
                o_tma_gmem,
            )
        else:
            with cute.arch.elect_one():
                cute.arch.fence_proxy("async.shared", space="cta")
                kv_page = self._o_page(page_ptr)
                sO = cute.make_tensor(
                    cute.make_ptr(self.q_dtype, kv_page, cute.AddressSpace.smem, assumed_align=128),
                    cute.make_layout(self._o_tma_smem_shape, stride=(1, self.D, self.D, self.D * self.tile_size_M)),
                )
                gO = cute.local_tile(
                    o_tma_gmem,
                    self._o_tma_smem_shape,
                    (None, None, None, None),
                )
                tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
                    o_tma,
                    Int32(0),
                    cute.make_layout(1),
                    cute.group_modes(sO, 0, 4),
                    cute.group_modes(gO, 0, 4),
                )
                cute.copy(o_tma, tOsO, tOgO[(None, tile_D, tile_H, tile_M, tile_B)])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=False)
                cute.arch.fence_proxy("async.global")


class Qwen3_5ForwardTwoPageTmaAttentionOp(Qwen3_5ForwardTmaAttentionOp):
    """Qwen TMA attention with separate Q and KV/O shared-memory pages.

    Page 0 holds the framework TMA-loaded Q tile, then the final O tile
    consumed by store after Q is in registers. Page 1 holds compute-issued K/V
    TMA buffers and D-block scratch.
    """

    requested_page_count = 2

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **kwargs):
        ops = super().schedule(tile_sizes=tile_sizes, page_size=page_size, **kwargs)
        q = kwargs.get("q")
        if q is not None:
            D = int(q.shape[-1])
            elem_bytes = int(q.element_size())
            k_d = min(D, int(getattr(cls, "d_block", 64))) if D > 64 else D
            copy_d = D
            tile_m = int((tile_sizes or {}).get("M", ops[0].tile_sizes.get("M", 16)))
            if D > 64 and tile_m * D * elem_bytes >= int(page_size):
                raise ValueError(
                    f"{cls.__name__} D={D} tile_M={tile_m} with page_size={int(page_size)} "
                    "is unsafe: the S2G TMA O tile fills the KV/O page. Use tile_M<=32 "
                    "for two-page 32KB attention, or use "
                    "one-page attention with page_size>=64KB/96KB for tile_M=64."
                )
            reserved_bytes = tile_m * D * elem_bytes + 256
            capacity = max(16, (int(page_size) - reserved_bytes) // (2 * copy_d * elem_bytes))
            n_block = max(16, min(64, (capacity // 16) * 16))
            for op in ops:
                op.static_dims["tma_n_block"] = n_block
                op.static_dims["tma_d_block"] = k_d
                op.static_dims["tma_copy_d_block"] = copy_d
        return ops

    @classmethod
    def page_release_after_compute_mask(cls, page_size: int) -> int:
        return 0

    @cute.jit
    def _q_page(self, page_ptr):
        return self.page_address(page_ptr, 0)

    @cute.jit
    def _kv_page(self, page_ptr):
        return self.page_address(page_ptr, 1)

    @cute.jit
    def _o_page(self, page_ptr):
        return self.page_address(page_ptr, 0)

    @cute.jit
    def load(self, page_ptr, tile_B, tile_M, tile_H, tile_D, q_tma, q_tma_gmem, work_mbar):
        _BaseFlashAttentionSm120Op.load(
            self,
            self._q_page(page_ptr),
            tile_B,
            tile_M,
            tile_H,
            tile_D,
            q_tma,
            q_tma_gmem,
            work_mbar,
        )

    @cute.jit
    def store(self, page_ptr, tile_B, tile_M, tile_H, tile_D, o_tma, o_tma_gmem):
        Qwen3_5ForwardTmaAttentionOp.store(
            self,
            page_ptr,
            tile_B,
            tile_M,
            tile_H,
            tile_D,
            o_tma,
            o_tma_gmem,
        )


class Qwen3_5ForwardThreePageTmaAttentionOp(Qwen3_5ForwardTmaAttentionOp):
    """Qwen TMA attention using three 32 KiB pages with the normal TMA compute path.

    Page 0 holds the framework TMA-loaded Q tile, then scratch/barriers after Q
    has been copied to registers, then the final O tile for store. Page 1 holds
    the compute-issued K TMA tile. Page 2 holds the compute-issued V TMA tile.
    This lets D=256 attention use a larger K/V block on 32 KiB pages without
    introducing another attention implementation.
    """

    requested_page_count = 3

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **kwargs):
        ops = super(Qwen3_5ForwardTmaAttentionOp, cls).schedule(
            tile_sizes=tile_sizes,
            page_size=page_size,
            **kwargs,
        )
        q = kwargs.get("q")
        if q is None:
            for op in ops:
                tile_m = int((tile_sizes or {}).get("M", op.tile_sizes.get("M", 16)))
                op.static_dims.setdefault("num_mma_warps", max(1, tile_m // 16))
                op.static_dims["tma_n_block"] = 32
                op.static_dims["tma_copy_d_block"] = HEAD_DIM
                op.static_dims["tma_separate_kv_pages"] = 1
                op.static_dims["tma_scratch_after_k"] = 1
            return ops

        D = int(q.shape[-1])
        elem_bytes = int(q.element_size())
        k_d = min(D, int(getattr(cls, "d_block", 64))) if D > 64 else D
        copy_d = D
        tile_m = int((tile_sizes or {}).get("M", ops[0].tile_sizes.get("M", 16)))
        page_size = int(page_size)
        q_o_bytes = tile_m * D * elem_bytes
        if q_o_bytes > page_size:
            raise ValueError(
                f"{cls.__name__} D={D} tile_M={tile_m} needs {q_o_bytes} bytes "
                f"for Q/O but page_size={page_size}."
            )

        n_block = 16
        scratch_after_k = 1
        for candidate in (64, 32, 16):
            k_tile_bytes = candidate * copy_d * elem_bytes
            v_tile_bytes = candidate * copy_d * elem_bytes
            q_chunk_bytes = tile_m * k_d * elem_bytes if D > 64 else 0
            k_chunk_bytes = candidate * k_d * elem_bytes if D > 64 else 0
            q_chunk_base = ((k_tile_bytes + 16 + 127) // 128) * 128
            k_chunk_base = ((q_chunk_base + q_chunk_bytes + 127) // 128) * 128
            scratch_after_k_end = k_chunk_base + k_chunk_bytes if D > 64 else k_tile_bytes + 16
            q_chunk_base = ((16 + 127) // 128) * 128
            k_chunk_base = ((q_chunk_base + q_chunk_bytes + 127) // 128) * 128
            scratch_q_page_end = k_chunk_base + k_chunk_bytes if D > 64 else 16
            if (
                k_tile_bytes <= page_size
                and v_tile_bytes <= page_size
                and scratch_after_k_end <= page_size
            ):
                n_block = candidate
                scratch_after_k = 1
                break
            if (
                k_tile_bytes <= page_size
                and v_tile_bytes <= page_size
                and scratch_q_page_end <= page_size
            ):
                n_block = candidate
                scratch_after_k = 0
                break

        for op in ops:
            op.static_dims["tma_n_block"] = n_block
            op.static_dims["tma_d_block"] = k_d
            op.static_dims["tma_copy_d_block"] = copy_d
            op.static_dims["tma_load_d_block"] = k_d
            op.static_dims["tma_separate_kv_pages"] = 1
            op.static_dims["tma_scratch_after_k"] = scratch_after_k
            op.static_dims["tma_chunked_kv_swizzle"] = 1
            op.static_dims["tma_even_n_block"] = int(int(kwargs["k"].shape[1]) % n_block == 0) if kwargs.get("k") is not None else 0
            op.static_dims.setdefault("num_mma_warps", max(1, tile_m // 16))
        return ops

    @classmethod
    def page_release_after_compute_mask(cls, page_size: int) -> int:
        # Store only reads page 0. K/V pages can be recycled after compute.
        return 0b110

    @cute.jit
    def _q_page(self, page_ptr):
        return self.page_address(page_ptr, 0)

    @cute.jit
    def _kv_page(self, page_ptr):
        return self.page_address(page_ptr, 1)

    @cute.jit
    def _k_page(self, page_ptr):
        return self.page_address(page_ptr, 1)

    @cute.jit
    def _v_page(self, page_ptr):
        return self.page_address(page_ptr, 2)

    @cute.jit
    def _scratch_page(self, page_ptr):
        if cutlass.const_expr(bool(getattr(self, "tma_scratch_after_k", 0))):
            return self.page_address(page_ptr, 1)
        return self.page_address(page_ptr, 0)

    @cute.jit
    def _o_page(self, page_ptr):
        return self.page_address(page_ptr, 0)

    @cute.jit
    def load(self, page_ptr, tile_B, tile_M, tile_H, tile_D, q_tma, q_tma_gmem, work_mbar):
        _BaseFlashAttentionSm120Op.load(
            self,
            self._q_page(page_ptr),
            tile_B,
            tile_M,
            tile_H,
            tile_D,
            q_tma,
            q_tma_gmem,
            work_mbar,
        )


def schedule_qwen3_5_forward_split_mma_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor | None = None,
    chunk_m: int = 64,
    num_splits: int = 0,
    page_size: int = 96 * 1024,
    causal: bool = True,
    kv_group_size: int = KV_GROUP_SIZE,
):
    """Schedule tensor-core split-KV attention over M chunks.

    This reuses the existing FlashDecoding split/combine MMA kernels, but
    applies them to prefill-sized tensors by slicing Q/O along M.  Split kernels
    get ``q_row_offset``/``total_M`` so causal masking uses global query rows.
    """

    B, M, H, D = q.shape
    if lse is None:
        lse = torch.empty(B, H, M, dtype=torch.float32, device=q.device)

    ops = []
    keep_alive = [lse]
    for m0 in range(0, M, int(chunk_m)):
        m1 = min(M, m0 + int(chunk_m))
        q_chunk = q[:, m0:m1, :, :]
        o_chunk = o[:, m0:m1, :, :]
        lse_chunk = lse[:, :, m0:m1]
        split_ops, o_partial, lse_partial = FlashDecodingSplitBSHDOp.schedule(
            q=q_chunk,
            k=k,
            v=v,
            num_splits=num_splits,
            page_size=page_size,
            causal=causal,
            kv_group_size=kv_group_size,
        )
        for op in split_ops:
            op.static_dims["q_row_offset"] = int(m0)
            op.static_dims["total_M"] = int(M)
        combine_ops = FlashDecodingCombineBSHDOp.schedule(
            o_partial=o_partial,
            lse_partial=lse_partial,
            o=o_chunk,
            lse=lse_chunk,
        )
        ops += split_ops + combine_ops
        keep_alive.extend([q_chunk, o_chunk, lse_chunk, o_partial, lse_partial])
    return ops, keep_alive


def schedule_qwen3_5_forward_fused_split_mma_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor | None = None,
    tile_m: int = 64,
    combine_tile_m: int = 32,
    num_splits: int = 2,
    page_size: int = 96 * 1024,
    causal: bool = True,
    kv_group_size: int = KV_GROUP_SIZE,
):
    """Schedule one torch-like split-KV MMA prefill op plus one combine op."""

    B, M, H, _D = q.shape
    if lse is None or tuple(lse.shape) != (B, H, M):
        lse = torch.empty(B, H, M, dtype=torch.float32, device=q.device)
    if int(num_splits) == 1:
        direct_cls = (
            Qwen3_5ForwardThreePagePrefillDirectBSHDOp
            if int(page_size) <= 32 * 1024 and int(tile_m) >= 64
            else Qwen3_5ForwardPrefillDirectBSHDOp
        )
        split_ops = direct_cls.schedule(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            page_size=page_size,
            causal=causal,
            kv_group_size=kv_group_size,
            write_lse=False,
            tile_sizes={"B": 1, "M": int(tile_m), "H": 1},
        )
        return split_ops, [lse]
    split_cls = (
        Qwen3_5ForwardThreePagePrefillSplitBSHDOp
        if int(page_size) <= 32 * 1024 and int(q.shape[-1]) >= 256
        else Qwen3_5ForwardPrefillSplitBSHDOp
    )
    split_ops, o_partial, lse_partial = split_cls.schedule(
        q=q,
        k=k,
        v=v,
        num_splits=num_splits,
        page_size=page_size,
        causal=causal,
        kv_group_size=kv_group_size,
        tile_sizes={"B": 1, "M": int(tile_m), "H": 1, "SPLIT": 1},
    )
    combine_ops = Qwen3_5ForwardPrefillCombineBSHDOp.schedule(
        o_partial=o_partial,
        lse_partial=lse_partial,
        o=o,
        lse=lse,
        tile_sizes={"B": 1, "M": int(combine_tile_m), "H": 1},
    )
    return split_ops + combine_ops, [lse, o_partial, lse_partial]


class Qwen3_5ForwardDirectGLUOp(DirectGLUOp):
    """Direct GLU with dependency-visible packed-input regions."""

    reads = {
        "x": (None, ("B", "S", "N")),
    }
    writes = DirectGLUOp.writes
    tile = DirectGLUOp.tile
    dynamic_dims = ("B", "S")
    inline_phases = ("load", "compute", "store")
    tma_loads = {"x"}
    tma_stores = {"y"}

    @classmethod
    def access_regions(cls, op) -> AccessRegions:
        return super().access_regions(op)

    @classmethod
    def schedule(cls, tile_sizes=None, activation="silu", page_size=DEFAULT_PAGE_SIZE, **tensors):
        return super().schedule(
            tile_sizes=tile_sizes,
            activation=activation,
            page_size=page_size,
            **tensors,
        )


class Qwen3_5ForwardGLUOp(_BaseGLUOp):
    """Qwen forward MLP activation op."""

    @classmethod
    def schedule(cls, tile_sizes=None, activation="silu", page_size=DEFAULT_PAGE_SIZE, **tensors):
        tile_sizes = dict(tile_sizes or {})
        if activation in ("silu", "swish") and tensors["y"].shape[-1] == INTERMEDIATE:
            # Current Qwen forward uses the direct-global GLU path. The generic
            # default S=16/D=512 creates too many small tiles between gate_up and
            # down. S=64/D=256 was the best full-layer variant in the local sweep.
            tile_sizes.setdefault("S", 64)
            tile_sizes.setdefault("D", 256)
            return Qwen3_5ForwardDirectGLUOp.schedule(
                tile_sizes=tile_sizes,
                activation=activation,
                page_size=page_size,
                **tensors,
            )
        return super().schedule(
            tile_sizes=tile_sizes,
            activation=activation,
            page_size=page_size,
            **tensors,
        )


class Qwen3_5ForwardResidualAddOp(_BaseResidualAddSm120Op):
    """Qwen forward final residual add op."""


class Qwen3_5ForwardPrefillDirectBSHDOp(FlashPrefillDirectBSHDOp):
    """Qwen-local direct prefill attention with declared sequence regions."""

    # Load only stages Q through TMA. K/V are streamed from compute, so their
    # dependency waits should not block Q prefetch in the controller.
    compute_wait_inputs = {"k", "v"}

    @classmethod
    def access_regions(cls, op) -> AccessRegions:
        return _qwen_attention_regions(op, super().access_regions(op))

    @classmethod
    def schedule(cls, *args, **kwargs):
        ops = super().schedule(*args, **kwargs)
        for op in ops:
            op.static_dims.pop("barrier_signal_o_alias_M", None)
        return ops

    @cute.jit
    def load(self, page_ptr, tile_B, tile_M, tile_H, q_tma, q_tma_gmem, work_mbar):
        FlashPrefillSplitBSHDOp.load(
            self,
            page_ptr,
            tile_B,
            tile_M,
            tile_H,
            Int32(0),
            q_tma,
            q_tma_gmem,
            work_mbar,
        )


class Qwen3_5ForwardThreePagePrefillDirectBSHDOp(Qwen3_5ForwardPrefillDirectBSHDOp):
    """Direct split-MMA prefill using three 32 KiB pages.

    Page 0 holds the TMA-loaded Q tile. Pages 1 and 2 hold double-buffered K/V
    stages, each laid out as K then V. This matches the 96 KiB single-page
    shared-memory budget for M=64,D=256 without forcing the whole megakernel to
    use 96 KiB pages.
    """

    requested_page_count = 3

    @classmethod
    def page_release_after_compute_mask(cls, page_size: int) -> int:
        return 0b110

    @classmethod
    def page_release_inside_compute_mask(cls, page_size: int) -> int:
        return 0b001

    @classmethod
    def schedule(cls, tile_sizes=None, causal=False, page_size=DEFAULT_PAGE_SIZE,
                 kv_group_size=1, write_lse=True, **tensors):
        q = tensors["q"]
        k = tensors["k"]
        v = tensors["v"]
        o = tensors["o"]
        lse = tensors["lse"]
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4 and o.ndim == 4
        B, M, H, D = q.shape
        N = k.shape[1]
        assert k.shape[0] == B and k.shape[3] == D and v.shape == k.shape
        assert o.shape == q.shape
        assert lse.shape == (B, H, M)

        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("M", 64)
        tile_sizes.setdefault("H", 1)

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        elem = q.element_size()
        q_tile_bytes = tile_sizes["M"] * D * elem
        effective_page_size = int(page_size) * int(cls.requested_page_count)
        n_block, _ = cls._pick_kv_blocking(effective_page_size, q_tile_bytes, D, elem, N)
        ops[0].static_dims["n_block"] = n_block
        ops[0].static_dims["page_size"] = int(page_size)
        ops[0].static_dims["multi_page_kv"] = 1
        ops[0].static_dims["multi_page_count"] = int(cls.requested_page_count)
        ops[0].static_dims["num_splits"] = 1
        ops[0].static_dims["write_lse"] = 1 if write_lse else 0
        ops[0].static_dims["M"] = M
        ops[0].static_dims["N"] = N
        ops[0].static_dims["H"] = H
        ops[0].static_dims["SPLIT"] = 1
        ops[0].static_dims["k_b_stride"] = k.stride(0)
        ops[0].static_dims["k_n_stride"] = k.stride(1)
        ops[0].static_dims["k_h_stride"] = k.stride(2)
        ops[0].static_dims["v_b_stride"] = v.stride(0)
        ops[0].static_dims["v_n_stride"] = v.stride(1)
        ops[0].static_dims["v_h_stride"] = v.stride(2)
        ops[0].static_dims["o_b_stride"] = o.stride(0)
        ops[0].static_dims["o_m_stride"] = o.stride(1)
        ops[0].static_dims["o_h_stride"] = o.stride(2)
        if causal:
            ops[0].static_dims["causal"] = 1
        if kv_group_size > 1:
            ops[0].static_dims["kv_group_size"] = int(kv_group_size)
        return ops

    @cute.jit
    def load(self, page_ptr, tile_B, tile_M, tile_H, q_tma, q_tma_gmem, work_mbar):
        FlashPrefillSplitBSHDOp.load(
            self,
            self.page_address(page_ptr, 0),
            tile_B,
            tile_M,
            tile_H,
            Int32(0),
            q_tma,
            q_tma_gmem,
            work_mbar,
        )

    @cute.jit
    def compute_mma_direct(
        self,
        page_ptr,
        tile_B,
        tile_M,
        tile_H,
        q,
        k,
        v,
        o,
        lse,
        page_release_table_ptr,
        page_release_page_base,
        page_release_mbar_base,
        page_release_mbar_stride,
        page_release_page_size,
    ):
        self._compute_mma_impl(
            page_ptr,
            tile_B,
            tile_M,
            tile_H,
            Int32(0),
            q,
            k,
            v,
            o,
            lse,
            o,
            lse,
            True,
            True,
            page_release_table_ptr,
            page_release_page_base,
            page_release_mbar_base,
            page_release_mbar_stride,
            page_release_page_size,
        )


class Qwen3_5ForwardThreePagePrefillSplitBSHDOp(FlashPrefillSplitBSHDOp):
    """Split-KV prefill using three 32 KiB physical pages.

    Page 0 holds the TMA-loaded Q tile. Pages 1 and 2 hold double-buffered K/V
    stages. This keeps Qwen D=256 split attention on 32 KiB allocator pages
    instead of requiring one monolithic 96 KiB page.
    """

    requested_page_count = 3
    compute_wait_inputs = {"k", "v"}

    @classmethod
    def access_regions(cls, op) -> AccessRegions:
        return _qwen_attention_regions(op, super().access_regions(op))

    @classmethod
    def page_release_after_compute_mask(cls, page_size: int) -> int:
        return 0b110

    @classmethod
    def page_release_inside_compute_mask(cls, page_size: int) -> int:
        return 0b001

    @classmethod
    def schedule(cls, tile_sizes=None, causal=False, page_size=DEFAULT_PAGE_SIZE,
                 kv_group_size=1, num_splits=2, **tensors):
        import torch

        q = tensors["q"]
        k = tensors["k"]
        v = tensors["v"]
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
        B, M, H, D = q.shape
        N = k.shape[1]
        assert k.shape[0] == B and k.shape[3] == D and v.shape == k.shape
        assert q.element_size() == 2

        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("M", 64)
        tile_sizes.setdefault("H", 1)
        tile_sizes.setdefault("SPLIT", 1)
        num_splits = max(1, int(num_splits))

        o_partial = torch.empty(B, H, num_splits, M, D, dtype=torch.float32, device=q.device)
        lse_partial = torch.empty(B, H, num_splits, M, dtype=torch.float32, device=q.device)
        tensors["o_partial"] = o_partial
        tensors["lse_partial"] = lse_partial

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        elem = q.element_size()
        q_tile_bytes = tile_sizes["M"] * D * elem
        effective_page_size = int(page_size) * int(cls.requested_page_count)
        n_block, _ = cls._pick_kv_blocking(effective_page_size, q_tile_bytes, D, elem, N)
        ops[0].static_dims["n_block"] = n_block
        ops[0].static_dims["page_size"] = int(page_size)
        ops[0].static_dims["multi_page_kv"] = 1
        ops[0].static_dims["multi_page_count"] = int(cls.requested_page_count)
        ops[0].static_dims["num_splits"] = num_splits
        ops[0].static_dims["M"] = M
        ops[0].static_dims["N"] = N
        ops[0].static_dims["H"] = H
        ops[0].static_dims["SPLIT"] = num_splits
        ops[0].static_dims["k_b_stride"] = k.stride(0)
        ops[0].static_dims["k_n_stride"] = k.stride(1)
        ops[0].static_dims["k_h_stride"] = k.stride(2)
        ops[0].static_dims["v_b_stride"] = v.stride(0)
        ops[0].static_dims["v_n_stride"] = v.stride(1)
        ops[0].static_dims["v_h_stride"] = v.stride(2)
        if causal:
            ops[0].static_dims["causal"] = 1
        if kv_group_size > 1:
            ops[0].static_dims["kv_group_size"] = int(kv_group_size)
        return ops, o_partial, lse_partial

    @cute.jit
    def load(self, page_ptr, tile_B, tile_M, tile_H, tile_SPLIT, q_tma, q_tma_gmem, work_mbar):
        FlashPrefillSplitBSHDOp.load(
            self,
            self.page_address(page_ptr, 0),
            tile_B,
            tile_M,
            tile_H,
            tile_SPLIT,
            q_tma,
            q_tma_gmem,
            work_mbar,
        )

    @cute.jit
    def compute_mma(
        self,
        page_ptr,
        tile_B,
        tile_M,
        tile_H,
        tile_SPLIT,
        q,
        k,
        v,
        o_partial,
        lse_partial,
        page_release_table_ptr,
        page_release_page_base,
        page_release_mbar_base,
        page_release_mbar_stride,
        page_release_page_size,
    ):
        self._compute_mma_impl(
            page_ptr,
            tile_B,
            tile_M,
            tile_H,
            tile_SPLIT,
            q,
            k,
            v,
            o_partial,
            lse_partial,
            o_partial,
            lse_partial,
            False,
            True,
            page_release_table_ptr,
            page_release_page_base,
            page_release_mbar_base,
            page_release_mbar_stride,
            page_release_page_size,
        )


class Qwen3_5ForwardPrefillSplitBSHDOp(FlashPrefillSplitBSHDOp):
    """Qwen-local split prefill attention with declared sequence regions."""

    compute_wait_inputs = {"k", "v"}

    @classmethod
    def access_regions(cls, op) -> AccessRegions:
        return _qwen_attention_regions(op, super().access_regions(op))


class Qwen3_5ForwardPrefillCombineBSHDOp(FlashPrefillCombineBSHDOp):
    """Qwen-local split-attention combine with declared sequence regions."""

    @classmethod
    def access_regions(cls, op) -> AccessRegions:
        return _rename_sequence_regions(super().access_regions(op))


def _alias_access_region_axes(
    region: TensorAccessRegion,
    aliases: dict[str, str],
) -> TensorAccessRegion:
    return replace(
        region,
        axes=tuple(
            replace(axis, name=aliases.get(axis.name, axis.name))
            for axis in region.axes
        ),
    )


class Qwen3_5QKNormRopeQBwdOp(QKNormRopeBwdOp):
    """Qwen Q-norm backward with attention dQ region names.

    The generic op uses sequence axis ``S``.  Attention declares dQ as ``M``,
    so alias only the ``dout`` read region.  Writes remain ``S`` because the
    next projection consumes the normalized-gradient tensor as a sequence.
    """

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=1e-6, **tensors):
        tile_sizes = dict(tile_sizes or {})
        q = tensors.get("q")
        if q is not None:
            tile_sizes.setdefault("B", 1)
            tile_sizes.setdefault("S", min(int(q.shape[1]), 128))
            tile_sizes.setdefault("H", int(q.shape[2]))
        return super().schedule(
            tile_sizes=tile_sizes,
            page_size=page_size,
            eps=eps,
            **tensors,
        )

    @classmethod
    def access_regions(cls, op) -> AccessRegions:
        regions = super().access_regions(op)
        dout = regions.reads.get("dout")
        if not isinstance(dout, TensorAccessRegion):
            return regions
        reads = dict(regions.reads)
        reads["dout"] = _alias_access_region_axes(dout, {"S": "M"})
        writes = dict(regions.writes)
        dq = writes.get("dq")
        if isinstance(dq, TensorAccessRegion):
            h_axis = dq.axis("H")
            if h_axis is not None and h_axis.tile_dim is not None:
                tile_h = int(op.tile_sizes.get(h_axis.tile_dim, 1))
                h_count = int(op.static_dims.get("H", 1))
                if tile_h > 0 and h_count % tile_h == 0:
                    writes["dq"] = _region_with_group(
                        dq,
                        group_dim="H",
                        group_tiles=1,
                        group_count=h_count // tile_h,
                    )
        return AccessRegions(reads=reads, writes=writes)


class Qwen3_5QKNormRopeKBwdOp(QKNormRopeBwdOp):
    """Qwen K-norm backward with attention dK region names."""

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=1e-6, **tensors):
        tile_sizes = dict(tile_sizes or {})
        q = tensors.get("q")
        if q is not None:
            tile_sizes.setdefault("B", 1)
            tile_sizes.setdefault("S", min(int(q.shape[1]), 128))
            tile_sizes.setdefault("H", int(q.shape[2]))
        return super().schedule(
            tile_sizes=tile_sizes,
            page_size=page_size,
            eps=eps,
            **tensors,
        )

    @classmethod
    def access_regions(cls, op) -> AccessRegions:
        regions = super().access_regions(op)
        dout = regions.reads.get("dout")
        if not isinstance(dout, TensorAccessRegion):
            return regions
        reads = dict(regions.reads)
        reads["dout"] = _alias_access_region_axes(dout, {"S": "N", "H": "H_kv"})
        writes = dict(regions.writes)
        dq = writes.get("dq")
        if isinstance(dq, TensorAccessRegion):
            h_axis = dq.axis("H")
            if h_axis is not None and h_axis.tile_dim is not None:
                tile_h = int(op.tile_sizes.get(h_axis.tile_dim, 1))
                h_count = int(op.static_dims.get("H", 1))
                if tile_h > 0 and h_count % tile_h == 0:
                    grouped = _region_with_group(
                        dq,
                        group_dim="H",
                        group_tiles=1,
                        group_count=h_count // tile_h,
                    )
                    writes["dq"] = _alias_access_region_axes(grouped, {"H": "H_kv"})
        return AccessRegions(reads=reads, writes=writes)


class Qwen3_5ProjectionDaReduceGemmOp(ProjectionDaReduceGemmOp):
    """Projection dA reduce-add op with optional attention-value sequence alias."""

    @classmethod
    def access_regions(cls, op) -> AccessRegions:
        regions = super().access_regions(op)
        a_region = regions.reads.get("a")
        if not isinstance(a_region, TensorAccessRegion):
            return regions
        reads = dict(regions.reads)
        if int(op.static_dims.get("alias_a_s_to_n", 0)):
            a_region = _alias_access_region_axes(a_region, {"S": "N"})

        group_axis_kind = int(op.static_dims.get("a_group_axis_kind", 0))
        group_axis = "H" if group_axis_kind == 1 else "H_kv" if group_axis_kind == 2 else ""
        group_count = int(op.static_dims.get("a_group_count", 0))
        group_index_group_tiles = int(op.static_dims.get("a_group_index_group_tiles", 0))
        if group_axis and group_count > 0 and group_index_group_tiles > 0:
            axes = []
            for axis in a_region.axes:
                if axis.name == "K":
                    axes.append(
                        RegionAxis(
                            name=group_axis,
                            extent=group_count,
                            tile_dim="R",
                            tile_size=int(op.tile_sizes.get("R", 1)),
                            tile_origin=int(op.tile_origins.get("R", 0)),
                        )
                    )
                else:
                    axes.append(axis)
            a_region = replace(
                a_region,
                axes=tuple(axes),
                group_index_dim=group_axis,
                group_index_group_tiles=group_index_group_tiles,
            )
        reads["a"] = a_region
        return AccessRegions(reads=reads, writes=regions.writes)


class Qwen3_5ForwardOverlapScheduler(OverlapTileScheduler):
    """Overlap scheduler for Qwen forward producer/consumer prefixes.

    The default overlap scheduler prioritizes source depth, so a ready GEMM can
    still wait until most RMSNorm tiles have been issued. Pure ready-consumer
    priority goes too far and starves the producer. This scheduler keeps a
    bounded mix in each fetch wave when both producer and consumer work is ready.
    """

    def __init__(self, *, consumer_fraction: float = 0.35, **kwargs):
        super().__init__(**kwargs)
        self.consumer_fraction = max(0.0, min(1.0, float(consumer_fraction)))
        self._balanced_tile_ranks = {}

    @staticmethod
    def _is_causal_prefill_attention_op(op) -> bool:
        if not op.static_dims.get("causal", 0):
            return False
        if "M" not in op.dim_names or "H" not in op.dim_names:
            return False
        name = op.op_cls.__name__
        return ("Prefill" in name and "BSHD" in name) or (
            name.startswith("Qwen3_5Forward") and "TmaAttentionOp" in name
        )

    def _build_balanced_tile_ranks(self, op_records, fetch_stride: int) -> dict[tuple[int, int], int]:
        """Return per-op tile priority ranks for causal attention.

        Causal prefill attention tiles have cost roughly proportional to the
        sequence tile M.  Row-major M,H ordering gives some persistent CTAs a
        pair of late-M tiles and others only early-M tiles.  This greedy order
        distributes expensive M tiles over the strided CTA streams while keeping
        the same dependency-visible tile granularity.
        """
        ranks: dict[tuple[int, int], int] = {}
        lanes = max(1, int(fetch_stride))
        for rec in op_records:
            op = rec.op
            if not self._is_causal_prefill_attention_op(op):
                continue
            m_axis = op.dim_names["M"]
            if op.total_tiles <= lanes:
                continue

            entries = [
                (int(tile[m_axis]) + 1, tile_idx, tile)
                for tile_idx, tile in enumerate(rec.tiles)
            ]
            entries.sort(reverse=True)
            lane_cost = [0] * lanes
            order: list[int | None] = [None] * len(entries)

            for cost, tile_idx, _tile in entries:
                best_pos = None
                best_key = None
                for pos in range(len(order)):
                    if order[pos] is not None:
                        continue
                    lane = pos % lanes
                    key = (lane_cost[lane], lane, pos)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = pos
                assert best_pos is not None
                order[best_pos] = tile_idx
                lane_cost[best_pos % lanes] += cost

            for rank, tile_idx in enumerate(order):
                if tile_idx is not None:
                    ranks[(rec.op_idx, int(tile_idx))] = rank
        return ranks

    def schedule_with_formulas(self, op_records, edges, formulas):
        fetch_stride = self._resolve_fetch_stride(sum(rec.op.total_tiles for rec in op_records))
        self._balanced_tile_ranks = self._build_balanced_tile_ranks(op_records, fetch_stride)
        try:
            return super().schedule_with_formulas(op_records, edges, formulas)
        finally:
            self._balanced_tile_ranks = {}

    def _priority(
        self,
        *,
        op_idx: int,
        tile_idx: int,
        depths,
        resource_scores,
        has_waits,
    ):
        base = super()._priority(
            op_idx=op_idx,
            tile_idx=tile_idx,
            depths=depths,
            resource_scores=resource_scores,
            has_waits=has_waits,
        )
        rank = self._balanced_tile_ranks.get((int(op_idx), int(tile_idx)))
        if rank is None:
            return base
        return (*base[:-1], -int(rank))

    def _schedule_stride_waves(
        self,
        candidates,
        formulas,
        depths,
        resource_scores,
        has_waits,
        fetch_stride,
    ):
        scheduled = []
        barrier_counts = {}
        barrier_ready_wave = {}
        current_wave = 0

        while candidates:
            ready = []
            fallback_ready = []
            base_ready_entries = []
            pending_by_op = {}
            base_ready_by_op = {}
            slack_ready_by_op = {}
            for op_idx, _tile_idx, _instr in candidates:
                pending_by_op[op_idx] = pending_by_op.get(op_idx, 0) + 1
            for pos, (op_idx, tile_idx, instr) in enumerate(candidates):
                wait_formulas = formulas.get(op_idx, ([], []))[0]
                if self._waits_ready(instr, wait_formulas, barrier_counts):
                    base_ready_by_op[op_idx] = base_ready_by_op.get(op_idx, 0) + 1
                    entry = (
                        *self._priority(
                            op_idx=op_idx,
                            tile_idx=tile_idx,
                            depths=depths,
                            resource_scores=resource_scores,
                            has_waits=has_waits,
                        ),
                        pos,
                    )
                    base_ready_entries.append((entry, op_idx, instr, wait_formulas))
                    fallback_ready.append(entry)
                    if self._waits_ready_with_slack(
                        instr,
                        wait_formulas,
                        barrier_counts,
                        barrier_ready_wave,
                        current_wave,
                        op_idx,
                    ):
                        slack_ready_by_op[op_idx] = slack_ready_by_op.get(op_idx, 0) + 1

            for entry, op_idx, instr, wait_formulas in base_ready_entries:
                if (
                    self._slack_applies(op_idx)
                    and (
                        base_ready_by_op.get(op_idx, 0) < pending_by_op.get(op_idx, 0)
                        or slack_ready_by_op.get(op_idx, 0) < pending_by_op.get(op_idx, 0)
                    )
                ):
                    continue
                if self._waits_ready_with_slack(
                    instr,
                    wait_formulas,
                    barrier_counts,
                    barrier_ready_wave,
                    current_wave,
                    op_idx,
                ):
                    ready.append(entry)

            if not ready:
                if fallback_ready:
                    ready = fallback_ready
                else:
                    raise RuntimeError(
                        "Qwen3_5ForwardOverlapScheduler could not find a ready tile."
                    )

            producers = [
                entry for entry in ready
                if not has_waits[candidates[entry[-1]][0]]
            ]
            consumers = [
                entry for entry in ready
                if has_waits[candidates[entry[-1]][0]]
            ]
            producers.sort(reverse=True)
            consumers.sort(reverse=True)
            if producers and consumers:
                consumer_quota = max(1, int(fetch_stride * self.consumer_fraction))
                producer_quota = max(1, fetch_stride - consumer_quota)
                selected = producers[:producer_quota] + consumers[:consumer_quota]
                if len(selected) < fetch_stride:
                    used = {entry[-1] for entry in selected}
                    rest = [entry for entry in ready if entry[-1] not in used]
                    rest.sort(reverse=True)
                    selected.extend(rest[: fetch_stride - len(selected)])
            else:
                ready.sort(reverse=True)
                selected = ready[:fetch_stride]

            selected_positions = [entry[-1] for entry in selected]
            selected_records = [(pos, candidates[pos]) for pos in selected_positions]

            for pos in sorted(selected_positions, reverse=True):
                candidates.pop(pos)

            for _pos, (_op_idx, _tile_idx, instr) in selected_records:
                scheduled.append(instr)
            for _pos, (op_idx, _tile_idx, instr) in selected_records:
                signal_formulas = formulas.get(op_idx, ([], []))[1]
                self._signal(
                    instr,
                    signal_formulas,
                    barrier_counts,
                    barrier_ready_wave,
                    current_wave,
                )
            current_wave += 1

        return scheduled


class Qwen3_5ForwardCausalRankScheduler(Qwen3_5ForwardOverlapScheduler):
    """Overlap scheduler with only causal-attention tile re-ranking.

    ``Qwen3_5ForwardOverlapScheduler`` also enforces a producer/consumer mix per
    wave, which is useful for some full-forward RMS/projection boundaries but can
    delay ready consumers after attention.  This variant keeps the generic
    readiness/interleaving policy and only changes the order of causal attention
    tiles so late, expensive M tiles are distributed across persistent CTAs.
    """

    def _schedule_stride_waves(
        self,
        candidates,
        formulas,
        depths,
        resource_scores,
        has_waits,
        fetch_stride,
    ):
        return OverlapTileScheduler._schedule_stride_waves(
            self,
            candidates,
            formulas,
            depths,
            resource_scores,
            has_waits,
            fetch_stride,
        )


def _rms_tile_sizes_for_overlap(
    batch: int,
    seq_len: int,
    page_size: int,
    scheduler,
    rms_tile_s: int | None,
) -> dict[str, int] | None:
    if rms_tile_s is not None:
        return {"S": int(rms_tile_s)}
    if scheduler is None:
        return None
    if int(page_size) <= 32 * 1024:
        # The largest fitting D=1024 RMS row tile is 15 at 32 KiB, but 15 is
        # incompatible with the following 128-row gate_up GEMM tile, so gate_up
        # waits for the whole RMS surface. S=8 divides 128 and turns that into
        # a row-slice wait (16 RMS tiles), which wins despite more RMS tiles.
        return {"S": 8}
    # Keep enough RMS producer tiles to feed the following row-tiled GEMMs
    # without creating excessive barrier/controller overhead. For the common
    # B=1,S=512 Qwen forward shape this selects S=4; S=3 has more overlap but
    # loses to the extra 171-tile barrier fan-out, while S>=6 delays consumers.
    target_tiles = 70 * 2
    tile_s = max(1, (int(batch) * int(seq_len) + target_tiles - 1) // target_tiles)
    return {"S": tile_s}


def _attention_tile_sizes_for_overlap(
    batch: int,
    seq_len: int,
    page_size: int,
    attention_tile_m: int | None = None,
) -> dict[str, int] | None:
    if attention_tile_m is not None:
        return {"M": int(attention_tile_m)}
    if int(batch) == 1 and int(seq_len) <= 256 and int(page_size) <= 32 * 1024:
        return {"M": 32}
    return None


def qwen3_5_forward_gemm_tile_sizes(
    batch: int,
    seq_len: int,
    input_k: int,
    output_n: int,
    scheduler,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    elem_bytes: int = 2,
) -> dict[str, int] | None:
    if scheduler is None:
        return None
    # Adaptive tiling for the 32KB / 3-page megakernel. The decisive lever for
    # GEMM throughput here is a DEEP K tile (large tile_k => fewer K-loop
    # iterations => less per-iteration LdMatrix/MMA overhead) combined with a C
    # tile small enough (<= half a page) that its store overlaps the next tile's
    # A/B loads across the 3-page ring, and enough tiles to fill all 70 SMs with
    # a couple of waves. The old heuristic maximized the C tile, which pinned
    # tile_k at 16 and lost ~10% vs torch; preferring deep K beats torch.
    min_tiles = 70 * 2
    best = None
    for tile_s in (256, 128, 64, 32, 16):
        if tile_s > int(seq_len):
            continue
        s_tiles = (int(seq_len) + tile_s - 1) // tile_s
        for tile_n in (256, 128, 64, 32):
            if output_n % tile_n != 0:
                continue
            for tile_k in (64, 32, 16):
                ab_total = 2 * (tile_s + tile_n) * tile_k * elem_bytes + 32
                c_total = tile_s * tile_n * elem_bytes
                if max(ab_total, c_total) > page_size:
                    continue
                tiles = int(batch) * s_tiles * ((int(output_n) + tile_n - 1) // tile_n)
                # Prefer (verified by sweep, e.g. qk -> S128/N64/K32 beats torch):
                #  1. a store-overlappable C tile (c fits in <= half a page so its
                #     TMA store overlaps the next tile's A/B loads on the 3-page ring),
                #  2. the LARGEST such C footprint (more MMA work per A/B load) up to
                #     that half-page cap,
                #  3. the deepest K (fewer K-loop iterations / less overhead),
                #  4. square-ish tiles with more rows (more MMA warps).
                c_overlap = 0 if c_total * 2 <= page_size else 1
                enough = 0 if tiles >= min_tiles else (min_tiles - tiles)
                score = (
                    c_overlap,
                    enough,
                    -min(tile_s * tile_n, page_size // (2 * elem_bytes)),
                    -tile_k,
                    abs(tile_s - tile_n),
                    -tile_s,
                )
                if best is None or score < best[0]:
                    best = (score, {"S": tile_s, "N": tile_n, "K": tile_k})
    return None if best is None else best[1]


def _complete_qwen_forward_gemm_tile_sizes(
    tile_sizes: dict[str, int],
    a: torch.Tensor,
    b: torch.Tensor,
    page_size: int,
) -> None:
    """Fill missing Qwen forward GEMM tiling, especially the inner K tile.

    Callers sometimes pin spatial tiling from an autotune/scheduler experiment
    and leave K unspecified. The generic GEMM default is K=32, which doubles the
    deepest down-projection K-loop. Keep Qwen forward on the deepest fitting K.
    """
    selected = None
    if "S" not in tile_sizes and "N" not in tile_sizes:
        selected = qwen3_5_forward_gemm_tile_sizes(
            int(a.shape[0]),
            int(a.shape[1]),
            int(a.shape[-1]),
            int(b.shape[0]),
            scheduler=True,
            page_size=page_size,
            elem_bytes=int(a.element_size()),
        )
        if selected is not None:
            for key, value in selected.items():
                tile_sizes.setdefault(key, value)

    if "K" in tile_sizes:
        return
    if "S" not in tile_sizes or "N" not in tile_sizes:
        return

    tile_s = int(tile_sizes["S"])
    tile_n = int(tile_sizes["N"])
    elem_bytes = int(a.element_size())
    for tile_k in (64, 32, 16):
        ab_total = 2 * (tile_s + tile_n) * tile_k * elem_bytes + 32
        c_total = tile_s * tile_n * elem_bytes
        if max(ab_total, c_total) <= page_size:
            tile_sizes["K"] = tile_k
            return


def _flatten_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    return x.reshape(x.shape[0] * x.shape[1], heads, HEAD_DIM)


def _unflatten_heads(x: torch.Tensor, batch: int, seq_len: int, heads: int) -> torch.Tensor:
    return x.reshape(batch, seq_len, heads, HEAD_DIM)


def schedule_qwen3_5_forward_ops(
    batch: int,
    seq_len: int,
    x: torch.Tensor,
    residual: torch.Tensor,
    attn_norm: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    mlp_norm: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    scheduler=None,
    gemm_tile_sizes: dict[str, dict[str, int]] | None = None,
    rms_tile_s: int | None = None,
    adaptive_gemm_tiling: bool = False,
    use_packed_qk: bool = True,
    use_qwen_projection: bool = False,
    use_packed_qkv_projection: bool = True,
    use_fused_rms_proj: bool = True,
    use_fused_down_residual: bool = True,
    use_qknorm_4d: bool = True,
    use_cpasync_projection: bool = False,
    use_split_attention: bool = False,
    split_attention_splits: int = 1,
    use_tma_attention: bool = True,
    use_two_page_attention: bool = False,
    use_three_page_attention: bool = True,
    attention_tile_m: int | None = None,
    op_limit: int | None = None,
) -> Qwen3_5ForwardSchedule:
    dtype = x.dtype
    device = x.device
    cos_flat = cos.repeat(batch, 1).contiguous()
    sin_flat = sin.repeat(batch, 1).contiguous()

    if use_tma_attention and not use_split_attention and not use_three_page_attention:
        # The current runtime S2G TMA path faults when the contiguous TMA mode
        # has extent > 2048 or when a split view keeps a larger parent stride.
        # Qwen's packed QK/QKV projections create 2560/3072-wide outputs, while
        # separate Q/K/V projections stay at 2048/512/512 and keep valid TMA
        # descriptors for the one-page D=256 attention path.
        use_packed_qk = False
        use_packed_qkv_projection = False
    if use_tma_attention and not use_split_attention and use_three_page_attention and page_size <= 32 * 1024 and attention_tile_m is None:
        # D=256 Q/O needs 2 * tile_M * D bytes over the Q/O page lifetime.
        # tile_M=128 overflows a 32 KiB page, so make the fast three-page
        # path valid by default.
        attention_tile_m = 64

    if use_packed_qkv_projection and not use_packed_qk:
        raise ValueError("use_packed_qkv_projection requires use_packed_qk=True")

    x0 = torch.empty_like(x)
    qkv = None
    w_qkv = None
    if use_packed_qk:
        if use_packed_qkv_projection:
            qkv_dim = Q_DIM + 2 * KV_DIM
            qkv = torch.empty(batch, seq_len, qkv_dim, dtype=dtype, device=device)
            qk = torch.as_strided(
                qkv,
                (batch, seq_len, Q_DIM + KV_DIM),
                (seq_len * qkv_dim, qkv_dim, 1),
            )
            q = torch.as_strided(
                qkv,
                (batch, seq_len, Q_DIM),
                (seq_len * qkv_dim, qkv_dim, 1),
            )
            k = torch.as_strided(
                qkv,
                (batch, seq_len, KV_DIM),
                (seq_len * qkv_dim, qkv_dim, 1),
                storage_offset=Q_DIM,
            )
            v = torch.as_strided(
                qkv,
                (batch, seq_len, KV_DIM),
                (seq_len * qkv_dim, qkv_dim, 1),
                storage_offset=Q_DIM + KV_DIM,
            )
            w_qkv = torch.cat((w_q, w_k, w_v), dim=0).contiguous()
            w_qk = None
        else:
            qk = torch.empty(batch, seq_len, Q_DIM + KV_DIM, dtype=dtype, device=device)
            q = qk[..., :Q_DIM]
            k = qk[..., Q_DIM:]
            v = torch.empty(batch, seq_len, KV_DIM, dtype=dtype, device=device)
            w_qk = torch.cat((w_q, w_k), dim=0).contiguous()
        qk_sumsq = torch.empty(
            batch,
            seq_len,
            NUM_Q_HEADS + NUM_KV_HEADS,
            HEAD_DIM // 64,
            dtype=torch.float32,
            device=device,
        )
    else:
        qk = None
        w_qk = None
        qk_sumsq = None
        q = torch.empty(batch, seq_len, Q_DIM, dtype=dtype, device=device)
        k = torch.empty(batch, seq_len, KV_DIM, dtype=dtype, device=device)
        v = torch.empty(batch, seq_len, KV_DIM, dtype=dtype, device=device)
    attn = torch.empty(batch, seq_len, Q_DIM, dtype=dtype, device=device)
    lse = torch.empty(batch, seq_len, NUM_Q_HEADS, dtype=torch.float32, device=device)
    attn_proj = torch.empty(
        batch,
        seq_len,
        HIDDEN,
        dtype=dtype,
        device=device,
    )
    residual1 = torch.empty_like(x)
    x1 = torch.empty_like(x)
    gate_up = torch.empty(batch, seq_len, 2 * INTERMEDIATE, dtype=dtype, device=device)
    mlp = torch.empty(batch, seq_len, INTERMEDIATE, dtype=dtype, device=device)
    mlp_out = torch.empty(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    y = torch.empty_like(x)

    def _packed_heads_view(base: torch.Tensor, heads: int, offset: int = 0) -> torch.Tensor:
        return torch.as_strided(
            base,
            (batch, seq_len, heads, HEAD_DIM),
            (base.stride(0), base.stride(1), HEAD_DIM, 1),
            storage_offset=base.storage_offset() + offset,
        )

    qh = _flatten_heads(q, NUM_Q_HEADS) if not use_packed_qk else _packed_heads_view(qk, NUM_Q_HEADS, 0)
    kh = _flatten_heads(k, NUM_KV_HEADS) if not use_packed_qk else _packed_heads_view(qk, NUM_KV_HEADS, Q_DIM)
    qkh = (
        torch.as_strided(
            qk,
            (batch * seq_len, NUM_Q_HEADS + NUM_KV_HEADS, HEAD_DIM),
            (qk.stride(1), HEAD_DIM, 1),
            storage_offset=qk.storage_offset(),
        )
        if qk is not None
        else None
    )
    qkh4 = (
        torch.as_strided(
            qk,
            (batch, seq_len, NUM_Q_HEADS + NUM_KV_HEADS, HEAD_DIM),
            (qk.stride(0), qk.stride(1), HEAD_DIM, 1),
            storage_offset=qk.storage_offset(),
        )
        if qk is not None
        else None
    )
    kh4 = _packed_heads_view(qk, NUM_KV_HEADS, Q_DIM) if use_packed_qk else _unflatten_heads(k, batch, seq_len, NUM_KV_HEADS)
    vh = _packed_heads_view(qkv, NUM_KV_HEADS, Q_DIM + KV_DIM) if use_packed_qkv_projection else _unflatten_heads(v, batch, seq_len, NUM_KV_HEADS)
    attn_h = _unflatten_heads(attn, batch, seq_len, NUM_Q_HEADS)

    user_gemm_tile_sizes = gemm_tile_sizes or {}
    resolved_gemm_tile_sizes: dict[str, dict[str, int]] = {}
    if adaptive_gemm_tiling:
        resolved_gemm_tile_sizes.update(
            {
                "qk": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, HIDDEN, Q_DIM + KV_DIM, scheduler, page_size=page_size),
                "qkv": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, HIDDEN, Q_DIM + 2 * KV_DIM, scheduler, page_size=page_size),
                "q": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, HIDDEN, Q_DIM, scheduler, page_size=page_size),
                "k": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, HIDDEN, KV_DIM, scheduler, page_size=page_size),
                "v": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, HIDDEN, KV_DIM, scheduler, page_size=page_size),
                "o": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, Q_DIM, HIDDEN, scheduler, page_size=page_size),
                "gate_up": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, HIDDEN, 2 * INTERMEDIATE, scheduler, page_size=page_size),
                "down": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, INTERMEDIATE, HIDDEN, scheduler, page_size=page_size),
            }
        )
        resolved_gemm_tile_sizes = {
            name: sizes for name, sizes in resolved_gemm_tile_sizes.items() if sizes is not None
        }
    for name, sizes in user_gemm_tile_sizes.items():
        base = dict(resolved_gemm_tile_sizes.get(name) or {})
        base.update(sizes)
        resolved_gemm_tile_sizes[name] = base
    if (
        "qkv" not in user_gemm_tile_sizes
        and use_packed_qkv_projection
        and page_size <= 32 * 1024
        and (int(seq_len) <= 128 or int(seq_len) >= 2048)
    ):
        resolved_gemm_tile_sizes["qkv"] = {"S": 128, "N": 128, "K": 16}
    if "down" not in user_gemm_tile_sizes and page_size <= 32 * 1024:
        resolved_gemm_tile_sizes["down"] = {
            "S": 64 if int(seq_len) <= 128 else 128,
            "N": 32 if int(seq_len) <= 128 else 64,
            "K": 32,
        }
    if "gate_up" not in user_gemm_tile_sizes and page_size <= 32 * 1024:
        if int(seq_len) >= 2048:
            resolved_gemm_tile_sizes["gate_up"] = {"S": 128, "N": 128, "K": 16}
        else:
            resolved_gemm_tile_sizes["gate_up"] = {"S": 128, "N": 64, "K": 64}
    resolved_gemm_tile_sizes.setdefault(
        "o",
        {"S": 64, "N": 128 if page_size <= 32 * 1024 else 64, "K": 32 if page_size <= 32 * 1024 else 64},
    )
    if use_tma_attention and not use_split_attention:
        for name in ("q", "k", "v"):
            base = dict(resolved_gemm_tile_sizes.get(name) or {})
            base["K"] = 32
            resolved_gemm_tile_sizes[name] = base
    rms_tile_sizes = _rms_tile_sizes_for_overlap(batch, seq_len, page_size, scheduler, rms_tile_s)
    def _gemm(name: str, *, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        if name == "qk" and use_qwen_projection:
            scheduled = Qwen3_5ForwardPackedQKProjectionOp.schedule(
                a=a,
                b=b,
                c=c,
                qk_sumsq=qk_sumsq,
                page_size=page_size,
                tile_sizes=resolved_gemm_tile_sizes.get(name),
            )
        else:
            projection_cls = (
                Qwen3_5ForwardCpAsyncProjectionOp
                if use_cpasync_projection
                else Qwen3_5ForwardProjectionOp
            )
            scheduled = projection_cls.schedule(
                a=a,
                b=b,
                c=c,
                page_size=page_size,
                tile_sizes=resolved_gemm_tile_sizes.get(name),
            )
        for op in scheduled:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        return scheduled

    extra_keep: list = []

    def _finish(output: torch.Tensor, residual_out: torch.Tensor = residual) -> Qwen3_5ForwardSchedule:
        keep_alive = [
            *extra_keep,
            cos_flat,
            sin_flat,
            x0,
            q,
            k,
            v,
            qk,
            qk_sumsq,
            qkv,
            w_qk,
            w_qkv,
            qh,
            kh,
            qkh,
            qkh4,
            kh4,
            vh,
            attn,
            attn_h,
            lse,
            attn_proj,
            residual1,
            x1,
            gate_up,
            mlp,
            mlp_out,
            y,
        ]
        return Qwen3_5ForwardSchedule(
            ops=ops,
            output=output,
            residual=residual_out,
            keep_alive=keep_alive,
        )

    def _fused_rms_proj(name, b_raw, c):
        # Bake attn_norm into the projection weight once, then fuse RMSNorm into
        # the projection: the op reads raw x, computes per-row rstd from the A
        # fragment in the K-loop, and scales the output. No x0 global round-trip.
        b_baked = (b_raw.float() * attn_norm.float()).to(dtype).contiguous()
        extra_keep.append(b_baked)
        scheduled = Qwen3_5RmsProjOp.schedule(
            a=x, b=b_baked, c=c, rmsnorm_weight=None,
            page_size=page_size, tile_sizes=resolved_gemm_tile_sizes.get(name),
        )
        for op in scheduled:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        return scheduled

    ops = []
    if not use_fused_rms_proj:
        ops += Qwen3_5ForwardRMSNormOp.schedule(
            x=x,
            weight=attn_norm,
            y=x0,
            page_size=page_size,
            eps=QWEN3_5_EPS,
            tile_sizes=rms_tile_sizes,
        )
        if op_limit is not None and len(ops) >= op_limit:
            return _finish(x0)
    proj_a = x if use_fused_rms_proj else x0
    if use_packed_qk:
        if use_packed_qkv_projection:
            ops += (
                _fused_rms_proj("qkv", w_qkv, qkv)
                if use_fused_rms_proj
                else _gemm("qkv", a=proj_a, b=w_qkv, c=qkv)
            )
        else:
            ops += _fused_rms_proj("qk", w_qk, qk) if use_fused_rms_proj else _gemm("qk", a=proj_a, b=w_qk, c=qk)
        qk_output = qk
    else:
        ops += _fused_rms_proj("q", w_q, q) if use_fused_rms_proj else _gemm("q", a=proj_a, b=w_q, c=q)
        ops += _fused_rms_proj("k", w_k, k) if use_fused_rms_proj else _gemm("k", a=proj_a, b=w_k, c=k)
        qk_output = k
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(qk_output)
    if not use_packed_qkv_projection:
        ops += _fused_rms_proj("v", w_v, v) if use_fused_rms_proj else _gemm("v", a=proj_a, b=w_v, c=v)
        if op_limit is not None and len(ops) >= op_limit:
            return _finish(v)
    if use_packed_qk:
        if use_qknorm_4d:
            qknorm_ops = Qwen3_5ForwardPackedQKNormRope4DOp.schedule(
                qk=qkh4,
                q_norm_weight=q_norm,
                k_norm_weight=k_norm,
                cos=cos_flat,
                sin=sin_flat,
                page_size=page_size,
                eps=QWEN3_5_EPS,
                num_q_heads=NUM_Q_HEADS,
                num_k_heads=NUM_KV_HEADS,
            )
        else:
            qknorm_ops = Qwen3_5ForwardPackedQKNormRopeOp.schedule(
                qk=qkh,
                q_norm_weight=q_norm,
                k_norm_weight=k_norm,
                cos=cos_flat,
                sin=sin_flat,
                page_size=page_size,
                eps=QWEN3_5_EPS,
                num_q_heads=NUM_Q_HEADS,
                num_k_heads=NUM_KV_HEADS,
            )
        ops += qknorm_ops
    else:
        ops += Qwen3_5ForwardQKNormRopeOp.schedule(q=qh, norm_weight=q_norm, cos=cos_flat, sin=sin_flat, page_size=page_size, eps=QWEN3_5_EPS)
        ops += Qwen3_5ForwardQKNormRopeOp.schedule(q=kh, norm_weight=k_norm, cos=cos_flat, sin=sin_flat, page_size=page_size, eps=QWEN3_5_EPS)
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(qk_output)
    _attn_q = qh if use_packed_qk else _unflatten_heads(q, batch, seq_len, NUM_Q_HEADS)
    if use_split_attention:
        split_attention_tile_m = attention_tile_m or (32 if int(seq_len) <= 256 else 64)
        _split_ops, _split_keep_alive = schedule_qwen3_5_forward_fused_split_mma_attention(
            q=_attn_q,
            k=kh4,
            v=vh,
            o=attn_h,
            lse=lse,
            tile_m=split_attention_tile_m,
            combine_tile_m=4,
            num_splits=split_attention_splits or 1,
            page_size=page_size,
            causal=True,
            kv_group_size=KV_GROUP_SIZE,
        )
        extra_keep.extend(_split_keep_alive)
        ops += _split_ops
    elif use_tma_attention:
        attention_cls = (
            Qwen3_5ForwardThreePageTmaAttentionOp
            if use_three_page_attention
            else Qwen3_5ForwardTwoPageTmaAttentionOp
            if use_two_page_attention
            else Qwen3_5ForwardTmaAttentionOp
        )
        ops += attention_cls.schedule(
            q=_attn_q,
            k=kh4,
            v=vh,
            o=attn_h,
            lse=lse,
            causal=True,
            kv_group_size=KV_GROUP_SIZE,
            page_size=page_size,
            tile_sizes=_attention_tile_sizes_for_overlap(
                batch,
                seq_len,
                page_size,
                attention_tile_m,
            ),
            write_lse=False,
        )
    else:
        raise ValueError(
            "Qwen 3.5 forward HEAD_DIM=256 requires TMA attention; "
            "the generic non-TMA attention path has been removed."
        )
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(attn)
    ops += _gemm("o", a=attn, b=w_o, c=attn_proj)
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(attn_proj)
    ops += Qwen3_5ForwardRMSNormOp.schedule(
        x=attn_proj,
        residual_in=residual,
        residual_out=residual1,
        weight=mlp_norm,
        y=x1,
        page_size=page_size,
        eps=QWEN3_5_EPS,
        tile_sizes=rms_tile_sizes,
    )
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(x1, residual1)
    ops += _gemm("gate_up", a=x1, b=w_gate_up, c=gate_up)
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(gate_up, residual1)
    ops += Qwen3_5ForwardGLUOp.schedule(x=gate_up, y=mlp, activation="silu", page_size=page_size)
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(mlp, residual1)
    if use_fused_down_residual:
        # Fuse y = (mlp @ w_down^T) + residual1 into the down GEMM epilogue,
        # removing the standalone residual-add op (a serial tail) + the mlp_out
        # round-trip.
        down_ops = Qwen3_5DownResidualOp.schedule(
            a=mlp, b=w_down, c=y, residual=residual1, page_size=page_size,
            tile_sizes=resolved_gemm_tile_sizes.get("down"),
        )
        for op in down_ops:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        ops += down_ops
        return _finish(y, residual1)

    ops += _gemm("down", a=mlp, b=w_down, c=mlp_out)
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(mlp_out, residual1)
    ops += Qwen3_5ForwardResidualAddOp.schedule(
        x=mlp_out,
        residual_in=residual1,
        residual_out=y,
        tile_sizes={"S": 16, "K": 256},
        page_size=page_size,
    )

    return _finish(y, residual1)


def schedule_qwen3_5_backward_ops(
    batch: int,
    seq_len: int,
    x: torch.Tensor,
    residual: torch.Tensor,
    attn_norm: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    mlp_norm: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    attention_bwd_batch_window: int = 4,
    projection_bwd_input_tile_s: int | None = None,
    projection_bwd_reduce_tile_n: int | None = None,
) -> Qwen3_5BackwardSchedule:
    """Build the Qwen 3.5 layer backward graph as one megakernel schedule."""
    effective_projection_bwd_reduce_tile_n = projection_bwd_reduce_tile_n
    if effective_projection_bwd_reduce_tile_n is None and seq_len <= 512:
        # At short sequence lengths the Q dA many-to-one wait is large enough
        # relative to attention work that exposing two-head Q chunks wins.  At
        # longer sequence lengths the added reduce tiles cost more than the
        # wait saved, so keep the original full-hidden reduction unless the
        # caller explicitly asks for a split.
        effective_projection_bwd_reduce_tile_n = 512

    dtype = x.dtype
    device = x.device

    dy = torch.randn_like(x)
    x0 = torch.randn_like(x)
    q_pre = torch.randn(batch, seq_len, Q_DIM, dtype=dtype, device=device)
    k_pre = torch.randn(batch, seq_len, KV_DIM, dtype=dtype, device=device)
    v = torch.randn(batch, seq_len, KV_DIM, dtype=dtype, device=device)
    q = torch.randn_like(q_pre)
    k = torch.randn_like(k_pre)
    attn = torch.randn(batch, seq_len, Q_DIM, dtype=dtype, device=device)
    lse = torch.randn(batch, seq_len, NUM_Q_HEADS, dtype=torch.float32, device=device)
    residual1 = torch.randn_like(x)
    x1 = torch.randn_like(x)
    gate_up = torch.randn(batch, seq_len, 2 * INTERMEDIATE, dtype=dtype, device=device)
    mlp = torch.randn(batch, seq_len, INTERMEDIATE, dtype=dtype, device=device)

    d_mlp = torch.empty_like(mlp)
    d_gate_up = torch.empty_like(gate_up)
    d_x1 = torch.empty_like(x1)
    d_residual1 = torch.empty_like(residual1)
    d_attn = torch.empty_like(attn)
    dpsum = torch.empty(batch, seq_len, NUM_Q_HEADS, dtype=torch.float32, device=device)
    d_q = torch.zeros(batch, seq_len, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float32, device=device)
    d_k = torch.zeros(batch, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    d_v = torch.zeros_like(d_k)
    d_q_pre = torch.empty_like(q_pre)
    d_k_pre = torch.empty_like(k_pre)
    d_x0 = torch.zeros_like(x0)
    dx = torch.empty_like(x)

    dw_down = torch.empty_like(w_down).unsqueeze(0)
    dw_gate_up = torch.empty_like(w_gate_up).unsqueeze(0)
    dw_o = torch.empty_like(w_o).unsqueeze(0)
    dw_q = torch.empty_like(w_q).unsqueeze(0)
    dw_k = torch.empty_like(w_k).unsqueeze(0)
    dw_v = torch.empty_like(w_v).unsqueeze(0)

    qh_pre = _unflatten_heads(q_pre, batch, seq_len, NUM_Q_HEADS)
    kh_pre = _unflatten_heads(k_pre, batch, seq_len, NUM_KV_HEADS)
    qh = _unflatten_heads(q, batch, seq_len, NUM_Q_HEADS)
    kh = _unflatten_heads(k, batch, seq_len, NUM_KV_HEADS)
    vh = _unflatten_heads(v, batch, seq_len, NUM_KV_HEADS)
    attn_h = _unflatten_heads(attn, batch, seq_len, NUM_Q_HEADS)
    d_attn_h = _unflatten_heads(d_attn, batch, seq_len, NUM_Q_HEADS)

    ops = []
    ops += _BaseGemmOp.schedule_backward(
        dout=dy, a=mlp, b=w_down, da=d_mlp, db=dw_down, page_size=page_size
    )
    ops += GLUBwdOp.schedule(
        dy=d_mlp, x=gate_up, dx=d_gate_up, activation="silu", page_size=page_size
    )
    ops += _BaseGemmOp.schedule_backward(
        dout=d_gate_up, a=x1, b=w_gate_up, da=d_x1, db=dw_gate_up, page_size=page_size
    )
    ops += RMSNormBwdOp.schedule(
        dout=d_x1,
        x=residual1,
        weight=mlp_norm,
        add=dy,
        dx=d_residual1,
        page_size=page_size,
        eps=QWEN3_5_EPS,
    )

    ops += _BaseGemmOp.schedule_backward(
        dout=d_residual1,
        a=attn,
        b=w_o,
        da=d_attn,
        db=dw_o,
        page_size=page_size,
    )
    attention_bwd_tile_h_kv = 1
    for dim_window in iter_dim_windows("B", batch, attention_bwd_batch_window):
        ops += AttentionDPSumOp.schedule(
            dout=d_attn_h,
            o=attn_h,
            dpsum=dpsum,
            page_size=page_size,
            dim_windows=dim_window,
        )
        attn_bwd_common = dict(
            k=kh,
            v=vh,
            q=qh,
            dout=d_attn_h,
            lse=lse,
            dpsum=dpsum,
            dq=d_q,
            dk=d_k,
            dv=d_v,
            causal=True,
            kv_group_size=KV_GROUP_SIZE,
            page_size=page_size,
            dim_windows=dim_window,
        )
        attn_bwd_ops = FlashAttentionSm120BwdOp.schedule(**attn_bwd_common)
        if attn_bwd_ops:
            attention_bwd_tile_h_kv = int(attn_bwd_ops[0].tile_sizes.get("H_kv", 1))
        ops += attn_bwd_ops

    projection_reduce_keep_alive = []

    def _schedule_projection_da_reduce(
        dout,
        weight,
        *,
        alias_a_s_to_n=False,
        a_group_axis_kind=0,
        a_head_count=1,
        a_producer_tile_heads=1,
    ):
        weight_t = weight.t().contiguous()
        projection_reduce_keep_alive.append(weight_t)
        reduce_tile_n = (
            int(effective_projection_bwd_reduce_tile_n)
            if effective_projection_bwd_reduce_tile_n is not None
            else int(dout.shape[-1])
        )
        scheduled = Qwen3_5ProjectionDaReduceGemmOp.schedule(
            a=dout,
            b=weight_t,
            c=d_x0,
            page_size=page_size,
            reduce_tile_n=reduce_tile_n,
            tile_sizes={"S": 128, "N": 64, "K": 32},
        )
        if alias_a_s_to_n:
            for op in scheduled:
                op.static_dims["alias_a_s_to_n"] = 1
        if a_group_axis_kind and a_head_count > 0 and a_producer_tile_heads > 0:
            head_dim = int(dout.shape[-1]) // int(a_head_count)
            if (
                head_dim > 0
                and reduce_tile_n % head_dim == 0
                and int(a_head_count) % int(a_producer_tile_heads) == 0
            ):
                chunk_heads = reduce_tile_n // head_dim
                if (
                    chunk_heads > 0
                    and int(a_head_count) % chunk_heads == 0
                    and chunk_heads % int(a_producer_tile_heads) == 0
                ):
                    group_count = int(a_head_count) // int(a_producer_tile_heads)
                    group_index_group_tiles = chunk_heads // int(a_producer_tile_heads)
                    if group_count <= group_index_group_tiles:
                        return scheduled
                    for op in scheduled:
                        op.static_dims["a_group_axis_kind"] = int(a_group_axis_kind)
                        op.static_dims["a_group_count"] = group_count
                        op.static_dims["a_group_index_group_tiles"] = group_index_group_tiles
        return scheduled

    def _schedule_projection_dweight(dout, weight, db):
        return _BaseGemmOp.schedule_backward(
            dout=dout,
            a=x0,
            b=weight,
            db=db,
            page_size=page_size,
        )

    q_norm_bwd_tile_sizes = None
    if effective_projection_bwd_reduce_tile_n is not None:
        q_chunk_heads = max(1, int(effective_projection_bwd_reduce_tile_n) // HEAD_DIM)
        if NUM_Q_HEADS % q_chunk_heads == 0:
            q_norm_bwd_tile_sizes = {"H": q_chunk_heads}

    q_norm_bwd_ops = Qwen3_5QKNormRopeQBwdOp.schedule(
        q=qh_pre,
        dout=d_q,
        norm_weight=q_norm,
        cos=cos,
        sin=sin,
        dq=_unflatten_heads(d_q_pre, batch, seq_len, NUM_Q_HEADS),
        page_size=page_size,
        eps=QWEN3_5_EPS,
        tile_sizes=q_norm_bwd_tile_sizes,
    )
    ops += q_norm_bwd_ops
    q_norm_bwd_tile_h = int(q_norm_bwd_ops[0].tile_sizes.get("H", NUM_Q_HEADS))

    k_norm_bwd_ops = Qwen3_5QKNormRopeKBwdOp.schedule(
        q=kh_pre,
        dout=d_k,
        norm_weight=k_norm,
        cos=cos,
        sin=sin,
        dq=_unflatten_heads(d_k_pre, batch, seq_len, NUM_KV_HEADS),
        page_size=page_size,
        eps=QWEN3_5_EPS,
    )
    ops += k_norm_bwd_ops
    k_norm_bwd_tile_h = int(k_norm_bwd_ops[0].tile_sizes.get("H", NUM_KV_HEADS))

    ops += _schedule_projection_da_reduce(
        d_q_pre,
        w_q,
        a_group_axis_kind=1,
        a_head_count=NUM_Q_HEADS,
        a_producer_tile_heads=q_norm_bwd_tile_h,
    )
    ops += _schedule_projection_dweight(d_q_pre, w_q, dw_q)
    ops += _schedule_projection_da_reduce(
        d_k_pre,
        w_k,
        a_group_axis_kind=2,
        a_head_count=NUM_KV_HEADS,
        a_producer_tile_heads=k_norm_bwd_tile_h,
    )
    ops += _schedule_projection_dweight(d_k_pre, w_k, dw_k)
    d_v_flat = d_v.reshape(batch, seq_len, KV_DIM)
    ops += _schedule_projection_da_reduce(
        d_v_flat,
        w_v,
        alias_a_s_to_n=True,
        a_group_axis_kind=2,
        a_head_count=NUM_KV_HEADS,
        a_producer_tile_heads=attention_bwd_tile_h_kv,
    )
    ops += _schedule_projection_dweight(d_v_flat, w_v, dw_v)
    ops += RMSNormBwdOp.schedule(
        dout=d_x0,
        x=x,
        weight=attn_norm,
        add=d_residual1,
        dx=dx,
        page_size=page_size,
        eps=QWEN3_5_EPS,
    )
    keep_alive = [
        dy,
        x0,
        q_pre,
        k_pre,
        v,
        q,
        k,
        attn,
        lse,
        residual1,
        x1,
        gate_up,
        mlp,
        d_mlp,
        d_gate_up,
        d_x1,
        d_residual1,
        d_attn,
        dpsum,
        d_q,
        d_k,
        d_v,
        d_q_pre,
        d_k_pre,
        d_v_flat,
        d_x0,
        dx,
        dw_down,
        dw_gate_up,
        dw_o,
        dw_q,
        dw_k,
        dw_v,
        qh_pre,
        kh_pre,
        qh,
        kh,
        vh,
        attn_h,
        d_attn_h,
        *projection_reduce_keep_alive,
    ]
    return Qwen3_5BackwardSchedule(ops=ops, output=dx, keep_alive=keep_alive)


def schedule_qwen3_5_mlp_backward_ops(
    batch: int,
    seq_len: int,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    pointwise_page_size: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda",
) -> Qwen3_5BackwardSchedule:
    """Build the optimized Qwen 3.5 MLP backward graph.

    This is the backward half that is currently supported end-to-end:
    down-projection backward, GLU backward, gate/up projection backward, and
    RMSNorm backward. It mirrors the forward benchmark's single-megakernel MLP
    block and keeps pointwise ops on a caller-selectable page size.
    """
    pointwise_page_size = int(pointwise_page_size or page_size)

    x = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    hidden = torch.randn_like(x)
    gate_up = torch.randn(batch, seq_len, 2 * INTERMEDIATE, dtype=dtype, device=device)
    mlp = torch.randn(batch, seq_len, INTERMEDIATE, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    d_mlp = torch.empty_like(mlp)
    d_gate_up = torch.empty_like(gate_up)
    d_hidden = torch.empty_like(hidden)
    dx = torch.empty_like(x)
    norm = torch.ones(HIDDEN, dtype=dtype, device=device)
    w_gate_up = torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=dtype, device=device) * 0.02
    w_down = torch.randn(HIDDEN, INTERMEDIATE, dtype=dtype, device=device) * 0.02
    dw_gate_up = torch.empty_like(w_gate_up).unsqueeze(0)
    dw_down = torch.empty_like(w_down).unsqueeze(0)

    ops = []
    ops += _BaseGemmOp.schedule_backward(
        dout=dy,
        a=mlp,
        b=w_down,
        da=d_mlp,
        db=dw_down,
        page_size=page_size,
    )
    ops += GLUBwdOp.schedule(
        dy=d_mlp,
        x=gate_up,
        dx=d_gate_up,
        activation="silu",
        page_size=pointwise_page_size,
    )
    ops += _BaseGemmOp.schedule_backward(
        dout=d_gate_up,
        a=hidden,
        b=w_gate_up,
        da=d_hidden,
        db=dw_gate_up,
        page_size=page_size,
    )
    ops += RMSNormBwdOp.schedule(
        dout=d_hidden,
        x=x,
        weight=norm,
        dx=dx,
        page_size=pointwise_page_size,
        eps=QWEN3_5_EPS,
    )

    keep_alive = [
        x,
        hidden,
        gate_up,
        mlp,
        dy,
        d_mlp,
        d_gate_up,
        d_hidden,
        dx,
        norm,
        w_gate_up,
        w_down,
        dw_gate_up,
        dw_down,
    ]
    return Qwen3_5BackwardSchedule(ops=ops, output=dx, keep_alive=keep_alive)


__all__ = [
    "DEFAULT_PAGE_SIZE",
    "HIDDEN",
    "INTERMEDIATE",
    "NUM_Q_HEADS",
    "NUM_KV_HEADS",
    "HEAD_DIM",
    "Q_DIM",
    "KV_DIM",
    "KV_GROUP_SIZE",
    "schedule_qwen3_5_forward_fused_split_mma_attention",
    "schedule_qwen3_5_forward_split_mma_attention",
    "Qwen3_5ForwardTmaAttentionOp",
    "Qwen3_5ForwardTwoPageTmaAttentionOp",
    "Qwen3_5ForwardThreePageTmaAttentionOp",
    "Qwen3_5ForwardGLUOp",
    "Qwen3_5ForwardCausalRankScheduler",
    "Qwen3_5ForwardOverlapScheduler",
    "Qwen3_5ForwardPackedQKProjectionOp",
    "Qwen3_5ForwardPackedQKNormRopeOp",
    "Qwen3_5ForwardPackedQKNormRope4DOp",
    "Qwen3_5ForwardProjectionOp",
    "Qwen3_5ForwardThreePagePrefillDirectBSHDOp",
    "Qwen3_5ForwardQKNormRopeOp",
    "Qwen3_5ForwardResidualAddOp",
    "Qwen3_5BackwardSchedule",
    "Qwen3_5ForwardSchedule",
    "Qwen3_5ForwardRMSNormOp",
    "qwen3_5_forward_gemm_tile_sizes",
    "schedule_qwen3_5_backward_ops",
    "schedule_qwen3_5_forward_ops",
    "schedule_qwen3_5_mlp_backward_ops",
]
