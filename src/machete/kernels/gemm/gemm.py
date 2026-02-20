# Copyright (c) 2025, Machete Authors
"""
GEMM Op for the Megakernel.

Computes C[M,N] = A[M,K] @ B[N,K]^T  (i.e., standard matmul with B pre-transposed).

Note: B is expected in (N, K) layout (K contiguous). If you have B in standard
PyTorch (K, N) layout, pass b.t().contiguous() when scheduling.

Architecture (SM_90+ / Hopper+):
    - Tensor core warp MMA: MmaF16BF16Op(16, 8, 16)
    - Supports fp16 and bf16 inputs, fp32 accumulation across all K blocks
    - K is handled via an inner loop in compute (not a framework tile dim)
    - TMA for G->S loads of A/B K-blocks (issued by DMA warp per inner iter)
    - Regular TMA store for C
    - LdMatrix for warp-cooperative smem->register reads

Pipelined phases:
    load:    TMA G->S of one K-block of A and B (called once per inner iter
             by the framework DMA warp). inner_iter_idx selects the K-block.
    compute: Inner K loop — wait for each K-block load, LdMatrix + MMA,
             signal smem_consumed so DMA can load the next K-block.
             Epilogue: R->S.
    store:   TMA store S->G of C[tile_M, tile_N]

Page layout (16KB):
    During K loop (depth=1): [A: tile_K x tile_M] [B: tile_K x tile_N]
    During K loop (depth=2): [buf0: A+B] [buf1: A+B]  (double buffer)
    Epilogue:                [C: tile_M x tile_N]  (reuses page from offset 0)

No output pre-zeroing needed — fp32 accumulator handles full K reduction.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32

from machete.megakernel.ops import Op
from machete.megakernel.interpreter import (
    mbarrier_arrive, mbarrier_arrive_expect_tx, mbarrier_wait,
    named_barrier_sync,
)


class GemmOp(Op):
    """GEMM operation for the megakernel framework.

    Computes C[M,N] = A[M,K] @ B[N,K]^T using tensor core MMA.
    B must be in (N, K) layout with K contiguous.

    K is handled via an inner loop: the framework DMA warp issues one
    TMA load per K-block (inner_iters = num_k_blocks), and compute
    processes all K-blocks with fp32 accumulation.
    """

    reads = {
        "a": (None, ("M", "K")),
        "b": (None, ("N", "K")),
    }
    writes = {"c": (None, ("M", "N"))}
    tile = ("M", "N")

    tma_loads = {"a", "b"}
    tma_stores = {"c"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        """Custom TMA tile shapes for inner K-block sub-tiling.

        A and B use tile_K (inner block size) instead of full K extent.
        C uses the standard tile_M x tile_N.
        """
        tile_K = static_dims.get("tile_K", 32)
        if tensor_name == "a":
            return (tile_sizes["M"], tile_K)
        elif tensor_name == "b":
            return (tile_sizes["N"], tile_K)
        else:  # "c"
            return (tile_sizes["M"], tile_sizes["N"])

    def __init__(self, **config):
        super().__init__(**config)

        if self.a_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            raise ValueError(f"GemmOp requires fp16 or bf16 input, got {self.a_dtype}")

        # tile_K and inner_depth injected via schedule_forward into static_dims
        self.tile_K = getattr(self, 'tile_K', 32)
        self.inner_depth = getattr(self, 'inner_depth', 1)
        self.page_size = getattr(self, 'page_size', 16384)

        self.a_tile_bytes = self.tile_size_M * self.tile_K * self.elem_bytes
        self.b_tile_bytes = self.tile_size_N * self.tile_K * self.elem_bytes
        self.c_tile_bytes = self.tile_size_M * self.tile_size_N * self.elem_bytes
        self.buf_stride = self.a_tile_bytes + self.b_tile_bytes

        # Smem budget: inner_depth copies of A+B during K loop, C during epilogue
        ab_bytes = self.inner_depth * self.buf_stride
        total_smem = max(ab_bytes, self.c_tile_bytes)
        assert total_smem <= self.page_size, (
            f"GemmOp: smem {total_smem}B exceeds page_size ({self.page_size}B). "
            f"tile_M={self.tile_size_M}, tile_N={self.tile_size_N}, "
            f"tile_K={self.tile_K}, inner_depth={self.inner_depth}"
        )

        assert self.tile_K >= 16 and self.tile_K % 16 == 0, (
            f"GemmOp: tile_K={self.tile_K} must be >= 16 and a multiple of 16."
        )

        self.num_k_blocks = (self.K + self.tile_K - 1) // self.tile_K
        self.num_mma_warps = self.threads_per_row // 32
        self.num_mma_threads = self.num_mma_warps * 32

        # Framework inner iterations: DMA warp calls load() once per K-block
        self.inner_iters = self.num_k_blocks

        # Swizzle parameters for bank-conflict-free LdMatrix reads.
        # Uses CUTLASS convention: make_swizzle(B, 4, 3) where B selects
        # the swizzle mode (SW128/SW64/SW32). Pick the largest mode whose
        # atom size evenly divides tile_K.
        # SW128: atom_K=64, SW64: atom_K=32, SW32: atom_K=16 (for fp16)
        if self.tile_K % 64 == 0 and self.tile_K >= 64:
            self.swz_B_ab = 3   # SW128
            self.atom_K = 64
        elif self.tile_K % 32 == 0:
            self.swz_B_ab = 2   # SW64
            self.atom_K = 32
        else:
            self.swz_B_ab = 1   # SW32
            self.atom_K = 16

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape,
                                tile_sizes, static_dims):
        """Return swizzled smem layout code for TMA descriptor creation.

        TMA hardware supports swizzle modes (32B/64B/128B). CuTe's
        make_tiled_tma_atom detects the swizzle from the composed layout
        and configures the appropriate TMA swizzle mode.

        Uses CUTLASS convention: make_swizzle(B, 4, 3) with fixed M=4, S=3.
        B selects the mode: 3=SW128, 2=SW64, 1=SW32.
        K-major atom: (atom_K, 8) stride (1, atom_K) in TMA (K,M/N) convention.
        """
        tile_K = static_dims.get("tile_K", 32)
        if tensor_name not in ("a", "b"):
            return None  # No swizzle for C

        # Pick largest swizzle whose atom divides tile_K
        if tile_K % 64 == 0 and tile_K >= 64:
            B = 3   # SW128
        elif tile_K % 32 == 0:
            B = 2   # SW64
        else:
            B = 1   # SW32

        dim0, dim1 = tma_tile_shape  # (tile_K, tile_M/N) in TMA convention
        # Flat composed layout: swizzle over full (K, M/N) col-major base.
        # make_tiled_tma_atom extracts the swizzle and sets TMA hardware mode.
        return (
            f"cute.make_composed_layout("
            f"cute.make_swizzle({B}, 4, 3), 0, "
            f"cute.make_layout(({dim0}, {dim1}), stride=(1, {dim0})))"
        )

    # =========================================================================
    # Forward Load: TMA G->S of one K-block of A and B
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_M, tile_N,
             a_tma, a_tma_gmem, b_tma, b_tma_gmem,
             work_mbar, inner_iter_idx):
        """TMA load of one K-block of A and B tiles.

        Called by the DMA warp once per inner iteration (K-block).
        inner_iter_idx selects which K-block to load.

        TMA transposes gmem: A(M,K) -> smem (K,M), B(N,K) -> smem (K,N).
        Smem uses swizzled layout for bank-conflict-free LdMatrix reads.
        """
        # Buffer offset for n-stage pipeline: inner_iter_idx selects K-block,
        # buf_idx = inner_iter_idx % inner_depth selects which smem buffer.
        # When inner_depth=1: inner_iter_idx % 1 = 0, so _buf_base = page_ptr.
        _buf_base = page_ptr + (inner_iter_idx % Int32(self.inner_depth)) * Int32(self.buf_stride)

        # Swizzle on pointer, plain layout: tma_partition requires plain layout
        # but needs the swizzle on the pointer to match TMA descriptor.
        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)
        sA_ptr = cute.recast_ptr(
            cute.make_ptr(self.a_dtype, _buf_base, cute.AddressSpace.smem),
            swz, dtype=self.a_dtype)
        sA = cute.make_tensor(
            sA_ptr,
            cute.make_layout((self.tile_K, self.tile_size_M),
                             stride=(1, self.tile_K)),
        )
        gA = cute.local_tile(
            a_tma_gmem, (self.tile_K, self.tile_size_M), (None, None),
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            a_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )

        sB_ptr = cute.recast_ptr(
            cute.make_ptr(self.b_dtype, _buf_base + Int32(self.a_tile_bytes),
                          cute.AddressSpace.smem),
            swz, dtype=self.b_dtype)
        sB = cute.make_tensor(
            sB_ptr,
            cute.make_layout((self.tile_K, self.tile_size_N),
                             stride=(1, self.tile_K)),
        )
        gB = cute.local_tile(
            b_tma_gmem, (self.tile_K, self.tile_size_N), (None, None),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            b_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )

        nbytes = Int32(self.a_tile_bytes + self.b_tile_bytes)
        mbar_ptr = cute.make_ptr(
            cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)

        cute.copy(a_tma, tAgA[(None, inner_iter_idx, tile_M)], tAsA,
                  tma_bar_ptr=mbar_ptr)
        cute.copy(b_tma, tBgB[(None, inner_iter_idx, tile_N)], tBsB,
                  tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Forward Compute: Inner K loop with DMA-driven loads
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_M, tile_N,
                work_mbar, smem_consumed_mbar, work_mbar_phase):
        """GEMM compute with inner K loop and fp32 accumulation.

        K-block 0 is already loaded by the DMA warp (framework waited on
        work_notify_mbar before calling compute).

        For K-blocks 1+:
            1. Signal smem_consumed for previous buffer (so DMA can reuse it)
            2. Wait on work_mbar for next K-block
            3. LdMatrix + MMA from current buffer

        With inner_depth > 1 (n-stage pipeline), multiple buffers overlap
        DMA loads with MMA compute. smem_consumed_mbar is the base address
        of per-buffer mbarriers (spaced 8 bytes apart).

        After all K-blocks: epilogue converts fp32 acc to output dtype in smem.
        """
        tidx = cute.arch.thread_idx()[0]
        lane_id = tidx % Int32(32)

        # --- Build tiled MMA ---
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            self.a_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        # --- LdMatrix tiled copies (warp-cooperative smem reads) ---
        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)

        smem_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False, num_matrices=4), self.a_dtype)
        smem_tiled_copy_A = cute.make_tiled_copy_A(
            smem_copy_atom_A, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)

        smem_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False, num_matrices=4), self.b_dtype)
        smem_tiled_copy_B = cute.make_tiled_copy_B(
            smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        # --- Buffer 0 smem tensors, MMA partitions, and fragments ---
        # Swizzle on pointer, plain layout — same pattern as CUTLASS.
        sA_ptr = cute.recast_ptr(
            cute.make_ptr(self.a_dtype, page_ptr,
                          cute.AddressSpace.smem, assumed_align=128),
            swz, dtype=self.a_dtype)
        sA = cute.make_tensor(
            sA_ptr,
            cute.make_layout((self.tile_size_M, self.tile_K),
                             stride=(self.tile_K, 1)),
        )
        sB_ptr = cute.recast_ptr(
            cute.make_ptr(self.b_dtype,
                          page_ptr + Int32(self.a_tile_bytes),
                          cute.AddressSpace.smem, assumed_align=128),
            swz, dtype=self.b_dtype)
        sB = cute.make_tensor(
            sB_ptr,
            cute.make_layout((self.tile_size_N, self.tile_K),
                             stride=(self.tile_K, 1)),
        )

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrA_ld = smem_thr_copy_A.retile(tCrA)
        tCrB_ld = smem_thr_copy_B.retile(tCrB)

        # --- Per-buffer LdMatrix smem partitions ---
        # Buffer 0 is already created above. Build list for K-loop indexing.
        _buf_tAsA_ld = [smem_thr_copy_A.partition_S(sA)]
        _buf_tBsB_ld = [smem_thr_copy_B.partition_S(sB)]
        for _d in cutlass.range_constexpr(self.inner_depth - 1):
            _buf_off = (_d + 1) * self.buf_stride
            _sA_d = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.a_dtype, page_ptr + Int32(_buf_off),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.a_dtype),
                cute.make_layout((self.tile_size_M, self.tile_K),
                                 stride=(self.tile_K, 1)),
            )
            _sB_d = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.b_dtype,
                                  page_ptr + Int32(_buf_off + self.a_tile_bytes),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.b_dtype),
                cute.make_layout((self.tile_size_N, self.tile_K),
                                 stride=(self.tile_K, 1)),
            )
            _buf_tAsA_ld.append(smem_thr_copy_A.partition_S(_sA_d))
            _buf_tBsB_ld.append(smem_thr_copy_B.partition_S(_sB_d))

        # --- Init fp32 accumulator ---
        acc = cute.make_fragment(
            tiled_mma.partition_shape_C((self.tile_size_M, self.tile_size_N)),
            Float32,
        )
        acc.fill(0.0)

        # --- Process all K-blocks ---
        # Per-buffer kblock_ready phase tracking:
        # buf 0: framework consumed phase 0, so next wait phase is 1
        # buf 1+: DMA prefilled (arrived phase 0→1), first wait is phase 0
        _kb_phases = [work_mbar_phase]
        for _d in cutlass.range_constexpr(self.inner_depth - 1):
            _kb_phases.append(Int32(0))
        for k in cutlass.range_constexpr(self.num_k_blocks):
            if k > 0:
                # Signal previous buffer consumed — all MMA warps done reading.
                # DMA warp polls per-buffer smem_consumed before reusing buffer.
                prev_buf = (k - 1) % self.inner_depth
                named_barrier_sync(
                    Int32(2), Int32(self.num_mma_threads))
                if lane_id == Int32(0):
                    mbarrier_arrive(
                        smem_consumed_mbar + Int32(prev_buf * 8))

                # Wait for DMA to load the next K-block (per-buffer mbar)
                cur_buf = k % self.inner_depth
                mbarrier_wait(work_mbar + Int32(cur_buf * 8), _kb_phases[cur_buf])
                _kb_phases[cur_buf] = Int32(1) - _kb_phases[cur_buf]

            # Select buffer for this K-block
            cur_buf = k % self.inner_depth
            tAsA_ld = _buf_tAsA_ld[cur_buf]
            tBsB_ld = _buf_tBsB_ld[cur_buf]

            # LdMatrix + MMA for this K-block
            for k_block in cutlass.range_constexpr(self.tile_K // 16):
                cute.copy(smem_tiled_copy_A,
                          tAsA_ld[None, None, k_block],
                          tCrA_ld[None, None, k_block])
                cute.copy(smem_tiled_copy_B,
                          tBsB_ld[None, None, k_block],
                          tCrB_ld[None, None, k_block])
                cute.gemm(tiled_mma, acc,
                          tCrA[None, None, k_block],
                          tCrB[None, None, k_block], acc)

        # --- Epilogue: R->S (write final C to smem for TMA store) ---
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
        sC = cute.make_tensor(
            cute.make_ptr(self.c_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.tile_size_M, self.tile_size_N),
                             stride=(self.tile_size_N, 1)),
        )
        tCsC = thr_mma.partition_C(sC)
        for i in cutlass.range_constexpr(cute.size(acc)):
            tCsC[i] = acc[i].to(self.c_dtype)

    # =========================================================================
    # Forward Store (S->G): Regular TMA store of C
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_M, tile_N, c_tma, c_tma_gmem):
        """TMA store of C from shared to global memory.

        Regular TMA store (overwrite, not atomic add).
        Compute writes C as swizzled (tile_M, tile_N) row-major.
        TMA sees smem as swizzled (tile_N, tile_M) col-major — same physical layout.
        """
        # Plain smem layout for TMA partition (TMA descriptor has swizzle)
        sC = cute.make_tensor(
            cute.make_ptr(self.c_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.tile_size_N, self.tile_size_M)),
        )

        gC = cute.local_tile(
            c_tma_gmem, (self.tile_size_N, self.tile_size_M), (None, None),
        )
        tCsC, tCgC = cute.nvgpu.cpasync.tma_partition(
            c_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(gC, 0, 2),
        )

        with cute.arch.elect_one():
            cute.copy(c_tma, tCsC, tCgC[(None, tile_N, tile_M)])

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def _auto_tiles(cls, page_size, inner_depth, elem_bytes=2):
        """Compute largest (tile_M, tile_N, tile_K) that fit in page_size.

        Constraints:
          - inner_depth * (tile_M + tile_N) * tile_K * elem <= page_size  (AB)
          - tile_M * tile_N * elem <= page_size  (C epilogue)
          - tile_K must be a multiple of 16
        """
        tile_K = 32
        for tile_M, tile_N in [(128, 64), (64, 64), (64, 32), (32, 32)]:
            ab = inner_depth * (tile_M + tile_N) * tile_K * elem_bytes
            c = tile_M * tile_N * elem_bytes
            if ab <= page_size and c <= page_size:
                return tile_M, tile_N, tile_K
        return 32, 32, 16

    @classmethod
    def schedule_forward(cls, tile_sizes=None, inner_depth=1, page_size=16384, **tensors):
        """Schedule forward GEMM.

        Accepts tile_sizes with M, N, K keys. K is the inner K-block size
        (not a framework tile dimension). Auto-computed from page_size when
        not specified.

        Args:
            inner_depth: Pipeline stages for K-block double buffering.
                1 = single buffer (default), 2 = double buffer.
                With depth > 1, DMA prefills multiple buffers before compute
                starts, overlapping loads with MMA. Requires more smem:
                inner_depth * (A_tile + B_tile) must fit in page_size.
            page_size: Shared memory page size in bytes. Default 16384 (16KB).
        """
        ts = dict(tile_sizes or {})
        if "M" not in ts or "N" not in ts:
            a = tensors.get('a')
            elem_bytes = a.element_size() if a is not None else 2
            auto_M, auto_N, auto_K = cls._auto_tiles(
                page_size, inner_depth, elem_bytes)
            ts.setdefault("M", auto_M)
            ts.setdefault("N", auto_N)
            ts.setdefault("K", auto_K)
        tile_K = ts.pop("K", 32)
        scheduled = cls._schedule_single(tile_sizes=ts, **tensors)
        scheduled.static_dims["tile_K"] = tile_K
        scheduled.static_dims["inner_depth"] = inner_depth
        scheduled.static_dims["page_size"] = page_size
        return [scheduled]

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for scheduled GemmOps."""
        from machete.megakernel import MegakernelConfig
        page_size = ops[0].static_dims.get('page_size', 16384)
        return MegakernelConfig(page_size=page_size)

    @classmethod
    def schedule_backward(cls, tile_sizes=None, inner_depth=1, **tensors):
        """Schedule GEMM backward as forward-equivalent ops.

        dA[M,K] = dout[M,N] @ B[N,K]  (contracts over N)
        dB[N,K] = dout^T[N,M] @ A[M,K]  (contracts over M)

        Each gradient is a standard forward GEMM with transposed inputs.
        No output zeroing needed (regular TMA store, not atomic add).
        """
        dout, a, b = tensors['dout'], tensors['a'], tensors['b']
        da, db = tensors.get('da'), tensors.get('db')
        ts = tile_sizes or {}
        tile_m = ts.get("M", 64)
        tile_n = ts.get("N", 32)
        tile_k = ts.get("K", 32)
        ops = []

        if da is not None:
            # dA(M,K) = dout(M,N) @ B(N,K)
            # GemmOp: C(M,N) = A(M,K) @ B(N,K)^T
            # Map: A→dout(M,N), B→B^T(K,N), C→dA(M,K)
            # GemmOp dims: M→M, N→K, K(inner)→N
            b_t = b.t().contiguous()  # (N,K) -> (K,N)
            ops.extend(cls.schedule_forward(
                a=dout, b=b_t, c=da,
                tile_sizes={"M": tile_m, "N": tile_k, "K": tile_n},
                inner_depth=inner_depth,
            ))

        if db is not None:
            # dB(N,K) = dout^T(N,M) @ A(M,K)
            # Map: A→dout^T(N,M), B→A^T(K,M), C→dB(N,K)
            # GemmOp dims: M→N, N→K, K(inner)→M
            dout_t = dout.t().contiguous()  # (M,N) -> (N,M)
            a_t = a.t().contiguous()  # (M,K) -> (K,M)
            ops.extend(cls.schedule_forward(
                a=dout_t, b=a_t, c=db,
                tile_sizes={"M": tile_n, "N": tile_k, "K": tile_m},
                inner_depth=inner_depth,
            ))

        return ops
