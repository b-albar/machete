# Copyright (c) 2025, Machete Authors
"""
GEMM Op for the Megakernel.

Computes C[M,N] = A[M,K] @ B[N,K]^T  (i.e., standard matmul with B pre-transposed).

Note: B is expected in (N, K) layout (K contiguous). If you have B in standard
PyTorch (K, N) layout, pass b.t().contiguous() when scheduling.

Architecture (SM_90+ / Hopper+):
    - Tensor core warp MMA: MmaF16BF16Op(16, 8, 16)
    - Supports fp16 and bf16 inputs, fp32 accumulation
    - K is a tile dimension — all (M, N, K) tiles are independent
    - TMA for G->S loads, TMA store_add (atomic) for S->G reduction
    - LdMatrix for warp-cooperative smem->register reads

Pipelined phases:
    load:    TMA G->S of A[tile_M, tile_K] + B[tile_N, tile_K]
    compute: LdMatrix smem->reg, single MMA pass. Epilogue: R->S.
    store:   TMA store_add S->G of C_partial[tile_M, tile_N]

Page layout (16KB):
    [A: tile_K x tile_M] [B: tile_K x tile_N]
    Epilogue: [C: tile_M x tile_N]  (reuses page from offset 0)

Output C must be zeroed before kernel launch. Each K tile atomically
adds its partial result via TMA store_add.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32


from machete.megakernel.ops import Op
from machete.megakernel.interpreter import mbarrier_arrive_expect_tx, named_barrier_sync
from machete.megakernel.paged_memory import PAGE_SIZE


class GemmOp(Op):
    """GEMM operation for the megakernel framework.

    Computes C[M,N] = A[M,K] @ B[N,K]^T using tensor core MMA.
    B must be in (N, K) layout with K contiguous.

    All (M, N, K) tiles are independent. K reduction is via TMA atomic add.
    Uses LdMatrix for warp-cooperative smem->register reads.
    """

    reads = {
        "a": (None, ("M", "K")),
        "b": (None, ("N", "K")),
    }
    writes = {"c": (None, ("M", "N"))}
    tile = ("M", "N", "K")

    tma_loads = {"a", "b"}
    tma_reduce_stores = {"c"}

    def __init__(self, **config):
        super().__init__(**config)

        if self.a_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            raise ValueError(f"GemmOp requires fp16 or bf16 input, got {self.a_dtype}")

        self.a_tile_bytes = self.tile_size_M * self.tile_size_K * self.elem_bytes
        self.b_tile_bytes = self.tile_size_N * self.tile_size_K * self.elem_bytes
        self.c_tile_bytes = self.tile_size_M * self.tile_size_N * self.elem_bytes

        # A+B must fit during load/compute; C reuses page during store
        ab_bytes = self.a_tile_bytes + self.b_tile_bytes
        assert max(ab_bytes, self.c_tile_bytes) <= PAGE_SIZE, (
            f"GemmOp: max(A+B={ab_bytes}B, C={self.c_tile_bytes}B) "
            f"exceeds PAGE_SIZE ({PAGE_SIZE}B)."
        )

        assert self.tile_size_K >= 16 and self.tile_size_K % 16 == 0, (
            f"GemmOp: tile_size_K={self.tile_size_K} must be >= 16 and a multiple of 16."
        )

        self.num_mma_warps = self.threads_per_row // 32
        self.num_mma_threads = self.num_mma_warps * 32

    # =========================================================================
    # Forward Load (TMA G->S): A[tile_M, tile_K] + B[tile_N, tile_K]
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_M, tile_N, tile_K,
             a_tma, a_tma_gmem, b_tma, b_tma_gmem, work_mbar):
        """TMA load of A and B tiles for one (M, N, K) tile.

        TMA transposes gmem: A(M,K) -> smem (K,M), B(N,K) -> smem (K,N).
        """
        # --- A tile: TMA G->S ---
        sA = cute.make_tensor(
            cute.make_ptr(self.a_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.tile_size_K, self.tile_size_M)),
        )
        gA = cute.local_tile(
            a_tma_gmem, (self.tile_size_K, self.tile_size_M), (None, None),
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            a_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )

        # --- B tile: TMA G->S ---
        sB = cute.make_tensor(
            cute.make_ptr(self.b_dtype, page_ptr + Int32(self.a_tile_bytes),
                          cute.AddressSpace.smem),
            cute.make_layout((self.tile_size_K, self.tile_size_N)),
        )
        gB = cute.local_tile(
            b_tma_gmem, (self.tile_size_K, self.tile_size_N), (None, None),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            b_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )

        # Signal mbarrier with expected bytes, then issue TMA copies
        nbytes = Int32(self.a_tile_bytes + self.b_tile_bytes)
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)

        # TMA copy: (None=TMA modes, K_tile, M/N_tile)
        cute.copy(a_tma, tAgA[(None, tile_K, tile_M)], tAsA, tma_bar_ptr=mbar_ptr)
        cute.copy(b_tma, tBgB[(None, tile_K, tile_N)], tBsB, tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Forward Compute: Single MMA pass over tile_K with LdMatrix
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_M, tile_N, tile_K):
        """GEMM compute: tensor core MMA with fp32 accumulation.

        Reads A/B from smem via LdMatrix (warp-cooperative 128-bit loads).
        Single pass over tile_K (no K-loop). Writes partial C to smem
        for atomic store_add in the store phase.
        """
        tidx = cute.arch.thread_idx()[0]

        # --- Build tiled MMA ---
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(self.a_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        # --- Smem tensors (compute layout: M/N mode 0, K mode 1, K contiguous) ---
        sA = cute.make_tensor(
            cute.make_ptr(self.a_dtype, page_ptr, cute.AddressSpace.smem,
                          assumed_align=128),
            cute.make_layout((self.tile_size_M, self.tile_size_K),
                             stride=(self.tile_size_K, 1)),
        )
        sB = cute.make_tensor(
            cute.make_ptr(self.b_dtype, page_ptr + Int32(self.a_tile_bytes),
                          cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout((self.tile_size_N, self.tile_size_K),
                             stride=(self.tile_size_K, 1)),
        )

        # --- MMA partitions and register fragments ---
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        # --- LdMatrix tiled copies (warp-cooperative smem reads) ---
        smem_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False, num_matrices=4), self.a_dtype)
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)

        smem_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False, num_matrices=4), self.b_dtype)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        # Partition smem for LdMatrix reads
        tAsA_ld = smem_thr_copy_A.partition_S(sA)
        tBsB_ld = smem_thr_copy_B.partition_S(sB)

        # Retile register fragments to match LdMatrix layout
        tCrA_ld = smem_thr_copy_A.retile(tCrA)
        tCrB_ld = smem_thr_copy_B.retile(tCrB)

        # --- Init fp32 accumulator ---
        acc = cute.make_fragment(
            tiled_mma.partition_shape_C((self.tile_size_M, self.tile_size_N)),
            Float32,
        )
        acc.fill(0.0)

        # --- MMA over tile_K in 16-element k-blocks (LdMatrix reads) ---
        for k_block in cutlass.range_constexpr(self.tile_size_K // 16):
            cute.copy(smem_tiled_copy_A,
                      tAsA_ld[None, None, k_block],
                      tCrA_ld[None, None, k_block])
            cute.copy(smem_tiled_copy_B,
                      tBsB_ld[None, None, k_block],
                      tCrB_ld[None, None, k_block])
            cute.gemm(tiled_mma, acc,
                      tCrA[None, None, k_block],
                      tCrB[None, None, k_block], acc)

        # --- Epilogue: R->S (write partial C to smem for store_add) ---
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
    # Forward Store (S->G): TMA store_add C partial from smem to global
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_M, tile_N, tile_K, c_tma, c_tma_gmem):
        """TMA atomic add store of partial C from shared to global memory.

        Compute epilogue writes C as (tile_M, tile_N) row-major stride (tile_N, 1).
        TMA sees smem as (tile_N, tile_M) col-major — same physical layout.
        TMA handles boundary conditions for non-divisible M/N.
        Multiple K tiles atomically accumulate into the same C[M,N] output.
        """
        # Smem view: (tile_N, tile_M) col-major matches compute's row-major write
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
    # Backward: reuse forward (GEMM backward = two forward GEMMs)
    # =========================================================================

    backward_reads = {
        "dout": (None, ("M", "N")),
        "a": (None, ("M", "K")),
        "b": (None, ("N", "K")),
    }
    backward_writes = {
        "da": (None, ("M", "K")),
        "db": (None, ("N", "K")),
    }

    backward_load = load
    backward_compute = compute
    backward_store = store

    @classmethod
    def schedule_backward(cls, tile_sizes=None, **tensors):
        """Schedule GEMM backward as forward-equivalent ops.

        dA[M,K] = dout[M,N] @ B[N,K]  (contracts over N)
        dB[N,K] = dout^T[N,M] @ A[M,K]  (contracts over M)

        Each gradient is a standard GEMM with transposed inputs.
        Output tensors are zeroed (TMA store_add accumulation).
        """
        dout, a, b = tensors['dout'], tensors['a'], tensors['b']
        da, db = tensors.get('da'), tensors.get('db')
        ts = tile_sizes or {}
        tile_m = ts.get("M", 64)
        tile_n = ts.get("N", 32)
        tile_k = ts.get("K", 32)
        ops = []

        if da is not None:
            # dA = dout(M,N) @ B(N,K) -> GemmOp(a=dout, b=B, c=dA)
            # Forward GemmOp: C(M,N) = A(M,K) @ B(N,K)^T
            # Here: dA(M,K) = dout(M,N) @ B(N,K) = dout(M,N) @ B(N,K)
            # Map: A->dout(M,N), B->B(N,K) transposed->B^T(K,N), C->dA(M,K)
            # GemmOp dims: M->M, K->N (contraction), N->K (output col)
            da.zero_()
            b_t = b.t().contiguous()  # (N,K) -> (K,N)
            ops.append(cls._schedule_single(
                a=dout, b=b_t, c=da,
                tile_sizes={"M": tile_m, "N": tile_k, "K": tile_n},
            ))

        if db is not None:
            # dB = dout^T(N,M) @ A(M,K) -> GemmOp(a=dout^T, b=A^T, c=dB)
            # Map: A->dout^T(N,M), B->A(M,K) transposed->A^T(K,M), C->dB(N,K)
            # GemmOp dims: M->N, K->M (contraction), N->K (output col)
            db.zero_()
            dout_t = dout.t().contiguous()  # (M,N) -> (N,M)
            a_t = a.t().contiguous()  # (M,K) -> (K,M)
            ops.append(cls._schedule_single(
                a=dout_t, b=a_t, c=db,
                tile_sizes={"M": tile_n, "N": tile_k, "K": tile_m},
            ))

        return ops
