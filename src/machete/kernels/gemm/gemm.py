# Copyright (c) 2025, Machete Authors
"""
GEMM Op for the Megakernel.

Computes C[M,N] = A[M,K] @ B[N,K]^T  (i.e., standard matmul with B pre-transposed).

Note: B is expected in (N, K) layout (K contiguous). If you have B in standard
PyTorch (K, N) layout, pass b.t().contiguous() when scheduling.

Architecture (SM_120 / Blackwell GeForce):
    - Tensor core warp MMA: MmaF16BF16Op(16, 8, 16)
    - Supports fp16 and bf16 inputs, fp32 accumulation
    - K dimension is an internal reduction loop (not a megakernel tile dim)
    - TMA for G->S loads (K=0), 2-stage cp.async pipeline for K>0

Pipelined phases:
    load:    TMA G->S of A[tile_M, tile_K] + B[tile_N, tile_K] for K=0
    compute: 2-stage K-loop: cp.async for K+1 overlaps with MMA for K.
             Double-buffered A/B in smem. Epilogue: R->S to page.
    store:   TMA S->G of C[tile_M, tile_N]

Page layout (16KB, double-buffered when num_k_steps > 1):
    buf[0]: [A0: tile_K x tile_M] [B0: tile_K x tile_N]
    buf[1]: [A1: tile_K x tile_M] [B1: tile_K x tile_N]
    Epilogue: [C: tile_M x tile_N]  (reuses page from offset 0)
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32


from machete.megakernel.ops import Op
from machete.megakernel.interpreter import mbarrier_arrive_expect_tx, named_barrier_sync
from machete.megakernel.paged_memory import PAGE_SIZE


class GemmOp(Op):
    """GEMM operation for the megakernel framework.

    Computes C[M,N] = A[M,K] @ B[N,K]^T using tensor core MMA (SM_120).
    B must be in (N, K) layout with K contiguous.

    Tensor declarations:
        a: (M, K) — input matrix A (fp16/bf16)
        b: (N, K) — input matrix B transposed (fp16/bf16)
        c: (M, N) — output matrix C (fp16/bf16)

    Tiling:
        tile over (M, N). K is reduced internally in compute().

    Pipeline:
        K=0 loaded by TMA (load phase). K>0 uses 2-stage cp.async pipeline
        in compute: copy K+1 overlaps with MMA K. Double-buffered A/B in smem.
    """

    reads = {
        "a": (None, ("M", "K")),
        "b": (None, ("N", "K")),
    }
    writes = {"c": (None, ("M", "N"))}
    tile = ("M", "N")

    tma_loads = {"a", "b"}
    tma_stores = {"c"}

    NUM_STAGES = 2  # Double-buffered cp.async pipeline for K>0

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        """Return custom TMA tile shape for A and B.

        K is not a megakernel tile dim but we only want tile_K elements
        per TMA load (not the full K extent). Compute tile_K from PAGE_SIZE.
        Uses double-buffer formula when multiple K steps are needed.
        """
        elem_bytes = 2  # fp16/bf16
        tile_M = tile_sizes["M"]
        tile_N = tile_sizes["N"]
        K = static_dims["K"]
        # Try single-buffer first
        max_tile_k = PAGE_SIZE // ((tile_M + tile_N) * elem_bytes)
        tile_K = min(max_tile_k, K)
        tile_K = (tile_K // 16) * 16
        num_k_steps = (K + tile_K - 1) // tile_K
        if num_k_steps > 1 and cls.NUM_STAGES > 1:
            # Need double-buffering: recompute with space for NUM_STAGES buffers
            max_tile_k = PAGE_SIZE // (cls.NUM_STAGES * (tile_M + tile_N) * elem_bytes)
            tile_K = min(max_tile_k, K)
            tile_K = (tile_K // 16) * 16

        if tensor_name == "a":
            return (tile_M, tile_K)
        elif tensor_name == "b":
            return (tile_N, tile_K)
        elif tensor_name == "c":
            return (tile_M, tile_N)
        raise ValueError(f"Unknown TMA tensor: {tensor_name}")

    def __init__(self, **config):
        super().__init__(**config)

        # Element size (fp16 or bf16 = 2 bytes)
        if self.a_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            raise ValueError(f"GemmOp requires fp16 or bf16 input, got {self.a_dtype}")

        # Compute tile_K from PAGE_SIZE constraint.
        # Try single-buffer first; if multiple K steps needed, use double-buffer.
        row_bytes = (self.tile_size_M + self.tile_size_N) * self.elem_bytes
        max_tile_k = PAGE_SIZE // row_bytes
        self.tile_K = min(max_tile_k, self.K)
        self.tile_K = (self.tile_K // 16) * 16
        self.num_k_steps = (self.K + self.tile_K - 1) // self.tile_K

        if self.num_k_steps > 1 and self.NUM_STAGES > 1:
            # Need double-buffering: recompute with space for NUM_STAGES buffers
            max_tile_k = PAGE_SIZE // (self.NUM_STAGES * row_bytes)
            self.tile_K = min(max_tile_k, self.K)
            self.tile_K = (self.tile_K // 16) * 16
            self.num_k_steps = (self.K + self.tile_K - 1) // self.tile_K

        assert self.tile_K >= 16, (
            f"GemmOp: tile_K={self.tile_K} < 16. Reduce tile_size_M or tile_size_N."
        )

        # Smem byte sizes (TMA layout: (tile_K, tile_M) and (tile_K, tile_N))
        self.a_tile_bytes = self.tile_size_M * self.tile_K * self.elem_bytes
        self.b_tile_bytes = self.tile_size_N * self.tile_K * self.elem_bytes
        self.ab_buffer_bytes = self.a_tile_bytes + self.b_tile_bytes
        assert self.ab_buffer_bytes <= PAGE_SIZE, (
            f"GemmOp: A+B tile ({self.ab_buffer_bytes}B) exceeds PAGE_SIZE ({PAGE_SIZE}B)."
        )
        if self.num_k_steps > 1:
            assert self.NUM_STAGES * self.ab_buffer_bytes <= PAGE_SIZE, (
                f"GemmOp: {self.NUM_STAGES}x A+B ({self.NUM_STAGES * self.ab_buffer_bytes}B) "
                f"exceeds PAGE_SIZE ({PAGE_SIZE}B)."
            )

        # C tile must also fit (reuses page after last K step)
        self.c_tile_bytes = self.tile_size_M * self.tile_size_N * self.elem_bytes
        assert self.c_tile_bytes <= PAGE_SIZE, (
            f"GemmOp: C tile ({self.c_tile_bytes}B) exceeds PAGE_SIZE ({PAGE_SIZE}B)."
        )

        # MMA warp count (threads_per_row already excludes the DMA warp)
        self.num_mma_warps = self.threads_per_row // 32
        self.num_mma_threads = self.num_mma_warps * 32

        # Thread layout for cooperative async G->S copy (K>0 steps)
        # cp.async min 32 bits = 2 fp16/bf16 elems per copy
        from math import gcd
        self.copy_vec = 2
        k_logical = self.tile_K // self.copy_vec
        k_thr = gcd(k_logical, self.num_mma_threads)
        while k_thr > 0:
            m_thr = self.num_mma_threads // k_thr
            if (k_logical % k_thr == 0
                    and self.num_mma_threads % k_thr == 0
                    and self.tile_size_M % m_thr == 0):
                break
            k_thr -= 1
        self.copy_k_threads = k_thr
        self.copy_m_threads = self.num_mma_threads // k_thr

    # =========================================================================
    # Forward Load (TMA G->S): first K-chunk
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_M, tile_N, a_tma, a_tma_gmem, b_tma, b_tma_gmem, work_mbar):
        """TMA load of A and B tiles for K=0.

        TMA transposes gmem: A(M,K) -> (K,M), B(N,K) -> (K,N).
        Smem layout: A at offset 0 as (tile_K, tile_M), B after A as (tile_K, tile_N).
        """
        # --- A tile: TMA G->S ---
        sA = cute.make_tensor(
            cute.make_ptr(self.a_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.tile_K, self.tile_size_M)),
        )
        gA = cute.local_tile(
            a_tma_gmem, (self.tile_K, self.tile_size_M), (None, None),
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            a_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )

        # --- B tile: TMA G->S ---
        sB = cute.make_tensor(
            cute.make_ptr(self.b_dtype, page_ptr + Int32(self.a_tile_bytes), cute.AddressSpace.smem),
            cute.make_layout((self.tile_K, self.tile_size_N)),
        )
        gB = cute.local_tile(
            b_tma_gmem, (self.tile_K, self.tile_size_N), (None, None),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            b_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )

        # Signal mbarrier with expected bytes (elect_one)
        # then issue TMA copies (all threads for warp convergence)
        nbytes = Int32(self.a_tile_bytes + self.b_tile_bytes)
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)

        # TMA copy: (None=TMA modes, K_tile=0, M/N_tile=tile index)
        cute.copy(a_tma, tAgA[(None, 0, tile_M)], tAsA, tma_bar_ptr=mbar_ptr)
        cute.copy(b_tma, tBgB[(None, 0, tile_N)], tBsB, tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Forward Compute: K-loop with tensor core MMA
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_M, tile_N, a, b, c):
        """GEMM compute: tensor core MMA with fp32 accumulation.

        2-stage pipelined K-loop:
            K=0: A/B in buf[0] from TMA load. Issue cp.async for K=1 → buf[1].
            K>0: Wait for copy, issue next copy (overlapped with MMA), MMA.
        Epilogue: R->S via partition_C to smem page.
        """
        row_start_m = tile_M * self.tile_size_M
        col_start_n = tile_N * self.tile_size_N
        tidx = cute.arch.thread_idx()[0]

        # --- Build tiled MMA ---
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(self.a_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        # --- Buffer 0 smem tensors (K=0, loaded by TMA) ---
        sA_0 = cute.make_tensor(
            cute.make_ptr(self.a_dtype, page_ptr, cute.AddressSpace.smem,
                          assumed_align=128),
            cute.make_layout((self.tile_size_M, self.tile_K), stride=(self.tile_K, 1)),
        )
        sB_0 = cute.make_tensor(
            cute.make_ptr(self.b_dtype, page_ptr + Int32(self.a_tile_bytes),
                          cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout((self.tile_size_N, self.tile_K), stride=(self.tile_K, 1)),
        )

        # --- Partition buf 0 and create register fragments ---
        tCsA_0 = thr_mma.partition_A(sA_0)
        tCsB_0 = thr_mma.partition_B(sB_0)
        tCrA = tiled_mma.make_fragment_A(tCsA_0)
        tCrB = tiled_mma.make_fragment_B(tCsB_0)

        # --- Init fp32 accumulator ---
        acc = cute.make_fragment(
            tiled_mma.partition_shape_C((self.tile_size_M, self.tile_size_N)),
            Float32,
        )
        acc.fill(0.0)

        # --- Tiled copy setup (cp.async G->S, shared across K steps) ---
        copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), self.a_dtype,
            num_bits_per_copy=self.a_dtype.width * self.copy_vec)
        copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), self.b_dtype,
            num_bits_per_copy=self.b_dtype.width * self.copy_vec)
        copy_thr = cute.make_layout(
            (self.copy_m_threads, self.copy_k_threads),
            stride=(self.copy_k_threads, 1),
        )
        copy_val = cute.make_layout((1, self.copy_vec))

        tiled_copy_A = cute.make_tiled_copy_tv(copy_atom_A, copy_thr, copy_val)
        tiled_copy_B = cute.make_tiled_copy_tv(copy_atom_B, copy_thr, copy_val)
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)

        # Identity tensors for predication (constant across K steps)
        id_A = cute.make_identity_tensor((self.tile_size_M, self.tile_K))
        id_B = cute.make_identity_tensor((self.tile_size_N, self.tile_K))
        copy_cA = thr_copy_A.partition_S(id_A)
        copy_cB = thr_copy_B.partition_S(id_B)

        # Pre-compute predicate shape from buf 0 partition (same for all buffers)
        copy_dA_0 = thr_copy_A.partition_D(sA_0)
        copy_dB_0 = thr_copy_B.partition_D(sB_0)

        # M/N boundaries (constant across K steps)
        m_bound = Int32(self.M) - row_start_m
        n_bound = Int32(self.N) - col_start_n

        # === PROLOGUE: Issue cp.async for K=1 → buf[1] ===
        if self.num_k_steps > 1:
            sA_1 = cute.make_tensor(
                cute.make_ptr(self.a_dtype,
                              page_ptr + Int32(self.ab_buffer_bytes),
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.tile_size_M, self.tile_K),
                                 stride=(self.tile_K, 1)),
            )
            sB_1 = cute.make_tensor(
                cute.make_ptr(self.b_dtype,
                              page_ptr + Int32(self.ab_buffer_bytes + self.a_tile_bytes),
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.tile_size_N, self.tile_K),
                                 stride=(self.tile_K, 1)),
            )
            copy_dA_1 = thr_copy_A.partition_D(sA_1)
            copy_dB_1 = thr_copy_B.partition_D(sB_1)

            gA_1 = cute.make_tensor(
                (a.iterator + row_start_m * self.K + self.tile_K).align(4),
                cute.make_layout((self.tile_size_M, self.tile_K),
                                 stride=(self.K, 1)),
            )
            gB_1 = cute.make_tensor(
                (b.iterator + col_start_n * self.K + self.tile_K).align(4),
                cute.make_layout((self.tile_size_N, self.tile_K),
                                 stride=(self.K, 1)),
            )

            k_bound_1 = Int32(self.K - self.tile_K)
            pred_A = cute.make_rmem_tensor(cute.make_layout((
                copy_dA_0.shape[0][1],
                cute.size(copy_dA_0, mode=[1]),
                cute.size(copy_dA_0, mode=[2]),
            )), cutlass.Boolean)
            for v in range(pred_A.shape[0]):
                for m in range(pred_A.shape[1]):
                    for k in range(pred_A.shape[2]):
                        pred_A[v, m, k] = cute.elem_less(
                            copy_cA[(0, v), m, k], (m_bound, k_bound_1))
            pred_B = cute.make_rmem_tensor(cute.make_layout((
                copy_dB_0.shape[0][1],
                cute.size(copy_dB_0, mode=[1]),
                cute.size(copy_dB_0, mode=[2]),
            )), cutlass.Boolean)
            for v in range(pred_B.shape[0]):
                for n in range(pred_B.shape[1]):
                    for k in range(pred_B.shape[2]):
                        pred_B[v, n, k] = cute.elem_less(
                            copy_cB[(0, v), n, k], (n_bound, k_bound_1))

            cute.copy(tiled_copy_A, thr_copy_A.partition_S(gA_1),
                      copy_dA_1, pred=pred_A)
            cute.copy(tiled_copy_B, thr_copy_B.partition_S(gB_1),
                      copy_dB_1, pred=pred_B)
            cute.arch.cp_async_commit_group()

        # === K=0: MMA from buffer 0 (TMA loaded) ===
        for k_block in cutlass.range_constexpr(self.tile_K // 16):
            cute.autovec_copy(tCsA_0[None, None, k_block],
                              tCrA[None, None, k_block])
            cute.autovec_copy(tCsB_0[None, None, k_block],
                              tCrB[None, None, k_block])
            cute.gemm(tiled_mma, acc,
                      tCrA[None, None, k_block],
                      tCrB[None, None, k_block], acc)

        # === K>0: Pipelined loop ===
        # Pattern: wait for K copy, issue K+1 copy (overlaps with MMA), MMA K.
        for k_step in range(1, self.num_k_steps, 1, unroll=1):
            # Wait for this step's copy to complete
            cute.arch.cp_async_wait_group(0)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # Issue cp.async for k_step+1 if more K steps remain
            if k_step + Int32(1) < Int32(self.num_k_steps):
                next_k_offset = (k_step + Int32(1)) * Int32(self.tile_K)
                next_buf = ((k_step + Int32(1)) % Int32(2)) * Int32(self.ab_buffer_bytes)
                sA_next = cute.make_tensor(
                    cute.make_ptr(self.a_dtype, page_ptr + next_buf,
                                  cute.AddressSpace.smem, assumed_align=128),
                    cute.make_layout((self.tile_size_M, self.tile_K),
                                     stride=(self.tile_K, 1)),
                )
                sB_next = cute.make_tensor(
                    cute.make_ptr(self.b_dtype,
                                  page_ptr + next_buf + Int32(self.a_tile_bytes),
                                  cute.AddressSpace.smem, assumed_align=128),
                    cute.make_layout((self.tile_size_N, self.tile_K),
                                     stride=(self.tile_K, 1)),
                )
                gA_next = cute.make_tensor(
                    (a.iterator + row_start_m * self.K + next_k_offset).align(4),
                    cute.make_layout((self.tile_size_M, self.tile_K),
                                     stride=(self.K, 1)),
                )
                gB_next = cute.make_tensor(
                    (b.iterator + col_start_n * self.K + next_k_offset).align(4),
                    cute.make_layout((self.tile_size_N, self.tile_K),
                                     stride=(self.K, 1)),
                )
                k_bound_next = Int32(self.K) - next_k_offset
                pred_An = cute.make_rmem_tensor(cute.make_layout((
                    copy_dA_0.shape[0][1],
                    cute.size(copy_dA_0, mode=[1]),
                    cute.size(copy_dA_0, mode=[2]),
                )), cutlass.Boolean)
                for v in range(pred_An.shape[0]):
                    for m in range(pred_An.shape[1]):
                        for k in range(pred_An.shape[2]):
                            pred_An[v, m, k] = cute.elem_less(
                                copy_cA[(0, v), m, k], (m_bound, k_bound_next))
                pred_Bn = cute.make_rmem_tensor(cute.make_layout((
                    copy_dB_0.shape[0][1],
                    cute.size(copy_dB_0, mode=[1]),
                    cute.size(copy_dB_0, mode=[2]),
                )), cutlass.Boolean)
                for v in range(pred_Bn.shape[0]):
                    for n in range(pred_Bn.shape[1]):
                        for k in range(pred_Bn.shape[2]):
                            pred_Bn[v, n, k] = cute.elem_less(
                                copy_cB[(0, v), n, k], (n_bound, k_bound_next))
                cute.copy(tiled_copy_A, thr_copy_A.partition_S(gA_next),
                          thr_copy_A.partition_D(sA_next), pred=pred_An)
                cute.copy(tiled_copy_B, thr_copy_B.partition_S(gB_next),
                          thr_copy_B.partition_D(sB_next), pred=pred_Bn)
                cute.arch.cp_async_commit_group()

            # MMA from current buffer (buf[k_step % 2])
            cur_buf = (k_step % Int32(2)) * Int32(self.ab_buffer_bytes)
            sA_cur = cute.make_tensor(
                cute.make_ptr(self.a_dtype, page_ptr + cur_buf,
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.tile_size_M, self.tile_K),
                                 stride=(self.tile_K, 1)),
            )
            sB_cur = cute.make_tensor(
                cute.make_ptr(self.b_dtype,
                              page_ptr + cur_buf + Int32(self.a_tile_bytes),
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.tile_size_N, self.tile_K),
                                 stride=(self.tile_K, 1)),
            )
            tCsA_k = thr_mma.partition_A(sA_cur)
            tCsB_k = thr_mma.partition_B(sB_cur)
            for k_block in cutlass.range_constexpr(self.tile_K // 16):
                cute.autovec_copy(tCsA_k[None, None, k_block],
                                  tCrA[None, None, k_block])
                cute.autovec_copy(tCsB_k[None, None, k_block],
                                  tCrB[None, None, k_block])
                cute.gemm(tiled_mma, acc,
                          tCrA[None, None, k_block],
                          tCrB[None, None, k_block], acc)

        # --- Epilogue: R->S (write C to smem page for store phase) ---
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
    # Forward Store (S->G): C tile from smem to global
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_M, tile_N, c_tma, c_tma_gmem):
        """TMA store C from shared to global memory.

        Compute epilogue writes C as (tile_M, tile_N) row-major stride (tile_N, 1).
        TMA sees smem as (tile_N, tile_M) col-major — same physical layout.
        TMA handles boundary conditions for non-divisible M/N.
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
