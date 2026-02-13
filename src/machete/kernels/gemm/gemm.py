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
    - TMA for G->S loads (K=0), cooperative G->S for K>0

Pipelined phases:
    load:    TMA G->S of A[tile_M, tile_K] + B[tile_N, tile_K] for K=0
    compute: K-loop with tensor core MMA (fp32 acc). K>0 cooperative G->S.
             Epilogue: R->S via StMatrix to page.
    store:   TMA S->G of C[tile_M, tile_N]

Page layout (16KB):
    Load/compute: [A: tile_K x tile_M elems] [B: tile_K x tile_N elems]
    Epilogue:     [C: tile_M x tile_N elems]  (reuses same page)
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
        """Return custom TMA tile shape for A and B.

        K is not a megakernel tile dim but we only want tile_K elements
        per TMA load (not the full K extent). Compute tile_K from PAGE_SIZE.
        """
        elem_bytes = 2  # fp16/bf16
        tile_M = tile_sizes["M"]
        tile_N = tile_sizes["N"]
        K = static_dims["K"]
        max_tile_k = PAGE_SIZE // ((tile_M + tile_N) * elem_bytes)
        tile_K = min(max_tile_k, K)
        tile_K = (tile_K // 16) * 16  # Multiple of 16 for MMA K-block

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

        # Compute tile_K from PAGE_SIZE constraint:
        # A[tile_M, tile_K] + B[tile_N, tile_K] must fit in PAGE_SIZE
        max_tile_k = PAGE_SIZE // ((self.tile_size_M + self.tile_size_N) * self.elem_bytes)
        self.tile_K = min(max_tile_k, self.K)
        self.tile_K = (self.tile_K // 16) * 16  # Multiple of 16 for MMA K-block
        assert self.tile_K >= 16, (
            f"GemmOp: tile_K={self.tile_K} < 16. Reduce tile_size_M or tile_size_N."
        )

        self.num_k_steps = (self.K + self.tile_K - 1) // self.tile_K

        # Smem byte sizes (TMA layout: (tile_K, tile_M) and (tile_K, tile_N))
        self.a_tile_bytes = self.tile_size_M * self.tile_K * self.elem_bytes
        self.b_tile_bytes = self.tile_size_N * self.tile_K * self.elem_bytes
        assert self.a_tile_bytes + self.b_tile_bytes <= PAGE_SIZE, (
            f"GemmOp: A+B tile ({self.a_tile_bytes + self.b_tile_bytes}B) "
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

        # Number of K elements each thread handles in cooperative loads
        self.num_k_per_thread = (self.tile_K + 31) // 32

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

        K=0: A/B loaded to smem by TMA (DMA warp).
        K>0: MMA warps cooperatively load A/B from global to smem.
        All K steps: partition_A/B -> make_fragment_A/B -> cute.gemm.
        Epilogue: R->S via StMatrix to page (for store phase).
        """
        row_start_m = tile_M * self.tile_size_M
        col_start_n = tile_N * self.tile_size_N
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.num_mma_warps

        # --- Build tiled MMA ---
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(self.a_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((num_warps, 1, 1)),
            permutation_mnk=(num_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        # --- Smem tensors (views for MMA partition_A/B) ---
        # TMA wrote (tile_K, tile_M) col-major: smem[k,m] at offset k + m*tile_K.
        # Create (tile_M, tile_K) view: sA[m,k] -> offset m*tile_K + k.
        sA = cute.make_tensor(
            cute.make_ptr(self.a_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.tile_size_M, self.tile_K), stride=(self.tile_K, 1)),
        )
        sB = cute.make_tensor(
            cute.make_ptr(self.b_dtype, page_ptr + Int32(self.a_tile_bytes), cute.AddressSpace.smem),
            cute.make_layout((self.tile_size_N, self.tile_K), stride=(self.tile_K, 1)),
        )

        # --- Partition smem and create register fragments ---
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        # --- Init fp32 accumulator ---
        acc = cute.make_fragment(
            tiled_mma.partition_shape_C((self.tile_size_M, self.tile_size_N)),
            Float32,
        )
        acc.fill(0.0)

        # --- K=0: explicit S->R copy then MMA (TMA loaded smem) ---
        for k_block in cutlass.range_constexpr(self.tile_K // 16):
            cute.autovec_copy(tCsA[None, None, k_block], tCrA[None, None, k_block])
            cute.autovec_copy(tCsB[None, None, k_block], tCrB[None, None, k_block])
            cute.gemm(tiled_mma, acc,
                      tCrA[None, None, k_block],
                      tCrB[None, None, k_block], acc)

        # --- K>0: cooperative G->S load, then MMA ---
        for k_step in range(1, self.num_k_steps, 1, unroll=1):
            k_offset = k_step * self.tile_K

            # Sync MMA warps before overwriting smem
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # Cooperative G->S load of A[tile_M, tile_K]
            # Write to smem matching col-major (tile_K, tile_M): offset = m*tile_K + k
            for m in range(warp_idx, self.tile_size_M, num_warps):
                row_idx = row_start_m + m
                if row_idx < self.M:
                    for i in range(self.num_k_per_thread):
                        k = lane_idx + Int32(i * 32)
                        if k < Int32(self.tile_K):
                            global_k = Int32(k_offset) + k
                            smem_byte = (Int32(m) * Int32(self.tile_K) + k) * Int32(self.elem_bytes)
                            s_elem = cute.make_tensor(
                                cute.make_ptr(self.a_dtype, page_ptr + smem_byte, cute.AddressSpace.smem),
                                cute.make_layout((1,)),
                            )
                            if global_k < Int32(self.K):
                                g_elem = cute.make_tensor(
                                    a.iterator + row_idx * self.K + global_k,
                                    cute.make_layout((1,)),
                                )
                                s_elem[0] = g_elem[0]
                            else:
                                s_elem[0] = Float32(0.0).to(self.a_dtype)

            # Cooperative G->S load of B[tile_N, tile_K]
            # Write to smem matching col-major (tile_K, tile_N): offset = n*tile_K + k
            b_smem_base = Int32(self.a_tile_bytes)
            for n in range(warp_idx, self.tile_size_N, num_warps):
                row_idx = col_start_n + n
                if row_idx < self.N:
                    for i in range(self.num_k_per_thread):
                        k = lane_idx + Int32(i * 32)
                        if k < Int32(self.tile_K):
                            global_k = Int32(k_offset) + k
                            smem_byte = b_smem_base + (Int32(n) * Int32(self.tile_K) + k) * Int32(self.elem_bytes)
                            s_elem = cute.make_tensor(
                                cute.make_ptr(self.b_dtype, page_ptr + smem_byte, cute.AddressSpace.smem),
                                cute.make_layout((1,)),
                            )
                            if global_k < Int32(self.K):
                                g_elem = cute.make_tensor(
                                    b.iterator + row_idx * self.K + global_k,
                                    cute.make_layout((1,)),
                                )
                                s_elem[0] = g_elem[0]
                            else:
                                s_elem[0] = Float32(0.0).to(self.b_dtype)

            # Sync to ensure smem is fully written
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # MMA from smem: explicit S->R copy then gemm
            for k_block in cutlass.range_constexpr(self.tile_K // 16):
                cute.autovec_copy(tCsA[None, None, k_block], tCrA[None, None, k_block])
                cute.autovec_copy(tCsB[None, None, k_block], tCrB[None, None, k_block])
                cute.gemm(tiled_mma, acc,
                          tCrA[None, None, k_block],
                          tCrB[None, None, k_block], acc)

        # --- Epilogue: R->S (write C to smem page for store phase) ---
        # Write accumulator to smem via partition_C (element-wise).
        # Row-major (tile_M, tile_N) so store phase can read rows sequentially.
        sC = cute.make_tensor(
            cute.make_ptr(self.c_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.tile_size_M, self.tile_size_N), stride=(self.tile_size_N, 1)),
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
