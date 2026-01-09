# Copyright (c) 2025, Machete Authors
import torch
from torch import Tensor
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
import cuda.bindings.driver as cuda
from typing import Dict, Tuple

from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.compile_utils import make_fake_tensor as fake_tensor


class RopeSM90Impl:
    """
    SM90 (Hopper) optimized RoPE kernel.
    Uses Threadblock Clusters and Vectorized memory access.
    """

    def __init__(self, dtype, head_dim, backward=False):
        self.dtype = dtype
        self.head_dim = head_dim
        self.backward = backward

    @cute.jit
    def forward(
        self,
        mQ: cute.Tensor,
        mCos: cute.Tensor,
        mSin: cute.Tensor,
        seqlen: Int32,
        stream: cuda.CUstream,
    ):
        # Configuration
        # We process multiple heads per block to amortize cos/sin load
        # Or use Clusters to share cos/sin.
        cluster_size = 4
        num_threads = 128

        # Grid: [TotalRows, Heads / HeadsPerBlock, 1]
        # For simplicity: [TotalRows, Heads, 1]
        grid = [mQ.shape[0], mQ.shape[1], 1]
        block = [num_threads, 1, 1]
        cluster = [cluster_size, 1, 1]  # Share across sequence dimension or head dimension

        self.kernel(mQ, mCos, mSin, seqlen).launch(grid=grid, block=block, cluster=cluster, stream=stream)

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mCos: cute.Tensor, mSin: cute.Tensor, seqlen: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, head_idx, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()

        # RoPE usually processes half_head_dim pairs
        half_D = const_expr(self.head_dim // 2)

        # 1. Shared Memory for Cos/Sin to avoid repeated L1/L2 hits
        # In a cluster, we could use DSMEM to share this, but here we just use it
        # as a fast cache for the current block.
        smem = cutlass.utils.SmemAllocator()
        sCos = smem.allocate_tensor(Float32, cute.make_layout(half_D))
        sSin = smem.allocate_tensor(Float32, cute.make_layout(half_D))

        row_pos = bidx % seqlen

        # Load Cos/Sin for this sequence position into SMEM (Vectorized)
        vec_size = const_expr(128 // mCos.element_type.width)
        tcopy_cs = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mCos.element_type, num_bits_per_copy=128),
            cute.make_layout(num_threads),
            cute.make_layout(vec_size),
        )

        thr_copy_cs = tcopy_cs.get_slice(tidx)
        tCs_gCos = thr_copy_cs.partition_S(mCos[row_pos, :])
        tCs_sCos = thr_copy_cs.partition_D(sCos)
        tCs_gSin = thr_copy_cs.partition_S(mSin[row_pos, :])
        tCs_sSin = thr_copy_cs.partition_D(sSin)

        # Vectorized load of Cos/Sin
        for i in range(cute.size(tCs_gCos)):
            tCs_sCos[i] = tCs_gCos[i].to(Float32)
            tCs_sSin[i] = tCs_gSin[i].to(Float32)

        cute.arch.cp_async_wait_all()  # Ensure Cos/Sin are in SMEM
        cute.arch.sync_threads()

        # 2. Process Q (Vectorized)
        # Q resides in (Row, Head, Dim)
        # We split Dim into [0:half_D] and [half_D:D]
        tcopy_q = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mQ.element_type, num_bits_per_copy=128),
            cute.make_layout(num_threads),
            cute.make_layout(vec_size),
        )
        thr_copy_q = tcopy_q.get_slice(tidx)

        # Partitions for Q1 [0:half_D] and Q2 [half_D:D]
        tQgQ1 = thr_copy_q.partition_S(mQ[bidx, head_idx, 0:half_D])
        tQgQ2 = thr_copy_q.partition_S(mQ[bidx, head_idx, half_D : self.head_dim])

        # We'll write back in-place
        tQgOut1 = thr_copy_q.partition_D(mQ[bidx, head_idx, 0:half_D])
        tQgOut2 = thr_copy_q.partition_D(mQ[bidx, head_idx, half_D : self.head_dim])

        # Partition Smem Cos/Sin for the arithmetic part
        tQsCos = thr_copy_cs.partition_S(sCos)
        tQsSin = thr_copy_cs.partition_S(sSin)

        for i in range(cute.size(tQgQ1)):
            q1_vec = tQgQ1[i]
            q2_vec = tQgQ2[i]
            cos_vec = tQsCos[i]
            sin_vec = tQsSin[i]

            for v in range(cute.size(q1_vec)):
                q1 = q1_vec[v].to(Float32)
                q2 = q2_vec[v].to(Float32)
                c = cos_vec[v]
                s = sin_vec[v]

                if const_expr(self.backward):
                    s = -s

                # Rotation logic
                out1 = q1 * c - q2 * s
                out2 = q2 * c + q1 * s

                tQgOut1[i][v] = out1.to(mQ.element_type)
                tQgOut2[i][v] = out2.to(mQ.element_type)


class RopeSM90(torch.autograd.Function):
    _compile_cache = {}

    @staticmethod
    def forward(ctx, q, cos, sin):
        """
        SM90-optimized Forward pass.
        Optimizations:
        1. 128-bit Vectorized Memory Access (LDG.128/STG.128)
        2. Per-block Shared Memory Caching of Cos/Sin tables
        3. Threadblock Clusters for sequence-wide cache stability
        """
        # q: [B, S, H, D] -> flattened [B*S, H, D]
        ori_shape = q.shape
        head_dim = q.shape[-1]

        if not q.is_contiguous():
            q = q.contiguous()

        q_flat = q.reshape(-1, q.shape[-2], head_dim)

        dtype = q.dtype
        cute_dtype = torch2cute_dtype_map[dtype]

        seqlen = cos.shape[0]

        compile_key = (dtype, head_dim, "forward")
        if compile_key not in RopeSM90._compile_cache:
            m_sym = cute.sym_int()
            num_heads = q.shape[-2]
            impl = RopeSM90Impl(cute_dtype, head_dim)
            RopeSM90._compile_cache[compile_key] = cute.compile(
                impl.forward,
                fake_tensor(cute_dtype, (m_sym, num_heads, head_dim)),
                fake_tensor(cute_dtype, (seqlen, head_dim // 2)),
                fake_tensor(cute_dtype, (seqlen, head_dim // 2)),
                Int32(seqlen),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )

        RopeSM90._compile_cache[compile_key](q_flat, cos, sin, seqlen)
        return q_flat.view(*ori_shape)
