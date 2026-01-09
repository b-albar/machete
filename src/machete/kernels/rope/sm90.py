import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
import cuda.bindings.driver as cuda

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
        mq: cute.Tensor,
        mcos: cute.Tensor,
        msin: cute.Tensor,
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
        grid = [mq.shape[0], mq.shape[1], 1]
        block = [num_threads, 1, 1]
        cluster = [cluster_size, 1, 1]  # Share across sequence dimension or head dimension

        # RoPE usually processes half_head_dim pairs
        half_d = const_expr(self.head_dim // 2)
        vec_size = const_expr(128 // mcos.element_type.width)

        # Pre-construct tiled copies in JIT context so layouts are static
        tcopy_cs = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mcos.element_type, num_bits_per_copy=128),
            cute.make_layout(num_threads),
            cute.make_layout(vec_size),
        )

        tcopy_q = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mq.element_type, num_bits_per_copy=128),
            cute.make_layout(num_threads),
            cute.make_layout(vec_size),
        )

        self.kernel(mq, mcos, msin, seqlen, tcopy_cs, tcopy_q).launch(
            grid=grid, block=block, cluster=cluster, stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mq: cute.Tensor,
        mcos: cute.Tensor,
        msin: cute.Tensor,
        seqlen: Int32,
        tcopy_cs: cute.TiledCopy,
        tcopy_q: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, head_idx, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()

        # RoPE usually processes half_head_dim pairs
        half_d = const_expr(self.head_dim // 2)

        # 1. Shared Memory for Cos/Sin to avoid repeated L1/L2 hits
        # In a cluster, we could use DSMEM to share this, but here we just use it
        # as a fast cache for the current block.
        smem = cutlass.utils.SmemAllocator()
        scos = smem.allocate_tensor(Float32, cute.make_layout(half_d))
        ssin = smem.allocate_tensor(Float32, cute.make_layout(half_d))

        row_pos = bidx % seqlen

        thr_copy_cs = tcopy_cs.get_slice(tidx)
        tcs_gcos = thr_copy_cs.partition_S(mcos[row_pos, :])
        tcs_scos = thr_copy_cs.partition_D(scos)
        tcs_gsin = thr_copy_cs.partition_S(msin[row_pos, :])
        tcs_ssin = thr_copy_cs.partition_D(ssin)

        # Vectorized load of Cos/Sin
        for i in range(cute.size(tcs_gcos)):
            tcs_scos[i] = tcs_gcos[i].to(Float32)
            tcs_ssin[i] = tcs_gsin[i].to(Float32)

        cute.arch.cp_async_wait_all()  # Ensure Cos/Sin are in SMEM
        cute.arch.sync_threads()

        # 2. Process Q (Vectorized)
        # Q resides in (Row, Head, Dim)
        # We split Dim into [0:half_d] and [half_d:D]
        thr_copy_q = tcopy_q.get_slice(tidx)

        # Partitions for Q1 [0:half_d] and Q2 [half_d:D]
        tqgq1 = thr_copy_q.partition_S(mq[bidx, head_idx, 0:half_d])
        tqgq2 = thr_copy_q.partition_S(mq[bidx, head_idx, half_d : self.head_dim])

        # We'll write back in-place
        tqgout1 = thr_copy_q.partition_D(mq[bidx, head_idx, 0:half_d])
        tqgout2 = thr_copy_q.partition_D(mq[bidx, head_idx, half_d : self.head_dim])

        # Partition Smem Cos/Sin for the arithmetic part
        tqscos = thr_copy_cs.partition_S(scos)
        tqssin = thr_copy_cs.partition_S(ssin)

        for i in range(cute.size(tqgq1)):
            q1_vec = tqgq1[i]
            q2_vec = tqgq2[i]
            cos_vec = tqscos[i]
            sin_vec = tqssin[i]

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

                tqgout1[i][v] = out1.to(mq.element_type)
                tqgout2[i][v] = out2.to(mq.element_type)


class RopeSM90(torch.autograd.Function):
    _compile_cache = {}

    @staticmethod
    def forward(ctx, q, cos, sin, backward=False):
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

        compile_key = (dtype, head_dim, backward)
        if compile_key not in RopeSM90._compile_cache:
            m_sym = cute.sym_int()
            num_heads = q.shape[-2]
            impl = RopeSM90Impl(cute_dtype, head_dim, backward=bool(backward))
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
