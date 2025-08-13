#include "a100_common_fwd.cuh"

namespace fa_a100 {

using namespace kittens;

__device__ static inline void
causal_mask(auto &reg_tile, int q_blk, int k_blk, int seqlen_q, int seqlen_k) {
    #pragma unroll
    for (auto i = 0; i < reg_tile.height; i++) {
        auto q_idx = q_blk + i * reg_tile.tile_size_row;
        #pragma unroll
        for (auto j = 0; j < reg_tile.width; j++) {
            auto k_idx = k_blk + j * reg_tile.tile_size_col;
            auto &attn_subtile = reinterpret_cast<rt_fl<kittens::TILE_ROW_DIM<float>, kittens::TILE_COL_DIM<float>>&>(reg_tile.tiles[i][j]);

            if (k_idx > q_idx || q_idx >= seqlen_q || k_idx >= seqlen_k) {
                neg_infty(attn_subtile);
            } else if (k_idx == q_idx) {
                make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty());
            }
            __syncwarp();
        }
    }
}

template<unsigned int HEAD_DIM, unsigned int QO_SIZE, unsigned int KV_SIZE, unsigned int STAGES, bool IS_CAUSAL, typename AttentionVariant>
__global__  __launch_bounds__(FWD_NUM_WORKERS*kittens::WARP_THREADS, 1)
void fwd_attend_ker(const __grid_constant__ fwd_globals<HEAD_DIM, QO_SIZE, KV_SIZE, STAGES> g, AttentionVariant& attention_variant) {

    // Shared memory allocation
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    constexpr float INV_LN2 = 1.44269504089f;
    constexpr float LN2 = 0.69314718056f;

    using load_group = kittens::group<2>; // pairs of workers collaboratively load k, v tiles
    int loadid = load_group::groupid(), workerid = kittens::warpid();
    constexpr int LOAD_BLOCKS = FWD_NUM_WORKERS / load_group::GROUP_WARPS;
    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int seq_idx = blockIdx.x * FWD_NUM_WORKERS + workerid;

    using ker_tile_dims = fwd_ker_tile_dims<HEAD_DIM, QO_SIZE, KV_SIZE, STAGES>;
    using q_tile = ker_tile_dims::q_tile;
    using k_tile = ker_tile_dims::k_tile;
    using v_tile = ker_tile_dims::v_tile;
    using l_col_vec = ker_tile_dims::l_col_vec;
    using o_tile = ker_tile_dims::o_tile;

    constexpr static int kv_height = ker_tile_dims::kv_height;
    constexpr static int qo_height = ker_tile_dims::qo_height;
    constexpr static int stages = ker_tile_dims::stages;

    const int seq_idx_q = seq_idx * qo_height;

    int kv_blocks = (g.seqlen_k + kv_height - 1) / kv_height;
    // number of iterations for the kv loop
    int kv_iters;
    if constexpr (IS_CAUSAL) {
        kv_iters = (g.seqlen_q + qo_height) / kv_height;
        kv_iters = min(kv_iters, kv_blocks);
    }
    else {
        kv_iters = kv_blocks;
    }

    // divide the kv_iters by the group size
    kv_iters = kv_iters / LOAD_BLOCKS;

    // shared memory allocation for k, v tiles
    k_tile (&k_smem)[LOAD_BLOCKS][stages] = al.allocate<k_tile, LOAD_BLOCKS, stages>();
    v_tile (&v_smem)[LOAD_BLOCKS][stages] = al.allocate<v_tile, LOAD_BLOCKS, stages>();

    // use k_smem for loading Q and O tiles
    q_tile (&q_smem)[FWD_NUM_WORKERS] = reinterpret_cast<q_tile(&)[FWD_NUM_WORKERS]>(k_smem);
    o_tile (&o_smem)[FWD_NUM_WORKERS] = reinterpret_cast<o_tile(&)[FWD_NUM_WORKERS]>(k_smem);

    // register allocation
    rt_bf<qo_height, HEAD_DIM, row_l> q_reg;
    rt_bf<kv_height, HEAD_DIM, row_l> k_reg;
    rt_bf<kv_height, HEAD_DIM, col_l> v_reg;
    rt_fl<qo_height, kv_height, row_l> att_block;
    rt_bf<qo_height, kv_height, row_l> att_block_mma;
    rt_fl<qo_height, HEAD_DIM, row_l> o_reg;

    // softmax requires l_i, which is the log of the normalization factor
    col_vec<rt_fl<qo_height, kv_height>> max_vec, norm_vec, max_vec_old;
    l_col_vec (*l_smem)[FWD_NUM_WORKERS];

    if constexpr (attention_variant.is_softmax) {
        l_smem = &al.allocate<l_col_vec, FWD_NUM_WORKERS>();
        // initialize the max and norm vectors
        neg_infty(max_vec);
        zero(norm_vec);
    }

    // initialize the attention variant
    attention_variant.initialize(g, al);

    // initialize some registers
    zero(o_reg);

    // load the Q tile
    if (seq_idx_q < g.Qg.rows()) {
        // going through shared memory improves coalescing of dram reads.
        load<2, false>(q_smem[workerid], g.Qg, {batch, head, seq_idx, 0});
        __syncwarp();
        // load the Q tile into the register
        load(q_reg, q_smem[workerid]);
    }
    __syncthreads();

    // launch the async load of the first stage k, v tiles
    int tic = 0;
    load_group::load_async<2, false>(k_smem[loadid][tic], g.Kg, {batch, head, loadid, 0});
    load_group::load_async<2, false>(v_smem[loadid][tic], g.Vg, {batch, head, loadid, 0});
    attention_variant.load_data(g, workerid, tic, batch, head, seq_idx, loadid);

    // iterate over k, v for these q's that have been loaded
    for(auto kv_idx = 0; kv_idx < kv_iters; kv_idx++, tic=(tic+1) % stages) {
        int next_load_idx = (kv_idx+1)*LOAD_BLOCKS + loadid;

        // load the next k, v tile if there are more iterations
        if(kv_idx+1 < kv_iters) {
            int next_tic = (tic+1) % stages;
            load_group::load_async<2, false>(k_smem[loadid][next_tic], g.Kg, {batch, head, next_load_idx, 0});
            load_group::load_async<2, false>(v_smem[loadid][next_tic], g.Vg, {batch, head, next_load_idx, 0});
            attention_variant.load_data(g, workerid, next_tic, batch, head, seq_idx, next_load_idx);
            load_async_wait<1>(); // next k, v can stay in flight.
        } else {
            load_async_wait();
        }

        __syncthreads();

        #pragma unroll LOAD_BLOCKS
        for(int subtile = 0; subtile < LOAD_BLOCKS && (kv_idx*LOAD_BLOCKS + subtile)*kv_height < g.seqlen_k; subtile++) {
            load(k_reg, k_smem[subtile][tic]); // load k from shared into registers

            attention_variant.qkv_transform(g, q_reg, k_reg, v_reg);

            zero(att_block); // zero attention tile
            mma_ABt(att_block, q_reg, k_reg, att_block); // Q@K.T

            // apply the variant transform of the logits
            attention_variant.logits_transform(g, att_block);

            // apply transformation of attention values (i.e. causal mask, bias, etc.)
            if constexpr (IS_CAUSAL) {
                causal_mask(
                    att_block,
                    seq_idx_q,
                    (kv_idx*LOAD_BLOCKS + subtile) * kv_height,
                    g.seqlen_q,
                    g.seqlen_k
                );
            }

            // scale by INV_LN2 for using exp2 and sm_scale
            if constexpr (attention_variant.is_softmax) {
                mul(att_block, att_block, INV_LN2 * g.sm_scale);
            } else {
                mul(att_block, att_block, g.sm_scale);
            }

            if constexpr (attention_variant.is_softmax) {
                // save the previous max_vec
                copy(max_vec_old, max_vec);

                // compute the max of the attention block - already scaled
                row_max(max_vec, att_block, max_vec);

                // compute exp2(S - m_i)
                sub_row(att_block, att_block, max_vec);
                exp2(att_block, att_block);

                // compute l_i = exp(m_prev - m_i) * l_i + rowsum(S)
                sub(max_vec_old, max_vec_old, max_vec);
                exp2(max_vec_old, max_vec_old);
                mul(norm_vec, norm_vec, max_vec_old);
                row_sum(norm_vec, att_block, norm_vec);

               attention_variant.finalize_step(g);
            }

            // copy the attention block from fp32 to bf16 register
            copy(att_block_mma, att_block);

            // load values and multiply by them
            load(v_reg, v_smem[subtile][tic]);
            mul_row(o_reg, o_reg, max_vec_old);
            mma_AB(o_reg, att_block_mma, v_reg, o_reg);
        }

        __syncthreads();
    }

    if constexpr (attention_variant.is_softmax) {
        div_row(o_reg, o_reg, norm_vec); // divide by l_i

        mul(max_vec, max_vec, LN2);
        log(norm_vec, norm_vec);
        add(norm_vec, norm_vec, max_vec);
    }

    if (seq_idx_q < g.Og.rows()) { // write out o.
        store(o_smem[workerid], o_reg);
        __syncwarp();
        store<2, false>(g.Og, o_smem[workerid], {batch, head, seq_idx, 0});

        if constexpr (attention_variant.is_softmax) {
            store((*l_smem)[workerid], norm_vec);
            store(g.Lg, (*l_smem)[workerid], {batch, head, 0, seq_idx});
        }
    }
    attention_variant.finalize(g, workerid, batch, head, seq_idx);
}

{{fwd_explicit_instantiations}}

} // namespace fa_a100