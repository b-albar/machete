#include "a100_common_bwd.cuh"

namespace fa_a100
{

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

    template <int D>
    __global__ __launch_bounds__((BWD_PREP_NUM_WORKERS)*kittens::WARP_THREADS, 1) void bwd_prep_ker(const __grid_constant__ bwd_prep_globals<D> g)
    {
        extern __shared__ int __shm[];
        shared_allocator al((int *)&__shm[0]);

        int workerid = kittens::warpid();
        const int batch = blockIdx.z;
        const int head = blockIdx.y;
        const int seq_idx = blockIdx.x * BWD_PREP_NUM_WORKERS + workerid;

        using ker_tile_dims = bwd_prep_ker_tile_dims<D>;

        using og_tile = ker_tile_dims::og_tile;
        using o_tile = ker_tile_dims::o_tile;
        using d_tile = ker_tile_dims::d_tile;

        og_tile(&og_smem)[BWD_PREP_NUM_WORKERS] = al.allocate<og_tile, BWD_PREP_NUM_WORKERS>();
        o_tile(&o_smem)[BWD_PREP_NUM_WORKERS] = al.allocate<o_tile, BWD_PREP_NUM_WORKERS>();
        d_tile(&d_smem)[BWD_PREP_NUM_WORKERS] = al.allocate<d_tile, BWD_PREP_NUM_WORKERS>();

        rt_fl<ker_tile_dims::qo_height, ker_tile_dims::tile_width> og_reg, o_reg;
        col_vec<rt_fl<ker_tile_dims::qo_height, ker_tile_dims::tile_width>> d_reg;

        // load the og and o tiles
        load<2, false>(o_smem[workerid], g.o, {batch, head, seq_idx, 0});
        load<2, false>(og_smem[workerid], g.og, {batch, head, seq_idx, 0});

        // load the o, og tiles in registers
        load(o_reg, o_smem[workerid]);
        load(og_reg, og_smem[workerid]);

        // compute the og * o tile
        mul(og_reg, og_reg, o_reg);
        row_sum(d_reg, og_reg);
        __syncthreads();

        store(d_smem[workerid], d_reg);
        __syncwarp();
        store(g.d, d_smem[workerid], {batch, head, 0, seq_idx});
    }

    template <int HEAD_DIM, bool IS_CAUSAL>
    __global__ __launch_bounds__(BWD_NUM_WORKERS*kittens::WARP_THREADS, 1) void bwd_attend_ker(const __grid_constant__ bwd_globals<HEAD_DIM> g)
    {
        extern __shared__ int __shm[];
        shared_allocator al((int *)&__shm[0]);

        using ker_tile_dims = bwd_ker_tile_dims<HEAD_DIM>;

        using q_tile = ker_tile_dims::q_tile;
        using k_tile = ker_tile_dims::k_tile;
        using v_tile = ker_tile_dims::v_tile;

        using og_tile = ker_tile_dims::og_tile;
        using qg_tile = ker_tile_dims::qg_tile;
        using kg_tile = ker_tile_dims::kg_tile;
        using vg_tile = ker_tile_dims::vg_tile;

        using l_tile = ker_tile_dims::l_tile;
        using d_tile = ker_tile_dims::d_tile;

        using attn_tile = ker_tile_dims::attn_tile;

        k_tile(&k_smem)[BWD_NUM_WORKERS] = al.allocate<k_tile, BWD_NUM_WORKERS>();
        v_tile(&v_smem)[BWD_NUM_WORKERS] = al.allocate<v_tile, BWD_NUM_WORKERS>();

        qg_tile(&max_qg_smem)[BWD_NUM_WORKERS] = reinterpret_cast<qg_tile(&)[BWD_NUM_WORKERS]>(k_smem);

        q_tile(&q_smem)[ker_tile_dims::stages] = al.allocate<q_tile, ker_tile_dims::stages>();
        og_tile(&og_smem)[ker_tile_dims::stages] = al.allocate<og_tile, ker_tile_dims::stages>();
        qg_tile(&qg_smem)[ker_tile_dims::stages] = al.allocate<qg_tile, ker_tile_dims::stages>();

        l_tile(&l_smem)[ker_tile_dims::stages] = al.allocate<l_tile, ker_tile_dims::stages>();
        d_tile(&d_smem)[ker_tile_dims::stages] = al.allocate<d_tile, ker_tile_dims::stages>();
        kg_tile(&kg_smem)[BWD_NUM_WORKERS] = reinterpret_cast<kg_tile(&)[BWD_NUM_WORKERS]>(k_smem);
        vg_tile(&vg_smem)[BWD_NUM_WORKERS] = reinterpret_cast<vg_tile(&)[BWD_NUM_WORKERS]>(v_smem);

        //attn_tile(&att_smem) = al.allocate<attn_tile>();

        constexpr float INV_LN2 = 1.44269504089f;
        constexpr float LN2 = 0.69314718056f;

        // TODO: understand why we cannot use group<4> here - causes incorrect results
        using load_group = kittens::group<4>; // share loading of q, og, l tiles

        const int warpid = kittens::warpid();
        const int batch = blockIdx.z;
        const int head = blockIdx.y;
        const int workerid = kittens::warpid();
        const int seq_idx = blockIdx.x * BWD_NUM_WORKERS + workerid;
        const int seq_idx_k = seq_idx * ker_tile_dims::kv_height;

        const int qo_blocks = (g.Qg.rows() + ker_tile_dims::qo_height - 1) / ker_tile_dims::qo_height;

        const int q_start = (IS_CAUSAL) ? 0 : 0;

        rt_bf<ker_tile_dims::qo_height, ker_tile_dims::tile_width, row_l> q_reg, og_reg;
        rt_bf<ker_tile_dims::qo_height, ker_tile_dims::tile_width, col_l> q_reg_col, og_reg_col;
        rt_bf<ker_tile_dims::kv_height, ker_tile_dims::tile_width, row_l> k_reg, v_reg;
        rt_bf<ker_tile_dims::kv_height, ker_tile_dims::tile_width, col_l> k_reg_col;

        rt_fl<ker_tile_dims::kv_height, ker_tile_dims::tile_width, row_l> kg_reg, vg_reg;
        rt_fl<ker_tile_dims::qo_height, ker_tile_dims::tile_width, row_l> qg_reg;

        // attention registers
        rt_fl<ker_tile_dims::qo_height, ker_tile_dims::kv_height, row_l> s_block, ds_block;
        rt_bf<ker_tile_dims::qo_height, ker_tile_dims::kv_height, row_l> ds_block_mma, s_block_mma;
        rt_bf<ker_tile_dims::qo_height, ker_tile_dims::kv_height, col_l> ds_block_mma_col, s_block_mma_col;

        // load the K/V tiles
        // going through shared memory
        if (seq_idx_k < g.Kg.rows()) {
            load<2, false>(k_smem[workerid], g.Kg, {batch, head, seq_idx, 0});
            load<2, false>(v_smem[workerid], g.Vg, {batch, head, seq_idx, 0});
            __syncwarp();
            // load the K/V tiles into the register
            load(k_reg, k_smem[workerid]);
            load(v_reg, v_smem[workerid]);

            // multiply the k by the sm_scale before matrix product
            mul(k_reg, k_reg, __float2bfloat16(g.sm_scale));
        }
        __syncthreads();

        // zero the blocks kg_reg and vg_reg
        zero(kg_reg);
        zero(vg_reg);

        int tic = 0;
        // load the first tile
        load_group::load_async<2, false>(q_smem[tic], g.Qg, {batch, head, q_start, 0});
        load_group::load_async<2, false>(og_smem[tic], g.OGg, {batch, head, q_start, 0});
        load_group::load_async(l_smem[tic], g.Lg, {batch, head, 0, q_start});
        load_group::load_async(d_smem[tic], g.Dg, {batch, head, 0, q_start});
        load_group::load_async(qg_smem[tic], g.QGg, {batch, head, q_start, 0});

        for (auto qo_idx = q_start; qo_idx < qo_blocks ; qo_idx++, tic=(tic+1) % ker_tile_dims::stages) {

            if (qo_idx + 1 < qo_blocks) {
                int next_tic = (tic + 1) % ker_tile_dims::stages;
                load_group::load_async<2, false>(q_smem[next_tic], g.Qg, {batch, head, qo_idx + 1, 0});
                load_group::load_async<2, false>(og_smem[next_tic], g.OGg, {batch, head, qo_idx + 1, 0});
                load_group::load_async(l_smem[next_tic], g.Lg, {batch, head, 0, qo_idx + 1});
                load_group::load_async(d_smem[next_tic], g.Dg, {batch, head, 0, qo_idx + 1});
                load_group::load_async(qg_smem[next_tic], g.QGg, {batch, head, qo_idx + 1, 0});
                load_async_wait<1>();
            } else {
                load_async_wait();
            }

            load(q_reg, q_smem[tic]);

            //zero the s_block
            stream_tile(s_block, l_smem, tic, -1.0f);

            // compute the s block and accumulate the result into s_block_t
            // result is P_i = S_ij - L_i
            mma_ABt(s_block, q_reg, k_reg, s_block);

            mul(s_block, s_block, INV_LN2);

            // apply the causal mask
            if constexpr (IS_CAUSAL) {
                causal_mask(s_block, qo_idx * ker_tile_dims::qo_height, seq_idx_k, g.seqlen_q, g.seqlen_k);
            }
            exp2(s_block, s_block);

            // load the og tile and compute the dp block
            load(og_reg, og_smem[tic]);

            // initialize the ds_block with the d vector
            stream_tile(ds_block, d_smem, tic, -1.0f);
            mma_ABt(ds_block, og_reg, v_reg, ds_block);

            mul(ds_block, s_block, ds_block);

            if (IS_CAUSAL) {
                causal_mask(ds_block, qo_idx * ker_tile_dims::qo_height, seq_idx_k, g.seqlen_q, g.seqlen_k);
            }

            // compute the dv block
            copy(s_block_mma, s_block);
            swap_layout(og_reg_col, og_reg);
            swap_layout(s_block_mma_col, s_block_mma);
            mma_AtB(vg_reg, s_block_mma_col, og_reg_col, vg_reg);

            // copy the ds_block to the ds_block_mma (bf16)
            copy(ds_block_mma, ds_block);

            // compute dq as P * k * sm_scale - sm_scale is already applied to k
            zero(qg_reg);
            swap_layout(k_reg_col, k_reg);
            mma_AB(qg_reg, ds_block_mma, k_reg_col, qg_reg);
            atomic_add(qg_smem[tic], qg_reg);
            __syncthreads();
            if (kittens::warpid() == 0) {
                store(g.QGg, qg_smem[tic], {batch, head, qo_idx, 0});
            }

            /* // implementation dq computation w/ sum reduction and warp 0
            load(qg_reg, qg_smem[tic]);
            swap_layout(k_reg_col, k_reg);
            mma_AB(qg_reg, ds_block_mma, k_reg_col, qg_reg);
            store(max_qg_smem[workerid], qg_reg);
            __syncthreads();
            if (kittens::warpid() == 0) {
                for (int i = 1; i < BWD_NUM_WORKERS; i++) {
                    add(max_qg_smem[0], max_qg_smem[0], max_qg_smem[i]);
                }
                __syncwarp();
                store(g.QGg, max_qg_smem[0], {batch, head, qo_idx, 0});
            }*/

            // compute dk as P^T * q * sm_scale
            swap_layout(q_reg_col, q_reg);
            swap_layout(ds_block_mma_col, ds_block_mma);
            mma_AtB(kg_reg, ds_block_mma_col, q_reg_col, kg_reg);
        }

        // store the kg and vg tiles
        mul(kg_reg, kg_reg, g.sm_scale);
        store(kg_smem[workerid], kg_reg);
        store(vg_smem[workerid], vg_reg);
        __syncwarp();

        // store the kg and vg tiles
        store(g.KGg, kg_smem[workerid], {batch, head, seq_idx, 0});
        store(g.VGg, vg_smem[workerid], {batch, head, seq_idx, 0});
    }

    template __global__ void bwd_prep_ker<64>(const __grid_constant__ bwd_prep_globals<64> g);
    template __global__ void bwd_prep_ker<128>(const __grid_constant__ bwd_prep_globals<128> g);

    template __global__ void bwd_attend_ker<64, false>(const __grid_constant__ bwd_globals<64> g);
    template __global__ void bwd_attend_ker<64, true>(const __grid_constant__ bwd_globals<64> g);

    template __global__ void bwd_attend_ker<128, false>(const __grid_constant__ bwd_globals<128> g);
    template __global__ void bwd_attend_ker<128, true>(const __grid_constant__ bwd_globals<128> g);

} // namespace fa_a100
