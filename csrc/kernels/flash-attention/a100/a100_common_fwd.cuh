#ifndef A100_FWD_COMMON_CUH
#define A100_FWD_COMMON_CUH

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

namespace fa_a100 {

using namespace kittens;

namespace cg = cooperative_groups;

constexpr int FWD_NUM_WORKERS = 4;

// Forward tile dimensions specializations
template<unsigned int HEAD_DIM, unsigned int QO_SIZE, unsigned int KV_SIZE, unsigned int STAGES> struct fwd_ker_tile_dims {
    constexpr static int tile_width = (HEAD_DIM);
    constexpr static int qo_height = (QO_SIZE*kittens::TILE_ROW_DIM<bf16>);
    constexpr static int kv_height = (KV_SIZE*kittens::TILE_ROW_DIM<bf16>);
    constexpr static int stages = (STAGES);

    using q_tile = st_bf<qo_height, tile_width>;
    using k_tile = st_bf<kv_height, tile_width>;
    using v_tile = st_bf<kv_height, tile_width>;
    using l_col_vec = col_vec<st_fl<qo_height, tile_width>>;
    using o_tile = st_bf<qo_height, tile_width>;
};

// Forward globals definition
template<unsigned int HEAD_DIM, unsigned int QO_SIZE, unsigned int KV_SIZE, unsigned int STAGES> struct fwd_globals {
    using ker_tile_dims = fwd_ker_tile_dims<HEAD_DIM, QO_SIZE, KV_SIZE, STAGES>;

    using q_tile = ker_tile_dims::q_tile;
    using k_tile = ker_tile_dims::k_tile;
    using v_tile = ker_tile_dims::v_tile;
    using l_col_vec = ker_tile_dims::l_col_vec;
    using o_tile = ker_tile_dims::o_tile;

    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16,  -1, -1, -1, -1, o_tile>;

    q_gl Qg;
    k_gl Kg;
    v_gl Vg;
    l_gl Lg;
    o_gl Og;

    {{variant_globals}}

    const int seqlen_q;
    const int seqlen_k;
    const int num_heads_q;
    const int num_heads_k;
    const float sm_scale;

    // Compute the maximum shared memory required for the forward pass
    template<typename AttentionVariant>
    size_t get_smem_size(AttentionVariant& av) {
        // Shared memory for K, V tiles
        int k_smem_size = 2 * ker_tile_dims::stages * sizeof(k_tile);
        int v_smem_size = 2 * ker_tile_dims::stages * sizeof(v_tile);

        int smem_size = k_smem_size + v_smem_size;

        // Shared memory for l tile
        if constexpr (AttentionVariant::is_softmax) {
            smem_size += FWD_NUM_WORKERS * sizeof(l_col_vec);
        }

        //smem_size += av.get_smem_size();

        return smem_size;
    }
};

{{variant_interface_fwd}}

// Forward declarations of kernel functions
template<unsigned int HEAD_DIM, unsigned int QO_SIZE, unsigned int KV_SIZE, unsigned int STAGES, bool IS_CAUSAL, typename AttentionVariant>
__global__ void fwd_attend_ker(const __grid_constant__ fwd_globals<HEAD_DIM, QO_SIZE, KV_SIZE, STAGES> g, AttentionVariant& variant);

} // namespace fa_a100

#endif // A100_FWD_COMMON_CUH