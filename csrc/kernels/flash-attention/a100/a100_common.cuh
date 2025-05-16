#ifndef A100_COMMON_CUH
#define A100_COMMON_CUH

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

namespace fa_a100 {

using namespace kittens;

namespace cg = cooperative_groups;

constexpr int FWD_NUM_WORKERS = 4;

// Forward declaration
template<unsigned int D> struct fwd_attend_ker_tile_dims;
template<unsigned int D> struct bwd_attend_ker_tile_dims;

// Forward pass constants and types

// Forward tile dimensions specializations
template<> struct fwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int qo_height = (4*kittens::TILE_ROW_DIM<bf16>);
    constexpr static int kv_height = (2*kittens::TILE_ROW_DIM<bf16>);
    constexpr static int stages = (2);

    using q_tile = st_bf<qo_height, tile_width>;
    using k_tile = st_bf<kv_height, tile_width>;
    using v_tile = st_bf<kv_height, tile_width>;
    using l_col_vec = col_vec<st_fl<qo_height, tile_width>>;
    using o_tile = st_bf<qo_height, tile_width>;
};

template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height = (2*kittens::TILE_ROW_DIM<bf16>);
    constexpr static int kv_height = (1*kittens::TILE_ROW_DIM<bf16>);
    constexpr static int stages = (2);

    using q_tile = st_bf<qo_height, tile_width>;
    using k_tile = st_bf<kv_height, tile_width>;
    using v_tile = st_bf<kv_height, tile_width>;
    using l_col_vec = col_vec<st_fl<qo_height, tile_width>>;
    using o_tile = st_bf<qo_height, tile_width>;
};

// Forward globals definition
template<int D> struct fwd_globals {
    using ker_tile_dims = fwd_attend_ker_tile_dims<D>;

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

    const int seqlen_k;
    const int seqlen_q;
    const int num_heads_k;
    const int num_heads_q;
    const float sm_scale;

    // Compute the maximum shared memory required for the forward pass
    size_t get_smem_size() {
        // Shared memory for K, V tiles
        int k_smem_size = 2 * ker_tile_dims::stages * sizeof(k_tile);
        int v_smem_size = 2 * ker_tile_dims::stages * sizeof(v_tile);

        // Shared memory for l tile
        int l_smem_size = FWD_NUM_WORKERS * sizeof(l_col_vec);

        return k_smem_size + v_smem_size + l_smem_size;
    }
};

// Forward declarations of kernel functions
template<int D, bool is_causal>
__global__ void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g);

} // namespace fa_a100

#endif // A100_COMMON_CUH