#ifndef A100_BWD_COMMON_CUH
#define A100_BWD_COMMON_CUH

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

namespace fa_a100 {

using namespace kittens;

namespace cg = cooperative_groups;

constexpr int BWD_PREP_NUM_WORKERS = 4;
constexpr int BWD_NUM_WORKERS = 4;

// Forward declaration
template<unsigned int D> struct bwd_ker_tile_dims;

// Forward tile dimensions specializations
template<> struct bwd_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int qo_height = (4*kittens::TILE_ROW_DIM<bf16>);
    constexpr static int kv_height = (2*kittens::TILE_ROW_DIM<bf16>);
    constexpr static int stages = (2);

    using q_tile = st_bf<qo_height, tile_width>;
    using k_tile = st_bf<kv_height, tile_width>;
    using v_tile = st_bf<kv_height, tile_width>;
    using og_tile = st_bf<qo_height, tile_width>;
    using qg_tile = st_fl<qo_height, tile_width>;
    using kg_tile = st_fl<kv_height, tile_width>;
    using vg_tile = st_fl<kv_height, tile_width>;
    using l_tile = row_vec<st_fl<qo_height, kv_height>>;
    using d_tile = row_vec<st_fl<qo_height, kv_height>>;
    using o_tile  = st_bf<qo_height, tile_width>;
};

template<> struct bwd_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height = (4*kittens::TILE_ROW_DIM<bf16>);
    constexpr static int kv_height = (2*kittens::TILE_ROW_DIM<bf16>);
    constexpr static int stages = (2);

    using q_tile = st_bf<qo_height, tile_width>;
    using k_tile = st_bf<kv_height, tile_width>;
    using v_tile = st_bf<kv_height, tile_width>;
    using og_tile = st_bf<qo_height, tile_width>;
    using qg_tile = st_fl<qo_height, tile_width>;
    using kg_tile = st_fl<kv_height, tile_width>;
    using vg_tile = st_fl<kv_height, tile_width>;
    using l_tile = row_vec<st_fl<qo_height, kv_height>>;
    using d_tile = row_vec<st_fl<qo_height, kv_height>>;
    using o_tile  = st_bf<qo_height, tile_width>;
};

template<int D>
struct bwd_prep_globals {
    using ker_tile_dims = bwd_ker_tile_dims<D>;

    using og_tile = ker_tile_dims::og_tile;
    using o_tile  = ker_tile_dims::o_tile;
    using d_tile  = ker_tile_dims::d_tile;

    using og_gl = gl<bf16, -1, -1, -1, -1>;
    using o_gl  = gl<bf16, -1, -1, -1, -1>;
    using d_gl  = gl<float, -1, -1, -1, -1>;

    // global memory for input tensors
    og_gl og;
    o_gl  o;
    d_gl  d;

    size_t get_smem_size() {
        return BWD_PREP_NUM_WORKERS * sizeof(og_tile) +
               BWD_PREP_NUM_WORKERS * sizeof(o_tile) +
               BWD_PREP_NUM_WORKERS * sizeof(d_tile);
    }
};

template<int D>
struct bwd_globals {
    using ker_tile_dims = bwd_ker_tile_dims<D>;

    using q_gl  = gl<bf16,  -1, -1, -1, -1>;
    using k_gl  = gl<bf16,  -1, -1, -1, -1>;
    using v_gl  = gl<bf16,  -1, -1, -1, -1>;
    using o_gl  = gl<bf16,  -1, -1, -1, -1>;

    using og_gl = gl<bf16,  -1, -1, -1, -1>;
    using qg_gl = gl<float, -1, -1, -1, -1>;
    using kg_gl = gl<float, -1, -1, -1, -1>;
    using vg_gl = gl<float, -1, -1, -1, -1>;

    using l_gl  = gl<float, -1, -1, -1, -1>;
    using d_gl  = gl<float, -1, -1, -1, -1>;

    // global memory for input tensors
    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;

    // global memory for gradient tensors
    og_gl og;
    qg_gl qg;
    kg_gl kg;
    vg_gl vg;

    // other tensors
    l_gl l;
    d_gl d;

    const int seqlen_k;
    const int seqlen_q;
    const int num_heads_k;
    const int num_heads_q;
    const float sm_scale;
};


// Forward declarations of kernel functions
template<int D>
__global__ void bwd_prep_ker(const __grid_constant__ bwd_prep_globals<D> g);

template<int D, bool is_causal, bool is_even_nm>
__global__ void bwd_attend_ker(const __grid_constant__ bwd_globals<D> g);

} // namespace fa_a100

#endif // A100_BWD_COMMON_CUH