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
template<unsigned int D> struct bwd_prep_ker_tile_dims;
template<unsigned int D> struct bwd_ker_tile_dims;

// Forward tile dimensions specializations

template<> struct bwd_prep_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int qo_height = (2*kittens::TILE_ROW_DIM<bf16>);

    using og_tile = st_bf<qo_height, tile_width>;
    using o_tile = st_bf<qo_height, tile_width>;
    using d_tile = col_vec<st_fl<qo_height, tile_width>>;
};

template<> struct bwd_prep_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height = (kittens::TILE_ROW_DIM<bf16>);

    using og_tile = st_bf<qo_height, tile_width>;
    using o_tile = st_bf<qo_height, tile_width>;
    using d_tile = col_vec<st_fl<qo_height, tile_width>>;
};

template<> struct bwd_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int qo_height = (1*kittens::TILE_ROW_DIM<bf16>);
    constexpr static int kv_height = (2*kittens::TILE_ROW_DIM<bf16>);
    constexpr static int stages = (2);

    // input
    using q_tile = st_bf<qo_height, tile_width>;
    using k_tile = st_bf<kv_height, tile_width>;
    using v_tile = st_bf<kv_height, tile_width>;
    using og_tile = st_bf<qo_height, tile_width>;

    // output
    using qg_tile = st_bf<qo_height, tile_width>;
    using kg_tile = st_bf<kv_height, tile_width>;
    using vg_tile = st_bf<kv_height, tile_width>;

    using l_tile = col_vec<st_fl<qo_height, kv_height>>;
    using d_tile = col_vec<st_fl<qo_height, tile_width>>;
};

template<> struct bwd_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height = (1*kittens::TILE_ROW_DIM<bf16>);
    constexpr static int kv_height = (2*kittens::TILE_ROW_DIM<bf16>);
    constexpr static int stages = (2);

    // input
    using q_tile = st_bf<qo_height, tile_width>;
    using k_tile = st_bf<kv_height, tile_width>;
    using v_tile = st_bf<kv_height, tile_width>;
    using og_tile = st_bf<qo_height, tile_width>;

    // output
    using qg_tile = st_bf<qo_height, tile_width>;
    using kg_tile = st_bf<kv_height, tile_width>;
    using vg_tile = st_bf<kv_height, tile_width>;

    // vectors
    using l_tile = col_vec<st_fl<qo_height, kv_height>>;
    using d_tile = col_vec<st_fl<qo_height, tile_width>>;
};

template<int D>
struct bwd_prep_globals {
    using ker_tile_dims = bwd_prep_ker_tile_dims<D>;

    using og_tile = ker_tile_dims::og_tile;
    using o_tile = ker_tile_dims::o_tile;
    using d_tile = ker_tile_dims::d_tile;

    using og_gl = gl<bf16, -1, -1, -1, -1, og_tile>;
    using o_gl = gl<bf16, -1, -1, -1, -1, o_tile>;
    using d_gl = gl<float, -1, -1, -1, -1, d_tile>;

    // global memory for input tensors
    og_gl og;
    o_gl o;
    d_gl d;

    size_t get_smem_size() {
        return BWD_PREP_NUM_WORKERS * (sizeof(og_tile) + sizeof(o_tile) + sizeof(d_tile));
    }
};

template<int D>
struct bwd_globals {
    using ker_tile_dims = bwd_ker_tile_dims<D>;

    // input tiles
    using q_tile = ker_tile_dims::q_tile;
    using k_tile = ker_tile_dims::k_tile;
    using v_tile = ker_tile_dims::v_tile;

    // output tiles
    using og_tile = ker_tile_dims::og_tile;
    using qg_tile = ker_tile_dims::qg_tile;
    using kg_tile = ker_tile_dims::kg_tile;
    using vg_tile = ker_tile_dims::vg_tile;

    // vectors
    using l_tile = ker_tile_dims::l_tile;
    using d_tile = ker_tile_dims::d_tile;

    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;

    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;
    using qg_gl = gl<bf16, -1, -1, -1, -1, qg_tile>;
    using kg_gl = gl<bf16, -1, -1, -1, -1, kg_tile>;
    using vg_gl = gl<bf16, -1, -1, -1, -1, vg_tile>;

    using l_gl = gl<float, -1, -1, -1, -1, l_tile>;
    using d_gl = gl<float, -1, -1, -1, -1, d_tile>;

    // global memory for input tensors
    q_gl Qg;
    k_gl Kg;
    v_gl Vg;
    og_gl OGg;

    // global memory for gradient tensors
    qg_gl QGg;
    kg_gl KGg;
    vg_gl VGg;

    // other tensors
    l_gl Lg;
    d_gl Dg;

    const int seqlen_q;
    const int seqlen_k;
    const int num_heads_q;
    const int num_heads_k;
    const float sm_scale;

    size_t get_smem_size() {

        size_t kv_smem_size = BWD_NUM_WORKERS * (sizeof(k_tile) + sizeof(v_tile));
        size_t qo_smem_size = ker_tile_dims::stages * (sizeof(q_tile) + sizeof(og_tile));
        size_t grad_smem_size = ker_tile_dims::stages * sizeof(qg_tile); // kg and vg use the same smem as q/k smem
        size_t vec_smem_size = ker_tile_dims::stages * (sizeof(l_tile) + sizeof(d_tile));

        return kv_smem_size + qo_smem_size + grad_smem_size + vec_smem_size;
    }
};

// preload a register tile from a smem vector
__device__ static inline void
stream_tile(auto &reg_tile, auto &smem_vec, int tic, float scale) {

    int base_col = kittens::laneid() / 4;

    __syncthreads();

    #pragma unroll
    for(int i = 0; i < reg_tile.height; i++) {
        float first_value = *(float*)&smem_vec[tic][i*16 + base_col + 0] * scale;
        float second_value = *(float*)&smem_vec[tic][i*16 + base_col + 8] * scale;
        #pragma unroll
        for(int j = 0; j < reg_tile.width; j++) {
            reg_tile.tiles[i][j].data[0] = make_float2(first_value, first_value);
            reg_tile.tiles[i][j].data[1] = make_float2(second_value, second_value);
            reg_tile.tiles[i][j].data[2] = make_float2(first_value, first_value);
            reg_tile.tiles[i][j].data[3] = make_float2(second_value, second_value);
        }
    }
}

// Forward declarations of kernel functions
template<int D>
__global__ void bwd_prep_ker(const __grid_constant__ bwd_prep_globals<D> g);

template<int D, bool IS_CAUSAL>
__global__ void bwd_attend_ker(const __grid_constant__ bwd_globals<D> g);

} // namespace fa_a100

#endif // A100_BWD_COMMON_CUH