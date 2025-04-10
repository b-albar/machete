#include "a100_common.cuh"

namespace fa_a100 {

using namespace kittens;

template<int D, bool is_causal>
__global__  __launch_bounds__((FWD_NUM_WORKERS)*kittens::WARP_THREADS, 1)
void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g) {

    // shared memory allocation
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    using K = fwd_attend_ker_tile_dims<D>;

    using q_tile = st_bf<K::qo_height, K::tile_width>;
    using k_tile = st_bf<K::kv_height, K::tile_width>;
    using v_tile = st_bf<K::kv_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using o_tile = st_bf<K::qo_height, K::tile_width>;

    k_tile    (&k_smem)[K::stages]   = al.allocate<k_tile, K::stages>();
    v_tile    (&v_smem)[K::stages]   = al.allocate<v_tile, K::stages>();
    l_col_vec (&l_smem)[FWD_NUM_WORKERS] = al.allocate<l_col_vec, FWD_NUM_WORKERS>();
    q_tile      (*q_smem)[FWD_NUM_WORKERS] = reinterpret_cast<q_tile(*)>(k_smem);
    o_tile      (*o_smem)              = reinterpret_cast<o_tile(*)>(k_smem);

    int kv_blocks   = g.seq_len / (K::kv_height);
    int kv_head_idx = blockIdx.y / g.heads_ratio;
    int seq_idx     = blockIdx.x * FWD_NUM_WORKERS;

    const float LN2 = 0.69314718056f;
    const float LN2_INV = 1.44269504089f;

    int pipe_idx = K::stages - 1;

    if (warpid == 0) {
        printf("In the kernel\n");
    }
}

// Explicit instantiations for D=64
template __global__ void fwd_attend_ker<64, true>(const __grid_constant__ fwd_globals<64> g);
template __global__ void fwd_attend_ker<64, false>(const __grid_constant__ fwd_globals<64> g);

// Explicit instantiations for D=128
template __global__ void fwd_attend_ker<128, true>(const __grid_constant__ fwd_globals<128> g);
template __global__ void fwd_attend_ker<128, false>(const __grid_constant__ fwd_globals<128> g);

} // namespace fa_a100