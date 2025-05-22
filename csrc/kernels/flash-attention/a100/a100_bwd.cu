#include "a100_common_bwd.cuh"

namespace fa_a100 {

using namespace kittens;

template<int D>
__global__  __launch_bounds__((BWD_PREP_NUM_WORKERS)*kittens::WARP_THREADS, 1)
void bwd_prep_ker(const __grid_constant__ bwd_prep_globals<D> g) {
    extern __shared__ int __shm[];
    shared_allocator al((int*)&__shm[0]);

    int workerid = kittens::warpid();
    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int seq_idx = blockIdx.x * BWD_PREP_NUM_WORKERS + workerid;

    using ker_tile_dims = bwd_prep_ker_tile_dims<D>;

    using og_tile = ker_tile_dims::og_tile;
    using o_tile = ker_tile_dims::o_tile;
    using d_tile = ker_tile_dims::d_tile;

    og_tile (&og_smem)[BWD_PREP_NUM_WORKERS] = al.allocate<og_tile, BWD_PREP_NUM_WORKERS>();
    o_tile (&o_smem)[BWD_PREP_NUM_WORKERS] = al.allocate<o_tile, BWD_PREP_NUM_WORKERS>();
    d_tile (&d_smem)[BWD_PREP_NUM_WORKERS] = al.allocate<d_tile, BWD_PREP_NUM_WORKERS>();

    rt_fl<ker_tile_dims::qo_height, ker_tile_dims::tile_width> og_reg, o_reg;
    col_vec<rt_fl<ker_tile_dims::qo_height, ker_tile_dims::tile_width>> d_reg;

    // load the og and o tiles
    load<2, false>(og_smem[workerid], g.og, {batch, head, seq_idx, 0});
    load<2, false>(o_smem[workerid], g.o, {batch, head, seq_idx, 0});

    // load the o, og tiles in registers
    load(o_reg, o_smem[workerid]);
    load(og_reg, og_smem[workerid]);

    // compute the og * o tile
    mul(og_reg, og_reg, o_reg);
    row_sum(d_reg, og_reg);
    __syncthreads();

    store(d_smem[workerid], d_reg);
    __syncwarp();
    store(g.d, d_smem[workerid], {batch, head, seq_idx, 0});
}

template<int HEAD_DIM, bool IS_CAUSAL>
__global__ __launch_bounds__(BWD_NUM_WORKERS*kittens::WARP_THREADS, 1)
void bwd_attend_ker(const __grid_constant__ bwd_globals<HEAD_DIM> g) {
    extern __shared__ int __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ker_tile_dims = bwd_ker_tile_dims<HEAD_DIM>;

    using q_tile = ker_tile_dims::q_tile;
    using k_tile = ker_tile_dims::k_tile;
    using v_tile = ker_tile_dims::v_tile;
    using o_tile = ker_tile_dims::o_tile;
    using og_tile = ker_tile_dims::og_tile;
    using qg_tile = ker_tile_dims::qg_tile;
    using kg_tile = ker_tile_dims::kg_tile;
    using vg_tile = ker_tile_dims::vg_tile;

    using l_tile = ker_tile_dims::l_tile;
    using d_tile = ker_tile_dims::d_tile;

    using attn_tile = ker_tile_dims::attn_tile;
    using attn_tile_mma = ker_tile_dims::attn_tile_mma;

    k_tile (&k_smem)[ker_tile_dims::stages] = al.allocate<k_tile, ker_tile_dims::stages>();
    v_tile (&v_smem)[ker_tile_dims::stages] = al.allocate<v_tile, ker_tile_dims::stages>();

    q_tile (&q_smem) = al.allocate<q_tile>();
    og_tile (&og_smem) = al.allocate<og_tile>();
    qg_tile (&qg_smem) = al.allocate<qg_tile>();

    l_tile (&l_smem) = al.allocate<l_tile>();
    d_tile (&d_smem) = al.allocate<d_tile>();
    kg_tile (&kg_smem) = reinterpret_cast<kg_tile(&)>(k_smem);
    vg_tile (&vg_smem) = reinterpret_cast<vg_tile(&)>(q_smem);

    attn_tile (&att_smem) = al.allocate<attn_tile>();
    attn_tile_mma (&att_mma_smem) = al.allocate<attn_tile_mma>();
}

template __global__ void bwd_prep_ker<64>(const __grid_constant__ bwd_prep_globals<64> g);
template __global__ void bwd_prep_ker<128>(const __grid_constant__ bwd_prep_globals<128> g);

template __global__ void bwd_attend_ker<64, false>(const __grid_constant__ bwd_globals<64> g);
template __global__ void bwd_attend_ker<64, true>(const __grid_constant__ bwd_globals<64> g);

template __global__ void bwd_attend_ker<128, false>(const __grid_constant__ bwd_globals<128> g);
template __global__ void bwd_attend_ker<128, true>(const __grid_constant__ bwd_globals<128> g);

} // namespace fa_a100
