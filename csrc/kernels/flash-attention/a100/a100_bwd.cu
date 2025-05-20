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

    using ker_tile_dims = bwd_ker_tile_dims<D>;

    using og_tile = bwd_ker_tile_dims<D>::og_tile;
    using o_tile = bwd_ker_tile_dims<D>::o_tile;
    using d_tile = bwd_ker_tile_dims<D>::d_tile;

    og_tile (&og_smem)[BWD_PREP_NUM_WORKERS] = al.allocate<og_tile, BWD_PREP_NUM_WORKERS>();
    o_tile (&o_smem)[BWD_PREP_NUM_WORKERS] = al.allocate<o_tile, BWD_PREP_NUM_WORKERS>();
    d_tile (&d_smem)[BWD_PREP_NUM_WORKERS] = al.allocate<d_tile, BWD_PREP_NUM_WORKERS>();

    rt_fl<ker_tile_dims::qo_height, ker_tile_dims::tile_width> og_reg, o_reg;
    col_vec<rt_fl<ker_tile_dims::qo_height, ker_tile_dims::tile_width>> d_reg;

     // launch the async load of the first stage of og and o tiles
    int tic = 0;
    load_async<2, false>(og_smem[workerid], g.og, {batch, head, seq_idx, 0});
    load_async<2, false>(o_smem[workerid], g.o, {batch, head, seq_idx, 0});
    load_async_wait();

    // load the o, og tiles in registers
    load(o_reg, o_smem[workerid]);
    load(og_reg, og_smem[workerid]);

    // compute the og * o tile
    mul(og_reg, og_reg, o_reg);
    row_sum(d_reg, og_reg);
    __syncthreads();

    /*store(d_smem[workerid], d_reg);
    __syncwarp();
    store(g.d, d_smem[workerid], {batch, head, seq_idx, 0});*/
}

template __global__ void bwd_prep_ker<64>(const __grid_constant__ bwd_prep_globals<64> g);
template __global__ void bwd_prep_ker<128>(const __grid_constant__ bwd_prep_globals<128> g);

} // namespace fa_a100
