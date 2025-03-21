#include "h100_common.cuh"
#include "pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

namespace fa_h100 {

using namespace kittens;

std::vector<torch::Tensor>
attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, float sm_scale)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    auto batch    = q.size(0);
    auto seq_len  = q.size(2);
    auto head_dim = q.size(3);
    auto is_causal = causal;
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(q.size(0) == batch, "Q batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0) == batch, "K batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0) == batch, "V batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q.size(2) == seq_len, "Q sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(k.size(2) == seq_len, "K sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(v.size(2) == seq_len, "V sequence length dimension - idx 2 - must match for all inputs");

    TORCH_CHECK(q.size(3) == head_dim, "Q head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3) == head_dim, "K head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3) == head_dim, "V head dimension - idx 3 - must match for all non-vector inputs");

    TORCH_CHECK(qo_heads >= kv_heads, "QO heads must be greater than or equal to KV heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "QO heads must be divisible by KV heads");
    TORCH_CHECK(q.size(1) == qo_heads, "QO head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");

    auto hr = qo_heads / kv_heads;

    c10::BFloat16* q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr = v.data_ptr<c10::BFloat16>();

    bf16*  d_q = reinterpret_cast<bf16*>(q_ptr);
    bf16*  d_k = reinterpret_cast<bf16*>(k_ptr);
    bf16*  d_v = reinterpret_cast<bf16*>(v_ptr);

    // for the returned outputs
    torch::Tensor o     = torch::empty({static_cast<const uint>(batch),
                                        static_cast<const uint>(qo_heads),
                                        static_cast<const uint>(seq_len),
                                        static_cast<const uint>(head_dim)}, v.options());

    torch::Tensor l_vec = torch::empty({static_cast<const uint>(batch),
                                        static_cast<const uint>(qo_heads),
                                        static_cast<const uint>(seq_len),
                                        static_cast<const uint>(1)},
                                        torch::TensorOptions().dtype(torch::kFloat).device(q.device()).memory_format(at::MemoryFormat::Contiguous));


    bf16*  o_ptr = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16*  d_o   = reinterpret_cast<bf16*>(o_ptr);

    float* l_ptr = reinterpret_cast<float*>(l_vec.data_ptr<float>());
    float* d_l   = reinterpret_cast<float*>(l_ptr);

    cudaDeviceSynchronize();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (head_dim == 64) {
        using q_tile    =         st_bf<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width>;
        using k_tile    =         st_bf<fwd_attend_ker_tile_dims<64>::kv_height, fwd_attend_ker_tile_dims<64>::tile_width>;
        using v_tile    =         st_bf<fwd_attend_ker_tile_dims<64>::kv_height, fwd_attend_ker_tile_dims<64>::tile_width>;
        using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width>>;
        using o_tile    =         st_bf<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width>;

        using q_global = gl<bf16,  -1, -1, -1, -1, q_tile>;
        using k_global = gl<bf16,  -1, -1, -1, -1, k_tile>;
        using v_global = gl<bf16,  -1, -1, -1, -1, v_tile>;
        using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
        using o_global = gl<bf16,  -1, -1, -1, -1, o_tile>;

        using globals      = fwd_globals<64>;

        q_global qg_arg{d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};
        k_global kg_arg{d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 64U};
        v_global vg_arg{d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 64U};
        l_global lg_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,  static_cast<unsigned int>(seq_len)};
        o_global og_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};

        globals g{qg_arg, kg_arg, vg_arg, lg_arg, og_arg, static_cast<int>(seq_len), static_cast<int>(hr), sm_scale};

        auto mem_size = kittens::MAX_SHARED_MEMORY;
        auto threads  = NUM_WORKERS * kittens::WARP_THREADS;

        // TORCH_CHECK(seq_len % (CONSUMER_WARPGROUPS*kittens::TILE_DIM*4) == 0, "sequence length must be divisible by 192");
        dim3 grid(seq_len/(CONSUMER_WARPGROUPS*kittens::TILE_ROW_DIM<bf16>*4), qo_heads, batch);

        if (is_causal) {
            cudaFuncSetAttribute(
                fwd_attend_ker<64, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<64, true><<<grid, (32*NUM_WORKERS), mem_size, stream>>>(g);
        }
        else {
            cudaFuncSetAttribute(
                fwd_attend_ker<64, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<64, false><<<grid, (32*NUM_WORKERS), mem_size, stream>>>(g);
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
        cudaStreamSynchronize(stream);
    }

    if (head_dim == 128) {
        using q_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>;
        using k_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width>;
        using v_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width>;
        using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>>;
        using o_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>;

        using q_global = gl<bf16,  -1, -1, -1, -1, q_tile>;
        using k_global = gl<bf16,  -1, -1, -1, -1, k_tile>;
        using v_global = gl<bf16,  -1, -1, -1, -1, v_tile>;
        using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
        using o_global = gl<bf16,  -1, -1, -1, -1, o_tile>;

        using globals      = fwd_globals<128>;

        q_global qg_arg{d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        k_global kg_arg{d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        v_global vg_arg{d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        l_global lg_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(seq_len)};
        o_global og_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};

        globals g{qg_arg, kg_arg, vg_arg, lg_arg, og_arg, static_cast<int>(seq_len), static_cast<int>(hr), sm_scale};

        auto mem_size = kittens::MAX_SHARED_MEMORY;
        auto threads  = NUM_WORKERS * kittens::WARP_THREADS;

        // TORCH_CHECK(seq_len % (CONSUMER_WARPGROUPS*kittens::TILE_DIM*4) == 0, "sequence length must be divisible by 192");
        dim3 grid(seq_len/(CONSUMER_WARPGROUPS*kittens::TILE_ROW_DIM<bf16>*4), qo_heads, batch);

        if (is_causal) {
            cudaFuncSetAttribute(
                fwd_attend_ker<128, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<128, true><<<grid, (32*NUM_WORKERS), mem_size, stream>>>(g);
        }
        else {
            cudaFuncSetAttribute(
                fwd_attend_ker<128, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            fwd_attend_ker<128, false><<<grid, (32*NUM_WORKERS), mem_size, stream>>>(g);
        }

        CHECK_CUDA_ERROR(cudaGetLastError());
        cudaStreamSynchronize(stream);
    }

    return {o, l_vec};
    cudaDeviceSynchronize();
}

std::vector<torch::Tensor>
attention_backward(torch::Tensor q,
                   torch::Tensor k,
                   torch::Tensor v,
                   torch::Tensor o,
                   torch::Tensor l_vec,
                   torch::Tensor og,
                   bool causal,
                   float sm_scale)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(l_vec);
    CHECK_INPUT(o);
    CHECK_INPUT(og);

    auto batch    = q.size(0);
    auto seq_len  = q.size(2);
    auto head_dim = q.size(3);

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(q.size(0)     == batch, "Q  batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0)     == batch, "K  batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0)     == batch, "V  batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(l_vec.size(0) == batch, "L  batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(o.size(0)     == batch, "O  batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(og.size(0)    == batch, "OG batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q.size(2)     == seq_len, "Q  sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(k.size(2)     == seq_len, "K  sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(v.size(2)     == seq_len, "V  sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(l_vec.size(2) == seq_len, "L  sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(o.size(2)     == seq_len, "O  sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(og.size(2)    == seq_len, "OG sequence length dimension - idx 2 - must match for all inputs");

    TORCH_CHECK(q.size(3)  == head_dim, "Q  head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3)  == head_dim, "K  head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3)  == head_dim, "V  head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(o.size(3)  == head_dim, "O  head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(og.size(3) == head_dim, "OG head dimension - idx 3 - must match for all non-vector inputs");

    // check if causal
    auto is_causal = causal;

    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);

    TORCH_CHECK(qo_heads >= kv_heads,     "Q heads must be greater than or equal to K and V heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "Q heads must be divisible by KV heads");

    TORCH_CHECK(q.size(1)     == qo_heads, "Q  heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(l_vec.size(1) == qo_heads, "L  heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(o.size(1)     == qo_heads, "O  heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(og.size(1)    == qo_heads, "OG heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k.size(1)  == kv_heads, "K  heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v.size(1)  == kv_heads, "V  heads dimension - idx 1 - must match for all inputs");

    auto hr = qo_heads / kv_heads;

    c10::BFloat16* q_ptr  = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr  = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr  = v.data_ptr<c10::BFloat16>();
    c10::BFloat16* o_ptr  = o.data_ptr<c10::BFloat16>();
    c10::BFloat16* og_ptr = og.data_ptr<c10::BFloat16>();
    float*         l_ptr  = l_vec.data_ptr<float>();

    torch::Tensor qg = torch::zeros({static_cast<const uint>(batch),
                                     static_cast<const uint>(qo_heads),
                                     static_cast<const uint>(seq_len),
                                     static_cast<const uint>(head_dim)},   l_vec.options());
    torch::Tensor kg = torch::zeros({static_cast<const uint>(batch),
                                     static_cast<const uint>(kv_heads),
                                     static_cast<const uint>(seq_len),
                                     static_cast<const uint>(head_dim)},   l_vec.options());
    torch::Tensor vg = torch::zeros({static_cast<const uint>(batch),
                                     static_cast<const uint>(kv_heads),
                                     static_cast<const uint>(seq_len),
                                     static_cast<const uint>(head_dim)},   l_vec.options());

    torch::Tensor d_vec = torch::empty({static_cast<const uint>(batch),
                                        static_cast<const uint>(qo_heads),
                                        static_cast<const uint>(seq_len),
                                        static_cast<const uint>(1)},       l_vec.options());

    float*         qg_ptr = qg.data_ptr<float>();
    float*         kg_ptr = kg.data_ptr<float>();
    float*         vg_ptr = vg.data_ptr<float>();
    float*         d_ptr  = d_vec.data_ptr<float>();

    bf16*  d_q  = reinterpret_cast<bf16*>(q_ptr);
    bf16*  d_k  = reinterpret_cast<bf16*>(k_ptr);
    bf16*  d_v  = reinterpret_cast<bf16*>(v_ptr);
    bf16*  d_o  = reinterpret_cast<bf16*>(o_ptr);
    bf16*  d_og = reinterpret_cast<bf16*>(og_ptr);
    float* d_l  = reinterpret_cast<float*>(l_ptr);
    float* d_d  = reinterpret_cast<float*>(d_ptr);
    float* d_qg = reinterpret_cast<float*>(qg_ptr);
    float* d_kg = reinterpret_cast<float*>(kg_ptr);
    float* d_vg = reinterpret_cast<float*>(vg_ptr);

    auto mem_size = kittens::MAX_SHARED_MEMORY;
    auto threads  = 4 * kittens::WARP_THREADS;

    cudaDeviceSynchronize();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    cudaStreamSynchronize(stream);

    // TORCH_CHECK(seq_len % (4*kittens::TILE_DIM*4) == 0, "sequence length must be divisible by 256");
    dim3 grid_bwd(seq_len/(4*kittens::TILE_ROW_DIM<bf16>*4), qo_heads, batch);

    if (head_dim == 64)  {
        using og_tile = st_bf<4*16, 64>;
        using o_tile  = st_bf<4*16, 64>;
        using d_tile  = col_vec<st_fl<4*16, 64>>;

        using og_global = gl<bf16,  -1, -1, -1, -1, og_tile>;
        using o_global  = gl<bf16,  -1, -1, -1, -1, o_tile>;
        using d_global  = gl<float, -1, -1, -1, -1, d_tile>;

        using bwd_prep_globals = bwd_prep_globals<64>;

        og_global prep_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};
        o_global  prep_o_arg {d_o,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};
        d_global  prep_d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,  static_cast<unsigned int>(seq_len)};

        bwd_prep_globals bwd_g{prep_og_arg, prep_o_arg, prep_d_arg};

        cudaFuncSetAttribute(
            bwd_attend_prep_ker<64>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        bwd_attend_prep_ker<64><<<grid_bwd, threads, mem_size, stream>>>(bwd_g);

        using bwd_q_tile    =         st_bf<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_width>;
        using bwd_k_tile    =         st_bf<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>;
        using bwd_v_tile    =         st_bf<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>;
        using bwd_og_tile   =         st_bf<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_width>;
        using bwd_qg_tile   =         st_fl<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_width>;
        using bwd_kg_tile   =         st_fl<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>;
        using bwd_vg_tile   =         st_fl<bwd_attend_ker_tile_dims<64>::tile_h,    bwd_attend_ker_tile_dims<64>::tile_width>;
        using bwd_l_tile    = row_vec<st_fl<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_h>>;
        using bwd_d_tile    = row_vec<st_fl<bwd_attend_ker_tile_dims<64>::tile_h_qo, bwd_attend_ker_tile_dims<64>::tile_h>>;

        using bwd_q_global  = gl<bf16,  -1, -1, -1, -1, bwd_q_tile>;
        using bwd_k_global  = gl<bf16,  -1, -1, -1, -1, bwd_k_tile>;
        using bwd_v_global  = gl<bf16,  -1, -1, -1, -1, bwd_v_tile>;

        using bwd_og_global = gl<bf16,  -1, -1, -1, -1, bwd_og_tile>;

        using bwd_qg_global = gl<float, -1, -1, -1, -1, bwd_qg_tile>;
        using bwd_kg_global = gl<float, -1, -1, -1, -1, bwd_kg_tile>;
        using bwd_vg_global = gl<float, -1, -1, -1, -1, bwd_vg_tile>;

        using bwd_l_global  = gl<float, -1, -1, -1, -1, bwd_l_tile>;
        using bwd_d_global  = gl<float, -1, -1, -1, -1, bwd_d_tile>;

        using bwd_global_args = bwd_globals<64>;

        bwd_q_global  bwd_q_arg {d_q,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};
        bwd_k_global  bwd_k_arg {d_k,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 64U};
        bwd_v_global  bwd_v_arg {d_v,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 64U};
        bwd_og_global bwd_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};
        bwd_qg_global bwd_qg_arg{d_qg, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 64U};
        bwd_kg_global bwd_kg_arg{d_kg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 64U};
        bwd_vg_global bwd_vg_arg{d_vg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 64U};
        bwd_l_global  bwd_l_arg {d_l,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,  static_cast<unsigned int>(seq_len)};
        bwd_d_global  bwd_d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,  static_cast<unsigned int>(seq_len)};

        bwd_global_args bwd_global{bwd_q_arg,
                        bwd_k_arg,
                        bwd_v_arg,
                        bwd_og_arg,
                        bwd_qg_arg,
                        bwd_kg_arg,
                        bwd_vg_arg,
                        bwd_l_arg,
                        bwd_d_arg,
                        static_cast<int>(seq_len),
                        static_cast<int>(hr),
                        sm_scale};

        // TORCH_CHECK(seq_len % (4*BWD_CONSUMER_WARPGROUPS*kittens::TILE_DIM) == 0, "sequence length must be divisible by 128");
        dim3 grid_bwd_2(seq_len/(4*BWD_CONSUMER_WARPGROUPS*kittens::TILE_ROW_DIM<bf16>), qo_heads, batch);
        threads = kittens::WARP_THREADS * BWD_NUM_WORKERS;

        cudaDeviceSynchronize();

        if (is_causal) {
            cudaFuncSetAttribute(
                bwd_attend_ker<64, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                194000
            );
            cudaFuncSetAttribute(
                bwd_attend_ker<64, true>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                85
            );

            bwd_attend_ker<64, true><<<grid_bwd_2, threads, 194000, stream>>>(bwd_global);
        }
        else {
            cudaFuncSetAttribute(
                bwd_attend_ker<64, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                194000
            );
            cudaFuncSetAttribute(
                bwd_attend_ker<64, false>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                85
            );

            bwd_attend_ker<64, false><<<grid_bwd_2, threads, 194000, stream>>>(bwd_global);
        }

        // CHECK_CUDA_ERROR(cudaGetLastError());
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();
        // const auto kernel_end = std::chrono::high_resolution_clock::now();
        // std::cout << "Kernel Time: " << std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - start).count() << "us" << std::endl;
        // std::cout << "---" << std::endl;
    }

    if (head_dim == 128) {
        using og_tile = st_bf<4*16, 128>;
        using o_tile  = st_bf<4*16, 128>;
        using d_tile  = col_vec<st_fl<4*16, 128>>;

        using og_global = gl<bf16,  -1, -1, -1, -1, og_tile>;
        using o_global  = gl<bf16,  -1, -1, -1, -1, o_tile>;
        using d_global  = gl<float, -1, -1, -1, -1, d_tile>;

        using bwd_prep_globals = bwd_prep_globals<128>;

        og_global prep_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        o_global  prep_o_arg {d_o,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        d_global  prep_d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(seq_len)};

        bwd_prep_globals bwd_g{prep_og_arg, prep_o_arg, prep_d_arg};

        cudaFuncSetAttribute(
            bwd_attend_prep_ker<128>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        bwd_attend_prep_ker<128><<<grid_bwd, threads, mem_size, stream>>>(bwd_g);

        using bwd_q_tile    =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>;
        using bwd_k_tile    =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
        using bwd_v_tile    =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
        using bwd_og_tile   =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>;
        using bwd_qg_tile   =         st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>;
        using bwd_kg_tile   =         st_fl<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
        using bwd_vg_tile   =         st_fl<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
        using bwd_l_tile    = row_vec<st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_h>>;
        using bwd_d_tile    = row_vec<st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_h>>;

        using bwd_q_global  = gl<bf16,  -1, -1, -1, -1, bwd_q_tile>;
        using bwd_k_global  = gl<bf16,  -1, -1, -1, -1, bwd_k_tile>;
        using bwd_v_global  = gl<bf16,  -1, -1, -1, -1, bwd_v_tile>;

        using bwd_og_global = gl<bf16,  -1, -1, -1, -1, bwd_og_tile>;

        using bwd_qg_global = gl<float, -1, -1, -1, -1, bwd_qg_tile>;
        using bwd_kg_global = gl<float, -1, -1, -1, -1, bwd_kg_tile>;
        using bwd_vg_global = gl<float, -1, -1, -1, -1, bwd_vg_tile>;

        using bwd_l_global  = gl<float, -1, -1, -1, -1, bwd_l_tile>;
        using bwd_d_global  = gl<float, -1, -1, -1, -1, bwd_d_tile>;

        using bwd_global_args = bwd_globals<128>;

        bwd_q_global  bwd_q_arg {d_q,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_k_global  bwd_k_arg {d_k,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_v_global  bwd_v_arg {d_v,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_og_global bwd_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_qg_global bwd_qg_arg{d_qg, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_kg_global bwd_kg_arg{d_kg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_vg_global bwd_vg_arg{d_vg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
        bwd_l_global  bwd_l_arg {d_l,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(seq_len)};
        bwd_d_global  bwd_d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(seq_len)};

        bwd_global_args bwd_global{
                        bwd_q_arg,
                        bwd_k_arg,
                        bwd_v_arg,
                        bwd_og_arg,
                        bwd_qg_arg,
                        bwd_kg_arg,
                        bwd_vg_arg,
                        bwd_l_arg,
                        bwd_d_arg,
                        static_cast<int>(seq_len),
                        static_cast<int>(hr),
                        sm_scale};

        // TORCH_CHECK(seq_len % (4*BWD_CONSUMER_WARPGROUPS*kittens::TILE_DIM) == 0, "sequence length must be divisible by 128");
        dim3 grid_bwd_2(seq_len/(4*BWD_CONSUMER_WARPGROUPS*kittens::TILE_ROW_DIM<bf16>), qo_heads, batch);
        threads = kittens::WARP_THREADS * BWD_NUM_WORKERS;

        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();

        if (is_causal) {
            cudaFuncSetAttribute(
                bwd_attend_ker<128, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                194000
            );
            cudaFuncSetAttribute(
                bwd_attend_ker<128, true>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                85
            );

            bwd_attend_ker<128, true><<<grid_bwd_2, threads, 194000, stream>>>(bwd_global);
        }
        else {
            cudaFuncSetAttribute(
                bwd_attend_ker<128, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                194000
            );
            cudaFuncSetAttribute(
                bwd_attend_ker<128, false>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                85
            );

            bwd_attend_ker<128, false><<<grid_bwd_2, threads, 194000, stream>>>(bwd_global);
        }

        // CHECK_CUDA_ERROR(cudaGetLastError());
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();
    }

    return {qg, kg, vg};
    cudaDeviceSynchronize();
}

} // namespace fa_h100


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fa_h100::attention_forward, "Forward pass");
    m.def("bwd", &fa_h100::attention_backward, "Backward pass");
}