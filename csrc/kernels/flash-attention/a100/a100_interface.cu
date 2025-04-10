#include "a100_common.cuh"
#include "pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include "static_switch.h"
#include "fa_switch.h"

namespace fa_a100 {

using namespace kittens;

std::vector<torch::Tensor>
attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, float sm_scale)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    auto batch = static_cast<const uint>(q.size(0));
    auto seq_len = static_cast<const uint>(q.size(2));
    auto head_dim = static_cast<const uint>(q.size(3));
    auto is_causal = static_cast<const bool>(causal);
    auto qo_heads = static_cast<const uint>(q.size(1));
    auto kv_heads = static_cast<const uint>(k.size(1));

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
    torch::Tensor o = torch::empty({batch, qo_heads, seq_len, head_dim}, v.options());

    torch::Tensor l_vec = torch::empty({batch, qo_heads, seq_len, 1},
                                        torch::TensorOptions().dtype(torch::kFloat).device(q.device()).memory_format(at::MemoryFormat::Contiguous));


    bf16* o_ptr = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16* d_o = reinterpret_cast<bf16*>(o_ptr);

    float* l_ptr = reinterpret_cast<float*>(l_vec.data_ptr<float>());
    float* d_l = reinterpret_cast<float*>(l_ptr);

    HEAD_DIM_SWITCH(head_dim, HEAD_DIM, [&] {
        BOOL_SWITCH(is_causal, is_causal_true, [&] {

            cudaDeviceSynchronize();
            auto stream = at::cuda::getCurrentCUDAStream().stream();

            using q_tile = st_bf<fwd_attend_ker_tile_dims<HEAD_DIM>::qo_height, fwd_attend_ker_tile_dims<HEAD_DIM>::tile_width>;
            using k_tile = st_bf<fwd_attend_ker_tile_dims<HEAD_DIM>::kv_height, fwd_attend_ker_tile_dims<HEAD_DIM>::tile_width>;
            using v_tile = st_bf<fwd_attend_ker_tile_dims<HEAD_DIM>::kv_height, fwd_attend_ker_tile_dims<HEAD_DIM>::tile_width>;
            using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<HEAD_DIM>::qo_height, fwd_attend_ker_tile_dims<HEAD_DIM>::tile_width>>;
            using o_tile = st_bf<fwd_attend_ker_tile_dims<HEAD_DIM>::qo_height, fwd_attend_ker_tile_dims<HEAD_DIM>::tile_width>;

            using q_global = gl<bf16,  -1, -1, -1, -1, q_tile>;
            using k_global = gl<bf16,  -1, -1, -1, -1, k_tile>;
            using v_global = gl<bf16,  -1, -1, -1, -1, v_tile>;
            using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
            using o_global = gl<bf16,  -1, -1, -1, -1, o_tile>;

            using globals = fwd_globals<HEAD_DIM>;

            q_global qg_arg{d_q, batch, qo_heads, seq_len, head_dim};
            k_global kg_arg{d_k, batch, kv_heads, seq_len, head_dim};
            v_global vg_arg{d_v, batch, kv_heads, seq_len, head_dim};
            l_global lg_arg{d_l, batch, qo_heads, 1U, seq_len};
            o_global og_arg{d_o, batch, qo_heads, seq_len, head_dim};

            globals g{qg_arg, kg_arg, vg_arg, lg_arg, og_arg, seq_len, hr, sm_scale};

            auto smem_size = kittens::MAX_SHARED_MEMORY;
            auto threads  = FWD_NUM_WORKERS * kittens::WARP_THREADS;

            dim3 grid(seq_len/(FWD_NUM_WORKERS*kittens::TILE_ROW_DIM<bf16>*4), qo_heads, batch);

            cudaFuncSetAttribute(
                fwd_attend_ker<HEAD_DIM, is_causal_true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size
            );

            fwd_attend_ker<HEAD_DIM, is_causal_true><<<grid, (32*FWD_NUM_WORKERS), smem_size, stream>>>(g);
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaStreamSynchronize(stream);
            cudaDeviceSynchronize();
        });
    });

    return {o, l_vec};
}

} // namespace fa_a100


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fa_a100::attention_forward, "Forward pass");
}