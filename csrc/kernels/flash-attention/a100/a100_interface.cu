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
    auto seqlen_q = static_cast<const uint>(q.size(2));
    auto seqlen_k = static_cast<const uint>(k.size(2));
    auto head_dim = static_cast<const uint>(q.size(3));
    auto is_causal = static_cast<const bool>(causal);
    auto qo_heads = static_cast<const uint>(q.size(1));
    auto kv_heads = static_cast<const uint>(k.size(1));

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(q.size(0) == batch, "Q batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0) == batch, "K batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0) == batch, "V batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q.size(3) == head_dim, "Q head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3) == head_dim, "K head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3) == head_dim, "V head dimension - idx 3 - must match for all non-vector inputs");

    TORCH_CHECK(qo_heads >= kv_heads, "QO heads must be greater than or equal to KV heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "QO heads must be divisible by KV heads");
    TORCH_CHECK(q.size(1) == qo_heads, "QO head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");

    bf16* q_ptr = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    bf16* k_ptr = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    bf16* v_ptr = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());

    // for the returned outputs
    torch::Tensor o = torch::empty({batch, qo_heads, seqlen_q, head_dim}, v.options());
    torch::Tensor l_vec = torch::empty({batch, qo_heads, seqlen_q, 1},
                                        torch::TensorOptions().dtype(torch::kFloat).device(q.device()).memory_format(at::MemoryFormat::Contiguous));


    bf16* o_ptr = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    float* l_ptr = reinterpret_cast<float*>(l_vec.data_ptr<float>());

    HEAD_DIM_SWITCH(head_dim, HEAD_DIM, [&] {
        using ker_tile_dims = fwd_attend_ker_tile_dims<HEAD_DIM>;
        const bool is_even_nm = (seqlen_q % ker_tile_dims::qo_height) == 0 && (seqlen_k % ker_tile_dims::kv_height) == 0;

        BOOL_SWITCH(is_causal, IS_CAUSAL, [&] {
            BOOL_SWITCH(is_even_nm, IS_EVEN_NM, [&] {

                cudaDeviceSynchronize();
                auto stream = at::cuda::getCurrentCUDAStream().stream();

                using globals = fwd_globals<HEAD_DIM>;

                using q_global = globals::q_gl;
                using k_global = globals::k_gl;
                using v_global = globals::v_gl;
                using l_global = globals::l_gl;
                using o_global = globals::o_gl;

                q_global qg_arg{q_ptr, batch, qo_heads, seqlen_q, head_dim};
                k_global kg_arg{k_ptr, batch, kv_heads, seqlen_k, head_dim};
                v_global vg_arg{v_ptr, batch, kv_heads, seqlen_k, head_dim};
                l_global lg_arg{l_ptr, batch, qo_heads, 1U, seqlen_q};
                o_global og_arg{o_ptr, batch, qo_heads, seqlen_q, head_dim};

                globals g{qg_arg, kg_arg, vg_arg, lg_arg, og_arg, seqlen_q, seqlen_k, qo_heads, kv_heads, sm_scale};

                int max_smem_size;
                cudaDeviceGetAttribute(&max_smem_size, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

                int smem_size = g.get_smem_size();

                assert(smem_size <= max_smem_size);

                auto threads = FWD_NUM_WORKERS * kittens::WARP_THREADS;

                int q_blocks = (seqlen_q + (FWD_NUM_WORKERS*ker_tile_dims::qo_height)/2) / (FWD_NUM_WORKERS*ker_tile_dims::qo_height);

                dim3 grid(q_blocks, qo_heads, batch);

                cudaFuncSetAttribute(
                    fwd_attend_ker<HEAD_DIM, IS_CAUSAL, IS_EVEN_NM>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_size
                );

                fwd_attend_ker<HEAD_DIM, IS_CAUSAL, IS_EVEN_NM><<<grid, threads, smem_size, stream>>>(g);
                CHECK_CUDA_ERROR(cudaGetLastError());
                cudaStreamSynchronize(stream);
                cudaDeviceSynchronize();
            });
        });
    });

    return {o, l_vec};
}

} // namespace fa_a100


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fa_a100::attention_forward, "Forward pass");
}