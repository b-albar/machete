#include "a100_common_fwd.cuh"
#include "a100_common_bwd.cuh"
#include "pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include "static_switch.h"
#include "fa_switch.h"

namespace fa_a100 {

using namespace kittens;

{{variant_declarations}}

std::vector<torch::Tensor>
attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    {{variant_inputs}}
    bool causal,
    float sm_scale
)
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

    {{variant_fwd_initialization}}

    // for the returned outputs
    torch::Tensor o = torch::empty({batch, qo_heads, seqlen_q, head_dim}, q.options());
    bf16* o_ptr = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());

    torch::Tensor l_vec;
    float *l_ptr;
    if constexpr (variant.is_softmax) {
        l_vec = torch::empty({batch, qo_heads, 1, seqlen_q},
                            torch::TensorOptions().dtype(torch::kFloat).device(q.device()).memory_format(at::MemoryFormat::Contiguous));
        l_ptr = reinterpret_cast<float*>(l_vec.data_ptr<float>());
    }

    using ker_tile_dims = fwd_ker_tile_dims<HEAD_DIM, QO_SIZE, KV_SIZE, STAGES>;

    BOOL_SWITCH(is_causal, IS_CAUSAL, [&] {

        cudaDeviceSynchronize();
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        using globals = fwd_globals<HEAD_DIM, QO_SIZE, KV_SIZE, STAGES>;

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

        {{variant_globals_decl}}

        globals g{
            qg_arg,
            kg_arg,
            vg_arg,
            lg_arg,
            og_arg,
            {{variant_globals_init}}
            seqlen_q,
            seqlen_k,
            qo_heads,
            kv_heads,
            sm_scale
        };

        int max_smem_size;
        cudaDeviceGetAttribute(&max_smem_size, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

        int smem_size = g.get_smem_size<{{variant_fwd_name}}>(variant);

        assert(smem_size <= max_smem_size);

        auto threads = FWD_NUM_WORKERS * kittens::WARP_THREADS;

        int q_blocks = (seqlen_q + (FWD_NUM_WORKERS*ker_tile_dims::qo_height) - 1) / (FWD_NUM_WORKERS*ker_tile_dims::qo_height);

        dim3 grid(q_blocks, qo_heads, batch);

        cudaFuncSetAttribute(
            fwd_attend_ker<HEAD_DIM, QO_SIZE, KV_SIZE, STAGES, IS_CAUSAL, {{variant_fwd_name}}>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );

        fwd_attend_ker<HEAD_DIM, QO_SIZE, KV_SIZE, STAGES, IS_CAUSAL, {{variant_fwd_name}}><<<grid, threads, smem_size, stream>>>(g, variant);
        CHECK_CUDA_ERROR(cudaGetLastError());
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();
    });

    return {
        o,
        l_vec,
        {{variant_outputs}}
    };
}

std::vector<torch::Tensor>
attention_backward(torch::Tensor q,
                   torch::Tensor k,
                   torch::Tensor v,
                   torch::Tensor o,
                   torch::Tensor l_vec,
                   torch::Tensor og,
                   bool is_causal,
                   float sm_scale)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(l_vec);
    CHECK_INPUT(o);
    CHECK_INPUT(og);

    auto batch = q.size(0);
    auto seqlen_q = q.size(2);
    auto seqlen_k = k.size(2);
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);
    auto head_dim = q.size(3);

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(q.size(0) == batch, "Q batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0) == batch, "K batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0) == batch, "V batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(l_vec.size(0) == batch, "L batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(o.size(0) == batch, "O batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(og.size(0) == batch, "OG batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q.size(2) == seqlen_q, "Q sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(k.size(2) == seqlen_k, "K sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(v.size(2) == seqlen_k, "V sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(l_vec.size(3) == seqlen_q, "L sequence length dimension - idx 3 - must match for all inputs");
    TORCH_CHECK(o.size(2) == seqlen_q, "O sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(og.size(2) == seqlen_q, "OG sequence length dimension - idx 2 - must match for all inputs");

    TORCH_CHECK(q.size(3) == head_dim, "Q head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3) == head_dim, "K head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3) == head_dim, "V head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(o.size(3) == head_dim, "O head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(og.size(3) == head_dim, "OG head dimension - idx 3 - must match for all non-vector inputs");

    TORCH_CHECK(qo_heads >= kv_heads, "Q heads must be greater than or equal to K and V heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "Q heads must be divisible by KV heads");

    TORCH_CHECK(q.size(1) == qo_heads, "Q heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(l_vec.size(1) == qo_heads, "L heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(o.size(1) == qo_heads, "O heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(og.size(1) == qo_heads, "OG heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k.size(1) == kv_heads, "K heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v.size(1) == kv_heads, "V heads dimension - idx 1 - must match for all inputs");

    // Initialize tensors for the gradients
    torch::Tensor qg = torch::zeros_like(q, q.options());
    torch::Tensor kg = torch::zeros_like(k, k.options());
    torch::Tensor vg = torch::zeros_like(v, v.options());
    torch::Tensor d_vec = torch::zeros_like(l_vec);

    bf16* d_q = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    bf16* d_k = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    bf16* d_v = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
    bf16* d_o = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());

    bf16* d_og = reinterpret_cast<bf16*>(og.data_ptr<c10::BFloat16>());
    bf16* d_qg = reinterpret_cast<bf16*>(qg.data_ptr<c10::BFloat16>());
    bf16* d_kg = reinterpret_cast<bf16*>(kg.data_ptr<c10::BFloat16>());
    bf16* d_vg = reinterpret_cast<bf16*>(vg.data_ptr<c10::BFloat16>());

    float* d_l = reinterpret_cast<float*>(l_vec.data_ptr<float>());
    float* d_d = reinterpret_cast<float*>(d_vec.data_ptr<float>());

    {{variant_bwd_initialization}}

    using ker_tile_dims_prep = bwd_prep_ker_tile_dims<HEAD_DIM, QO_SIZE, KV_SIZE>;
    using ker_tile_dims = bwd_ker_tile_dims<HEAD_DIM, QO_SIZE, KV_SIZE, STAGES>;

    BOOL_SWITCH(is_causal, IS_CAUSAL, [&] {

        cudaDeviceSynchronize();
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        auto threads = 0;
        auto seqlen_per_worker = 0;
        int smem_size = 0;

        int max_smem_size;
        cudaDeviceGetAttribute(&max_smem_size, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

        // prepare and launch the backward preparation kernel
        using bwd_prep_globals = bwd_prep_globals<HEAD_DIM, QO_SIZE, KV_SIZE>;

        using o_prep_global = bwd_prep_globals::o_gl;
        using og_prep_global = bwd_prep_globals::og_gl;
        using d_prep_global = bwd_prep_globals::d_gl;

        o_prep_global o_prep_arg{d_o, batch, qo_heads, seqlen_q, head_dim};
        og_prep_global og_prep_arg{d_og, batch, qo_heads, seqlen_q, head_dim};
        d_prep_global d_prep_arg{d_d, batch, qo_heads, 1U, seqlen_q};

        bwd_prep_globals pg{o_prep_arg, og_prep_arg, d_prep_arg};

        smem_size = pg.get_smem_size();
        assert(smem_size <= max_smem_size);

        threads = BWD_PREP_NUM_WORKERS * kittens::WARP_THREADS;
        seqlen_per_worker = BWD_PREP_NUM_WORKERS * ker_tile_dims_prep::qo_height;

        int q_blocks = (seqlen_q + seqlen_per_worker - 1) / seqlen_per_worker;

        dim3 grid_prep(q_blocks, qo_heads, batch);

        cudaFuncSetAttribute(
            bwd_prep_ker<HEAD_DIM, QO_SIZE, KV_SIZE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );

        bwd_prep_ker<HEAD_DIM, QO_SIZE, KV_SIZE><<<grid_prep, threads, smem_size, stream>>>(pg);

        // launch the main backward kernel
        using bwd_globals = bwd_globals<HEAD_DIM, QO_SIZE, KV_SIZE, STAGES>;

        using q_global = bwd_globals::q_gl;
        using k_global = bwd_globals::k_gl;
        using v_global = bwd_globals::v_gl;
        using og_global = bwd_globals::og_gl;

        using qg_global = bwd_globals::qg_gl;
        using kg_global = bwd_globals::kg_gl;
        using vg_global = bwd_globals::vg_gl;

        using d_global = bwd_globals::d_gl;
        using l_global = bwd_globals::l_gl;

        q_global q_arg{d_q, batch, qo_heads, seqlen_q, head_dim};
        k_global k_arg{d_k, batch, kv_heads, seqlen_k, head_dim};
        v_global v_arg{d_v, batch, kv_heads, seqlen_k, head_dim};
        og_global og_arg{d_og, batch, qo_heads, seqlen_q, head_dim};

        qg_global qg_arg{d_qg, batch, qo_heads, seqlen_q, head_dim};
        kg_global kg_arg{d_kg, batch, kv_heads, seqlen_k, head_dim};
        vg_global vg_arg{d_vg, batch, kv_heads, seqlen_k, head_dim};

        d_global d_arg{d_d, batch, qo_heads, 1U, seqlen_q};
        l_global l_arg{d_l, batch, qo_heads, 1U, seqlen_q};

        bwd_globals g{
            q_arg, k_arg, v_arg,
            og_arg, qg_arg, kg_arg, vg_arg,
            l_arg, d_arg,
            seqlen_q, seqlen_k, qo_heads, kv_heads, sm_scale
        };

        smem_size = g.get_smem_size(variant);
        assert(smem_size <= max_smem_size);

        threads = BWD_NUM_WORKERS * kittens::WARP_THREADS;
        seqlen_per_worker = BWD_NUM_WORKERS * ker_tile_dims::kv_height;

        int k_blocks = (seqlen_k + seqlen_per_worker - 1) / seqlen_per_worker;

        dim3 grid_bwd(k_blocks, kv_heads, batch);

        cudaFuncSetAttribute(
            bwd_attend_ker<HEAD_DIM, QO_SIZE, KV_SIZE, STAGES, IS_CAUSAL, {{variant_bwd_name}}>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );

        bwd_attend_ker<HEAD_DIM, QO_SIZE, KV_SIZE, STAGES, IS_CAUSAL, {{variant_bwd_name}}><<<grid_bwd, threads, smem_size, stream>>>(g, variant);

        CHECK_CUDA_ERROR(cudaGetLastError());
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();

    });

    return {qg, kg, vg};
}

} // namespace fa_a100

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fa_a100::attention_forward, "Forward pass");
    m.def("bwd", &fa_a100::attention_backward, "Backward pass");
}