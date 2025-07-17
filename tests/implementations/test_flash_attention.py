import torch
import pytest
from machete.kernels.flash_attention import flash_attention
from machete.utils.references.attention.attention_ref import attn_ref, attn_bwd_ref

#from flash_attn import flash_attn_func

def max_diff(a, b):
    return (a - b).abs().max().item()

@pytest.mark.parametrize('scale', [0.125, 1.0])
@pytest.mark.parametrize('b', [1, 2, 3, 4])
@pytest.mark.parametrize('h', [4, 5, 8, 16])
@pytest.mark.parametrize('m', [64, 128, 256])
@pytest.mark.parametrize('n', [64, 128, 256])
@pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
def test_attention_fwd(b, h, m, n, d, causal, dtype, scale):
    q = torch.empty((b, h, m, d), dtype=dtype, device="cuda").normal_(mean=0., std=0.1)
    k = torch.empty((b, h, n, d), dtype=dtype, device="cuda").normal_(mean=0., std=0.1)
    v = torch.empty((b, h, n, d), dtype=dtype, device="cuda").normal_(mean=0., std=0.1)

    o_ref, l_vec_ref = attn_ref(q, k, v, causal=causal, sm_scale=scale, upcast=True)
    o_torch, l_vec_torch = attn_ref(q, k, v, causal=causal, sm_scale=scale, upcast=False)
    #o_spda = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    #o_fa2 = flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
    o_hyp, l_vec_hyp = flash_attention(q, k, v, causal, scale)

    machete_max_diff = max_diff(o_ref, o_hyp)
    ref_max_diff = max_diff(o_ref, o_torch)

    l_vec_max_diff = max_diff(l_vec_ref, l_vec_hyp)
    l_vec_ref_max_diff = max_diff(l_vec_ref, l_vec_torch)

    assert machete_max_diff < 2 * ref_max_diff
    assert l_vec_max_diff < 2 * l_vec_ref_max_diff

@pytest.mark.parametrize('scale', [0.125])
@pytest.mark.parametrize('b', [2, 3, 5])
@pytest.mark.parametrize('h', [4, 16])
@pytest.mark.parametrize('m', [128])
@pytest.mark.parametrize('n', [128])
@pytest.mark.parametrize('d', [64, 128])
@pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
def test_attention_bwd(b, h, m, n, d, causal, dtype, scale):
    torch.manual_seed(0)
    # Create tensors with gradients enabled
    q = torch.empty((b, h, m, d), dtype=dtype, device="cuda").normal_(mean=0., std=0.1).requires_grad_(True)
    k = torch.empty((b, h, n, d), dtype=dtype, device="cuda").normal_(mean=0., std=0.1).requires_grad_(True)
    v = torch.empty((b, h, n, d), dtype=dtype, device="cuda").normal_(mean=0., std=0.1).requires_grad_(True)

    # Reference implementation tensors
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    q_torch = q.detach().clone().requires_grad_(True)
    k_torch = k.detach().clone().requires_grad_(True)
    v_torch = v.detach().clone().requires_grad_(True)

    # Our implementation
    o_hyp, l_vec = flash_attention(q, k, v, causal, scale)

    # Reference implementation
    o_ref, l_vec_ref = attn_ref(q_ref, k_ref, v_ref, causal=causal, sm_scale=scale, upcast=True)
    o_torch, l_vec_torch = attn_ref(q_torch, k_torch, v_torch, causal=causal, sm_scale=scale, upcast=False)

    # Create gradient for backprop
    grad_output = torch.randn_like(o_hyp).normal_(mean=0., std=0.1) * 100
    grad_output_ref = grad_output.clone()
    grad_output_torch = grad_output.clone()

    # Backward pass
    q_grad, k_grad, v_grad = torch.ops.machete.flash_attention_bwd(grad_output, o_hyp, q, k, v, l_vec, causal, scale)

    dq_ref, dk_ref, dv_ref = attn_bwd_ref(
        q, k, v, o_ref, grad_output_ref, l_vec.permute(0, 1, 3, 2), b=None, causal=causal, sm_scale=scale, upcast=True
    )
    dq_torch, dk_torch, dv_torch = attn_bwd_ref(
        q_torch,
        k_torch,
        v_torch,
        o_torch,
        grad_output_torch,
        l_vec.permute(0, 1, 3, 2),
        b=None,
        causal=causal,
        sm_scale=scale,
        upcast=False
    )

    dq_diff = max_diff(q_grad, dq_ref)
    dk_diff = max_diff(k_grad, dk_ref)
    dv_diff = max_diff(v_grad, dv_ref)

    dq_diff_ref = max_diff(dq_ref, dq_torch)
    dk_diff_ref = max_diff(dk_ref, dk_torch)
    dv_diff_ref = max_diff(dv_ref, dv_torch)

    # Assert that gradients are close enough
    assert dq_diff < 2 * dq_diff_ref, f"dQ difference too large: {dq_diff} vs {dq_diff_ref}"
    assert dk_diff < 2 * dk_diff_ref, f"dK difference too large: {dk_diff} vs {dk_diff_ref}"
    assert dv_diff < 2 * dv_diff_ref, f"dV difference too large: {dv_diff} vs {dv_diff_ref}"

test_attention_fwd(3, 16, 128, 128, 64, False, torch.bfloat16, 0.125)
#test_attention_bwd(3, 16, 128, 128, 64, False, torch.bfloat16, 0.125)
