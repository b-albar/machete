import torch
import pytest
from machete.kernels.flash_attention import flash_attention
from machete.utils.references.attention.attention_ref import attn_ref

from flash_attn import flash_attn_func

def max_diff(a, b):
    return (a - b).abs().max().item()

@pytest.mark.parametrize('scale', [1.0])
@pytest.mark.parametrize('b, h, m, n, d', [
    (2, 4, 512, 512, 64),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
def test_attention_fwd(b, h, m, n, d, causal, dtype, scale):
    q = torch.empty((b, h, m, d), dtype=dtype, device="cuda").normal_(mean=0., std=scale)
    k = torch.empty((b, h, n, d), dtype=dtype, device="cuda").normal_(mean=0., std=scale)
    v = torch.empty((b, h, n, d), dtype=dtype, device="cuda").normal_(mean=0., std=scale)

    o_ref, _ = attn_ref(q, k, v, causal=causal, sm_scale=scale, upcast=True)
    o_torch, l_torch = attn_ref(q, k, v, causal=causal, sm_scale=scale, upcast=False)
    #o_spda = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    #o_fa2 = flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
    o_hyp = flash_attention(q, k, v, causal, scale)

    machete_max_diff = max_diff(o_ref, o_hyp)
    ref_max_diff = max_diff(o_ref, o_torch)

    assert machete_max_diff < 2 * ref_max_diff

test_attention_fwd(1, 1, 512, 512, 128, False, torch.bfloat16, 0.125)
