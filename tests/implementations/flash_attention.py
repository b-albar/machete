import torch
import pytest
from machete.kernels.flash_attention import flash_attention
from machete.utils.references.attention.attention_ref import attn_ref

def max_diff(a, b):
    return (a - b).abs().max().item()

@pytest.mark.parametrize('scale', [1.0])
@pytest.mark.parametrize('b, h, m, n, d', [
    (2, 4, 512, 612, 128),
    (2, 4, 1024, 1045, 64),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_attention_fwd(b, h, m, n, d, causal, dtype, scale):
    q = torch.empty((b, h, m, d), dtype=dtype, device="cuda").normal_(mean=0., std=scale)
    k = torch.empty((b, h, n, d), dtype=dtype, device="cuda").normal_(mean=0., std=scale)
    v = torch.empty((b, h, n, d), dtype=dtype, device="cuda").normal_(mean=0., std=scale)

    o_ref = attn_ref(q, k, v, causal=causal, sm_scale=scale, upcast=True)
    o_torch = attn_ref(q, k, v, causal=causal, sm_scale=scale, upcast=False)
    o_hyp = flash_attention(q, k, v, causal, scale)

    torch_max_diff = max_diff(o_torch, o_ref)
    machete_max_diff = max_diff(o_hyp, o_ref)
    assert machete_max_diff <= 2 * torch_max_diff + 1e-5
