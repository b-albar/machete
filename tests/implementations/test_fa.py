import torch
from machete.kernels.flash_attention import flash_attention
from machete.utils.references.attention.attention_ref import attn_ref

def max_diff(a, b):
    return (a - b).abs().max().item()

b = 4
h = 8
n = 128
m = 128
d = 64
causal = True
scale = 1.0
dtype = torch.bfloat16

q = torch.empty((b, h, m, d), dtype=dtype, device="cuda").normal_(mean=0., std=scale)
k = torch.empty((b, h, n, d), dtype=dtype, device="cuda").normal_(mean=0., std=scale)
v = torch.empty((b, h, n, d), dtype=dtype, device="cuda").normal_(mean=0., std=scale)

o_ref = attn_ref(q, k, v, causal=causal, sm_scale=scale, upcast=True)
o_torch = attn_ref(q, k, v, causal=causal, sm_scale=scale, upcast=False)
o_hyp = flash_attention(q, k, v, causal, scale)

torch_max_diff = max_diff(o_torch, o_ref)
machete_max_diff = max_diff(o_hyp, o_ref)

print(f"torch_max_diff: {torch_max_diff}")
print(f"machete_max_diff: {machete_max_diff}")