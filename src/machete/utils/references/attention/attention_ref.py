from typing import Optional
import torch
import math

def attn_ref(q: torch.Tensor,
             k: torch.Tensor,
             v: torch.Tensor,
             b: Optional[torch.Tensor] = None,
             sm_scale: Optional[float] = None,
             dropout_p: float = 0.0,
             causal: bool = False,
             upcast: bool = False) -> torch.Tensor:
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
        if b is not None:
            b = b.float()

    if b is not None:
        if (b.shape[0] != q.shape[0]) or (b.shape[1] != q.shape[1]):
            b = b.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[3])

    ms = torch.arange(q.shape[2], device=q.device).unsqueeze(-1)
    ns = torch.arange(k.shape[2], device=q.device)

    p = torch.matmul(q, k.transpose(2, 3))
    p *= sm_scale
    if b is not None:
        p += b

    if causal:
        p = torch.where(ms + k.shape[2] - q.shape[2] >= ns, p, float("-inf"))

    max_score = torch.max(p, dim=-1, keepdim=True)[0]
    exp_p = torch.exp(p - max_score)
    sum_exp_p = torch.sum(exp_p, dim=-1, keepdim=True)
    l_vec = torch.log(sum_exp_p) + max_score

    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    if dropout_p > 0.0:
        p = torch.dropout(p, dropout_p, train=True)

    ref_out = torch.matmul(p, v)
    return ref_out, l_vec
