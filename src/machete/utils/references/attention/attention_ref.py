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
             upcast: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
        if b is not None:
            b = b.float()

    if b is not None:
        if (b.shape[0] != q.shape[0]) or (b.shape[1] != q.shape[1]):
            b = b.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[3])

    p = torch.matmul(q, k.transpose(2, 3))
    p *= sm_scale
    if b is not None:
        p += b

    if causal:
        ms = torch.arange(q.shape[2], device=q.device).unsqueeze(-1)
        ns = torch.arange(k.shape[2], device=q.device)
        p = torch.where(ms >= ns, p, float("-inf"))

    max_score = torch.max(p, dim=-1, keepdim=True)[0]
    exp_p = torch.exp(p - max_score)
    sum_exp_p = torch.sum(exp_p, dim=-1, keepdim=True)
    l_vec = torch.log(sum_exp_p) + max_score

    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    if dropout_p > 0.0:
        p = torch.dropout(p, dropout_p, train=True)

    ref_out = torch.matmul(p, v)
    return ref_out, l_vec

def attn_bwd_ref(q: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor,
                 o: torch.Tensor,
                 og: torch.Tensor,
                 l: torch.Tensor,
                 b: Optional[torch.Tensor] = None,
                 sm_scale: Optional[float] = None,
                 causal: bool = False,
                 upcast: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if upcast:
        q, k, v, o, og = q.float(), k.float(), v.float(), o.float(), og.float()
        if b is not None:
            b = b.float()

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[3])

    # Compute attention scores (needed for softmax backward)
    s = torch.matmul(q, k.transpose(2, 3) * sm_scale)

    # Compute attention probabilities using provided l tensor
    # l contains log(sum(exp(s))), so exp(s - l) gives us the probabilities
    prob = torch.exp(s - l).to(og.dtype)

    if causal:
        # Zero out probabilities for masked positions
        ms = torch.arange(q.shape[2], device=q.device).unsqueeze(-1)
        ns = torch.arange(k.shape[2], device=q.device)
        prob = torch.where(ms >= ns, prob, 0.0)

    # Backward pass gradients
    # Gradient w.r.t. v: dv = P^T @ og
    dv = torch.matmul(prob.transpose(2, 3), og)

    # Gradient w.r.t. attention weights: dp = og @ v^T
    dp = torch.matmul(og, v.transpose(2, 3))
    # Gradient w.r.t. scores (softmax backward):
    # ds = P * (dp - sum(P * dp, dim=-1, keepdim=True))
    ds = prob * (dp - torch.sum(prob * dp, dim=-1, keepdim=True))

    if causal:
        # Zero out gradients for masked positions
        ms = torch.arange(q.shape[2], device=q.device).unsqueeze(-1)
        ns = torch.arange(k.shape[2], device=q.device)
        ds = torch.where(ms >= ns, ds, 0.0)

    # Gradient w.r.t. q: dq = ds @ k * sm_scale
    dq = torch.matmul(ds, k * sm_scale)

    # Gradient w.r.t. k: dk = ds^T @ q * sm_scale
    dk = torch.matmul(ds.transpose(2, 3), q) * sm_scale

    return dq, dk, dv










