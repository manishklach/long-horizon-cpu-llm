"""Head-by-head attention (cache-friendly FlashAttention-style for CPU).

Idea: basic unit = one token on one KV head. Finish one head fully before
moving to the next, so that head's KV stays resident in L3 instead of being
evicted by parallel-head access. Matches CPU profile: few cores, huge cache.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def repeat_kv_for_gqa(kv: torch.Tensor, n_heads: int) -> torch.Tensor:
    """kv: [L_kv, H_kv, D] -> [L_kv, H, D] by repeating for grouped-query attention."""
    L, Hkv, D = kv.shape
    if Hkv == n_heads:
        return kv
    assert n_heads % Hkv == 0
    return kv.repeat_interleave(n_heads // Hkv, dim=1)


def attention_head_by_head(
    q: torch.Tensor,  # [L_q, H, D]
    k: torch.Tensor,  # [L_kv, H_kv, D]
    v: torch.Tensor,  # [L_kv, H_kv, D]
    causal: bool = True,
    kv_offset: int = 0,  # absolute start pos of k/v in full sequence (for incremental prefill)
) -> torch.Tensor:
    """Sequential per-head SDPA. Returns [L_q, H, D]."""
    Lq, H, D = q.shape
    k_rep = repeat_kv_for_gqa(k, H)  # [Lkv, H, D]
    v_rep = repeat_kv_for_gqa(v, H)
    Lk = k_rep.shape[0]
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(D)
    for h in range(H):  # one head resident at a time
        qh = q[:, h, :]            # [Lq, D]
        kh = k_rep[:, h, :]        # [Lk, D]
        vh = v_rep[:, h, :]        # [Lk, D]
        scores = (qh @ kh.T) * scale  # [Lq, Lk]
        if causal:
            # causal against absolute positions: query row i is abs pos kv_offset+i
            q_pos = torch.arange(Lq) + kv_offset          # [Lq]
            kv_pos = torch.arange(Lk)                      # [Lk] (k already prefix+new)
            mask = kv_pos.unsqueeze(0) > q_pos.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))
        out[:, h, :] = torch.softmax(scores, dim=-1) @ vh
    return out


def attention_batched(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
) -> torch.Tensor:
    """Conventional parallel-head baseline for comparison."""
    Lq, H, D = q.shape
    k_rep = repeat_kv_for_gqa(k, H).transpose(0, 1)  # [H, Lk, D]
    v_rep = repeat_kv_for_gqa(v, H).transpose(0, 1)
    qq = q.transpose(0, 1)  # [H, Lq, D]
    scores = (qq @ k_rep.transpose(1, 2)) / math.sqrt(D)
    if causal:
        Lk = k_rep.shape[1]
        mask = torch.triu(torch.ones(Lq, Lk, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    return (torch.softmax(scores, dim=-1) @ v_rep).transpose(0, 1)


def ortho_check(n_heads: int = 8, head_dim: int = 64, tol: float = 1e-4) -> float:
    q = torch.randn(16, n_heads, head_dim)
    k = torch.randn(16, 4, head_dim)
    v = torch.randn(16, 4, head_dim)
    a = attention_head_by_head(q, k, v)
    b = attention_batched(q, k, v)
    return float((a - b).abs().max())
