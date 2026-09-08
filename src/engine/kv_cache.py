"""eLLM-style CPU inference — Python prototype.

Static KV cache with dimension-first layout, preallocated max_seq.
Mirrors eLLM's "elastic static graph + non-paged KV + massive-dim + session cache".
Shape convention: [max_seq, n_kv_heads, head_dim] per layer, seq-dim first
so reads along seq are contiguous (good spatial locality on CPU).
"""
from __future__ import annotations
import os
import torch


class StaticKVCache:
    """Preallocated, non-paged KV cache for batch=1.

    - One (K, V) pair per layer, fixed shape, no reallocation.
    - Direct coordinate indexing: write at [seq_pos:seq_pos+n].
    - Contiguous along seq dim for cache-friendly head-by-head attention.
    """

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int,
                 max_seq_len: int = 50000, dtype: torch.dtype = torch.float32):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        # dimension-first: seq is dim 0 -> contiguous slice per head
        self.K = [torch.zeros(max_seq_len, n_kv_heads, head_dim, dtype=dtype)
                  for _ in range(n_layers)]
        self.V = [torch.zeros(max_seq_len, n_kv_heads, head_dim, dtype=dtype)
                  for _ in range(n_layers)]
        self.seq_len = 0  # logical fill pointer, graph never rebuilds

    def __len__(self):
        return self.seq_len

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> int:
        """Append k_new/v_new shaped [n_tokens, n_kv_heads, head_dim]. Returns start pos."""
        n = k_new.shape[0]
        assert self.seq_len + n <= self.max_seq_len, \
            f"KV overflow: {self.seq_len}+{n} > {self.max_seq_len}"
        pos = self.seq_len
        self.K[layer_idx][pos:pos + n].copy_(k_new)
        self.V[layer_idx][pos:pos + n].copy_(v_new)
        return pos

    def commit(self, n_tokens: int):
        self.seq_len += n_tokens

    def get(self, layer_idx: int, length: int | None = None):
        """Contiguous read along seq dim: [L, H, D]."""
        L = length if length is not None else self.seq_len
        return self.K[layer_idx][:L], self.V[layer_idx][:L]

    def reset(self):
        self.seq_len = 0  # reuse buffers, no free/malloc (elastic static graph)

    def trim(self, new_len: int):
        assert 0 <= new_len <= self.seq_len
        self.seq_len = new_len

    def memory_bytes(self) -> int:
        per = self.max_seq_len * self.n_kv_heads * self.head_dim * torch.finfo(self.dtype).bits // 8
        return per * 2 * self.n_layers

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "seq_len": self.seq_len,
            "K": self.K, "V": self.V,
            "meta": (self.n_layers, self.n_kv_heads, self.head_dim, self.max_seq_len, str(self.dtype)),
        }, path)

    @classmethod
    def load(cls, path: str) -> "StaticKVCache":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        n_layers, n_kv_heads, head_dim, max_seq, dtype_s = ckpt["meta"]
        dtype = getattr(torch, dtype_s.split(".")[-1])
        c = cls(n_layers, n_kv_heads, head_dim, max_seq, dtype)
        c.K, c.V, c.seq_len = ckpt["K"], ckpt["V"], ckpt["seq_len"]
        return c
