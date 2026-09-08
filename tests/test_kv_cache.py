import torch
from src.engine.kv_cache import StaticKVCache
from src.engine.attention import attention_head_by_head, attention_batched


def test_static_cache_append_read():
    c = StaticKVCache(n_layers=2, n_kv_heads=2, head_dim=8, max_seq_len=16)
    k = torch.randn(4, 2, 8)
    v = torch.randn(4, 2, 8)
    c.append(0, k, v)
    c.commit(4)
    K, V = c.get(0)
    assert K.shape == (4, 2, 8)
    assert len(c) == 4
    c.reset()
    assert len(c) == 0


def test_head_by_head_matches_batched():
    torch.manual_seed(0)
    q = torch.randn(8, 4, 16)
    k = torch.randn(8, 2, 16)
    v = torch.randn(8, 2, 16)
    a = attention_head_by_head(q, k, v)
    b = attention_batched(q, k, v)
    assert torch.allclose(a, b, atol=1e-4), float((a - b).abs().max())
