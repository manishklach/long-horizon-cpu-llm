"""Pytest wrapper: fast 50K-shape check without downloading a model."""
import torch
from scripts.long_context_test import phase_a_static_50k, phase_b_attention_long


def test_static_holds_50k_shape():
    # small dims keep it fast; proves no-realloc accounting at 50K length
    r = phase_a_static_50k(n_layers=2, n_kv_heads=2, head_dim=16, seq=50000)
    assert r["ok"] and r["seq"] == 50000


def test_attention_matches_at_long_slice():
    r = phase_b_attention_long(seq=1024)
    assert r["match"], r["max_err"]
