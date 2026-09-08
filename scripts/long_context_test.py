"""Real 50K long-context test: full single-pass vs chunked.

Runs in 3 phases so it works on small CPU boxes AND big Linux servers:

Phase A (no model, always runs): StaticKVCache @ 50K accounting + memory math.
Phase B (no model): head-by-head attention timing at long seq (4K) vs batched.
Phase C (model): real HF prefill full vs chunked + 3-turn incremental session,
  capped at the model's own max position (e.g. OPT-125M -> 2048) so tiny CPUs
  don't OOM. Pass --seq-len 50000 with a long-context model (Qwen3) on a big
  box to do the true 50K run.

Usage:
  python scripts/long_context_test.py --model tiny-opt-125m --seq-len 2048
  python scripts/long_context_test.py --model qwen3-0.6b --seq-len 50000 --max-new 16
  pytest tests/test_long_context_50k.py -v
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.kv_cache import StaticKVCache
from src.engine.attention import attention_head_by_head, attention_batched
from src.engine.quant import kv_bytes_estimate


def phase_a_static_50k(n_layers=12, n_kv_heads=4, head_dim=64, seq=50000):
    """Prove the static graph holds 50K without realloc (accounting level)."""
    c = StaticKVCache(n_layers, n_kv_heads, head_dim, max_seq_len=seq)
    chunk = 5000
    t0 = time.perf_counter()
    for start in range(0, seq, chunk):
        n = min(chunk, seq - start)
        k = torch.randn(n, n_kv_heads, head_dim)
        v = torch.randn(n, n_kv_heads, head_dim)
        c.append(0, k, v)  # layer 0 representative; others same shape
        c.commit(n)
    dt = time.perf_counter() - t0
    mem_gb = c.memory_bytes() / 1e9
    assert len(c) == seq, f"expected {seq}, got {len(c)}"
    return {"phase": "A-static-50k", "seq": seq, "fill_s": round(dt, 3),
            "static_mem_gb": round(mem_gb, 3), "ok": True}


def phase_b_attention_long(seq=4096, n_heads=8, n_kv=2, d=64):
    torch.manual_seed(0)
    q = torch.randn(seq, n_heads, d)
    k = torch.randn(seq, n_kv, d)
    v = torch.randn(seq, n_kv, d)
    # correctness on a slice (full 4K softmax is heavy but OK on CPU once)
    t0 = time.perf_counter()
    a = attention_head_by_head(q[:512], k[:512], v[:512])
    t_hbh = time.perf_counter() - t0
    t0 = time.perf_counter()
    b = attention_batched(q[:512], k[:512], v[:512])
    t_bat = time.perf_counter() - t0
    err = float((a - b).abs().max())
    return {"phase": "B-attention", "seq_slice": 512, "full_seq": seq,
            "head_by_head_s": round(t_hbh, 3), "batched_s": round(t_bat, 3),
            "max_err": err, "match": bool(err < 1e-4)}


def phase_c_model(model_id: str, seq: int, max_new=16, chunk=512):
    from src.engine.inference import CPUEngine
    e = CPUEngine(model_id, max_seq_len=seq + max_new + 64)
    # cap at model's real position limit so tiny models don't OOM/fail
    model_max = getattr(getattr(e.model.config, "to_dict", lambda: {})(), "get", lambda *a, **k: None)("max_position_embeddings", None) \
        if hasattr(e.model.config, "to_dict") else getattr(e.model.config, "max_position_embeddings", None)
    if model_max is None:
        model_max = getattr(e.model.config, "n_positions", 2048)
    run_seq = min(seq, int(model_max))
    capped = run_seq < seq
    vocab = getattr(e.tok, "vocab_size", 50265) or 50265
    ids = [(100 + i) % vocab for i in range(run_seq)]
    _, t_full = e.prefill_full(ids)
    _, t_chunk = e.prefill_chunked(ids, chunk)
    # 3-turn incremental session (the long-horizon win: no recompute)
    sid = "lc-test"
    t0 = time.perf_counter()
    r1 = e.generate("Context: DDR memory holds the full prompt. " * 8, session_id=sid, max_new_tokens=8)
    r2 = e.generate("Summarize turn 1 in one line.", session_id=sid, max_new_tokens=8)
    r3 = e.generate("Add one more fact about KV caching.", session_id=sid, max_new_tokens=8)
    t_sess = time.perf_counter() - t0
    est_gb = kv_bytes_estimate(e.info["n_layers"], e.info["n_kv_heads"],
                               e.info["head_dim"], seq) / 1e9
    return {"phase": "C-model", "model": model_id, "requested_seq": seq,
            "ran_seq": run_seq, "capped_to_model_max": capped,
            "model_max_pos": int(model_max), "ttft_full_s": round(t_full, 4),
            "ttft_chunked_s": round(t_chunk, 4),
            "speedup_full_vs_chunked": round(t_chunk / max(t_full, 1e-9), 2),
            "turns": [r1["ttft_s"], r2["ttft_s"], r3["ttft_s"]],
            "incremental_reuse_t2": r2["reused_tokens"],
            "session_3turn_s": round(t_sess, 3),
            "kv_50k_est_gb_fp16": round(est_gb, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny-opt-125m")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--skip-model", action="store_true")
    a = ap.parse_args()
    out = {"model": a.model, "requested_seq": a.seq_len}
    out["A"] = phase_a_static_50k(seq=50000 if a.seq_len >= 50000 else a.seq_len)
    out["B"] = phase_b_attention_long()
    if not a.skip_model:
        out["C"] = phase_c_model(a.model, a.seq_len, a.max_new, a.chunk)
    print(json.dumps(out, indent=2))
    with open("long_context_report.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved long_context_report.json")


if __name__ == "__main__":
    main()
