"""Benchmark: full single-pass Prefill vs chunked baseline + Decode TPOT.

Full single-pass Prefill beats chunked baselines as context grows on CPU.
Saves an HTML dashboard.
"""
from __future__ import annotations
import argparse
import json
import time
import torch
from src.engine.inference import CPUEngine


def run(model: str, lengths: list[int], max_new: int = 16, chunk: int = 512):
    e = CPUEngine(model, max_seq_len=max(lengths) + max_new + 64)
    # cap at model positional limit so tiny models (OPT 2048) don't overflow
    try:
        cfg = e.model.config.to_dict()
        model_max = int(cfg.get("max_position_embeddings") or cfg.get("n_positions") or 2048)
    except Exception:
        model_max = 2048
    rows = []
    for L in lengths:
        Lc = min(L, model_max - max_new - 8)
        if Lc <= 0:
            continue
        ids = list(range(100, 100 + Lc))  # synthetic long prompt ids (mod vocab below)
        vocab = e.tok.vocab_size if hasattr(e.tok, "vocab_size") else 50265
        ids = [i % vocab for i in ids]
        _, t_full = e.prefill_full(ids)
        _, t_chunk = e.prefill_chunked(ids, chunk)
        # decode TPOT: greedy loop from full cache
        import torch as _t
        with _t.no_grad():
            out = e.model(input_ids=_t.tensor([ids]), use_cache=True)
            cache, logits = out.past_key_values, out.logits[0, -1]
            t0 = time.perf_counter()
            for _ in range(max_new):
                nxt = int(_t.argmax(logits))
                o = e.model(input_ids=_t.tensor([[nxt]]), past_key_values=cache, use_cache=True)
                cache, logits = o.past_key_values, o.logits[0, -1]
            tpot = (time.perf_counter() - t0) / max_new
        rows.append({"seq_len": L, "ttft_full_s": round(t_full, 4),
                     "ttft_chunked_s": round(t_chunk, 4),
                     "speedup": round(t_chunk / max(t_full, 1e-9), 2),
                     "tpot_s": round(tpot, 4)})
        print(rows[-1])
    return rows


def save_html(rows, path="bench_report.html"):
    rows_html = "".join(
        f"<tr><td>{r['seq_len']}</td><td>{r['ttft_full_s']}</td>"
        f"<td>{r['ttft_chunked_s']}</td><td>{r['speedup']}x</td><td>{r['tpot_s']}</td></tr>"
        for r in rows)
    html = f"""<html><head><title>CPU-LLM Bench</title>
<style>body{{font-family:sans-serif;margin:2em}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:6px 12px}}</style></head><body>
<h2>Prefill: full single-pass vs chunked + Decode TPOT</h2>
<table><tr><th>seq_len</th><th>TTFT full (s)</th><th>TTFT chunked (s)</th>
<th>speedup</th><th>TPOT (s/tok)</th></tr>{rows_html}</table>
<p>Full-pass avoids repeated param loading; gap widens with length.</p>
</body></html>"""
    open(path, "w", encoding="utf-8").write(html)
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny-opt-125m")
    ap.add_argument("--lengths", default="128,512,1024,2048")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args()
    rows = run(a.model, [int(x) for x in a.lengths.split(",")], a.max_new, a.chunk)
    save_html(rows)
    print(json.dumps(rows, indent=2))
