"""0.5B vs 1.5B comparison on recall + missing-record evals.

- Same seeded stores/questions for every model (held-out seeds by default).
- Greedy decoding (temperature=0) for determinism.
- Per-call streaming timing: TTFT + TPOT. Peak RSS sampled during eval.
- Models load one at a time (16GB box) and are freed between runs.
- Writes JSON results + prints a markdown table for the PR body.

Usage:
  python -m src.eval.compare_models --models models/qwen2.5-0.5b-instruct-q4_k_m.gguf models/qwen2.5-1.5b-instruct-q4_k_m.gguf
  python -m src.eval.compare_models --models <a.gguf> --seeds 101,202,303 --n-records 20
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import random
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.eval.records import (  # noqa: E402
    generate_store, build_messages, recall_item, missing_item,
    score_recall, score_abstain)

DEV_SEEDS = [11, 22]
HELDOUT_SEEDS = [101, 202, 303]


class RamSampler:
    """Peak-RSS sampler thread (MB)."""

    def __init__(self):
        import psutil
        self.proc = psutil.Process()
        self.peak = 0.0
        self._stop = threading.Event()

    def _loop(self):
        while not self._stop.wait(0.05):
            rss = self.proc.memory_info().rss / 1e6
            if rss > self.peak:
                self.peak = rss

    def __enter__(self):
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()


def timed_completion(llm, messages: list[dict], max_tokens: int) -> dict:
    """Streamed chat completion with true TTFT/TPOT. Greedy."""
    t0 = time.perf_counter()
    first, text, usage, pieces = None, "", {}, 0
    for chunk in llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens,
            temperature=0.0, stream=True):
        ch = chunk["choices"][0]
        delta = ch.get("delta", {})
        if delta.get("content"):
            if first is None:
                first = time.perf_counter()
            pieces += 1  # llama.cpp streams one token per chunk
        text += delta.get("content", "")
        if "usage" in chunk:
            usage = chunk["usage"]
    t1 = time.perf_counter()
    # streamed chunks rarely carry usage; fall back to per-piece count (~1 tok/chunk)
    n_gen = usage.get("completion_tokens", 0) or pieces
    ttft = (first - t0) if first else (t1 - t0)
    tpot = ((t1 - first) / max(n_gen, 1)) if first and n_gen else 0.0
    return {"text": text, "ttft_s": round(ttft, 4), "tpot_s": round(tpot, 4),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": n_gen}


def eval_model(path: str, seeds: list[int], n_records: int,
               n_recall: int, n_missing: int, n_ctx: int, max_tokens: int) -> dict:
    from llama_cpp import Llama
    import psutil
    proc = psutil.Process()
    base_mb = proc.memory_info().rss / 1e6
    llm = Llama(model_path=path, n_ctx=n_ctx, verbose=False)
    load_mb = proc.memory_info().rss / 1e6 - base_mb
    rec_ok = rec_n = mis_ok = mis_n = 0
    ttfts, tpots = [], []
    with RamSampler() as ram:
        for seed in seeds:
            store = generate_store(seed, n_records)
            rng = random.Random(1000 + seed)
            for _ in range(n_recall):
                q, exp = recall_item(store, rng)
                r = timed_completion(llm, build_messages(store, q), max_tokens)
                rec_n += 1
                rec_ok += score_recall(r["text"], exp)
                ttfts.append(r["ttft_s"])
                tpots.append(r["tpot_s"])
            for _ in range(n_missing):
                q, _name = missing_item(store, rng)
                r = timed_completion(llm, build_messages(store, q), max_tokens)
                mis_n += 1
                mis_ok += score_abstain(r["text"])
                ttfts.append(r["ttft_s"])
                tpots.append(r["tpot_s"])
    peak_mb = ram.peak - base_mb
    del llm
    gc.collect()
    n = max(rec_n + mis_n, 1)
    return {"model": os.path.basename(path),
            "seeds": seeds, "n_records": n_records,
            "recall_acc": round(rec_ok / max(rec_n, 1), 3),
            "recall_n": rec_n,
            "abstain_rate": round(mis_ok / max(mis_n, 1), 3),
            "missing_n": mis_n,
            "ttft_mean_s": round(sum(ttfts) / n, 4),
            "tpot_mean_s": round(sum(tpots) / n, 4),
            "load_ram_mb": round(load_mb, 1),
            "peak_ram_mb": round(peak_mb, 1)}


def md_table(rows: list[dict]) -> str:
    h = ("| model | recall acc | abstain rate | TTFT mean (s) | TPOT mean (s) | "
         "load RAM (MB) | peak RAM (MB) |")
    s = ["|---|---|---|---|---|---|---|", h]
    for r in rows:
        s.append(f"| {r['model']} | {r['recall_acc']} ({r['recall_n']}) | "
                 f"{r['abstain_rate']} ({r['missing_n']}) | {r['ttft_mean_s']} | "
                 f"{r['tpot_mean_s']} | {r['load_ram_mb']} | {r['peak_ram_mb']} |")
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", default=",".join(map(str, HELDOUT_SEEDS)))
    ap.add_argument("--n-records", type=int, default=20)
    ap.add_argument("--n-recall", type=int, default=6)
    ap.add_argument("--n-missing", type=int, default=4)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default="eval/model_compare.json")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    rows = [eval_model(m, seeds, a.n_records, a.n_recall,
                       a.n_missing, a.n_ctx, a.max_tokens) for m in a.models]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print(md_table(rows))
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
