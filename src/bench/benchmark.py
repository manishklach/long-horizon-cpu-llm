"""Repeated CPU prefill benchmark; timings are prefill-only, not API TTFT."""
from __future__ import annotations
import argparse
import html
import importlib.metadata
import json
import platform
import statistics
import threading
import time
import psutil
import torch
from src.engine.inference import CPUEngine


def metadata():
    return {"platform": platform.platform(), "cpu": platform.processor(),
            "logical_cpus": psutil.cpu_count(), "threads": torch.get_num_threads(),
            "ram_bytes": psutil.virtual_memory().total,
            "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "psutil")}}


def measured(fn):
    """Sample whole-process RSS (includes model); not allocator-only memory."""
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    stop = threading.Event()
    def sample():
        while not stop.wait(0.005):
            samples.append(process.memory_info().rss)
    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        value = fn()
        samples.append(process.memory_info().rss)
        return value, {"rss_baseline_bytes": baseline, "rss_peak_sampled_bytes": max(samples)}
    finally:
        stop.set()
        worker.join()


def run(model, lengths, max_new=16, chunk=512, repeats=5, warmups=1, threads=None):
    if not lengths or min(lengths) <= 0 or min(max_new, chunk, repeats) <= 0 or warmups < 0:
        raise ValueError("lengths, max_new, chunk, repeats must be positive; warmups nonnegative")
    e = CPUEngine(model, max_seq_len=max(lengths) + max_new, threads=threads)
    env = metadata()
    rows = []
    for requested in lengths:
        if requested + max_new > e.max_seq_len:
            rows.append({"requested_seq_len": requested, "seq_len": None, "status": "skipped",
                         "reason": f"prompt plus output exceeds context limit {e.max_seq_len}", "environment": env})
            continue
        ids = [(100 + i) % e.tok.vocab_size for i in range(requested)]
        samples = []
        for trial in range(warmups + repeats):
            order = ("full", "chunked") if trial % 2 == 0 else ("chunked", "full")
            for mode in order:
                fn = (lambda: e.prefill_full(ids)) if mode == "full" else (lambda: e.prefill_chunked(ids, chunk))
                (cache, seconds), mem = measured(fn)
                del cache
                if trial >= warmups:
                    samples.append({"trial": trial - warmups, "mode": mode, "prefill_s": seconds, **mem})
        decode = []
        for trial in range(repeats):
            e.reset_session("benchmark")
            result, mem = measured(lambda: e.generate(ids, session_id="benchmark", max_new_tokens=max_new,
                                                      prompt_mode="full"))
            decode.append({**result, **mem})
        e.reset_session("benchmark")
        full = [s["prefill_s"] for s in samples if s["mode"] == "full"]
        chunked = [s["prefill_s"] for s in samples if s["mode"] == "chunked"]
        tpots = [s["tpot_s"] for s in decode if s["tpot_s"] is not None]
        rows.append({"requested_seq_len": requested, "seq_len": len(ids), "status": "ok",
                     "prefill_full_s": statistics.median(full), "prefill_chunked_s": statistics.median(chunked),
                     "prefill_full_stdev_s": statistics.stdev(full) if repeats > 1 else 0,
                     "prefill_chunked_stdev_s": statistics.stdev(chunked) if repeats > 1 else 0,
                     "speedup": statistics.median(chunked) / statistics.median(full),
                     "ttft_s": statistics.median(s["ttft_s"] for s in decode),
                     "tpot_s": statistics.median(tpots) if tpots else None,
                     "samples": samples, "decode_samples": decode, "environment": env,
                     "model": model, "input_kind": "synthetic token IDs", "chunk": chunk,
                     "warmups": warmups, "repeats": repeats, "max_new": max_new,
                     "memory_measurement": "whole-process RSS sampled every 5ms; may miss brief peaks"})
    return rows


def save_html(rows, path="bench_report.html"):
    with open(path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>CPU benchmark</title></head><body>"
                "<h1>CPU prefill and generation measurements</h1>"
                "<p>Prefill timing excludes sampling. Speedup is chunked/full median. "
                "Unsupported lengths are skipped. No quality conclusion follows from synthetic prompts.</p><pre>"
                + html.escape(json.dumps(rows, indent=2)) + "</pre></body></html>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny-opt-125m")
    ap.add_argument("--lengths", default="128,512,1024,2000")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--output", default="bench_report.json")
    a = ap.parse_args()
    rows = run(a.model, [int(x) for x in a.lengths.split(",")], a.max_new, a.chunk,
               a.repeats, a.warmups, a.threads)
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    save_html(rows, a.output + ".html")
    print(json.dumps(rows, indent=2))
