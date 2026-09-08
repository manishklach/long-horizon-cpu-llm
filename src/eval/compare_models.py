"""Cross-size recall-controls comparison (e.g. 0.5B vs 1.5B GGUF).

Drives the EXISTING recall-controls protocol
(experiments/recall-controls-v1.json: 4K initial budget, 8K absent budget,
predeclared seeds, strict exact-answer scoring, token-level TTFT/TPOT where
TPOT is None for single-token outputs) once per model spec, then summarizes
head-to-head. Raw per-case records, sha256 manifests and runtime environments
are saved by the protocol runner itself, so results are auditable by
construction. This module adds no new timing or scoring code.

Seeds 17/29/41 are held out for this comparison: no prompt tuning was done
after observing any model output; seed 17 additionally replicates prior work.

Usage:
  python -m src.eval.compare_models --prepare          # download+checksum artifacts
  python -m src.eval.compare_models                    # run both sizes + summarize
  python -m src.eval.compare_models --summarize-only   # re-summarize saved reports
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.bench import recall_controls  # noqa: E402
from src.bench.runners import prepare  # noqa: E402

DEFAULT_SPECS = ["experiments/qwen2.5-0.5b.json", "experiments/qwen2.5-1.5b.json"]
PROTOCOL = "experiments/recall-controls-v1.json"


def roots_for(specs: list[str], root: str) -> list[str]:
    return [os.path.join(root + "-" + json.load(open(s))["slug"]) for s in specs]


def summarize_size(report: dict) -> dict:
    recs = [r for r in report["records"] if r.get("status") == "ok"]
    ttfts = [r["ttft_s"] for r in recs]
    tpots = [r["tpot_s"] for r in recs if r.get("tpot_s") is not None]
    initial = [r for r in recs if r["kind"] == "retrieval"]
    absent = [r for r in recs if r["kind"] == "absent"]
    elig = {}
    for v in ("implicit", "explicit"):
        rows = [r for r in recs if r.get("variant") == v and r.get("initial_exact_match")]
        elig[v] = (sum(r["exact_match"] for r in rows), len(rows))
    return {
        "cases": len(recs),
        "initial": [sum(r["exact_match"] for r in initial), len(initial)],
        "implicit_cond": list(elig["implicit"]),
        "explicit_cond": list(elig["explicit"]),
        "absent": [sum(r["exact_match"] for r in absent), len(absent)],
        "ttft_median_s": round(statistics.median(ttfts), 4) if ttfts else None,
        "tpot_median_s": round(statistics.median(tpots), 4) if tpots else None,
        "tpot_n": len(tpots),
        "peak_rss_gib": round(max(r["rss_peak_sampled_bytes"] for r in recs) / 2**30, 3) if recs else None,
    }


def md_table(rows: list[tuple[str, dict]]) -> str:
    h = ("| model | initial | implicit (cond) | explicit (cond) | absent | "
         "TTFT med (s) | TPOT med (s) | peak RSS (GiB) |")
    out = [h, "|---|---|---|---|---|---|---|---|"]
    for slug, s in rows:
        out.append(f"| {slug} | {s['initial'][0]}/{s['initial'][1]} | "
                   f"{s['implicit_cond'][0]}/{s['implicit_cond'][1]} | "
                   f"{s['explicit_cond'][0]}/{s['explicit_cond'][1]} | "
                   f"{s['absent'][0]}/{s['absent'][1]} | {s['ttft_median_s']} | "
                   f"{s['tpot_median_s']} (n={s['tpot_n']}) | {s['peak_rss_gib']} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specs", nargs="+", default=DEFAULT_SPECS)
    ap.add_argument("--protocol", default=PROTOCOL)
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--out-dir", default="eval")
    ap.add_argument("--tag", default="",
                    help="suffix for report names, e.g. v2 -> recall-controls-<slug>-v2.json")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    protocol = json.load(open(a.protocol, encoding="utf-8-sig"))
    os.makedirs(a.out_dir, exist_ok=True)
    summaries = []
    for spec_path in a.specs:
        spec = json.load(open(spec_path, encoding="utf-8-sig"))
        slug = spec["slug"]
        root = os.path.join(a.models_root + "-" + slug)
        tag = f"-{a.tag}" if a.tag else ""
        report_path = os.path.join(a.out_dir, f"recall-controls-{slug}{tag}.json")
        if a.prepare:
            prepare(spec, root)
            print(f"prepared {slug} -> {root}", flush=True)
            continue
        if a.summarize_only:
            report = json.load(open(report_path, encoding="utf-8"))
        else:
            report = recall_controls.run(protocol, spec, root, report_path)
        md = recall_controls.summarize(report)
        open(report_path.replace(".json", ".md"), "w", encoding="utf-8").write(md)
        summaries.append((slug, summarize_size(report)))
    if a.prepare:
        return
    table = md_table(summaries)
    print(table)
    tag = f"-{a.tag}" if a.tag else ""
    cmp_path = os.path.join(a.out_dir, f"size_compare{tag}.json")
    json.dump({"protocol": protocol["name"], "seeds": protocol["seeds"],
               "models": {slug: s for slug, s in summaries}}, open(cmp_path, "w"), indent=2)
    open(cmp_path.replace(".json", ".md"), "w", encoding="utf-8").write(
        "# Cross-size recall controls\n\n" + table + "\n")
    print(f"\nsaved {cmp_path}")


if __name__ == "__main__":
    main()
