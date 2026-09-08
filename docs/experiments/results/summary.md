# CPU baseline and context evaluation

These are measurements on one CPU host with a small synthetic task set. HF uses FP32 and GGUF uses Q4_K_M; both runtime and precision differ. This does not isolate a kernel or architecture speedup.

## Cold-prompt baseline

Model loading/tokenization are excluded. Each measured request resets its cache. Warmup runs are excluded. TPOT covers intervals after the first sampled token. RSS is sampled whole-process memory and may miss brief peaks.

| Backend | Prompt tokens | Trials | Output token range | Median TTFT (s) | TTFT stdev (s) | Median TPOT (s) | Peak sampled RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gguf | 511 | 3 | 8-8 | 4.017 | 0.270 | 0.0249 | 0.952 |
| gguf | 2030 | 3 | 8-8 | 19.778 | 0.367 | 0.0320 | 1.012 |
| hf | 511 | 3 | 8-8 | 4.534 | 1.236 | 0.1725 | 2.617 |
| hf | 2030 | 3 | 8-8 | 15.823 | 1.108 | 0.1915 | 3.706 |

Cross-backend prompt hashes and context/thread/output settings match: **True**.
Observed greedy output token sequences match across backends: **True**.

## Retrieval quality

A strict exact-answer score allows surrounding whitespace/quotes/punctuation, but rejects extra prose. The answer occurs once in a seeded distractor corpus. Absent-record cases test UNKNOWN responses. A budget is an upper bound; whole records are retained and actual lengths are reported.

| Backend | Budget | Actual prompt range | Retrieval exact | Absent exact | Retrieval by requested position |
|---|---:|---:|---:|---:|---|
| gguf | 512 | 510-511 | 3/3 | 1/1 | 10%: 1/1, 50%: 1/1, 90%: 1/1 |
| gguf | 4096 | 4082-4084 | 3/3 | 1/1 | 10%: 1/1, 50%: 1/1, 90%: 1/1 |
| gguf | 8192 | 8173-8192 | 3/3 | 0/1 | 10%: 1/1, 50%: 1/1, 90%: 1/1 |

Observed exact-answer failures:

- gguf, 8192 tokens, absent, seed 17: expected `UNKNOWN`, received `CODE-PTUUNWWF`.

## Growing conversations

Each turn compares cached and freshly reset execution of identical prompt IDs. The next turn extends the fresh control conversation. These are single observations, not repeated speedup estimates.

| Backend | Turn | Prompt tokens | Reused tokens | Cached TTFT (s) | Fresh TTFT (s) | Same output IDs | Exact answer |
|---|---:|---:|---:|---:|---:|---|---|
| gguf | 1 | 511 | 0 | 2.961 | 2.962 | True | True |
| gguf | 2 | 896 | 518 | 4.126 | 7.639 | True | True |
| gguf | 3 | 1281 | 903 | 3.641 | 11.522 | True | True |
| hf | 1 | 511 | 0 | 3.795 | 3.527 | True | True |
| hf | 2 | 896 | 518 | 2.973 | 7.588 | True | True |
| hf | 3 | 1281 | 903 | 3.261 | 10.577 | True | True |
| gguf | 1 | 4084 | 0 | 45.539 | 45.711 | True | True |
| gguf | 2 | 4469 | 4091 | 4.987 | 44.726 | True | False |
| gguf | 3 | 4848 | 4470 | 6.091 | 54.662 | True | False |

## Reproducibility and limits

Host: AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD; 4 physical / 8 logical CPUs; 15.34 GiB RAM.

Model revisions, artifact hashes, prompts/token IDs, seeds, generation outputs, native CPU features, dependency versions, settings and per-case timing are preserved in the raw JSON. The harness is identified by source hashes even when run before its commit.

Supported model context is 32,768 tokens. The tested lengths establish neither 32K nor 50K quality. One seed per position is a diagnostic sample, not a statistically reliable quality estimate. Whole-process RSS includes shared Python/PyTorch imports in both runners and is not isolated KV memory.
