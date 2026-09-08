# Architecture and research boundaries

## Serving path

`src/server/app.py` validates requests and selects a backend. HFBackend delegates to
`CPUEngine`; LlamaCppBackend owns one native llama.cpp instance. Backend mutation is
serialized. Named sessions keep history, while anonymous API sessions are discarded.

HF full prefill evaluates the prompt in one forward. Incremental generation reuses only
a verified cached token prefix. After generation, the final sampled token remains pending,
so KV covers `input_ids[:-1]`, including EOS handling. On the next turn that token and the
new suffix are evaluated. Context validation reserves the requested output before inference.

The GGUF serving wrapper keeps separate transcripts, but a shared native instance does not
guarantee separate resident KV per named session. The evaluator tests one resident conversation.

## Experimental components

- `src/engine/kv_cache.py`: preallocated buffers shaped `[sequence, KV heads, head dimension]`.
  This is a standalone cache experiment, not the storage used for HF logits.
- `src/engine/attention.py`: head-by-head and batched attention correctness experiments.
  They materialize attention scores. At 50K tokens, one FP32 square score matrix alone is 10 GB.
- `scripts/long_context_test.py`: legacy allocation/attention/model diagnostic. Its allocation
  phase and small attention slice do not establish 50K model quality or scalable attention.
- `src/memory/session_store.py`: JSONL transcript logging and supplied-document prompt injection.
  It does not implement retrieval, persistent KV restoration or confidence estimation.

A cache capacity estimate excludes model weights, activations, attention workspace and other
runtime memory. Fitting KV in RAM does not establish that full prefill is feasible or useful.

## Evidence path

`src/bench/benchmark.py` measures repeated HF full/chunked prefill and generation. The pinned
`runners.py` / `evaluate.py` path verifies artifact checksums and HF/GGUF token parity, then
records exact IDs, output, timing and sampled RSS. `tasks.py` builds seeded synthetic inputs.
`recall_controls.py` evaluates the predeclared paired-reference protocol with fresh caches.

`docs/experiments/results/` retains successful and failed cases. Performance and quality are
separate outcomes: cache parity preserved wrong answers in the longer conversation test.
The explicit-ID control improved tested follow-ups, while missing-record failures persisted.
