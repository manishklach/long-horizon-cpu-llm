# Long-horizon CPU LLM inference

A CPU inference research prototype with Hugging Face and optional llama.cpp/GGUF backends.
The working inference path uses Hugging Face's native KV cache. Static KV and head-by-head
attention are standalone experiments; neither currently accelerates model generation.

## Current milestone: correct sessions and credible measurements

- API chat uses the complete supplied message history and the tokenizer's chat template.
  Models without a chat template use a plain role-labelled fallback (not an instruction-tuning guarantee).
- Requests without `session_id` are independent and their cache is released afterward.
- With `session_id`, send the complete conversation on every API request. The ID enables cache
  reuse; it does not cause the server to append missing messages. Changed history safely recomputes.
- Local `CPUEngine.generate` defaults to raw-token append mode. Use `prompt_mode="full"` for
  authoritative prompts, or `generate_chat(messages, ...)` for templated conversations.
- The final generated token remains pending in the cache, including EOS. Reported reused tokens
  count only tokens actually cached. Context checks reserve the requested output before inference.
- Reset removes both cached history and the turn count. Engine mutations are serialized.
- Chat SSE receives text during decoding, with incomplete words buffered by the HF backend.
  It is not one SSE event per token. A disconnected client currently does not cancel generation.
- JSONL files are transcript logs, not restartable KV checkpoints. RAG accepts supplied documents;
  retrieval itself is outside this prototype.

## Quickstart

```powershell
pip install -r requirements.txt
python -m scripts.local_chat --model tiny-opt-125m
python -m scripts.run_server
python -m scripts.chat
pytest tests/ -q
streamlit run dashboard/app.py
```

Set `CPU_LLM_MODEL`, `CPU_LLM_MAX_SEQ`, and `CPU_LLM_INT8=1` to configure the API server.
The API supports basic text chat/completion requests, not the entire OpenAI/vLLM API surface.
Streaming errors are emitted as SSE error objects because response headers have already been sent.

Optional GGUF backend:

```powershell
pip install llama-cpp-python
python scripts/run_gguf.py --model path/to/model.gguf --n-ctx 8192
```

GGUF TTFT measures first content-chunk arrival. Exact token usage, TPOT and reused-token counts
are currently returned as `null` rather than inferred from words or total request time.
One llama.cpp instance is shared; separate transcript histories do not guarantee resident KV per session.

## Reproducible benchmark

```powershell
python -m src.bench.benchmark --model tiny-opt-125m --lengths 128,512,1024,2000 --max-new 8 --chunk 512 --warmups 1 --repeats 5 --threads 4 --output bench_report.json
```

The JSON and accompanying HTML contain:

- Actual and requested prompt lengths; unsupported prompt-plus-output lengths are explicitly skipped.
- Full/chunked prefill medians, standard deviations and raw trials, with alternating execution order.
- Separate generation TTFT (through first sampled token) and TPOT (between subsequent tokens).
  TPOT is `null` when generation emits only one token.
- CPU/platform, thread count, RAM and dependency versions.
- Whole-process RSS sampled every 5 ms, including model weights. Brief peaks can be missed;
  allocator retention and earlier runs affect the baseline. This is not isolated per-method peak memory.

Inputs are synthetic token IDs. These timings do not demonstrate long-context answer quality.
The previous OPT-125M table was a single-run exploratory result; it is not retained as evidence of
an architectural speedup. Full and chunked prefill use the same resident model; the harness does
not measure weight traffic and cannot attribute differences to parameter loading.

## Research limits and next experiments

`tests/test_long_context_50k.py` verifies 50K allocation with small dimensions. Its attention
comparison uses a 512-token slice. These are not 50K model-generation or quality results.
The legacy `scripts/long_context_test.py` is a diagnostic; use the benchmark above for repeated timing.

Next: compare against a pinned GGUF baseline, evaluate fact retrieval and growing conversations at
verified model context lengths, then profile before integrating one optimization for one architecture.
A 50K FP32 attention score matrix alone is 10 GB per head. The experimental attention implementation
materializes this matrix; tiled attention is needed before treating it as a scalable kernel.
KV sizing excludes weights, activations and attention workspace. Model context support must be
verified independently of available RAM.

## Validation

Tests include a small randomly initialized OPT model (no model downloads) for cached/uncached
output equivalence, changed-history recomputation, EOS alignment, reset, limits, streaming text,
and full/chunked prefill equivalence; API isolation/history and benchmark-label tests are also included.
These tests establish implementation behavior, not pretrained model quality or hardware performance.

MIT licensed. See [LICENSE](LICENSE).

## Pinned GGUF and quality evaluation

The next milestone adds token-matched HF FP32 / official GGUF Q4_K_M baselines,
seeded retrieval-position and absent-record cases, and cached/fresh growing-conversation
checks. Model revisions, checksums and the optional native runtime are pinned.

See [evaluation setup and commands](docs/experiments/README.md). The default quality
sweep covers prompt budgets of 512, 4096 and 8192 tokens; it does not claim 50K support.

[Measured results and failure cases](docs/experiments/results/README.md) include a real
8K retrieval sweep and growing conversations, with raw reports checked in.

[Multi-seed recall controls](docs/experiments/recall-controls.md) replicate the follow-up
failure and test explicit record IDs, with all outcomes and raw data published.
