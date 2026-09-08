# Long-horizon CPU LLM inference

A CPU-only research prototype for persistent KV reuse, inference measurements, and long-context
answer-quality evaluation. Includes Hugging Face and optional llama.cpp/GGUF backends, a basic
chat API, a Streamlit dashboard, and reproducible experiments with raw results.

**[v0.2.0](https://github.com/manishklach/long-horizon-cpu-llm/releases/tag/v0.2.0)** |
[Documentation](docs/README.md) | [Results](docs/experiments/results/README.md) |
[Roadmap](docs/ROADMAP.md) | [MIT license](LICENSE)

## What works today

- Full or incremental HF prefill, with verified token-prefix reuse and context bounds.
- Complete-history chat requests, isolated anonymous sessions, reset, and streamed text.
- CPU GGUF execution and a separate token-level evaluation runner.
- Repeated prefill/generation measurements, model checksums, prompt-token parity checks,
  hardware metadata, sampled RAM, and saved raw trials.
- Retrieval-position, missing-record, growing-conversation, and explicit-ID controls.

The HF inference path uses its native dynamic cache. Static KV and head-by-head attention
remain standalone experiments; neither currently accelerates model generation. This repo
does not establish that CPU inference beats GPUs, or that 50K-token generation is validated.

## Quickstart

Python 3.11 was used for the published Windows CPU run. From a Python environment:

```powershell
git clone https://github.com/manishklach/long-horizon-cpu-llm.git
cd long-horizon-cpu-llm
python -m pip install -r requirements.txt
python -m scripts.local_chat --model tiny-opt-125m
```

The tiny OPT model is a smoke-test model, not a strong assistant. For instruction chat, set
`CPU_LLM_MODEL=qwen2.5-0.5b` and run the API; see [setup and examples](docs/getting-started.md).

```powershell
python -m pytest tests/ -q
python -m uvicorn src.server.app:app --host 127.0.0.1 --port 8000
# In another terminal:
python -m scripts.chat
```

Models download on first use. Optional pinned CPU GGUF support:

```powershell
python -m pip install --only-binary=:all: -r requirements-eval.txt
python -m src.bench.evaluate prepare
python -m src.bench.recall_controls
```

The last command runs 12 real-model cases and can take many minutes on a laptop CPU.
See [all experiment commands](docs/experiments/README.md).

## Evidence, including failures

Published on one four-core/eight-thread Windows CPU, using four inference threads:

| Measurement | Observation |
|---|---|
| 2,030-token prompt, median TTFT | HF FP32: 15.823 s; GGUF Q4_K_M: 19.778 s |
| Same prompt, median decode interval | HF: 0.1915 s/token; GGUF: 0.0320 s/token |
| Same prompt, sampled process RSS peak | HF: 3.706 GiB; GGUF: 1.012 GiB |
| Cached versus fresh generation | Identical output IDs in nine tested turn comparisons |
| Three-seed follow-up control | Implicit reference: 0/3 correct; explicit record ID: 3/3 |
| Three-seed absent-record control near 8K | 0/3 correct |

Runtime and weight precision both differ in the backend comparison. The quality cases are
small synthetic diagnostics, not general reliability scores. Faster cache reuse preserved
outputs but did not fix wrong answers. [Read the methods, raw results, and limits](docs/experiments/results/README.md).

## API and research scope

When using `session_id`, send the **complete conversation** on every chat request. The ID
allows prefix reuse; it does not append omitted messages. Requests without an ID are independent.
See the [API contract and v0.1 migration notes](docs/api.md).

The model in the published evaluation supports 32,768 configured context tokens; the experiments
here do not validate 32K or 50K quality. The 50K test checks allocation with small dimensions.
Transcript logs are not persistent KV checkpoints, and RAG currently injects supplied documents
without implementing retrieval. The API is a local research server, not a production service.

## Contributing and next work

Run the 23-test download-free suite before changing inference or evaluation behavior. Keep model
revisions and protocols fixed before measuring, and publish failures alongside successes.
See [architecture](docs/architecture.md), [the next PR proposal](docs/ROADMAP.md), and
[release notes](docs/releases/v0.2.0.md).
