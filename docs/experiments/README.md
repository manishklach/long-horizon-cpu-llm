# Pinned CPU evaluation

This harness measures exact token-level generation separately from the serving API.
It uses the same Qwen2.5-0.5B-Instruct tokenizer/chat template and verifies GGUF token IDs
match before each case. A mismatch stops the run.

## Setup

```powershell
pip install -r requirements.txt
pip install --only-binary=:all: -r requirements-eval.txt
python -m src.bench.evaluate prepare
```

The core package versions used for the checked-in run are in
[`experiments/requirements-tested.txt`](../../experiments/requirements-tested.txt).
The optional runtime is pinned to llama-cpp-python 0.3.35. Its CPU wheel comes from the
maintainer's package index. If a wheel is unavailable on your platform, build that exact
version from source and record the build options; do not treat different builds as identical.

The checked-in specification pins both model repository revisions, the official GGUF SHA256
and the 32,768-token context limit. `prepare` records hashes of downloaded HF artifacts too.
Every run verifies this manifest. Models remain under the ignored `models/` directory.

## Baselines

Run backends sequentially to avoid competing CPU workloads:

```powershell
python -m src.bench.evaluate baseline --backend gguf --context 4096 --lengths 512,2048 --max-new 16 --threads 4 --output data/experiments/baseline-gguf.json
python -m src.bench.evaluate baseline --backend hf --context 4096 --lengths 512,2048 --max-new 16 --threads 4 --output data/experiments/baseline-hf.json
```

Defaults: one warmup, three measured repeats, greedy decoding with repeat penalty 1.
Both runners stop on the shared tokenizer EOS token or the requested output limit;
the model card's stochastic generation defaults are deliberately not used. Prompt budgets retain complete
records, so actual lengths can be slightly below the budget. Both backends receive identical
IDs. These are HF FP32 versus GGUF Q4_K_M measurements: runtime and precision both differ.
No result isolates a kernel improvement. The GGUF native prefill batch size defaults to 512;
HF full prefill is a single forward. `--batch` affects only GGUF.

## Quality and session reuse

```powershell
python -m src.bench.evaluate quality --backend gguf --context 16384 --lengths 512,4096,8192 --seeds 17 --threads 4 --output data/experiments/quality-gguf.json
python -m src.bench.evaluate sessions --backend gguf --context 4096 --lengths 512 --threads 4 --output data/experiments/sessions-gguf.json
python -m src.bench.evaluate sessions --backend hf --context 4096 --lengths 512 --threads 4 --output data/experiments/sessions-hf.json
python -m src.bench.evaluate sessions --backend gguf --context 8192 --lengths 4096 --threads 4 --output data/experiments/sessions-long-gguf.json
```

Quality cases place a seeded access-code record near 10%, 50% and 90% of the record corpus,
plus one absent-record control, at each length. Actual needle token offsets are retained.
The exact-answer scorer ignores case and surrounding punctuation, but extra prose fails. The corpus
is synthetic; success does not establish general reasoning, RAG or agent-task quality.
Increase `--seeds 17,29,41` for replication. Report each seed/position; do not tune prompts
on the measured seed and then claim held-out performance.

Sessions use complete templated conversations, append assistant/user turns, and compare cached
and fresh execution of identical IDs. The runner only reuses a fully verified cached prefix.
It evaluates the pending token and suffix explicitly and records actual reused-token counts.
Each fresh control replaces resident state; the next turn extends that control's transcript.
This tests one resident conversation, not concurrent per-session caches or restart persistence.

Results checkpoint after every measured case. `complete: false` marks an interrupted run;
the summary tool rejects incomplete reports. Unsupported lengths are explicitly skipped.
The run fails on errors rather than scoring them as wrong answers.

## Summaries

```powershell
python -m src.bench.summarize data/experiments/baseline-gguf.json data/experiments/baseline-hf.json data/experiments/quality-gguf.json data/experiments/sessions-gguf.json data/experiments/sessions-hf.json data/experiments/sessions-long-gguf.json --output data/experiments/summary.md
```

TTFT is measured through the first sampled token, excluding tokenization and model loading.
TPOT averages subsequent token intervals and is null for one-token outputs. Whole-process RSS
is sampled every 5 ms and includes common Python/PyTorch imports in both runners. It may miss
brief peaks; allocator retention affects baselines. Model hashes, prompt IDs, outputs, settings,
versions, native CPU features and source hashes accompany the raw measurements.

These experiments do not validate 50K context. This model's configured limit is 32,768 tokens;
`--context` above that is rejected. Long-context runs can be slow on a laptop CPU. HF full
prefill also materializes large logits tensors; use conservative lengths until that path is profiled.

## Extending the evidence

The three-seed recall replication is complete; see the controls linked below. The next
comparison should add a stronger model and held-out seeds, as described in the
[roadmap](../ROADMAP.md), followed by additional prompt families and matched-precision controls.
Keep precision, context, prompt IDs, thread counts and output lengths explicit. Run backends
in alternating process order across replications to assess host-load and thermal effects;
the initial runs execute one backend after the other and cannot remove those effects.

The initial run is documented in [results and failure cases](results/README.md).

## Recall failure replication

See [the declared multi-seed recall controls](recall-controls.md) for paired implicit/explicit
record references and repeated 8K absent-record cases. The protocol is fixed before the run.
