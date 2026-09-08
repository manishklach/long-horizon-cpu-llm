# Getting started

## Environment

The published experiment used Python 3.11 on Windows. Create a virtual environment if you
want to keep these dependencies separate. Commands below run from the repository root.

Windows PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest tests/ -q
```

On Linux/macOS, use `.venv/bin/python` instead. The reported performance results are from
Windows; the other platforms have not been benchmarked by this project.

For the following examples, use your environment's Python executable in place of `python`.

## Local smoke test

```powershell
python -m scripts.local_chat --model tiny-opt-125m
```

This is a raw-text append-mode CLI. OPT-125M is useful for checking the installation but is
not instruction-tuned. Exit with `exit` or `quit`.

## Instruction chat API

In PowerShell:

```powershell
$env:CPU_LLM_MODEL = "qwen2.5-0.5b"
$env:CPU_LLM_MAX_SEQ = "8192"
python -m uvicorn src.server.app:app --host 127.0.0.1 --port 8000
```

In another terminal using the same environment:

```powershell
python -m scripts.chat
```

The client sends the full transcript each turn. HF instruction chat uses the tokenizer's
chat template. The research API has no authentication; these examples bind to localhost.
`python -m scripts.run_server` is an alternative launcher that binds all interfaces.

## GGUF

```powershell
python -m pip install --only-binary=:all: -r requirements-eval.txt
python -m src.bench.evaluate prepare
python scripts/run_gguf.py --model models/qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf --n-ctx 8192
```

`prepare` downloads both pinned HF and GGUF artifacts and verifies their checksums. Models
are kept in the ignored `models/` directory. Optional CPU wheels are platform-dependent;
see [the evaluation guide](experiments/README.md) for the pinned runtime and build caveats.

## Dashboard and benchmark

```powershell
python -m streamlit run dashboard/app.py
python -m src.bench.benchmark --model tiny-opt-125m --lengths 128,512,1024,2000 --max-new 8 --chunk 512 --warmups 1 --repeats 5 --threads 4 --output bench_report.json
```

The benchmark produces JSON and HTML. Requests above the model's prompt-plus-output limit
are skipped rather than relabelled as longer runs. This benchmark uses synthetic token IDs.
For quality measurements, use the [pinned evaluation workflows](experiments/README.md).
