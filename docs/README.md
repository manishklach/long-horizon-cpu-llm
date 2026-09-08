# Documentation

Start with [getting started](getting-started.md), then choose the workflow you need.

| Guide | Covers |
|---|---|
| [API contract](api.md) | Sessions, history, streaming, metrics and migration |
| [Architecture](architecture.md) | Working inference paths and experimental components |
| [Evaluation guide](experiments/README.md) | Pinned artifacts and reproducible commands |
| [Measured results](experiments/results/README.md) | Backend comparisons and observed failures |
| [Recall controls](experiments/recall-controls.md) | Predeclared multi-seed explicit-ID experiment |
| [Roadmap](ROADMAP.md) | Next PR scope and acceptance criteria |
| [v0.2.0 release](releases/v0.2.0.md) | Changes, validation and compatibility |

Raw JSON reports under `experiments/results/` include synthetic prompts, token IDs, outputs,
model/source hashes and measurements. Model weights and local transcript logs are ignored by Git.

The original `benchmark.png` is a legacy exploratory chart, not evidence for the current
benchmark harness. Use the linked raw reports and generated tables for current findings.
