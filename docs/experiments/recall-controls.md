# Multi-seed recall controls

The protocol in [recall-controls-v1.json](../../experiments/recall-controls-v1.json) was
committed before the measured run. It uses seeds 17, 29 and 41 with the same pinned Qwen
GGUF model and runtime as the initial evaluation. Seed 17 repeats the earlier setup;
the other seeds are new.

```powershell
python -m src.bench.recall_controls --output data/experiments/recall-controls.json
python -m src.bench.recall_controls --output data/experiments/recall-controls.json --summarize-only
```

Each seed runs four cases:

1. Initial retrieval with a 4,096-token prompt budget and an early target record.
2. A follow-up referring to "the same requested record".
3. The same follow-up containing the explicit record ID.
4. An absent-record control with an 8,192-token prompt budget.

The two follow-ups share the exact initial messages, the model's actual initial answer,
and the same irrelevant notes. The reference phrase is the only text change. Their order
alternates by seed. All four cases start with a fresh KV cache, so this experiment evaluates
answer quality independently of prefix reuse.

The scorer and generation settings are unchanged. All outcomes are reported, and paired
follow-up counts are also conditioned on a correct initial answer. This avoids treating
an already-corrupted transcript as a clean follow-up comparison. The absent-record cases
are scored separately from positive retrieval and follow-up cases.

The JSON checkpoints after each case and remains marked incomplete until all 12 finish.
It includes the declared protocol, raw prompts and token IDs, actual answers, model/source
hashes and generation metrics. The CLI generates a Markdown summary only for completed runs.

This is a small synthetic replication on one model and host. Changing a reference phrase
can identify sensitivity to that prompt contrast; it cannot establish a general explanation
for long-context failures. The 4K positive controls are not paired 8K positive controls.

## Multi-seed follow-up findings

The [three-seed control run](results/recall-controls.md), fixed in commit `0b2cdc4` before
execution, completed all 12 cases. Initial retrieval was correct for all three seeds.
The implicit follow-up was correct 0/3 times, while the explicit record-ID version was
correct 3/3. The paired transcripts shared the same actual initial answer and differed
only in the record-reference phrase. All cases used fresh KV state.

The 8K absent-record controls failed for all three seeds. This replicates the earlier
failure on seed 17 and extends it to seeds 29 and 41. Explicit identifiers helped resolve
follow-up references in this small diagnostic; they did not establish reliable abstention.

The evidence supports preserving explicit entity IDs when constructing record-oriented
follow-ups. It does not justify a generic rewrite of arbitrary user messages, nor prove
that all long-context recall problems are reference-resolution problems. Before choosing
a serving-model change, compare a stronger model on this unchanged protocol and add
independently designed document-level retrieval/abstention tasks.

Raw inputs and outputs: [recall-controls.json](results/recall-controls.json).
