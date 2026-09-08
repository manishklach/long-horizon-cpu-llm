# Roadmap

## Proposed next PR: model comparison for recall and abstention

The 0.5B Qwen baseline answered initial retrieval correctly, but implicit follow-ups and
8K missing-record controls failed across three declared seeds. Explicit record IDs recovered
all three follow-up answers. The next question is whether a larger small model improves
those outcomes at an acceptable CPU latency and RAM cost.

Proposed scope:

1. Make evaluation artifact paths and architecture metadata specification-driven instead
   of hardcoded to the current 0.5B preset. Keep strict hashes and tokenizer parity checks.
2. Add a pinned [official Qwen2.5-1.5B-Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
   candidate at Q4_K_M. Verify model revision, file checksum and supported context before running.
3. Compare it with the existing 0.5B baseline on the unchanged recall protocol. Declare
   additional held-out seeds before execution; retain the original seeds as regressions.
4. Add positive retrieval controls at the same 8K lengths as absent-record controls, so
   abstention improvements cannot be explained by always replying UNKNOWN.
5. Publish raw cases, paired quality outcomes, first-token latency, decode timing and RAM.
   Keep the serving default unchanged unless the measured tradeoff supports changing it.

Acceptance criteria:

- Multiple pinned model specifications work without disabling checksum/context validation.
- Each model's prompt construction is verified against its own tokenizer; cross-model
  tokenization differences are recorded instead of assuming identical token IDs.
- Every planned case is reported, including failures and incomplete runs; no hidden prompt tuning.
- The report distinguishes correct positive retrieval, correct abstention, and follow-up recall.
- Existing tests and cached/fresh equivalence checks pass. Performance claims state precision,
  hardware, context and output-length differences.

This PR is proposed, not implemented in v0.2.0. A larger model is a candidate, not a promised fix.

## Later work

- Independently designed document-level retrieval and abstention tasks beyond synthetic records.
- Matched-precision controls and profiling of prefill/logits allocation before kernel changes.
- Integration of one optimized cache or tiled-attention path, with isolated comparisons.
- Streaming cancellation, bounded session retention and persistence if moving toward a service.

Neither this roadmap nor the current results claim validated 50K model quality.
