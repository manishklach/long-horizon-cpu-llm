# Findings from the initial Windows CPU run

See [the generated tables](summary.md) and the raw JSON files in this directory.
All runs used the pinned Qwen2.5-0.5B-Instruct artifacts, four CPU threads and greedy decoding.

- **The backend tradeoff depends on the phase.** At 2,030 prompt tokens, GGUF Q4_K_M had
  median TPOT of 0.0320 s versus HF FP32's 0.1915 s, and sampled RSS of 1.012 versus
  3.706 GiB. HF had lower median TTFT: 15.823 versus 19.778 s. Runtime and precision
  both differ, so this is not evidence of an isolated architectural improvement.
- **Positive retrieval passed at each tested position**, including three cases at 8,173 tokens.
  This is one seed per position, not a broad quality claim.
- **Missing-record detection failed at 8,192 tokens.** The requested record was absent, but
  the model returned `CODE-PTUUNWWF`, the code belonging to distractor `item-00040`, instead
  of `UNKNOWN`. The shorter absent-record controls passed.
- **Cache reuse preserved outputs in all nine tested turn comparisons.** The longer GGUF
  conversation grew from 4,084 to 4,848 prompt tokens. Its second and third turns reused
  4,091 and 4,470 tokens respectively.
- **Follow-up recall failed in that longer conversation.** Both cached and fresh execution
  returned `UNKNOWN` on turns two and three despite the code remaining in the transcript.
  The failure is therefore present without KV reuse too. The follow-up uses a reference to
  the previously requested record; a future explicit-record-ID control can separate
  reference resolution from other recall failures.

The longer conversation's cached TTFT was about 5-6 seconds on later turns versus 45-55
seconds for fresh execution, but those answers were wrong. Faster wrong answers should
not be treated as successful long-horizon performance.

## Validation

- 21 automated tests passed, including exact cache reuse, prompt/tokenizer parity guards,
  seeded task construction, strict scoring, checksum verification and incomplete-run rejection.
- The production GGUF backend also completed a real-model stream/reset smoke check. It
  returned `OK.` in two content chunks; concatenated streamed text matched the final output.
- Six experiment reports completed: two baselines, one quality sweep, two short conversation
  runs and one longer conversation run. Raw prompts, IDs, outputs and hashes are retained.

## What this evidence suggests next

Replicate the negative controls and follow-up failures with more predeclared seeds, including
an explicit-ID follow-up control. Then test a stronger small model using the same tasks.
For performance work, add a matched-precision control and profile prefill and decode separately.
These results do not validate 32K or 50K quality, concurrent session residency, or persistence.

## Raw reports

- [HF FP32 baseline](baseline-hf.json)
- [GGUF Q4_K_M baseline](baseline-gguf.json)
- [GGUF retrieval and absent-record sweep](quality-gguf.json)
- [HF short conversation](sessions-hf.json)
- [GGUF short conversation](sessions-gguf.json)
- [GGUF longer conversation](sessions-long-gguf.json)
