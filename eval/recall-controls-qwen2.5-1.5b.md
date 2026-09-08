# Multi-seed recall controls

Every generation starts with a fresh KV cache. Paired follow-ups share the exact initial transcript and actual assistant response; only the record reference changes.

| Seed | Initial retrieval | Implicit follow-up | Explicit ID follow-up | Absent-record control |
|---:|---|---|---|---|
| 17 | PASS: `CODE-PKMKFYYT` | PASS: `CODE-PKMKFYYT` | PASS: `CODE-PKMKFYYT` | PASS: `UNKNOWN` |
| 29 | PASS: `CODE-CMVVKCSM` | PASS: `CODE-CMVVKCSM` | PASS: `CODE-CMVVKCSM` | PASS: `UNKNOWN` |
| 41 | PASS: `CODE-LHFNUYKT` | PASS: `CODE-LHFNUYKT` | PASS: `CODE-LHFNUYKT` | PASS: `UNKNOWN` |

## Counts

- implicit follow-up, conditional on correct initial retrieval: 3/3.
- explicit follow-up, conditional on correct initial retrieval: 3/3.
- Absent-record controls: 3/3.

This is a small synthetic diagnostic on one model, quantization and host. Seed 17 is a replication; 29 and 41 are new. These results do not establish general reliability or a causal explanation beyond the tested prompt contrast. Raw prompts, token IDs, outputs, seeds, hashes and timing are preserved in the accompanying JSON.
