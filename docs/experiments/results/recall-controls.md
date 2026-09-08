# Multi-seed recall controls

Every generation starts with a fresh KV cache. Paired follow-ups share the exact initial transcript and actual assistant response; only the record reference changes.

| Seed | Initial retrieval | Implicit follow-up | Explicit ID follow-up | Absent-record control |
|---:|---|---|---|---|
| 17 | PASS: `CODE-PKMKFYYT` | FAIL: `UNKNOWN` | PASS: `CODE-PKMKFYYT` | FAIL: `CODE-PTUUNWWF` |
| 29 | PASS: `CODE-CMVVKCSM` | FAIL: `UNKNOWN` | PASS: `CODE-CMVVKCSM` | FAIL: `CODE-WBDRSHRU` |
| 41 | PASS: `CODE-LHFNUYKT` | FAIL: `UNKNOWN` | PASS: `CODE-LHFNUYKT` | FAIL: `CODE-ULCSFMLU` |

## Counts

- implicit follow-up, conditional on correct initial retrieval: 0/3.
- explicit follow-up, conditional on correct initial retrieval: 3/3.
- Absent-record controls: 0/3.

This is a small synthetic diagnostic on one model, quantization and host. Seed 17 is a replication; 29 and 41 are new. These results do not establish general reliability or a causal explanation beyond the tested prompt contrast. Raw prompts, token IDs, outputs, seeds, hashes and timing are preserved in the accompanying JSON.
