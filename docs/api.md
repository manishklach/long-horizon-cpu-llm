# API contract

The server implements a basic text subset of OpenAI-style requests, plus research-specific
fields. It does not implement tools/function calling, multimodal content or the full API.

## Endpoints

| Endpoint | Behavior |
|---|---|
| `POST /v1/chat/completions` | Full-history chat; optional SSE |
| `POST /v1/completions` | Authoritative prompt; non-streaming |
| `POST /v1/rag/chat` | Inject supplied documents into the final user message |
| `POST /v1/sessions/{sid}/reset` | Clear in-memory session state and turn count |
| `GET /health` | Process status and configured model name; not a model-load check |
| `GET /v1/models` | Configured model ID |
| `GET /metrics` | Backend and in-memory session statistics |

## Chat example

```python
import requests

messages = [{"role": "user", "content": "Explain KV caching briefly."}]
response = requests.post("http://127.0.0.1:8000/v1/chat/completions", json={
    "messages": messages,
    "session_id": "example-session",
    "max_tokens": 64,
    "temperature": 0,
}, timeout=300)
response.raise_for_status()
answer = response.json()["choices"][0]["message"]
messages.extend([answer, {"role": "user", "content": "What changes on the next turn?"}])
# Send the complete messages list on the next request, with the same session_id.
```

Roles are `system`, `user` and `assistant`; content is text. `max_tokens` must be positive
and temperature nonnegative. Session IDs accept letters, digits, underscores and hyphens.

The complete supplied history is authoritative. With an explicit session ID, verified
cached prefixes can be reused; changed history recomputes safely. Without an ID, each
request gets an independent cache that is released afterward. IDs are cache identifiers,
not authentication credentials. Explicit-session turns are logged as JSONL; reset does
not delete those transcript files, and a process restart does not restore KV from them.

The lower-level HF `generate` method defaults to raw append mode for local callers.
`prompt_mode="full"` and `generate_chat` use authoritative history. The API's completion
endpoint uses full mode; the GGUF serving wrapper formats that prompt as a user chat message.

## Streaming and errors

Set `stream=true` on chat or RAG chat for SSE. Content chunks arrive during generation,
followed by a finish-reason chunk and `[DONE]`. HF buffers incomplete words, so a chunk is
not necessarily one token. Disconnecting a client currently does not cancel generation.

Validation errors use HTTP 422; non-streaming context errors use HTTP 400. Generation
errors after streaming starts are sent as SSE error objects because headers were already sent.
The finish reason is `stop` for EOS or `length` for the output limit on the HF path.

## Metrics

HF reports full prompt-token count, actual reused tokens, generated tokens, TTFT through
the first sampled token, and TPOT between subsequent sampled tokens. TPOT is null when
only one token is generated. These timings exclude network/client latency.

The GGUF serving wrapper reports first content-chunk latency; exact usage, reused tokens
and TPOT are null. Its separate [evaluation runner](experiments/README.md) does measure
exact token-level counts and timing. Do not compare those two metric definitions directly.

## Migrating from v0.1

Send full histories instead of only the latest message when using a session ID. Anonymous
requests no longer share a default conversation. Handle nullable metrics. Streaming now
reflects decoding rather than replaying the final response split into words. Benchmark
prefill fields are `prefill_full_s` and `prefill_chunked_s`, not TTFT-labelled prefill values.
