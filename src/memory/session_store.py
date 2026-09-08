"""Long-horizon session memory + RAG injection.

- Persistent session transcripts (JSONL per session).
- RAG: prepend retrieved docs as context block, still benefits from
  incremental prefill (only new turn forwarded).
"""
from __future__ import annotations
import json
import os
import time

STORE_DIR = os.environ.get("CPU_LLM_STORE", "data/sessions")


def _path(sid: str) -> str:
    os.makedirs(STORE_DIR, exist_ok=True)
    safe = "".join(c for c in sid if c.isalnum() or c in "-_")
    return os.path.join(STORE_DIR, f"{safe or 'default'}.jsonl")


def append_turn(sid: str, role: str, text: str):
    with open(_path(sid), "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "role": role, "text": text}) + "\n")


def load_history(sid: str, limit: int = 50):
    p = _path(sid)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        lines = f.readlines()[-limit:]
    return [json.loads(l) for l in lines]


def build_rag_prompt(user_msg: str, docs: list[str] | None) -> str:
    if not docs:
        return user_msg
    ctx = "\n\n".join(f"[doc{i+1}] {d}" for i, d in enumerate(docs))
    return (f"Use the context below to answer. If unsure, say so.\n\n"
            f"<context>\n{ctx}\n</context>\n\n<user>\n{user_msg}\n</user>")
