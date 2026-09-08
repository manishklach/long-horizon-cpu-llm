"""OpenAI-compatible FastAPI server.

Endpoints:
- POST /v1/chat/completions  (OpenAI chat, stream + non-stream)
- POST /v1/completions       (legacy completion)
- POST /v1/rag/chat          (RAG injection)
- GET  /health, GET /metrics, GET /v1/models
- POST /v1/sessions/{sid}/reset
"""
from __future__ import annotations
import os
import time
import uuid
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from src.engine.backends import create_backend
from src.memory.session_store import append_turn, build_rag_prompt

MODEL_ID = os.environ.get("CPU_LLM_MODEL", "tiny-opt-125m")
MAX_SEQ = int(os.environ.get("CPU_LLM_MAX_SEQ", "8192"))
QUANT = os.environ.get("CPU_LLM_INT8", "0") == "1"

app = FastAPI(title="CPU-LLM-Inference")
engine = None


def get_engine():
    global engine
    if engine is None:
        engine = create_backend(MODEL_ID, max_seq=MAX_SEQ, quant_int8=QUANT)
    return engine


class ChatMsg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    messages: List[ChatMsg]
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = False
    session_id: str = "default"


class CompleteReq(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    session_id: str = "default"


class RagReq(ChatReq):
    documents: Optional[List[str]] = None


def _last_user_text(msgs: List[ChatMsg]) -> str:
    for m in reversed(msgs):
        if m.role == "user":
            return m.content
    return msgs[-1].content if msgs else ""


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.get("/v1/models")
def models():
    return {"data": [{"id": MODEL_ID, "owned_by": "cpu-llm"}]}


@app.get("/metrics")
def metrics():
    e = get_engine()
    return e.stats()


@app.post("/v1/sessions/{sid}/reset")
def reset_session(sid: str):
    get_engine().reset(sid)
    return {"ok": True, "session": sid}


@app.post("/v1/completions")
def completions(r: CompleteReq):
    e = get_engine()
    out = e.generate(r.prompt, session_id=r.session_id,
                     max_new_tokens=r.max_tokens, temperature=r.temperature)
    append_turn(r.session_id, "user", r.prompt)
    append_turn(r.session_id, "assistant", out["text"])
    return {"id": f"cmpl-{uuid.uuid4().hex[:8]}", "model": MODEL_ID,
            "choices": [{"text": out["text"], "finish_reason": "stop"}],
            "usage": {"prompt_tokens": out["prompt_tokens"],
                      "completion_tokens": out["generated_tokens"]},
            "perf": {"ttft_s": out["ttft_s"], "tpot_s": out["tpot_s"]}}


@app.post("/v1/chat/completions")
def chat(r: ChatReq):
    e = get_engine()
    prompt = _last_user_text(r.messages)
    # system messages become prefix context on first turn
    sys = "\n".join(m.content for m in r.messages if m.role == "system")
    if sys and e.turns(r.session_id) == 0:
        prompt = sys + "\n\n" + prompt
    if r.stream:
        def gen():
            out = e.generate(prompt, session_id=r.session_id,
                             max_new_tokens=r.max_tokens, temperature=r.temperature)
            append_turn(r.session_id, "user", prompt)
            append_turn(r.session_id, "assistant", out["text"])
            import json as _json
            for tok in out["text"].split(" "):
                chunk = {"choices": [{"delta": {"content": tok + " "}}]}
                yield "data: " + _json.dumps(chunk) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    out = e.generate(prompt, session_id=r.session_id,
                     max_new_tokens=r.max_tokens, temperature=r.temperature)
    append_turn(r.session_id, "user", prompt)
    append_turn(r.session_id, "assistant", out["text"])
    return {"id": f"chat-{uuid.uuid4().hex[:8]}", "model": MODEL_ID,
            "choices": [{"message": {"role": "assistant", "content": out["text"]},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": out["prompt_tokens"],
                      "completion_tokens": out["generated_tokens"]},
            "perf": out}


@app.post("/v1/rag/chat")
def rag_chat(r: RagReq):
    prompt = _last_user_text(r.messages)
    full = build_rag_prompt(prompt, r.documents)
    return chat(ChatReq(messages=[ChatMsg(role="user", content=full)],
                        max_tokens=r.max_tokens, temperature=r.temperature,
                        stream=False, session_id=r.session_id))
