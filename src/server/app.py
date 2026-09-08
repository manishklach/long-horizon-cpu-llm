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
import threading
import time
import uuid
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.engine.backends import create_backend
from src.memory.session_store import append_turn, build_rag_prompt

MODEL_ID = os.environ.get("CPU_LLM_MODEL", "tiny-opt-125m")
MAX_SEQ = int(os.environ.get("CPU_LLM_MAX_SEQ", "8192"))
QUANT = os.environ.get("CPU_LLM_INT8", "0") == "1"

app = FastAPI(title="CPU-LLM-Inference")
engine = None
_engine_lock = threading.Lock()


def get_engine():
    global engine
    with _engine_lock:
        if engine is None:
            engine = create_backend(MODEL_ID, max_seq=MAX_SEQ, quant_int8=QUANT)
    return engine


class ChatMsg(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatReq(BaseModel):
    messages: List[ChatMsg] = Field(min_length=1)
    max_tokens: int = Field(default=128, gt=0)
    temperature: float = Field(default=0.0, ge=0)
    stream: bool = False
    session_id: Optional[str] = Field(default=None, min_length=1, pattern=r"^[A-Za-z0-9_-]+$")


class CompleteReq(BaseModel):
    prompt: str
    max_tokens: int = Field(default=128, gt=0)
    temperature: float = Field(default=0.0, ge=0)
    session_id: Optional[str] = Field(default=None, min_length=1, pattern=r"^[A-Za-z0-9_-]+$")


class RagReq(ChatReq):
    documents: Optional[List[str]] = None


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


def _run_request(r, messages=None, on_text=None):
    e = get_engine()
    sid = r.session_id or uuid.uuid4().hex
    try:
        kwargs = dict(session_id=sid, max_new_tokens=r.max_tokens, temperature=r.temperature)
        if messages is not None:
            out = e.generate_chat(messages, on_text=on_text, **kwargs)
        else:
            out = e.generate(r.prompt, prompt_mode="full", **kwargs)
        if r.session_id:
            append_turn(sid, "user", messages[-1]["content"] if messages else r.prompt)
            append_turn(sid, "assistant", out["text"])
        return out
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if r.session_id is None:
            e.reset(sid)


def _usage(out):
    prompt, completion = out.get("prompt_tokens"), out.get("generated_tokens")
    if prompt is None or completion is None:
        return None
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}


@app.post("/v1/completions")
def completions(r: CompleteReq):
    out = _run_request(r)
    return {"id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion",
            "created": int(time.time()), "model": MODEL_ID,
            "choices": [{"index": 0, "text": out["text"], "finish_reason": out["finish_reason"]}],
            "usage": _usage(out), "perf": out}


@app.post("/v1/chat/completions")
def chat(r: ChatReq):
    if r.stream:
        import json
        import queue
        import threading
        events = queue.Queue()
        messages = [m.model_dump() for m in r.messages]
        def work():
            try:
                out = _run_request(r, messages, on_text=lambda text: events.put(("text", text)))
                events.put(("finish", out["finish_reason"]))
            except Exception as exc:
                events.put(("error", str(exc)))
            finally:
                events.put(("done", None))
        threading.Thread(target=work, daemon=True).start()
        ident, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        def gen():
            while True:
                kind, value = events.get()
                if kind == "done":
                    yield "data: [DONE]\n\n"
                    break
                if kind == "error":
                    payload = {"error": {"message": value}}
                else:
                    payload = {"id": ident, "object": "chat.completion.chunk", "created": created,
                               "model": MODEL_ID, "choices": [{"index": 0,
                               "delta": {"content": value} if kind == "text" else {},
                               "finish_reason": value if kind == "finish" else None}]}
                yield "data: " + json.dumps(payload) + "\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    out = _run_request(r, [m.model_dump() for m in r.messages])
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": out["text"]},
                         "finish_reason": out["finish_reason"]}], "usage": _usage(out), "perf": out}


@app.post("/v1/rag/chat")
def rag_chat(r: RagReq):
    if r.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="RAG requires a final user message")
    full = build_rag_prompt(r.messages[-1].content, r.documents)
    return chat(ChatReq(messages=[*r.messages[:-1], ChatMsg(role="user", content=full)],
                        max_tokens=r.max_tokens, temperature=r.temperature,
                        stream=r.stream, session_id=r.session_id))
