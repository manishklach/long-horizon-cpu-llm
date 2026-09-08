"""CPU inference with exact prefix reuse; static KV remains a separate experiment."""
from __future__ import annotations
import time
import threading
import torch
from typing import Dict, List, Optional
from transformers import DynamicCache

from .model_loader import load_model_and_tokenizer
from .quant import apply_dynamic_int8


class Session:
    def __init__(self, sid: str):
        self.sid = sid
        self.input_ids: List[int] = []   # full history ids (prompt+generated)
        self.cache: Optional[DynamicCache] = None
        self.turns = 0


class CPUEngine:
    def __init__(self, model_name: str = "tiny-opt-125m", max_seq_len: int = 8192,
                 dtype: str = "fp32", quant_int8: bool = False, threads: int | None = None):
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if threads:
            torch.set_num_threads(threads)
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.model, self.tok, self.info = load_model_and_tokenizer(model_name, dtype)
        if quant_int8:
            self.model = apply_dynamic_int8(self.model)
        self.model.eval()
        self.sessions: Dict[str, Session] = {}
        self.lock = threading.RLock()
        limit = getattr(self.model.config, "max_position_embeddings", None) or getattr(self.model.config, "n_positions", max_seq_len)
        self.max_seq_len = min(max_seq_len, limit)

    # ---------- sessions ----------
    def get_session(self, sid: str) -> Session:
        if sid not in self.sessions:
            s = Session(sid)
            self.sessions[sid] = s
        return self.sessions[sid]

    def reset_session(self, sid: str):
        with self.lock:
            self.sessions.pop(sid, None)

    # ---------- prefill modes ----------
    @torch.no_grad()
    def _forward(self, ids: torch.Tensor, cache: Optional[DynamicCache]):
        return self.model(input_ids=ids, past_key_values=cache, use_cache=True)

    @torch.no_grad()
    def prefill_full(self, ids: List[int]) -> tuple[DynamicCache, float]:
        """Single-pass full prefill (no chunking)."""
        self._validate_prefill(ids)
        t0 = time.perf_counter()
        out = self._forward(torch.tensor([ids]), None)
        return out.past_key_values, time.perf_counter() - t0

    @torch.no_grad()
    def prefill_chunked(self, ids: List[int], chunk: int = 512) -> tuple[DynamicCache, float]:
        """Chunked HF baseline using the same resident model."""
        self._validate_prefill(ids)
        if chunk <= 0:
            raise ValueError("chunk must be positive")
        t0 = time.perf_counter()
        cache = None
        for i in range(0, len(ids), chunk):
            out = self._forward(torch.tensor([ids[i:i + chunk]]), cache)
            cache = out.past_key_values
        return cache, time.perf_counter() - t0

    def _validate_prefill(self, ids):
        if not ids or len(ids) > self.max_seq_len:
            raise ValueError("prefill must be nonempty and fit the context limit")

    def generate_chat(self, messages, **kwargs):
        if self.tok.chat_template:
            ids = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
            ids = self.tok.encode(text, add_special_tokens=True)
        return self.generate(ids, prompt_mode="full", **kwargs)

    @torch.no_grad()
    def generate(self, prompt, session_id="default", max_new_tokens=64,
                 temperature=0.0, top_p=1.0, prompt_mode="append", on_text=None):
        """Append raw text locally; full mode accepts authoritative token history.

        After generation the final token is pending: cache covers input_ids[:-1].
        This also holds for EOS, which is evaluated on the next turn.
        """
        if max_new_tokens <= 0 or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("invalid generation parameters")
        if prompt_mode not in ("append", "full"):
            raise ValueError("prompt_mode must be append or full")
        with self.lock:
            s = self.get_session(session_id)
            ids = (self.tok.encode(prompt, add_special_tokens=prompt_mode == "full" or not s.input_ids)
                   if isinstance(prompt, str) else list(prompt))
            if not ids:
                raise ValueError("prompt must contain tokens")
            full = s.input_ids + ids if prompt_mode == "append" else ids
            if len(full) + max_new_tokens > self.max_seq_len:
                raise ValueError(f"context overflow: {len(full)} + {max_new_tokens} > {self.max_seq_len}")
            cached = s.input_ids[:-1]
            reuse = len(cached) if s.cache is not None and full[:len(cached)] == cached and len(full) > len(cached) else 0
            cache = s.cache if reuse else None
            start = time.perf_counter()
            gen_ids, arrivals = [], []
            emitted = ""
            try:
                out = self._forward(torch.tensor([full[reuse:]]), cache)
                cache, logits = out.past_key_values, out.logits[0, -1]
                for i in range(max_new_tokens):
                    if temperature > 0:
                        probs = torch.softmax(logits / temperature, dim=-1)
                        if top_p < 1:
                            sp, idx = torch.sort(probs, descending=True)
                            sp[torch.cumsum(sp, -1) - sp >= top_p] = 0
                            probs = torch.zeros_like(probs).scatter_(-1, idx, sp / sp.sum())
                        nxt = int(torch.multinomial(probs, 1))
                    else:
                        nxt = int(torch.argmax(logits))
                    gen_ids.append(nxt)
                    arrivals.append(time.perf_counter())
                    if on_text:
                        decoded = self.tok.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        # Keep an incomplete word/UTF-8 tail until it is stable.
                        boundary = max(decoded.rfind(" "), decoded.rfind("\n")) + 1
                        stable = decoded[:boundary]
                        if stable.startswith(emitted) and len(stable) > len(emitted):
                            on_text(stable[len(emitted):])
                            emitted = stable
                    if nxt == self.tok.eos_token_id or i + 1 == max_new_tokens:
                        break
                    out = self._forward(torch.tensor([[nxt]]), cache)
                    cache, logits = out.past_key_values, out.logits[0, -1]
            except Exception:
                self.sessions.pop(session_id, None)
                raise
            decoded = self.tok.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if on_text and decoded.startswith(emitted):
                on_text(decoded[len(emitted):])
            s.cache, s.input_ids = cache, full + gen_ids
            s.turns += 1
            return {"text": self.tok.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    "prompt_tokens": len(full), "reused_tokens": reuse,
                    "generated_tokens": len(gen_ids), "ttft_s": arrivals[0] - start,
                    "tpot_s": (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) if len(arrivals) > 1 else None,
                    "session_len": len(s.input_ids), "turn": s.turns,
                    "finish_reason": "stop" if gen_ids[-1] == self.tok.eos_token_id else "length"}

    def stats(self) -> dict:
        with self.lock:
            return {"model": self.info, "max_seq": self.max_seq_len,
                    "sessions": {k: {"len": len(v.input_ids), "turns": v.turns} for k, v in self.sessions.items()}}
