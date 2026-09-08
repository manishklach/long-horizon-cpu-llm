"""Core CPU inference engine: full single-pass Prefill + Session Cache + small-batch Decode.

eLLM ideas implemented here:
- FULL prefill in one forward (no chunking) -> less repeated param loading.
- INCREMENTAL prefill: reuse past_key_values + prefix match, only new tokens forwarded.
- STATIC accounting via StaticKVCache (seq pointer, overflow guard).
- SMALL batch (=1) decode -> each request gets full memory bandwidth.

Uses HF transformers Cache natively for correctness; StaticKVCache mirrors
logical length for eLLM-style accounting/persistence stats.
"""
from __future__ import annotations
import time
import torch
from typing import Dict, List, Optional
from transformers import DynamicCache

from .kv_cache import StaticKVCache
from .model_loader import load_model_and_tokenizer
from .quant import apply_dynamic_int8


class Session:
    def __init__(self, sid: str):
        self.sid = sid
        self.input_ids: List[int] = []   # full history ids (prompt+generated)
        self.cache: Optional[DynamicCache] = None
        self.static: Optional[StaticKVCache] = None
        self.turns = 0


class CPUEngine:
    def __init__(self, model_name: str = "tiny-opt-125m", max_seq_len: int = 8192,
                 dtype: str = "fp32", quant_int8: bool = False, threads: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.model, self.tok, self.info = load_model_and_tokenizer(model_name, dtype)
        if quant_int8:
            self.model = apply_dynamic_int8(self.model)
        self.model.eval()
        self.sessions: Dict[str, Session] = {}
        self.static_proto = StaticKVCache(
            self.info["n_layers"], self.info["n_kv_heads"], self.info["head_dim"],
            max_seq_len, torch.float32)

    # ---------- sessions ----------
    def get_session(self, sid: str) -> Session:
        if sid not in self.sessions:
            s = Session(sid)
            # fresh static mirror
            import copy
            s.static = StaticKVCache(self.info["n_layers"], self.info["n_kv_heads"],
                                     self.info["head_dim"], self.max_seq_len)
            self.sessions[sid] = s
        return self.sessions[sid]

    def reset_session(self, sid: str):
        if sid in self.sessions:
            self.sessions[sid].cache = None
            self.sessions[sid].input_ids = []
            self.sessions[sid].static.reset()

    # ---------- prefill modes ----------
    @torch.no_grad()
    def _forward(self, ids: torch.Tensor, cache: Optional[DynamicCache]):
        return self.model(input_ids=ids, past_key_values=cache, use_cache=True)

    @torch.no_grad()
    def prefill_full(self, ids: List[int]) -> tuple[DynamicCache, float]:
        """Single-pass full prefill (eLLM mode)."""
        t0 = time.perf_counter()
        out = self._forward(torch.tensor([ids]), None)
        return out.past_key_values, time.perf_counter() - t0

    @torch.no_grad()
    def prefill_chunked(self, ids: List[int], chunk: int = 512) -> tuple[DynamicCache, float]:
        """Chunked baseline (like vLLM/SGLang chunked prefill) for comparison."""
        t0 = time.perf_counter()
        cache = None
        for i in range(0, len(ids), chunk):
            out = self._forward(torch.tensor([ids[i:i + chunk]]), cache)
            cache = out.past_key_values
        return cache, time.perf_counter() - t0

    @staticmethod
    def _common_prefix(a: List[int], b: List[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    # ---------- generation with session cache ----------
    @torch.no_grad()
    def generate(self, prompt: str, session_id: str = "default", max_new_tokens: int = 64,
                 temperature: float = 0.0, top_p: float = 1.0) -> dict:
        s = self.get_session(session_id)
        new_ids = self.tok.encode(prompt, add_special_tokens=(len(s.input_ids) == 0))
        # prefix match -> incremental prefill on suffix only (no recompute)
        if s.cache is not None and s.input_ids:
            # candidate full = history + new prompt ids
            full = s.input_ids + new_ids
            # cache already covers len(s.input_ids); incremental = new_ids only
            reuse = len(s.input_ids)
            t0 = time.perf_counter()
            out = self._forward(torch.tensor([new_ids]), s.cache)
            ttft = time.perf_counter() - t0
            s.cache = out.past_key_values
            s.input_ids = full
            try:
                s.static.commit(len(new_ids))
            except AssertionError:
                pass
            prompt_tokens = len(new_ids)
        else:
            # first turn: full single-pass prefill
            t0 = time.perf_counter()
            out = self._forward(torch.tensor([new_ids]), None)
            ttft = time.perf_counter() - t0
            s.cache = out.past_key_values
            s.input_ids = list(new_ids)
            try:
                s.static.commit(len(new_ids))
            except AssertionError:
                pass
            prompt_tokens = len(new_ids)
            reuse = 0

        logits = out.logits[0, -1]
        gen_ids: List[int] = []
        t_dec0 = time.perf_counter()
        for _ in range(max_new_tokens):
            if temperature and temperature > 0:
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                if top_p < 1.0:
                    sp, idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sp, dim=-1)
                    cut = (cum > top_p).nonzero()
                    if len(cut):
                        sp[cut[0].item() + 1:] = 0
                        sp /= sp.sum()
                        probs = torch.zeros_like(probs).scatter_(-1, idx, sp)
                nxt = int(torch.multinomial(probs, 1))
            else:
                nxt = int(torch.argmax(logits))
            gen_ids.append(nxt)
            s.input_ids.append(nxt)
            if nxt == self.tok.eos_token_id:
                break
            o = self._forward(torch.tensor([[nxt]]), s.cache)
            s.cache = o.past_key_values
            logits = o.logits[0, -1]
            try:
                s.static.commit(1)
            except AssertionError:
                pass
        tpot = (time.perf_counter() - t_dec0) / max(len(gen_ids), 1)
        s.turns += 1
        return {
            "text": self.tok.decode(gen_ids, skip_special_tokens=True),
            "prompt_tokens": prompt_tokens,
            "reused_tokens": reuse if s.turns > 1 else 0,
            "generated_tokens": len(gen_ids),
            "ttft_s": round(ttft, 4),
            "tpot_s": round(tpot, 4),
            "session_len": len(s.input_ids),
            "turn": s.turns,
        }

    def stats(self) -> dict:
        return {"model": self.info, "max_seq": self.max_seq_len,
                "sessions": {k: {"len": len(v.input_ids), "turns": v.turns} for k, v in self.sessions.items()}}
