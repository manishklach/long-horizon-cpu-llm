"""Backend abstraction: same generate() API over HF transformers and GGUF/llama.cpp.

- HFBackend: wraps CPUEngine (full single-pass + session cache).
- LlamaCppBackend: loads .gguf via llama-cpp-python (CPU, n_threads auto).
  Transcript history is retained, but per-session resident KV reuse is not measured.

Auto-select: if model path ends with .gguf (or CPU_LLM_BACKEND=gguf) -> GGUF.

Install GGUF support (optional, Windows CPU wheels exist):
  pip install llama-cpp-python
  # or with OpenBLAS: set CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"

GGUF models to try (download once, no HF torch weights needed):
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF  (qwen2.5-0.5b-instruct-q4_k_m.gguf)
  https://huggingface.co/bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF
"""
from __future__ import annotations
import os
import time
from typing import Dict, List


def is_gguf_target(model_id: str) -> bool:
    m = model_id.lower()
    be = os.environ.get("CPU_LLM_BACKEND", "auto").lower()
    if be == "gguf":
        return True
    if be == "hf":
        return False
    return m.endswith(".gguf") or "gguf" in m


class HFBackend:
    kind = "hf"

    def __init__(self, model_id="tiny-opt-125m", max_seq=8192, quant_int8=False):
        from src.engine.inference import CPUEngine
        self.e = CPUEngine(model_id, max_seq_len=max_seq, quant_int8=quant_int8)

    def generate(self, prompt, session_id="default", max_new_tokens=64, temperature=0.0, **kwargs):
        return self.e.generate(prompt, session_id, max_new_tokens, temperature, **kwargs)

    def generate_chat(self, messages, **kwargs):
        return self.e.generate_chat(messages, **kwargs)

    def reset(self, sid):
        self.e.reset_session(sid)

    def turns(self, sid: str) -> int:
        s = self.e.sessions.get(sid)
        return s.turns if s else 0

    def stats(self):
        return {"backend": "hf-transformers", **self.e.stats()}


class LlamaCppBackend:
    """GGUF backend with session-cache (history reuse = incremental prefill)."""
    kind = "gguf"

    def __init__(self, model_path: str, n_ctx: int = 8192, n_threads: int | None = None,
                 n_gpu_layers: int = 0, verbose: bool = False):
        try:
            from llama_cpp import Llama
        except ImportError as ex:
            raise RuntimeError("pip install llama-cpp-python to use GGUF backend") from ex
        import multiprocessing
        self.Llama = Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx,
                         n_threads=n_threads or multiprocessing.cpu_count(),
                         n_gpu_layers=n_gpu_layers, verbose=verbose)
        import threading
        self.lock = threading.RLock()
        self.histories: Dict[str, List[dict]] = {}
        self.n_ctx = n_ctx
        self.model_path = model_path

    def _hist(self, sid: str) -> List[dict]:
        return self.histories.setdefault(sid, [])

    def generate(self, prompt, session_id="default", max_new_tokens=64, temperature=0.0,
                 prompt_mode="append"):
        if prompt_mode not in ("append", "full"):
            raise ValueError("prompt_mode must be append or full")
        with self.lock:
            history = self._hist(session_id) if prompt_mode == "append" else []
            return self.generate_chat([*history, {"role": "user", "content": prompt}],
                                      session_id, max_new_tokens, temperature)

    def generate_chat(self, messages, session_id="default", max_new_tokens=64, temperature=0.0, on_text=None):
        if max_new_tokens <= 0 or temperature < 0:
            raise ValueError("invalid generation parameters")
        with self.lock:
            start = time.perf_counter()
            # Native streaming supplies observable content-arrival latency.
            chunks = self.llm.create_chat_completion(messages=messages, max_tokens=max_new_tokens,
                                                       temperature=temperature, stream=True)
            pieces, arrivals = [], []
            finish = None
            for chunk in chunks:
                choice = chunk["choices"][0]
                content = choice.get("delta", {}).get("content")
                if content:
                    pieces.append(content)
                    arrivals.append(time.perf_counter())
                    if on_text:
                        on_text(content)
                finish = choice.get("finish_reason") or finish
            text = "".join(pieces)
            self.histories[session_id] = [*messages, {"role": "assistant", "content": text}]
            return {"text": text, "prompt_tokens": None, "reused_tokens": None,
                    "generated_tokens": None,
                    "ttft_s": arrivals[0] - start if arrivals else None,
                    "tpot_s": None, "total_s": time.perf_counter() - start,
                    "session_len": None, "turn": self.turns(session_id),
                    "finish_reason": finish,
                    "metrics_note": "TTFT is first content chunk; exact token usage and cache reuse unavailable."}

    def reset(self, sid):
        with self.lock:
            self.histories.pop(sid, None)

    def turns(self, sid: str) -> int:
        return len(self.histories.get(sid, [])) // 2

    def stats(self):
        with self.lock:
            return {"backend": "llama.cpp-gguf", "model": self.model_path,
                    "n_ctx": self.n_ctx, "sessions": {k: len(v) for k, v in self.histories.items()}}


def create_backend(model_id="tiny-opt-125m", max_seq=8192, quant_int8=False):
    if is_gguf_target(model_id):
        return LlamaCppBackend(model_path=model_id, n_ctx=max_seq)
    return HFBackend(model_id, max_seq, quant_int8)
