"""Backend abstraction: same generate() API over HF transformers and GGUF/llama.cpp.

- HFBackend: wraps CPUEngine (full single-pass + session cache, exact eLLM mirror).
- LlamaCppBackend: loads .gguf via llama-cpp-python (CPU, n_threads auto).
  llama.cpp already does key things eLLM wants on CPU: quantized weights in
  RAM, KV cache reuse across turns (we keep session prompt history so only new
  tokens are evaluated), small-batch decode.

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

    def generate(self, prompt, session_id="default", max_new_tokens=64, temperature=0.0):
        return self.e.generate(prompt, session_id, max_new_tokens, temperature)

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
        self.histories: Dict[str, List[dict]] = {}
        self.n_ctx = n_ctx
        self.model_path = model_path

    def _hist(self, sid: str) -> List[dict]:
        return self.histories.setdefault(sid, [])

    def generate(self, prompt, session_id="default", max_new_tokens=64, temperature=0.0):
        hist = self._hist(session_id)
        # session cache: keep prior turns in the evaluated context, only the new
        # user message is "incremental" work for the sampler loop
        msgs = [*hist, {"role": "user", "content": prompt}]
        t0 = time.perf_counter()
        out = self.llm.create_chat_completion(
            messages=msgs, max_tokens=max_new_tokens,
            temperature=temperature, stream=False)
        ttft = time.perf_counter() - t0  # llama.cpp lumps prefill+first token; reported as TTFT
        text = out["choices"][0]["message"]["content"]
        usage = out.get("usage", {})
        hist.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": text}])
        pt = usage.get("prompt_tokens", len(prompt.split()))
        ct = usage.get("completion_tokens", max_new_tokens)
        return {"text": text, "prompt_tokens": pt, "reused_tokens": sum(len(m["content"].split()) for m in hist[:-2]),
                "generated_tokens": ct, "ttft_s": round(ttft, 4),
                "tpot_s": round(ttft / max(ct, 1), 4),
                "session_len": len(hist), "turn": len(hist) // 2}

    def reset(self, sid):
        self.histories.pop(sid, None)

    def turns(self, sid: str) -> int:
        return len(self.histories.get(sid, [])) // 2

    def stats(self):
        return {"backend": "llama.cpp-gguf", "model": self.model_path,
                "n_ctx": self.n_ctx, "sessions": {k: len(v) for k, v in self.histories.items()}}


def create_backend(model_id="tiny-opt-125m", max_seq=8192, quant_int8=False):
    if is_gguf_target(model_id):
        return LlamaCppBackend(model_path=model_id, n_ctx=max_seq)
    return HFBackend(model_id, max_seq, quant_int8)
