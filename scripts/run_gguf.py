"""Chat with a GGUF model via llama.cpp backend (quantized, CPU RAM friendly).

  pip install llama-cpp-python
  python scripts/run_gguf.py --model path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.engine.backends import LlamaCppBackend

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="path to .gguf file")
ap.add_argument("--n-ctx", type=int, default=8192)
a = ap.parse_args()
b = LlamaCppBackend(a.model, n_ctx=a.n_ctx)
print(f"GGUF ready: {a.model} (n_ctx={a.n_ctx}). Type exit to quit.")
sid = "gguf"
while True:
    try:
        q = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if q.lower() in ("exit", "quit"):
        break
    o = b.generate(q, session_id=sid, max_new_tokens=128)
    print("AI:", o["text"])
    print(f"(ttft~{o['ttft_s']}s ctx_msgs={o['session_len']})")
