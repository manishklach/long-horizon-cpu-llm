"""Download GGUF quantized models (CPU RAM friendly, no torch weights needed).

Presets (Q4_K_M ~4-bit, best size/quality for CPU):
  qwen2.5-0.5b  -> Qwen/Qwen2.5-0.5B-Instruct-GGUF / qwen2.5-0.5b-instruct-q4_k_m.gguf (~400MB)
  qwen2.5-1.5b  -> Qwen/Qwen2.5-1.5B-Instruct-GGUF / qwen2.5-1.5b-instruct-q4_k_m.gguf (~1GB)
  tinyllama-1.1b -> bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF / TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf (~700MB)
  qwen3-0.6b    -> Qwen/Qwen3-0.6B-GGUF / Qwen3-0.6B-Q4_K_M.gguf

Usage:
  python scripts/download_gguf.py --preset qwen2.5-0.5b --out models/
  python scripts/download_gguf.py --repo Qwen/Qwen2.5-0.5B-Instruct-GGUF --file qwen2.5-0.5b-instruct-q4_k_m.gguf
  python scripts/run_gguf.py --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --n-ctx 8192
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRESETS = {
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "tinyllama-1.1b": ("bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF", "TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"),
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--preset", choices=list(PRESETS), default="qwen2.5-0.5b")
ap.add_argument("--repo", default=None)
ap.add_argument("--file", default=None)
ap.add_argument("--out", default="models")
a = ap.parse_args()

repo, fname = PRESETS[a.preset] if a.repo is None else (a.repo, a.file)
assert fname, "--file required with custom --repo"
os.makedirs(a.out, exist_ok=True)
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise SystemExit("pip install huggingface_hub (comes with transformers)")

path = hf_hub_download(repo_id=repo, filename=fname, local_dir=a.out,
                       local_dir_use_symlinks=False)
print(f"downloaded -> {path}")
print(f"chat: python scripts/run_gguf.py --model {path} --n-ctx 8192")
