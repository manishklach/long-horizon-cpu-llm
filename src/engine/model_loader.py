"""Multi-model loader (OPT, TinyLlama, Qwen, Phi, Llama, Mistral)."""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REGISTRY = {
    # light defaults for CPU demo (no 30B download needed to try the framework)
    "tiny-opt-125m": "facebook/opt-125m",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    # full-size long-context target (needs ~60GB+ RAM):
    "qwen3-coder-30b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
}

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def resolve_model_id(name_or_path: str) -> str:
    return MODEL_REGISTRY.get(name_or_path, name_or_path)


def load_model_and_tokenizer(name_or_path: str, dtype: str = "fp32",
                             trust_remote_code: bool = True):
    mid = resolve_model_id(name_or_path)
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        dtype=DTYPE_MAP.get(dtype, torch.float32),
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    return model, tok, {
        "model_id": mid,
        "n_layers": getattr(cfg, "num_hidden_layers", 12),
        "n_heads": getattr(cfg, "num_attention_heads", 12),
        "n_kv_heads": getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 12)),
        "head_dim": getattr(cfg, "hidden_size", 768) // getattr(cfg, "num_attention_heads", 12),
        "hidden": getattr(cfg, "hidden_size", 768),
    }
