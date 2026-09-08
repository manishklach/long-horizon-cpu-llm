"""Quantization + memory helpers (extra beyond eLLM baseline).

- Dynamic INT8 for CPU Linear layers (torch.ao, no calibration needed).
- FP16/BF16 dtype selection.
- Memory estimator so users can size DDR for million-token contexts.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def model_bytes_estimate(num_params: int, dtype: torch.dtype) -> int:
    bits = torch.finfo(dtype).bits if dtype in (torch.float16, torch.bfloat16, torch.float32) else 8
    return num_params * bits // 8


def kv_bytes_estimate(n_layers: int, n_kv_heads: int, head_dim: int, seq_len: int,
                      dtype: torch.dtype = torch.float16) -> int:
    b = torch.finfo(dtype).bits // 8 if dtype != torch.int8 else 1
    return 2 * n_layers * seq_len * n_kv_heads * head_dim * b


def apply_dynamic_int8(model: nn.Module) -> nn.Module:
    """Quantize nn.Linear to INT8 dynamically (CPU-friendly, latency win in decode)."""
    try:
        q = torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        return q
    except Exception as e:
        print(f"[quant] dynamic INT8 failed ({e}), returning FP model")
        return model


def apply_better_dtype(model: nn.Module, dtype_str: str = "fp32") -> nn.Module:
    d = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype_str, torch.float32)
    if d != torch.float32:
        try:
            return model.to(d)
        except Exception as e:
            print(f"[quant] dtype {dtype_str} failed ({e})")
    return model
