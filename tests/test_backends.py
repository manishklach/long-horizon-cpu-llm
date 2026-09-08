"""Backend selection is tested without downloading models."""
from src.engine.backends import is_gguf_target


def test_gguf_detection():
    assert is_gguf_target("path/to/model-q4_k_m.gguf")
    assert not is_gguf_target("tiny-opt-125m")
