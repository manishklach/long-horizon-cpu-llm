"""Backend selection is tested without downloading models."""
from src.engine.backends import is_gguf_target


def test_gguf_detection():
    assert is_gguf_target("path/to/model-q4_k_m.gguf")
    assert not is_gguf_target("tiny-opt-125m")


def test_gguf_does_not_invent_token_metrics():
    import threading
    from src.engine.backends import LlamaCppBackend
    class Llama:
        def create_chat_completion(self, **kwargs):
            assert kwargs['stream'] is True
            yield {'choices': [{'delta': {'content': 'hi'}, 'finish_reason': None}]}
            yield {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend.llm = Llama()
    backend.lock = threading.RLock()
    backend.histories = {}
    result = backend.generate_chat([{'role': 'user', 'content': 'hello'}])
    assert result['text'] == 'hi'
    assert result['ttft_s'] >= 0
    assert result['reused_tokens'] is None
    assert result['tpot_s'] is None
    assert result['generated_tokens'] is None
