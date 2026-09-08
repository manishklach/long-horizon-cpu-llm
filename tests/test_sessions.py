import pytest
import torch
from transformers import OPTConfig, OPTForCausalLM
from src.engine import inference


class Tokenizer:
    eos_token_id = 2
    chat_template = 'test'
    vocab_size = 32
    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [3 + ord(c) % 29 for c in text]
    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return ' '.join(str(i) for i in ids if i != 2)
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1] + [3 + len(m['content']) % 29 for m in messages] + [4]


@pytest.fixture
def engine(monkeypatch):
    torch.manual_seed(7)
    model = OPTForCausalLM(OPTConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                                   ffn_dim=32, num_attention_heads=2, max_position_embeddings=64,
                                   dropout=0, attention_dropout=0))
    monkeypatch.setattr(inference, 'load_model_and_tokenizer', lambda *args: (model, Tokenizer(), {}))
    return inference.CPUEngine(max_seq_len=128, threads=1)


def test_cached_matches_uncached_and_changed_history(engine):
    engine.generate([1, 5, 6], session_id='cached', max_new_tokens=3, prompt_mode='full')
    history = engine.sessions['cached'].input_ids + [7, 8]
    cached = engine.generate(history, session_id='cached', max_new_tokens=4, prompt_mode='full')
    fresh = engine.generate(history, session_id='fresh', max_new_tokens=4, prompt_mode='full')
    assert cached['text'] == fresh['text']
    assert cached['reused_tokens'] > 0
    changed = engine.generate([1, 12, 13], session_id='cached', max_new_tokens=4, prompt_mode='full')
    reference = engine.generate([1, 12, 13], session_id='reference', max_new_tokens=4, prompt_mode='full')
    assert changed['reused_tokens'] == 0
    assert changed['text'] == reference['text']


def test_eos_pending_alignment(engine):
    from types import SimpleNamespace
    original = engine._forward
    def force_eos(ids, cache):
        out = original(ids, cache)
        logits = torch.full_like(out.logits, -100)
        logits[..., 2] = 100
        return SimpleNamespace(logits=logits, past_key_values=out.past_key_values)
    engine._forward = force_eos
    result = engine.generate([1, 5], max_new_tokens=3)
    assert result['finish_reason'] == 'stop'
    assert result['tpot_s'] is None
    s = engine.sessions['default']
    assert s.input_ids[-1] == 2
    assert s.cache.get_seq_length() == len(s.input_ids) - 1
    engine.generate([6], max_new_tokens=3)
    assert s.cache.get_seq_length() == len(s.input_ids) - 1


def test_reset_limits_and_stream(engine):
    pieces = []
    out = engine.generate([1, 6], max_new_tokens=4, on_text=pieces.append)
    assert ''.join(pieces) == out['text']
    assert engine.max_seq_len == 64
    with pytest.raises(ValueError, match='overflow'):
        engine.generate([3] * 64, max_new_tokens=1)
    engine.reset_session('default')
    assert 'default' not in engine.sessions
    assert engine.generate([1, 6], max_new_tokens=1)['turn'] == 1
    with pytest.raises(ValueError):
        engine.generate([], max_new_tokens=1)


def test_prefill_full_chunked_logits(engine):
    ids = [1, 4, 7, 8, 9]
    full, _ = engine.prefill_full(ids)
    chunked, _ = engine.prefill_chunked(ids, chunk=2)
    a = engine._forward(torch.tensor([[10]]), full).logits
    b = engine._forward(torch.tensor([[10]]), chunked).logits
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_chat_template_is_used(engine):
    messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'hi'}]
    ids = engine.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    a = engine.generate_chat(messages, session_id='chat', max_new_tokens=3)
    b = engine.generate(ids, session_id='raw', max_new_tokens=3, prompt_mode='full')
    assert a['text'] == b['text']
    assert engine.sessions['chat'].input_ids == engine.sessions['raw'].input_ids


def test_failed_forward_invalidates_mutated_cache(engine):
    engine.generate([1, 5], max_new_tokens=2)
    def failure(ids, cache):
        raise RuntimeError('failure')
    engine._forward = failure
    with pytest.raises(RuntimeError):
        engine.generate([6], max_new_tokens=2)
    assert 'default' not in engine.sessions
