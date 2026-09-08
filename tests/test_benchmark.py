from types import SimpleNamespace
from src.bench import benchmark


class FakeEngine:
    def __init__(self, *args, **kwargs):
        self.max_seq_len = 10
        self.tok = SimpleNamespace(vocab_size=32)
    def prefill_full(self, ids):
        return None, 2.0
    def prefill_chunked(self, ids, chunk):
        return None, 3.0
    def reset_session(self, sid):
        pass
    def generate(self, ids, **kwargs):
        return {'ttft_s': 2.1, 'tpot_s': .1, 'generated_tokens': 2}


def test_actual_lengths_repeats_and_skips(monkeypatch):
    monkeypatch.setattr(benchmark, 'CPUEngine', FakeEngine)
    rows = benchmark.run('fake', [4, 100], max_new=2, repeats=3, warmups=1)
    assert rows[0]['seq_len'] == 4
    assert len(rows[0]['samples']) == 6
    assert rows[0]['speedup'] == 1.5
    assert rows[0]['ttft_s'] == 2.1
    assert rows[1]['status'] == 'skipped'
    assert rows[1]['seq_len'] is None
