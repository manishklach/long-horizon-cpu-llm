from src.bench.tasks import build_case, exact_answer, token_hash


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return '\n'.join(m['role'] + ': ' + m['content'] for m in messages) + '\nassistant:'
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


def test_retrieval_tasks_fit_and_have_no_answer_leak():
    tok = CharacterTokenizer()
    a = build_case(tok, 4096, .1, 17)
    b = build_case(tok, 4096, .9, 17)
    assert a['input_ids'] == build_case(tok, 4096, .1, 17)['input_ids']
    assert a['prompt_tokens'] <= 4096
    assert a['prompt'].count(a['expected']) == 1
    assert a['needle_token_offset'] < b['needle_token_offset']
    absent = build_case(tok, 4096, .5, 17, absent=True)
    assert a['expected'] not in absent['prompt']
    assert absent['expected'] == 'UNKNOWN'
    assert absent['needle_token_offset'] is None
    assert token_hash(a['input_ids']) != token_hash(b['input_ids'])


def test_exact_match_does_not_reward_answer_lists():
    assert exact_answer(' "CODE-ABCDEFGH". ', 'CODE-ABCDEFGH')
    assert not exact_answer('Maybe CODE-ABCDEFGH or CODE-XYZ.', 'CODE-ABCDEFGH')
    assert not exact_answer('The answer is CODE-ABCDEFGH', 'CODE-ABCDEFGH')
    assert exact_answer('UNKNOWN', 'UNKNOWN')


def test_gguf_runner_counts_only_verified_cached_tokens(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from src.bench.runners import Runner
    class FakeLlama:
        def __init__(self):
            self.n_tokens = 0
            self.input_ids = np.zeros(64, dtype=int)
        def reset(self):
            self.n_tokens = 0
        def generate(self, tokens, **kwargs):
            pending = list(tokens)
            while True:
                self.input_ids[self.n_tokens:self.n_tokens+len(pending)] = pending
                self.n_tokens += len(pending)
                token = int(sum(self.input_ids[:self.n_tokens]) % 20 + 3)
                yield token
                pending = [token]
    runner = Runner.__new__(Runner)
    runner.backend = 'gguf'
    runner.context = 64
    runner.llm = FakeLlama()
    runner.tok = SimpleNamespace(eos_token_id=2, decode=lambda ids, **kwargs: str(ids))
    first = runner.generate([1, 4, 6], max_new=3)
    ids = [1, 4, 6] + first['output_ids'] + [8]
    warm = runner.generate(ids, max_new=3, cold=False)
    cold = runner.generate(ids, max_new=3, cold=True)
    assert warm['output_ids'] == cold['output_ids']
    assert warm['reused_tokens'] == 5
    assert warm['evaluated_prompt_tokens'] == len(ids) - 5
    assert cold['reused_tokens'] == 0
    changed = runner.generate([1, 5], max_new=1, cold=False)
    assert changed['reused_tokens'] == 0
    assert changed['tpot_s'] is None


def test_runner_rejects_context_overflow():
    import pytest
    from src.bench.runners import Runner
    runner = Runner.__new__(Runner)
    runner.context = 10
    with pytest.raises(ValueError):
        runner.generate([1] * 9, max_new=2)


def test_summary_rejects_partial_runs():
    import pytest
    from src.bench.summarize import summarize
    with pytest.raises(ValueError, match='incomplete'):
        summarize([{'complete': False}])


def test_prepare_handles_absolute_download_paths_and_checksums(tmp_path, monkeypatch):
    import json
    import pytest
    import huggingface_hub
    from src.bench.runners import prepare, verify, sha256
    monkeypatch.chdir(tmp_path)
    hf = tmp_path / 'models' / 'qwen2.5-0.5b-hf'
    gguf = tmp_path / 'models' / 'qwen2.5-0.5b-gguf' / 'tiny.gguf'
    hf.mkdir(parents=True)
    gguf.parent.mkdir(parents=True)
    (hf / 'config.json').write_text('{}')
    gguf.write_bytes(b'test model')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda *a, **kw: str(hf))
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda *a, **kw: str(gguf))
    spec = {'hf_repo': 'test', 'hf_revision': 'abc', 'gguf_repo': 'test-gguf',
            'gguf_revision': 'def', 'gguf_file': 'tiny.gguf', 'gguf_sha256': sha256(gguf)}
    manifest = prepare(spec, 'models')
    assert verify(spec, 'models') == manifest
    (hf / 'config.json').write_text('{"modified": true}')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify(spec, 'models')


def test_tokenizer_mismatch_stops_comparison():
    import pytest
    from types import SimpleNamespace
    from src.bench.runners import Runner
    runner = Runner.__new__(Runner)
    runner.backend = 'gguf'
    runner.tok = SimpleNamespace(encode=lambda *a, **kw: [1, 2])
    runner.llm = SimpleNamespace(tokenize=lambda *a, **kw: [1, 3])
    with pytest.raises(ValueError, match='tokenizer mismatch'):
        runner.check_prompt({'prompt': 'example', 'input_ids': [1, 2]})


def test_followup_pair_changes_only_reference():
    from src.bench.tasks import build_followups
    tok = CharacterTokenizer()
    initial = build_case(tok, 4096, .1, 29)
    implicit, explicit = build_followups(tok, initial, initial['expected'])
    assert implicit['messages'][:-1] == explicit['messages'][:-1]
    a = implicit['messages'][-1]['content']
    b = explicit['messages'][-1]['content']
    assert a.replace('the same requested record', 'record ' + initial['target_record']) == b
    assert initial['expected'] not in b
    assert implicit['initial_prompt_sha256'] == explicit['initial_prompt_sha256']


def test_recall_protocol_runs_all_seeds_cold_and_conditions_scores(tmp_path, monkeypatch):
    from src.bench import recall_controls
    calls = []
    class FakeRunner:
        tok = CharacterTokenizer()
        settings = {'backend': 'fake'}
        manifest = {}
        def __init__(self, *args):
            pass
        def check_prompt(self, case):
            pass
        def generate(self, ids, max_new, cold):
            calls.append(cold)
            return {'text': 'UNKNOWN', 'ttft_s': 0}
    monkeypatch.setattr(recall_controls, 'Runner', FakeRunner)
    monkeypatch.setattr(recall_controls, 'environment', lambda: {})
    protocol = {'seeds': [17, 29, 41], 'expected_cases': 12, 'backend': 'fake', 'context': 16384,
                'threads': 1, 'batch': 512, 'initial_budget': 4096, 'absent_budget': 8192,
                'position': .1, 'max_new': 24}
    report = recall_controls.run(protocol, {}, '.', tmp_path / 'report.json')
    assert report['complete'] and len(report['records']) == 12
    assert calls == [True] * 12
    assert [r.get('variant') for r in report['records'] if r['kind'] == 'followup'] == [
        'implicit', 'explicit', 'explicit', 'implicit', 'implicit', 'explicit']
    summary = recall_controls.summarize(report)
    assert 'conditional on correct initial retrieval: 0/0' in summary
    assert 'Absent-record controls: 3/3' in summary
