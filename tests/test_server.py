import json
from fastapi.testclient import TestClient
from src.server import app as server


class Backend:
    def __init__(self):
        self.calls = []
        self.resets = []
    def generate_chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs['session_id']))
        if kwargs.get('on_text'):
            kwargs['on_text']('hello ')
            kwargs['on_text']('world')
        return {'text': 'hello world', 'finish_reason': 'length'}
    def reset(self, sid):
        self.resets.append(sid)


def test_history_isolation_and_stream(monkeypatch):
    backend = Backend()
    monkeypatch.setattr(server, 'engine', backend)
    client = TestClient(server.app)
    messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'first'},
                {'role': 'assistant', 'content': 'answer'}, {'role': 'user', 'content': 'second'}]
    for _ in range(2):
        response = client.post('/v1/chat/completions', json={'messages': messages})
        assert response.status_code == 200
    assert backend.calls[0][0] == messages
    assert backend.calls[0][1] != backend.calls[1][1]
    assert len(backend.resets) == 2
    response = client.post('/v1/chat/completions', json={'messages': messages, 'stream': True})
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith('data: ') and line != 'data: [DONE]']
    assert ''.join(e['choices'][0]['delta'].get('content', '') for e in events) == 'hello world'
    assert events[-1]['choices'][0]['finish_reason'] == 'length'
    assert client.post('/v1/chat/completions', json={'messages': []}).status_code == 422
