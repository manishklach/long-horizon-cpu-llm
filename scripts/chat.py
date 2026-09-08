"""Streaming chat client (OpenAI-compatible)."""
import os
import requests

BASE = os.environ.get("CPU_LLM_BASE", "http://127.0.0.1:8000")
SID = "demo-session"

print("CPU-LLM chat — type exit/quit to stop.")
while True:
    try:
        q = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if q.lower() in ("exit", "quit"):
        break
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": q}],
        "max_tokens": 128, "session_id": SID}, timeout=300)
    r.raise_for_status()
    j = r.json()
    print("AI:", j["choices"][0]["message"]["content"])
    print(f"(ttft={j['perf']['ttft_s']}s tpot={j['perf']['tpot_s']}s/tok "
          f"ctx={j['perf']['session_len']})")
