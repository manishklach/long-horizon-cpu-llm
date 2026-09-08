"""Long-horizon demo: multi-turn incremental prefill + RAG (no server needed)."""
from src.engine.inference import CPUEngine

e = CPUEngine("tiny-opt-125m", max_seq_len=4096)
sid = "research-demo"
turns = [
    "The project uses large DDR memory to avoid chunked prefill.",
    "Summarize what I just said in one line.",
    "Now add: head-by-head attention keeps KV in L3 cache.",
]
for t in turns:
    out = e.generate(t, session_id=sid, max_new_tokens=48)
    print(f"\nYou: {t}\nAI: {out['text']}\n"
          f"[ttft={out['ttft_s']}s tpot={out['tpot_s']} reused={out['reused_tokens']} ctx={out['session_len']}]")
print("\nSession stats:", e.stats())
