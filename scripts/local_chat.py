"""Local multi-turn chat without server (fastest way to try Session Cache)."""
import argparse
from src.engine.inference import CPUEngine
from src.memory.session_store import append_turn

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="tiny-opt-125m")
ap.add_argument("--max-seq", type=int, default=8192)
a = ap.parse_args()
e = CPUEngine(a.model, max_seq_len=a.max_seq)
print(f"Loaded {a.model}. Multi-turn uses incremental prefill (no recompute). Type exit to quit.")
sid = "local"
while True:
    try:
        q = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if q.lower() in ("exit", "quit"):
        break
    out = e.generate(q, session_id=sid, max_new_tokens=128)
    print("AI:", out["text"])
    print(f"(ttft={out['ttft_s']}s tpot={out['tpot_s']}s reused={out['reused_tokens']} ctx={out['session_len']})")
    append_turn(sid, "user", q)
    append_turn(sid, "assistant", out["text"])
