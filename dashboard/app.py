"""Streamlit dashboard: chat + benchmark + long-context memory.

Run:  streamlit run dashboard/app.py
Env:  CPU_LLM_MODEL=tiny-opt-125m  (or a .gguf path for llama.cpp)
"""
import json
import time
import streamlit as st

from src.engine.backends import create_backend, is_gguf_target
from src.engine.quant import kv_bytes_estimate, model_bytes_estimate
from src.bench.benchmark import run as bench_run

st.set_page_config(page_title="CPU-LLM Inference", layout="wide")
st.title("CPU-LLM Inference — eLLM-style + extras")

with st.sidebar:
    st.header("Engine")
    model = st.text_input("Model (HF id or .gguf path)", value="tiny-opt-125m")
    max_seq = st.number_input("Max seq / n_ctx", 512, 200000, 8192, step=512)
    use_int8 = st.checkbox("Dynamic INT8 (HF only)", value=False)
    st.caption(f"Backend auto: {'GGUF/llama.cpp' if is_gguf_target(model) else 'HF transformers'}")
    if st.button("Load / reload engine"):
        st.cache_resource.clear()

@st.cache_resource(show_spinner="Loading model (first run downloads)...")
def _engine(model_id, seq, int8):
    return create_backend(model_id, max_seq=seq, quant_int8=int8)

try:
    eng = _engine(model, int(max_seq), bool(use_int8))
    st.success(f"Engine ready: {eng.stats()}")
except Exception as e:
    st.error(f"Engine failed to load: {e}")
    st.stop()

tab_chat, tab_bench, tab_mem = st.tabs(["Chat (session cache)", "Benchmark", "Long-context memory"])

with tab_chat:
    sid = st.text_input("session_id", value="dash")
    q = st.text_area("You", value="Explain why full single-pass prefill beats chunking on CPUs.")
    c1, c2 = st.columns(2)
    max_new = c1.slider("max_new_tokens", 8, 512, 96)
    temp = c2.slider("temperature", 0.0, 1.5, 0.0)
    if st.button("Send"):
        t0 = time.perf_counter()
        with st.spinner("decoding on CPU..."):
            o = eng.generate(q, session_id=sid, max_new_tokens=max_new, temperature=temp)
        st.write(o["text"])
        st.json({k: o[k] for k in ("prompt_tokens", "reused_tokens", "generated_tokens",
                                   "ttft_s", "tpot_s", "session_len", "turn")})
        st.caption(f"wall {time.perf_counter()-t0:.2f}s — incremental prefill reused "
                   f"{o['reused_tokens']} tokens (no recompute)")
    if st.button("Reset session"):
        eng.reset(sid)
        st.info(f"session {sid} reset")

with tab_bench:
    st.write("Full single-pass Prefill vs chunked baseline + decode TPOT (HF backend).")
    lengths = st.text_input("seq lengths", value="128,512,1024,2048")
    chunk = st.number_input("chunk size (baseline)", 64, 8192, 512)
    if st.button("Run benchmark"):
        if is_gguf_target(model):
            st.warning("Benchmark loop needs HF backend; switch model to an HF id.")
        else:
            Ls = [int(x) for x in lengths.split(",") if x.strip().isdigit()]
            with st.spinner("benchmarking..."):
                rows = bench_run(eng.e.model_name if hasattr(eng, "e") else model,
                                 Ls, max_new=16, chunk=int(chunk))
            st.line_chart([{"full": r["ttft_full_s"], "chunked": r["ttft_chunked_s"]} for r in rows])
            st.table(rows)
            st.json(rows)

with tab_mem:
    st.write("DDR sizing for million-token contexts (the eLLM trade: storage for compute).")
    n_layers = st.number_input("layers", 1, 128, 12)
    n_kv = st.number_input("kv heads", 1, 64, 4)
    hd = st.number_input("head_dim", 16, 256, 64)
    seq = st.number_input("context tokens", 1000, 1000000, 50000, step=1000)
    gb = kv_bytes_estimate(int(n_layers), int(n_kv), int(hd), int(seq)) / 1e9
    st.metric("KV cache @FP16 (GB)", f"{gb:.2f}")
    st.caption("Static preallocation: reserve once, never rebuild the graph. "
               "If this fits your DDR, you can run full single-pass prefill at this length.")
    if st.button("Run 50K-shape check (no model)"):
        from scripts.long_context_test import phase_a_static_50k, phase_b_attention_long
        with st.spinner("filling 50K static cache..."):
            a = phase_a_static_50k(n_layers=2, n_kv_heads=2, head_dim=16, seq=50000)
            b = phase_b_attention_long(seq=1024)
        st.json({"static_50k": a, "attention": b})
