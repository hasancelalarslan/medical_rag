"""
ui_app.py


Usage
------
# 1) Run with Docker Compose (recommended):
#    - Ensure your API service is named "api" (or set API_URL env)
#    - UI service should set: API_URL=http://api:8000/query
#
# 2) Local dev:
#    pip install streamlit requests
#    streamlit run ui_app.py
"""

from __future__ import annotations
import os
import time
import json
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

st.set_page_config(
    page_title=" 🩺 Medical RAG — BAYKAR ",
    page_icon="🩺",
    layout="wide",
)

def _format_ms(ms: Any) -> str:
    try:
        return f"{float(ms):.0f}"
    except Exception:
        return "?"

def _retry_post(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 120,
    retries: int = 2,
    backoff_sec: float = 1.0,
) -> requests.Response:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff_sec * (2 ** attempt))
    raise last_exc or requests.exceptions.RequestException("Unknown request error")

def _trim(text: str, max_chars: int = 2000) -> str:
    if not text:
        return ""
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

def _copyable(text: str, label: str = "Copy"):
    st.code(text)
    st.button(f"📋 {label}", on_click=st.session_state.__setitem__, args=("clipboard", text))


if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []
if "clipboard" not in st.session_state:
    st.session_state.clipboard = ""

st.sidebar.header("API Settings")
default_api_url = os.getenv("API_URL", "http://localhost:8000/query")
api_url = st.sidebar.text_input(
    "API URL",
    value=default_api_url,
    help="The POST endpoint of your Medical RAG API (e.g., http://api:8000/query in Docker).",
)

with st.sidebar.expander("Optional HTTP Headers (JSON)", expanded=False):
    default_headers = os.getenv("API_HEADERS_JSON", "").strip()
    headers_json = st.text_area(
        "Headers JSON",
        value=default_headers,
        height=80,
        placeholder='e.g., {"Authorization":"Bearer <token>"}',
        help="Custom headers to include in the request. Leave empty if not needed.",
    )
    req_headers: Optional[Dict[str, str]] = None
    if headers_json:
        try:
            req_headers = json.loads(headers_json)
        except Exception:
            st.sidebar.error("Invalid JSON in headers. They will be ignored.")
            req_headers = None

st.sidebar.markdown("---")
st.sidebar.header("Retrieval & Generation")
top_k = st.sidebar.slider(
    "Top-K Retrieved Chunks",
    min_value=1, max_value=20, value=5, step=1,
    help="Number of chunks to retrieve from the vector index (FAISS)."
)
temperature = st.sidebar.slider(
    "Generation Temperature",
    min_value=0.0, max_value=1.5, value=0.2, step=0.05,
    help="Lower = more deterministic; higher = more creative."
)
nprobe = st.sidebar.number_input(
    "nprobe (FAISS IVF)",
    min_value=1, max_value=4096, value=32, step=1,
    help="Controls recall/speed trade-off for FAISS IVF search."
)

with st.sidebar.expander("Advanced Parameters", expanded=False):
    top_p = st.slider("top_p", 0.0, 1.0, 1.0, 0.05)
    top_k_gen = st.number_input("top_k (generation)", min_value=0, max_value=200, value=0, step=1, help="0 = model default.")
    repetition_penalty = st.slider("repetition_penalty", 0.5, 2.5, 1.0, 0.05)
    max_new_tokens = st.number_input("max_new_tokens", min_value=0, max_value=4096, value=512, step=16)
    use_hallucination_guard = st.checkbox("Hallucination Guard (if supported)", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: When running with Docker Compose, set API_URL to http://api:8000/query.")

st.title("🩺 Medical RAG — Interactive Demo")
st.caption("Ask a medical question, tune retrieval/generation, and inspect the evidence. For clinical use, always verify with standard guidelines.")

cols_h = st.columns([1, 1, 3])
with cols_h[0]:
    if st.button("🔍 Check API /health", use_container_width=True):
        health_url = api_url.replace("/query", "/health")
        try:
            r = requests.get(health_url, timeout=10)
            if r.ok:
                st.success(f"Health OK: {r.json()}")
            else:
                st.error(f"Health HTTP {r.status_code}: {r.text[:300]}")
        except Exception as e:
            st.error(f"Health check failed: {e}")

with cols_h[1]:
    if st.button("🧹 Clear History", use_container_width=True):
        st.session_state.history.clear()
        st.success("History cleared.")


query = st.text_area(
    "Enter your medical question:",
    height=120,
    placeholder="e.g., What are first-line treatments for Type 2 diabetes?",
)

col_submit_left, col_submit_right = st.columns([3, 2])
with col_submit_left:
    ask_clicked = st.button("Ask", type="primary", use_container_width=True)
with col_submit_right:
    retries_opt = st.selectbox("Retries", options=[0, 1, 2], index=1, help="Auto-retry on transient API errors.")

if ask_clicked:
    if not query.strip():
        st.warning("Please enter a medical question before asking.")
    else:
        payload: Dict[str, Any] = {
            "query": query,
            "top_k": int(top_k),
            "temperature": float(temperature),
            "nprobe": int(nprobe),
            "top_p": float(top_p),
            "top_k_gen": int(top_k_gen),
            "repetition_penalty": float(repetition_penalty),
            "max_new_tokens": int(max_new_tokens),
            "hallucination_guard": bool(use_hallucination_guard),
        }

        with st.spinner("Querying API..."):
            t0 = time.time()
            try:
                resp = _retry_post(
                    api_url, payload, headers=req_headers, timeout=120,
                    retries=int(retries_opt), backoff_sec=1.0
                )
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
                st.stop()
            t1 = time.time()

        if resp.status_code != 200:
            st.error(f"API returned HTTP {resp.status_code}:\n{_trim(resp.text, max_chars=1000)}")
        else:
            try:
                data = resp.json()
            except Exception:
                st.error("API returned non-JSON response.")
                data = {}

            timings = data.get("timings", {}) or {}
            retrieval_ms = _format_ms(timings.get("retrieval_ms"))
            generation_ms = _format_ms(timings.get("generation_ms"))
            total_ms = _format_ms(timings.get("total_ms"))
            roundtrip_ms = (t1 - t0) * 1000.0

            st.success("Response received successfully.")
            st.markdown(
                f"**Latency:** Retrieval = `{retrieval_ms}` ms | "
                f"Generation = `{generation_ms}` ms | "
                f"Total = `{total_ms}` ms | "
                f"Round-trip = `{roundtrip_ms:.0f}` ms"
            )

            answer = data.get("answer") or data.get("text") or "No answer returned."
            st.markdown("###  Answer")
            st.write(answer)

            
            dl_cols = st.columns([1, 1, 2, 3])
            with dl_cols[0]:
                st.download_button(
                    "⬇ Answer (MD)",
                    data=f"# Answer\n\n{answer}\n",
                    file_name="answer.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with dl_cols[1]:
                st.download_button(
                    "⬇ Raw JSON",
                    data=json.dumps(data, ensure_ascii=False, indent=2),
                    file_name="response.json",
                    mime="application/json",
                    use_container_width=True,
                )

            # ---------------- SOURCES DISPLAY ----------------
            st.markdown("---")
            st.markdown("###  Retrieved Context")
            sources: List[Dict[str, Any]] = data.get("sources") or data.get("contexts") or []
            if not sources:
                st.info("No sources returned.")
            else:
                for i, src in enumerate(sources, start=1):
                    title = src.get("title") or src.get("source") or f"Source {i}"
                    score = src.get("score", "")
                    url = src.get("url")
                    chunk = src.get("text") or src.get("content") or ""
                    with st.expander(f"[{i}] {title}"):
                        srow = st.columns([2, 1, 2])
                        with srow[0]:
                            if url:
                                st.markdown(f"[🔗 Open Source Link]({url})")
                        with srow[1]:
                            st.markdown(f"**Score:** `{score}`")
                        with srow[2]:
                            meta = {k: v for k, v in src.items() if k not in ("text", "content")}
                            st.caption(f"Meta: `{_trim(json.dumps(meta, ensure_ascii=False), 300)}`")
                        st.write(chunk)

            st.session_state.history.insert(
                0,
                {
                    "query": query,
                    "answer": answer,
                    "sources": sources,
                    "timings": timings,
                    "roundtrip_ms": roundtrip_ms,
                    "params": {
                        "top_k": top_k, "temperature": temperature, "nprobe": nprobe,
                        "top_p": top_p, "top_k_gen": top_k_gen, "repetition_penalty": repetition_penalty,
                        "max_new_tokens": max_new_tokens, "hallucination_guard": use_hallucination_guard,
                    },
                },
            )

st.markdown("---")
st.subheader("Session History")

if not st.session_state.history:
    st.info("No history yet. Ask something to populate this section.")
else:
    hist_cols = st.columns([1, 1, 3, 3])
    with hist_cols[0]:
        # Prepare CSV-ish export
        rows = []
        for item in st.session_state.history:
            rows.append({
                "query": item["query"],
                "answer": _trim(item["answer"], 500).replace("\n", " "),
                "retrieval_ms": item["timings"].get("retrieval_ms"),
                "generation_ms": item["timings"].get("generation_ms"),
                "total_ms": item["timings"].get("total_ms"),
                "roundtrip_ms": f"{item['roundtrip_ms']:.0f}",
                "top_k": item["params"]["top_k"],
                "temperature": item["params"]["temperature"],
                "nprobe": item["params"]["nprobe"],
            })
        csv_lines = ["query,answer,retrieval_ms,generation_ms,total_ms,roundtrip_ms,top_k,temperature,nprobe"]
        for r in rows:
            q = r["query"].replace('"', '""')
            a = r["answer"].replace('"', '""')
            csv_lines.append(
                f"\"{q}\",\"{a}\",{r['retrieval_ms']},{r['generation_ms']},{r['total_ms']},{r['roundtrip_ms']},{r['top_k']},{r['temperature']},{r['nprobe']}"
            )
        st.download_button(
            "⬇ Export History (CSV)",
            data="\n".join(csv_lines),
            file_name="rag_ui_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with hist_cols[1]:
        md_parts = ["# RAG UI History Export\n"]
        for idx, item in enumerate(st.session_state.history, start=1):
            md_parts.append(textwrap.dedent(f"""
            ## {idx}. Query
            {item['query']}

            **Latency:** retrieval `{_format_ms(item['timings'].get('retrieval_ms'))}` ms |
            generation `{_format_ms(item['timings'].get('generation_ms'))}` ms |
            total `{_format_ms(item['timings'].get('total_ms'))}` ms |
            round-trip `{item['roundtrip_ms']:.0f}` ms

            ### Answer
            {item['answer']}

            ### Sources (top {len(item['sources'])})
            """).strip())
            for j, s in enumerate(item["sources"], start=1):
                ttl = s.get("title") or s.get("source") or f"Source {j}"
                url = s.get("url")
                score = s.get("score", "")
                md_parts.append(f"- [{j}] {ttl}  ({url if url else 'no-url'})  score={score}")
            md_parts.append("\n")
        st.download_button(
            "⬇ Export History (Markdown)",
            data="\n".join(md_parts),
            file_name="rag_ui_history.md",
            mime="text/markdown",
            use_container_width=True,
        )

    for idx, item in enumerate(st.session_state.history, start=1):
        with st.expander(f"{idx}. {item['query'][:80]}"):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("**Answer**")
                st.write(item["answer"])
            with c2:
                st.markdown("**Timings (ms)**")
                st.json({
                    "retrieval_ms": item["timings"].get("retrieval_ms"),
                    "generation_ms": item["timings"].get("generation_ms"),
                    "total_ms": item["timings"].get("total_ms"),
                    "roundtrip_ms": f"{item['roundtrip_ms']:.0f}",
                })
                st.markdown("**Params**")
                st.json(item["params"])

            st.markdown("**Sources**")
            if not item["sources"]:
                st.caption("No sources.")
            else:
                for j, s in enumerate(item["sources"], start=1):
                    ttl = s.get("title") or s.get("source") or f"Source {j}"
                    url = s.get("url")
                    score = s.get("score", "")
                    st.write(f"[{j}] {ttl} — score: `{score}`")
                    if url:
                        st.markdown(f"[🔗 Link]({url})")
                    st.caption(_trim(s.get("text") or s.get("content") or "", 800))


st.markdown("---")
st.caption("Powered by Medical RAG | Streamlit UI. This tool is for research/demonstration only and does not provide medical advice.")
