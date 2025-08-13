
import os, csv, time, json, sys
from typing import Dict, List, Optional
import httpx

# Config
API = os.getenv("API_URL", "http://127.0.0.1:8000")
IN_CSV  = os.getenv("PERF_INPUT",  "examples/queries_100.csv")
OUT_CSV = os.getenv("PERF_OUTPUT", "eval/perf_results.csv")
MODEL_OUT_CSV = os.getenv("MODEL_OUTPUTS", "eval/model_outputs.csv")
K = int(os.getenv("TOPK", "5"))


LOG_EVERY = int(os.getenv("LOG_EVERY", "10"))   
VERBOSE   = os.getenv("VERBOSE", "0") == "1"    



def _ensure_dirs():
    for p in [OUT_CSV, MODEL_OUT_CSV]:
        d = os.path.dirname(p) or "."
        os.makedirs(d, exist_ok=True)

def _flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def ping_health(client: httpx.Client) -> None:
    try:
        r = client.get(f"{API}/health", timeout=10)
        r.raise_for_status()
        _flush_print("[health]", r.json())
    except Exception as e:
        raise SystemExit(f"API not healthy at {API} → {e}")


ALIAS_QUERY      = ["query", "question", "prompt", "Query", "Question", "Prompt"]
ALIAS_Q_LEN      = ["query_length", "Query_Length", "length", "Length"]
ALIAS_REFERENCE  = ["reference", "gold", "target", "ground_truth",
                    "Reference", "Gold", "Target", "Ground_Truth"]

def _find_col(header: List[str], candidates: List[str]) -> Optional[str]:
    hmap = { (h or "").strip().lower(): h for h in header if h is not None }
    for c in candidates:
        key = (c or "").strip().lower()
        if key in hmap:
            return hmap[key]
    return None

def _compact_sources(sources: Optional[List[Dict]]) -> str:

    if not sources:
        return ""
    parts = []
    for s in sources:
        if not isinstance(s, dict):
            parts.append(str(s)); continue
        sid = s.get("id") or s.get("doc_id") or s.get("source_id") or ""
        title = s.get("title") or s.get("name") or s.get("filename") or ""
        page = s.get("page") or s.get("page_number") or ""
        if page != "":
            parts.append(f"{title or sid}#p{page}")
        else:
            parts.append(title or sid or json.dumps(s, ensure_ascii=False))
    return " | ".join([p for p in parts if p])

def post_query(client: httpx.Client, q: str) -> Dict:
    payload = {
        "query": q,
        "k": K,
        "temperature": 0.0, 
    }
    for attempt in range(3):
        try:
            r = client.post(f"{API}/query", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.5)
    return {}

def main():
    _ensure_dirs()

    
    try:
        with open(IN_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            src_rows = list(reader)
            header = reader.fieldnames or []
    except Exception as e:
        raise SystemExit(f"[error] Failed to read {IN_CSV}: {e}")

    total = len(src_rows)
    if total == 0:
        raise SystemExit(f"[error] No rows found in {IN_CSV}")

    # Resolve input columns (query, length, reference – optional)
    col_query     = _find_col(header, ALIAS_QUERY) or "Query"
    col_q_len     = _find_col(header, ALIAS_Q_LEN) or "Query_Length"
    col_reference = _find_col(header, ALIAS_REFERENCE)  # may be None

    _flush_print("=== run_perf: configuration ===")
    _flush_print(f"API_URL:        {API}")
    _flush_print(f"IN_CSV:         {IN_CSV}  (rows: {total})")
    _flush_print(f"OUT_CSV:        {OUT_CSV}")
    _flush_print(f"MODEL_OUT_CSV:  {MODEL_OUT_CSV}")
    _flush_print(f"TOPK:           {K}")
    _flush_print(f"VERBOSE:        {VERBOSE}")
    _flush_print("===============================\n")

    with httpx.Client() as client, \
         open(OUT_CSV, "w", newline="", encoding="utf-8") as perf_out, \
         open(MODEL_OUT_CSV, "w", newline="", encoding="utf-8") as model_out:

        ping_health(client)

        
        perf_writer = csv.writer(perf_out)
        perf_writer.writerow([
            "Query", "Query_Length",
            "Retrieval_Time_MS", "Generation_Time_MS", "Total_Time_MS"
        ])

        model_writer = csv.writer(model_out)
        model_writer.writerow(["Query", "Answer", "Reference", "Sources_Compact", "Sources_JSON"])

        
        try:
            client.post(f"{API}/query", json={"query": "warmup", "k": min(3, K), "temperature": 0.0}, timeout=120)
        except Exception:
            pass

        done = 0
        t_start_all = time.perf_counter()

        for i, row in enumerate(src_rows, start=1):
            q = (row.get(col_query) or "").strip()
            if not q:
                continue

            reference = (row.get(col_reference) or "").strip() if col_reference else ""
            q_len = (row.get(col_q_len) or "").strip()

            if VERBOSE:
                _flush_print(f"[{i}/{total}] Q: {q[:80]}{'…' if len(q) > 80 else ''}")

            t0 = time.perf_counter()
            resp = post_query(client, q)
            t1 = time.perf_counter()

            timings = resp.get("timings_ms", {}) or {}
            answer  = (resp.get("answer") or "").strip()
            sources = resp.get("sources") or resp.get("documents") or resp.get("refs") or []

            
            perf_writer.writerow([
                q,
                q_len,
                timings.get("retrieval", ""),
                timings.get("generation", ""),
                round((t1 - t0) * 1000, 2),
            ])

            
            model_writer.writerow([
                q,
                answer,
                reference,
                _compact_sources(sources),
                json.dumps(sources, ensure_ascii=False),
            ])

            done += 1
            if done % LOG_EVERY == 0:
                elapsed = time.perf_counter() - t_start_all
                _flush_print(f"[{done}/{total}] processed in {elapsed:.1f}s")

        total_elapsed = time.perf_counter() - t_start_all

    _flush_print(f" Wrote perf results   → {OUT_CSV}")
    _flush_print(f" Wrote model outputs  → {MODEL_OUT_CSV}")
    _flush_print(f"  Total time: {total_elapsed:.2f}s for {done} queries")

if __name__ == "__main__":
    main()
