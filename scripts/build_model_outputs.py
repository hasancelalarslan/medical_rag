
import os, csv, time, httpx, argparse, json
from pathlib import Path
import pandas as pd

"""
 eval/model_outputs.csv from queries by the RAG API

Input:
from_perf eval/perf_results.csv 
from_queries examples/queries_100.csv 

Env:
  API_URL (default: http://127.0.0.1:8000)

Output:
  eval/model_outputs.csv with columns: query,answer,reference
"""

DEF_API = os.getenv("API_URL", "http://127.0.0.1:8000")
OUT_CSV = "eval/model_outputs.csv"

def call_api(client, api, q: str):
    
    req = {"query": q, "top_k": 5}
    r = client.post(f"{api}/query", json=req, timeout=60)
    r.raise_for_status()
    data = r.json()
    
    answer = data.get("answer") or data.get("output") or data.get("text") or ""
    
    ref = ""
    sources = data.get("sources") or data.get("contexts") or data.get("documents") or []
    if isinstance(sources, list) and sources:
        # try common shapes
        first = sources[0]
        if isinstance(first, dict):
            ref = first.get("snippet") or first.get("text") or first.get("content") or ""
        elif isinstance(first, str):
            ref = first
    return answer, ref

def load_queries(args):
    if args.from_perf:
        df = pd.read_csv(args.from_perf)
        
        for col in ["Query", "query", "question", "prompt"]:
            if col in df.columns:
                return df[col].astype(str).tolist()
        raise SystemExit(f"Couldn't find a query column in {args.from_perf}")
    elif args.from_queries:
        df = pd.read_csv(args.from_queries)
        col = "query" if "query" in df.columns else df.columns[0]
        return df[col].astype(str).tolist()
    else:
        raise SystemExit("Provide --from_perf or --from_queries")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_perf", default="eval/perf_results.csv")
    ap.add_argument("--from_queries", default=None)
    ap.add_argument("--api", default=DEF_API)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep_ms", type=int, default=0, help="sleep between calls")
    args = ap.parse_args()

    queries = load_queries(args)
    if args.limit:
        queries = queries[:args.limit]
    os.makedirs("eval", exist_ok=True)

    rows = []
    with httpx.Client() as client:
        for i, q in enumerate(queries, 1):
            try:
                ans, ref = call_api(client, args.api, q)
            except Exception as e:
                print(f"[warn] {i}/{len(queries)} failed → {e}")
                ans, ref = "", ""
            rows.append({"query": q, "answer": ans, "reference": ref})
            if args.sleep_ms:
                time.sleep(args.sleep_ms / 1000.0)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {OUT_CSV} with {len(rows)} rows.")

if __name__ == "__main__":
    main()
