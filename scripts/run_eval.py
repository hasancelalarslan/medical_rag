#!/usr/bin/env python3
from __future__ import annotations
"""
run_eval.py — Evaluate and visualize performance results for Medical RAG.

Usage:
------
python run_eval.py
python run_eval.py --csv path/to/results.csv --out eval/report \
    --title "Medical RAG Perf — Baykar Submission" --topk 20
"""

import argparse
import os
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = BASE_DIR / "eval" / "perf_results.csv"
DEFAULT_OUT = BASE_DIR / "eval" / "report"

PERCENTILES = [50, 75, 90, 95, 99, 99.9]
STAGE_COLS_CANON = [
    ("retrieval_time_ms", ["retrieval_time_ms", "retrieval_ms", "retrieval", "retrieval_time"]),
    ("generation_time_ms", ["generation_time_ms", "generation_ms", "generation", "llm_time", "decode_ms"]),
    ("total_time_ms", ["total_time_ms", "total_ms", "latency_ms", "latency", "total_time"]),
]
OPTIONAL_COLS = {
    "query": ["query", "question", "prompt"],
    "query_length": ["query_length", "q_len", "length", "kategori", "category"],
    "status": ["status", "ok", "result"],
    "model": ["model", "model_name", "llm", "engine"],
    "timestamp": ["timestamp", "time", "created_at"],
}
BUCKET_MAP = {
    "kisa": "Short", "orta": "Medium", "uzun": "Long",
    "short": "Short", "medium": "Medium", "long": "Long",
}

@dataclass
class EvalConfig:
    csv: str
    compare: Optional[str]
    out: str
    title: str
    topk: int

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})
    for canon, aliases in STAGE_COLS_CANON:
        found = _find_col(df, aliases)
        if found and found != canon:
            df = df.rename(columns={found: canon})
    for canon, aliases in OPTIONAL_COLS.items():
        found = _find_col(df, aliases)
        if found and found != canon:
            df = df.rename(columns={found: canon})
    for col in ["retrieval_time_ms", "generation_time_ms", "total_time_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "query_length" in df.columns:
        df["query_length"] = (
            df["query_length"].astype(str).str.strip().str.lower().map(BUCKET_MAP).fillna(df["query_length"])
        )
    return df

def _numeric_quality_cols(df: pd.DataFrame) -> List[str]:
    skip = {"retrieval_time_ms", "generation_time_ms", "total_time_ms"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

def _percentiles(s: pd.Series, percentiles=PERCENTILES) -> Dict[str, float]:
    return {f"p{p}": float(np.percentile(s.dropna(), p)) if s.notna().any() else float("nan") for p in percentiles}

def summarize_latency(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary = {}
    for col in ["retrieval_time_ms", "generation_time_ms", "total_time_ms"]:
        if col in df.columns:
            s = df[col]
            summary[col] = {
                "count": int(s.notna().sum()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),
                "min": float(s.min()),
                "max": float(s.max()),
                **_percentiles(s),
            }
    return summary

def summarize_by_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if "query_length" not in df.columns:
        return pd.DataFrame()
    gb = df.groupby("query_length")
    rows = []
    for bucket, g in gb:
        row = {"bucket": bucket}
        for col in ["retrieval_time_ms", "generation_time_ms", "total_time_ms"]:
            if col in g.columns:
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_p95"] = np.percentile(g[col].dropna(), 95)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("bucket")

def top_n_slowest(df: pd.DataFrame, n: int) -> pd.DataFrame:
    cols = [c for c in ["query", "query_length", "retrieval_time_ms", "generation_time_ms", "total_time_ms"] if c in df.columns]
    return df[cols].sort_values("total_time_ms", ascending=False).head(n) if "total_time_ms" in df.columns else pd.DataFrame(columns=cols)

def _ensure_outdirs(out_dir: str) -> Tuple[str, str]:
    figs_dir = os.path.join(out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    return out_dir, figs_dir

def plot_histograms(df: pd.DataFrame, figs_dir: str) -> Dict[str, str]:
    paths = {}
    for col in ["retrieval_time_ms", "generation_time_ms", "total_time_ms"]:
        if col in df.columns:
            plt.figure()
            df[col].dropna().plot(kind="hist", bins=30, alpha=0.8)
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.title(f"Histogram — {col}")
            p = os.path.join(figs_dir, f"hist_{col}.png")
            plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
            paths[col] = p
    return paths

def plot_cdf(df: pd.DataFrame, figs_dir: str) -> Dict[str, str]:
    paths = {}
    for col in ["retrieval_time_ms", "generation_time_ms", "total_time_ms"]:
        if col in df.columns:
            x = np.sort(df[col].dropna().values)
            y = np.arange(1, len(x) + 1) / len(x)
            plt.figure()
            plt.plot(x, y)
            plt.xlabel(col)
            plt.ylabel("CDF")
            plt.title(f"CDF — {col}")
            p = os.path.join(figs_dir, f"cdf_{col}.png")
            plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
            paths[col] = p
    return paths

def _compare_runs(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    key = "query" if "query" in df_a.columns and "query" in df_b.columns else "_idx"
    if key == "_idx":
        df_a = df_a.copy(); df_b = df_b.copy()
        df_a["_idx"] = range(len(df_a))
        df_b["_idx"] = range(len(df_b))
    cols = ["retrieval_time_ms", "generation_time_ms", "total_time_ms"]
    a = df_a[[key] + [c for c in cols if c in df_a.columns]].rename(columns={c: f"{c}_A" for c in cols if c in df_a.columns})
    b = df_b[[key] + [c for c in cols if c in df_b.columns]].rename(columns={c: f"{c}_B" for c in cols if c in df_b.columns})
    j = pd.merge(a, b, on=key, how="inner")
    for c in cols:
        if f"{c}_A" in j.columns and f"{c}_B" in j.columns:
            j[f"delta_{c}"] = j[f"{c}_B"] - j[f"{c}_A"]
            j[f"delta_{c}_pct"] = (j[f"delta_{c}"] / j[f"{c}_A"].replace(0, np.nan)) * 100
    if key == "_idx":
        j = j.drop(columns=["_idx"])
    return j

def make_markdown_report(cfg: EvalConfig, df: pd.DataFrame, out_dir: str, figs: Dict[str, str], compare_df: Optional[pd.DataFrame] = None) -> str:
    md_path = os.path.join(out_dir, "perf_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {cfg.title}\n\n")
        f.write(f"**Input CSV:** `{os.path.abspath(cfg.csv)}`\n\n")
        if compare_df is not None:
            f.write(f"**Compare CSV:** `{os.path.abspath(cfg.compare)}`\n\n")
        f.write("## Overall Latency Summary\n\n")
        summ = summarize_latency(df)
        f.write(pd.DataFrame(summ).T.to_markdown() + "\n\n")
        byb = summarize_by_bucket(df)
        if not byb.empty:
            f.write("## Latency by Query Length\n\n")
            f.write(byb.to_markdown(index=False) + "\n\n")
        topn = top_n_slowest(df, cfg.topk)
        if not topn.empty:
            f.write(f"## Top {cfg.topk} Slowest Queries\n\n")
            f.write(topn.to_markdown(index=False) + "\n\n")
        if compare_df is not None:
            f.write("## Comparison vs Baseline\n\n")
            f.write(_compare_runs(df, compare_df).to_markdown(index=False) + "\n\n")
    return md_path

def load_csv_normalized(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    if "total_time_ms" in df.columns:
        df = df[df["total_time_ms"].notna()]
    return df.reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Medical RAG perf CSV and generate a report.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--compare", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--title", default="Medical RAG Perf — Baykar Submission")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    cfg = EvalConfig(csv=args.csv, compare=args.compare, out=args.out, title=args.title, topk=args.topk)

    os.makedirs(cfg.out, exist_ok=True)
    out_dir, figs_dir = _ensure_outdirs(cfg.out)

    df = load_csv_normalized(cfg.csv)
    if df.empty:
        raise SystemExit("No valid rows found in CSV.")

    compare_df = None
    if cfg.compare:
        compare_df = load_csv_normalized(cfg.compare)
        if compare_df.empty:
            compare_df = None

    # Figures
    figs = {}
    figs.update({f"Histogram {k}": v for k, v in plot_histograms(df, figs_dir).items()})
    figs.update({f"CDF {k}": v for k, v in plot_cdf(df, figs_dir).items()})

    # Summary JSON
    summary_data = {
        "title": cfg.title,
        "rows": int(len(df)),
        "latency": summarize_latency(df),
        "by_bucket": summarize_by_bucket(df).to_dict(orient="records"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary_data, fp, ensure_ascii=False, indent=2)

    # Markdown report
    md_path = make_markdown_report(cfg, df, out_dir, figs, compare_df=compare_df)

    print(f"✅ Report written to: {md_path}")
    print(f"📊 Figures under: {figs_dir}")
    print(f"📄 Summary JSON: {os.path.join(out_dir, 'summary.json')}")

if __name__ == "__main__":
    main()
