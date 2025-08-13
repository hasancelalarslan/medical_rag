#!/usr/bin/env python3
"""
run_manual_eval.py — Aggregate manual evaluation scores for Medical RAG

Inputs:
- CSV (default: eval/manual_eval.csv) with:
  query, answer, reference, relevance_1to5, accuracy_1to5, fluency_1to5, source_present_yes_no

Outputs:
- manual_scores_clean.csv   (normalized per-row)
- manual_summary.json       (aggregates)
- manual_summary.md         (Markdown report)
- manual_lowest10.csv       (lowest-scoring items for review)

Usage:
python scripts/run_manual_eval.py --input eval/manual_eval.csv --outdir eval
"""

import argparse, json, os
from pathlib import Path
import pandas as pd
import numpy as np

def _bool_from_str(x):
    s = str(x).strip().lower()
    return 1 if s in {"yes","y","true","1","evet","var","present"} else 0

def _coerce_1to5(s):
    try:
        v = float(s)
    except Exception:
        return np.nan
    return float(max(1.0, min(5.0, v))) if not np.isnan(v) else np.nan

def load_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize column names
    aliases = {
        "relevance_1to5": ["relevance","relevancy"],
        "accuracy_1to5": ["accuracy","correctness"],
        "fluency_1to5": ["fluency","readability"],
        "source_present_yes_no": ["source","has_source","citation_present"]
    }
    for canon, cands in aliases.items():
        if canon not in df.columns:
            for c in cands:
                if c in df.columns:
                    df = df.rename(columns={c: canon})
                    break

    # Coerce types
    df["relevance_1to5"] = df["relevance_1to5"].apply(_coerce_1to5)
    df["accuracy_1to5"]  = df["accuracy_1to5"].apply(_coerce_1to5)
    df["fluency_1to5"]   = df["fluency_1to5"].apply(_coerce_1to5)
    df["source_present_yes_no"] = df["source_present_yes_no"].apply(_bool_from_str)

    # Drop empty rows
    mask = df[["relevance_1to5","accuracy_1to5","fluency_1to5"]].notna().any(axis=1)
    df = df[mask].reset_index(drop=True)

    # ✅ Compute overall score here so it's always present
    w_rel, w_acc, w_flu = 0.4, 0.4, 0.2
    df["overall_score_1to5"] = (
        df["relevance_1to5"].fillna(0)*w_rel +
        df["accuracy_1to5"].fillna(0)*w_acc +
        df["fluency_1to5"].fillna(0)*w_flu
    ) / (w_rel + w_acc + w_flu)

    return df

def summarize(df: pd.DataFrame) -> dict:
    def stat(s):
        s = s.dropna()
        return {
            "count": int(s.size),
            "mean": float(s.mean()) if s.size else None,
            "median": float(s.median()) if s.size else None,
            "std": float(s.std(ddof=0)) if s.size else None,
            "min": float(s.min()) if s.size else None,
            "max": float(s.max()) if s.size else None,
        }

    # Weighted overall score
    w_rel, w_acc, w_flu = 0.4, 0.4, 0.2
    df["overall_score_1to5"] = (
        df["relevance_1to5"].fillna(0)*w_rel +
        df["accuracy_1to5"].fillna(0)*w_acc +
        df["fluency_1to5"].fillna(0)*w_flu
    ) / (w_rel + w_acc + w_flu)

    summary = {
        "n_rows": int(len(df)),
        "scores": {
            "relevance_1to5": stat(df["relevance_1to5"]),
            "accuracy_1to5":  stat(df["accuracy_1to5"]),
            "fluency_1to5":   stat(df["fluency_1to5"]),
            "overall_score_1to5": stat(df["overall_score_1to5"]),
        },
        "source_present_rate": float(df["source_present_yes_no"].mean()) if len(df)>0 else None,
        "lowest_10_preview": df.nsmallest(10, "overall_score_1to5")[["query","answer","reference","relevance_1to5","accuracy_1to5","fluency_1to5","overall_score_1to5"]].to_dict(orient="records")
    }
    return summary

def write_md(summary: dict, out_md: str):
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Manual Evaluation Summary\n\n")
        f.write(f"**Items evaluated:** {summary['n_rows']}\n\n")
        f.write("## Averages (1–5)\n\n")
        for k in ["relevance_1to5","accuracy_1to5","fluency_1to5","overall_score_1to5"]:
            s = summary["scores"][k]
            if s["mean"] is not None:
                f.write(f"- **{k}**: {s['mean']:.2f} (σ={s['std']:.2f}, median={s['median']:.2f})\n")
        if summary.get("source_present_rate") is not None:
            f.write(f"\n**Source present rate:** {summary['source_present_rate']*100:.1f}%\n\n")
        f.write("## Lowest 10 by Overall Score\n\n")
        if summary["lowest_10_preview"]:
            df_low = pd.DataFrame(summary["lowest_10_preview"])
            f.write(df_low.to_markdown(index=False))
        f.write("\n_Weights: relevance 0.4, accuracy 0.4, fluency 0.2._\n")

def main():
    ap = argparse.ArgumentParser(description="Aggregate manual evaluation scores")
    ap.add_argument("--input", default="eval/manual_eval.csv")
    ap.add_argument("--outdir", default="eval")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df = load_clean(args.input)

    # Save cleaned scores
    clean_csv = os.path.join(args.outdir, "manual_scores_clean.csv")
    df.to_csv(clean_csv, index=False)

    # Save lowest-10
    low10_csv = os.path.join(args.outdir, "manual_lowest10.csv")
    df.nsmallest(10, "overall_score_1to5").to_csv(low10_csv, index=False)

    # Summary outputs
    summary = summarize(df)
    with open(os.path.join(args.outdir, "manual_summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    write_md(summary, os.path.join(args.outdir, "manual_summary.md"))

    print(f"[ok] Wrote {clean_csv}")
    print(f"[ok] Wrote {low10_csv}")
    print(f"[ok] Wrote manual_summary.json and manual_summary.md")

if __name__ == "__main__":
    main()
