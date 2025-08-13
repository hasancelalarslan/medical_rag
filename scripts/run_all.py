#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import nltk

def ensure_nltk():
    """Ensure NLTK corpora are present inside Docker container."""
    for pkg in ["punkt", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
        except LookupError:
            print(f"[info] Downloading NLTK resource: {pkg}")
            nltk.download(pkg, quiet=True)

def run_cmd(cmd_list, desc):
    print(f"\n$ {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list)
    if res.returncode != 0:
        print(f"[error] {desc} failed!")
        sys.exit(res.returncode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", default="http://127.0.0.1:8000")
    parser.add_argument("--perf_input", default="examples/queries_100.csv")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--perf_output", default="eval/perf_results.csv")
    parser.add_argument("--report_out", default="eval/report")
    parser.add_argument("--quality_input", default="eval/model_outputs.csv")
    parser.add_argument("--quality_out", default="eval")
    parser.add_argument("--bert_model", default="roberta-large")
    parser.add_argument("--ppl_model", default="gpt2")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    ensure_nltk()

    # Performance test
    run_cmd([
        sys.executable, "scripts/run_perf.py",
        ], "Performance run")

    
    run_cmd([
        sys.executable, "scripts/run_eval.py",
        "--csv", args.perf_output,
        "--out", args.report_out,
        "--title", "Medical RAG Perf — Baykar Submission"
    ], "Evaluation report")

    #  Auto-create model_outputs.csv if missing
    if not os.path.exists(args.quality_input):
        import pandas as pd
        if os.path.exists(args.perf_output):
            df = pd.read_csv(args.perf_output)
            if {"Query", "Answer", "Reference"}.issubset(df.columns):
                new_df = df.rename(columns={
                    "Query": "query",
                    "Answer": "answer",
                    "Reference": "reference"
                })
                new_df.to_csv(args.quality_input, index=False)
                print(f"[info] Created {args.quality_input} from {args.perf_output}")
            else:
                print("[warn] perf_results.csv found but missing required columns to build model_outputs.csv")

    # Quality metrics
    run_cmd([
        sys.executable, "scripts/run_quality.py",
        "--input", args.quality_input,
        "--outdir", args.quality_out,
        "--bert_model", args.bert_model,
        "--ppl_model", args.ppl_model,
        "--lang", args.lang,
        "--batch_size", str(args.batch_size)
    ], "Quality metrics")

    print("\n ALL DONE!")
    print(f" Perf CSV: {args.perf_output}")
    print(f" Report: {os.path.join(args.report_out, 'perf_report.md')}")
    print(f" Quality CSV: {os.path.join(args.quality_out, 'quality_scores.csv')}")
    print(f" Quality report: {os.path.join(args.quality_out, 'quality_report.md')}")

if __name__ == "__main__":
    main()
