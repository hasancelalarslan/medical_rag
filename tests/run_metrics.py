#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import csv
import json
import math
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import statistics
import sys
from typing import Dict, List, Tuple, Optional


def _flag(name: str, ok: bool, reason: str = ""):
    """Emit a short diagnostic line to stderr for optional components."""
    print(f"[metrics:{name}] ok={ok}{(' — ' + reason) if reason else ''}", file=sys.stderr)


try:
    import sacrebleu  
    _HAS_SACREBLEU = True
    _flag("sacrebleu", True)
except Exception as e:
    _HAS_SACREBLEU = False
    _flag("sacrebleu", False, f"import failed: {e!r}")

try:
    from rouge_score import rouge_scorer
    _HAS_ROUGE = True
    _flag("rouge-score", True)
except Exception as e:
    _HAS_ROUGE = False
    _flag("rouge-score", False, f"import failed: {e!r}")

try:
    import nltk
    _HAS_METEOR_BASE = True
    _flag("nltk", True)
except Exception as e:
    nltk = None
    _HAS_METEOR_BASE = False
    _flag("nltk", False, f"import failed: {e!r}")

try:
    from bert_score import score as bert_score
    _HAS_BERTSCORE = True
    _flag("bert-score", True)
except Exception as e:
    bert_score = None
    _HAS_BERTSCORE = False
    _flag("bert-score", False, f"import failed: {e!r}")

DEFAULT_BERTSCORE_MODEL = os.getenv("BERT_SCORE_MODEL_TYPE", "roberta-base")


def _norm_dec_str(x) -> Optional[str]:
    """Normalize decimal strings: '469,3' -> '469.3' and strip spaces."""
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if ',' in s and '.' not in s:
        s = s.replace(',', '.')
    s = s.replace(' ', '')
    return s

def to_float_list(series: List) -> List[float]:
    vals = []
    for v in series or []:
        s = _norm_dec_str(v)
        if s is None:
            continue
        try:
            vals.append(float(s))
        except Exception:
            continue
    return vals

def ptiles(xs: List[float], qs: List[float]) -> Dict[float, float]:
    if not xs:
        return {q: float('nan') for q in qs}
    xs_sorted = sorted(xs)
    out = {}
    for q in qs:
        k = max(0, min(len(xs_sorted)-1, int(round(q * (len(xs_sorted)-1)))))
        out[q] = xs_sorted[k]
    return out

def safe_mean(xs: List[float]) -> float:
    return float('nan') if not xs else (sum(xs) / len(xs))

def read_csv(path: str) -> Tuple[List[Dict], List[str]]:
    """Read CSV preserving headers; handle potential stray empty column."""
    with open(path, 'r', encoding='utf-8', newline='') as f:
        try:
            sniff_buf = f.read(4096)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sniff_buf)
        except Exception:
            f.seek(0)
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)
        headers = reader.fieldnames or []
    clean_rows = []
    for r in rows:
        if '' in r:
            r.pop('', None)
        clean_rows.append(r)
    headers = [h for h in headers if h and h.strip() != '']
    return clean_rows, headers

def get_col_name(headers: List[str], candidates: List[str]) -> Optional[str]:
    """Case-insensitive header resolution."""
    hset = {h.lower(): h for h in headers}
    for c in candidates:
        cl = c.lower()
        if cl in hset:
            return hset[cl]
    return None

def tokenize_words(s: str) -> List[str]:
    if not s:
        return []
    return [w for w in s.strip().split() if w]

def compute_latency(rows: List[Dict], headers: List[str]) -> Dict:
    ret_h = get_col_name(headers, ["Retrieval_Time_MS", "retrieval_time_ms", "retrieval_ms"])
    gen_h = get_col_name(headers, ["Generation_Time_MS", "generation_time_ms", "generation_ms"])
    tot_h = get_col_name(headers, ["Total_Time_MS", "total_time_ms", "total_ms"])

    ret = to_float_list([r.get(ret_h) for r in rows]) if ret_h else []
    gen = to_float_list([r.get(gen_h) for r in rows]) if gen_h else []
    tot = to_float_list([r.get(tot_h) for r in rows]) if tot_h else []

    def pack(xs: List[float]) -> Dict:
        if not xs:
            return {"count": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
        pts = ptiles(xs, [0.50, 0.95])
        return {"count": len(xs), "mean": safe_mean(xs), "p50": pts[0.50], "p95": pts[0.95]}

    return {
        "retrieval_ms": pack(ret),
        "generation_ms": pack(gen),
        "total_ms": pack(tot),
    }

def compute_perplexity_proxy(ans_list: List[str]) -> float:
    """Unigram self-perplexity proxy with add-one smoothing (2^H)."""
    tokens = []
    for a in ans_list:
        tokens.extend(tokenize_words(a))
    V = len(set(tokens)) or 1
    N = len(tokens)
    if N == 0:
        return float('nan')
    from collections import Counter
    cnt = Counter(tokens)
    denom = N + V
    H = 0.0
    for w in tokens:
        p = (cnt[w] + 1.0) / denom
        H += -math.log2(p)
    H /= N
    return 2 ** H

def compute_bleu(references: List[str], hypotheses: List[str]) -> Optional[float]:
    if not _HAS_SACREBLEU:
        return None
    try:
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        return float(bleu.score)
    except Exception as e:
        _flag("BLEU", False, f"compute failed: {e!r}")
        return None

def compute_rouge(references: List[str], hypotheses: List[str]) -> Optional[Dict[str, float]]:
    if not _HAS_ROUGE:
        return None
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rL = [], [], []
        for ref, hyp in zip(references, hypotheses):
            scores = scorer.score(ref or "", hyp or "")
            r1.append(scores["rouge1"].fmeasure)
            r2.append(scores["rouge2"].fmeasure)
            rL.append(scores["rougeL"].fmeasure)
        def avg(xs): return float('nan') if not xs else sum(xs)/len(xs)
        return {"ROUGE-1_F1": avg(r1), "ROUGE-2_F1": avg(r2), "ROUGE-L_F1": avg(rL)}
    except Exception as e:
        _flag("ROUGE", False, f"compute failed: {e!r}")
        return None

def _ensure_nltk_for_meteor() -> bool:
    """Ensure required NLTK data is present. Return True if ready."""
    if not _HAS_METEOR_BASE:
        return False
    
    try:
        from nltk.corpus import wordnet as wn  # noqa
        _ = wn.synsets("test")
        return True
    except Exception:
        pass

    
    try:
        import pathlib
        dl_dir = pathlib.Path(os.environ.get("NLTK_DATA", "/home/appuser/nltk_data"))
        dl_dir.mkdir(parents=True, exist_ok=True)
        if str(dl_dir) not in nltk.data.path:
            nltk.data.path.append(str(dl_dir))
        for pkg in ("punkt", "wordnet", "omw-1.4"):
            try:
                nltk.download(pkg, download_dir=str(dl_dir), quiet=True)
            except Exception:
                
                pass
        from nltk.corpus import wordnet as wn  # re-check
        _ = wn.synsets("test")
        _flag("nltk-data", True, f"downloaded to {dl_dir}")
        return True
    except Exception as e:
        _flag("nltk-data", False, f"setup failed: {e!r}")
        return False

def compute_meteor(references: List[str], hypotheses: List[str]) -> Optional[float]:
    
    if not _HAS_METEOR_BASE:
        return None
    if not _ensure_nltk_for_meteor():
        return None
    try:
        
        from nltk.translate.meteor_score import single_meteor_score as nltk_meteor
    except Exception as e:
        _flag("METEOR", False, f"import failed post-setup: {e!r}")
        return None

    scores = []
    for ref, hyp in zip(references, hypotheses):
        try:
            scores.append(nltk_meteor(ref or "", hyp or ""))
        except Exception:
            continue
    return float('nan') if not scores else float(sum(scores) / len(scores))

def compute_bertscore(references: List[str], hypotheses: List[str], model_type: str = "roberta-base") -> Optional[Dict[str, float]]:
    if not _HAS_BERTSCORE:
        _flag("BERTScore", False, "bert-score not installed")
        return None
    try:
        P, R, F1 = bert_score(
            hypotheses, references,
            lang="en",
            model_type=model_type or "roberta-base",
            rescale_with_baseline=True
        )
        return {"BERTScore_P": float(P.mean()), "BERTScore_R": float(R.mean()), "BERTScore_F1": float(F1.mean())}
    except Exception as e:
        _flag("BERTScore", False, f"compute failed: {e!r}")
        return None

def summarize_manual(df_rows: List[Dict], headers: List[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for col in ["relevance_1to5", "accuracy_1to5", "fluency_1to5", "overall_score_1to5"]:
        h = get_col_name(headers, [col])
        if not h:
            continue
        vals = to_float_list([r.get(h) for r in df_rows])
        if not vals:
            continue
        out[col] = {
            "count": len(vals),
            "mean": sum(vals)/len(vals),
            "median": statistics.median(vals),
            "std": (statistics.pstdev(vals) if len(vals) > 1 else 0.0),
            "min": min(vals),
            "max": max(vals),
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default=os.getenv("PERF_INPUT", "results/perf_results.csv"),
                    help="Input perf CSV (default: results/perf_results.csv)")
    ap.add_argument("--outdir", dest="outdir", default=os.getenv("OUT_DIR", "results"),
                    help="Output directory for metrics_report.* (default: results)")
    args = ap.parse_args()

    in_csv = args.in_csv
    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    rows, headers = read_csv(in_csv)
    count_rows = len(rows)

    ans_h = get_col_name(headers, ["Answer", "answer"])
    ref_h = get_col_name(headers, ["Reference", "reference", "gold", "target"])

    answers = [r.get(ans_h, "") for r in rows] if ans_h else []
    references = [r.get(ref_h, "") for r in rows] if ref_h else []

    latency = compute_latency(rows, headers)
    ppl_proxy = compute_perplexity_proxy(answers) if answers else float('nan')

    text_metrics: Dict[str, float] = {}
    has_refs = bool(references) and any(bool(x) for x in references)

    if has_refs and answers:
        bleu = compute_bleu(references, answers)
        if bleu is not None:
            text_metrics["BLEU"] = bleu

        rouge = compute_rouge(references, answers)
        if rouge is not None:
            text_metrics.update(rouge)

        meteor = compute_meteor(references, answers)
        if meteor is not None:
            text_metrics["METEOR"] = meteor

        bscore = compute_bertscore(references, answers, model_type=DEFAULT_BERTSCORE_MODEL)
        if bscore is not None:
            text_metrics.update(bscore)

    manual = summarize_manual(rows, headers)

    def fmt(x):
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return ""
            return f"{x:.4f}"
        return x

    csv_path = os.path.join(out_dir, "metrics_report.csv")
    md_path = os.path.join(out_dir, "metrics_report.md")

    
    csv_rows = []
    for key, block in [
        ("Retrieval_Time_MS", latency["retrieval_ms"]),
        ("Generation_Time_MS", latency["generation_ms"]),
        ("Total_Time_MS", latency["total_ms"]),
    ]:
        csv_rows.append({
            "Metric": key, "Count": block["count"],
            "Mean": fmt(block["mean"]), "P50": fmt(block["p50"]), "P95": fmt(block["p95"]),
        })
    csv_rows.append({"Metric": "Perplexity_Proxy_Unigram", "Count": count_rows, "Mean": fmt(ppl_proxy), "P50": "", "P95": ""})
    for k, v in text_metrics.items():
        csv_rows.append({"Metric": k, "Count": count_rows, "Mean": fmt(v), "P50": "", "P95": ""})
    for k, d in manual.items():
        csv_rows.append({"Metric": k, "Count": d["count"], "Mean": fmt(d["mean"]), "P50": fmt(d["median"]), "P95": ""})

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Metric", "Count", "Mean", "P50", "P95"])
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    def md_line(label, block):
        return f"- **{label}** — count: {block['count']}, mean: {fmt(block['mean'])} ms, p50: {fmt(block['p50'])} ms, p95: {fmt(block['p95'])} ms"

    md_lines = []
    md_lines.append("# Metrics Report\n")
    md_lines.append(f"- **Samples (rows)**: {count_rows}\n")
    md_lines.append("## Latency (ms)\n")
    md_lines.append(md_line("Retrieval", latency["retrieval_ms"]))
    md_lines.append(md_line("Generation", latency["generation_ms"]))
    md_lines.append(md_line("Total", latency["total_ms"]))
    md_lines.append("")
    md_lines.append("## Text Metrics\n")
    md_lines.append(f"- **Perplexity (unigram proxy)**: {fmt(ppl_proxy)}")
    if has_refs:
        if "BLEU" in text_metrics:        md_lines.append(f"- **BLEU**: {fmt(text_metrics['BLEU'])}")
        if "ROUGE-1_F1" in text_metrics:  md_lines.append(f"- **ROUGE-1 F1**: {fmt(text_metrics['ROUGE-1_F1'])}")
        if "ROUGE-2_F1" in text_metrics:  md_lines.append(f"- **ROUGE-2 F1**: {fmt(text_metrics['ROUGE-2_F1'])}")
        if "ROUGE-L_F1" in text_metrics:  md_lines.append(f"- **ROUGE-L F1**: {fmt(text_metrics['ROUGE-L_F1'])}")
        if "METEOR" in text_metrics:      md_lines.append(f"- **METEOR**: {fmt(text_metrics['METEOR'])}")
        if "BERTScore_P" in text_metrics: md_lines.append(f"- **BERTScore** (P/R/F1): {fmt(text_metrics['BERTScore_P'])} / {fmt(text_metrics['BERTScore_R'])} / {fmt(text_metrics['BERTScore_F1'])}")
    else:
        md_lines.append("- No `reference` column found; BLEU/ROUGE/METEOR/BERTScore skipped.")
    md_lines.append("")
    if manual:
        md_lines.append("## Manual Metrics (if provided)\n")
        for k, d in manual.items():
            md_lines.append(f"- **{k}** — count: {d['count']}, mean: {fmt(d['mean'])}, median: {fmt(d['median'])}, std: {fmt(d['std'])}, min: {fmt(d['min'])}, max: {fmt(d['max'])}")
        md_lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("[run_metrics] Wrote:")
    print("  -", csv_path)
    print("  -", md_path)
    # Echo which libs were active (useful in CI logs)
    print("[run_metrics] deps:", {
        "sacrebleu": _HAS_SACREBLEU,
        "rouge-score": _HAS_ROUGE,
        "meteor(nltk)": _HAS_METEOR_BASE,
        "bertscore": _HAS_BERTSCORE,
        "bertscore_model": DEFAULT_BERTSCORE_MODEL,
    })

if __name__ == "__main__":
    main()
