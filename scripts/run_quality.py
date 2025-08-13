#!/usr/bin/env python3
from __future__ import annotations
"""
run_quality.py 

Inputs
 eval/model_outputs.csv with columns: query, answer, reference


Outputs (under --outdir, default eval/)
quality_scores.csv       (per-example)
 quality_report.md        (corpus summary)
-quality_summary.json     (machine-readable)

Usage
pip install pandas numpy tqdm nltk bert-score rouge-score torch transformers regex

python scripts/run_quality.py \
  --input eval/model_outputs.csv \
  --outdir eval \
  --bert_model roberta-large \
  --ppl_model gpt2 \
  --lang en \
  --batch_size 16
"""
import argparse, json, math, os, sys
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


from rouge_score import rouge_scorer


try:
    from bert_score import score as bert_score
    _HAS_BERTSCORE = True
except Exception:
    _HAS_BERTSCORE = False


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import regex as re

@dataclass
class ExampleScores:
    idx: int
    query: str
    answer: str
    reference: str
    bleu1: float
    bleu2: float
    bleu3: float
    bleu4: float
    rougeL: float
    meteor: float
    bert_f1: float | None
    perplexity: float | None

def _ensure_nltk():
    for pkg in ['punkt', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'tokenizers/{pkg}' if pkg=='punkt' else f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)

_WS_RE = re.compile(r"\s+")
def simple_tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.strip().lower()
    if not text:
        return []
    return _WS_RE.sub(" ", text).split(" ")

def compute_bleu(references_tok, hypotheses_tok):
    smoothie = SmoothingFunction().method3
    weights_list = [
        (1.0, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (1/3, 1/3, 1/3, 0),
        (0.25, 0.25, 0.25, 0.25),
    ]
    out = []
    for w in weights_list:
        try:
            s = corpus_bleu(references_tok, hypotheses_tok, weights=w, smoothing_function=smoothie)
        except ZeroDivisionError:
            s = 0.0
        out.append(float(s))
    return out

def compute_rougeL(refs: List[str], hyps: List[str]) -> List[float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = []
    for ref, hyp in zip(refs, hyps):
        try:
            sc = scorer.score(ref or "", hyp or "")
            vals.append(float(sc["rougeL"].fmeasure))
        except Exception:
            vals.append(0.0)
    return vals

def compute_meteor(refs: List[str], hyps: List[str]) -> List[float]:
    vals = []
    for ref, hyp in zip(refs, hyps):
        try:
            vals.append(float(meteor_score([ref or ""], hyp or "")))
        except Exception:
            vals.append(0.0)
    return vals

def compute_bertscore(refs: List[str], hyps: List[str], model: str, batch_size: int, lang: str) -> List[float]:
    if not _HAS_BERTSCORE:
        return [None] * len(hyps)
    try:
        P, R, F1 = bert_score(hyps, refs, model_type=model, lang=lang, verbose=False, batch_size=batch_size)
        return [float(f) for f in F1.tolist()]
    except Exception:
        return [None] * len(hyps)

@torch.no_grad()
def compute_perplexities(texts: List[str], model_name: str, max_length: int = 1024) -> List[float | None]:
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"[warn] Perplexity disabled: failed to load {model_name}: {e}")
        return [None] * len(texts)

    vals: List[float | None] = []
    for t in tqdm(texts, desc="Perplexity"):
        try:
            if not isinstance(t, str) or not t.strip():
                vals.append(None); continue
            enc = tok(t, return_tensors='pt', truncation=True, max_length=max_length)
            input_ids = enc['input_ids'].to(device)
            attn = enc.get('attention_mask')
            attn = attn.to(device) if attn is not None else None
            outputs = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            ppl = float(math.exp(outputs.loss.item()))
            vals.append(ppl)
        except Exception:
            vals.append(None)
    return vals

def _write_template(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmpl = pd.DataFrame([
        {"query": "What is the first-line therapy for type 2 diabetes?",
         "answer": "Metformin is typically first-line unless contraindicated.",
         "reference": "Metformin is recommended as initial pharmacologic therapy."},
        {"query": "Define myocardial infarction.",
         "answer": "It is myocardial necrosis due to ischemia, diagnosed by troponin rise with symptoms/ECG.",
         "reference": "Myocardial infarction is myocardial cell death due to prolonged ischemia."}
    ])
    tmpl.to_csv(path, index=False)
    print(f"[info] Created template CSV at: {path}")

def _remap_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower(): c for c in df.columns}
    
    df.columns = [c.lower() for c in df.columns]
    
    alias_map = {
        "answer": ["answer", "prediction", "output", "response"],
        "reference": ["reference", "gold", "target", "ground_truth"],
        "query": ["query", "question", "prompt"]
    }
    for canon, cands in alias_map.items():
        if canon in df.columns:
            continue
        for cand in cands:
            if cand in df.columns:
                df = df.rename(columns={cand: canon})
                break
    required = {"query", "answer", "reference"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"[error] Missing required columns after remap: {missing}. Present: {list(df.columns)}")
    return df

def main():
    _ensure_nltk()

    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='eval/model_outputs.csv', help='CSV with columns: query, answer, reference (aliases supported)')
    ap.add_argument('--outdir', default='eval', help='Output directory')
    ap.add_argument('--bert_model', default='roberta-large', help='bert-score model_type (e.g., roberta-large, distilroberta-base, bert-base-multilingual-cased)')
    ap.add_argument('--lang', default='en', help='Language code for BERTScore (e.g., en, tr).')
    ap.add_argument('--ppl_model', default='gpt2', help='HF causal LM name for perplexity')
    ap.add_argument('--batch_size', type=int, default=16, help='Batch size for BERTScore')
    ap.add_argument('--create_template', action='store_true', default=True, help='If input is missing, create a template CSV and exit')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    
    if not os.path.exists(args.input):
        if args.create_template:
            _write_template(args.input)
            print("[hint] Fill the template then rerun the script.")
            sys.exit(0)
        else:
            print(f"[error] Failed to read {args.input}: file not found"); sys.exit(1)

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"[error] Failed to read {args.input}: {e}"); sys.exit(1)

    df = _remap_columns(df)
    
    df['query'] = df['query'].astype(str)
    df['answer'] = df['answer'].fillna("").astype(str)
    df['reference'] = df['reference'].fillna("").astype(str)

    refs = df['reference'].tolist()
    hyps = df['answer'].tolist()

    
    tokenized_refs = [[[tok for tok in simple_tokenize(r)]] for r in refs]
    tokenized_hyps = [[tok for tok in simple_tokenize(h)] for h in hyps]
    bleu1, bleu2, bleu3, bleu4 = compute_bleu(tokenized_refs, tokenized_hyps)

    
    smoothie = SmoothingFunction().method3
    sent_bleu4 = []
    for r, h in zip(tokenized_refs, tokenized_hyps):
        try:
            sent_bleu4.append(float(nltk.translate.bleu_score.sentence_bleu(r, h, weights=(0.25,)*4, smoothing_function=smoothie)))
        except ZeroDivisionError:
            sent_bleu4.append(0.0)

    rougeL_vals = compute_rougeL(refs, hyps)
    meteor_vals = compute_meteor(refs, hyps)
    bert_f1_vals = compute_bertscore(refs, hyps, model=args.bert_model, batch_size=args.batch_size, lang=args.lang)
    ppl_vals = compute_perplexities(hyps, model_name=args.ppl_model)

    rows = []
    for i, (q, a, r, rl, mt, bf1, ppl, sbleu4) in enumerate(zip(df['query'], hyps, refs, rougeL_vals, meteor_vals, bert_f1_vals, ppl_vals, sent_bleu4)):
        rows.append(ExampleScores(
            idx=i, query=q, answer=a, reference=r,
            bleu1=0.0, bleu2=0.0, bleu3=0.0, bleu4=float(sbleu4),
            rougeL=float(rl), meteor=float(mt),
            bert_f1=(None if bf1 is None else float(bf1)),
            perplexity=(None if ppl is None else float(ppl)),
        ))

    out_csv = os.path.join(args.outdir, 'quality_scores.csv')
    pd.DataFrame([asdict(r) for r in rows]).to_csv(out_csv, index=False)

    def _avg(vals):
        arr = np.array([v for v in vals if v is not None], dtype=float)
        return float(arr.mean()) if arr.size > 0 else None

    corpus = {
        'BLEU1': float(bleu1), 'BLEU2': float(bleu2), 'BLEU3': float(bleu3), 'BLEU4': float(bleu4),
        'ROUGE-L': _avg(rougeL_vals), 'METEOR': _avg(meteor_vals),
        'BERTScore_F1': _avg(bert_f1_vals), 'Perplexity': _avg(ppl_vals),
        'num_examples': int(len(df)), 'bert_model': args.bert_model if _HAS_BERTSCORE else None,
        'ppl_model': args.ppl_model, 'lang': args.lang,
    }

    out_json = os.path.join(args.outdir, 'quality_summary.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    out_md = os.path.join(args.outdir, 'quality_report.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('# Quality Report\n\n')
        f.write('| Metric | Value |\n|---|---:|\n')
        for k in ['BLEU1','BLEU2','BLEU3','BLEU4','ROUGE-L','METEOR','BERTScore_F1','Perplexity']:
            v = corpus[k]
            if v is None:
                f.write(f"| {k} | n/a |\n")
            else:
                f.write(f"| {k} | {v*100:.2f}% |\n" if k != 'Perplexity' else f"| {k} | {v:.2f} |\n")
        f.write('\n')
        f.write('**Notes**:\n')
        f.write('- BLEU above is corpus-level; per-row BLEU-4 is in quality_scores.csv.\n')
        f.write(f"- BERTScore model: {corpus['bert_model'] or 'n/a'} (lang={args.lang})\n")
        f.write(f"- Perplexity model: {args.ppl_model}\n")
        f.write('\nGenerated by scripts/run_quality.py\n')

    print(f"[ok] Wrote {out_csv}\n[ok] Wrote {out_md}\n[ok] Wrote {out_json}")

if __name__ == '__main__':
    main()
