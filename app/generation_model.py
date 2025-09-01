# app/generation_model.py

from typing import Any, List, Optional, Sequence
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch, re, os

PROMPT_TMPL = """You are a medical assistant.
Answer the QUESTION strictly using only facts in CONTEXT.
Do NOT include any country/city/province names or percentages unless they appear verbatim in CONTEXT.
Do NOT ask a question. Give a direct, neutral medical answer in 3–6 sentences.

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""

FALLBACK_TMPL = """You are a medical assistant.
Answer the QUESTION in a detailed and comprehensive way, using your general medical knowledge.
Provide 3–6 sentences. If uncertain, say you are not certain.

QUESTION: {question}
ANSWER:"""

_num_re = re.compile(r"\d+(?:\.\d+)?%?")
_word_re = re.compile(r"[A-Za-zğüşöçıİĞÜŞÖÇ0-9]+")

_STOPWORDS = {
    # EN
    "the","a","an","and","or","of","to","in","for","on","with","as","by","at","from","that","this","be","is","are","was","were","it",
    # TR
    "ve","veya","ile","için","da","de","bu","şu","o","bir","olan","olarak","gibi","ama","fakat","ancak","ya","en",
}

def _coerce_texts(docs: Sequence[Any]) -> List[str]:
    texts: List[str] = []
    for d in docs:
        if isinstance(d, str):
            t = d
        elif isinstance(d, dict):
            t = d.get("text") or d.get("page_content") or d.get("content") or ""
        else:
            t = getattr(d, "text", getattr(d, "page_content", str(d)))
        t = (t or "").strip()
        if t:
            texts.append(t)
    return texts

def build_prompt(question: str, docs: Sequence[Any]) -> str:
    ctx = "\n\n".join(_coerce_texts(docs)[:5])
    return PROMPT_TMPL.format(context=ctx, question=question)

def _numbers(text: str) -> set[str]:
    return set(_num_re.findall(text or ""))

def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _word_re.findall(text or "") if w]

def _content_words(text: str) -> List[str]:
    return [w for w in _tokenize(text) if w not in _STOPWORDS and len(w) > 2]

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _context_relevant(question: str, docs: Sequence[Any], min_ratio: float = 0.12) -> bool:
    """
    Returns True only if retrieved docs actually overlap with the question
    (basic lexical signal). Prevents using unrelated chunks as 'context'.
    """
    q = set(_content_words(question))
    if not q:
        return False
    ctx_words: set = set()
    for t in _coerce_texts(docs):
        ctx_words |= set(_content_words(t))
    return _jaccard(q, ctx_words) >= min_ratio

def _strip_unseen_numbers(answer: str, allowed_nums: set[str]) -> str:
    if not answer:
        return answer
    sents = re.split(r"(?<=[.!?])\s+", answer)
    keep = []
    for s in sents:
        nums = _numbers(s)
        # drop sentence if it introduces numbers not present in context
        if nums and any(n not in allowed_nums for n in nums):
            continue
        keep.append(s)
    cleaned = " ".join(keep).strip()
    return cleaned if cleaned else answer

def _filter_sentences_not_in_context(answer: str, context_text: str, min_overlap: float = 0.10) -> str:
    """
    Extra guard: drop sentences whose content words barely appear in the context.
    This reduces irrelevant/hallucinated lines.
    """
    if not answer:
        return answer
    ctx_set = set(_content_words(context_text))
    sents = re.split(r"(?<=[.!?])\s+", answer)
    kept = []
    for s in sents:
        cw = set(_content_words(s))
        if not cw:
            continue
        overlap = _jaccard(cw, ctx_set)
        if overlap >= min_overlap:
            kept.append(s)
    cleaned = " ".join(kept).strip()
    return cleaned if cleaned else answer  # fall back to original if all dropped


class Generator:
    def __init__(self, model_id: str = "microsoft/BioGPT",
                 offload_folder: Optional[str] = None,
                 device: Optional[str] = None):
        self.model_id = model_id
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # dtype: on CUDA prefer bf16 if supported, else fp16; CPU uses fp32
        if self.device == "cuda":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32

        is_biogpt = "biogpt" in (self.model_id or "").lower()

        # Tokenizer: BioGPT has no fast tokenizer -> use_fast=False there
        tok_kwargs = {"use_fast": False} if is_biogpt else {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tok_kwargs)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        common_kwargs = dict(
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            offload_folder=offload_folder,
            trust_remote_code=False,
        )

        try:
            if is_biogpt:
                # BioGPT does NOT support device_map='auto' → load on single device
                # Some BioGPT weights are picky with fp16/bf16; if it fails, we fall back to fp32 below.
                if self.device == "cpu":
                    common_kwargs["torch_dtype"] = torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **common_kwargs)
                self.model.to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map=("auto" if self.device != "cpu" else None),
                    **common_kwargs,
                )
        except Exception as e:
            # Generic fallback: most robust path → fp32 on a single device
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                trust_remote_code=False,
                offload_folder=offload_folder,
            )
            self.model.to(self.device)

        self.model.eval()

        # CUDA speed hint
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

        # Pipeline will use the model as-is (already on device if applicable)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=(0 if self.device == "cuda" else -1),  # GPU varsa 0, yoksa CPU
        )


        # longer answers
        self.max_input_tokens = min(getattr(self.tokenizer, "model_max_length", 1024), 1024) - 96
        self.max_new_tokens = 512

        print(f"[Generator] Loaded {self.model_id} on {self.device.upper()} with {self.model.dtype} precision.")

    def _truncate_prompt(self, prompt: str) -> str:
        toks = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if toks.shape[-1] <= self.max_input_tokens:
            return prompt
        kept = toks[-self.max_input_tokens:]
        return self.tokenizer.decode(kept, skip_special_tokens=True)

    def _dynamic_token_cap(self, query_text: Optional[str]) -> int:
        if not query_text:
            return self.max_new_tokens
        q_len = len(query_text.strip())
        if q_len < 40:   return 256
        if q_len < 100:  return 384
        return 512

    def _generate_text(self, prompt: str, **gen_kwargs) -> str:
        out = self.pipe(prompt, **gen_kwargs)
        return (out[0]["generated_text"] or "").strip()

    def generate(
        self,
        prompt: Optional[str] = None,
        *,
        docs: Optional[Sequence[Any]] = None,
        query_text: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = 1.0,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = 1.05,
        max_new_tokens: Optional[int] = None,
    ) -> str:

        #  deciding mechanism
        have_docs = bool(docs) and len(_coerce_texts(docs)) > 0
        ctx_is_relevant = have_docs and bool(query_text) and _context_relevant(query_text, docs)

        if not ctx_is_relevant:
            #  fallback (no/low context overlap)
            fb_prompt = FALLBACK_TMPL.format(question=query_text or prompt or "")
            if max_new_tokens is None:
                max_new_tokens = self._dynamic_token_cap(query_text)
            gen_kwargs = {
                "max_new_tokens": int(max_new_tokens),
                "return_full_text": False,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "do_sample": True,
                "temperature": max(0.3, float(temperature or 0.7)),
                "top_p": 0.9,
            }
            fb_raw = self._generate_text(fb_prompt, **gen_kwargs)
            return fb_raw.strip()

        # normal RAG
        if prompt is None and query_text:
            prompt = build_prompt(query_text, docs)

        safe_prompt = self._truncate_prompt(prompt or "")
        if max_new_tokens is None:
            max_new_tokens = self._dynamic_token_cap(query_text)

        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "return_full_text": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature is None or float(temperature) <= 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = max(0.05, min(float(temperature), 2.0))
            if top_p is not None:
                gen_kwargs["top_p"] = max(0.05, min(float(top_p), 1.0))
            if top_k is not None:
                gen_kwargs["top_k"] = max(1, int(top_k))

        rp = max(1.05, float(repetition_penalty) if repetition_penalty is not None else 1.05)
        gen_kwargs["repetition_penalty"] = rp
        gen_kwargs["no_repeat_ngram_size"] = 3

        raw = self._generate_text(safe_prompt, **gen_kwargs)

        # cleanup relative to context
        ctx_start = safe_prompt.find("CONTEXT:")
        ctx_end = safe_prompt.find("\n\nQUESTION:")
        context_text = safe_prompt[ctx_start:ctx_end] if (ctx_start != -1 and ctx_end != -1) else safe_prompt
        allowed_nums = _numbers(context_text)

        cleaned = _strip_unseen_numbers(raw, allowed_nums)
        cleaned = _filter_sentences_not_in_context(cleaned, context_text, min_overlap=0.10)

        return cleaned
