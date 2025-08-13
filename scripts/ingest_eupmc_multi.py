# scripts/ingest_eupmc_multi.py
import sys, os, re, json, time
import httpx

# make project root importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.retrieval_faiss import FaissStore  # needs index_dir support

# ---------- OUTPUT PATHS ----------
OUT_JSONL = "data/processed/eupmc_multi.jsonl"
INDEX_DIR  = "data/index_eupmc"
os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ---------- TOPICS (edit/extend freely) ----------
TOPICS = [
    "hypertension", "heart failure", "myocardial infarction", "stroke",
    "coronary artery disease", "arrhythmia", "atrial fibrillation", "peripheral artery disease", "hyperlipidemia",
    "type 1 diabetes", "type 2 diabetes", "gestational diabetes", "obesity", "metabolic syndrome",
    "lung cancer", "breast cancer", "colorectal cancer", "prostate cancer", "pancreatic cancer",
    "Alzheimer's disease", "Parkinson's disease", "multiple sclerosis", "epilepsy", "migraine",
    "COVID-19", "influenza", "tuberculosis", "HIV", "hepatitis B", "hepatitis C",
    "asthma", "chronic obstructive pulmonary disease", "pneumonia", "pulmonary fibrosis",
    "Crohn's disease", "ulcerative colitis", "irritable bowel syndrome", "celiac disease",
    "chronic kidney disease", "acute kidney injury", "kidney stones", "urinary tract infection",
    "rheumatoid arthritis", "osteoarthritis", "lupus", "psoriasis", "ankylosing spondylitis",
    "anemia", "hemophilia", "thalassemia", "sickle cell disease",
    "preeclampsia", "endometriosis", "polycystic ovary syndrome", "menopause", "infertility",
    "mental health", "depression", "anxiety disorders", "vaccination", "antibiotic resistance",
    "telemedicine", "precision medicine", "gene therapy", "stem cell therapy"
]

# ---------- HELPERS ----------
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def search_eupmc(query, page_size=100, max_pages=5):
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    for page in range(1, max_pages + 1):
        params = {
            "query": query,
            "format": "json",
            "pageSize": str(page_size),
            "page": str(page)
        }
        r = httpx.get(base_url, params=params, timeout=60)
        r.raise_for_status()
        for rec in r.json().get("resultList", {}).get("result", []):
            yield rec
        time.sleep(0.2)  # be polite

def main():
    # --------- 1) INGEST TO JSONL ---------
    seen = set()
    total = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for topic in TOPICS:
            query = f'TITLE_ABS:"{topic}" AND PUB_YEAR:2018-2025'
            print(f"[Europe PMC] Fetching topic: {topic}")
            for rec in search_eupmc(query, page_size=100, max_pages=5):
                doc_id = rec.get("id") or rec.get("pmid") or rec.get("doi")
                title = rec.get("title", "")
                abstract = rec.get("abstractText", "")
                if not doc_id or not (title or abstract):
                    continue
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                doc = {
                    "id": doc_id,
                    "text": normalize(f"{title}. {abstract}"),
                    "source": f"{rec.get('source','eupmc')}:{doc_id}"
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                total += 1
    print(f"[Europe PMC] Wrote {total} unique docs → {OUT_JSONL}")

    # --------- 2) BUILD FAISS INDEX ---------
    print("[Indexing] Building FAISS index from Europe PMC data...")
    store = FaissStore(index_dir=INDEX_DIR)  # requires your FaissStore to accept index_dir
    # stream to limit RAM if file is large
    texts, metas = [], []
    with open(OUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if not d.get("text"): 
                continue
            texts.append(d["text"])
            metas.append(d)
            # optional: flush in chunks to save memory
            if len(texts) >= 2000:
                store.add(texts, metas=metas)
                texts, metas = [], []
    if texts:
        store.add(texts, metas=metas)
    store.save()
    print(f"[Indexing] Saved FAISS index to {INDEX_DIR}")

if __name__ == "__main__":
    main()
