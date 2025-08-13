import sys, os, re, json, time
import httpx
from xml.etree import ElementTree as ET


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.retrieval_faiss import FaissStore


OUT_JSONL = "data/processed/pubmed_multi.jsonl"
INDEX_DIR = "data/index_pubmed"
os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# PubMed API 
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
EMAIL = os.getenv("EMAIL", "your_email@example.com")
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# same topics
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


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def esearch(term, retmax=200):
    """Search PubMed and return a list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": str(retmax),
        "api_key": NCBI_API_KEY,
        "email": EMAIL
    }
    r = httpx.get(f"{BASE}/esearch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]

def efetch(pmids):
    """Fetch details for given PMIDs in batches."""
    BATCH = 200
    for i in range(0, len(pmids), BATCH):
        chunk = pmids[i:i+BATCH]
        params = {
            "db": "pubmed",
            "id": ",".join(chunk),
            "retmode": "xml",
            "api_key": NCBI_API_KEY,
            "email": EMAIL
        }
        r = httpx.get(f"{BASE}/efetch.fcgi", params=params, timeout=120)
        r.raise_for_status()
        yield r.text
        time.sleep(0.34) 

def parse_pubmed_xml(xml_text):
    """Extract title + abstract from PubMed XML."""
    root = ET.fromstring(xml_text)
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//PMID") or "").strip()
        title = article.findtext(".//ArticleTitle") or ""
        abstract_text = " ".join(
            (a.text or "") for a in article.findall(".//AbstractText") if a is not None
        )
        yield {
            "id": pmid,
            "text": normalize(f"{title}. {abstract_text}"),
            "source": f"pubmed:{pmid}"
        }


def main():
    seen_ids = set()
    total = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for topic in TOPICS:
            term = f"{topic}[Title/Abstract] AND (2018:3000[Date - Publication])"
            print(f"[PubMed] Fetching topic: {topic}")
            pmids = esearch(term, retmax=300)  
            for xml in efetch(pmids):
                for doc in parse_pubmed_xml(xml):
                    if not doc["id"] or not doc["text"]:
                        continue
                    if doc["id"] in seen_ids:
                        continue
                    seen_ids.add(doc["id"])
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    total += 1
    print(f"[PubMed] Wrote {total} unique docs → {OUT_JSONL}")

    # faiss index
    print("[Indexing] Building FAISS index from ingested PubMed data...")
    store = FaissStore(index_dir=INDEX_DIR)
    with open(OUT_JSONL, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    store.add([d["text"] for d in docs], metas=docs)
    store.save()
    print(f"[Indexing] Saved FAISS index to {INDEX_DIR}")

if __name__ == "__main__":
    main()
