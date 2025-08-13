import sys, os, json, re
from typing import List
from pypdf import PdfReader


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from app.retrieval_faiss import FaissStore
except Exception:
    FaissStore = None  


RAW_DIR = "data/raw/guidelines"            
OUT_JSONL = "data/processed/guidelines.jsonl"
INDEX_DIR = "data/index_guidelines"         
BUILD_INDEX = True                           

os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
if BUILD_INDEX:
    os.makedirs(INDEX_DIR, exist_ok=True)


def normalize(txt: str) -> str:
    return re.sub(r"\s+", " ", (txt or "")).strip()

def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    parts: List[str] = []
    for p in reader.pages:
        t = p.extract_text() or ""
        parts.append(t)
    return normalize("\n".join(parts))


def main():
    total = 0
    docs = []

    
    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for fn in sorted(os.listdir(RAW_DIR)):
            if not fn.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(RAW_DIR, fn)
            print(f"[Guidelines] Extracting: {fn}")
            text = pdf_to_text(pdf_path)
            if not text:
                print(f"[Guidelines] Skipped (no text): {fn}")
                continue
            doc = {"id": fn, "text": text, "source": f"guideline:{fn}"}
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            docs.append(doc)
            total += 1
    print(f"[Guidelines] Wrote {total} docs → {OUT_JSONL}")

    
    if BUILD_INDEX:
        if FaissStore is None:
            raise RuntimeError("FaissStore import failed; set BUILD_INDEX=False or fix app.retrieval_faiss import.")
        if not docs:
            print("[Indexing] No docs, skipping FAISS build.")
            return
        print("[Indexing] Building FAISS index from guideline docs...")
        store = FaissStore(index_dir=INDEX_DIR)
        
        B = 100
        for i in range(0, len(docs), B):
            chunk = docs[i:i+B]
            store.add([d["text"] for d in chunk], metas=chunk)
        store.save()
        print(f"[Indexing] Saved FAISS index → {INDEX_DIR}")

if __name__ == "__main__":
    
    os.makedirs(RAW_DIR, exist_ok=True)
    main()
