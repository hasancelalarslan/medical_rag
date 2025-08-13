# app/main.py

import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import get_settings
from app.retrieval_faiss import FaissStore
from app.generation_model import Generator

S = get_settings()
app = FastAPI(title="Medical RAG", version="0.1.0")

#  Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response 
class QueryRequest(BaseModel):
    query: str = Field(..., description="Medical question to answer")
    k: int = Field(5, ge=1, le=20, description="Top-k passages to retrieve")
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0, le=1, description="Top-p nucleus sampling")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k sampling")
    repetition_penalty: Optional[float] = Field(None, ge=0, description="Repetition penalty")

class SourceDoc(BaseModel):
    text: str
    score: float
    source: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]
    timings_ms: Dict[str, float]

#App 
startup_error: Optional[str] = None
retriever: Optional[FaissStore] = None
generator: Optional[Generator] = None

@app.on_event("startup")
def _startup():
    global retriever, generator, startup_error
    try:
        retriever = FaissStore(index_dir="data/index")
        index_path = os.path.join("data", "index", "index.faiss")
        if os.path.exists(index_path):
            retriever.load()
            print(f"[FAISS] Loaded index from data/index ({retriever.index.ntotal} docs)")
        else:
            startup_error = "FAISS index not found in data/index. Please build or merge first."
            print(f"⚠ {startup_error}")
    except Exception as e:
        startup_error = f"Failed to initialize FAISS retriever: {e}"
        print(f" {startup_error}")

    try:
        model_id = getattr(S, "gen_model_id", None) or os.getenv("GEN_MODEL_ID", "microsoft/BioGPT")
        generator = Generator(model_id=model_id)
        print("[Generator] Model loaded successfully.")
    except Exception as e:
        startup_error = f"Failed to initialize generation model: {e}"
        print(f" {startup_error}")

#  Route
@app.get("/health")
def health():
    return {
        "status": "ok" if startup_error is None else "degraded",
        "startup_error": startup_error,
        "model_id": getattr(generator, "model_id", None),
        "retriever_ready": retriever is not None,
        "generator_ready": generator is not None,
    }

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Service not ready: {startup_error}")
    if retriever is None or generator is None:
        raise HTTPException(status_code=503, detail="Service initializing, try again in a moment.")

    t0 = time.perf_counter()

    #  Retrieve
    t_r0 = time.perf_counter()
    hits = retriever.search(req.query, k=req.k)  # returns (meta_dict, score)
    t_r1 = time.perf_counter()

    if not hits:
        raise HTTPException(status_code=400, detail="Index is empty. Build the FAISS index first.")

    #  Build sources 
    sources: List[SourceDoc] = []
    doc_texts: List[str] = []
    for meta, score in hits:
        text = meta.get("text") if isinstance(meta, dict) else str(meta)
        sources.append(SourceDoc(
            text=text,
            score=float(score),
            source=(meta.get("source") if isinstance(meta, dict) else None),
            meta=(meta if isinstance(meta, dict) else {"raw": meta})
        ))
        if text:
            doc_texts.append(text)

    #  Generate 
    t_g0 = time.perf_counter()
    answer = generator.generate(
        docs=doc_texts,                # top-k snippets
        query_text=req.query,          # raw user question
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
    )
    t_g1 = time.perf_counter()

    t1 = time.perf_counter()
    return QueryResponse(
        answer=answer,
        sources=sources,
        timings_ms={
            "retrieval": round((t_r1 - t_r0) * 1000, 2),
            "generation": round((t_g1 - t_g0) * 1000, 2),
            "total": round((t1 - t0) * 1000, 2),
        },
    )

# Run local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
