# Medical RAG 

FastAPI RAG for medical Q&A using FAISS retrieval + BioGPT generation. Dockerized API and optional Streamlit UI. Includes scripts for perf/quality eval.

> Disclaimer: Research/demo only. Not a medical device.

---

#TL;DR
```bash
# Build API
docker build -t medical-rag:latest .

# Run API
docker run --rm -p 8000:8000 --env-file .env \
  -v "$(pwd)/.cache:/cache/huggingface" -v "$(pwd)/data:/app/data" \
  medical-rag:latest

#  Build index (if missing)
python scripts/build_index.py --docs_dir docs --out_dir data/index

#  Test
curl -s -X POST http://127.0.0.1:8000/query \
 -H "Content-Type: application/json" \
 -d '{"query":"What is hypertension?","k":5,"temperature":0.0}'
```

---

 Layout
```
app/
  main.py                
  generation_model.py    
  retrieval_faiss.py    
  config.py             
ui/
  ui_app.py             
data/
  index/                 
  processed/             
docs/                    
examples/
  queries_100.csv        
scripts/
  build_index.py         
  build_model_outputs.py 
  run_perf.py            
  run_quality.py         
  run_manual_eval.py     
eval/                    
Dockerfile               
Dockerfile.ui            
docker-compose.yml       
requirements.txt
.dockerignore


---

## .env (example)
```ini
EMB_MODEL_ID=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
GEN_MODEL_ID=microsoft/BioGPT
INDEX_DIR=/app/data/index
PROCESSED_DIR=/app/data/processed
HF_TOKEN=
```

---

## docker-compose (API + UI)
```yaml
services:
  api:
    image: medical-rag:latest
    ports: ["8000:8000"]
    env_file: .env
    volumes: [".cache:/cache/huggingface","./data:/app/data"]
  ui:
    image: medical-rag-ui:latest
    ports: ["8501:8501"]
    environment: { API_URL: "http://api:8000" }
    depends_on: [api]
```
Run: `docker compose up -d` → API: `:8000`, UI: `:8501`

---

Evaluation
```bash
# Performance eval/perf_results.csv
API_URL=http://127.0.0.1:8000 python scripts/run_perf.py

# Quality  eval/quality_report.md
python scripts/build_model_outputs.py --api_url http://127.0.0.1:8000 \
  --in_csv examples/queries_100.csv --out_jsonl eval/model_outputs.jsonl
python scripts/run_quality.py --inputs eval/model_outputs.jsonl --out eval/quality_report.md
```

---


No index / degraded health** → Build `data/index/index.faiss`.
Slow on CPU** → Lower `k`, persist `./.cache`; consider smaller embeddings.
CORS/UI** → UI uses `API_URL`; API has CORS enabled.

---


Uses FAISS, Hugging Face Transformers, and BioGPT. See code for notices.

Googel Drive mp4 File: https://drive.google.com/file/d/1Ja4L6xLPoWVB2ybL5Y-NGPxX1vzARYs2/view?usp=sharing

