# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    HF_HOME=/cache/huggingface \
    NLTK_DATA=/home/appuser/nltk_data


RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir rouge-score

RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt wordnet omw-1.4

COPY . .

RUN mkdir -p /cache/huggingface

RUN useradd -m appuser && \
    mkdir -p /home/appuser/nltk_data && \
    chown -R appuser:appuser /app /cache /home/appuser/nltk_data
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${UVICORN_PORT}/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
