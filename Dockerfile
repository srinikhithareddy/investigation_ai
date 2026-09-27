FROM python:3.11-slim

WORKDIR /app

# System deps needed by sentence-transformers / chromadb wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/chroma

ENV PYTHONUNBUFFERED=1 \
    DATABASE_PATH=/app/data/app.db \
    CHROMA_PATH=/app/data/chroma \
    DOCUMENTS_PATH=/app/data/documents.json

EXPOSE 8000

# Ingest the bundled documents.json at container start (idempotent), then serve.
CMD ["sh", "-c", "python -m app.ingestion.documents && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
