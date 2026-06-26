# Stage 1: install dependencies and pre-download model files
FROM python:3.12-slim AS builder

ENV HF_HOME=/model_cache \
    PIP_NO_CACHE_DIR=1 \
    PIP_TMPDIR=/build/tmp

WORKDIR /build
RUN mkdir -p /build/tmp

COPY requirements.txt .

RUN python -m venv /venv

# Step 1: torch seul (~190MB) — couche séparée pour limiter la pression mémoire
RUN /venv/bin/pip install torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Step 2: reste des dépendances
RUN /venv/bin/pip install \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.29.0" \
    sentence-transformers==3.0.1 \
    httpx==0.27.0 \
    Pillow==10.3.0 \
    numpy==1.26.4 \
    pydantic==2.7.1

# Step 3: téléchargement des fichiers du modèle (sans chargement en RAM)
RUN /venv/bin/python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('sentence-transformers/clip-ViT-B-32')"

# Stage 2: runtime
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:$PATH" \
    HF_HOME=/model_cache

COPY --from=builder /venv /venv
COPY --from=builder /model_cache /model_cache

RUN useradd --no-create-home appuser && \
    chmod -R 755 /model_cache

USER appuser

WORKDIR /app
COPY main.py clip_service.py models.py ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
