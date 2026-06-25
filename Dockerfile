# Stage 1: install dependencies
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/deps -r requirements.txt


# Stage 2: runtime
FROM python:3.12-slim

ENV PYTHONPATH=/install/deps/lib/python3.12/site-packages \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /install/deps /install/deps

WORKDIR /app
COPY main.py clip_service.py models.py ./

# Pre-download CLIP model into the image so cold starts are instant
RUN PYTHONPATH=/install/deps/lib/python3.12/site-packages \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('clip-ViT-B-32')"

EXPOSE 8000

CMD ["/install/deps/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
