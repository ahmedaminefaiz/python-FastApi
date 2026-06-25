import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
from fastapi.middleware.cors import CORSMiddleware

from clip_service import clip_service
from models import SimilarityRequest, SimilarityResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    clip_service.load_model()
    yield


app = FastAPI(title="Similarity Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest) -> SimilarityResponse:
    results = await clip_service.compare(request.sourceImageUrl, request.candidates)
    return SimilarityResponse(results=results)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": clip_service.model is not None}
