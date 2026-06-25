import asyncio
import io
import logging

import httpx
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from models import SimilarityScore

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.82


class ClipService:
    def __init__(self):
        self.model = None

    def load_model(self):
        self.model = SentenceTransformer("clip-ViT-B-32")

    async def _download_image(self, client: httpx.AsyncClient, url: str) -> Image.Image | None:
        try:
            response = await client.get(url, timeout=10.0, follow_redirects=True)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception:
            return None

    async def _download_all(
        self, source_url: str, candidates: list
    ) -> tuple[Image.Image | None, dict[int, Image.Image]]:
        headers = {"User-Agent": "SimilarityService/1.0 (image similarity bot)"}
        async with httpx.AsyncClient(headers=headers) as client:
            source_task = self._download_image(client, source_url)
            candidate_tasks = [self._download_image(client, c.imageUrl) for c in candidates]
            results = await asyncio.gather(source_task, *candidate_tasks)

        source_image = results[0]
        candidate_images = {
            candidates[i].id: results[i + 1]
            for i in range(len(candidates))
            if results[i + 1] is not None
        }
        return source_image, candidate_images

    def _embed(self, images: list) -> np.ndarray:
        return self.model.encode(images, batch_size=8, normalize_embeddings=True)

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b))

    async def compare(self, source_url: str, candidates: list) -> list[SimilarityScore]:
        source_image, candidate_images = await self._download_all(source_url, candidates)

        if source_image is None:
            logger.warning("Source image could not be downloaded: %s", source_url)
            return []
        if not candidate_images:
            logger.warning("No candidate images could be downloaded (all failed)")
            return []

        candidate_ids = list(candidate_images.keys())
        all_images = [source_image] + [candidate_images[cid] for cid in candidate_ids]

        embeddings = self._embed(all_images)
        source_vec = embeddings[0]
        candidate_vecs = embeddings[1:]

        scores = [
            SimilarityScore(id=cid, score=self._cosine_similarity(source_vec, candidate_vecs[i]))
            for i, cid in enumerate(candidate_ids)
        ]

        logger.info("Raw scores before threshold: %s", [(s.id, round(s.score, 4)) for s in scores])

        filtered = [s for s in scores if s.score >= SIMILARITY_THRESHOLD]
        filtered.sort(key=lambda s: s.score, reverse=True)

        logger.info("After threshold (%.2f): %d/%d results", SIMILARITY_THRESHOLD, len(filtered), len(scores))
        return filtered


clip_service = ClipService()
