from pydantic import BaseModel


class ImageCandidate(BaseModel):
    id: int
    imageUrl: str


class SimilarityRequest(BaseModel):
    sourceImageUrl: str
    candidates: list[ImageCandidate]


class SimilarityScore(BaseModel):
    id: int
    score: float


class SimilarityResponse(BaseModel):
    results: list[SimilarityScore]
