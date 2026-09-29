"""
Retriever interface + implementation.

LLD doc 2 section 11: "Paper -> Chunks -> Embedding -> Vector Index ->
Extraction Query -> Retrieve relevant chunks -> Optional reranking -> LLM."
This module is the "Retrieve" step, scoped per document_version — retrieval
for extraction is always "find the relevant chunks *within this paper*",
never a cross-corpus search.

Reranking (LLD section 11's "optional reranking") is deliberately not here:
it's tightly coupled to how an LLM extractor phrases its
per-field queries, so it belongs with that extractor, not the generic
retrieval layer.
"""

from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod

from pydantic import BaseModel

from storage.read_models import ChunkRecord
from storage.repository import PaperRepository


class RetrievedChunk(BaseModel):
    chunk: ChunkRecord
    score: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class Retriever(ABC):
    @abstractmethod
    def search(
        self, version_id: uuid.UUID, query_embedding: list[float], top_k: int = 10
    ) -> list[RetrievedChunk]: ...


class CosineRetriever(Retriever):
    """
    Brute-force cosine similarity over one document_version's chunks.

    For a 50-100 paper corpus (a few thousand chunks total, at most), scoring
    every chunk in Python is fast enough that a dedicated ANN index isn't
    justified — the same "don't over-engineer for this scale" reasoning the
    LLD uses to justify pgvector over a separate vector DB in the first
    place (Step 2's README).

    If retrieval volume or corpus size grows enough that this becomes the
    bottleneck, replace it with a native SQL query using pgvector's
    similarity operator:

        SELECT * FROM chunks
        WHERE version_id = :version_id
        ORDER BY embedding <=> :query_embedding
        LIMIT :top_k

    which requires switching `storage.db_types.EmbeddingType` to
    `pgvector.sqlalchemy.Vector(dim)` — the one place flagged for that
    change back in Step 2 — but nothing in this module's interface changes.
    """

    def __init__(self, repository: PaperRepository):
        self.repository = repository

    def search(
        self, version_id: uuid.UUID, query_embedding: list[float], top_k: int = 10
    ) -> list[RetrievedChunk]:
        chunks = self.repository.get_chunks(version_id)
        scored = [
            RetrievedChunk(chunk=c, score=cosine_similarity(query_embedding, c.embedding))
            for c in chunks
            if c.embedding is not None
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
