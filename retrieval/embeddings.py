"""
Embedding models.

The LLD picks BGE/E5-family models. Loading either requires downloading
weights from the Hugging Face Hub, and this environment can't reach it
(network is allow-listed to package registries only — see the sandbox's
network_configuration). So, same pattern as GrobidClient in Step 3: an
interface (`EmbeddingModel`), a real production implementation that can't be
exercised here (`SentenceTransformerEmbedder`), and a deterministic offline
implementation the whole retrieval pipeline is actually tested against
(`HashingEmbedder`).

`HashingEmbedder` is NOT a semantic embedding model — it has no notion of
synonyms or meaning. It's a signed feature-hashing bag-of-words vectorizer
(the "hashing trick"), included so chunking -> embedding -> retrieval can be
tested end-to-end, deterministically, with zero external dependencies. Swap
in `SentenceTransformerEmbedder` for real semantic retrieval quality; nothing
else in the pipeline needs to change since both implement `EmbeddingModel`.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Sequence

_TOKEN_RE = re.compile(r"[A-Za-z0-9°]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class EmbeddingModel(ABC):
    name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder(EmbeddingModel):
    """Deterministic, offline, dependency-free. Two texts sharing more
    tokens get a higher cosine similarity — enough to meaningfully test
    chunking and retrieval logic without a real model."""

    def __init__(self, dimension: int = 256):
        self.dimension = dimension
        self.name = f"hashing-{dimension}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec


class SentenceTransformerEmbedder(EmbeddingModel):
    """
    Production embedder for the BGE/E5 family via `sentence-transformers`.

    Not exercised by the test suite — instantiating this downloads model
    weights from the Hugging Face Hub, which this environment can't reach.
    Swap this in for `HashingEmbedder` wherever the app is deployed with
    model-hub access; both satisfy `EmbeddingModel`, so nothing downstream
    (chunking, indexing, retrieval) needs to change.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # local import: optional heavy dep

        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()
