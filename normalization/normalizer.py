"""
Entity normalization (LLD section 19):

    "AgNO3"
    "Silver nitrate"
    "silver nitrate"
            |
            v
      Silver nitrate
      canonical_id = AGNO3

Matching hierarchy, in order (LLD's own list is "exact alias -> normalized
string -> domain dictionary -> ontology lookup -> embedding similarity";
here the domain dictionary IS the exact-alias table, ontology lookup is a
separate concern (linking.py), and embedding similarity is the fallback for
names not in the dictionary verbatim):

  1. Exact alias match, case/whitespace-normalized, against
     canonical_dictionary.CANONICAL_ENTITIES.
  2. Embedding-similarity fallback (reuses Step 4's EmbeddingModel) for
     names close to — but not exactly — a known alias, e.g. a hyphenation
     or pluralization the dictionary doesn't happen to list. Given a
     similarity score, not a fixed 1.0, so callers can see it was a fuzzy
     match. Inherits HashingEmbedder's lexical-only limitation (Step 4) —
     it catches surface variation, not true synonyms with no shared
     tokens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from normalization.canonical_dictionary import CANONICAL_ENTITIES, CanonicalEntity
from retrieval.embeddings import EmbeddingModel
from retrieval.retriever import cosine_similarity


class NormalizedEntity(BaseModel):
    canonical_id: str
    canonical_name: str
    confidence: float
    match_method: str  # "exact_alias" | "embedding_similarity"


class EntityNormalizer(ABC):
    @abstractmethod
    def normalize(self, name: str) -> NormalizedEntity | None: ...


def _normalize_string(s: str) -> str:
    return " ".join(s.strip().lower().split())


class DictionaryNormalizer(EntityNormalizer):
    def __init__(
        self,
        entities: list[CanonicalEntity] | None = None,
        embedder: EmbeddingModel | None = None,
        similarity_threshold: float = 0.6,
    ):
        self.entities = entities if entities is not None else CANONICAL_ENTITIES
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

        self._alias_index: dict[str, CanonicalEntity] = {}
        for entity in self.entities:
            self._alias_index[_normalize_string(entity.canonical_name)] = entity
            for synonym in entity.synonyms:
                self._alias_index[_normalize_string(synonym)] = entity

        self._entity_vectors: dict[str, list[float]] = {}
        if embedder is not None:
            names = [e.canonical_name for e in self.entities]
            vectors = embedder.embed(names)
            self._entity_vectors = {e.canonical_id: v for e, v in zip(self.entities, vectors)}

    def normalize(self, name: str) -> NormalizedEntity | None:
        exact = self._alias_index.get(_normalize_string(name))
        if exact is not None:
            return NormalizedEntity(
                canonical_id=exact.canonical_id, canonical_name=exact.canonical_name,
                confidence=1.0, match_method="exact_alias",
            )

        if self.embedder is not None and self.entities:
            query_vector = self.embedder.embed([name])[0]
            best_entity, best_score = None, 0.0
            for entity in self.entities:
                score = cosine_similarity(query_vector, self._entity_vectors[entity.canonical_id])
                if score > best_score:
                    best_entity, best_score = entity, score
            if best_entity is not None and best_score >= self.similarity_threshold:
                return NormalizedEntity(
                    canonical_id=best_entity.canonical_id, canonical_name=best_entity.canonical_name,
                    confidence=best_score, match_method="embedding_similarity",
                )

        return None
