"""
NerModel interface (LLD section 12).

Same pattern as EmbeddingModel (Step 4) and GrobidClient (Step 3): a real
domain NER model needs weights that aren't reachable from this environment,
so there's an interface, an offline-testable implementation the pipeline
actually runs against (GazetteerNER), and a production implementation that
isn't exercised here (TransformerNER).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from extractors.ner.labels import EntityLabel


class Entity(BaseModel):
    text: str
    label: EntityLabel
    char_start: int
    char_end: int
    confidence: float = 1.0


class NerModel(ABC):
    name: str

    @abstractmethod
    def predict(self, text: str) -> list[Entity]: ...
