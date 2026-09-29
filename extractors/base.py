"""
Extractor interface (LLD section 7 — Strategy pattern).

    Extractor
       |
       +-- RuleExtractor   (this step)
       +-- NerExtractor    (Step 6)
       +-- PromptLlmExtractor (Step 7; replaced RagLlmExtractor)
       +-- HybridExtractor (Step 8)

Every subclass takes the same `CanonicalDocument` (Step 3's output) and
returns the same `ExtractionResult` (Step 1's schema). That uniformity is
the entire point: it's what makes the four-way ablation study a fair
comparison instead of four incompatible pipelines.

Note this drops the `schema: ExtractionSchema` parameter the LLD sketch
shows in `Extractor.extract(document, schema)`. That made sense when the
schema was still hypothetical; Step 1 already fixed it as
`SynthesisExtraction` for the whole project, so threading it through every
call site would only risk one extractor silently getting a different schema
version than the others — worse for comparability, not better.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ingestion.canonical import CanonicalDocument
from schemas.extraction_schema import ExtractionResult, ExtractorName


class Extractor(ABC):
    name: ExtractorName
    version: str

    @abstractmethod
    def extract(self, document: CanonicalDocument) -> ExtractionResult: ...
