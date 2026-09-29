"""
Provenance & confidence models.

Every fact any extractor (rule / NER / Prompt+LLM / hybrid) produces gets wrapped
in a CandidateFact. This is what lets the fusion layer, the evaluator, and a
human reviewer all answer: "where did this value come from, and how much
should I trust it?"

Nothing here is extractor-specific — this file has zero imports from rules/,
ner/, rag/, etc. That's intentional: provenance is a cross-cutting concern,
not something owned by any one extraction method.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    """Which subsystem produced a candidate fact."""

    RULE = "RULE"
    NER = "NER"
    LLM = "LLM"
    HYBRID = "HYBRID"
    HUMAN = "HUMAN"  # gold-dataset annotations flow through the same model


class EvidenceSpan(BaseModel):
    """
    Pointer back into the source document: paper -> version -> page -> text.

    char_start/char_end are offsets into the block's `text` field (see the
    document_blocks table in the LLD) and are optional because not every
    extractor can localize a fact down to a character span (e.g. an LLM
    answer synthesized across two paragraphs).
    """

    paper_id: str
    version: int = Field(ge=1, description="document_versions.version this evidence was found in")
    page: Optional[int] = Field(default=None, ge=1)
    section: Optional[str] = None
    text: str = Field(description="The verbatim snippet supporting the fact")
    char_start: Optional[int] = Field(default=None, ge=0)
    char_end: Optional[int] = Field(default=None, ge=0)

    @field_validator("char_end")
    @classmethod
    def _end_after_start(cls, v, info):
        start = info.data.get("char_start")
        if v is not None and start is not None and v < start:
            raise ValueError("char_end must be >= char_start")
        return v


class CandidateFact(BaseModel):
    """
    A single (field, value) proposal from one extractor, with the evidence
    and confidence needed to resolve conflicts later (LLD section 18).

    `field` uses dotted paths so it can address nested schema fields, e.g.
    "precursors.0.concentration" or "temperature".
    """

    field: str
    value: Any
    source: SourceType
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: EvidenceSpan
    extractor_version: str = Field(
        default="unset",
        description="e.g. 'rule@0.1.0', 'ner@bio-bert-v2', 'llm@claude-sonnet-5'. "
        "Required for reproducible experiment tracking (LLD section 23).",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "field": "temperature",
                    "value": {"value": 80, "unit": "C"},
                    "source": "RULE",
                    "confidence": 0.98,
                    "evidence": {
                        "paper_id": "P001",
                        "version": 1,
                        "page": 4,
                        "section": "Experimental",
                        "text": "heated at 80 \u00b0C for 2 h",
                        "char_start": 12,
                        "char_end": 34,
                    },
                    "extractor_version": "rule@0.1.0",
                }
            ]
        }
    }
