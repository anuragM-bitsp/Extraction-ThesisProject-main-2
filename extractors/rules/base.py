"""
ExtractionRule interface (LLD section 11: "rules/temperature.py, .../time.py, ...").

Each rule is a self-contained regex + unit-normalization unit, independently
testable and independently addable — the rule engine (rule_extractor.py)
just runs whatever list of rules it's given over every block's text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class RuleMatch(BaseModel):
    """One regex hit, already unit-normalized, with its span within the
    text it was found in (relative to `block.text`, not the whole page —
    the caller combines this with block metadata to build an EvidenceSpan)."""

    field: str
    value: dict  # e.g. {"value": 80.0, "unit": "C"}
    text: str  # the exact matched substring, used as evidence
    char_start: int
    char_end: int
    confidence: float = 0.95


class ExtractionRule(ABC):
    field: str

    @abstractmethod
    def apply(self, text: str) -> list[RuleMatch]: ...
