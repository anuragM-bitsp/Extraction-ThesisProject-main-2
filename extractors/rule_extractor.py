"""
RuleExtractor: the first of the four extraction strategies (LLD sections
7 and 11-14).

Deterministic, cheap, high precision, no external calls — the strength LLD
section 8 names for rules is exactly numeric/unit patterns, and that's all
this extractor claims to do well. It's not trying to identify precursors,
materials, or relationships between entities; that's what NER (Step 6) and
the Prompt+LLM extractor (Step 7) are for.
"""

from __future__ import annotations

from extractors.base import Extractor
from extractors.rules.base import ExtractionRule
from extractors.rules.concentration import ConcentrationRule
from extractors.rules.mass import MassRule
from extractors.rules.reaction_time import ReactionTimeRule
from extractors.rules.temperature import TemperatureRule
from extractors.rules.volume import VolumeRule
from ingestion.canonical import CanonicalDocument
from schemas.extraction_schema import ExtractionResult, ExtractorName, Quantity, SynthesisExtraction
from schemas.provenance import CandidateFact, EvidenceSpan, SourceType

RULE_EXTRACTOR_VERSION = "rule@0.1.0"

DEFAULT_RULES: list[ExtractionRule] = [
    TemperatureRule(),
    ReactionTimeRule(),
    ConcentrationRule(),
    VolumeRule(),
    MassRule(),
]

# Fields safe to write directly onto SynthesisExtraction: singular,
# document-level facts a rule match can populate with no ambiguity about
# *what it's a property of*. `concentration`/`volume`/`mass` are NOT in this
# set — "10 mM" is meaningless in the schema until we know which precursor
# it belongs to, which needs entity linking (Steps 6-9). Those rule matches
# still get returned as CandidateFacts (see extract() below), just not
# written into `prediction`.
DIRECT_SCHEMA_FIELDS = {"temperature", "reaction_time"}


class RuleExtractor(Extractor):
    name = ExtractorName.RULE
    version = RULE_EXTRACTOR_VERSION

    def __init__(self, rules: list[ExtractionRule] | None = None):
        self.rules = rules if rules is not None else DEFAULT_RULES

    def extract(self, document: CanonicalDocument) -> ExtractionResult:
        candidates: list[CandidateFact] = []

        for block in document.blocks:
            if not block.text:
                continue
            for rule in self.rules:
                for match in rule.apply(block.text):
                    evidence = EvidenceSpan(
                        paper_id=document.paper_id,
                        version=document.version,
                        page=block.page_number,
                        section=block.section,
                        text=match.text,
                        char_start=match.char_start,
                        char_end=match.char_end,
                    )
                    candidates.append(
                        CandidateFact(
                            field=match.field,
                            value=match.value,
                            source=SourceType.RULE,
                            confidence=match.confidence,
                            evidence=evidence,
                            extractor_version=self.version,
                        )
                    )

        prediction = SynthesisExtraction(paper_id=document.paper_id)
        for field in DIRECT_SCHEMA_FIELDS:
            # First occurrence in document (reading) order wins. Papers
            # typically state primary synthesis conditions once, in the
            # Experimental section; when a paper genuinely reports multiple
            # values (e.g. a temperature series), "first wins" is a known
            # simplification — proper handling belongs to the hybrid fusion
            # layer (Step 8), which has cross-source conflict-resolution
            # logic this single-strategy extractor intentionally doesn't.
            field_candidates = [c for c in candidates if c.field == field]
            if field_candidates:
                setattr(prediction, field, Quantity(**field_candidates[0].value))

        return ExtractionResult(
            paper_id=document.paper_id,
            version=document.version,
            extractor=self.name,
            extractor_version=self.version,
            prediction=prediction,
            provenance=candidates,
        )
