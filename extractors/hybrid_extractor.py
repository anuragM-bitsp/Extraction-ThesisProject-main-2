"""
HybridExtractor: the fourth and final extraction strategy (LLD sections
16-18).

    "Don't simply: hybrid = rule + ner + llm... For example, I would rely
    heavily on rules for units and numerical conditions, use NER for entity
    detection, and use the LLM for semantic relations and information that
    requires broader context. A fusion layer would retain provenance and
    confidence and resolve conflicts between systems."

The LLM component is the Prompt + LLM extractor (`PromptLlmExtractor`,
full document + editable prompt), which replaced the earlier RAG + LLM
extractor; fusion logic is unchanged.

This runs all three prior strategies against the same document, then merges
their outputs field-by-field via `ConflictResolver` rather than
concatenating whatever each one found.

Two things this fusion step explicitly does NOT do, both by design:
  - Merge synonymous precursor names ("AgNO3" vs "silver nitrate"). Matching
    is exact (case-insensitive) surface-form only — canonicalizing
    synonyms is Step 9's job (entity normalization). Tested directly so
    it's documented behavior, not a silently-discovered gap.
  - Reconcile unit-mismatched numeric conflicts ("80 C" vs "353 K") as
    equivalent. That's evaluation-time unit normalization (Step 11), not
    fusion.
"""

from __future__ import annotations

from extractors.base import Extractor
from extractors.fusion.conflict_resolver import ConflictResolver
from extractors.ner_extractor import NerExtractor
from extractors.llm_extractor import PromptLlmExtractor
from extractors.rule_extractor import RuleExtractor
from ingestion.canonical import CanonicalDocument
from schemas.extraction_schema import (
    ExtractionResult,
    ExtractorName,
    Material,
    Precursor,
    Quantity,
    SynthesisExtraction,
)
from schemas.provenance import CandidateFact

# Fields that are a single Quantity {value, unit} rather than a bare string.
QUANTITY_SCALAR_FIELDS = {"temperature", "reaction_time"}
# Fields that are a bare string on SynthesisExtraction.
STRING_SCALAR_FIELDS = {"solvent", "synthesis_method"}
# List-valued fields carried through as-is from whichever source populated
# them first, in priority order — currently only ever populated by one
# strategy each (steps/properties by Prompt+LLM), so there's no real conflict
# to resolve yet; the passthrough exists so adding a second source later
# (e.g. an LLM "steps" rewrite of NER's characterization) doesn't require
# restructuring this extractor.
PASSTHROUGH_LIST_FIELDS = ["steps", "properties"]


class HybridExtractor(Extractor):
    name = ExtractorName.HYBRID

    def __init__(
        self,
        rule_extractor: RuleExtractor | None = None,
        ner_extractor: NerExtractor | None = None,
        llm_extractor: PromptLlmExtractor | None = None,
        resolver: ConflictResolver | None = None,
    ):
        if llm_extractor is None:
            raise ValueError("HybridExtractor requires a configured PromptLlmExtractor (needs an LLM client)")
        self.rule_extractor = rule_extractor or RuleExtractor()
        self.ner_extractor = ner_extractor or NerExtractor()
        self.llm_extractor = llm_extractor
        self.resolver = resolver or ConflictResolver()
        self.version = (
            f"hybrid@{self.rule_extractor.version}"
            f"+{self.ner_extractor.version}"
            f"+{self.llm_extractor.version}"
        )

    def extract(self, document: CanonicalDocument) -> ExtractionResult:
        rule_result = self.rule_extractor.extract(document)
        ner_result = self.ner_extractor.extract(document)
        llm_result = self.llm_extractor.extract(document)
        sub_results = [rule_result, ner_result, llm_result]

        all_candidates: list[CandidateFact] = (
            rule_result.provenance + ner_result.provenance + llm_result.provenance
        )

        prediction = SynthesisExtraction(paper_id=document.paper_id)
        fused: list[CandidateFact] = []

        self._fuse_quantity_scalars(all_candidates, prediction, fused)
        self._fuse_string_scalars(all_candidates, prediction, fused)
        self._fuse_material(all_candidates, prediction, fused)
        self._fuse_characterization(sub_results, prediction, fused)
        self._fuse_precursors(sub_results, prediction, fused)
        self._fuse_passthrough_lists(sub_results, prediction, fused)

        return ExtractionResult(
            paper_id=document.paper_id,
            version=document.version,
            extractor=self.name,
            extractor_version=self.version,
            prediction=prediction,
            provenance=fused,
            llm_run=llm_result.llm_run,
        )

    # -- scalar fields --------------------------------------------------------------

    def _fuse_quantity_scalars(self, all_candidates, prediction, fused):
        for field in QUANTITY_SCALAR_FIELDS:
            chosen = self.resolver.resolve(field, [c for c in all_candidates if c.field == field])
            if chosen is not None:
                setattr(prediction, field, Quantity(**chosen.value))
                fused.append(chosen)

    def _fuse_string_scalars(self, all_candidates, prediction, fused):
        for field in STRING_SCALAR_FIELDS:
            chosen = self.resolver.resolve(field, [c for c in all_candidates if c.field == field])
            if chosen is not None:
                setattr(prediction, field, chosen.value)
                fused.append(chosen)

    def _fuse_material(self, all_candidates, prediction, fused):
        chosen = self.resolver.resolve("material.name", [c for c in all_candidates if c.field == "material.name"])
        if chosen is not None:
            prediction.material = Material(name=chosen.value)
            fused.append(chosen)

    # -- list fields ------------------------------------------------------------------

    def _fuse_characterization(self, sub_results, prediction, fused):
        seen: set[str] = set()
        for result in sub_results:
            for idx, technique in enumerate(result.prediction.characterization):
                key = technique.technique.lower()
                if key in seen:
                    continue
                seen.add(key)
                prediction.characterization.append(technique)
                fact = next((c for c in result.provenance if c.field == f"characterization[{idx}].technique"), None)
                if fact is not None:
                    fused.append(fact)

    def _fuse_precursors(self, sub_results, prediction, fused):
        # Group by exact (case-insensitive) name across sources — NOT by
        # canonical chemical identity. "AgNO3" and "silver nitrate" get
        # separate entries even though they're the same compound; see the
        # module docstring and Step 9's role in fixing this.
        entries: dict[str, dict[str, list[CandidateFact]]] = {}

        for result in sub_results:
            for idx, precursor in enumerate(result.prediction.precursors):
                key = precursor.name.lower()
                entry = entries.setdefault(key, {"name": [], "concentration": [], "amount": []})
                for suffix in ("name", "concentration", "amount"):
                    fact = next(
                        (c for c in result.provenance if c.field == f"precursors[{idx}].{suffix}"), None
                    )
                    if fact is not None:
                        entry[suffix].append(fact)

        for entry in entries.values():
            name_fact = self.resolver.resolve("precursors.name", entry["name"])
            if name_fact is None:
                continue  # shouldn't happen (every precursor has a name candidate), but stay defensive
            concentration_fact = self.resolver.resolve("concentration", entry["concentration"])
            amount_fact = self.resolver.resolve("amount", entry["amount"])

            prediction.precursors.append(
                Precursor(
                    name=name_fact.value,
                    concentration=Quantity(**concentration_fact.value) if concentration_fact else None,
                    amount=Quantity(**amount_fact.value) if amount_fact else None,
                )
            )
            fused.append(name_fact)
            if concentration_fact is not None:
                fused.append(concentration_fact)
            if amount_fact is not None:
                fused.append(amount_fact)

    def _fuse_passthrough_lists(self, sub_results, prediction, fused):
        for field in PASSTHROUGH_LIST_FIELDS:
            for result in sub_results:
                values = getattr(result.prediction, field)
                if not values:
                    continue
                setattr(prediction, field, values)
                prefix = f"{field}["
                fused.extend(c for c in result.provenance if c.field.startswith(prefix))
                break  # first non-empty source wins; see PASSTHROUGH_LIST_FIELDS docstring
