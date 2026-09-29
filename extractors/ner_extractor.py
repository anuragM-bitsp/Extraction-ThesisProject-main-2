"""
NerExtractor: the second of the four extraction strategies (LLD section 12).

Per block:
    text -> NerModel.predict() -> entities (PRECURSOR/SOLVENT/MATERIAL/CHARACTERIZATION)
                                        |
                       for each PRECURSOR entity: search the SAME block's
                       text for nearby ConcentrationRule/MassRule matches
                       and link them (linking.py) into one Precursor
                                        v
                             SynthesisExtraction fields + provenance

Reusing ConcentrationRule/MassRule here (rather than duplicating regex) is
deliberate: the *number-finding* logic is identical to Step 5's; what's new
in this extractor is entity recognition and the linking decision. All
CandidateFacts this extractor produces are tagged `source=NER` regardless —
`source` identifies which extraction *strategy* produced the fact for the
ablation study, not which regex happened to fire internally.
"""

from __future__ import annotations

from extractors.base import Extractor
from extractors.ner.base import NerModel
from extractors.ner.gazetteer import GazetteerNER
from extractors.ner.labels import EntityLabel
from extractors.ner.linking import find_nearest_match
from extractors.rules.concentration import ConcentrationRule
from extractors.rules.mass import MassRule
from ingestion.canonical import CanonicalDocument
from schemas.extraction_schema import (
    CharacterizationMethod,
    ExtractionResult,
    ExtractorName,
    Material,
    Precursor,
    Quantity,
    SynthesisExtraction,
)
from schemas.provenance import CandidateFact, EvidenceSpan, SourceType

NER_EXTRACTOR_VERSION = "ner@gazetteer-0.1.0"
LINKING_MAX_DISTANCE = 60  # characters; see extractors/ner/linking.py


class NerExtractor(Extractor):
    name = ExtractorName.NER
    version = NER_EXTRACTOR_VERSION

    def __init__(self, model: NerModel | None = None):
        self.model = model if model is not None else GazetteerNER()
        self._concentration_rule = ConcentrationRule()
        self._mass_rule = MassRule()

    def extract(self, document: CanonicalDocument) -> ExtractionResult:
        prediction = SynthesisExtraction(paper_id=document.paper_id)
        candidates: list[CandidateFact] = []
        seen_precursor_names: set[str] = set()
        seen_characterization: set[str] = set()

        for block in document.blocks:
            if not block.text:
                continue
            entities = self.model.predict(block.text)
            if not entities:
                continue

            concentration_matches = self._concentration_rule.apply(block.text)
            mass_matches = self._mass_rule.apply(block.text)

            for entity in entities:
                evidence = EvidenceSpan(
                    paper_id=document.paper_id,
                    version=document.version,
                    page=block.page_number,
                    section=block.section,
                    text=entity.text,
                    char_start=entity.char_start,
                    char_end=entity.char_end,
                )

                if entity.label == EntityLabel.MATERIAL:
                    self._handle_material(prediction, candidates, entity, evidence)
                elif entity.label == EntityLabel.SOLVENT:
                    self._handle_solvent(prediction, candidates, entity, evidence)
                elif entity.label == EntityLabel.CHARACTERIZATION:
                    self._handle_characterization(prediction, candidates, entity, evidence, seen_characterization)
                elif entity.label == EntityLabel.PRECURSOR:
                    self._handle_precursor(
                        prediction, candidates, entity, evidence, block,
                        concentration_matches, mass_matches, seen_precursor_names,
                    )

        return ExtractionResult(
            paper_id=document.paper_id,
            version=document.version,
            extractor=self.name,
            extractor_version=self.version,
            prediction=prediction,
            provenance=candidates,
        )

    # -- per-label handlers, each a documented simplification ---------------------

    def _handle_material(self, prediction, candidates, entity, evidence):
        # First material mention wins — a paper may name its product
        # several times; disambiguating "the" material vs. a comparison
        # material mentioned in passing is future work, same class of
        # limitation as RuleExtractor's "first occurrence wins" (Step 5).
        if prediction.material is not None:
            return
        prediction.material = Material(name=entity.text)
        candidates.append(self._fact("material.name", entity.text, evidence, entity.confidence))

    def _handle_solvent(self, prediction, candidates, entity, evidence):
        if prediction.solvent is not None:
            return
        prediction.solvent = entity.text
        candidates.append(self._fact("solvent", entity.text, evidence, entity.confidence))

    def _handle_characterization(self, prediction, candidates, entity, evidence, seen):
        key = entity.text.lower()
        if key in seen:
            return
        seen.add(key)
        prediction.characterization.append(CharacterizationMethod(technique=entity.text))
        idx = len(prediction.characterization) - 1
        candidates.append(self._fact(f"characterization[{idx}].technique", entity.text, evidence, entity.confidence))

    def _handle_precursor(
        self, prediction, candidates, entity, evidence, block,
        concentration_matches, mass_matches, seen_names,
    ):
        # Dedup by exact surface form, not canonical identity: "AgNO3" and
        # "silver nitrate" will each become their own Precursor entry even
        # though they're the same compound. Collapsing synonymous surface
        # forms into one canonical entity is Step 9's job (entity
        # normalization) — deliberately not attempted here.
        key = entity.text.lower()
        if key in seen_names:
            return
        seen_names.add(key)

        concentration_match = find_nearest_match(
            entity.char_start, entity.char_end, concentration_matches, LINKING_MAX_DISTANCE
        )
        amount_match = find_nearest_match(entity.char_start, entity.char_end, mass_matches, LINKING_MAX_DISTANCE)

        prediction.precursors.append(
            Precursor(
                name=entity.text,
                concentration=Quantity(**concentration_match.value) if concentration_match else None,
                amount=Quantity(**amount_match.value) if amount_match else None,
            )
        )
        idx = len(prediction.precursors) - 1
        candidates.append(self._fact(f"precursors[{idx}].name", entity.text, evidence, entity.confidence))

        for match, field_suffix in ((concentration_match, "concentration"), (amount_match, "amount")):
            if match is None:
                continue
            match_evidence = EvidenceSpan(
                paper_id=evidence.paper_id,
                version=evidence.version,
                page=block.page_number,
                section=block.section,
                text=match.text,
                char_start=match.char_start,
                char_end=match.char_end,
            )
            candidates.append(
                self._fact(f"precursors[{idx}].{field_suffix}", match.value, match_evidence, 0.85)
            )

    def _fact(self, field: str, value, evidence: EvidenceSpan, confidence: float) -> CandidateFact:
        return CandidateFact(
            field=field,
            value=value,
            source=SourceType.NER,
            confidence=confidence,
            evidence=evidence,
            extractor_version=self.version,
        )
