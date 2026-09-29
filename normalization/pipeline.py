"""
Post-extraction normalization (LLD sections 15/19).

Deliberately NOT tied to HybridExtractor — this is a transformation applied
to the OUTPUT of any of the four strategies. NER's or Prompt+LLM's raw output
can carry the exact same synonym-duplication problem on its own (a paper
naming both "AgNO3" and "silver nitrate" gives NER two Precursor entries
even with no fusion involved) — Step 6 and Step 8 both documented and
tested for this gap without fixing it. This module is where it finally
gets fixed, uniformly, for any extractor's output.
"""

from __future__ import annotations

from normalization.linking import OntologyLinker
from normalization.normalizer import EntityNormalizer
from schemas.extraction_schema import ExtractionResult, Precursor
from schemas.provenance import CandidateFact, EvidenceSpan, SourceType

NORMALIZATION_VERSION = "normalization@0.1.0"


def normalize_extraction(
    result: ExtractionResult,
    normalizer: EntityNormalizer,
    linker: OntologyLinker | None = None,
) -> ExtractionResult:
    prediction = result.prediction.model_copy(deep=True)

    prediction.precursors, precursor_facts = _normalize_precursors(
        prediction.precursors, normalizer, linker, result
    )

    if prediction.material is not None and prediction.material.name:
        normalized = normalizer.normalize(prediction.material.name)
        if normalized is not None:
            prediction.material.name = normalized.canonical_name
            prediction.material.canonical_id = normalized.canonical_id

    if prediction.solvent:
        normalized = normalizer.normalize(prediction.solvent)
        if normalized is not None:
            prediction.solvent = normalized.canonical_name

    return ExtractionResult(
        paper_id=result.paper_id,
        version=result.version,
        extractor=result.extractor,
        extractor_version=f"{result.extractor_version}+normalized",
        prediction=prediction,
        provenance=result.provenance + precursor_facts,
        llm_run=result.llm_run,  # keep the LLM's raw JSON/prompt attached after normalization
    )


def _normalize_precursors(
    precursors: list[Precursor],
    normalizer: EntityNormalizer,
    linker: OntologyLinker | None,
    result: ExtractionResult,
) -> tuple[list[Precursor], list[CandidateFact]]:
    merged: dict[str, Precursor] = {}
    order: list[str] = []
    facts: list[CandidateFact] = []
    linked_ids: set[str] = set()

    for precursor in precursors:
        normalized = normalizer.normalize(precursor.name)
        key = normalized.canonical_id if normalized else precursor.name.strip().lower()
        display_name = normalized.canonical_name if normalized else precursor.name
        canonical_id = normalized.canonical_id if normalized else None

        if key not in merged:
            merged[key] = Precursor(
                name=display_name,
                canonical_id=canonical_id,
                amount=precursor.amount,
                concentration=precursor.concentration,
            )
            order.append(key)
        else:
            existing = merged[key]
            # Backfill gaps from the duplicate rather than discarding it —
            # if NER's "AgNO3" mention had no linked amount but the LLM's
            # "silver nitrate" mention did, the merged entry should keep it.
            if existing.amount is None and precursor.amount is not None:
                existing.amount = precursor.amount
            if existing.concentration is None and precursor.concentration is not None:
                existing.concentration = precursor.concentration
            facts.append(
                CandidateFact(
                    field=f"precursors.normalized_merge.{key}",
                    value={"merged_surface_form": precursor.name, "canonical_name": display_name},
                    source=SourceType.HYBRID,
                    confidence=normalized.confidence if normalized else 0.5,
                    evidence=EvidenceSpan(
                        paper_id=result.paper_id,
                        version=result.version,
                        text=f"'{precursor.name}' normalized to canonical entity '{display_name}'",
                    ),
                    extractor_version=NORMALIZATION_VERSION,
                )
            )

        if linker is not None and canonical_id is not None and canonical_id not in linked_ids:
            linked_ids.add(canonical_id)
            external_id = linker.link(canonical_id)
            if external_id is not None:
                facts.append(
                    CandidateFact(
                        field=f"precursors.external_id.{key}",
                        value={"system": external_id.system, "identifier": external_id.identifier},
                        source=SourceType.HYBRID,
                        confidence=1.0,
                        evidence=EvidenceSpan(
                            paper_id=result.paper_id,
                            version=result.version,
                            text=f"Linked '{display_name}' to {external_id.system}:{external_id.identifier}",
                        ),
                        extractor_version=NORMALIZATION_VERSION,
                    )
                )

    return [merged[k] for k in order], facts
