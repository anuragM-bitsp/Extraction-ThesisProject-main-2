"""
Adjudication (LLD doc 2 section 20-21).

Deliberately NOT automatic conflict resolution the way Step 8's
ConflictResolver is. Step 8 ranks EXTRACTION STRATEGIES against a prior
belief about their relative reliability (rules are precise for numbers,
etc.) — that prior doesn't exist for two human annotators; there's no
general reason to trust "annotator A" over "annotator B" by default. Real
adjudication is a human decision (discussion, or a third annotator), and
this module supports that process rather than pretending to automate it
away: `adjudicate()` identifies exactly which fields need a human decision
and requires the caller to supply one for each, rather than guessing.
"""

from __future__ import annotations

from datetime import datetime, timezone

from annotation.paths import flatten_payload, unflatten_payload
from schemas.extraction_schema import ExtractionResult, ExtractorName, SynthesisExtraction
from schemas.provenance import CandidateFact, EvidenceSpan, SourceType

GOLD_SOURCE_VERSION_PREFIX = "gold"


class UnresolvedDisagreement(RuntimeError):
    """Raised when annotators disagree on one or more fields and no
    override was supplied for at least one of them. Lists every such field
    at once rather than one at a time, so a human adjudicator can resolve
    them all in a single pass instead of hitting this error repeatedly."""

    def __init__(self, fields: list[str]):
        self.fields = fields
        super().__init__(f"unresolved disagreement on field(s): {fields}")


def adjudicate(payload_a: dict, payload_b: dict, overrides: dict[str, object] | None = None) -> dict:
    """
    Merge two annotators' payloads into one, field by field:
      - fields where both agree (including "neither annotated it"): kept as-is
      - fields where they disagree: require `overrides[field]`, or raise
        UnresolvedDisagreement listing every unresolved field

    Returns a plain nested dict, not yet schema-validated — see
    `build_gold_result` for the validated `ExtractionResult` wrapper.
    """
    overrides = overrides or {}
    flat_a = flatten_payload(payload_a)
    flat_b = flatten_payload(payload_b)
    all_fields = sorted(set(flat_a) | set(flat_b))

    resolved: dict[str, object] = {}
    unresolved: list[str] = []

    for field in all_fields:
        value_a = flat_a.get(field)
        value_b = flat_b.get(field)
        if field in overrides:
            resolved[field] = overrides[field]
        elif value_a == value_b:
            resolved[field] = value_a
        else:
            unresolved.append(field)

    if unresolved:
        raise UnresolvedDisagreement(unresolved)

    return unflatten_payload(resolved)


def build_gold_result(
    paper_id: str,
    version: int,
    payload_a: dict,
    payload_b: dict,
    adjudicated_by: str,
    overrides: dict[str, object] | None = None,
) -> ExtractionResult:
    """
    Runs `adjudicate()`, validates the merged payload against
    `SynthesisExtraction` (so a gold record is exactly as schema-conformant
    as any extractor's prediction — same guarantee, same comparability in
    Step 11), and wraps it as an `ExtractionResult` with
    `extractor=ExtractorName.HUMAN`, generating one `CandidateFact` per
    resolved field so gold data has the same provenance-first shape as
    every extractor's output.
    """
    merged_flat = flatten_payload(adjudicate(payload_a, payload_b, overrides))
    prediction = SynthesisExtraction.model_validate(unflatten_payload({**merged_flat, "paper_id": paper_id}))

    version_string = f"{GOLD_SOURCE_VERSION_PREFIX}@{adjudicated_by}"
    now = datetime.now(timezone.utc)
    provenance = [
        CandidateFact(
            field=field,
            value=value,
            source=SourceType.HUMAN,
            confidence=1.0,
            evidence=EvidenceSpan(paper_id=paper_id, version=version, text="human annotation (adjudicated)"),
            extractor_version=version_string,
            created_at=now,
        )
        for field, value in merged_flat.items()
        if value is not None
    ]

    return ExtractionResult(
        paper_id=paper_id,
        version=version,
        extractor=ExtractorName.HUMAN,
        extractor_version=version_string,
        prediction=prediction,
        provenance=provenance,
    )
