import json

import pytest
from pydantic import ValidationError

from schemas.extraction_schema import (
    ExtractionResult,
    ExtractorName,
    Precursor,
    Quantity,
    SynthesisExtraction,
)
from schemas.provenance import CandidateFact, EvidenceSpan, SourceType


def _sample_evidence(**overrides) -> EvidenceSpan:
    base = dict(
        paper_id="P001",
        version=1,
        page=4,
        section="Experimental",
        text="heated at 80 \u00b0C for 2 h",
    )
    base.update(overrides)
    return EvidenceSpan(**base)


def test_minimal_extraction_is_valid_with_everything_optional():
    """A paper where nothing was found yet should still be a valid instance —
    extractors run incrementally, field by field (LLD section 14)."""
    result = SynthesisExtraction(paper_id="P001")
    assert result.schema_version == "v2"  # bumped in Step 9 when canonical_id was added
    assert result.precursors == []


def test_full_extraction_round_trips_through_json():
    extraction = SynthesisExtraction(
        paper_id="P001",
        solvent="Ethanol",
        temperature=Quantity(value=80, unit="C"),
        reaction_time=Quantity(value=2, unit="h"),
        precursors=[
            Precursor(
                name="Silver nitrate",
                concentration=Quantity(value=10, unit="mM"),
            )
        ],
    )
    dumped = extraction.model_dump_json()
    restored = SynthesisExtraction.model_validate_json(dumped)
    assert restored == extraction


def test_evidence_span_rejects_inverted_char_range():
    with pytest.raises(ValidationError):
        _sample_evidence(char_start=50, char_end=10)


def test_candidate_fact_confidence_must_be_0_to_1():
    with pytest.raises(ValidationError):
        CandidateFact(
            field="temperature",
            value={"value": 80, "unit": "C"},
            source=SourceType.RULE,
            confidence=1.5,  # invalid
            evidence=_sample_evidence(),
        )


def test_extraction_result_links_prediction_to_provenance():
    """This is the shape every extractor (rule/NER/Prompt+LLM/hybrid) must
    return — a SynthesisExtraction plus one CandidateFact per populated
    field, so the fusion layer and evaluator can trace every value back
    to its source (LLD section 17)."""
    prediction = SynthesisExtraction(
        paper_id="P001",
        temperature=Quantity(value=80, unit="C"),
    )
    provenance = [
        CandidateFact(
            field="temperature",
            value={"value": 80, "unit": "C"},
            source=SourceType.RULE,
            confidence=0.98,
            evidence=_sample_evidence(),
            extractor_version="rule@0.1.0",
        )
    ]
    result = ExtractionResult(
        paper_id="P001",
        version=1,
        extractor=ExtractorName.RULE,
        extractor_version="rule@0.1.0",
        prediction=prediction,
        provenance=provenance,
    )
    assert result.provenance[0].field == "temperature"
    assert result.prediction.temperature.value == 80


def test_two_extractors_can_produce_disagreeing_results_same_schema():
    """Rule says 80C, LLM says 85C — both are still valid SynthesisExtraction
    instances. Resolving the disagreement is the fusion layer's job (a later
    step), not the schema's job. The schema's only job is to make this
    disagreement comparable in the first place."""
    rule_pred = SynthesisExtraction(paper_id="P001", temperature=Quantity(value=80, unit="C"))
    llm_pred = SynthesisExtraction(paper_id="P001", temperature=Quantity(value=85, unit="C"))
    assert rule_pred.temperature.value != llm_pred.temperature.value
    assert rule_pred.schema_version == llm_pred.schema_version  # same schema, comparable


def test_json_schema_export_for_llm_structured_output():
    """The LLM extractor (step 7) can hand this JSON schema to the model
    for constrained generation (LLD section 15/12). Prove it exports cleanly
    now so nothing downstream breaks on a schema that doesn't serialize."""
    schema = SynthesisExtraction.model_json_schema()
    assert schema["title"] == "SynthesisExtraction"
    # must be JSON-serializable as-is
    json.dumps(schema)
