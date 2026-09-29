from __future__ import annotations

import json
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from annotation.adjudication import UnresolvedDisagreement, adjudicate, build_gold_result
from annotation.agreement import cohens_kappa, compute_agreement
from annotation.dataset import export_gold_jsonl, paper_level_split
from annotation.paths import flatten_payload, unflatten_payload
from schemas.extraction_schema import ExtractionResult, ExtractorName, Precursor, Quantity, SynthesisExtraction
from storage.models import Base
from storage.object_store import LocalObjectStore
from storage.repository import PaperRepository

# ---- flatten / unflatten -----------------------------------------------------------


def test_flatten_produces_expected_field_paths():
    payload = {
        "paper_id": "P001",
        "temperature": {"value": 80.0, "unit": "C"},
        "precursors": [{"name": "Silver nitrate", "concentration": {"value": 10.0, "unit": "mM"}, "amount": None}],
    }
    flat = flatten_payload(payload)

    assert "paper_id" not in flat  # excluded — not annotatable content
    assert flat["temperature.value"] == 80.0
    assert flat["temperature.unit"] == "C"
    assert flat["precursors[0].name"] == "Silver nitrate"
    assert flat["precursors[0].concentration.value"] == 10.0
    assert flat["precursors[0].amount"] is None


def test_unflatten_reconstructs_nested_structure():
    flat = {
        "temperature.value": 80.0,
        "temperature.unit": "C",
        "precursors[0].name": "Silver nitrate",
        "precursors[0].concentration.value": 10.0,
        "precursors[0].concentration.unit": "mM",
    }
    nested = unflatten_payload(flat)
    assert nested["temperature"] == {"value": 80.0, "unit": "C"}
    assert nested["precursors"][0]["name"] == "Silver nitrate"
    assert nested["precursors"][0]["concentration"] == {"value": 10.0, "unit": "mM"}


def test_flatten_unflatten_round_trips_a_real_synthesis_extraction():
    """The generic path machinery must handle everything a real extraction
    schema instance can contain, not just the toy cases above."""
    extraction = SynthesisExtraction(
        paper_id="P001",
        solvent="ethanol",
        temperature=Quantity(value=80, unit="C"),
        precursors=[
            Precursor(name="Silver nitrate", concentration=Quantity(value=10, unit="mM")),
            Precursor(name="Trisodium citrate", amount=Quantity(value=100, unit="mg")),
        ],
    )
    payload = extraction.model_dump()
    round_tripped = unflatten_payload(flatten_payload(payload))
    round_tripped["paper_id"] = "P001"  # excluded field, add back for comparison
    round_tripped["schema_version"] = payload["schema_version"]

    assert SynthesisExtraction.model_validate(round_tripped) == extraction


# ---- cohens_kappa ---------------------------------------------------------------------


def test_cohens_kappa_matches_hand_computed_value():
    a = [True, True, True, False, False]
    b = [True, True, False, False, False]
    # po = 4/5 = 0.8; pa1=0.6, pb1=0.4; pe = 0.6*0.4 + 0.4*0.6 = 0.48
    # kappa = (0.8 - 0.48) / (1 - 0.48) = 8/13
    assert cohens_kappa(a, b) == pytest.approx(8 / 13, rel=1e-9)


def test_cohens_kappa_perfect_agreement_is_one():
    assert cohens_kappa([True, False, True], [True, False, True]) == 1.0


def test_cohens_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cohens_kappa([True], [True, False])


# ---- compute_agreement ----------------------------------------------------------------


def test_compute_agreement_counts_agreements_and_disagreements():
    payload_a = {
        "paper_id": "P001", "solvent": "ethanol",
        "temperature": {"value": 80.0, "unit": "C"},
        "precursors": [{"name": "Silver nitrate", "concentration": {"value": 10.0, "unit": "mM"}, "amount": None}],
    }
    payload_b = {
        "paper_id": "P001", "solvent": "ethanol",  # agree
        "temperature": {"value": 80.0, "unit": "C"},  # agree
        "precursors": [{"name": "Silver nitrate", "concentration": {"value": 12.0, "unit": "mM"}, "amount": None}],  # disagree
    }
    report = compute_agreement(payload_a, payload_b)

    disagreement_fields = {d.field for d in report.disagreements}
    assert "precursors[0].concentration.value" in disagreement_fields
    assert "solvent" not in disagreement_fields
    assert 0.0 < report.agreement_rate < 1.0


def test_compute_agreement_full_agreement_gives_rate_one():
    payload = {"paper_id": "P001", "solvent": "ethanol"}
    report = compute_agreement(payload, dict(payload))
    assert report.agreement_rate == 1.0
    assert report.disagreements == []


# ---- adjudicate -----------------------------------------------------------------------


def test_adjudicate_keeps_agreed_fields_without_override():
    payload_a = {"paper_id": "P001", "solvent": "ethanol"}
    payload_b = {"paper_id": "P001", "solvent": "ethanol"}
    merged = adjudicate(payload_a, payload_b)
    assert merged["solvent"] == "ethanol"


def test_adjudicate_raises_listing_every_unresolved_field():
    payload_a = {"paper_id": "P001", "solvent": "ethanol", "synthesis_method": "reduction"}
    payload_b = {"paper_id": "P001", "solvent": "methanol", "synthesis_method": "precipitation"}

    with pytest.raises(UnresolvedDisagreement) as exc_info:
        adjudicate(payload_a, payload_b)
    assert set(exc_info.value.fields) == {"solvent", "synthesis_method"}


def test_adjudicate_applies_override_for_disagreement():
    payload_a = {"paper_id": "P001", "solvent": "ethanol"}
    payload_b = {"paper_id": "P001", "solvent": "methanol"}
    merged = adjudicate(payload_a, payload_b, overrides={"solvent": "ethanol"})
    assert merged["solvent"] == "ethanol"


# ---- build_gold_result ------------------------------------------------------------------


def test_build_gold_result_produces_valid_extraction_result_with_human_source():
    payload_a = {
        "paper_id": "P001",
        "temperature": {"value": 80.0, "unit": "C"},
        "precursors": [{"name": "Silver nitrate", "concentration": {"value": 10.0, "unit": "mM"}, "amount": None}],
    }
    payload_b = {
        "paper_id": "P001",
        "temperature": {"value": 80.0, "unit": "C"},
        "precursors": [{"name": "Silver nitrate", "concentration": {"value": 12.0, "unit": "mM"}, "amount": None}],
    }
    result = build_gold_result(
        paper_id="P001", version=1, payload_a=payload_a, payload_b=payload_b,
        adjudicated_by="lead_annotator",
        overrides={"precursors[0].concentration.value": 10.0},
    )

    assert isinstance(result, ExtractionResult)
    assert result.extractor == ExtractorName.HUMAN
    assert result.prediction.temperature.value == 80.0
    assert result.prediction.precursors[0].concentration.value == 10.0
    assert all(c.source.value == "HUMAN" for c in result.provenance)
    assert any(c.field == "temperature.value" for c in result.provenance)


def test_build_gold_result_propagates_unresolved_disagreement():
    payload_a = {"paper_id": "P001", "solvent": "ethanol"}
    payload_b = {"paper_id": "P001", "solvent": "methanol"}
    with pytest.raises(UnresolvedDisagreement):
        build_gold_result(paper_id="P001", version=1, payload_a=payload_a, payload_b=payload_b, adjudicated_by="x")


# ---- paper_level_split ------------------------------------------------------------------


def test_paper_level_split_covers_all_papers_with_no_overlap():
    paper_ids = [f"P{i:03d}" for i in range(20)]
    train, test = paper_level_split(paper_ids, test_fraction=0.25, seed=1)
    assert set(train) | set(test) == set(paper_ids)
    assert set(train) & set(test) == set()
    assert len(test) == 5


def test_paper_level_split_is_deterministic_given_same_seed():
    paper_ids = [f"P{i:03d}" for i in range(15)]
    train1, test1 = paper_level_split(paper_ids, seed=7)
    train2, test2 = paper_level_split(paper_ids, seed=7)
    assert train1 == train2
    assert test1 == test2


def test_paper_level_split_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        paper_level_split(["P001"], test_fraction=1.5)


# ---- export_gold_jsonl -------------------------------------------------------------------


def test_export_gold_jsonl_round_trips_through_json():
    result = build_gold_result(
        paper_id="P001", version=1,
        payload_a={"paper_id": "P001", "solvent": "ethanol"},
        payload_b={"paper_id": "P001", "solvent": "ethanol"},
        adjudicated_by="lead_annotator",
    )
    jsonl = export_gold_jsonl([result])
    lines = jsonl.strip().split("\n")
    assert len(lines) == 1
    restored = ExtractionResult.model_validate(json.loads(lines[0]))
    assert restored.prediction.solvent == "ethanol"


# ---- full pipeline: repository -> agreement -> adjudication -> gold storage --------------


def test_end_to_end_annotation_to_gold_via_repository():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with tempfile.TemporaryDirectory() as tmp, Session(engine) as session:
        repo = PaperRepository(session, LocalObjectStore(tmp))
        paper = repo.create_paper(title="P1")
        version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")

        payload_a = {
            "paper_id": str(paper.paper_id),
            "solvent": "ethanol",
            "temperature": {"value": 80.0, "unit": "C"},
        }
        payload_b = {
            "paper_id": str(paper.paper_id),
            "solvent": "ethanol",
            "temperature": {"value": 85.0, "unit": "C"},  # disagreement
        }
        ann_a = repo.submit_annotation(version.version_id, "annotator_a", payload_a)
        ann_b = repo.submit_annotation(version.version_id, "annotator_b", payload_b)

        report = compute_agreement(payload_a, payload_b)
        assert any(d.field == "temperature.value" for d in report.disagreements)

        gold_result = build_gold_result(
            paper_id=str(paper.paper_id), version=1,
            payload_a=payload_a, payload_b=payload_b,
            adjudicated_by="lead_annotator",
            overrides={"temperature.value": 80.0},
        )

        gold_record = repo.submit_gold(
            version.version_id,
            payload=gold_result.prediction.model_dump(),
            adjudicated_by="lead_annotator",
            source_annotation_ids=[str(ann_a.annotation_id), str(ann_b.annotation_id)],
        )

        fetched = repo.get_gold(version.version_id)
        assert fetched.payload["temperature"]["value"] == 80.0
        assert fetched.payload == gold_record.payload
