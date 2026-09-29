from __future__ import annotations

import pytest

from evaluation.ablation import ablation_table, aggregate_reports
from evaluation.comparators import (
    compare_scalar_fields,
    entity_prf1,
    field_prf1,
    match_precursors,
    relation_prf1,
)
from evaluation.error_analysis import ErrorType, analyze_errors
from evaluation.evaluator import evaluate
from evaluation.metrics import PRF1
from evaluation.units import quantities_equal, to_canonical
from schemas.extraction_schema import (
    CharacterizationMethod,
    ExtractionResult,
    ExtractorName,
    Material,
    Precursor,
    Quantity,
    SynthesisExtraction,
)

# ---- units --------------------------------------------------------------------------


def test_to_canonical_recognizes_each_dimension():
    assert to_canonical(80, "C")[1] == "temperature"
    assert to_canonical(2, "h")[1] == "time"
    assert to_canonical(10, "mM")[1] == "concentration"
    assert to_canonical(100, "mg")[1] == "mass"
    assert to_canonical(20, "mL")[1] == "volume"
    assert to_canonical(5, "bogus_unit") is None


def test_quantities_equal_across_different_units_same_dimension():
    assert quantities_equal({"value": 0.5, "unit": "M"}, {"value": 500.0, "unit": "mM"})
    assert quantities_equal({"value": 80.0, "unit": "C"}, {"value": 353.15, "unit": "K"})


def test_quantities_equal_false_across_different_dimensions():
    assert not quantities_equal({"value": 10.0, "unit": "mM"}, {"value": 10.0, "unit": "mg"})


def test_quantities_equal_false_when_either_is_none():
    assert not quantities_equal(None, {"value": 1.0, "unit": "M"})
    assert not quantities_equal({"value": 1.0, "unit": "M"}, None)
    assert not quantities_equal(None, None)


def test_quantities_equal_respects_tolerance():
    assert quantities_equal({"value": 80.0, "unit": "C"}, {"value": 80.0001, "unit": "C"})
    assert not quantities_equal({"value": 80.0, "unit": "C"}, {"value": 85.0, "unit": "C"})


# ---- PRF1 -------------------------------------------------------------------------------


def test_prf1_precision_recall_f1():
    m = PRF1(tp=8, fp=2, fn=2)
    assert m.precision == 0.8
    assert m.recall == 0.8
    assert m.f1 == pytest.approx(0.8)


def test_prf1_handles_zero_denominators():
    m = PRF1(tp=0, fp=0, fn=0)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_prf1_addition_sums_counts():
    total = PRF1(tp=1, fp=1, fn=1) + PRF1(tp=2, fp=0, fn=1)
    assert total == PRF1(tp=3, fp=1, fn=2)


# ---- comparators: scalar fields ----------------------------------------------------------


def test_compare_scalar_fields_classifies_every_status():
    prediction = SynthesisExtraction(
        paper_id="P001",
        solvent="ethanol",  # correct
        synthesis_method="chemical reduction",  # spurious (gold has none)
        temperature=Quantity(value=85, unit="C"),  # incorrect (gold says 80)
    )
    gold = SynthesisExtraction(
        paper_id="P001",
        solvent="ethanol",
        temperature=Quantity(value=80, unit="C"),
        reaction_time=Quantity(value=2, unit="h"),  # missing in prediction
    )
    statuses = compare_scalar_fields(prediction, gold)

    assert statuses["solvent"] == "correct"
    assert statuses["synthesis_method"] == "spurious"
    assert statuses["temperature"] == "incorrect"
    assert statuses["reaction_time"] == "missing"
    assert statuses["material.name"] == "both_absent"


def test_compare_scalar_fields_unit_aware_temperature():
    prediction = SynthesisExtraction(paper_id="P001", temperature=Quantity(value=353.15, unit="K"))
    gold = SynthesisExtraction(paper_id="P001", temperature=Quantity(value=80, unit="C"))
    assert compare_scalar_fields(prediction, gold)["temperature"] == "correct"


# ---- comparators: precursor matching ------------------------------------------------------


def test_match_precursors_by_canonical_id_when_present():
    prediction = SynthesisExtraction(
        paper_id="P001", precursors=[Precursor(name="AgNO3", canonical_id="silver_nitrate")]
    )
    gold = SynthesisExtraction(
        paper_id="P001", precursors=[Precursor(name="Silver nitrate", canonical_id="silver_nitrate")]
    )
    match = match_precursors(prediction, gold)
    assert len(match.matched) == 1
    assert match.unmatched_pred == []
    assert match.unmatched_gold == []


def test_match_precursors_falls_back_to_name_without_canonical_id():
    prediction = SynthesisExtraction(paper_id="P001", precursors=[Precursor(name="Silver nitrate")])
    gold = SynthesisExtraction(paper_id="P001", precursors=[Precursor(name="silver nitrate")])  # case differs
    match = match_precursors(prediction, gold)
    assert len(match.matched) == 1


def test_match_precursors_without_normalization_treats_synonyms_as_unmatched():
    """Documented consequence of skipping Step 9: 'AgNO3' vs 'Silver
    nitrate' with no canonical_id set counts as one missing + one spurious
    entity, not a match. Evaluating un-normalized predictions is a real,
    not theoretical, way to understate an extractor's true performance."""
    prediction = SynthesisExtraction(paper_id="P001", precursors=[Precursor(name="AgNO3")])
    gold = SynthesisExtraction(paper_id="P001", precursors=[Precursor(name="Silver nitrate")])
    match = match_precursors(prediction, gold)
    assert match.matched == []
    assert len(match.unmatched_pred) == 1
    assert len(match.unmatched_gold) == 1


# ---- entity / relation / field PRF1 --------------------------------------------------------


def test_entity_prf1_counts_matched_missing_and_spurious():
    prediction = SynthesisExtraction(
        paper_id="P001",
        precursors=[Precursor(name="Silver nitrate", canonical_id="silver_nitrate"), Precursor(name="Extra reagent")],
    )
    gold = SynthesisExtraction(
        paper_id="P001",
        precursors=[Precursor(name="Silver nitrate", canonical_id="silver_nitrate"), Precursor(name="Trisodium citrate")],
    )
    metrics = entity_prf1(prediction, gold)
    assert metrics.tp == 1  # Silver nitrate matched
    assert metrics.fp == 1  # Extra reagent spurious
    assert metrics.fn == 1  # Trisodium citrate missed


def test_entity_prf1_includes_material_solvent_and_characterization():
    prediction = SynthesisExtraction(
        paper_id="P001",
        material=Material(name="Silver nanoparticles"),
        solvent="ethanol",
        characterization=[CharacterizationMethod(technique="TEM"), CharacterizationMethod(technique="XRD")],
    )
    gold = SynthesisExtraction(
        paper_id="P001",
        material=Material(name="Silver nanoparticles"),
        solvent="methanol",  # wrong
        characterization=[CharacterizationMethod(technique="TEM")],
    )
    metrics = entity_prf1(prediction, gold)
    # material: tp+1; solvent: wrong -> fp+1, fn+1; characterization: TEM tp+1, XRD fp+1
    assert metrics.tp == 2
    assert metrics.fp == 2
    assert metrics.fn == 1


def test_relation_prf1_only_scores_matched_entities():
    prediction = SynthesisExtraction(
        paper_id="P001",
        precursors=[
            Precursor(name="Silver nitrate", canonical_id="silver_nitrate", concentration=Quantity(value=12, unit="mM")),
            Precursor(name="Unmatched reagent", concentration=Quantity(value=5, unit="mM")),
        ],
    )
    gold = SynthesisExtraction(
        paper_id="P001",
        precursors=[
            Precursor(name="Silver nitrate", canonical_id="silver_nitrate", concentration=Quantity(value=10, unit="mM")),
        ],
    )
    metrics = relation_prf1(prediction, gold)
    # Only the matched "Silver nitrate" pair's concentration is scored (wrong: 12 vs 10 -> fp+fn).
    # The unmatched "Unmatched reagent" entity's concentration is NOT counted here —
    # that entity itself is already an entity-level error.
    assert metrics.tp == 0
    assert metrics.fp == 1
    assert metrics.fn == 1


def test_relation_prf1_correct_when_unit_aware_match():
    prediction = SynthesisExtraction(
        paper_id="P001",
        precursors=[Precursor(name="Silver nitrate", canonical_id="silver_nitrate", concentration=Quantity(value=0.01, unit="M"))],
    )
    gold = SynthesisExtraction(
        paper_id="P001",
        precursors=[Precursor(name="Silver nitrate", canonical_id="silver_nitrate", concentration=Quantity(value=10, unit="mM"))],
    )
    metrics = relation_prf1(prediction, gold)
    assert metrics.tp == 1
    assert metrics.fp == 0
    assert metrics.fn == 0


def test_field_prf1_from_statuses():
    statuses = {"solvent": "correct", "temperature": "incorrect", "reaction_time": "missing", "pH": "spurious"}
    metrics = field_prf1(statuses)
    assert metrics.tp == 1
    assert metrics.fp == 2  # incorrect + spurious
    assert metrics.fn == 2  # incorrect + missing


# ---- evaluator ------------------------------------------------------------------------------


def _result(paper_id, extractor, prediction) -> ExtractionResult:
    return ExtractionResult(paper_id=paper_id, version=1, extractor=extractor, extractor_version="test@0", prediction=prediction)


def test_evaluate_produces_full_report():
    prediction = _result(
        "P001", ExtractorName.RULE,
        SynthesisExtraction(paper_id="P001", temperature=Quantity(value=80, unit="C")),
    )
    gold = _result(
        "P001", ExtractorName.HUMAN,
        SynthesisExtraction(paper_id="P001", temperature=Quantity(value=80, unit="C"), solvent="ethanol"),
    )
    report = evaluate(prediction, gold)
    assert report.extractor == "RULE"
    assert report.field.tp == 1  # temperature correct
    assert report.field.fn == 1  # solvent missing


def test_evaluate_rejects_mismatched_paper_ids():
    prediction = _result("P001", ExtractorName.RULE, SynthesisExtraction(paper_id="P001"))
    gold = _result("P002", ExtractorName.HUMAN, SynthesisExtraction(paper_id="P002"))
    with pytest.raises(ValueError):
        evaluate(prediction, gold)


# ---- ablation --------------------------------------------------------------------------------


def test_aggregate_reports_sums_across_documents_for_same_extractor():
    gold1 = _result("P001", ExtractorName.HUMAN, SynthesisExtraction(paper_id="P001", solvent="ethanol"))
    gold2 = _result("P002", ExtractorName.HUMAN, SynthesisExtraction(paper_id="P002", solvent="methanol"))
    pred1 = _result("P001", ExtractorName.RULE, SynthesisExtraction(paper_id="P001", solvent="ethanol"))
    pred2 = _result("P002", ExtractorName.RULE, SynthesisExtraction(paper_id="P002", solvent=None))

    reports = [evaluate(pred1, gold1), evaluate(pred2, gold2)]
    aggregated = aggregate_reports(reports)

    assert aggregated["RULE:field"].tp == 1  # P001 solvent correct
    assert aggregated["RULE:field"].fn == 1  # P002 solvent missing


def test_ablation_table_has_one_row_per_extractor_and_level():
    gold = _result("P001", ExtractorName.HUMAN, SynthesisExtraction(paper_id="P001", solvent="ethanol"))
    pred = _result("P001", ExtractorName.NER, SynthesisExtraction(paper_id="P001", solvent="ethanol"))
    table = ablation_table([evaluate(pred, gold)])

    assert len(table) == 3  # entity, relation, field — all for NER
    assert all(row["extractor"] == "NER" for row in table)
    field_row = next(r for r in table if r["level"] == "field")
    assert field_row["precision"] == 1.0


# ---- error analysis --------------------------------------------------------------------------


def test_analyze_errors_labels_every_failure_type():
    prediction = SynthesisExtraction(
        paper_id="P001",
        solvent="ethanol",  # correct, no error
        synthesis_method="precipitation",  # spurious
        temperature=Quantity(value=85, unit="C"),  # wrong value
        precursors=[
            Precursor(name="Silver nitrate", canonical_id="silver_nitrate", concentration=Quantity(value=12, unit="mM")),
            Precursor(name="Spurious reagent"),
        ],
    )
    gold = SynthesisExtraction(
        paper_id="P001",
        solvent="ethanol",
        temperature=Quantity(value=80, unit="C"),
        reaction_time=Quantity(value=2, unit="h"),  # missing in prediction
        precursors=[
            Precursor(name="Silver nitrate", canonical_id="silver_nitrate", concentration=Quantity(value=10, unit="mM")),
            Precursor(name="Trisodium citrate"),
        ],
    )
    prediction_result = _result("P001", ExtractorName.HYBRID, prediction)
    gold_result = _result("P001", ExtractorName.HUMAN, gold)

    errors = analyze_errors(prediction_result, gold_result)
    error_types = {e.error_type for e in errors}

    assert ErrorType.MISSING_FIELD in error_types  # reaction_time
    assert ErrorType.SPURIOUS_FIELD in error_types  # synthesis_method
    assert ErrorType.WRONG_VALUE in error_types  # temperature
    assert ErrorType.MISSING_ENTITY in error_types  # Trisodium citrate
    assert ErrorType.SPURIOUS_ENTITY in error_types  # Spurious reagent
    assert ErrorType.WRONG_RELATION in error_types  # concentration 12 vs 10


def test_analyze_errors_empty_when_extraction_is_perfect():
    prediction = SynthesisExtraction(paper_id="P001", solvent="ethanol")
    gold = SynthesisExtraction(paper_id="P001", solvent="ethanol")
    errors = analyze_errors(
        _result("P001", ExtractorName.RULE, prediction), _result("P001", ExtractorName.HUMAN, gold)
    )
    assert errors == []
