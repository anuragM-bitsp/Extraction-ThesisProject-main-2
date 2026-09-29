"""
Evaluator: one prediction, one gold, three levels of score (LLD section 22).

This is deliberately per-document. Corpus-level aggregation across many
papers (needed for the four-way ablation study) is `ablation.py`'s job —
summing many small, easily-tested per-document reports is far easier to
get right than a single function trying to do both at once.
"""

from __future__ import annotations

from pydantic import BaseModel

from evaluation.comparators import compare_scalar_fields, entity_prf1, field_prf1, relation_prf1
from evaluation.metrics import PRF1
from schemas.extraction_schema import ExtractionResult


class EvaluationReport(BaseModel):
    paper_id: str
    extractor: str
    entity: PRF1
    relation: PRF1
    field: PRF1
    scalar_field_statuses: dict[str, str]


def evaluate(prediction: ExtractionResult, gold: ExtractionResult) -> EvaluationReport:
    if prediction.paper_id != gold.paper_id:
        raise ValueError(
            f"prediction is for paper {prediction.paper_id!r} but gold is for {gold.paper_id!r} — "
            "refusing to score two different papers against each other"
        )

    statuses = compare_scalar_fields(prediction.prediction, gold.prediction)
    return EvaluationReport(
        paper_id=prediction.paper_id,
        extractor=prediction.extractor.value,
        entity=entity_prf1(prediction.prediction, gold.prediction),
        relation=relation_prf1(prediction.prediction, gold.prediction),
        field=field_prf1(statuses),
        scalar_field_statuses=statuses,
    )
