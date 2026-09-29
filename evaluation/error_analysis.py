"""
Error taxonomy (LLD section 26): "two systems with similar F1 scores can
fail in very different ways... After computing aggregate metrics, I would
perform qualitative error analysis."

`analyze_errors` walks the same comparison logic `evaluator.py` uses, but
instead of just counting into tp/fp/fn, it returns one labeled
`ErrorRecord` per individual failure — enough to answer "which specific
fields does this extractor get wrong, and how" rather than only "what's
its F1."

Scoped to the extraction-level error categories `comparators.py` can
actually distinguish (missing/spurious/wrong field or entity, wrong
relation). LLD section 26 also lists PDF-parsing and retrieval failures as
categories — those live upstream of what an ExtractionResult can express
on its own (they'd need Step 3's ingestion pipeline or Step 4/7's retrieval
step in scope, not just the final prediction/gold pair), so they aren't
reconstructable from this module's inputs and aren't claimed here.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from evaluation.comparators import compare_scalar_fields, match_precursors, scalar_value
from evaluation.units import quantities_equal
from schemas.extraction_schema import ExtractionResult


class ErrorType(str, Enum):
    MISSING_FIELD = "MISSING_FIELD"      # gold has a value; prediction has none
    SPURIOUS_FIELD = "SPURIOUS_FIELD"    # prediction has a value; gold has none
    WRONG_VALUE = "WRONG_VALUE"          # both have a value; they disagree
    MISSING_ENTITY = "MISSING_ENTITY"    # a gold precursor was never predicted
    SPURIOUS_ENTITY = "SPURIOUS_ENTITY"  # a predicted precursor isn't in gold
    WRONG_RELATION = "WRONG_RELATION"    # entity matched, but its attached quantity is wrong


class ErrorRecord(BaseModel):
    error_type: ErrorType
    field: str
    predicted_value: object = None
    gold_value: object = None


def analyze_errors(prediction: ExtractionResult, gold: ExtractionResult) -> list[ErrorRecord]:
    pred_ex, gold_ex = prediction.prediction, gold.prediction
    errors: list[ErrorRecord] = []

    statuses = compare_scalar_fields(pred_ex, gold_ex)
    for field, status in statuses.items():
        if status in ("correct", "both_absent"):
            continue
        pred_val, gold_val = scalar_value(pred_ex, field), scalar_value(gold_ex, field)
        if status == "missing":
            errors.append(ErrorRecord(error_type=ErrorType.MISSING_FIELD, field=field, gold_value=gold_val))
        elif status == "spurious":
            errors.append(ErrorRecord(error_type=ErrorType.SPURIOUS_FIELD, field=field, predicted_value=pred_val))
        elif status == "incorrect":
            errors.append(
                ErrorRecord(error_type=ErrorType.WRONG_VALUE, field=field, predicted_value=pred_val, gold_value=gold_val)
            )

    match = match_precursors(pred_ex, gold_ex)
    for gold_p in match.unmatched_gold:
        errors.append(ErrorRecord(error_type=ErrorType.MISSING_ENTITY, field="precursors", gold_value=gold_p.name))
    for pred_p in match.unmatched_pred:
        errors.append(ErrorRecord(error_type=ErrorType.SPURIOUS_ENTITY, field="precursors", predicted_value=pred_p.name))

    for pred_p, gold_p in match.matched:
        for attr in ("concentration", "amount"):
            pred_q, gold_q = getattr(pred_p, attr), getattr(gold_p, attr)
            pred_dict = pred_q.model_dump() if pred_q else None
            gold_dict = gold_q.model_dump() if gold_q else None
            if gold_dict is None:
                continue
            if pred_dict is None or not quantities_equal(pred_dict, gold_dict):
                errors.append(
                    ErrorRecord(
                        error_type=ErrorType.WRONG_RELATION,
                        field=f"precursors[{gold_p.name}].{attr}",
                        predicted_value=pred_dict,
                        gold_value=gold_dict,
                    )
                )

    return errors
