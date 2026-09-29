"""
Comparison logic between one prediction and one gold `SynthesisExtraction`
(LLD sections 22-23) — the building block `evaluator.py` and
`error_analysis.py` both sit on top of.

Precursor matching uses `canonical_id` when present, falling back to
lowercased `name`. This means evaluation quality is directly coupled to
whether Step 9's normalization ran on the prediction being evaluated: an
un-normalized "AgNO3" prediction against a gold "Silver nitrate" entry will
count as a missed entity AND a spurious one, even though a human would
call that correct. Running `normalize_extraction()` (Step 9) before
`evaluate()` is a real prerequisite for fair scoring, not a nice-to-have —
stated here rather than left as a silent source of misleading numbers.

The "wrong value counts as both a false positive and a false negative"
convention below matches standard span-based NER evaluation practice: a
predicted value that disagrees with gold is simultaneously "gold's correct
value was missed" and "an incorrect value was spuriously asserted" — never
scored as a free pass just because *something* was predicted.
"""

from __future__ import annotations

from pydantic import BaseModel

from evaluation.metrics import PRF1
from evaluation.units import quantities_equal
from schemas.extraction_schema import Precursor, SynthesisExtraction

SCALAR_STRING_FIELDS = ["solvent", "synthesis_method"]
SCALAR_QUANTITY_FIELDS = ["temperature", "reaction_time"]


def _precursor_key(p: Precursor) -> str:
    return p.canonical_id if p.canonical_id else p.name.lower()


def _classify(pred_value, gold_value, equal_fn) -> str:
    if pred_value is None and gold_value is None:
        return "both_absent"
    if gold_value is None:
        return "spurious"
    if pred_value is None:
        return "missing"
    return "correct" if equal_fn(pred_value, gold_value) else "incorrect"


def scalar_value(extraction: SynthesisExtraction, field: str):
    """Pulls one scalar field's comparable value off a SynthesisExtraction,
    normalizing Quantity fields to plain dicts so callers never need to
    know which fields are Quantity-typed vs plain str/float."""
    if field == "material.name":
        return extraction.material.name if extraction.material else None
    if field in SCALAR_QUANTITY_FIELDS:
        q = getattr(extraction, field)
        return q.model_dump() if q is not None else None
    return getattr(extraction, field, None)


def compare_scalar_fields(prediction: SynthesisExtraction, gold: SynthesisExtraction) -> dict[str, str]:
    """Returns {field: status}, status one of
    'correct' | 'incorrect' | 'missing' | 'spurious' | 'both_absent'."""
    statuses: dict[str, str] = {}

    for field in SCALAR_STRING_FIELDS:
        statuses[field] = _classify(scalar_value(prediction, field), scalar_value(gold, field), lambda a, b: a == b)

    for field in SCALAR_QUANTITY_FIELDS:
        statuses[field] = _classify(scalar_value(prediction, field), scalar_value(gold, field), quantities_equal)

    statuses["material.name"] = _classify(
        scalar_value(prediction, "material.name"), scalar_value(gold, "material.name"), lambda a, b: a == b
    )

    if gold.pH is not None or prediction.pH is not None:
        statuses["pH"] = _classify(prediction.pH, gold.pH, lambda a, b: a == b)

    return statuses


class PrecursorMatch(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    matched: list[tuple[Precursor, Precursor]]  # (prediction, gold)
    unmatched_pred: list[Precursor]  # spurious entities
    unmatched_gold: list[Precursor]  # missed entities


def match_precursors(prediction: SynthesisExtraction, gold: SynthesisExtraction) -> PrecursorMatch:
    gold_by_key = {_precursor_key(p): p for p in gold.precursors}
    matched: list[tuple[Precursor, Precursor]] = []
    unmatched_pred: list[Precursor] = []
    used_keys: set[str] = set()

    for p in prediction.precursors:
        key = _precursor_key(p)
        if key in gold_by_key and key not in used_keys:
            matched.append((p, gold_by_key[key]))
            used_keys.add(key)
        else:
            unmatched_pred.append(p)

    unmatched_gold = [g for k, g in gold_by_key.items() if k not in used_keys]
    return PrecursorMatch(matched=matched, unmatched_pred=unmatched_pred, unmatched_gold=unmatched_gold)


def _entity_counts(pred_value: str | None, gold_value: str | None) -> tuple[int, int, int]:
    """(tp, fp, fn) for a single named, single-valued entity field
    (material name, solvent) — same wrong-value-counts-twice convention as
    the module docstring describes."""
    pred_key = pred_value.lower() if pred_value else None
    gold_key = gold_value.lower() if gold_value else None
    if pred_key is None and gold_key is None:
        return 0, 0, 0
    if pred_key == gold_key:
        return 1, 0, 0
    return 0, (1 if pred_key is not None else 0), (1 if gold_key is not None else 0)


def entity_prf1(prediction: SynthesisExtraction, gold: SynthesisExtraction) -> PRF1:
    """Entity-level: does the extraction identify the right named things —
    precursors (by canonical key), the material, the solvent, and
    characterization techniques — independent of whether attached
    attributes are ALSO correct. `relation_prf1` checks attributes."""
    match = match_precursors(prediction, gold)
    tp, fp, fn = len(match.matched), len(match.unmatched_pred), len(match.unmatched_gold)

    for pred_val, gold_val in (
        (prediction.material.name if prediction.material else None, gold.material.name if gold.material else None),
        (prediction.solvent, gold.solvent),
    ):
        t, f, n = _entity_counts(pred_val, gold_val)
        tp, fp, fn = tp + t, fp + f, fn + n

    pred_techniques = {c.technique.lower() for c in prediction.characterization}
    gold_techniques = {c.technique.lower() for c in gold.characterization}
    tp += len(pred_techniques & gold_techniques)
    fp += len(pred_techniques - gold_techniques)
    fn += len(gold_techniques - pred_techniques)

    return PRF1(tp=tp, fp=fp, fn=fn)


def relation_prf1(prediction: SynthesisExtraction, gold: SynthesisExtraction) -> PRF1:
    """A 'relation' here is (matched entity, attribute) -> value, e.g.
    (silver_nitrate, concentration) -> 10 mM (LLD section 17-19's whole
    point: an entity and a value are not useful separately). Only defined
    over MATCHED entity pairs — an attribute on an entity that's itself
    missing or spurious is already scored as an entity-level error and
    isn't double-counted here."""
    match = match_precursors(prediction, gold)
    tp = fp = fn = 0

    for pred_p, gold_p in match.matched:
        for attr in ("concentration", "amount"):
            pred_q = getattr(pred_p, attr)
            gold_q = getattr(gold_p, attr)
            pred_dict = pred_q.model_dump() if pred_q else None
            gold_dict = gold_q.model_dump() if gold_q else None

            if pred_dict is None and gold_dict is None:
                continue
            if gold_dict is not None and quantities_equal(pred_dict, gold_dict):
                tp += 1
                continue
            if gold_dict is not None:
                fn += 1
            if pred_dict is not None:
                fp += 1

    return PRF1(tp=tp, fp=fp, fn=fn)


def field_prf1(statuses: dict[str, str]) -> PRF1:
    tp = sum(1 for s in statuses.values() if s == "correct")
    fp = sum(1 for s in statuses.values() if s in ("incorrect", "spurious"))
    fn = sum(1 for s in statuses.values() if s in ("incorrect", "missing"))
    return PRF1(tp=tp, fp=fp, fn=fn)
