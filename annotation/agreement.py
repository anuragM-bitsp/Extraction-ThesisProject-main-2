"""
Inter-annotator agreement (LLD doc 2 section 20).

Two levels of agreement, both worth looking at:
  - Value agreement: for a field BOTH annotators marked, did they record
    the same value? (`AgreementReport.agreement_rate`)
  - Presence agreement: did the two annotators even notice the same
    information existed at all, independent of whether they transcribed
    the value identically? (`AgreementReport.presence_kappa`) A paper's
    Discussion section might mention a property one annotator flags and
    the other skips as "not really synthesis data" — that's a different,
    and arguably more useful, disagreement signal than a typo-level value
    mismatch.

Deliberately exact-match only for value comparison — no unit-aware
tolerance here. That's Step 11's job for comparing predictions to gold; an
annotator disagreement is a data-quality signal about the GOLD data itself,
which should stay maximally strict so annotation guidelines can be fixed
before anything gets frozen as ground truth (LLD doc 2 section 21: "I would
use disagreement analysis to refine the annotation guidelines before
freezing the gold dataset").
"""

from __future__ import annotations

from pydantic import BaseModel

from annotation.paths import flatten_payload


def cohens_kappa(labels_a: list[bool], labels_b: list[bool]) -> float:
    """Standard Cohen's kappa for two raters' binary judgments over the
    same set of items. 1.0 = perfect agreement, 0.0 = chance-level."""
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        raise ValueError("cohens_kappa requires two equal-length, non-empty label lists")

    observed_agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    p_a_true = sum(labels_a) / n
    p_b_true = sum(labels_b) / n
    expected_agree = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)

    if expected_agree >= 1.0:
        return 1.0 if observed_agree >= 1.0 else 0.0
    return (observed_agree - expected_agree) / (1 - expected_agree)


class FieldDisagreement(BaseModel):
    field: str
    value_a: object
    value_b: object


class AgreementReport(BaseModel):
    fields_compared: int
    agreements: int
    disagreements: list[FieldDisagreement]
    agreement_rate: float
    presence_kappa: float


def compute_agreement(payload_a: dict, payload_b: dict) -> AgreementReport:
    flat_a = flatten_payload(payload_a)
    flat_b = flatten_payload(payload_b)
    all_fields = sorted(set(flat_a) | set(flat_b))

    _MISSING = object()
    disagreements: list[FieldDisagreement] = []
    agreements = 0
    presence_a: list[bool] = []
    presence_b: list[bool] = []

    for field in all_fields:
        value_a = flat_a.get(field, _MISSING)
        value_b = flat_b.get(field, _MISSING)
        presence_a.append(value_a is not _MISSING and value_a is not None)
        presence_b.append(value_b is not _MISSING and value_b is not None)

        if value_a == value_b:
            agreements += 1
        else:
            disagreements.append(FieldDisagreement(
                field=field,
                value_a=None if value_a is _MISSING else value_a,
                value_b=None if value_b is _MISSING else value_b,
            ))

    total = len(all_fields)
    return AgreementReport(
        fields_compared=total,
        agreements=agreements,
        disagreements=disagreements,
        agreement_rate=(agreements / total) if total else 1.0,
        presence_kappa=cohens_kappa(presence_a, presence_b) if total else 1.0,
    )
