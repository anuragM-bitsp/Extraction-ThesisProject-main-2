"""
Ablation study aggregation (LLD doc 2 section 25): turn many per-document
`EvaluationReport`s into one comparison row per (extractor, level), so
Rules/NER/Prompt+LLM/Hybrid can be compared side by side on the same gold set.

Aggregation sums tp/fp/fn across documents first, THEN computes
precision/recall/F1 from the totals ("micro" averaging) rather than
averaging each document's F1 score ("macro" averaging). Micro avoids a
subtle distortion macro is prone to: a document with zero gold entities for
some field would have undefined/zero recall dragging down a macro average
even though there was nothing to find, whereas summing counts first means
that document simply contributes 0 to both tp and fn and doesn't skew
anything.
"""

from __future__ import annotations

from evaluation.evaluator import EvaluationReport
from evaluation.metrics import PRF1

LEVELS = ("entity", "relation", "field")


def aggregate_reports(reports: list[EvaluationReport]) -> dict[str, PRF1]:
    """Returns {'{extractor}:{level}': PRF1}, summed across every report
    for that extractor."""
    aggregated: dict[str, PRF1] = {}
    for report in reports:
        for level in LEVELS:
            key = f"{report.extractor}:{level}"
            aggregated[key] = aggregated.get(key, PRF1()) + getattr(report, level)
    return aggregated


def ablation_table(reports: list[EvaluationReport]) -> list[dict]:
    """One row per (extractor, level), ready to print or hand to
    pandas/a markdown table — this is the table LLD section 25 sketches:

        System   Entity F1   Relation F1   Field F1
        Rules       ?             ?            ?
        NER         ?             ?            ?
        ...
    """
    aggregated = aggregate_reports(reports)
    rows = []
    for key, prf1 in sorted(aggregated.items()):
        extractor, level = key.split(":")
        rows.append(
            {
                "extractor": extractor,
                "level": level,
                "precision": round(prf1.precision, 3),
                "recall": round(prf1.recall, 3),
                "f1": round(prf1.f1, 3),
                "tp": prf1.tp,
                "fp": prf1.fp,
                "fn": prf1.fn,
            }
        )
    return rows
