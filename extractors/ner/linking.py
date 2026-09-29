"""
Entity-relation linking.

LLD doc 2 section 8: "NER identifies entities... but it doesn't tell us how
those entities are related... NER -> Entities -> Relation Extraction ->
Structured JSON." This module is that relation-extraction step for one
specific, common relation: a numeric quantity (concentration, mass) sitting
next to a precursor entity almost always describes that precursor —
"silver nitrate (10 mM)" — so proximity is a reasonable link rule.

This is a heuristic, stated plainly: nearest-span matching, not a trained
relation classifier. It will get confused by a sentence with two precursors
and one shared, ambiguous quantity ("silver nitrate and sodium citrate were
mixed with 10 mM solutions") — a real system would use dependency parsing
or a trained relation-extraction model for that case. Good enough to
demonstrate the linking step exists and is independently testable; not a
claim of production-grade relation extraction.
"""

from __future__ import annotations

from extractors.rules.base import RuleMatch


def _span_distance(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    if a_end <= b_start:
        return b_start - a_end
    if b_end <= a_start:
        return a_start - b_end
    return 0  # overlapping spans


def find_nearest_match(
    entity_start: int, entity_end: int, matches: list[RuleMatch], max_distance: int = 60
) -> RuleMatch | None:
    best: RuleMatch | None = None
    best_distance: int | None = None
    for match in matches:
        distance = _span_distance(entity_start, entity_end, match.char_start, match.char_end)
        if distance <= max_distance and (best_distance is None or distance < best_distance):
            best, best_distance = match, distance
    return best
