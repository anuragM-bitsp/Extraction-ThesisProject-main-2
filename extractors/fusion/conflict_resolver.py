"""
ConflictResolver (LLD sections 16-18).

    "The fusion layer needs an explicit conflict-resolution policy... If two
    systems disagree, I can use source reliability, evidence quality,
    contextual consistency... to resolve the conflict." (LLD section 14)
    "I would trust deterministic rules strongly for units and numerical
    conditions, use NER for entity detection, and use the LLM for semantic
    relations..." (LLD section 18)

This resolver operationalizes exactly that: a per-field ranking of which
*strategy* to trust, with confidence as a tiebreaker within an equally-
ranked source. It deliberately does NOT attempt semantic equivalence —
"80 C" and "353 K" are different values here, not reconciled — unit-aware
comparison is Step 11's job (evaluation), not fusion's, for the same reason
Step 1 kept Quantity un-normalized: normalizing too early can silently
destroy information if a unit was misread upstream.
"""

from __future__ import annotations

from schemas.provenance import CandidateFact, SourceType

# Per-field priority order, one entry per field NAME (not per list index —
# "precursors[0].concentration" and "precursors[3].concentration" both look
# up under "concentration"). Reflects LLD section 18's own reasoning:
# rules for numeric/unit patterns, NER for entity identification, LLM for
# context requiring broader semantic understanding. Not exhaustive of every
# schema field — fields no current extractor populates (e.g. pH) simply
# never reach the resolver with any candidates.
DEFAULT_FIELD_PRIORITY: dict[str, list[SourceType]] = {
    "temperature": [SourceType.RULE, SourceType.NER, SourceType.LLM],
    "reaction_time": [SourceType.RULE, SourceType.NER, SourceType.LLM],
    "concentration": [SourceType.RULE, SourceType.NER, SourceType.LLM],
    "amount": [SourceType.RULE, SourceType.NER, SourceType.LLM],
    "solvent": [SourceType.NER, SourceType.LLM],
    "material.name": [SourceType.NER, SourceType.LLM],
    "synthesis_method": [SourceType.LLM, SourceType.NER],
    "pH": [SourceType.RULE, SourceType.LLM],
}

DEFAULT_PRIORITY_FALLBACK: list[SourceType] = [
    SourceType.RULE, SourceType.NER, SourceType.LLM, SourceType.HYBRID, SourceType.HUMAN,
]


class ConflictResolver:
    def __init__(self, field_priority: dict[str, list[SourceType]] | None = None):
        self.field_priority = field_priority if field_priority is not None else DEFAULT_FIELD_PRIORITY

    def resolve(self, field_key: str, candidates: list[CandidateFact]) -> CandidateFact | None:
        """`field_key` is the priority-lookup key, e.g. 'concentration' or
        'temperature' — unindexed, since priority depends on field type, not
        position within a list. Returns the winning CandidateFact unchanged
        (with its original `source`), so provenance still shows which
        strategy the fused value actually came from — that traceability is
        the entire point of keeping provenance through fusion."""
        if not candidates:
            return None

        priority = self.field_priority.get(field_key, DEFAULT_PRIORITY_FALLBACK)

        def sort_key(candidate: CandidateFact) -> tuple[int, float]:
            try:
                rank = priority.index(candidate.source)
            except ValueError:
                rank = len(priority)  # unranked source: sorts after every ranked one
            return (rank, -candidate.confidence)

        return sorted(candidates, key=sort_key)[0]
