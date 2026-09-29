"""
Gold dataset assembly helpers (LLD doc 2 sections 21 and 27; LLD's own tech
table names JSONL/Parquet as the dataset format).

`paper_level_split` exists specifically because of LLD doc 2 section 21:
"Because the corpus is relatively small, I need to be especially careful
about data leakage... I would consider paper-level cross-validation or
leave-one-paper-out evaluation... Don't split chunks randomly across train
and test." Splitting is done on paper_id, never on chunks or individual
annotations, so no fact from a paper's Experimental section can end up in
training/tuning while another fact from the SAME paper ends up in test.
"""

from __future__ import annotations

import random

from schemas.extraction_schema import ExtractionResult


def paper_level_split(
    paper_ids: list[str], test_fraction: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    shuffled = list(dict.fromkeys(paper_ids))  # de-dup while preserving first-seen order
    random.Random(seed).shuffle(shuffled)

    test_size = max(1, round(len(shuffled) * test_fraction)) if shuffled else 0
    test_ids = shuffled[:test_size]
    train_ids = shuffled[test_size:]
    return train_ids, test_ids


def export_gold_jsonl(results: list[ExtractionResult]) -> str:
    """One ExtractionResult per line, each a complete JSON object — the
    ML-friendly gold dataset format the LLD's tech table names directly.
    Every record includes full provenance, so the exported file alone is
    enough to trace any gold value back to which annotation(s) produced
    it, without a live database."""
    return "\n".join(result.model_dump_json() for result in results)
