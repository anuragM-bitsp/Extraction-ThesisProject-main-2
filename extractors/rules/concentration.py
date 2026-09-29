from __future__ import annotations

import re

from extractors.rules.base import ExtractionRule, RuleMatch
from extractors.rules.units import CONCENTRATION_UNITS, normalize_unit

_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg/mL|mg/L|mM|\u00b5M|uM|nM|M|wt%)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


class ConcentrationRule(ExtractionRule):
    """
    Field is left as generic `concentration`, NOT written into any specific
    `precursors[i].concentration` slot — this rule has no way to know which
    precursor a bare "10 mM" refers to without entity linking (LLD sections
    16-19, Steps 6-9 of this project). The rule extractor surfaces these as
    unassigned CandidateFacts; attaching them to a named precursor is later
    work, not a gap in this rule.
    """

    field = "concentration"

    def apply(self, text: str) -> list[RuleMatch]:
        matches = []
        for m in _PATTERN.finditer(text):
            unit = normalize_unit(m.group(2), CONCENTRATION_UNITS)
            if unit is None:
                continue
            matches.append(
                RuleMatch(
                    field=self.field,
                    value={"value": float(m.group(1)), "unit": unit},
                    text=m.group(0),
                    char_start=m.start(),
                    char_end=m.end(),
                    confidence=0.9,
                )
            )
        return matches
