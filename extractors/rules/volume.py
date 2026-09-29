from __future__ import annotations

import re

from extractors.rules.base import ExtractionRule, RuleMatch
from extractors.rules.units import VOLUME_UNITS, normalize_unit

_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mL|\u00b5L|uL|L)\b",
    re.IGNORECASE,
)


class VolumeRule(ExtractionRule):
    field = "volume"

    def apply(self, text: str) -> list[RuleMatch]:
        matches = []
        for m in _PATTERN.finditer(text):
            unit = normalize_unit(m.group(2), VOLUME_UNITS)
            if unit is None:
                continue
            matches.append(
                RuleMatch(
                    field=self.field,
                    value={"value": float(m.group(1)), "unit": unit},
                    text=m.group(0),
                    char_start=m.start(),
                    char_end=m.end(),
                    confidence=0.93,
                )
            )
        return matches
