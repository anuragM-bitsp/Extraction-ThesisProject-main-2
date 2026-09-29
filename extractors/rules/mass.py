from __future__ import annotations

import re

from extractors.rules.base import ExtractionRule, RuleMatch
from extractors.rules.units import MASS_UNITS, normalize_unit

# The trailing (?!\s*/) stops "5 mg/mL" from being read as a 5 mg mass —
# that's a concentration (ConcentrationRule's job), not a standalone mass.
_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|kg|g|\u00b5g|ug)\b(?!\s*/)",
    re.IGNORECASE,
)


class MassRule(ExtractionRule):
    field = "mass"

    def apply(self, text: str) -> list[RuleMatch]:
        matches = []
        for m in _PATTERN.finditer(text):
            unit = normalize_unit(m.group(2), MASS_UNITS)
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
