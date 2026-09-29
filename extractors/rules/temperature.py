from __future__ import annotations

import re

from extractors.rules.base import ExtractionRule, RuleMatch
from extractors.rules.units import TEMPERATURE_UNITS, normalize_unit

_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*\u00b0?\s*(Celsius|Fahrenheit|Kelvin|C|K|F)\b",
    re.IGNORECASE,
)


class TemperatureRule(ExtractionRule):
    field = "temperature"

    def apply(self, text: str) -> list[RuleMatch]:
        matches = []
        for m in _PATTERN.finditer(text):
            unit = normalize_unit(m.group(2), TEMPERATURE_UNITS)
            if unit is None:
                continue
            matches.append(
                RuleMatch(
                    field=self.field,
                    value={"value": float(m.group(1)), "unit": unit},
                    text=m.group(0),
                    char_start=m.start(),
                    char_end=m.end(),
                    confidence=0.97,
                )
            )
        return matches
