from __future__ import annotations

import re

from extractors.rules.base import ExtractionRule, RuleMatch
from extractors.rules.units import TIME_UNITS, normalize_unit

_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(hours|hour|hrs|hr|h|minutes|minute|mins|min|seconds|second|secs|sec|s)\b",
    re.IGNORECASE,
)


class ReactionTimeRule(ExtractionRule):
    """Named `reaction_time` (not `time`) to match the schema field it feeds
    (SynthesisExtraction.reaction_time) and to avoid any confusion with the
    stdlib `time` module."""

    field = "reaction_time"

    def apply(self, text: str) -> list[RuleMatch]:
        matches = []
        for m in _PATTERN.finditer(text):
            unit = normalize_unit(m.group(2), TIME_UNITS)
            if unit is None:
                continue
            matches.append(
                RuleMatch(
                    field=self.field,
                    value={"value": float(m.group(1)), "unit": unit},
                    text=m.group(0),
                    char_start=m.start(),
                    char_end=m.end(),
                    confidence=0.95,
                )
            )
        return matches
