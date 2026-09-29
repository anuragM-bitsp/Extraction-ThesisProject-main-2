"""
Unit-aware numeric comparison for evaluation.

LLD doc 2 section 23: "For scientific quantities, exact string matching is
insufficient. I would normalize units and compare canonical numerical
values within appropriate tolerances." — e.g. 0.5 M and 500 mM should
compare equal.

This is deliberately the ONLY place in the whole project unit conversion
happens. Steps 1, 8, and 9 all kept `Quantity` un-normalized on purpose,
reasoning that normalizing early risks silently destroying information if
a unit was misread upstream. That argument doesn't apply here: evaluation
compares a prediction against an already-finalized gold value, so
converting both to a common basis for comparison can't corrupt anything
upstream — there is no "upstream" left at this point.
"""

from __future__ import annotations

import math

_TEMPERATURE_TO_KELVIN = {
    "C": lambda c: c + 273.15,
    "K": lambda k: k,
    "F": lambda f: (f - 32) * 5 / 9 + 273.15,
}
_TIME_TO_SECONDS = {"s": 1, "min": 60, "h": 3600}
_CONCENTRATION_TO_MOLAR = {"M": 1, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9}
_MASS_TO_GRAMS = {"g": 1, "mg": 1e-3, "kg": 1e3, "ug": 1e-6}
_VOLUME_TO_LITERS = {"L": 1, "mL": 1e-3, "uL": 1e-6}

_LINEAR_TABLES = {
    "time": _TIME_TO_SECONDS,
    "concentration": _CONCENTRATION_TO_MOLAR,
    "mass": _MASS_TO_GRAMS,
    "volume": _VOLUME_TO_LITERS,
}


def to_canonical(value: float, unit: str) -> tuple[float, str] | None:
    """Returns (canonical_value, dimension_name), or None if `unit` isn't
    recognized in any dimension — an unrecognized unit is a data problem
    worth surfacing, not something to silently treat as unequal-by-default
    without saying why."""
    if unit in _TEMPERATURE_TO_KELVIN:
        return _TEMPERATURE_TO_KELVIN[unit](value), "temperature"
    for dimension, table in _LINEAR_TABLES.items():
        if unit in table:
            return value * table[unit], dimension
    return None


def quantities_equal(a: dict | None, b: dict | None, rel_tol: float = 1e-3, abs_tol: float = 1e-9) -> bool:
    """
    `a`/`b` are {'value': float, 'unit': str} dicts (a Quantity's
    model_dump()), or None.

    Two `None`s are NOT considered equal here — "both extractors found
    nothing" is a different, and not necessarily meaningful, case that
    callers (comparators.py) handle explicitly as `both_absent` rather than
    silently folding it into "correct".
    """
    if a is None or b is None:
        return False
    canonical_a = to_canonical(a["value"], a["unit"])
    canonical_b = to_canonical(b["value"], b["unit"])
    if canonical_a is None or canonical_b is None:
        return False
    (value_a, dim_a), (value_b, dim_b) = canonical_a, canonical_b
    if dim_a != dim_b:
        return False
    return math.isclose(value_a, value_b, rel_tol=rel_tol, abs_tol=abs_tol)
