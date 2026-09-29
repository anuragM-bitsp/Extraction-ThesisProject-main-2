"""
Unit normalization.

Kept separate from any single rule because the same "raw string -> canonical
unit" lookup pattern is shared by every numeric rule. Case-insensitive
matching is a deliberate approximation — "mM" vs "MM" is technically
ambiguous, but scientific-paper OCR/PDF-extraction quality makes exact-case
matching too brittle to be worth the false negatives it would cause. This is
a known, documented trade-off, not an oversight.
"""

from __future__ import annotations

TEMPERATURE_UNITS = {
    "c": "C", "°c": "C", "celsius": "C",
    "k": "K", "kelvin": "K",
    "f": "F", "fahrenheit": "F",
}

TIME_UNITS = {
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "min": "min", "mins": "min", "minute": "min", "minutes": "min",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
}

CONCENTRATION_UNITS = {
    "mm": "mM", "\u00b5m": "uM", "um": "uM", "nm": "nM", "m": "M",
    "mg/ml": "mg/mL", "mg/l": "mg/L", "wt%": "wt%",
}

VOLUME_UNITS = {
    "ml": "mL", "\u00b5l": "uL", "ul": "uL", "l": "L",
}

MASS_UNITS = {
    "mg": "mg", "g": "g", "kg": "kg", "\u00b5g": "ug", "ug": "ug",
}


def normalize_unit(raw: str, table: dict[str, str]) -> str | None:
    return table.get(raw.strip().lower())
