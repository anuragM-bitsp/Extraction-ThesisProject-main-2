"""
LLM JSON -> SynthesisExtraction + CandidateFacts.

The LLM's parsed JSON is always kept verbatim (ExtractionResult.llm_run.output).
This module additionally maps whatever parts of it use the project schema's
field names onto `SynthesisExtraction`, so the downstream steps that already
exist — hybrid fusion, normalization/entity linking, evaluation, MLflow —
keep working unchanged.

Mapping is deliberately lenient about *shape* (models and custom prompts
vary): a quantity may arrive as {"value": 80, "unit": "°C"}, as "80 °C", or
as a one-element list; a precursor may be an object or a bare string; a
step may be a string or {"description": ...}. It is strict about *content*:
anything that can't be interpreted unambiguously (e.g. "room temperature"
as a Quantity, or a range like "25–30 °C") is left out of the prediction
rather than guessed — it is still visible in the raw LLM JSON.

Units are written in the same spelling the rule-based extractor uses
(C, min, h, mM, uM, mL, ...) so rule and LLM values are comparable in
fusion and in evaluation's unit-aware comparison.

Evidence: an LLM answer has no character offsets, so each value is looked
up verbatim in the document. When found, the fact gets a precise
page/section/character-span EvidenceSpan (confidence 0.8); otherwise a
document-level EvidenceSpan that says so (confidence 0.6).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ingestion.canonical import BlockType, CanonicalDocument
from schemas.extraction_schema import (
    CharacterizationMethod,
    Material,
    MaterialProperty,
    Precursor,
    Quantity,
    SynthesisExtraction,
    SynthesisStep,
)
from schemas.provenance import CandidateFact, EvidenceSpan, SourceType

LOCATED_CONFIDENCE = 0.8
UNLOCATED_CONFIDENCE = 0.6
UNLOCATED_EVIDENCE_TEXT = "Reported by the LLM; no verbatim match was found in the document text."

_MISSING_STRINGS = {"", "null", "none", "n/a", "na", "not stated", "not reported", "not mentioned", "unknown", "-"}

# Case-sensitive first: "nm" (length) vs "nM" (concentration), "M" vs "m".
_UNIT_EXACT = {
    "°C": "C", "ºC": "C", "℃": "C", "C": "C", "degC": "C", "K": "K", "°F": "F", "F": "F",
    "s": "s", "sec": "s", "secs": "s", "min": "min", "mins": "min", "h": "h", "hr": "h", "hrs": "h",
    "M": "M", "mM": "mM", "µM": "uM", "μM": "uM", "uM": "uM", "nM": "nM",
    "mL": "mL", "ml": "mL", "µL": "uL", "μL": "uL", "uL": "uL", "ul": "uL", "L": "L",
    "mg": "mg", "g": "g", "kg": "kg", "µg": "ug", "μg": "ug", "ug": "ug",
    "nm": "nm", "µm": "um", "μm": "um", "mm": "mm",
    "mol": "mol", "mmol": "mmol", "µmol": "umol", "μmol": "umol", "umol": "umol",
    "wt%": "wt%", "wt.%": "wt%", "%": "%",
    "mg/mL": "mg/mL", "mg/ml": "mg/mL", "mg/L": "mg/L", "mg/l": "mg/L",
    "rpm": "rpm", "eV": "eV", "mV": "mV",
}
_UNIT_WORDS = {
    "celsius": "C", "degrees celsius": "C", "degree celsius": "C", "deg c": "C", "degrees c": "C",
    "kelvin": "K", "fahrenheit": "F",
    "second": "s", "seconds": "s", "minute": "min", "minutes": "min", "hour": "h", "hours": "h",
    "molar": "M", "millimolar": "mM", "micromolar": "uM", "nanomolar": "nM",
    "milliliter": "mL", "milliliters": "mL", "millilitre": "mL", "microliter": "uL", "microliters": "uL",
    "nanometer": "nm", "nanometers": "nm", "nanometre": "nm",
}

_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_QUANTITY_STRING_RE = re.compile(rf"^\s*(?:~|≈|ca\.?|approx\.?|about)?\s*({_NUMBER_RE})\s*(.+?)\s*$")


# ---- primitive coercions ------------------------------------------------------------------


def canonical_unit(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    unit = str(raw).strip()
    if not unit:
        return None
    unit = re.sub(r"°\s+", "°", unit).replace("wt %", "wt%").replace("wt. %", "wt%")
    if unit in _UNIT_EXACT:
        return _UNIT_EXACT[unit]
    compact = unit.replace(" ", "")
    if compact in _UNIT_EXACT:
        return _UNIT_EXACT[compact]
    lowered = unit.lower().rstrip(".")
    if lowered in _UNIT_WORDS:
        return _UNIT_WORDS[lowered]
    return unit


def clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        for item in value:
            s = clean_str(item)
            if s:
                return s
        return None
    if isinstance(value, dict):
        for key in ("name", "value", "description", "text"):
            if key in value:
                return clean_str(value[key])
        return None
    s = str(value).strip()
    return None if s.lower() in _MISSING_STRINGS else s


def to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.fullmatch(rf"\s*(?:~|≈)?\s*({_NUMBER_RE})\s*", value)
        if m:
            return float(m.group(1))
    return None


def to_quantity(value: Any) -> Optional[Quantity]:
    """{"value": 80, "unit": "°C"} | "80 °C" | [first, ...] -> Quantity, else None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, list):
        return to_quantity(value[0]) if value else None
    if isinstance(value, dict):
        raw_value = next((value[k] for k in ("value", "amount", "number", "magnitude", "quantity") if k in value), None)
        raw_unit = next((value[k] for k in ("unit", "units") if k in value), None)
        number = to_float(raw_value)
        if number is None and isinstance(raw_value, str) and raw_unit is None:
            return to_quantity(raw_value)  # {"value": "80 °C"}
        unit = canonical_unit(raw_unit)
        if number is None or not unit:
            return None
        return Quantity(value=number, unit=unit)
    if isinstance(value, str):
        m = _QUANTITY_STRING_RE.match(value)
        if not m:
            return None
        unit_part = m.group(2)
        # Reject ranges / multiple values ("25-30 °C", "2 or 3 h") — ambiguous.
        if re.match(r"^(?:-|–|—|to\b|or\b|and\b|/)\s*\d", unit_part):
            return None
        unit = canonical_unit(unit_part)
        return Quantity(value=float(m.group(1)), unit=unit) if unit else None
    return None


def _first_key(data: dict, *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


# ---- evidence ---------------------------------------------------------------------------


class EvidenceLocator:
    """Finds the block (page/section/char span) where a value appears."""

    def __init__(self, document: CanonicalDocument):
        self.document = document
        self.blocks = [
            b for b in document.blocks if b.text and b.block_type != BlockType.REFERENCE
        ]
        self._lower = [b.text.lower() for b in self.blocks]

    def _span(self, block_idx: int, start: int, end: int) -> EvidenceSpan:
        block = self.blocks[block_idx]
        return EvidenceSpan(
            paper_id=self.document.paper_id,
            version=self.document.version,
            page=block.page_number,
            section=block.section,
            text=block.text[start:end],
            char_start=start,
            char_end=end,
        )

    def unlocated(self) -> EvidenceSpan:
        return EvidenceSpan(
            paper_id=self.document.paper_id, version=self.document.version, text=UNLOCATED_EVIDENCE_TEXT
        )

    def find_text(self, text: Optional[str]) -> Optional[EvidenceSpan]:
        if not text or len(text.strip()) < 2:
            return None
        needle = text.strip().lower()
        for i, hay in enumerate(self._lower):
            pos = hay.find(needle)
            if pos >= 0:
                return self._span(i, pos, pos + len(needle))
        return None

    def find_quantity(self, quantity: Optional[Quantity]) -> Optional[EvidenceSpan]:
        if quantity is None:
            return None
        number = f"{quantity.value:g}"
        unit_forms = {quantity.unit}
        unit_forms.update(k for k, v in _UNIT_EXACT.items() if v == quantity.unit)
        unit_alt = "|".join(re.escape(u.lstrip("°º")) for u in sorted(unit_forms, key=len, reverse=True))
        pattern = re.compile(rf"(?<![\d.]){re.escape(number)}(?:\.0+)?\s*[°º]?\s*(?:{unit_alt})(?![A-Za-z])")
        for i, block in enumerate(self.blocks):
            m = pattern.search(block.text)
            if m:
                return self._span(i, m.start(), m.end())
        return None


# ---- main mapping -------------------------------------------------------------------------


_TOP_LEVEL_ALIASES = {
    "precursors": ("precursors", "reagents", "chemicals"),
    "reaction_time": ("reaction_time", "time", "duration"),
    "synthesis_method": ("synthesis_method", "method"),
    "characterization": ("characterization", "characterization_methods", "techniques"),
    "steps": ("steps", "procedure"),
}


def map_llm_output(
    data: Any,
    document: CanonicalDocument,
    extractor_version: str,
) -> tuple[SynthesisExtraction, list[CandidateFact]]:
    prediction = SynthesisExtraction(paper_id=document.paper_id)
    facts: list[CandidateFact] = []
    if not isinstance(data, dict):
        return prediction, facts

    locator = EvidenceLocator(document)

    def add_fact(field: str, value: Any, evidence: Optional[EvidenceSpan]) -> None:
        facts.append(
            CandidateFact(
                field=field,
                value=value,
                source=SourceType.LLM,
                confidence=LOCATED_CONFIDENCE if evidence is not None else UNLOCATED_CONFIDENCE,
                evidence=evidence or locator.unlocated(),
                extractor_version=extractor_version,
            )
        )

    def get(field: str) -> Any:
        return _first_key(data, *_TOP_LEVEL_ALIASES.get(field, (field,)))

    # -- material
    raw_material = data.get("material")
    if isinstance(raw_material, str):
        raw_material = {"name": raw_material}
    if isinstance(raw_material, dict):
        material = Material(
            name=clean_str(raw_material.get("name")),
            composition=clean_str(raw_material.get("composition")),
            size=to_quantity(raw_material.get("size")),
            morphology=clean_str(raw_material.get("morphology") or raw_material.get("shape")),
        )
        if any([material.name, material.composition, material.size, material.morphology]):
            prediction.material = material
            if material.name:
                add_fact("material.name", material.name, locator.find_text(material.name))
            if material.composition:
                add_fact("material.composition", material.composition, locator.find_text(material.composition))
            if material.size:
                add_fact("material.size", material.size.model_dump(), locator.find_quantity(material.size))
            if material.morphology:
                add_fact("material.morphology", material.morphology, locator.find_text(material.morphology))

    # -- precursors
    for item in _as_list(get("precursors")):
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = clean_str(_first_key(item, "name", "chemical", "compound", "reagent"))
        if not name:
            continue
        amount = to_quantity(_first_key(item, "amount", "volume", "mass", "moles"))
        concentration = to_quantity(item.get("concentration"))
        prediction.precursors.append(Precursor(name=name, amount=amount, concentration=concentration))
        idx = len(prediction.precursors) - 1
        add_fact(f"precursors[{idx}].name", name, locator.find_text(name))
        if concentration is not None:
            add_fact(f"precursors[{idx}].concentration", concentration.model_dump(), locator.find_quantity(concentration))
        if amount is not None:
            add_fact(f"precursors[{idx}].amount", amount.model_dump(), locator.find_quantity(amount))

    # -- scalar strings
    for field in ("solvent", "synthesis_method"):
        value = clean_str(get(field))
        if value:
            setattr(prediction, field, value)
            add_fact(field, value, locator.find_text(value))

    # -- scalar quantities
    for field in ("temperature", "reaction_time"):
        quantity = to_quantity(get(field))
        if quantity is not None:
            setattr(prediction, field, quantity)
            add_fact(field, quantity.model_dump(), locator.find_quantity(quantity))

    # -- pH
    ph = to_float(get("pH") if get("pH") is not None else data.get("ph"))
    if ph is not None and 0 <= ph <= 14:
        prediction.pH = ph
        add_fact("pH", ph, None)

    # -- steps
    for item in _as_list(get("steps")):
        description = clean_str(
            _first_key(item, "description", "action", "step", "text") if isinstance(item, dict) else item
        )
        if not description:
            continue
        prediction.steps.append(SynthesisStep(order=len(prediction.steps) + 1, description=description))
        idx = len(prediction.steps) - 1
        add_fact(f"steps[{idx}].description", description, locator.find_text(description))

    # -- characterization
    for item in _as_list(get("characterization")):
        if isinstance(item, str):
            item = {"technique": item}
        if not isinstance(item, dict):
            continue
        technique = clean_str(_first_key(item, "technique", "method", "name"))
        if not technique:
            continue
        finding = clean_str(_first_key(item, "finding", "result", "description"))
        prediction.characterization.append(CharacterizationMethod(technique=technique, finding=finding))
        idx = len(prediction.characterization) - 1
        add_fact(f"characterization[{idx}].technique", technique, locator.find_text(technique))

    # -- properties
    for item in _as_list(get("properties")):
        if not isinstance(item, dict):
            continue
        name = clean_str(_first_key(item, "name", "property"))
        if not name:
            continue
        raw_value = item.get("value")
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool) and item.get("unit"):
            raw_value = {"value": raw_value, "unit": item.get("unit")}
        quantity = to_quantity(raw_value)
        qualitative = clean_str(item.get("qualitative_value"))
        if quantity is None and qualitative is None and isinstance(raw_value, str):
            qualitative = clean_str(raw_value)
        prediction.properties.append(MaterialProperty(name=name, value=quantity, qualitative_value=qualitative))
        idx = len(prediction.properties) - 1
        add_fact(f"properties[{idx}].name", name, locator.find_text(name))
        if quantity is not None:
            add_fact(f"properties[{idx}].value", quantity.model_dump(), locator.find_quantity(quantity))

    return prediction, facts
