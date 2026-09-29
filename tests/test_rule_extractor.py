from __future__ import annotations

import pytest

from extractors.rule_extractor import RuleExtractor
from extractors.rules.concentration import ConcentrationRule
from extractors.rules.mass import MassRule
from extractors.rules.reaction_time import ReactionTimeRule
from extractors.rules.temperature import TemperatureRule
from extractors.rules.volume import VolumeRule
from ingestion.canonical import BlockType, CanonicalBlock, CanonicalDocument
from schemas.provenance import SourceType

# ---- individual rules -------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_value,expected_unit",
    [
        ("heated at 80 \u00b0C for 2 h", 80.0, "C"),
        ("annealed at 300 K", 300.0, "K"),
        ("cooled to -20 C", -20.0, "C"),
        ("body temperature is 98.6 Fahrenheit", 98.6, "F"),
    ],
)
def test_temperature_rule(text, expected_value, expected_unit):
    matches = TemperatureRule().apply(text)
    assert len(matches) == 1
    assert matches[0].value == {"value": expected_value, "unit": expected_unit}


@pytest.mark.parametrize(
    "text,expected_value,expected_unit",
    [
        ("stirred for 2 h", 2.0, "h"),
        ("stirred for 2 hours", 2.0, "h"),
        ("incubated for 30 min", 30.0, "min"),
        ("flashed for 45 s", 45.0, "s"),
        ("reacted for 1 hr", 1.0, "h"),
    ],
)
def test_reaction_time_rule(text, expected_value, expected_unit):
    matches = ReactionTimeRule().apply(text)
    assert len(matches) == 1
    assert matches[0].value == {"value": expected_value, "unit": expected_unit}


@pytest.mark.parametrize(
    "text,expected_value,expected_unit",
    [
        ("10 mM silver nitrate", 10.0, "mM"),
        ("0.5 M NaOH solution", 0.5, "M"),
        ("5 mg/mL solution", 5.0, "mg/mL"),
        ("20 nM concentration", 20.0, "nM"),
        ("1 wt% loading", 1.0, "wt%"),
    ],
)
def test_concentration_rule(text, expected_value, expected_unit):
    matches = ConcentrationRule().apply(text)
    assert len(matches) == 1
    assert matches[0].value == {"value": expected_value, "unit": expected_unit}


@pytest.mark.parametrize(
    "text,expected_value,expected_unit",
    [
        ("dissolved in 20 mL ethanol", 20.0, "mL"),
        ("added 5 L of water", 5.0, "L"),
        ("100 \u00b5L aliquot", 100.0, "uL"),
    ],
)
def test_volume_rule(text, expected_value, expected_unit):
    matches = VolumeRule().apply(text)
    assert len(matches) == 1
    assert matches[0].value == {"value": expected_value, "unit": expected_unit}


def test_mass_rule_matches_standalone_mass():
    matches = MassRule().apply("weighed 100 mg of powder")
    assert len(matches) == 1
    assert matches[0].value == {"value": 100.0, "unit": "mg"}


def test_mass_rule_does_not_misfire_on_compound_concentration_unit():
    """'5 mg/mL' is a concentration, not a 5 mg mass — ConcentrationRule's
    job, not MassRule's (the negative lookahead in mass.py exists for
    exactly this case)."""
    assert MassRule().apply("solution at 5 mg/mL") == []


def test_concentration_rule_catches_the_compound_unit_mass_rule_correctly_avoids():
    matches = ConcentrationRule().apply("solution at 5 mg/mL")
    assert len(matches) == 1
    assert matches[0].value == {"value": 5.0, "unit": "mg/mL"}


# ---- RuleExtractor (full pipeline) -------------------------------------------------


def block(text, page=1, section=None):
    return CanonicalBlock(block_type=BlockType.PARAGRAPH, page_number=page, section=section, text=text)


def test_rule_extractor_populates_direct_schema_fields():
    document = CanonicalDocument(
        paper_id="P001",
        version=1,
        blocks=[
            block("Silver nanoparticles have attracted attention.", page=1, section="Introduction"),
            block(
                "Silver nitrate (10 mM) was dissolved in 20 mL ethanol and heated at 80 \u00b0C for 2 h.",
                page=2,
                section="Experimental",
            ),
        ],
    )
    result = RuleExtractor().extract(document)

    assert result.extractor.value == "RULE"
    assert result.prediction.temperature.value == 80.0
    assert result.prediction.temperature.unit == "C"
    assert result.prediction.reaction_time.value == 2.0
    assert result.prediction.reaction_time.unit == "h"


def test_rule_extractor_leaves_unlinked_quantities_out_of_prediction_but_in_provenance():
    """Concentration/volume/mass are found but NOT written into
    `prediction.precursors` — that requires entity linking this extractor
    doesn't do. They must still show up in `provenance` so nothing is
    silently thrown away."""
    document = CanonicalDocument(
        paper_id="P001",
        version=1,
        blocks=[
            block(
                "Silver nitrate (10 mM) was dissolved in 20 mL ethanol and heated at 80 C for 2 h.",
                page=2,
                section="Experimental",
            ),
        ],
    )
    result = RuleExtractor().extract(document)

    assert result.prediction.precursors == []  # not linked yet — by design

    fields_found = {c.field for c in result.provenance}
    assert {"temperature", "reaction_time", "concentration", "volume"} <= fields_found
    concentration_facts = [c for c in result.provenance if c.field == "concentration"]
    assert concentration_facts[0].value == {"value": 10.0, "unit": "mM"}


def test_rule_extractor_evidence_carries_page_and_section():
    document = CanonicalDocument(
        paper_id="P001",
        version=1,
        blocks=[block("heated at 80 C for 2 h", page=5, section="Experimental")],
    )
    result = RuleExtractor().extract(document)

    temp_fact = next(c for c in result.provenance if c.field == "temperature")
    assert temp_fact.source == SourceType.RULE
    assert temp_fact.evidence.page == 5
    assert temp_fact.evidence.section == "Experimental"
    assert temp_fact.evidence.paper_id == "P001"
    assert "80" in temp_fact.evidence.text


def test_rule_extractor_first_occurrence_wins_for_repeated_field():
    """Documented simplification: if a paper reports a temperature more than
    once, the first mention in reading order is used. Proper multi-source
    conflict resolution belongs to the hybrid fusion layer (Step 8)."""
    document = CanonicalDocument(
        paper_id="P001",
        version=1,
        blocks=[
            block("First heated to 60 C to dissolve the precursor.", page=2, section="Experimental"),
            block("Then heated to 80 C for the main reaction.", page=2, section="Experimental"),
        ],
    )
    result = RuleExtractor().extract(document)
    assert result.prediction.temperature.value == 60.0

    all_temps = [c.value["value"] for c in result.provenance if c.field == "temperature"]
    assert all_temps == [60.0, 80.0]  # both preserved in provenance, nothing lost


def test_rule_extractor_produces_no_fields_when_document_has_no_matches():
    document = CanonicalDocument(
        paper_id="P001",
        version=1,
        blocks=[block("This paper discusses green chemistry principles broadly.", page=1)],
    )
    result = RuleExtractor().extract(document)
    assert result.prediction.temperature is None
    assert result.prediction.reaction_time is None
    assert result.provenance == []
