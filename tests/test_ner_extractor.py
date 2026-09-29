from __future__ import annotations

from extractors.ner.gazetteer import find_entities
from extractors.ner.labels import EntityLabel
from extractors.ner.linking import find_nearest_match
from extractors.ner_extractor import NerExtractor
from extractors.rules.concentration import ConcentrationRule
from ingestion.canonical import BlockType, CanonicalBlock, CanonicalDocument

# ---- gazetteer entity finding -------------------------------------------------------


def test_finds_precursor_solvent_and_material_in_one_sentence():
    text = "Silver nitrate was dissolved in ethanol to form silver nanoparticles."
    entities = find_entities(text)
    labels_and_text = [(e.label, e.text) for e in entities]

    assert (EntityLabel.PRECURSOR, "Silver nitrate") in labels_and_text
    assert (EntityLabel.SOLVENT, "ethanol") in labels_and_text
    assert (EntityLabel.MATERIAL, "silver nanoparticles") in labels_and_text


def test_matching_is_case_insensitive():
    entities = find_entities("SILVER NITRATE was used as the precursor.")
    assert any(e.label == EntityLabel.PRECURSOR for e in entities)


def test_longer_phrase_wins_over_shorter_substring():
    """'deionized water' should be one SOLVENT entity, not 'water' matched
    separately inside it."""
    entities = find_entities("Rinsed thoroughly with deionized water before use.")
    solvent_entities = [e for e in entities if e.label == EntityLabel.SOLVENT]
    assert len(solvent_entities) == 1
    assert solvent_entities[0].text == "deionized water"


def test_word_boundary_prevents_substring_false_positive():
    """'water' must not match inside 'watertight'."""
    entities = find_entities("The container must remain watertight during synthesis.")
    assert not any(e.text.lower() == "water" for e in entities)


def test_repeated_term_found_at_each_occurrence():
    entities = find_entities("TEM confirmed particle size. TEM images showed uniform spheres.")
    tem_entities = [e for e in entities if e.text == "TEM"]
    assert len(tem_entities) == 2
    assert tem_entities[0].char_start < tem_entities[1].char_start


def test_synonymous_precursor_names_are_not_merged_by_the_gazetteer():
    """Documented limitation: 'AgNO3' and 'silver nitrate' surface as two
    distinct entities. Canonicalizing them is Step 9's job."""
    entities = find_entities("Silver nitrate (AgNO3) was the precursor used.")
    precursor_texts = {e.text for e in entities if e.label == EntityLabel.PRECURSOR}
    assert precursor_texts == {"Silver nitrate", "AgNO3"}


# ---- linking --------------------------------------------------------------------------


def test_find_nearest_match_picks_the_closer_of_two_candidates():
    text = "Silver nitrate (10 mM) was dissolved; separately 0.5 M NaOH was prepared."
    matches = ConcentrationRule().apply(text)
    entity_start, entity_end = text.index("Silver nitrate"), text.index("Silver nitrate") + len("Silver nitrate")

    nearest = find_nearest_match(entity_start, entity_end, matches, max_distance=60)
    assert nearest.value == {"value": 10.0, "unit": "mM"}


def test_find_nearest_match_returns_none_beyond_max_distance():
    text = "Silver nitrate was used." + (" " * 100) + "Separately, 10 mM NaCl was prepared elsewhere."
    matches = ConcentrationRule().apply(text)
    entity_start, entity_end = 0, len("Silver nitrate")

    assert find_nearest_match(entity_start, entity_end, matches, max_distance=60) is None


# ---- full NerExtractor ------------------------------------------------------------------


def block(text, page=1, section=None):
    return CanonicalBlock(block_type=BlockType.PARAGRAPH, page_number=page, section=section, text=text)


def test_ner_extractor_links_concentration_to_the_right_precursor():
    document = CanonicalDocument(
        paper_id="P001",
        version=1,
        blocks=[
            block(
                "Silver nitrate (10 mM) was dissolved in ethanol and heated at 80 C for 2 h.",
                page=2,
                section="Experimental",
            )
        ],
    )
    result = NerExtractor().extract(document)

    assert len(result.prediction.precursors) == 1
    precursor = result.prediction.precursors[0]
    assert precursor.name == "Silver nitrate"
    assert precursor.concentration.value == 10.0
    assert precursor.concentration.unit == "mM"
    assert result.prediction.solvent == "ethanol"


def test_ner_extractor_links_mass_amount_to_precursor():
    document = CanonicalDocument(
        paper_id="P001",
        version=1,
        blocks=[block("100 mg of AgNO3 was added slowly to the flask.", page=3, section="Experimental")],
    )
    result = NerExtractor().extract(document)

    assert len(result.prediction.precursors) == 1
    assert result.prediction.precursors[0].name == "AgNO3"
    assert result.prediction.precursors[0].amount.value == 100.0
    assert result.prediction.precursors[0].amount.unit == "mg"


def test_ner_extractor_deduplicates_repeated_characterization_technique():
    document = CanonicalDocument(
        paper_id="P001",
        version=1,
        blocks=[block("TEM confirmed particle size. TEM images showed uniform spheres.", page=6, section="Results")],
    )
    result = NerExtractor().extract(document)

    assert len(result.prediction.characterization) == 1
    assert result.prediction.characterization[0].technique == "TEM"


def test_ner_extractor_provenance_traces_precursor_and_linked_concentration():
    document = CanonicalDocument(
        paper_id="P001",
        version=1,
        blocks=[block("Silver nitrate (10 mM) was the precursor.", page=4, section="Experimental")],
    )
    result = NerExtractor().extract(document)

    fields = {c.field for c in result.provenance}
    assert "precursors[0].name" in fields
    assert "precursors[0].concentration" in fields
    for candidate in result.provenance:
        assert candidate.source.value == "NER"
        assert candidate.evidence.section == "Experimental"


def test_ner_extractor_leaves_concentration_unlinked_when_no_precursor_nearby():
    document = CanonicalDocument(
        paper_id="P001",
        version=1,
        blocks=[block("The solution was maintained at 10 mM throughout the reaction.", page=4)],
    )
    result = NerExtractor().extract(document)
    assert result.prediction.precursors == []


def test_ner_extractor_handles_document_with_no_entities():
    document = CanonicalDocument(
        paper_id="P001",
        version=1,
        blocks=[block("This section discusses broader environmental implications.", page=1)],
    )
    result = NerExtractor().extract(document)
    assert result.prediction.precursors == []
    assert result.prediction.material is None
    assert result.provenance == []
