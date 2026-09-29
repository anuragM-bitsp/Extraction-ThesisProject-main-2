from __future__ import annotations

from extractors.fusion.conflict_resolver import ConflictResolver
from extractors.hybrid_extractor import HybridExtractor
from extractors.ner_extractor import NerExtractor
from extractors.llm.llm_client import FakeLLMClient
from extractors.llm_extractor import PromptLlmExtractor
from extractors.rule_extractor import RuleExtractor
from ingestion.canonical import BlockType, CanonicalBlock, CanonicalDocument
from schemas.provenance import CandidateFact, EvidenceSpan, SourceType


def fact(field, value, source, confidence=0.9) -> CandidateFact:
    return CandidateFact(
        field=field,
        value=value,
        source=source,
        confidence=confidence,
        evidence=EvidenceSpan(paper_id="P001", version=1, text="evidence"),
        extractor_version="test@0.0",
    )


def block(text, page=1, section=None):
    return CanonicalBlock(block_type=BlockType.PARAGRAPH, page_number=page, section=section, text=text)


# ---- ConflictResolver ---------------------------------------------------------------


def test_resolver_returns_none_for_no_candidates():
    assert ConflictResolver().resolve("temperature", []) is None


def test_resolver_prefers_higher_priority_source_over_higher_confidence():
    """LLD section 18: priority ranking beats raw confidence — a low-
    confidence RULE match still outranks a high-confidence LLM guess for a
    numeric field, because that's the whole point of ranking by strategy."""
    rule_fact = fact("temperature", {"value": 80.0, "unit": "C"}, SourceType.RULE, confidence=0.5)
    llm_fact = fact("temperature", {"value": 85.0, "unit": "C"}, SourceType.LLM, confidence=0.99)

    winner = ConflictResolver().resolve("temperature", [llm_fact, rule_fact])
    assert winner is rule_fact


def test_resolver_breaks_ties_by_confidence_within_same_source():
    low = fact("solvent", "ethanol", SourceType.NER, confidence=0.6)
    high = fact("solvent", "ethanol", SourceType.NER, confidence=0.95)

    winner = ConflictResolver().resolve("solvent", [low, high])
    assert winner is high


def test_resolver_falls_back_to_default_priority_for_unknown_field():
    rule_fact = fact("mystery_field", "x", SourceType.RULE)
    llm_fact = fact("mystery_field", "y", SourceType.LLM)

    winner = ConflictResolver().resolve("mystery_field", [llm_fact, rule_fact])
    assert winner is rule_fact  # DEFAULT_PRIORITY_FALLBACK ranks RULE first


def test_resolver_ignores_unranked_source_when_ranked_one_present():
    ranked = fact("solvent", "ethanol", SourceType.NER)
    unranked = fact("solvent", "methanol", SourceType.HUMAN)  # not in solvent's priority list
    winner = ConflictResolver().resolve("solvent", [unranked, ranked])
    assert winner is ranked


# ---- HybridExtractor (full pipeline) ----------------------------------------------------


def _document():
    return CanonicalDocument(
        paper_id="P001",
        version=1,
        blocks=[
            block(
                "Silver nanoparticles have attracted attention for their antibacterial properties.",
                page=1,
                section="Introduction",
            ),
            block(
                "Silver nitrate (10 mM) was dissolved in ethanol and heated at 80 C for 2 h.",
                page=2,
                section="Experimental",
            ),
            block(
                "TEM confirmed an average particle size of 15 nm with spherical morphology.",
                page=6,
                section="Results",
            ),
        ],
    )


def _make_hybrid_extractor(llm_answer_parts):
    """The Prompt + LLM extractor makes ONE call returning one JSON object;
    the per-field dicts below are merged into that single answer."""
    answer: dict = {}
    for part in llm_answer_parts:
        answer.update(part)
    llm_client = FakeLLMClient([answer])
    llm = PromptLlmExtractor(llm_client=llm_client, retry_delay_seconds=0)
    return HybridExtractor(rule_extractor=RuleExtractor(), ner_extractor=NerExtractor(), llm_extractor=llm)


def test_hybrid_extractor_requires_an_llm_extractor():
    import pytest

    with pytest.raises(ValueError):
        HybridExtractor(llm_extractor=None)


def test_hybrid_result_carries_the_llm_run():
    extractor = _make_hybrid_extractor([{"synthesis_method": "citrate reduction"}])
    result = extractor.extract(_document())
    assert result.llm_run is not None
    assert result.llm_run.output == {"synthesis_method": "citrate reduction"}


def test_hybrid_pulls_temperature_and_time_from_rule_since_its_the_only_source():
    extractor = _make_hybrid_extractor(
        [
            {"synthesis_method": None}, {"steps": []}, {"precursors": []}, {"properties": []},
        ]
    )
    result = extractor.extract(_document())

    assert result.extractor.value == "HYBRID"
    assert result.prediction.temperature.value == 80.0
    assert result.prediction.reaction_time.value == 2.0
    temp_fact = next(c for c in result.provenance if c.field == "temperature")
    assert temp_fact.source.value == "RULE"  # provenance shows the winning strategy


def test_hybrid_pulls_solvent_and_material_from_ner():
    extractor = _make_hybrid_extractor(
        [{"synthesis_method": None}, {"steps": []}, {"precursors": []}, {"properties": []}]
    )
    result = extractor.extract(_document())

    assert result.prediction.solvent == "ethanol"
    assert result.prediction.material.name == "Silver nanoparticles"


def test_hybrid_resolves_concentration_conflict_in_favor_of_ner_over_llm():
    """NER's concentration for 'Silver nitrate' is regex-derived (10 mM,
    genuinely present in the text). The LLM is scripted to hallucinate a
    different value (12 mM) for the SAME precursor name. Fusion must prefer
    NER's per the 'concentration' priority ranking — this is the concrete
    payoff of hybrid fusion the LLD keeps promising."""
    scripted = [
        {"synthesis_method": None},
        {"steps": []},
        {"precursors": [{"name": "Silver nitrate", "concentration": {"value": 12.0, "unit": "mM"}, "amount": None}]},
        {"properties": []},
    ]
    extractor = _make_hybrid_extractor(scripted)
    result = extractor.extract(_document())

    silver_nitrate = next(p for p in result.prediction.precursors if p.name.lower() == "silver nitrate")
    assert silver_nitrate.concentration.value == 10.0  # NER's real value, not the LLM's 12.0

    conc_fact = next(c for c in result.provenance if c.field == "precursors[0].concentration" or "concentration" in c.field)
    assert conc_fact.source.value == "NER"


def test_hybrid_does_not_merge_synonymous_precursor_names():
    """Documented limitation carried forward from Step 6: 'AgNO3' (LLM) and
    'Silver nitrate' (NER) refer to the same compound but surface as two
    separate Precursor entries, since fusion only matches exact
    (case-insensitive) surface form. Canonicalizing synonyms is Step 9."""
    scripted = [
        {"synthesis_method": None},
        {"steps": []},
        {"precursors": [{"name": "AgNO3", "concentration": None, "amount": None}]},
        {"properties": []},
    ]
    extractor = _make_hybrid_extractor(scripted)
    result = extractor.extract(_document())

    names = {p.name.lower() for p in result.prediction.precursors}
    assert names == {"silver nitrate", "agno3"}  # NOT merged into one


def test_hybrid_passes_through_llm_only_fields():
    scripted = [
        {"synthesis_method": "chemical reduction using citrate"},
        {"steps": [{"description": "Dissolve silver nitrate in ethanol"}]},
        {"precursors": []},
        {"properties": [{"name": "particle size", "value": {"value": 15.0, "unit": "nm"}, "qualitative_value": None}]},
    ]
    extractor = _make_hybrid_extractor(scripted)
    result = extractor.extract(_document())

    assert result.prediction.synthesis_method == "chemical reduction using citrate"
    assert result.prediction.steps[0].description == "Dissolve silver nitrate in ethanol"
    assert result.prediction.properties[0].name == "particle size"


def test_hybrid_deduplicates_characterization_across_sources():
    extractor = _make_hybrid_extractor(
        [{"synthesis_method": None}, {"steps": []}, {"precursors": []}, {"properties": []}]
    )
    result = extractor.extract(_document())

    assert len(result.prediction.characterization) == 1
    assert result.prediction.characterization[0].technique == "TEM"


def test_hybrid_version_string_encodes_all_three_sub_extractor_versions():
    extractor = _make_hybrid_extractor(
        [{"synthesis_method": None}, {"steps": []}, {"precursors": []}, {"properties": []}]
    )
    assert extractor.rule_extractor.version in extractor.version
    assert extractor.ner_extractor.version in extractor.version
    assert extractor.llm_extractor.version in extractor.version
