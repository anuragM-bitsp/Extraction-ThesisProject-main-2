from __future__ import annotations

from extractors.ner.gazetteer import _ENTITY_TERMS
from extractors.ner.labels import EntityLabel
from ingestion.canonical import BlockType, CanonicalBlock, CanonicalDocument
from normalization.canonical_dictionary import CANONICAL_ENTITIES
from normalization.linking import LocalCasLinker
from normalization.normalizer import DictionaryNormalizer
from normalization.pipeline import normalize_extraction
from extractors.ner_extractor import NerExtractor
from retrieval.embeddings import HashingEmbedder
from schemas.extraction_schema import ExtractionResult, ExtractorName, Precursor, SynthesisExtraction

# ---- dictionary coverage (regression guard) ------------------------------------------


def test_canonical_dictionary_covers_every_chemical_entity_the_gazetteer_recognizes():
    """CHARACTERIZATION terms (TEM, XRD, ...) are technique acronyms, not
    chemical entities — they're intentionally excluded from the ontology.
    Every PRECURSOR/SOLVENT/MATERIAL term the gazetteer can produce must
    have a normalization entry, or a real gazetteer hit would silently fail
    to normalize. This test caught exactly that gap once already (acetone,
    DMF, DMSO, toluene, chloroform, isopropanol, and three nanoparticle
    materials were missing) — it stays here to stop it recurring."""
    chemical_terms = {t.lower() for t, label in _ENTITY_TERMS if label != EntityLabel.CHARACTERIZATION}

    covered = set()
    for entity in CANONICAL_ENTITIES:
        covered.add(entity.canonical_name.lower())
        covered.update(s.lower() for s in entity.synonyms)

    missing = chemical_terms - covered
    assert not missing, f"canonical_dictionary.py is missing: {sorted(missing)}"


# ---- DictionaryNormalizer -----------------------------------------------------------


def test_exact_alias_match_is_case_and_whitespace_insensitive():
    normalizer = DictionaryNormalizer()
    result = normalizer.normalize("  Silver Nitrate  ")
    assert result is not None
    assert result.canonical_id == "AGNO3"
    assert result.match_method == "exact_alias"
    assert result.confidence == 1.0


def test_synonymous_surface_forms_resolve_to_the_same_canonical_id():
    normalizer = DictionaryNormalizer()
    a = normalizer.normalize("silver nitrate")
    b = normalizer.normalize("AgNO3")
    assert a.canonical_id == b.canonical_id == "AGNO3"


def test_unknown_name_without_embedder_returns_none():
    normalizer = DictionaryNormalizer(embedder=None)
    assert normalizer.normalize("some completely novel compound xyz123") is None


def test_embedding_fallback_catches_near_miss_not_in_dictionary():
    """'silver nitrate solution' isn't a listed alias verbatim, but shares
    enough vocabulary with 'Silver nitrate' that the embedding-similarity
    fallback should still resolve it — with a similarity-based confidence,
    not a fixed 1.0."""
    normalizer = DictionaryNormalizer(embedder=HashingEmbedder(dimension=128), similarity_threshold=0.5)
    result = normalizer.normalize("silver nitrate solution")
    assert result is not None
    assert result.canonical_id == "AGNO3"
    assert result.match_method == "embedding_similarity"
    assert 0.5 <= result.confidence < 1.0


def test_embedding_fallback_declines_when_nothing_is_close_enough():
    normalizer = DictionaryNormalizer(embedder=HashingEmbedder(dimension=128), similarity_threshold=0.9)
    assert normalizer.normalize("a sentence entirely unrelated to any known compound") is None


# ---- LocalCasLinker -----------------------------------------------------------------


def test_local_cas_linker_returns_known_cas_number():
    linker = LocalCasLinker()
    result = linker.link("AGNO3")
    assert result is not None
    assert result.system == "CAS"
    assert result.identifier == "7761-88-8"


def test_local_cas_linker_returns_none_for_entity_without_cas_number():
    linker = LocalCasLinker()
    assert linker.link("AGNP") is None  # nanoparticle material, no CAS number in our dictionary


def test_local_cas_linker_returns_none_for_unknown_id():
    assert LocalCasLinker().link("NOT_A_REAL_ID") is None


# ---- normalize_extraction (full pipeline) --------------------------------------------


def _result_with_duplicate_precursors() -> ExtractionResult:
    prediction = SynthesisExtraction(
        paper_id="P001",
        precursors=[
            Precursor(name="Silver nitrate", concentration=None, amount=None),
            Precursor(name="AgNO3", concentration={"value": 10.0, "unit": "mM"}, amount=None),
        ],
    )
    return ExtractionResult(
        paper_id="P001", version=1, extractor=ExtractorName.HYBRID,
        extractor_version="hybrid@test", prediction=prediction, provenance=[],
    )


def test_normalize_extraction_merges_synonymous_precursors():
    """This is the fix for the gap Steps 6 and 8 both documented and tested
    for without closing: 'AgNO3' and 'Silver nitrate' finally collapse into
    one Precursor."""
    normalizer = DictionaryNormalizer()
    result = normalize_extraction(_result_with_duplicate_precursors(), normalizer)

    assert len(result.prediction.precursors) == 1
    merged = result.prediction.precursors[0]
    assert merged.canonical_id == "AGNO3"
    assert merged.name == "Silver nitrate"  # canonical display name, not either raw surface form


def test_normalize_extraction_backfills_missing_fields_from_the_duplicate():
    """The first-seen 'Silver nitrate' entry had no concentration; the
    duplicate 'AgNO3' entry did. Merging should keep it, not discard it."""
    normalizer = DictionaryNormalizer()
    result = normalize_extraction(_result_with_duplicate_precursors(), normalizer)

    merged = result.prediction.precursors[0]
    assert merged.concentration is not None
    assert merged.concentration.value == 10.0


def test_normalize_extraction_records_the_merge_in_provenance():
    normalizer = DictionaryNormalizer()
    result = normalize_extraction(_result_with_duplicate_precursors(), normalizer)

    merge_facts = [c for c in result.provenance if c.field.startswith("precursors.normalized_merge.")]
    assert len(merge_facts) == 1
    assert merge_facts[0].value["merged_surface_form"] == "AgNO3"
    assert merge_facts[0].value["canonical_name"] == "Silver nitrate"


def test_normalize_extraction_links_external_identifier_when_linker_given():
    normalizer = DictionaryNormalizer()
    linker = LocalCasLinker()
    result = normalize_extraction(_result_with_duplicate_precursors(), normalizer, linker=linker)

    link_facts = [c for c in result.provenance if c.field.startswith("precursors.external_id.")]
    assert len(link_facts) == 1
    assert link_facts[0].value == {"system": "CAS", "identifier": "7761-88-8"}


def test_normalize_extraction_marks_extractor_version_as_normalized():
    normalizer = DictionaryNormalizer()
    result = normalize_extraction(_result_with_duplicate_precursors(), normalizer)
    assert result.extractor_version == "hybrid@test+normalized"
    assert result.extractor == ExtractorName.HYBRID  # unchanged — normalization isn't a 5th strategy


def test_normalize_extraction_leaves_unknown_precursor_names_unmerged_but_intact():
    prediction = SynthesisExtraction(
        paper_id="P001",
        precursors=[Precursor(name="some obscure lab reagent XJ-9", concentration=None, amount=None)],
    )
    result = ExtractionResult(
        paper_id="P001", version=1, extractor=ExtractorName.NER,
        extractor_version="ner@test", prediction=prediction, provenance=[],
    )
    normalized = normalize_extraction(result, DictionaryNormalizer())

    assert len(normalized.prediction.precursors) == 1
    assert normalized.prediction.precursors[0].name == "some obscure lab reagent XJ-9"
    assert normalized.prediction.precursors[0].canonical_id is None


def test_normalize_extraction_normalizes_solvent_and_material():
    prediction = SynthesisExtraction(
        paper_id="P001",
        solvent="DI water",
        material={"name": "AgNPs"},
    )
    result = ExtractionResult(
        paper_id="P001", version=1, extractor=ExtractorName.NER,
        extractor_version="ner@test", prediction=prediction, provenance=[],
    )
    normalized = normalize_extraction(result, DictionaryNormalizer())

    assert normalized.prediction.solvent == "Deionized water"
    assert normalized.prediction.material.name == "Silver nanoparticles"
    assert normalized.prediction.material.canonical_id == "AGNP"


# ---- end-to-end: real gazetteer NER output, actually merged -------------------------


def block(text, page=1, section=None):
    return CanonicalBlock(block_type=BlockType.PARAGRAPH, page_number=page, section=section, text=text)


def test_end_to_end_ner_duplication_is_fixed_by_normalization():
    """The exact scenario Step 6 documented as a known limitation
    ('Silver nitrate' and 'AgNO3' both found by the gazetteer, producing
    two Precursor entries) — now actually resolved."""
    document = CanonicalDocument(
        paper_id="P001",
        version=1,
        blocks=[block("Silver nitrate (AgNO3, 10 mM) was the precursor used in this synthesis.", page=2, section="Experimental")],
    )
    ner_result = NerExtractor().extract(document)
    assert len(ner_result.prediction.precursors) == 2  # still true before normalization

    normalized = normalize_extraction(ner_result, DictionaryNormalizer())
    assert len(normalized.prediction.precursors) == 1
    assert normalized.prediction.precursors[0].canonical_id == "AGNO3"
    assert normalized.prediction.precursors[0].concentration.value == 10.0
