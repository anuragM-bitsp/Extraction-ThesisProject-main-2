"""
Canonical entity dictionary.

A small internal stand-in for a real chemical ontology/database lookup
(PubChem, ChemSpider, CAS) — this sandbox's network allow-list doesn't
include any of those services, so this is the offline-testable source of
truth `DictionaryNormalizer` matches against, in the same spirit as
`extractors/ner/gazetteer.py`'s term list. Covers every PRECURSOR, SOLVENT,
and MATERIAL entry that gazetteer recognizes (verified by a coverage check
in tests/test_normalization.py), since normalization exists specifically to
collapse the duplicate entries that gazetteer can produce (see Step 6/8's
documented "AgNO3 vs silver nitrate" limitation). CHARACTERIZATION entries
(TEM, XRD, ...) are deliberately excluded — they're technique acronyms, not
chemical entities, and have no CAS number or synonym-duplication problem to
resolve.
"""

from __future__ import annotations

from pydantic import BaseModel


class CanonicalEntity(BaseModel):
    canonical_id: str
    canonical_name: str
    synonyms: list[str]
    cas_number: str | None = None


CANONICAL_ENTITIES: list[CanonicalEntity] = [
    CanonicalEntity(canonical_id="AGNO3", canonical_name="Silver nitrate",
                     synonyms=["silver nitrate", "agno3"], cas_number="7761-88-8"),
    CanonicalEntity(canonical_id="NABH4", canonical_name="Sodium borohydride",
                     synonyms=["sodium borohydride", "nabh4"], cas_number="16940-66-2"),
    CanonicalEntity(canonical_id="HAUCL4", canonical_name="Chloroauric acid",
                     synonyms=["chloroauric acid", "gold chloride", "haucl4"], cas_number="16903-35-8"),
    CanonicalEntity(canonical_id="FECL3", canonical_name="Ferric chloride",
                     synonyms=["ferric chloride", "fecl3"], cas_number="7705-08-0"),
    CanonicalEntity(canonical_id="ZNOAC", canonical_name="Zinc acetate",
                     synonyms=["zinc acetate"], cas_number="557-34-6"),
    CanonicalEntity(canonical_id="CUSO4", canonical_name="Copper sulfate",
                     synonyms=["copper sulfate", "cuso4"], cas_number="7758-98-7"),
    CanonicalEntity(canonical_id="NA3CIT", canonical_name="Trisodium citrate",
                     synonyms=["trisodium citrate", "sodium citrate"], cas_number="6132-04-3"),
    CanonicalEntity(canonical_id="PVP", canonical_name="Polyvinylpyrrolidone",
                     synonyms=["polyvinylpyrrolidone", "pvp"], cas_number="9003-39-8"),
    CanonicalEntity(canonical_id="ETOH", canonical_name="Ethanol",
                     synonyms=["ethanol"], cas_number="64-17-5"),
    CanonicalEntity(canonical_id="MEOH", canonical_name="Methanol",
                     synonyms=["methanol"], cas_number="67-56-1"),
    CanonicalEntity(canonical_id="H2O", canonical_name="Deionized water",
                     synonyms=["deionized water", "di water", "distilled water", "water"], cas_number="7732-18-5"),
    CanonicalEntity(canonical_id="ISOPROP", canonical_name="Isopropanol",
                     synonyms=["isopropanol"], cas_number="67-63-0"),
    CanonicalEntity(canonical_id="ACETONE", canonical_name="Acetone",
                     synonyms=["acetone"], cas_number="67-64-1"),
    CanonicalEntity(canonical_id="DMF", canonical_name="Dimethylformamide",
                     synonyms=["dimethylformamide", "dmf"], cas_number="68-12-2"),
    CanonicalEntity(canonical_id="DMSO", canonical_name="Dimethyl sulfoxide",
                     synonyms=["dimethyl sulfoxide", "dmso"], cas_number="67-68-5"),
    CanonicalEntity(canonical_id="TOLUENE", canonical_name="Toluene",
                     synonyms=["toluene"], cas_number="108-88-3"),
    CanonicalEntity(canonical_id="CHLOROFORM", canonical_name="Chloroform",
                     synonyms=["chloroform"], cas_number="67-66-3"),
    CanonicalEntity(canonical_id="AGNP", canonical_name="Silver nanoparticles",
                     synonyms=["silver nanoparticles", "agnps"]),
    CanonicalEntity(canonical_id="AUNP", canonical_name="Gold nanoparticles",
                     synonyms=["gold nanoparticles", "aunps"]),
    CanonicalEntity(canonical_id="ZNONP", canonical_name="Zinc oxide nanoparticles",
                     synonyms=["zinc oxide nanoparticles", "zno nanoparticles"]),
    CanonicalEntity(canonical_id="FEOXNP", canonical_name="Iron oxide nanoparticles",
                     synonyms=["iron oxide nanoparticles"]),
    CanonicalEntity(canonical_id="CUNP", canonical_name="Copper nanoparticles",
                     synonyms=["copper nanoparticles"]),
]
