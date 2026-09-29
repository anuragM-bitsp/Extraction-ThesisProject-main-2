"""
GazetteerNER: dictionary + word-boundary matching over a fixed term list.

NOT a claim that dictionary lookup is a good substitute for a trained
domain NER model — it obviously isn't (LLD section 12 calls for
fine-tuning/applying a real model). It IS a genuine, deterministic,
offline-testable NerModel implementation, which is what lets the rest of
this step (entity linking, NerExtractor, schema population) be built and
tested without Hugging Face Hub access. Swap in TransformerNER once
deployed somewhere with model access; both implement `NerModel`.

Known limitations, stated plainly rather than papered over:
  - Fixed vocabulary: any precursor/solvent/material/technique not in
    `_ENTITY_TERMS` is invisible to this implementation. A trained model
    generalizes to novel chemical names; this doesn't.
  - No entity normalization: "AgNO3" and "silver nitrate" are two
    different dictionary entries and will surface as two separate
    entities even when they refer to the same compound. Canonicalizing
    synonymous surface forms is Step 9's job (entity normalization/linking
    to canonical IDs), not this one's.
"""

from __future__ import annotations

from extractors.ner.base import Entity, NerModel
from extractors.ner.labels import EntityLabel

_ENTITY_TERMS: list[tuple[str, EntityLabel]] = [
    # Precursors (nanoparticle-synthesis literature)
    ("silver nitrate", EntityLabel.PRECURSOR),
    ("AgNO3", EntityLabel.PRECURSOR),
    ("gold chloride", EntityLabel.PRECURSOR),
    ("chloroauric acid", EntityLabel.PRECURSOR),
    ("HAuCl4", EntityLabel.PRECURSOR),
    ("sodium borohydride", EntityLabel.PRECURSOR),
    ("NaBH4", EntityLabel.PRECURSOR),
    ("ferric chloride", EntityLabel.PRECURSOR),
    ("FeCl3", EntityLabel.PRECURSOR),
    ("zinc acetate", EntityLabel.PRECURSOR),
    ("copper sulfate", EntityLabel.PRECURSOR),
    ("CuSO4", EntityLabel.PRECURSOR),
    ("trisodium citrate", EntityLabel.PRECURSOR),
    ("sodium citrate", EntityLabel.PRECURSOR),
    ("polyvinylpyrrolidone", EntityLabel.PRECURSOR),
    ("PVP", EntityLabel.PRECURSOR),
    # Solvents
    ("deionized water", EntityLabel.SOLVENT),
    ("DI water", EntityLabel.SOLVENT),
    ("distilled water", EntityLabel.SOLVENT),
    ("ethanol", EntityLabel.SOLVENT),
    ("methanol", EntityLabel.SOLVENT),
    ("isopropanol", EntityLabel.SOLVENT),
    ("acetone", EntityLabel.SOLVENT),
    ("dimethylformamide", EntityLabel.SOLVENT),
    ("DMF", EntityLabel.SOLVENT),
    ("dimethyl sulfoxide", EntityLabel.SOLVENT),
    ("DMSO", EntityLabel.SOLVENT),
    ("toluene", EntityLabel.SOLVENT),
    ("chloroform", EntityLabel.SOLVENT),
    ("water", EntityLabel.SOLVENT),
    # Materials (the synthesized product)
    ("silver nanoparticles", EntityLabel.MATERIAL),
    ("AgNPs", EntityLabel.MATERIAL),
    ("gold nanoparticles", EntityLabel.MATERIAL),
    ("AuNPs", EntityLabel.MATERIAL),
    ("zinc oxide nanoparticles", EntityLabel.MATERIAL),
    ("ZnO nanoparticles", EntityLabel.MATERIAL),
    ("iron oxide nanoparticles", EntityLabel.MATERIAL),
    ("copper nanoparticles", EntityLabel.MATERIAL),
    # Characterization techniques
    ("TEM", EntityLabel.CHARACTERIZATION),
    ("SEM", EntityLabel.CHARACTERIZATION),
    ("XRD", EntityLabel.CHARACTERIZATION),
    ("UV-Vis", EntityLabel.CHARACTERIZATION),
    ("FTIR", EntityLabel.CHARACTERIZATION),
    ("DLS", EntityLabel.CHARACTERIZATION),
    ("XPS", EntityLabel.CHARACTERIZATION),
    ("EDX", EntityLabel.CHARACTERIZATION),
    ("zeta potential", EntityLabel.CHARACTERIZATION),
]

# Longer phrases are tried first so "silver nitrate" wins over any
# coincidental substring match, and "deionized water" wins over "water".
_SORTED_TERMS = sorted(_ENTITY_TERMS, key=lambda t: -len(t[0]))


def _is_word_char(ch: str) -> bool:
    return ch.isalnum()


def _overlaps(span: tuple[int, int], used: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < u_end and end > u_start for u_start, u_end in used)


def find_entities(text: str) -> list[Entity]:
    lower_text = text.lower()
    used_spans: list[tuple[int, int]] = []
    entities: list[Entity] = []

    for term, label in _SORTED_TERMS:
        term_lower = term.lower()
        search_from = 0
        while True:
            idx = lower_text.find(term_lower, search_from)
            if idx == -1:
                break
            end = idx + len(term)
            search_from = idx + 1  # allow overlapping *candidate* search; overlap resolved below

            before_ok = idx == 0 or not _is_word_char(text[idx - 1])
            after_ok = end == len(text) or not _is_word_char(text[end])
            if before_ok and after_ok and not _overlaps((idx, end), used_spans):
                entities.append(Entity(text=text[idx:end], label=label, char_start=idx, char_end=end))
                used_spans.append((idx, end))

    entities.sort(key=lambda e: e.char_start)
    return entities


class GazetteerNER(NerModel):
    name = "gazetteer-v1"

    def predict(self, text: str) -> list[Entity]:
        return find_entities(text)
