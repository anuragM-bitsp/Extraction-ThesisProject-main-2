"""
Ontology linking (LLD section 16):

    "Where possible, I would link normalized entities to domain ontologies
    or external identifiers. This turns the dataset from a collection of
    strings into a machine-readable scientific knowledge representation."

Same pattern as every other external-service boundary in this project: an
interface, an offline implementation that's genuinely functional (looks up
the CAS number already baked into our own canonical dictionary), and a
production implementation that queries a real external service and isn't
exercised by the test suite because that service isn't reachable from this
sandbox.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from normalization.canonical_dictionary import CANONICAL_ENTITIES, CanonicalEntity


class ExternalIdentifier(BaseModel):
    system: str  # e.g. "CAS", "PubChem"
    identifier: str


class OntologyLinker(ABC):
    @abstractmethod
    def link(self, canonical_id: str) -> ExternalIdentifier | None: ...


class LocalCasLinker(OntologyLinker):
    """Looks up the CAS registry number already stored on our own
    CanonicalEntity dictionary. Genuinely functional and network-free, but
    limited to the handful of compounds that dictionary knows about."""

    def __init__(self, entities: list[CanonicalEntity] | None = None):
        entities = entities if entities is not None else CANONICAL_ENTITIES
        self._by_id = {e.canonical_id: e for e in entities}

    def link(self, canonical_id: str) -> ExternalIdentifier | None:
        entity = self._by_id.get(canonical_id)
        if entity is None or entity.cas_number is None:
            return None
        return ExternalIdentifier(system="CAS", identifier=entity.cas_number)


class PubChemLinker(OntologyLinker):
    """
    Production: queries PubChem's PUG REST API for a Compound ID (CID) by
    name. Not exercised by the test suite — pubchem.ncbi.nlm.nih.gov isn't
    in this sandbox's network allow-list. Swap this in for LocalCasLinker
    once deployed somewhere with outbound internet access; both implement
    `OntologyLinker`.
    """

    def __init__(self, base_url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"):
        self.base_url = base_url

    def link(self, canonical_name_or_id: str) -> ExternalIdentifier | None:
        import requests  # local import: optional dep, only needed for this path

        resp = requests.get(
            f"{self.base_url}/compound/name/{canonical_name_or_id}/cids/JSON", timeout=10
        )
        if resp.status_code != 200:
            return None
        cids = resp.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            return None
        return ExternalIdentifier(system="PubChem", identifier=str(cids[0]))
