"""
Target schema for nanoparticle-synthesis extraction (LLD section 1 / doc 2 section 1).

Design rule this file exists to enforce: ALL FOUR extraction strategies
(rule-based, NER, Prompt+LLM, hybrid) must produce an instance of
`SynthesisExtraction`. If they don't share this schema, the four-way
ablation study in your thesis isn't comparing like with like.

This module has two kinds of models:
  - Value models (Quantity, Precursor, ...) — the actual predicted content.
  - `SynthesisExtraction` — the top-level document the extractors return.

Provenance (CandidateFact/EvidenceSpan) is deliberately NOT embedded inside
these value models. Keeping "what is the value" separate from "where did
each field come from" means the same SynthesisExtraction type can represent
a gold annotation, a raw model prediction, or a post-fusion result — see
ExtractionResult at the bottom, which is the only place the two are joined.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from schemas.provenance import CandidateFact

SCHEMA_VERSION = "v2"  # v2 (Step 9): added canonical_id to Precursor/Material
                        # for entity normalization. Bumping this is the whole
                        # point of versioning a schema (LLD section 23) — it's
                        # what lets an experiment record "extracted against
                        # schema v1" vs "v2" instead of silently comparing
                        # incompatible shapes later.


class Quantity(BaseModel):
    """
    A numeric value + unit, e.g. 80 degC or 10 mM.

    Kept as (value, unit) rather than a single normalized number because:
      1. We want to preserve exactly what the paper said (provenance).
      2. Unit normalization for comparison (0.5 M == 500 mM) is an
         *evaluation-time* concern (LLD doc 2, section 23), not an
         extraction-time one — normalizing too early can silently destroy
         information if a unit was misread.
    """

    value: float
    unit: str


class Precursor(BaseModel):
    name: str
    amount: Optional[Quantity] = None
    concentration: Optional[Quantity] = None
    canonical_id: Optional[str] = Field(
        default=None,
        description="Set by Step 9's entity normalization once a surface form is "
        "matched against the ontology (e.g. 'AgNO3' -> 'silver_nitrate'). None "
        "means either normalization hasn't run yet, or the name isn't in the "
        "fixed ontology — both are valid, distinguishable only by whether this "
        "extraction went through normalize_extraction_result().",
    )
    canonical_id: Optional[str] = Field(
        default=None,
        description="Set by Step 9's entity normalization — e.g. 'AgNO3' and "
        "'silver nitrate' both resolve to canonical_id='AGNO3'. None until "
        "normalization has run; extractors themselves never set this.",
    )


class Material(BaseModel):
    name: Optional[str] = None
    composition: Optional[str] = None
    size: Optional[Quantity] = None
    morphology: Optional[str] = None
    canonical_id: Optional[str] = None
    canonical_id: Optional[str] = Field(default=None, description="Set by Step 9's entity normalization; see Precursor.canonical_id.")


class SynthesisStep(BaseModel):
    order: int = Field(ge=1)
    description: str


class CharacterizationMethod(BaseModel):
    technique: str  # e.g. "TEM", "XRD", "UV-Vis"
    finding: Optional[str] = None


class MaterialProperty(BaseModel):
    name: str  # e.g. "band gap", "zeta potential"
    value: Optional[Quantity] = None
    qualitative_value: Optional[str] = None


class SynthesisExtraction(BaseModel):
    """The structured recipe extracted from (or annotated for) one paper."""

    paper_id: str
    schema_version: str = SCHEMA_VERSION

    material: Optional[Material] = None
    precursors: list[Precursor] = Field(default_factory=list)
    solvent: Optional[str] = None
    temperature: Optional[Quantity] = None
    reaction_time: Optional[Quantity] = None
    pH: Optional[float] = None
    synthesis_method: Optional[str] = None
    steps: list[SynthesisStep] = Field(default_factory=list)
    characterization: list[CharacterizationMethod] = Field(default_factory=list)
    properties: list[MaterialProperty] = Field(default_factory=list)


class ExtractorName(str, Enum):
    RULE = "RULE"
    NER = "NER"
    LLM = "LLM"  # Prompt + LLM (full document + editable prompt, no retrieval)
    RAG_LLM = "RAG_LLM"  # legacy: kept only so previously stored results still load
    HYBRID = "HYBRID"
    HUMAN = "HUMAN"


class LlmRunInfo(BaseModel):
    """
    Everything needed to reproduce / audit one Prompt + LLM call: which
    model, which exact prompt, and the model's full JSON answer. `output` is
    kept verbatim even when the prompt asks for fields outside
    SynthesisExtraction — `prediction` only holds the schema-mappable part.
    """

    model: str
    prompt_name: Optional[str] = None
    prompt_sha: str
    prompt: str
    output: Any = None
    stop_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    document_truncated: bool = False
    attempts: int = 1


class ExtractionResult(BaseModel):
    """
    What an Extractor.extract(...) call returns (see Extractor interface,
    LLD section 7): the predicted schema instance, PLUS full provenance for
    every field that was populated.

    This is the object that gets persisted, compared against gold, and fed
    into the fusion/evaluation layers in later steps.
    """

    paper_id: str
    version: int = Field(ge=1)
    extractor: ExtractorName
    extractor_version: str
    schema_version: str = SCHEMA_VERSION

    prediction: SynthesisExtraction
    provenance: list[CandidateFact] = Field(
        default_factory=list,
        description="One CandidateFact per populated field/sub-field in `prediction`.",
    )
    llm_run: Optional[LlmRunInfo] = Field(
        default=None,
        description="Set by the Prompt + LLM extractor (and Hybrid, which embeds it): model, "
        "prompt and the raw JSON the LLM returned. None for Rule/NER results.",
    )
