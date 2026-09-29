"""
build_extractor(): ExperimentConfig -> a real Extractor instance.

Deliberately takes `llm_client` as an explicit argument rather than
constructing it from `config.llm_model_name`. That string is for logging
(see ExperimentConfig's docstring); turning it into a live
`AnthropicLLMClient` — or a `FakeLLMClient` for a test run — is a
deployment/test decision the caller already has to make.
"""

from __future__ import annotations

from experiment.config import ExperimentConfig
from extractors.base import Extractor
from extractors.hybrid_extractor import HybridExtractor
from extractors.llm.llm_client import LLMClient
from extractors.llm_extractor import PromptLlmExtractor
from extractors.ner_extractor import NerExtractor
from extractors.rule_extractor import RuleExtractor


def build_extractor(config: ExperimentConfig, llm_client: LLMClient | None = None) -> Extractor:
    if config.extractor == "rule":
        return RuleExtractor()

    if config.extractor == "ner":
        return NerExtractor()

    if config.extractor in ("llm", "rag_llm", "hybrid"):
        if llm_client is None:
            raise ValueError(f"extractor={config.extractor!r} requires an `llm_client`")
        llm_extractor = PromptLlmExtractor(
            llm_client=llm_client,
            prompt=config.prompt,
            prompt_name=config.prompt_name,
            max_retries=config.max_retries,
        )
        if config.extractor == "hybrid":
            return HybridExtractor(llm_extractor=llm_extractor)
        return llm_extractor

    raise ValueError(f"unknown extractor kind: {config.extractor!r}")
