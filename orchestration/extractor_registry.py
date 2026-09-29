"""
get_extractor(): a plain string (as it arrives over HTTP, through a Celery
task's JSON arguments, or from the Streamlit UI) -> a real `Extractor`.

RULE and NER need nothing and always work offline. LLM ("Prompt + LLM")
and HYBRID need an Anthropic API key; without one `get_extractor` raises a
clear, actionable error instead of returning something that would fail
deep inside extraction.

Configuration (all optional):
    ANTHROPIC_API_KEY       used when no session key is supplied
    LLM_MODEL               model id (default: DEFAULT_LLM_MODEL)
    LLM_MAX_TOKENS          max output tokens (default: 16000)
    EXTRACTION_PROMPT_FILE  path to a prompt .txt (default: prompts/synthesis_extraction.txt)

"rag_llm" is accepted as a legacy alias for "llm" so existing API clients
and experiment configs keep working.
"""

from __future__ import annotations

import os
from typing import Optional

from extractors.base import Extractor
from extractors.hybrid_extractor import HybridExtractor
from extractors.llm.llm_client import DEFAULT_LLM_MODEL, DEFAULT_MAX_TOKENS, AnthropicLLMClient
from extractors.llm_extractor import PromptLlmExtractor
from extractors.ner_extractor import NerExtractor
from extractors.rule_extractor import RuleExtractor

LLM_EXTRACTOR_ALIASES = {"llm", "prompt_llm", "rag_llm"}


class ExtractorUnavailableError(RuntimeError):
    pass


def resolve_anthropic_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Session-supplied key wins; otherwise the process environment. Never logs the value."""
    if api_key and api_key.strip():
        return api_key.strip()
    env = os.environ.get("ANTHROPIC_API_KEY")
    return env.strip() if env and env.strip() else None


def configured_llm_model(model: Optional[str] = None) -> str:
    if model and model.strip():
        return model.strip()
    return os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL


def configured_max_tokens(max_tokens: Optional[int] = None) -> int:
    if max_tokens:
        return int(max_tokens)
    env = os.environ.get("LLM_MAX_TOKENS")
    return int(env) if env else DEFAULT_MAX_TOKENS


def canonical_extractor_id(name: str) -> str:
    name = (name or "").strip().lower()
    return "llm" if name in LLM_EXTRACTOR_ALIASES else name


def extractor_status(api_key: Optional[str] = None, model: Optional[str] = None) -> list[dict]:
    """Which extractors `get_extractor` can currently construct — used by the UI."""
    has_key = bool(resolve_anthropic_api_key(api_key))
    llm_model = configured_llm_model(model)
    missing_key = "Anthropic API key is not configured."

    return [
        {
            "id": "rule",
            "label": "Rule-based",
            "available": True,
            "requires_api_key": False,
            "implementation": "RuleExtractor",
            "reason": None,
        },
        {
            "id": "ner",
            "label": "NER",
            "available": True,
            "requires_api_key": False,
            "implementation": "NerExtractor",
            "ner_implementation": "GazetteerNER",
            "mode": "Offline",
            "reason": None,
        },
        {
            "id": "llm",
            "label": "Prompt + LLM",
            "available": has_key,
            "requires_api_key": True,
            "implementation": "PromptLlmExtractor",
            "llm_model": llm_model,
            "reason": None if has_key else missing_key,
        },
        {
            "id": "hybrid",
            "label": "Hybrid",
            "available": has_key,
            "requires_api_key": True,
            "implementation": "HybridExtractor",
            "llm_model": llm_model,
            "reason": None if has_key else missing_key,
        },
    ]


def describe_extractor(extractor: Extractor) -> dict:
    """Report configuration from a live extractor instance — no invented fields."""
    info: dict = {
        "class_name": type(extractor).__name__,
        "name": extractor.name.value if hasattr(extractor.name, "value") else str(extractor.name),
        "version": extractor.version,
    }
    if isinstance(extractor, NerExtractor):
        model = extractor.model
        info["ner_implementation"] = type(model).__name__
        info["ner_model_name"] = getattr(model, "name", type(model).__name__)
        info["mode"] = "Offline" if type(model).__name__ == "GazetteerNER" else "Model"
    if isinstance(extractor, PromptLlmExtractor):
        info["llm"] = extractor.model_name
        info["prompt_name"] = extractor.prompt_name
        info["prompt_sha"] = extractor.prompt_sha
    if isinstance(extractor, HybridExtractor):
        info["rule_version"] = extractor.rule_extractor.version
        info["ner_version"] = extractor.ner_extractor.version
        info["llm_extractor"] = describe_extractor(extractor.llm_extractor)
        info["llm"] = extractor.llm_extractor.model_name
        info["prompt_name"] = extractor.llm_extractor.prompt_name
        info["prompt_sha"] = extractor.llm_extractor.prompt_sha
        info["resolver"] = type(extractor.resolver).__name__
    return info


def get_extractor(
    name: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
    prompt_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Extractor:
    name = canonical_extractor_id(name)

    if name == "rule":
        return RuleExtractor()
    if name == "ner":
        return NerExtractor()

    if name in ("llm", "hybrid"):
        resolved_key = resolve_anthropic_api_key(api_key)
        if not resolved_key:
            raise ExtractorUnavailableError(
                f"extractor={name!r} requires an Anthropic API key. Enter it in the sidebar and click "
                "'Apply' (or set ANTHROPIC_API_KEY). For offline tests, construct PromptLlmExtractor "
                "with FakeLLMClient directly."
            )
        llm_model = configured_llm_model(model)
        try:
            llm_client = AnthropicLLMClient(
                model=llm_model, api_key=resolved_key, max_tokens=configured_max_tokens(max_tokens)
            )
        except ImportError as exc:
            raise ExtractorUnavailableError(
                "The 'anthropic' package is not installed. Run: pip install -r requirements.txt"
            ) from exc
        except Exception as exc:
            raise ExtractorUnavailableError(f"AnthropicLLMClient could not be constructed: {exc}") from exc

        llm_extractor = PromptLlmExtractor(llm_client=llm_client, prompt=prompt, prompt_name=prompt_name)
        return llm_extractor if name == "llm" else HybridExtractor(llm_extractor=llm_extractor)

    raise ValueError(f"unknown extractor name: {name!r} (expected one of: rule, ner, llm, hybrid)")
