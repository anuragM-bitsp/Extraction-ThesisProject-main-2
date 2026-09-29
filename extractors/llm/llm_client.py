"""
LLM client boundary for the Prompt + LLM extractor.

Same pattern as every other external-service boundary in this project
(GrobidClient, EmbeddingModel, NerModel): an interface, a real
implementation (`AnthropicLLMClient`) and an offline, scriptable fake
(`FakeLLMClient`) for tests.

The client's only job is: (system prompt, user prompt) -> raw text. Turning
that text into JSON (`extractors.llm.json_parsing`) and then into the
project schema (`extractors.llm.mapping`) are separate, independently
testable steps. Keeping the client schema-agnostic is what lets the
extraction prompt be changed freely — a new prompt asking for a different
JSON shape needs no code change here.

Why this replaced the old tool-use client
-----------------------------------------
The previous `AnthropicLLMClient` forced a tool call (`tool_choice`) with
`max_tokens=1024` and swallowed every API error inside `call_with_retry`.
On current Claude models that combination fails (forced tool use is not
allowed while thinking is on, sampling parameters are fixed, and 1024
tokens is too small for a full recipe), and because the errors were
swallowed the UI only ever saw an empty result or an opaque failure. This
client asks for plain JSON text, streams the response, and classifies
errors so that permanent ones (bad key, unknown model, bad request) are
reported immediately and clearly instead of being retried and hidden.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_LLM_MODEL = "claude-sonnet-5-5"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TIMEOUT_SECONDS = 600.0


# ---- errors -------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Base class for every error raised by an LLM client."""


class LLMTransportError(LLMError):
    """Network / timeout / rate limit / 5xx. Retryable."""


class LLMOutputInvalid(LLMError):
    """The model answered but the answer was not usable (e.g. not JSON).
    Retryable — another sample usually succeeds."""


class LLMPermanentError(LLMError):
    """Retrying cannot help: invalid API key, unknown model, malformed
    request, output cut off by max_tokens, etc. Surfaced to the user as-is."""


class LLMExtractionError(LLMError):
    """Raised once all retries are exhausted."""


# ---- response ------------------------------------------------------------------------


@dataclass
class LLMResponse:
    text: str
    model: str
    stop_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    raw: dict = field(default_factory=dict)


# ---- interface --------------------------------------------------------------------------


class LLMClient(ABC):
    name: str = "llm"

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return the model's text answer. Must raise LLMTransportError for
        retryable transport failures and LLMPermanentError for failures a
        retry cannot fix."""
        ...


# ---- Anthropic ------------------------------------------------------------------------


class AnthropicLLMClient(LLMClient):
    """
    Production client for the Anthropic Messages API.

    - No `temperature`/`top_p`/`top_k`: current Claude models reject
      non-default sampling parameters with a 400.
    - No forced `tool_choice`: incompatible with (adaptive) thinking.
    - Streaming: required by the SDK for large `max_tokens` values, and it
      avoids HTTP timeouts on long papers.
    - Only `text` content blocks are returned; `thinking` blocks are ignored.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        http_client: Any = None,
    ):
        import anthropic  # local import: optional dependency

        self._anthropic = anthropic
        # max_retries=0: retries are handled (and made visible) by
        # PromptLlmExtractor, not silently inside the SDK.
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout, "max_retries": 0}
        if http_client is not None:  # tests inject an httpx.Client with a mock transport
            kwargs["http_client"] = http_client
        self._client = anthropic.Anthropic(**kwargs)
        self.model = model
        self.name = model
        self.max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        anthropic = self._anthropic
        try:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                message = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMPermanentError("Anthropic rejected the API key (401). Check the key in the sidebar.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMPermanentError(f"API key is not allowed to use model {self.model!r}: {_api_message(exc)}") from exc
        except anthropic.NotFoundError as exc:
            raise LLMPermanentError(
                f"Model {self.model!r} was not found. Pick another model (use 'Load models' in the sidebar). "
                f"Details: {_api_message(exc)}"
            ) from exc
        except anthropic.BadRequestError as exc:
            raise LLMPermanentError(f"Anthropic rejected the request (400): {_api_message(exc)}") from exc
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as exc:
            raise LLMTransportError(f"Temporary Anthropic API problem: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500 or exc.status_code == 429:
                raise LLMTransportError(f"Temporary Anthropic API problem: {exc}") from exc
            raise LLMPermanentError(f"Anthropic API error ({exc.status_code}): {_api_message(exc)}") from exc
        except anthropic.APIError as exc:
            raise LLMTransportError(f"Anthropic API error: {exc}") from exc

        text = "".join(getattr(b, "text", "") for b in message.content if getattr(b, "type", None) == "text")
        usage = getattr(message, "usage", None)
        return LLMResponse(
            text=text,
            model=getattr(message, "model", self.model),
            stop_reason=getattr(message, "stop_reason", None),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )


def _api_message(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return str(exc)


def list_anthropic_models(api_key: str) -> list[dict[str, str]]:
    """Models this API key can use, newest first: [{'id', 'display_name'}].
    Used by the UI so nobody has to guess model IDs."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, max_retries=1, timeout=30.0)
    try:
        page = client.models.list(limit=100)
    except anthropic.AuthenticationError as exc:
        raise LLMPermanentError("Anthropic rejected the API key (401).") from exc
    except anthropic.APIError as exc:
        raise LLMTransportError(f"Could not list models: {exc}") from exc
    return [{"id": m.id, "display_name": getattr(m, "display_name", m.id) or m.id} for m in page.data]


# ---- fake (tests / offline) ---------------------------------------------------------------


class FakeLLMClient(LLMClient):
    """
    Offline, deterministic, scriptable. Each `complete()` pops the next item:
      - str            -> returned as the response text
      - dict / list    -> JSON-encoded and returned as the response text
      - LLMResponse    -> returned as-is
      - Exception      -> raised
    Every call is recorded in `.calls` as (system_prompt, user_prompt).
    """

    def __init__(self, scripted_responses: list[Any], name: str = "fake-llm"):
        self._queue = list(scripted_responses)
        self.calls: list[tuple[str, str]] = []
        self.name = name
        self.model = name

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls.append((system_prompt, user_prompt))
        if not self._queue:
            raise LLMTransportError("FakeLLMClient: scripted response queue exhausted")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResponse):
            return item
        if isinstance(item, (dict, list)):
            item = json.dumps(item)
        return LLMResponse(text=str(item), model=self.name, stop_reason="end_turn")
