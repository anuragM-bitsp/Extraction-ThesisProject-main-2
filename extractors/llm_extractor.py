"""
PromptLlmExtractor: the "Prompt + LLM" extraction strategy.

Replaces the earlier RAG + LLM extractor. Retrieval was dropped because the
question asked of every paper is always the same and a whole paper fits in
the model's context window, so retrieval added failure modes (embedding
model downloads, chunks missing the relevant sentence, four separate LLM
calls) without adding information.

    CanonicalDocument (full text, page/section markers)
        + editable extraction prompt (prompts/*.txt or the UI text box)
        -> one LLM call
        -> JSON parsing (tolerant of code fences / stray prose)
        -> SynthesisExtraction + CandidateFacts   (extractors/llm/mapping.py)
        -> ExtractionResult (with llm_run: model, prompt, raw JSON)

Everything downstream — hybrid fusion, normalization/entity linking,
evaluation, experiment tracking — consumes the same ExtractionResult as
before.

Errors are never swallowed: a bad API key, unknown model, truncated output
or unparseable answer raises with a clear message (after retries where a
retry can help), so the UI shows the real cause instead of an empty result
— and an empty result is never cached as if it were a success.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from extractors.base import Extractor
from extractors.llm.json_parsing import parse_json_response
from extractors.llm.llm_client import (
    LLMClient,
    LLMExtractionError,
    LLMOutputInvalid,
    LLMPermanentError,
    LLMResponse,
    LLMTransportError,
)
from extractors.llm.mapping import map_llm_output
from extractors.llm.prompting import (
    DEFAULT_MAX_DOCUMENT_CHARS,
    build_prompt,
    load_default_prompt,
    prompt_fingerprint,
)
from ingestion.canonical import CanonicalDocument
from schemas.extraction_schema import ExtractionResult, ExtractorName, LlmRunInfo

_RETRY_FEEDBACK = (
    "\n\nIMPORTANT: your previous answer could not be parsed as JSON ({error}). "
    "Reply again with ONLY the JSON value — no code fences, no explanation."
)


class PromptLlmExtractor(Extractor):
    name = ExtractorName.LLM

    def __init__(
        self,
        llm_client: LLMClient,
        prompt: Optional[str] = None,
        prompt_name: Optional[str] = None,
        max_retries: int = 2,
        max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
        retry_delay_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.llm_client = llm_client
        self.prompt = (prompt if prompt and prompt.strip() else load_default_prompt()).strip()
        self.prompt_name = prompt_name
        self.prompt_sha = prompt_fingerprint(self.prompt)
        self.max_retries = max_retries
        self.max_document_chars = max_document_chars
        self.retry_delay_seconds = retry_delay_seconds
        self._sleep = sleep
        self.model_name = getattr(llm_client, "name", llm_client.__class__.__name__)
        # The prompt hash is part of the version: PipelineService caches
        # results by (extractor, extractor_version), so a changed prompt
        # must be a different version or the old answer would be returned.
        self.version = f"llm@{self.model_name}+prompt-{self.prompt_sha}"

    # ------------------------------------------------------------------------------

    def extract(self, document: CanonicalDocument) -> ExtractionResult:
        system_prompt, user_prompt, truncated = build_prompt(
            self.prompt, document, max_chars=self.max_document_chars
        )
        if not document.full_text().strip():
            raise LLMPermanentError(
                "The document has no extractable text (scanned PDF without OCR?), so there is nothing to send to the LLM."
            )

        response, data, attempts = self._call_with_retry(system_prompt, user_prompt)
        prediction, facts = map_llm_output(data, document, self.version)

        return ExtractionResult(
            paper_id=document.paper_id,
            version=document.version,
            extractor=self.name,
            extractor_version=self.version,
            prediction=prediction,
            provenance=facts,
            llm_run=LlmRunInfo(
                model=response.model or self.model_name,
                prompt_name=self.prompt_name,
                prompt_sha=self.prompt_sha,
                prompt=self.prompt,
                output=data,
                stop_reason=response.stop_reason,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                document_truncated=truncated,
                attempts=attempts,
            ),
        )

    # ------------------------------------------------------------------------------

    def _call_with_retry(self, system_prompt: str, user_prompt: str) -> tuple[LLMResponse, object, int]:
        last_error: Exception | None = None
        prompt = user_prompt
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                response = self.llm_client.complete(system_prompt, prompt)
                self._check_stop_reason(response)
                data = parse_json_response(response.text)
                return response, data, attempt
            except LLMPermanentError:
                raise
            except LLMOutputInvalid as exc:
                last_error = exc
                prompt = user_prompt + _RETRY_FEEDBACK.format(error=str(exc)[:200])
            except LLMTransportError as exc:
                last_error = exc
                if attempt < total_attempts and self.retry_delay_seconds > 0:
                    self._sleep(self.retry_delay_seconds * attempt)

        raise LLMExtractionError(
            f"LLM extraction failed after {total_attempts} attempt(s). Last error: {last_error}"
        ) from last_error

    def _check_stop_reason(self, response: LLMResponse) -> None:
        if response.stop_reason == "max_tokens":
            max_tokens = getattr(self.llm_client, "max_tokens", "the configured")
            raise LLMPermanentError(
                f"The LLM's answer was cut off at max_tokens={max_tokens} before the JSON was complete. "
                "Increase LLM_MAX_TOKENS (or the 'Max output tokens' setting) or ask for less output in the prompt."
            )
        if response.stop_reason == "refusal":
            raise LLMPermanentError("The model declined to answer this request (stop_reason=refusal).")
