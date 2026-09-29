"""
Robust "LLM text -> JSON value" parsing.

Even when told to return only JSON, models sometimes wrap it in a Markdown
code fence or add a sentence before/after it. This module accepts those
harmless variations and raises `LLMOutputInvalid` (retryable) for anything
that genuinely isn't JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any

from extractors.llm.llm_client import LLMOutputInvalid

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)


def parse_json_response(text: str) -> Any:
    if text is None or not text.strip():
        raise LLMOutputInvalid("The model returned an empty response.")

    stripped = text.strip().lstrip("﻿")

    # 1. The whole response is JSON (the expected case).
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 2. JSON inside a ```json ... ``` fence.
    for match in _FENCE_RE.finditer(stripped):
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # 3. The first complete JSON object/array embedded in surrounding prose.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(stripped):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(stripped[i:])
                return value
            except json.JSONDecodeError:
                continue

    preview = stripped[:300].replace("\n", " ")
    raise LLMOutputInvalid(f"The model response is not valid JSON. Beginning of response: {preview!r}")
