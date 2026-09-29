"""
flatten_payload / unflatten_payload: convert between a nested
SynthesisExtraction.model_dump() and the same flat field-path convention
CandidateFact.field already uses everywhere else in this project
("precursors[0].concentration.value").

This is what lets two annotators' full, independently-submitted payloads
(storage.models.AnnotationORM.payload — a whole SynthesisExtraction dict
each) be compared and merged field-by-field without writing bespoke
per-field-type merge logic the way an ad-hoc "walk the schema by hand"
approach would need. Flatten both, compare/resolve per key, unflatten the
result, then let SynthesisExtraction.model_validate() do the actual schema
enforcement — Pydantic re-earns its keep here for free.
"""

from __future__ import annotations

import re
from typing import Any

_INDEX_OR_KEY = re.compile(r"[^.\[\]]+|\[\d+\]")

# Not annotatable content — every payload has these, comparing them would
# only ever show trivial 100% agreement (or a mismatch that's a bug
# elsewhere, not an annotation disagreement).
_EXCLUDED_TOP_LEVEL_KEYS = {"paper_id", "schema_version"}


def flatten_payload(obj: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not prefix and key in _EXCLUDED_TOP_LEVEL_KEYS:
                continue
            path = f"{prefix}.{key}" if prefix else key
            flat.update(flatten_payload(value, path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            flat.update(flatten_payload(item, f"{prefix}[{i}]"))
    else:
        flat[prefix] = obj
    return flat


def _tokenize(path: str) -> list[str]:
    return _INDEX_OR_KEY.findall(path)


def _ensure_length(lst: list, index: int) -> None:
    while len(lst) <= index:
        lst.append(None)


def _set(container, tokens: list[str], value: Any) -> None:
    token = tokens[0]
    is_index = token.startswith("[")
    key: Any = int(token[1:-1]) if is_index else token

    if len(tokens) == 1:
        if isinstance(key, int):
            _ensure_length(container, key)
        container[key] = value
        return

    next_is_index = tokens[1].startswith("[")
    default = [] if next_is_index else {}

    if isinstance(key, int):
        _ensure_length(container, key)
        if container[key] is None:
            container[key] = default
    else:
        if key not in container or container[key] is None:
            container[key] = default

    _set(container[key], tokens[1:], value)


def unflatten_payload(flat: dict[str, Any]) -> dict:
    root: dict = {}
    for path, value in flat.items():
        _set(root, _tokenize(path), value)
    return root
