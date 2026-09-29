"""Tests for the Prompt + LLM extractor (replaces the old RAG + LLM tests)."""

from __future__ import annotations

import json

import pytest

try:  # anthropic>=1.x ships on `httpx2`; older SDKs used `httpx`
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

from extractors.llm.json_parsing import parse_json_response
from extractors.llm.llm_client import (
    AnthropicLLMClient,
    FakeLLMClient,
    LLMExtractionError,
    LLMOutputInvalid,
    LLMPermanentError,
    LLMResponse,
    LLMTransportError,
)
from extractors.llm.mapping import UNLOCATED_EVIDENCE_TEXT, canonical_unit, map_llm_output, to_quantity
from extractors.llm.prompting import (
    DOCUMENT_PLACEHOLDER,
    SYSTEM_PROMPT,
    build_prompt,
    list_prompt_presets,
    load_default_prompt,
    prompt_fingerprint,
    render_document,
)
from extractors.llm_extractor import PromptLlmExtractor
from ingestion.canonical import BlockType, CanonicalBlock, CanonicalDocument
from schemas.extraction_schema import ExtractionResult


def block(text, page=1, section=None, block_type=BlockType.PARAGRAPH):
    return CanonicalBlock(block_type=block_type, page_number=page, section=section, text=text)


def document():
    return CanonicalDocument(
        paper_id="P001",
        version=1,
        blocks=[
            block("Gold nanorods have attracted attention for plasmonics.", page=1, section="Introduction"),
            block(
                "HAuCl4 (25 mM, 0.2 mL) was added to CTAB solution in water and heated at 35 °C for 30 min.",
                page=2,
                section="Experimental",
            ),
            block("TEM showed rods with a length of 45 nm.", page=3, section="Results"),
            block("[1] Somebody et al. Heated at 99 °C.", page=4, section="References", block_type=BlockType.REFERENCE),
        ],
    )


FULL_ANSWER = {
    "material": {"name": "gold nanorods", "composition": "Au", "size": {"value": 45, "unit": "nm"}, "morphology": "rod"},
    "precursors": [
        {"name": "HAuCl4", "amount": {"value": 0.2, "unit": "mL"}, "concentration": {"value": 25, "unit": "mM"}},
        {"name": "CTAB", "amount": None, "concentration": None},
    ],
    "solvent": "water",
    "temperature": {"value": 35, "unit": "°C"},
    "reaction_time": {"value": 30, "unit": "min"},
    "pH": None,
    "synthesis_method": "seed-mediated growth",
    "steps": [{"order": 1, "description": "Add HAuCl4 to CTAB solution"}, {"order": 2, "description": "Heat at 35 °C"}],
    "characterization": [{"technique": "TEM", "finding": "rods of 45 nm"}],
    "properties": [{"name": "length", "value": {"value": 45, "unit": "nm"}, "qualitative_value": None}],
}


def make_extractor(answers, **kwargs):
    client = FakeLLMClient(answers)
    return PromptLlmExtractor(llm_client=client, retry_delay_seconds=0, **kwargs), client


# ---- JSON parsing ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Here is the result:\n{"a": 1}\nHope this helps.',
        '﻿  {"a": 1}  ',
    ],
)
def test_parse_json_response_accepts_common_wrappings(text):
    assert parse_json_response(text) == {"a": 1}


@pytest.mark.parametrize("text", ["", "   ", "no json here", '{"a": 1'])
def test_parse_json_response_rejects_non_json(text):
    with pytest.raises(LLMOutputInvalid):
        parse_json_response(text)


# ---- unit / quantity coercion ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("°C", "C"), ("° C", "C"), ("ºC", "C"), ("celsius", "C"), ("K", "K"),
        ("minutes", "min"), ("hours", "h"), ("h", "h"),
        ("mM", "mM"), ("μM", "uM"), ("µM", "uM"), ("M", "M"),
        ("nm", "nm"), ("nM", "nM"),  # case matters: length vs concentration
        ("ml", "mL"), ("µL", "uL"), ("wt %", "wt%"), ("furlongs", "furlongs"),
    ],
)
def test_canonical_unit(raw, expected):
    assert canonical_unit(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"value": 80, "unit": "°C"}, (80.0, "C")),
        ({"value": "0.5", "unit": "M"}, (0.5, "M")),
        ("80 °C", (80.0, "C")),
        ("~2 h", (2.0, "h")),
        ("25mM", (25.0, "mM")),
        ([{"value": 10, "unit": "min"}], (10.0, "min")),
        ({"value": "80 °C"}, (80.0, "C")),
    ],
)
def test_to_quantity_accepts_shapes(raw, expected):
    q = to_quantity(raw)
    assert (q.value, q.unit) == expected


@pytest.mark.parametrize(
    "raw",
    [None, "room temperature", "25-30 °C", "25–30 °C", "2 or 3 h", 80, {"value": 80}, {"value": None, "unit": "C"}, True],
)
def test_to_quantity_rejects_ambiguous(raw):
    assert to_quantity(raw) is None


# ---- mapping ------------------------------------------------------------------------------


def test_map_full_answer_to_schema():
    prediction, facts = map_llm_output(FULL_ANSWER, document(), "llm@test")
    assert prediction.material.name == "gold nanorods"
    assert prediction.material.size.unit == "nm"
    assert [p.name for p in prediction.precursors] == ["HAuCl4", "CTAB"]
    assert prediction.precursors[0].concentration.model_dump() == {"value": 25.0, "unit": "mM"}
    assert prediction.precursors[0].amount.model_dump() == {"value": 0.2, "unit": "mL"}
    assert prediction.solvent == "water"
    assert prediction.temperature.model_dump() == {"value": 35.0, "unit": "C"}
    assert prediction.reaction_time.model_dump() == {"value": 30.0, "unit": "min"}
    assert prediction.pH is None
    assert [s.order for s in prediction.steps] == [1, 2]
    assert prediction.characterization[0].technique == "TEM"
    assert prediction.properties[0].value.value == 45.0
    fields = {f.field for f in facts}
    assert {"temperature", "reaction_time", "solvent", "synthesis_method", "material.name",
            "precursors[0].name", "precursors[0].concentration", "precursors[0].amount",
            "precursors[1].name", "steps[0].description", "characterization[0].technique",
            "properties[0].name"} <= fields
    assert all(f.source.value == "LLM" and f.extractor_version == "llm@test" for f in facts)


def test_mapping_locates_evidence_in_document():
    _, facts = map_llm_output(FULL_ANSWER, document(), "llm@test")
    by_field = {f.field: f for f in facts}

    temp = by_field["temperature"].evidence
    assert temp.page == 2 and temp.section == "Experimental"
    assert temp.text == "35 °C"
    assert temp.char_start is not None and temp.char_end > temp.char_start
    assert by_field["temperature"].confidence == 0.8

    name = by_field["precursors[0].name"].evidence
    assert name.text == "HAuCl4" and name.page == 2

    conc = by_field["precursors[0].concentration"].evidence
    assert conc.text == "25 mM"

    method = by_field["synthesis_method"]  # paraphrase — not verbatim in the paper
    assert method.evidence.text == UNLOCATED_EVIDENCE_TEXT
    assert method.evidence.page is None
    assert method.confidence == 0.6


def test_mapping_never_uses_reference_list_as_evidence():
    _, facts = map_llm_output({"temperature": "99 °C"}, document(), "v")
    assert facts[0].evidence.text == UNLOCATED_EVIDENCE_TEXT


def test_mapping_is_lenient_about_shapes():
    answer = {
        "material": "gold nanorods",
        "reagents": ["HAuCl4", {"chemical": "CTAB", "concentration": "0.1 M"}],
        "solvent": ["water", "ethanol"],
        "temperature": "35 °C",
        "time": "30 min",
        "pH": "7.4",
        "method": "seeded growth",
        "procedure": ["mix", {"action": "heat"}, "", None],
        "characterization": ["TEM", "UV-vis"],
        "properties": [{"property": "LSPR", "value": 780, "unit": "nm"}, {"name": "shape", "value": "rod"}],
    }
    p, _ = map_llm_output(answer, document(), "v")
    assert p.material.name == "gold nanorods"
    assert [x.name for x in p.precursors] == ["HAuCl4", "CTAB"]
    assert p.precursors[1].concentration.model_dump() == {"value": 0.1, "unit": "M"}
    assert p.solvent == "water"
    assert p.temperature.unit == "C" and p.reaction_time.unit == "min"
    assert p.pH == 7.4
    assert p.synthesis_method == "seeded growth"
    assert [s.description for s in p.steps] == ["mix", "heat"]
    assert [c.technique for c in p.characterization] == ["TEM", "UV-vis"]
    assert p.properties[0].value.model_dump() == {"value": 780.0, "unit": "nm"}
    assert p.properties[1].qualitative_value == "rod"


@pytest.mark.parametrize("answer", [[], "text", 42, None, {"protocols": [{"name": "seed"}]}, {"solvent": "N/A", "pH": 99}])
def test_mapping_of_non_schema_answers_gives_empty_prediction(answer):
    p, facts = map_llm_output(answer, document(), "v")
    assert p.precursors == [] and p.solvent is None and p.pH is None and p.material is None
    assert facts == []


# ---- prompt building ----------------------------------------------------------------------


def test_render_document_has_page_and_section_markers_and_drops_references():
    text, truncated = render_document(document())
    assert "[Page 2]" in text and "[Section: Experimental]" in text
    assert "HAuCl4 (25 mM" in text
    assert "Somebody et al" not in text
    assert truncated is False


def test_render_document_truncates_long_documents():
    text, truncated = render_document(document(), max_chars=50)
    assert truncated is True
    assert text.endswith("[... document truncated ...]")


def test_build_prompt_puts_paper_first_then_instructions():
    system, user, _ = build_prompt("Extract the solvent as JSON.", document())
    assert system == SYSTEM_PROMPT
    assert "untrusted" in system and "JSON" in system
    assert user.index("<paper>") < user.index("<instructions>")
    assert "Extract the solvent as JSON." in user
    assert "HAuCl4" in user
    assert "HAuCl4" not in system  # paper text never goes into the system prompt


def test_build_prompt_supports_document_placeholder():
    _, user, _ = build_prompt(f"Read this:\n{DOCUMENT_PLACEHOLDER}\nNow return JSON with {{\"a\": 1}}.", document())
    assert "<instructions>" not in user
    assert user.startswith("Read this:\n<paper>")
    assert '{"a": 1}' in user  # literal braces in the prompt are left alone


def test_default_prompt_and_presets_exist():
    default = load_default_prompt()
    assert "precursors" in default and "JSON" in default
    presets = list_prompt_presets()
    assert {"synthesis_extraction", "protocols_extraction"} <= set(presets)


def test_prompt_file_env_override(tmp_path, monkeypatch):
    path = tmp_path / "p.txt"
    path.write_text("Only the solvent, as JSON.")
    monkeypatch.setenv("EXTRACTION_PROMPT_FILE", str(path))
    assert load_default_prompt() == "Only the solvent, as JSON."
    extractor, _ = make_extractor([])
    assert extractor.prompt == "Only the solvent, as JSON."
    monkeypatch.setenv("EXTRACTION_PROMPT_FILE", str(tmp_path / "missing.txt"))
    with pytest.raises(FileNotFoundError):
        load_default_prompt()


# ---- extractor ----------------------------------------------------------------------------


def test_extractor_end_to_end():
    extractor, client = make_extractor([FULL_ANSWER])
    result = extractor.extract(document())

    assert isinstance(result, ExtractionResult)
    assert result.extractor.value == "LLM"
    assert result.prediction.temperature.value == 35.0
    assert result.llm_run.output == FULL_ANSWER  # raw JSON preserved verbatim
    assert result.llm_run.prompt == extractor.prompt
    assert result.llm_run.prompt_sha == prompt_fingerprint(extractor.prompt)
    assert result.llm_run.attempts == 1
    assert len(client.calls) == 1  # ONE call per paper — no per-field retrieval calls
    ExtractionResult.model_validate_json(result.model_dump_json())  # round-trips


def test_custom_prompt_is_sent_and_arbitrary_json_is_kept():
    custom = {"protocols": [{"name": "seed synthesis", "reagents": [{"name": "HAuCl4"}]}]}
    extractor, client = make_extractor([custom], prompt="List every protocol as JSON.", prompt_name="protocols")
    result = extractor.extract(document())
    assert "List every protocol as JSON." in client.calls[0][1]
    assert result.llm_run.output == custom
    assert result.llm_run.prompt_name == "protocols"
    assert result.prediction.precursors == []  # not schema-shaped -> only in llm_run.output


def test_version_changes_with_prompt_and_model():
    a, _ = make_extractor([], prompt="prompt A")
    b, _ = make_extractor([], prompt="prompt B")
    a2, _ = make_extractor([], prompt="  prompt A  ")
    assert a.version != b.version
    assert a.version == a2.version
    assert a.version.startswith("llm@fake-llm+prompt-")


def test_invalid_json_is_retried_with_feedback():
    extractor, client = make_extractor(["Sorry, here you go: not json", {"solvent": "water"}])
    result = extractor.extract(document())
    assert result.prediction.solvent == "water"
    assert result.llm_run.attempts == 2
    assert "could not be parsed as JSON" in client.calls[1][1]
    assert "could not be parsed as JSON" not in client.calls[0][1]


def test_transport_errors_are_retried_with_backoff():
    sleeps = []
    client = FakeLLMClient([LLMTransportError("timeout"), LLMTransportError("529 overloaded"), {"solvent": "water"}])
    extractor = PromptLlmExtractor(llm_client=client, retry_delay_seconds=1.5, sleep=sleeps.append)
    assert extractor.extract(document()).prediction.solvent == "water"
    assert sleeps == [1.5, 3.0]


def test_exhausted_retries_raise_with_last_error():
    extractor, client = make_extractor([LLMTransportError("down")] * 3, max_retries=2)
    with pytest.raises(LLMExtractionError, match="down"):
        extractor.extract(document())
    assert len(client.calls) == 3


def test_permanent_errors_are_not_retried():
    extractor, client = make_extractor([LLMPermanentError("bad key"), {"solvent": "water"}])
    with pytest.raises(LLMPermanentError, match="bad key"):
        extractor.extract(document())
    assert len(client.calls) == 1


def test_truncated_output_is_a_clear_permanent_error():
    truncated = LLMResponse(text='{"steps": ["a", "b"', model="m", stop_reason="max_tokens")
    extractor, client = make_extractor([truncated, {"solvent": "water"}])
    with pytest.raises(LLMPermanentError, match="max_tokens"):
        extractor.extract(document())
    assert len(client.calls) == 1


def test_empty_document_raises_instead_of_calling_llm():
    extractor, client = make_extractor([{"solvent": "water"}])
    with pytest.raises(LLMPermanentError, match="no extractable text"):
        extractor.extract(CanonicalDocument(paper_id="P", version=1, blocks=[]))
    assert client.calls == []


# ---- AnthropicLLMClient against the real SDK (mocked HTTP) ----------------------------------------


def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def _stream_body(text, stop_reason="end_turn"):
    return _sse(
        [
            {"type": "message_start", "message": {
                "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test", "content": [],
                "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1234, "output_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "{\"not\": \"this\"}"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text[: len(text) // 2]}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text[len(text) // 2:]}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": 56}},
            {"type": "message_stop"},
        ]
    )


def _anthropic_client(handler, **kwargs):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return AnthropicLLMClient(model="claude-test", api_key="sk-test", http_client=http, **kwargs)


def test_anthropic_client_request_and_streamed_response():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return httpx.Response(200, text=_stream_body('{"solvent": "water"}'), headers={"content-type": "text/event-stream"})

    client = _anthropic_client(handler, max_tokens=4096)
    response = client.complete("SYS", "USER")

    body = seen["body"]
    assert seen["url"].endswith("/v1/messages")
    assert body["model"] == "claude-test" and body["max_tokens"] == 4096 and body["stream"] is True
    assert body["system"] == "SYS"
    assert body["messages"] == [{"role": "user", "content": "USER"}]
    # Parameters that break current models must NOT be sent:
    for forbidden in ("temperature", "top_p", "top_k", "tools", "tool_choice"):
        assert forbidden not in body
    assert response.text == '{"solvent": "water"}'  # thinking block ignored
    assert response.stop_reason == "end_turn"
    assert response.input_tokens == 1234 and response.output_tokens == 56

    extractor = PromptLlmExtractor(llm_client=_anthropic_client(handler), retry_delay_seconds=0)
    assert extractor.extract(document()).prediction.solvent == "water"


@pytest.mark.parametrize(
    "status,error_type,expected,match",
    [
        (401, "authentication_error", LLMPermanentError, "API key"),
        (404, "not_found_error", LLMPermanentError, "not found"),
        (400, "invalid_request_error", LLMPermanentError, "boom"),
        (429, "rate_limit_error", LLMTransportError, "Temporary"),
        (529, "overloaded_error", LLMTransportError, "Temporary"),
        (500, "api_error", LLMTransportError, "Temporary"),
    ],
)
def test_anthropic_client_classifies_errors(status, error_type, expected, match):
    def handler(request):
        return httpx.Response(status, json={"type": "error", "error": {"type": error_type, "message": "boom"}})

    with pytest.raises(expected, match=match):
        _anthropic_client(handler).complete("s", "u")


def test_anthropic_client_connection_error_is_transient():
    def handler(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(LLMTransportError):
        _anthropic_client(handler).complete("s", "u")
