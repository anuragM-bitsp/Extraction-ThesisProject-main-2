from __future__ import annotations

import tempfile
from pathlib import Path

import pymupdf
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from extractors.hybrid_extractor import HybridExtractor
from extractors.ner_extractor import NerExtractor
from extractors.llm.llm_client import FakeLLMClient
from extractors.rule_extractor import RuleExtractor
from frontend.session_pipeline import (
    availability,
    extract_record,
    extract_records,
    ingest_records,
    register_upload,
    result_json_bytes,
)
from ingestion.canonical import CanonicalDocument
from orchestration.extractor_registry import get_extractor
from schemas.extraction_schema import ExtractionResult
from storage.models import Base
from storage.object_store import LocalObjectStore
from storage.repository import PaperRepository


def make_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    return doc.tobytes()


EXPERIMENTAL_PDF = make_pdf("Silver nitrate was dissolved in ethanol and heated at 80 C for 2 h.")
SECOND_PDF = make_pdf("The mixture was stirred at 25 C.")


@pytest.fixture()
def repo():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with tempfile.TemporaryDirectory() as tmp:
        with Session(engine) as session:
            yield PaperRepository(session, LocalObjectStore(tmp))


@pytest.fixture(autouse=True)
def _no_grobid(monkeypatch):
    monkeypatch.setattr("frontend.session_pipeline.maybe_grobid_client", lambda: None)


def test_frontend_app_imports():
    import frontend.app as app

    assert callable(app.render_app)


def test_register_and_ingest_calls_existing_pipeline(repo):
    record = register_upload(repo, "paper.pdf", EXPERIMENTAL_PDF)
    assert record["status"] == "Uploaded"
    ingested = ingest_records(repo, [record])[0]
    assert ingested["status"] == "Ready"
    doc = CanonicalDocument.model_validate(ingested["canonical_doc"])
    assert any("Silver nitrate" in b.text for b in doc.blocks)


def test_rule_extraction_calls_real_rule_extractor(repo, monkeypatch):
    calls = []
    original = RuleExtractor.extract

    def spy(self, document):
        calls.append(type(self).__name__)
        return original(self, document)

    monkeypatch.setattr(RuleExtractor, "extract", spy)
    record = ingest_records(repo, [register_upload(repo, "paper.pdf", EXPERIMENTAL_PDF)])[0]
    extracted = extract_record(repo, record, "rule")
    assert calls == ["RuleExtractor"]
    attempt = extracted["extractions"]["rule"]
    assert attempt["status"] == "Complete"
    result = ExtractionResult.model_validate(attempt["result"])
    assert result.prediction.temperature.value == 80.0
    assert result.extractor.value == "RULE"


def test_ner_extraction_calls_real_ner_extractor(repo, monkeypatch):
    calls = []
    original = NerExtractor.extract

    def spy(self, document):
        calls.append(type(self.model).__name__)
        return original(self, document)

    monkeypatch.setattr(NerExtractor, "extract", spy)
    record = ingest_records(repo, [register_upload(repo, "paper.pdf", EXPERIMENTAL_PDF)])[0]
    extracted = extract_record(repo, record, "ner")
    assert calls == ["GazetteerNER"]
    attempt = extracted["extractions"]["ner"]
    assert attempt["status"] == "Complete"
    result = ExtractionResult.model_validate(attempt["result"])
    assert result.extractor.value == "NER"
    assert attempt["extractor_info"]["ner_implementation"] == "GazetteerNER"
    assert attempt["extractor_info"]["mode"] == "Offline"


def test_json_is_serialized_from_extraction_result(repo):
    record = ingest_records(repo, [register_upload(repo, "paper.pdf", EXPERIMENTAL_PDF)])[0]
    extracted = extract_record(repo, record, "rule")
    payload = result_json_bytes(extracted["extractions"]["rule"]["result"])
    round_trip = ExtractionResult.model_validate_json(payload)
    assert round_trip.prediction.temperature.value == 80.0
    assert round_trip.provenance  # real CandidateFacts from RuleExtractor, not a UI-built list


def test_llm_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    by_id = {row["id"]: row for row in availability(api_key=None)}
    assert by_id["rule"]["available"] is True
    assert by_id["ner"]["available"] is True
    assert by_id["llm"]["available"] is False
    assert by_id["hybrid"]["available"] is False
    assert by_id["llm"]["reason"] == "Anthropic API key is not configured."
    assert "rag_llm" not in by_id  # RAG + LLM was replaced by Prompt + LLM


def test_llm_available_when_session_api_key_supplied(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    by_id = {row["id"]: row for row in availability(api_key="sk-session-only", model="claude-test")}
    assert by_id["llm"]["available"] is True
    assert by_id["llm"]["label"] == "Prompt + LLM"
    assert by_id["hybrid"]["available"] is True
    assert by_id["llm"]["llm_model"] == "claude-test"


def _fake_anthropic(monkeypatch, answers, seen=None):
    def factory(model=None, api_key=None, max_tokens=None):
        if seen is not None:
            seen.append({"model": model, "api_key": api_key, "max_tokens": max_tokens})
        return FakeLLMClient(list(answers), name=model or "fake")

    monkeypatch.setattr("orchestration.extractor_registry.AnthropicLLMClient", factory)


def test_llm_extraction_through_frontend_with_custom_prompt(repo, monkeypatch):
    seen = []
    _fake_anthropic(
        monkeypatch,
        [{"precursors": [{"name": "Silver nitrate", "concentration": "10 mM"}], "temperature": {"value": 80, "unit": "°C"}}],
        seen,
    )
    record = ingest_records(repo, [register_upload(repo, "paper.pdf", EXPERIMENTAL_PDF)])[0]
    extracted = extract_record(
        repo, record, "llm", api_key="sk-test",
        llm_options={"model": "claude-x", "prompt": "Give me precursors as JSON", "prompt_name": "custom", "max_tokens": 4096},
    )
    attempt = extracted["extractions"]["llm"]
    assert attempt["status"] == "Complete", attempt["error"]
    assert seen == [{"model": "claude-x", "api_key": "sk-test", "max_tokens": 4096}]
    result = ExtractionResult.model_validate(attempt["result"])
    assert result.extractor.value == "LLM"
    assert result.prediction.temperature.value == 80.0 and result.prediction.temperature.unit == "C"
    assert result.prediction.precursors[0].concentration.unit == "mM"
    assert result.llm_run.prompt == "Give me precursors as JSON"
    assert result.llm_run.model == "claude-x"
    assert attempt["extractor_info"]["prompt_name"] == "custom"
    assert result_json_bytes(attempt["result"])  # still serializes via the real schema


def test_changed_prompt_is_not_served_from_cache(repo, monkeypatch):
    _fake_anthropic(monkeypatch, [{"solvent": "water"}, {"solvent": "ethanol"}])
    record = ingest_records(repo, [register_upload(repo, "paper.pdf", EXPERIMENTAL_PDF)])[0]
    first = extract_record(repo, record, "llm", api_key="k", llm_options={"prompt": "prompt A"})
    _fake_anthropic(monkeypatch, [{"solvent": "ethanol"}])
    second = extract_record(repo, record, "llm", api_key="k", llm_options={"prompt": "prompt B"})
    a = ExtractionResult.model_validate(first["extractions"]["llm"]["result"])
    b = ExtractionResult.model_validate(second["extractions"]["llm"]["result"])
    assert a.prediction.solvent == "water"
    assert b.prediction.solvent == "ethanol"
    assert a.extractor_version != b.extractor_version


def test_llm_failure_is_reported_not_hidden(repo, monkeypatch):
    from extractors.llm.llm_client import LLMPermanentError

    _fake_anthropic(monkeypatch, [LLMPermanentError("Anthropic rejected the API key (401).")])
    record = ingest_records(repo, [register_upload(repo, "paper.pdf", EXPERIMENTAL_PDF)])[0]
    extracted = extract_record(repo, record, "llm", api_key="bad-key")
    attempt = extracted["extractions"]["llm"]
    assert attempt["status"] == "Failed"
    assert "401" in attempt["error"]
    assert "bad-key" not in attempt["error"]


def test_legacy_rag_llm_name_maps_to_prompt_llm(repo, monkeypatch):
    _fake_anthropic(monkeypatch, [{"solvent": "water"}])
    record = ingest_records(repo, [register_upload(repo, "paper.pdf", EXPERIMENTAL_PDF)])[0]
    extracted = extract_record(repo, record, "rag_llm", api_key="k")
    assert extracted["extractions"]["llm"]["status"] == "Complete"


def test_prompt_presets_default_first():
    from frontend.session_pipeline import prompt_presets

    names = list(prompt_presets())
    assert names[0] == "synthesis_extraction"
    assert "protocols_extraction" in names


def test_hybrid_uses_existing_hybrid_extractor(repo, monkeypatch):
    _fake_anthropic(monkeypatch, [{}])
    calls = []
    original = HybridExtractor.extract

    def spy(self, document):
        calls.append((type(self).__name__, type(self.rule_extractor).__name__, type(self.ner_extractor).__name__))
        return original(self, document)

    monkeypatch.setattr(HybridExtractor, "extract", spy)
    record = ingest_records(repo, [register_upload(repo, "paper.pdf", EXPERIMENTAL_PDF)])[0]
    extracted = extract_record(repo, record, "hybrid", api_key="sk-test-not-used")
    assert calls == [("HybridExtractor", "RuleExtractor", "NerExtractor")]
    attempt = extracted["extractions"]["hybrid"]
    assert attempt["status"] == "Complete"
    result = ExtractionResult.model_validate(attempt["result"])
    assert result.extractor.value == "HYBRID"
    assert attempt["extractor_info"]["resolver"] == "ConflictResolver"


def test_multiple_pdfs_processed_independently(repo):
    records = [
        register_upload(repo, "paper1.pdf", EXPERIMENTAL_PDF),
        register_upload(repo, "paper2.pdf", SECOND_PDF),
    ]
    ingested = ingest_records(repo, records)
    assert [r["status"] for r in ingested] == ["Ready", "Ready"]
    extracted = extract_records(repo, ingested, "rule")
    temps = [
        ExtractionResult.model_validate(r["extractions"]["rule"]["result"]).prediction.temperature.value
        for r in extracted
    ]
    assert temps == [80.0, 25.0]


def test_one_failed_document_does_not_erase_success(repo):
    good = register_upload(repo, "good.pdf", EXPERIMENTAL_PDF)
    bad = register_upload(repo, "bad.pdf", b"%PDF-1.4 not a real pdf at all")
    ingested = ingest_records(repo, [good, bad])
    statuses = {r["filename"]: r["status"] for r in ingested}
    assert statuses["good.pdf"] == "Ready"
    assert statuses["bad.pdf"] == "Failed"
    assert ingested[1]["error"]
    assert "Silver nitrate" in CanonicalDocument.model_validate(ingested[0]["canonical_doc"]).full_text()

    extracted = extract_records(repo, ingested, "rule")
    assert extracted[0]["extractions"]["rule"]["status"] == "Complete"
    assert extracted[1]["extractions"]["rule"]["status"] == "Failed"
    assert extracted[0]["extractions"]["rule"]["result"] is not None


def test_frontend_source_has_no_hardcoded_extraction_results():
    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    banned = (
        "Silver nitrate was dissolved",
        '"precursors": ["Silver nitrate"]',
        "temperature\": 80",
    )
    for path in frontend_dir.glob("*.py"):
        text = path.read_text()
        for snippet in banned:
            assert snippet not in text, f"{path.name} contains banned demo snippet {snippet!r}"


def test_get_extractor_still_rejects_llm_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name in ("llm", "rag_llm", "hybrid"):
        with pytest.raises(Exception):
            get_extractor(name)
