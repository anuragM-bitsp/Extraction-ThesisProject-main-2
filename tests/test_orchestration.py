from __future__ import annotations

import os
import tempfile
import uuid

import pymupdf
import pytest
from celery.exceptions import Retry
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from extractors.rule_extractor import RuleExtractor
from orchestration.extractor_registry import ExtractorUnavailableError, get_extractor
from orchestration.service import PipelineService, VersionNotFoundError
from storage.models import Base
from storage.object_store import LocalObjectStore
from storage.repository import PaperRepository


def make_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    return doc.tobytes()


EXPERIMENTAL_PDF = make_pdf("Silver nitrate was dissolved in ethanol and heated at 80 C for 2 h.")


# ---- PipelineService (no HTTP, no Celery) ----------------------------------------------


@pytest.fixture()
def repo():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with tempfile.TemporaryDirectory() as tmp:
        with Session(engine) as session:
            yield PaperRepository(session, LocalObjectStore(tmp))


def test_ingest_document_persists_blocks_and_canonical_json(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, EXPERIMENTAL_PDF)

    canonical_doc = PipelineService(repo).ingest_document(version.version_id)

    assert any("Silver nitrate" in b.text for b in canonical_doc.blocks)
    assert len(repo.get_blocks(version.version_id)) == len(canonical_doc.blocks)


def test_load_canonical_document_returns_stored_step3_json(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, EXPERIMENTAL_PDF)
    ingested = PipelineService(repo).ingest_document(version.version_id)

    loaded = PipelineService(repo).load_canonical_document(version.version_id)
    assert loaded.paper_id == ingested.paper_id
    assert loaded.was_ocred == ingested.was_ocred
    assert loaded.used_grobid == ingested.used_grobid
    assert [b.text for b in loaded.blocks] == [b.text for b in ingested.blocks]


def test_run_extraction_is_idempotent(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, EXPERIMENTAL_PDF)
    canonical_doc = PipelineService(repo).ingest_document(version.version_id)

    service = PipelineService(repo)
    first = service.run_extraction(version.version_id, RuleExtractor(), canonical_doc)
    second = service.run_extraction(version.version_id, RuleExtractor(), canonical_doc)

    assert first == second
    assert len(repo.get_extraction_results(version.version_id)) == 1


def test_run_extraction_reconstructs_canonical_document_from_stored_blocks(repo):
    """Re-extraction (a second extractor on an already-ingested paper)
    shouldn't require re-parsing the PDF."""
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, EXPERIMENTAL_PDF)
    service = PipelineService(repo)
    service.ingest_document(version.version_id)  # persists blocks, discards the CanonicalDocument

    result = service.run_extraction(version.version_id, RuleExtractor())  # no canonical_doc passed
    assert result.prediction.temperature.value == 80.0


def test_run_extraction_raises_for_unknown_version(repo):
    with pytest.raises(VersionNotFoundError):
        PipelineService(repo).run_extraction(uuid.uuid4(), RuleExtractor())


# ---- extractor_registry -----------------------------------------------------------------


def test_get_extractor_rule_and_ner_always_available():
    from extractors.ner_extractor import NerExtractor

    assert isinstance(get_extractor("rule"), RuleExtractor)
    assert isinstance(get_extractor("ner"), NerExtractor)


def test_get_extractor_llm_fails_clearly_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ExtractorUnavailableError):
        get_extractor("hybrid")


def test_extractor_status_requires_api_key_for_rag_and_hybrid(monkeypatch):
    from orchestration.extractor_registry import extractor_status

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    by_id = {row["id"]: row for row in extractor_status()}
    assert by_id["rule"]["available"] is True
    assert by_id["ner"]["available"] is True
    assert by_id["llm"]["available"] is False
    assert by_id["hybrid"]["available"] is False
    assert "API key" in by_id["llm"]["reason"]

    by_id = {row["id"]: row for row in extractor_status(api_key="session-key")}
    assert by_id["llm"]["available"] is True
    assert by_id["hybrid"]["available"] is True
    assert by_id["llm"]["reason"] is None


# ---- Celery tasks (eager mode) -----------------------------------------------------------


@pytest.fixture()
def env_db(monkeypatch):
    """Points orchestration/db.py's env-var-driven engine/object-store at
    per-test temp locations, so Celery tasks (which open their own session)
    and the test's own assertions see the same data."""
    db_path = tempfile.mktemp(suffix=".db")
    store_path = tempfile.mkdtemp()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("OBJECT_STORE_ROOT", store_path)
    yield
    if os.path.exists(db_path):
        os.remove(db_path)


def _repo(env_db):
    from orchestration.db import get_object_store, get_session

    return PaperRepository(get_session(), get_object_store())


def test_process_document_task_succeeds(env_db):
    from orchestration.tasks import process_document_task

    repo = _repo(env_db)
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, EXPERIMENTAL_PDF)
    job = repo.create_job(version.version_id, job_type="process")
    repo.session.commit()

    process_document_task.delay(str(job.job_id), str(version.version_id))

    fresh_repo = _repo(env_db)
    assert fresh_repo.get_job(job.job_id).status == "SUCCESS"
    assert len(fresh_repo.get_blocks(version.version_id)) > 0


def test_extract_task_succeeds_with_rule_extractor(env_db):
    from orchestration.tasks import extract_task, process_document_task

    repo = _repo(env_db)
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, EXPERIMENTAL_PDF)
    process_job = repo.create_job(version.version_id, job_type="process")
    repo.session.commit()
    process_document_task.delay(str(process_job.job_id), str(version.version_id))

    extract_job = repo.create_job(version.version_id, job_type="extract", extractor="RULE")
    repo.session.commit()
    extract_task.delay(str(extract_job.job_id), str(version.version_id), "rule")

    fresh_repo = _repo(env_db)
    assert fresh_repo.get_job(extract_job.job_id).status == "SUCCESS"
    results = fresh_repo.get_extraction_results(version.version_id)
    assert results[0].payload["prediction"]["temperature"]["value"] == 80.0


def _run_with_simulated_retries(task, args, max_attempts=5):
    """Eager-mode Celery has no scheduler to catch Retry and reschedule the
    task (see tasks.py's docstring) — this plays that missing scheduler's
    part for tests, re-invoking .delay() each time Retry is raised."""
    for _ in range(max_attempts):
        try:
            task.delay(*args)
            return
        except Retry:
            continue
    raise AssertionError(f"task did not terminate within {max_attempts} simulated retries")


def test_process_document_task_retries_then_dead_letters_on_permanent_failure(env_db):
    """An unparseable 'PDF' is a permanent failure — retrying it 3 times
    and landing in DEAD_LETTER (not looping forever) is LLD section 26's
    retry/DLQ policy, exercised for real against actual Celery retry
    semantics rather than asserted in prose."""
    from orchestration.tasks import process_document_task

    repo = _repo(env_db)
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"not a real pdf at all")
    job = repo.create_job(version.version_id, job_type="process")
    repo.session.commit()

    _run_with_simulated_retries(process_document_task, (str(job.job_id), str(version.version_id)))

    fresh_repo = _repo(env_db)
    final_job = fresh_repo.get_job(job.job_id)
    assert final_job.status == "DEAD_LETTER"
    assert final_job.retry_count == 3


# ---- FastAPI app (TestClient, no live server) ---------------------------------------------


@pytest.fixture()
def client(env_db):
    from orchestration.api import app

    return TestClient(app)


def test_create_paper_uploads_pdf(client):
    response = client.post(
        "/papers", files={"file": ("paper.pdf", EXPERIMENTAL_PDF, "application/pdf")}, data={"title": "Test Paper"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == 1


def test_create_paper_rejects_non_pdf(client):
    response = client.post("/papers", files={"file": ("paper.txt", b"not a pdf", "text/plain")})
    assert response.status_code == 400


def test_create_paper_rejects_oversized_upload(client, monkeypatch):
    import orchestration.api as api_module

    monkeypatch.setattr(api_module, "MAX_UPLOAD_BYTES", 10)
    response = client.post("/papers", files={"file": ("paper.pdf", EXPERIMENTAL_PDF, "application/pdf")})
    assert response.status_code == 413


def test_full_flow_process_extract_status_results(client):
    create_response = client.post("/papers", files={"file": ("paper.pdf", EXPERIMENTAL_PDF, "application/pdf")})
    paper_id = create_response.json()["paper_id"]

    process_response = client.post(f"/papers/{paper_id}/process")
    assert process_response.status_code == 200
    assert process_response.json()["status"] == "QUEUED"  # the response reflects job state AT ENQUEUE TIME —
    # true regardless of whether eager mode already finished it by the time this returns; a client
    # polls /status (below) for the actual outcome, exactly as it would have to against a real broker.

    extract_response = client.post(f"/papers/{paper_id}/extract", json={"extractor": "rule"})
    assert extract_response.status_code == 200

    status_response = client.get(f"/papers/{paper_id}/status")
    jobs = status_response.json()
    assert len(jobs) == 2  # one process job, one extract job
    assert all(job["status"] == "SUCCESS" for job in jobs)  # eager mode: both actually completed by now

    results_response = client.get(f"/papers/{paper_id}/results")
    results = results_response.json()
    assert results[0]["payload"]["prediction"]["temperature"]["value"] == 80.0


def test_submit_annotation_via_api(client):
    create_response = client.post("/papers", files={"file": ("paper.pdf", EXPERIMENTAL_PDF, "application/pdf")})
    paper_id = create_response.json()["paper_id"]

    response = client.post(
        f"/papers/{paper_id}/annotations",
        json={"annotator_id": "annotator_a", "payload": {"paper_id": paper_id, "solvent": "ethanol"}},
    )
    assert response.status_code == 200
    assert response.json()["annotator_id"] == "annotator_a"


def test_status_and_results_404_for_unknown_paper(client):
    fake_id = str(uuid.uuid4())
    assert client.get(f"/papers/{fake_id}/status").status_code == 404
    assert client.get(f"/papers/{fake_id}/results").status_code == 404


def test_invalid_paper_id_format_is_a_400_not_a_500(client):
    assert client.get("/papers/not-a-uuid/status").status_code == 400


def test_api_llm_extraction_with_custom_prompt(client, monkeypatch):
    from extractors.llm.llm_client import FakeLLMClient

    seen = []

    def factory(model=None, api_key=None, max_tokens=None):
        seen.append(model)
        return FakeLLMClient([{"temperature": "80 °C", "solvent": "ethanol"}], name=model)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    monkeypatch.setattr("orchestration.extractor_registry.AnthropicLLMClient", factory)

    paper_id = client.post("/papers", files={"file": ("paper.pdf", EXPERIMENTAL_PDF, "application/pdf")}).json()["paper_id"]
    client.post(f"/papers/{paper_id}/process")
    response = client.post(
        f"/papers/{paper_id}/extract",
        json={"extractor": "llm", "prompt": "Return temperature and solvent as JSON.", "model": "claude-custom"},
    )
    assert response.status_code == 200
    jobs = client.get(f"/papers/{paper_id}/status").json()
    assert all(job["status"] == "SUCCESS" for job in jobs), jobs
    assert seen == ["claude-custom"]

    payload = client.get(f"/papers/{paper_id}/results").json()[0]["payload"]
    assert payload["extractor"] == "LLM"
    assert payload["prediction"]["temperature"] == {"value": 80.0, "unit": "C"}
    assert payload["llm_run"]["prompt"] == "Return temperature and solvent as JSON."


def test_api_rejects_unknown_extractor(client):
    paper_id = client.post("/papers", files={"file": ("paper.pdf", EXPERIMENTAL_PDF, "application/pdf")}).json()["paper_id"]
    assert client.post(f"/papers/{paper_id}/extract", json={"extractor": "string"}).status_code == 400
