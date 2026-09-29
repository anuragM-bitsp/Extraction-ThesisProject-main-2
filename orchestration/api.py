"""
FastAPI layer (LLD section 25): the HTTP surface over everything built in
Steps 1-13.

    POST /papers
    POST /papers/{id}/process
    POST /papers/{id}/extract
    GET  /papers/{id}/status
    GET  /papers/{id}/results
    POST /papers/{id}/annotations

Anything that touches a PDF or an LLM (parsing, extraction) enqueues a
Celery task and responds immediately with a job id — per LLD section 25's
own point: "Not `{'result': '...'}` because extraction is asynchronous."

Security posture (LLD section 29): uploaded PDFs are untrusted input.
Content-type and magic-byte checks plus a size limit are enforced here,
at the boundary, before anything else touches the bytes. The deeper
untrusted-input concern LLD section 29 raises — a paper's text containing
something that looks like an instruction to an LLM — is handled
architecturally in the Prompt + LLM extractor (`extractors/llm/prompting.py`
puts paper content only inside <paper> in the user message and tells the
model to treat it as data, never as instructions) rather than re-litigated here.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from orchestration.db import get_object_store, get_session
from orchestration.extractor_registry import canonical_extractor_id
from orchestration.tasks import extract_task, process_document_task
from storage.repository import PaperRepository

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB

app = FastAPI(title="Scientific Paper Extraction API")


def get_repo():
    session = get_session()
    try:
        repo = PaperRepository(session, get_object_store())
        yield repo
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _require_version(repo: PaperRepository, paper_id: str):
    try:
        paper_uuid = uuid.UUID(paper_id)
    except ValueError:
        raise HTTPException(400, f"{paper_id!r} is not a valid paper id")
    version = repo.get_latest_version(paper_uuid)
    if version is None:
        raise HTTPException(404, f"no paper/version found for {paper_id}")
    return version


# ---- schemas ------------------------------------------------------------------------


class PaperCreateResponse(BaseModel):
    paper_id: str
    version_id: str
    version: int


class JobResponse(BaseModel):
    job_id: str
    status: str


class ExtractRequest(BaseModel):
    extractor: str  # "rule" | "ner" | "llm" | "hybrid"   ("rag_llm" = legacy alias of "llm")
    # Optional overrides for "llm"/"hybrid". Omitted -> LLM_MODEL /
    # EXTRACTION_PROMPT_FILE / prompts/synthesis_extraction.txt.
    prompt: Optional[str] = None
    model: Optional[str] = None


class AnnotationRequest(BaseModel):
    annotator_id: str
    payload: dict


# ---- routes -------------------------------------------------------------------------


@app.post("/papers", response_model=PaperCreateResponse)
async def create_paper(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    doi: Optional[str] = Form(None),
    repo: PaperRepository = Depends(get_repo),
):
    pdf_bytes = await file.read()
    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES} byte limit")
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(400, "uploaded file does not look like a PDF (missing %PDF header)")

    paper = repo.create_paper(title=title, doi=doi)
    version = repo.add_document_version(paper.paper_id, pdf_bytes)
    return PaperCreateResponse(paper_id=str(paper.paper_id), version_id=str(version.version_id), version=version.version)


@app.post("/papers/{paper_id}/process", response_model=JobResponse)
def process_paper(paper_id: str, repo: PaperRepository = Depends(get_repo)):
    version = _require_version(repo, paper_id)
    job = repo.create_job(version.version_id, job_type="process")
    # Commit BEFORE enqueueing: with CELERY_TASK_ALWAYS_EAGER (this
    # sandbox's default — no real broker), .delay() runs the task
    # synchronously, in-process, right here — not after this request
    # finishes the way a real broker's message delivery would guarantee.
    # The task opens its OWN session (orchestration/tasks.py, deliberately,
    # to match what a separate worker process would do) and won't see this
    # job row unless it's committed first. Skipping this commit is exactly
    # the kind of bug that only appears in eager/offline mode and passes
    # silently against a real broker where the delay is enough to hide it.
    repo.session.commit()
    process_document_task.delay(str(job.job_id), str(version.version_id))
    return JobResponse(job_id=str(job.job_id), status="QUEUED")


@app.post("/papers/{paper_id}/extract", response_model=JobResponse)
def extract_paper(paper_id: str, body: ExtractRequest, repo: PaperRepository = Depends(get_repo)):
    version = _require_version(repo, paper_id)
    extractor_id = canonical_extractor_id(body.extractor)
    if extractor_id not in ("rule", "ner", "llm", "hybrid"):
        raise HTTPException(400, f"unknown extractor {body.extractor!r} (expected rule, ner, llm or hybrid)")
    job = repo.create_job(version.version_id, job_type="extract", extractor=extractor_id.upper())
    repo.session.commit()  # see process_paper's comment — same eager-mode ordering requirement
    extract_task.delay(str(job.job_id), str(version.version_id), extractor_id, body.prompt, body.model)
    return JobResponse(job_id=str(job.job_id), status="QUEUED")


@app.get("/papers/{paper_id}/status")
def get_status(paper_id: str, repo: PaperRepository = Depends(get_repo)):
    version = _require_version(repo, paper_id)
    return [job.model_dump(mode="json") for job in repo.get_jobs(version.version_id)]


@app.get("/papers/{paper_id}/results")
def get_results(paper_id: str, repo: PaperRepository = Depends(get_repo)):
    version = _require_version(repo, paper_id)
    return [r.model_dump(mode="json") for r in repo.get_extraction_results(version.version_id)]


@app.post("/papers/{paper_id}/annotations")
def submit_annotation(paper_id: str, body: AnnotationRequest, repo: PaperRepository = Depends(get_repo)):
    version = _require_version(repo, paper_id)
    record = repo.submit_annotation(version.version_id, body.annotator_id, body.payload)
    return record.model_dump(mode="json")
