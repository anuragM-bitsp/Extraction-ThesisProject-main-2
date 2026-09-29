"""
UI-facing calls into the existing orchestration layer.

This module does not parse PDFs, run regexes, run NER, or call an LLM.
It registers bytes with `PaperRepository`, ingests via `PipelineService`
(which uses `DocumentIngestionPipeline`), and extracts via `get_extractor`
+ `PipelineService.run_extraction`.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

from ingestion.canonical import CanonicalDocument
from ingestion.grobid_client import HttpGrobidClient
from orchestration.api import MAX_UPLOAD_BYTES
from extractors.llm.llm_client import list_anthropic_models
from extractors.llm.prompting import DEFAULT_PROMPT_NAME, list_prompt_presets, load_default_prompt
from orchestration.extractor_registry import (
    ExtractorUnavailableError,
    canonical_extractor_id,
    describe_extractor,
    extractor_status,
    get_extractor,
    resolve_anthropic_api_key,
)
from orchestration.service import PipelineService
from schemas.extraction_schema import ExtractionResult
from storage.repository import PaperRepository

from frontend.state import DocumentRecord, ExtractionAttempt, empty_document


def redact_secrets(message: str, api_key: Optional[str] = None) -> str:
    text = message
    if api_key:
        text = text.replace(api_key, "[redacted]")
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        text = text.replace(env_key, "[redacted]")
    return text


def maybe_grobid_client() -> HttpGrobidClient | None:
    """Use the existing HTTP GROBID client only when the service answers /api/isalive."""
    url = os.environ.get("GROBID_URL", "http://localhost:8070")
    client = HttpGrobidClient(base_url=url)
    return client if client.is_alive() else None


def register_upload(repo: PaperRepository, filename: str, pdf_bytes: bytes) -> DocumentRecord:
    if not filename.lower().endswith(".pdf"):
        raise ValueError(f"{filename}: only PDF files are supported")
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError(f"{filename}: uploaded file does not look like a PDF (missing %PDF header)")
    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(f"{filename}: file exceeds {MAX_UPLOAD_BYTES} byte limit")

    record = empty_document(filename, len(pdf_bytes))
    paper = repo.create_paper(title=filename, source_uri=filename)
    version = repo.add_document_version(paper.paper_id, pdf_bytes)
    record["paper_id"] = str(paper.paper_id)
    record["version_id"] = str(version.version_id)
    record["status"] = "Uploaded"
    repo.session.commit()
    return record


def ingest_record(repo: PaperRepository, record: DocumentRecord) -> DocumentRecord:
    """Run Step 3 via PipelineService. Failures stay on this record only."""
    updated = dict(record)
    updated["status"] = "Ingesting"
    updated["error"] = None
    version_id = updated.get("version_id")
    if not version_id:
        updated["status"] = "Failed"
        updated["error"] = "Ingestion failed: document was never registered (no version_id)."
        return updated

    try:
        grobid = maybe_grobid_client()
        canonical_doc = PipelineService(repo).ingest_document(uuid.UUID(version_id), grobid_client=grobid)
        repo.session.commit()
        updated["canonical_doc"] = canonical_doc.model_dump(mode="json")
        updated["status"] = "Ready"
        updated["error"] = None
    except Exception as exc:
        repo.session.rollback()
        updated["status"] = "Failed"
        updated["error"] = redact_secrets(f"Ingestion failed: {exc}")
    return updated


def ingest_records(repo: PaperRepository, records: list[DocumentRecord]) -> list[DocumentRecord]:
    return [ingest_record(repo, rec) for rec in records]


def extract_record(
    repo: PaperRepository,
    record: DocumentRecord,
    extractor_name: str,
    api_key: Optional[str] = None,
    llm_options: Optional[dict] = None,
) -> DocumentRecord:
    """Run one existing extractor against one already-ingested document.

    `llm_options` (model, prompt, prompt_name, max_tokens) only matter for
    the "llm" and "hybrid" extractors."""
    extractor_name = canonical_extractor_id(extractor_name)
    updated = dict(record)
    extractions: dict[str, ExtractionAttempt] = dict(updated.get("extractions") or {})
    version_id = updated.get("version_id")

    if updated.get("status") != "Ready" or not version_id:
        extractions[extractor_name] = {
            "status": "Failed",
            "error": "Extraction skipped: document is not Ready (ingestion did not complete).",
            "result": None,
            "extractor_info": None,
        }
        updated["extractions"] = extractions
        return updated

    canonical_doc = None
    if updated.get("canonical_doc"):
        canonical_doc = CanonicalDocument.model_validate(updated["canonical_doc"])

    try:
        extractor = get_extractor(extractor_name, api_key=api_key, **(llm_options or {}))
        info = describe_extractor(extractor)
        result = PipelineService(repo).run_extraction(uuid.UUID(version_id), extractor, canonical_doc)
        repo.session.commit()
        extractions[extractor_name] = {
            "status": "Complete",
            "error": None,
            "result": result.model_dump(mode="json"),
            "extractor_info": info,
        }
    except ExtractorUnavailableError as exc:
        repo.session.rollback()
        extractions[extractor_name] = {
            "status": "Failed",
            "error": redact_secrets(str(exc), api_key),
            "result": None,
            "extractor_info": None,
        }
    except Exception as exc:
        repo.session.rollback()
        extractions[extractor_name] = {
            "status": "Failed",
            "error": redact_secrets(f"Extraction failed: {exc}", api_key),
            "result": None,
            "extractor_info": None,
        }

    updated["extractions"] = extractions
    return updated


def extract_records(
    repo: PaperRepository,
    records: list[DocumentRecord],
    extractor_name: str,
    api_key: Optional[str] = None,
    llm_options: Optional[dict] = None,
) -> list[DocumentRecord]:
    return [
        extract_record(repo, rec, extractor_name, api_key=api_key, llm_options=llm_options) for rec in records
    ]


def result_json_bytes(result_dump: dict) -> bytes:
    """Serialize an already-produced ExtractionResult dump. Validates against the real schema."""
    result = ExtractionResult.model_validate(result_dump)
    return result.model_dump_json(indent=2).encode("utf-8")


def availability(api_key: Optional[str] = None, model: Optional[str] = None) -> list[dict]:
    return extractor_status(api_key=api_key, model=model)


def available_models(api_key: Optional[str] = None) -> tuple[list[dict], Optional[str]]:
    """(models, error). Models this key can use, newest first."""
    key = resolve_anthropic_api_key(api_key)
    if not key:
        return [], "Apply an API key first."
    try:
        return list_anthropic_models(key), None
    except Exception as exc:
        return [], redact_secrets(str(exc), api_key)


def prompt_presets() -> dict[str, str]:
    """Prompt presets from the prompts/ folder, default prompt first."""
    presets = list_prompt_presets()
    if not presets:
        return {"default": load_default_prompt()}
    ordered = {DEFAULT_PROMPT_NAME: presets.pop(DEFAULT_PROMPT_NAME)} if DEFAULT_PROMPT_NAME in presets else {}
    ordered.update(presets)
    return ordered


def document_metadata(canonical_dump: dict | None, filename: str, paper_id: str | None, version_id: str | None) -> dict:
    """Only fields that exist on CanonicalDocument / the session record."""
    meta: dict = {"Filename": filename}
    if paper_id:
        meta["Paper ID"] = paper_id
    if canonical_dump:
        doc = CanonicalDocument.model_validate(canonical_dump)
        meta["Version"] = doc.version
        pages = {b.page_number for b in doc.blocks if b.page_number is not None}
        if pages:
            meta["Number of pages"] = max(pages)
        if doc.title:
            meta["Title"] = doc.title
        if doc.authors:
            meta["Authors"] = ", ".join(a.full_name for a in doc.authors)
        sections = sorted({b.section for b in doc.blocks if b.section})
        if sections:
            meta["Sections"] = ", ".join(sections)
        meta["OCR used"] = doc.was_ocred
        meta["GROBID used"] = doc.used_grobid
    if version_id:
        meta["Version ID"] = version_id
    return meta
