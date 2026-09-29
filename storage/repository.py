"""
Repository layer.

This is the only module in the project allowed to know about SQLAlchemy
Sessions and ORM objects. Everything else — extractors, the future API
layer, tests — talks to `PaperRepository` and gets/gives Pydantic models
(storage.read_models) and plain bytes. That boundary is what let Step 1's
schemas stay completely ignorant of how (or whether) anything is persisted.

Implements the idempotency behavior from LLD section 4 ("content_hash allows
us to detect whether a newly uploaded document is actually different from an
existing version") and section 27 (reprocessing the same input should not
silently duplicate state).
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from storage.models import (
    AnnotationORM,
    ChunkORM,
    DocumentBlockORM,
    DocumentVersionORM,
    ExtractionResultORM,
    GoldAnnotationORM,
    JobORM,
    PaperORM,
)
from storage.object_store import ObjectStore, canonical_json_key, raw_pdf_key
from storage.read_models import (
    AnnotationRecord,
    ChunkRecord,
    DocumentBlockRecord,
    DocumentVersionRecord,
    ExtractionResultRecord,
    GoldAnnotationRecord,
    JobRecord,
    PaperRecord,
)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PaperRepository:
    def __init__(self, session: Session, object_store: ObjectStore):
        self.session = session
        self.object_store = object_store

    # ---- papers ---------------------------------------------------------

    def create_paper(
        self,
        title: str | None = None,
        doi: str | None = None,
        publication_year: int | None = None,
        source_uri: str | None = None,
    ) -> PaperRecord:
        paper = PaperORM(
            title=title, doi=doi, publication_year=publication_year, source_uri=source_uri
        )
        self.session.add(paper)
        self.session.flush()
        return PaperRecord.model_validate(paper)

    def get_paper(self, paper_id: uuid.UUID) -> Optional[PaperRecord]:
        paper = self.session.get(PaperORM, paper_id)
        return PaperRecord.model_validate(paper) if paper else None

    # ---- document versions ------------------------------------------------

    def add_document_version(self, paper_id: uuid.UUID, pdf_bytes: bytes) -> DocumentVersionRecord:
        """
        Store `pdf_bytes` in object storage and register a new version row —
        UNLESS the content hash matches the paper's latest version, in which
        case the existing version is returned untouched. This is what stops
        re-uploading the same PDF from creating an unbounded number of
        versions and re-triggering the whole processing pipeline.
        """
        content_hash = _hash_bytes(pdf_bytes)
        latest = self._latest_version_orm(paper_id)

        if latest is not None and latest.content_hash == content_hash:
            return DocumentVersionRecord.model_validate(latest)

        next_version = (latest.version + 1) if latest is not None else 1
        version = DocumentVersionORM(
            paper_id=paper_id,
            version=next_version,
            content_hash=content_hash,
            status="INGESTED",
        )
        self.session.add(version)
        self.session.flush()  # need version.version_id before writing the object key

        self.object_store.put_bytes(raw_pdf_key(paper_id, next_version), pdf_bytes)

        return DocumentVersionRecord.model_validate(version)

    def get_latest_version(self, paper_id: uuid.UUID) -> Optional[DocumentVersionRecord]:
        latest = self._latest_version_orm(paper_id)
        return DocumentVersionRecord.model_validate(latest) if latest else None

    def get_version(self, version_id: uuid.UUID) -> Optional[DocumentVersionRecord]:
        version = self.session.get(DocumentVersionORM, version_id)
        return DocumentVersionRecord.model_validate(version) if version else None

    def _latest_version_orm(self, paper_id: uuid.UUID) -> Optional[DocumentVersionORM]:
        stmt = (
            select(DocumentVersionORM)
            .where(DocumentVersionORM.paper_id == paper_id)
            .order_by(DocumentVersionORM.version.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def store_canonical_document(self, paper_id: uuid.UUID, version: int, canonical_json: bytes) -> str:
        """Persist the parsed canonical-document JSON (Step 3 output) to
        object storage and return its key."""
        key = canonical_json_key(paper_id, version)
        self.object_store.put_bytes(key, canonical_json)
        return key

    # ---- blocks & chunks --------------------------------------------------

    def add_blocks(self, version_id: uuid.UUID, blocks: Iterable[dict]) -> list[DocumentBlockRecord]:
        orm_blocks = [DocumentBlockORM(version_id=version_id, **b) for b in blocks]
        self.session.add_all(orm_blocks)
        self.session.flush()
        return [DocumentBlockRecord.model_validate(b) for b in orm_blocks]

    def get_blocks(self, version_id: uuid.UUID) -> list[DocumentBlockRecord]:
        stmt = (
            select(DocumentBlockORM)
            .where(DocumentBlockORM.version_id == version_id)
            .order_by(DocumentBlockORM.page_number)
        )
        rows = self.session.execute(stmt).scalars().all()
        return [DocumentBlockRecord.model_validate(r) for r in rows]

    def add_chunks(self, version_id: uuid.UUID, chunks: Iterable[dict]) -> list[ChunkRecord]:
        orm_chunks = [ChunkORM(version_id=version_id, **c) for c in chunks]
        self.session.add_all(orm_chunks)
        self.session.flush()
        return [ChunkRecord.model_validate(c) for c in orm_chunks]

    def get_chunks(self, version_id: uuid.UUID) -> list[ChunkRecord]:
        stmt = select(ChunkORM).where(ChunkORM.version_id == version_id)
        rows = self.session.execute(stmt).scalars().all()
        return [ChunkRecord.model_validate(r) for r in rows]

    # ---- annotations & gold (Step 10) --------------------------------------------

    def submit_annotation(
        self, version_id: uuid.UUID, annotator_id: str, payload: dict
    ) -> AnnotationRecord:
        """Upsert: resubmitting the same (version_id, annotator_id) pair
        overwrites the previous submission rather than duplicating it —
        mirrors the idempotency reasoning behind document_versions'
        content_hash from Step 2."""
        stmt = select(AnnotationORM).where(
            AnnotationORM.version_id == version_id, AnnotationORM.annotator_id == annotator_id
        )
        existing = self.session.execute(stmt).scalar_one_or_none()
        if existing is not None:
            existing.payload = payload
            self.session.flush()
            return AnnotationRecord.model_validate(existing)

        annotation = AnnotationORM(version_id=version_id, annotator_id=annotator_id, payload=payload)
        self.session.add(annotation)
        self.session.flush()
        return AnnotationRecord.model_validate(annotation)

    def get_annotations(self, version_id: uuid.UUID) -> list[AnnotationRecord]:
        stmt = select(AnnotationORM).where(AnnotationORM.version_id == version_id)
        rows = self.session.execute(stmt).scalars().all()
        return [AnnotationRecord.model_validate(r) for r in rows]

    def submit_gold(
        self,
        version_id: uuid.UUID,
        payload: dict,
        adjudicated_by: str,
        source_annotation_ids: list,
    ) -> GoldAnnotationRecord:
        stmt = select(GoldAnnotationORM).where(GoldAnnotationORM.version_id == version_id)
        existing = self.session.execute(stmt).scalar_one_or_none()
        if existing is not None:
            existing.payload = payload
            existing.adjudicated_by = adjudicated_by
            existing.source_annotation_ids = source_annotation_ids
            self.session.flush()
            return GoldAnnotationRecord.model_validate(existing)

        gold = GoldAnnotationORM(
            version_id=version_id,
            payload=payload,
            adjudicated_by=adjudicated_by,
            source_annotation_ids=source_annotation_ids,
        )
        self.session.add(gold)
        self.session.flush()
        return GoldAnnotationRecord.model_validate(gold)

    def get_gold(self, version_id: uuid.UUID) -> Optional[GoldAnnotationRecord]:
        stmt = select(GoldAnnotationORM).where(GoldAnnotationORM.version_id == version_id)
        gold = self.session.execute(stmt).scalar_one_or_none()
        return GoldAnnotationRecord.model_validate(gold) if gold else None

    # ---- extraction results (Step 13, idempotent per LLD section 27) ------------

    def store_extraction_result(
        self, version_id: uuid.UUID, extractor: str, extractor_version: str, schema_version: str, payload: dict
    ) -> ExtractionResultRecord:
        """If a result already exists for this exact
        (version_id, extractor, extractor_version, schema_version), return
        it unchanged instead of inserting a duplicate — a re-queued or
        retried extraction task must be a no-op, not a second row."""
        stmt = select(ExtractionResultORM).where(
            ExtractionResultORM.version_id == version_id,
            ExtractionResultORM.extractor == extractor,
            ExtractionResultORM.extractor_version == extractor_version,
            ExtractionResultORM.schema_version == schema_version,
        )
        existing = self.session.execute(stmt).scalar_one_or_none()
        if existing is not None:
            return ExtractionResultRecord.model_validate(existing)

        result = ExtractionResultORM(
            version_id=version_id, extractor=extractor, extractor_version=extractor_version,
            schema_version=schema_version, payload=payload,
        )
        self.session.add(result)
        self.session.flush()
        return ExtractionResultRecord.model_validate(result)

    def get_extraction_results(
        self, version_id: uuid.UUID, extractor: str | None = None
    ) -> list[ExtractionResultRecord]:
        stmt = select(ExtractionResultORM).where(ExtractionResultORM.version_id == version_id)
        if extractor is not None:
            stmt = stmt.where(ExtractionResultORM.extractor == extractor)
        rows = self.session.execute(stmt).scalars().all()
        return [ExtractionResultRecord.model_validate(r) for r in rows]

    # ---- jobs (Step 13) -----------------------------------------------------------

    def create_job(self, version_id: uuid.UUID, job_type: str, extractor: str | None = None) -> JobRecord:
        job = JobORM(version_id=version_id, job_type=job_type, extractor=extractor, status="QUEUED")
        self.session.add(job)
        self.session.flush()
        return JobRecord.model_validate(job)

    def mark_job_running(self, job_id: uuid.UUID) -> JobRecord:
        return self._update_job(job_id, status="RUNNING")

    def mark_job_success(self, job_id: uuid.UUID) -> JobRecord:
        return self._update_job(job_id, status="SUCCESS", error_message=None)

    def mark_job_failed(self, job_id: uuid.UUID, error_message: str, max_retries: int = 3) -> JobRecord:
        """LLD section 26's retry/DLQ diagram: increment retry_count; below
        max_retries the job goes back to QUEUED for Celery to retry, at or
        beyond it the job is DEAD_LETTER — a permanent failure that stops
        retrying rather than looping forever."""
        job = self.session.get(JobORM, job_id)
        if job is None:
            raise ValueError(f"no job {job_id}")
        job.retry_count += 1
        job.status = "DEAD_LETTER" if job.retry_count >= max_retries else "QUEUED"
        job.error_message = error_message
        job.updated_at = _utcnow()
        self.session.flush()
        return JobRecord.model_validate(job)

    def get_job(self, job_id: uuid.UUID) -> Optional[JobRecord]:
        job = self.session.get(JobORM, job_id)
        return JobRecord.model_validate(job) if job else None

    def get_jobs(self, version_id: uuid.UUID) -> list[JobRecord]:
        stmt = select(JobORM).where(JobORM.version_id == version_id).order_by(JobORM.created_at)
        rows = self.session.execute(stmt).scalars().all()
        return [JobRecord.model_validate(r) for r in rows]

    def _update_job(self, job_id: uuid.UUID, **fields) -> JobRecord:
        job = self.session.get(JobORM, job_id)
        if job is None:
            raise ValueError(f"no job {job_id}")
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = _utcnow()
        self.session.flush()
        return JobRecord.model_validate(job)
