"""
ORM models — the PostgreSQL schema from LLD sections 4-6, expressed with
SQLAlchemy 2.0's typed declarative style.

Provenance chain this schema exists to support (LLD section 5):

    paper -> version -> page -> block -> character span

Every extracted fact (see schemas.provenance.EvidenceSpan) points back into
this chain, which is why `document_blocks.text` + block/page identity has to
be stable and queryable — it's the thing evidence spans reference.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from storage.db_types import EmbeddingType, GUID, JSONVariant


class Base(DeclarativeBase):
    pass


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PaperORM(Base):
    __tablename__ = "papers"

    paper_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=_uuid)
    title: Mapped[str | None] = mapped_column(default=None)
    doi: Mapped[str | None] = mapped_column(default=None)
    publication_year: Mapped[int | None] = mapped_column(default=None)
    source_uri: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    versions: Mapped[list["DocumentVersionORM"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", order_by="DocumentVersionORM.version"
    )


class DocumentVersionORM(Base):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("paper_id", "version", name="uq_paper_version"),)

    version_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=_uuid)
    paper_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("papers.paper_id"))
    version: Mapped[int]
    content_hash: Mapped[str] = mapped_column(index=True)
    status: Mapped[str] = mapped_column(default="PENDING")
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    paper: Mapped["PaperORM"] = relationship(back_populates="versions")
    blocks: Mapped[list["DocumentBlockORM"]] = relationship(
        back_populates="version", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["ChunkORM"]] = relationship(
        back_populates="version", cascade="all, delete-orphan"
    )


class DocumentBlockORM(Base):
    __tablename__ = "document_blocks"
    __table_args__ = (Index("ix_blocks_version_page", "version_id", "page_number"),)

    block_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=_uuid)
    version_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("document_versions.version_id"))

    block_type: Mapped[str]  # "paragraph" | "table" | "figure" | "heading" | ...
    page_number: Mapped[int | None] = mapped_column(default=None)
    section: Mapped[str | None] = mapped_column(default=None)
    text: Mapped[str] = mapped_column(default="")
    bbox: Mapped[dict | None] = mapped_column(JSONVariant(), default=None)

    version: Mapped["DocumentVersionORM"] = relationship(back_populates="blocks")


class ChunkORM(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=_uuid)
    version_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("document_versions.version_id"))

    text: Mapped[str]
    section: Mapped[str | None] = mapped_column(default=None)
    page_start: Mapped[int | None] = mapped_column(default=None)
    page_end: Mapped[int | None] = mapped_column(default=None)
    token_count: Mapped[int | None] = mapped_column(default=None)

    embedding_model: Mapped[str | None] = mapped_column(default=None)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingType(), default=None)

    version: Mapped["DocumentVersionORM"] = relationship(back_populates="chunks")


class AnnotationORM(Base):
    """
    One annotator's independent, from-scratch labeling of one document
    version (LLD section 20 — "Backend: Annotator -> Annotation API ->
    PostgreSQL"). `payload` stores a full `SynthesisExtraction.model_dump()`
    — annotators produce the exact same schema every extractor does, which
    is what makes gold data directly comparable to any extractor's output
    in Step 11.

    UNIQUE(version_id, annotator_id): one annotation per annotator per
    version — resubmitting overwrites rather than duplicating (mirrors the
    idempotency reasoning behind document_versions' content_hash in Step 2).
    """

    __tablename__ = "annotations"
    __table_args__ = (UniqueConstraint("version_id", "annotator_id", name="uq_version_annotator"),)

    annotation_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=_uuid)
    version_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("document_versions.version_id"))
    annotator_id: Mapped[str]
    payload: Mapped[dict] = mapped_column(JSONVariant())
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)


class GoldAnnotationORM(Base):
    """
    The adjudicated, final ground truth for one document version — at most
    one per version (LLD section 19: gold is independent of, and the
    reference standard against, every extractor's predictions).
    `source_annotation_ids` keeps the audit trail back to which raw
    annotations were adjudicated into this gold record.
    """

    __tablename__ = "gold_annotations"
    __table_args__ = (UniqueConstraint("version_id", name="uq_gold_version"),)

    gold_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=_uuid)
    version_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("document_versions.version_id"))
    payload: Mapped[dict] = mapped_column(JSONVariant())
    adjudicated_by: Mapped[str] = mapped_column(default="unspecified")
    source_annotation_ids: Mapped[list] = mapped_column(JSONVariant(), default=list)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)


class ExtractionResultORM(Base):
    """
    A persisted `ExtractionResult` (any extractor — RULE/NER/RAG_LLM/
    HYBRID). `payload` stores `ExtractionResult.model_dump()` wholesale,
    same pattern as annotations/gold, so nothing here needs to know about
    SynthesisExtraction's internal shape.

    UNIQUE(version_id, extractor, extractor_version, schema_version) is
    LLD section 27's idempotency key verbatim: "unique(paper_id,
    document_version, extractor, extractor_version, schema_version)... if
    the worker receives the same task twice, we return the existing result
    instead of executing duplicate extraction." `PaperRepository.
    store_extraction_result` enforces exactly that.
    """

    __tablename__ = "extraction_results"
    __table_args__ = (
        UniqueConstraint(
            "version_id", "extractor", "extractor_version", "schema_version",
            name="uq_extraction_identity",
        ),
    )

    extraction_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=_uuid)
    version_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("document_versions.version_id"))
    extractor: Mapped[str]
    extractor_version: Mapped[str]
    schema_version: Mapped[str]
    payload: Mapped[dict] = mapped_column(JSONVariant())
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)


class JobORM(Base):
    """
    Async job status (LLD section 25's `GET /papers/{id}/status` and
    section 26's retry/DLQ diagram). Celery owns the actual retry
    mechanics (`orchestration/tasks.py`'s `self.retry(...)`); this row is
    the durable, queryable record of a job's outcome and final retry
    count — what an API client polls, independent of whether Celery,
    Redis, or the worker process are still around to ask directly.
    """

    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_version_status", "version_id", "status"),)

    job_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=_uuid)
    version_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("document_versions.version_id"))
    job_type: Mapped[str]  # "process" | "extract"
    extractor: Mapped[str | None] = mapped_column(default=None)  # set for job_type="extract"
    status: Mapped[str] = mapped_column(default="QUEUED")  # QUEUED|RUNNING|SUCCESS|DEAD_LETTER
    retry_count: Mapped[int] = mapped_column(default=0)
    error_message: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow)
