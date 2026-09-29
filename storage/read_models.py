"""
Read-side Pydantic models for the storage layer.

These mirror storage.models 1:1 but are what repositories return and what
(eventually) the FastAPI layer in Step 13 serializes — application code
should depend on these, not on SQLAlchemy ORM objects directly, so the DB
layer can change without rippling through the rest of the codebase.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class PaperRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    paper_id: uuid.UUID
    title: Optional[str] = None
    doi: Optional[str] = None
    publication_year: Optional[int] = None
    source_uri: Optional[str] = None
    created_at: datetime


class DocumentVersionRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    version_id: uuid.UUID
    paper_id: uuid.UUID
    version: int
    content_hash: str
    status: str
    created_at: datetime


class DocumentBlockRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    block_id: uuid.UUID
    version_id: uuid.UUID
    block_type: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    text: str
    bbox: Optional[dict] = None


class ChunkRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: uuid.UUID
    version_id: uuid.UUID
    text: str
    section: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    token_count: Optional[int] = None
    embedding_model: Optional[str] = None
    embedding: Optional[list[float]] = None


class AnnotationRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    annotation_id: uuid.UUID
    version_id: uuid.UUID
    annotator_id: str
    payload: dict
    created_at: datetime


class GoldAnnotationRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    gold_id: uuid.UUID
    version_id: uuid.UUID
    payload: dict
    adjudicated_by: str
    source_annotation_ids: list
    created_at: datetime


class ExtractionResultRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    extraction_id: uuid.UUID
    version_id: uuid.UUID
    extractor: str
    extractor_version: str
    schema_version: str
    payload: dict
    created_at: datetime


class JobRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: uuid.UUID
    version_id: uuid.UUID
    job_type: str
    extractor: Optional[str] = None
    status: str
    retry_count: int
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
