"""
PipelineService: the single place ingestion (Step 3), indexing (Step 4),
and extraction (Steps 5-8) get called from once a paper is in storage.

Both the FastAPI routes and the Celery tasks call THIS, never each other's
internals — matching the "business logic is plain Python, transport is a
thin wrapper" split Step 12's `ExperimentRunner` already established. A
`PipelineService` needs nothing but a `PaperRepository` and an optional
embedder; it's fully testable with an in-memory SQLite session and a
`LocalObjectStore`, the same fixtures every storage test in this project
already uses — no HTTP client, no Celery worker, no running broker.
"""

from __future__ import annotations

import uuid

from extractors.base import Extractor
from ingestion.canonical import BBox, CanonicalBlock, CanonicalDocument
from ingestion.pipeline import DocumentIngestionPipeline
from retrieval.embeddings import EmbeddingModel
from retrieval.indexing import index_document
from schemas.extraction_schema import SCHEMA_VERSION, ExtractionResult
from storage.object_store import canonical_json_key, raw_pdf_key
from storage.repository import PaperRepository


class VersionNotFoundError(RuntimeError):
    pass


class PipelineService:
    def __init__(self, repository: PaperRepository, embedder: EmbeddingModel | None = None):
        self.repository = repository
        self.embedder = embedder  # only required if indexing (chunk+embed) is wanted

    def ingest_document(self, version_id: uuid.UUID, grobid_client=None) -> CanonicalDocument:
        """LLD section 2's "Document Worker": pulls the raw PDF back out of
        object storage, runs it through Step 3's ingestion pipeline,
        persists the resulting blocks, and — if an embedder was supplied —
        chunks and embeds them for retrieval (Step 4)."""
        version = self._require_version(version_id)
        pdf_bytes = self.repository.object_store.get_bytes(raw_pdf_key(version.paper_id, version.version))

        pipeline = DocumentIngestionPipeline(grobid_client=grobid_client)
        canonical_doc = pipeline.ingest(paper_id=str(version.paper_id), version=version.version, pdf_bytes=pdf_bytes)

        self.repository.add_blocks(
            version_id,
            [
                dict(
                    block_type=block.block_type.value,
                    page_number=block.page_number,
                    section=block.section,
                    text=block.text,
                    bbox=block.bbox.model_dump() if block.bbox else None,
                )
                for block in canonical_doc.blocks
            ],
        )
        self.repository.store_canonical_document(
            version.paper_id, version.version, canonical_doc.model_dump_json().encode("utf-8")
        )

        if self.embedder is not None:
            index_document(self.repository, self.embedder, version_id, canonical_doc)

        return canonical_doc

    def run_extraction(
        self, version_id: uuid.UUID, extractor: Extractor, canonical_doc: CanonicalDocument | None = None
    ) -> ExtractionResult:
        """
        Idempotent per LLD section 27's exact wording: "if the worker
        receives the same task twice, we return the existing result
        instead of executing duplicate extraction." Identity is
        (version, extractor, extractor_version, schema_version) — the same
        key `storage.repository.store_extraction_result` enforces at the
        database level.
        """
        version = self._require_version(version_id)

        already_run = [
            r
            for r in self.repository.get_extraction_results(version_id, extractor=extractor.name.value)
            if r.extractor_version == extractor.version and r.schema_version == SCHEMA_VERSION
        ]
        if already_run:
            return ExtractionResult.model_validate(already_run[0].payload)

        if canonical_doc is None:
            canonical_doc = self.load_canonical_document(version_id)

        result = extractor.extract(canonical_doc)
        stored = self.repository.store_extraction_result(
            version_id, extractor.name.value, extractor.version, SCHEMA_VERSION, result.model_dump(mode="json")
        )
        return ExtractionResult.model_validate(stored.payload)

    def _require_version(self, version_id: uuid.UUID):
        version = self.repository.get_version(version_id)
        if version is None:
            raise VersionNotFoundError(f"no document_version {version_id}")
        return version

    def load_canonical_document(self, version_id: uuid.UUID) -> CanonicalDocument:
        """Prefer the full Step 3 JSON (title/authors/OCR/GROBID flags) over
        reconstructing from persisted blocks, which drop that metadata."""
        version = self._require_version(version_id)
        try:
            raw = self.repository.object_store.get_bytes(canonical_json_key(version.paper_id, version.version))
            return CanonicalDocument.model_validate_json(raw)
        except FileNotFoundError:
            return self._canonical_document_from_stored_blocks(version)

    def _canonical_document_from_stored_blocks(self, version) -> CanonicalDocument:
        """Re-extraction (a second extractor run on an already-ingested
        paper) shouldn't require re-parsing the PDF — the persisted blocks
        from Step 3's ingestion are the source of truth from here on."""
        blocks = self.repository.get_blocks(version.version_id)
        canonical_blocks = [
            CanonicalBlock(
                block_type=b.block_type,
                page_number=b.page_number,
                section=b.section,
                text=b.text,
                bbox=BBox(**b.bbox) if b.bbox else None,
            )
            for b in blocks
        ]
        return CanonicalDocument(paper_id=str(version.paper_id), version=version.version, blocks=canonical_blocks)
