"""
Indexing: CanonicalDocument -> chunked, embedded, persisted chunks.

This is the glue between Step 3 (ingestion), Step 2 (storage), and this
step's chunking/embeddings modules — the function `index_document` is what
a real ingestion job would call right after `DocumentIngestionPipeline.ingest`.
"""

from __future__ import annotations

import uuid

from ingestion.canonical import CanonicalDocument
from retrieval.chunking import DEFAULT_OVERLAP_TOKENS, DEFAULT_TARGET_TOKENS, chunk_blocks
from retrieval.embeddings import EmbeddingModel
from storage.read_models import ChunkRecord
from storage.repository import PaperRepository


def index_document(
    repository: PaperRepository,
    embedder: EmbeddingModel,
    version_id: uuid.UUID,
    canonical_doc: CanonicalDocument,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[ChunkRecord]:
    drafts = chunk_blocks(canonical_doc.blocks, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    if not drafts:
        return []

    vectors = embedder.embed([d.text for d in drafts])

    chunk_rows = [
        dict(
            text=draft.text,
            section=draft.section,
            page_start=draft.page_start,
            page_end=draft.page_end,
            token_count=draft.token_count,
            embedding_model=embedder.name,
            embedding=vector,
        )
        for draft, vector in zip(drafts, vectors)
    ]
    return repository.add_chunks(version_id, chunk_rows)
