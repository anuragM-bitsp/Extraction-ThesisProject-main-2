"""
Ingestion pipeline: PDF bytes in, CanonicalDocument out.

    PDF
     |
     +-- PyMuPDF (always runs: text + bbox per block)
     |
     +-- needs_ocr()? --yes--> OCRmyPDF --> re-run PyMuPDF on the OCR'd PDF
     |
     +-- GrobidClient given? --yes--> title/authors/abstract/references
     |                                + best-effort section tagging
     v
  CanonicalDocument

GROBID is treated as optional and best-effort by design: it's a separate
service that can be down, slow, or simply not deployed yet, and per LLD
section 8 it's a structure/metadata enhancement layer on top of PyMuPDF, not
a replacement for it. If GrobidClient.process_fulltext raises for any
reason, ingestion still succeeds — just without title/authors/abstract and
without section labels on blocks.
"""

from __future__ import annotations

import logging

from ingestion.canonical import Author, CanonicalDocument
from ingestion.grobid_client import GrobidClient
from ingestion.grobid_tei import GrobidMetadata, GrobidSection, parse_tei
from ingestion.ocr import needs_ocr, ocr_is_available, run_ocr
from ingestion.pymupdf_parser import extract_blocks, page_count, total_extractable_chars

logger = logging.getLogger(__name__)


def _normalize(s: str) -> str:
    return " ".join(s.split())


def _assign_sections(blocks, sections: list[GrobidSection]) -> None:
    """
    Best-effort section tagging: a block is labeled with a GROBID section
    heading if a leading snippet of the block's text appears inside that
    section's text.

    This is a heuristic, not a guarantee — PyMuPDF and GROBID tokenize and
    segment text independently, so their boundaries don't always align
    (e.g. GROBID may merge two PyMuPDF blocks into one <p>, or split one
    PyMuPDF block across two). Good enough to make `section` usable for
    downstream retrieval filtering (Step 4); not good enough to treat as
    ground truth for evaluation (Step 11) without spot-checking.
    """
    candidates = [(s.heading, _normalize(s.text)) for s in sections if s.text]
    for block in blocks:
        snippet = _normalize(block.text)[:60]
        if not snippet:
            continue
        for heading, section_text in candidates:
            if snippet in section_text:
                block.section = heading
                break


class DocumentIngestionPipeline:
    def __init__(self, grobid_client: GrobidClient | None = None):
        self.grobid_client = grobid_client

    def ingest(self, paper_id: str, version: int, pdf_bytes: bytes) -> CanonicalDocument:
        pages = page_count(pdf_bytes)
        chars = total_extractable_chars(pdf_bytes)

        was_ocred = False
        if needs_ocr(pages, chars):
            if ocr_is_available():
                logger.info("Native text layer inadequate (%d chars / %d pages) — running OCR", chars, pages)
                pdf_bytes = run_ocr(pdf_bytes)
                was_ocred = True
            else:
                logger.warning(
                    "Document needs OCR (%d chars / %d pages) but ocrmypdf is not installed; "
                    "proceeding with whatever native text is available",
                    chars,
                    pages,
                )

        blocks = extract_blocks(pdf_bytes)

        metadata: GrobidMetadata | None = None
        used_grobid = False
        if self.grobid_client is not None:
            try:
                tei_xml = self.grobid_client.process_fulltext(pdf_bytes)
                metadata = parse_tei(tei_xml)
                used_grobid = True
                _assign_sections(blocks, metadata.sections)
            except Exception:
                logger.warning("GROBID enrichment failed; continuing with PyMuPDF-only output", exc_info=True)

        return CanonicalDocument(
            paper_id=paper_id,
            version=version,
            title=metadata.title if metadata else None,
            authors=[Author(full_name=a) for a in metadata.authors] if metadata else [],
            abstract=metadata.abstract if metadata else None,
            references=metadata.references if metadata else [],
            blocks=blocks,
            was_ocred=was_ocred,
            used_grobid=used_grobid,
        )
