"""
PyMuPDF parsing layer.

LLD section 8: "I would use GROBID for scientific structure and PyMuPDF as
a lightweight extraction/fallback layer." This module IS that fallback
layer — it never fails to produce *something* for a native-text PDF, has
no external service dependency, and gives us the per-page character counts
that `needs_ocr()` (ocr.py) uses to decide whether OCR is required at all.

Deliberately does not attempt section detection — PyMuPDF sees text blocks
and font sizes, not semantic structure ("Experimental" vs "Introduction").
That's GROBID's job (grobid_client.py / grobid_tei.py). A block from this
module has `section=None`; the ingestion pipeline (pipeline.py) fills it in
when GROBID output is available.
"""

from __future__ import annotations

import pymupdf

from ingestion.canonical import BBox, BlockType, CanonicalBlock


def extract_blocks(pdf_bytes: bytes) -> list[CanonicalBlock]:
    """Extract one CanonicalBlock per text block per page, in reading order.
    Tables/figures are NOT distinguished here — everything native PyMuPDF
    sees as a text block comes back as BlockType.PARAGRAPH. Table extraction
    proper (Camelot/pdfplumber, LLD section 10) is a later step; wiring it
    in only changes which blocks get block_type=TABLE, not this function's
    contract."""
    blocks: list[CanonicalBlock] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc, start=1):
            for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
                text = text.strip()
                if not text:
                    continue
                blocks.append(
                    CanonicalBlock(
                        block_type=BlockType.PARAGRAPH,
                        page_number=page_index,
                        section=None,
                        text=text,
                        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                    )
                )
    return blocks


def page_count(pdf_bytes: bytes) -> int:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count


def total_extractable_chars(pdf_bytes: bytes) -> int:
    """Total characters PyMuPDF can pull from the native text layer. Used by
    ocr.needs_ocr() to decide whether a document is likely scanned (no text
    layer) rather than actually having little text."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return sum(len(page.get_text("text")) for page in doc)
