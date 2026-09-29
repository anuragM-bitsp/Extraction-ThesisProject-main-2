"""
Canonical document representation.

LLD doc 2, section 5: "A plain PDF-to-text conversion can destroy the
relationships between [two-column layouts, equations, tables, headers,
footnotes]... Therefore I want a canonical document representation that
preserves text together with page number, section, table, figure and
positional metadata."

`CanonicalDocument` is that representation. It's the single output of
Step 3 (this step) and the single input to every extractor built in Steps
5-8 — rules, NER, and Prompt+LLM all read the same `CanonicalDocument`, never
a raw PDF. It's also, block-for-block, what gets persisted via
`PaperRepository.add_blocks` (Step 2): `CanonicalBlock.model_dump()` matches
the kwargs `add_blocks` expects.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    FIGURE_CAPTION = "figure_caption"
    REFERENCE = "reference"


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class CanonicalBlock(BaseModel):
    block_type: BlockType
    page_number: Optional[int] = Field(default=None, ge=1)
    section: Optional[str] = None
    text: str = ""
    bbox: Optional[BBox] = None


class Author(BaseModel):
    full_name: str


class CanonicalDocument(BaseModel):
    """
    Full parsed representation of one paper (one document_version).

    `blocks` comes primarily from PyMuPDF (always available). `title`,
    `authors`, `abstract`, and `references` come from GROBID when a GROBID
    client was supplied to the pipeline; they're None otherwise, since
    PyMuPDF alone can't reliably tell a title from a running header.
    """

    paper_id: str
    version: int = Field(ge=1)

    title: Optional[str] = None
    authors: list[Author] = Field(default_factory=list)
    abstract: Optional[str] = None
    references: list[str] = Field(default_factory=list)

    blocks: list[CanonicalBlock] = Field(default_factory=list)

    was_ocred: bool = Field(
        default=False, description="True if the native text layer was missing/inadequate and OCRmyPDF ran"
    )
    used_grobid: bool = False

    def full_text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text)
