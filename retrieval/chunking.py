"""
Chunking: CanonicalBlock list -> ChunkDraft list.

Word count is used as the token-count proxy throughout (no tokenizer
dependency) — good enough for sizing chunks; swap for a real tokenizer's
count if a specific LLM's context budget needs to be hit exactly.

Two things this chunker deliberately does, both traceable to the LLD:

  1. Prefers to break at a section boundary over exceeding the token target,
     so a chunk doesn't straddle e.g. "Introduction" and "Experimental" —
     LLD doc 2 section 10 leans on section metadata ("synthesis conditions
     are more likely in Experimental than Introduction") for RAG retrieval,
     which only works if chunks don't span sections.
  2. Carries the tail of one chunk into the start of the next (overlap), so
     a fact split across a chunk boundary (e.g. "...heated at 80 °C" |
     "for 2 h.") is still findable as one contiguous span in at least one
     chunk.
"""

from __future__ import annotations

from pydantic import BaseModel

from ingestion.canonical import CanonicalBlock

DEFAULT_TARGET_TOKENS = 200
DEFAULT_OVERLAP_TOKENS = 40


class ChunkDraft(BaseModel):
    """Pre-embedding chunk. Field names match storage.repository.add_chunks'
    expected kwargs (plus `embedding`/`embedding_model`, added at index
    time) so a draft converts to a DB row with no remapping."""

    text: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    token_count: int


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_blocks(
    blocks: list[CanonicalBlock],
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []

    texts: list[str] = []
    section: str | None = None
    pages: list[int] = []

    def current_token_count() -> int:
        return sum(_word_count(t) for t in texts)

    def flush() -> None:
        if not texts:
            return
        text = "\n".join(texts)
        drafts.append(
            ChunkDraft(
                text=text,
                section=section,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                token_count=_word_count(text),
            )
        )

    for block in blocks:
        block_text = block.text.strip()
        if not block_text:
            continue
        block_tokens = _word_count(block_text)

        crosses_section = (
            bool(texts) and section is not None and block.section is not None and block.section != section
        )
        would_overflow = bool(texts) and (current_token_count() + block_tokens > target_tokens)

        if crosses_section or would_overflow:
            flush()
            if would_overflow and not crosses_section:
                # Overlap only makes sense when we split due to length —
                # carrying trailing words across a section boundary would
                # leak, say, Introduction text into an Experimental-tagged
                # chunk, undermining the whole point of tagging by section.
                prev_words = "\n".join(texts).split() if texts else []
                overlap_words = prev_words[-overlap_tokens:] if prev_words else []
                texts = [" ".join(overlap_words)] if overlap_words else []
            else:
                texts = []
            pages = []
            section = block.section
        elif section is None:
            section = block.section

        texts.append(block_text)
        if block.page_number is not None:
            pages.append(block.page_number)

    flush()
    return [d for d in drafts if d.text.strip()]
