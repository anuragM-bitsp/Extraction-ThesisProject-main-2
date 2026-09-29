import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.canonical import BlockType, CanonicalBlock, CanonicalDocument
from retrieval.chunking import ChunkDraft, chunk_blocks
from retrieval.embeddings import HashingEmbedder
from retrieval.indexing import index_document
from retrieval.retriever import CosineRetriever, cosine_similarity
from storage.models import Base
from storage.object_store import LocalObjectStore
from storage.repository import PaperRepository


def block(text, page=1, section=None, block_type=BlockType.PARAGRAPH):
    return CanonicalBlock(block_type=block_type, page_number=page, section=section, text=text)


# ---- chunking -------------------------------------------------------------------


def test_small_blocks_merge_into_a_single_chunk():
    blocks = [block("Alpha beta gamma."), block("Delta epsilon zeta.")]
    chunks = chunk_blocks(blocks, target_tokens=200, overlap_tokens=10)
    assert len(chunks) == 1
    assert "Alpha" in chunks[0].text and "epsilon" in chunks[0].text


def test_chunk_splits_when_exceeding_target_tokens():
    long_block_a = block(" ".join(f"word{i}" for i in range(150)))
    long_block_b = block(" ".join(f"term{i}" for i in range(150)))
    chunks = chunk_blocks([long_block_a, long_block_b], target_tokens=200, overlap_tokens=10)
    assert len(chunks) == 2
    assert "word0" in chunks[0].text
    assert "term0" in chunks[1].text


def test_chunk_overlap_carries_tail_words_forward():
    long_block_a = block(" ".join(f"word{i}" for i in range(150)))
    long_block_b = block(" ".join(f"term{i}" for i in range(150)))
    chunks = chunk_blocks([long_block_a, long_block_b], target_tokens=200, overlap_tokens=10)
    tail_of_first = chunks[0].text.split()[-10:]
    start_of_second = chunks[1].text.split()[:10]
    assert tail_of_first == start_of_second


def test_chunk_breaks_at_section_boundary_even_under_token_target():
    intro = block("Short intro sentence.", page=1, section="Introduction")
    experimental = block("Short experimental sentence.", page=2, section="Experimental")
    chunks = chunk_blocks([intro, experimental], target_tokens=200, overlap_tokens=10)
    assert len(chunks) == 2
    assert chunks[0].section == "Introduction"
    assert chunks[1].section == "Experimental"


def test_section_boundary_does_not_leak_overlap_text_across_sections():
    """Regression test: overlap must only apply when a chunk splits due to
    length, never when it splits because the section changed — otherwise a
    short Introduction block gets carried whole into an Experimental-tagged
    chunk, contaminating section-scoped retrieval (LLD doc 2 section 10)."""
    intro = block("Silver nanoparticles have attracted attention for their antibacterial properties.", page=1, section="Introduction")
    experimental = block("Silver nitrate was dissolved in ethanol and heated at 80 C.", page=2, section="Experimental")
    results = block("TEM confirmed a particle size of 15 nm.", page=6, section="Results")

    chunks = chunk_blocks([intro, experimental, results], target_tokens=200, overlap_tokens=40)

    assert len(chunks) == 3
    assert "antibacterial" not in chunks[1].text
    assert "antibacterial" not in chunks[2].text
    assert "Silver nitrate" not in chunks[2].text


def test_chunk_tracks_page_range():
    blocks = [block("first", page=4), block("second", page=5), block("third", page=5)]
    chunks = chunk_blocks(blocks, target_tokens=200, overlap_tokens=5)
    assert len(chunks) == 1
    assert chunks[0].page_start == 4
    assert chunks[0].page_end == 5


def test_blank_blocks_are_skipped():
    blocks = [block("   "), block("real content here")]
    chunks = chunk_blocks(blocks)
    assert len(chunks) == 1
    assert "real content" in chunks[0].text


# ---- embeddings -------------------------------------------------------------------


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder(dimension=64)
    v1 = embedder.embed(["silver nitrate was dissolved in ethanol"])[0]
    v2 = embedder.embed(["silver nitrate was dissolved in ethanol"])[0]
    assert v1 == v2


def test_hashing_embedder_produces_unit_vectors():
    embedder = HashingEmbedder(dimension=64)
    vec = embedder.embed(["some reasonably long piece of scientific text"])[0]
    norm = sum(v * v for v in vec) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_hashing_embedder_similar_text_scores_higher_than_unrelated():
    embedder = HashingEmbedder(dimension=128)
    query = "concentration of silver nitrate precursor"
    relevant = "the silver nitrate precursor concentration was 10 mM"
    unrelated = "the cat sat quietly on the warm windowsill"

    q, r, u = embedder.embed([query, relevant, unrelated])
    assert cosine_similarity(q, r) > cosine_similarity(q, u)


# ---- indexing + retrieval (integration) --------------------------------------------


@pytest.fixture()
def repo():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with tempfile.TemporaryDirectory() as tmp:
        with Session(engine) as session:
            yield PaperRepository(session, LocalObjectStore(tmp))


def test_index_document_persists_embedded_chunks(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")

    canonical_doc = CanonicalDocument(
        paper_id=str(paper.paper_id),
        version=1,
        blocks=[
            block("Silver nanoparticles were characterized by TEM and UV-Vis.", page=1, section="Introduction"),
            block("Silver nitrate (10 mM) was dissolved in ethanol and heated at 80 C for 2 h.", page=2, section="Experimental"),
        ],
    )
    embedder = HashingEmbedder(dimension=64)

    records = index_document(repo, embedder, version.version_id, canonical_doc)

    assert len(records) == 2
    assert all(r.embedding_model == "hashing-64" for r in records)
    assert all(len(r.embedding) == 64 for r in records)

    fetched = repo.get_chunks(version.version_id)
    assert len(fetched) == 2


def test_cosine_retriever_ranks_the_relevant_chunk_first(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")

    canonical_doc = CanonicalDocument(
        paper_id=str(paper.paper_id),
        version=1,
        blocks=[
            block("Silver nitrate (10 mM) was dissolved in ethanol and heated at 80 C for 2 h.", page=2, section="Experimental"),
            block("The synthesized nanoparticles showed strong antibacterial activity against E. coli.", page=6, section="Results"),
        ],
    )
    embedder = HashingEmbedder(dimension=128)
    index_document(repo, embedder, version.version_id, canonical_doc)

    retriever = CosineRetriever(repo)
    query_embedding = embedder.embed(["what was the concentration of the silver nitrate precursor?"])[0]
    results = retriever.search(version.version_id, query_embedding, top_k=2)

    assert len(results) == 2
    assert "Silver nitrate (10 mM)" in results[0].chunk.text
    assert results[0].score > results[1].score


def test_cosine_retriever_respects_top_k(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")
    canonical_doc = CanonicalDocument(
        paper_id=str(paper.paper_id),
        version=1,
        blocks=[block(f"sentence number {i} about nanoparticle synthesis", page=1) for i in range(5)],
    )
    embedder = HashingEmbedder(dimension=32)
    # Small target forces each short sentence into its own chunk, so there
    # are enough distinct chunks for top_k to meaningfully truncate.
    index_document(repo, embedder, version.version_id, canonical_doc, target_tokens=5, overlap_tokens=0)

    retriever = CosineRetriever(repo)
    results = retriever.search(version.version_id, embedder.embed(["synthesis"])[0], top_k=3)
    assert len(results) == 3
