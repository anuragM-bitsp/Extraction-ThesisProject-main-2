import tempfile
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from storage.models import Base
from storage.object_store import LocalObjectStore, canonical_json_key, raw_pdf_key
from storage.repository import PaperRepository


@pytest.fixture()
def session():
    """Fresh in-memory SQLite DB per test. Same ORM models will run
    unchanged against real Postgres — see storage/db_types.py."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture()
def object_store():
    with tempfile.TemporaryDirectory() as tmp:
        yield LocalObjectStore(tmp)


@pytest.fixture()
def repo(session, object_store):
    return PaperRepository(session, object_store)


def test_create_paper(repo):
    paper = repo.create_paper(title="Green synthesis of AgNPs", doi="10.1000/xyz")
    assert paper.paper_id is not None
    assert repo.get_paper(paper.paper_id).title == "Green synthesis of AgNPs"


def test_first_upload_creates_version_1(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 fake bytes")
    assert version.version == 1
    assert version.status == "INGESTED"
    assert len(version.content_hash) == 64  # sha256 hex


def test_reuploading_identical_pdf_does_not_create_a_new_version(repo):
    """LLD section 4: content_hash lets us detect a no-op re-upload instead
    of re-triggering the whole processing pipeline."""
    paper = repo.create_paper(title="P1")
    pdf_bytes = b"%PDF-1.4 identical content"

    v1 = repo.add_document_version(paper.paper_id, pdf_bytes)
    v1_again = repo.add_document_version(paper.paper_id, pdf_bytes)

    assert v1.version_id == v1_again.version_id
    assert v1.version == v1_again.version == 1


def test_uploading_different_pdf_creates_version_2(repo):
    paper = repo.create_paper(title="P1")
    v1 = repo.add_document_version(paper.paper_id, b"%PDF-1.4 original")
    v2 = repo.add_document_version(paper.paper_id, b"%PDF-1.4 revised manuscript")

    assert v2.version == 2
    assert v1.version_id != v2.version_id
    assert repo.get_latest_version(paper.paper_id).version_id == v2.version_id


def test_raw_pdf_bytes_are_retrievable_from_object_store(repo, object_store):
    paper = repo.create_paper(title="P1")
    pdf_bytes = b"%PDF-1.4 the actual paper"
    version = repo.add_document_version(paper.paper_id, pdf_bytes)

    key = raw_pdf_key(paper.paper_id, version.version)
    assert object_store.exists(key)
    assert object_store.get_bytes(key) == pdf_bytes


def test_canonical_document_stored_separately_from_raw_pdf(repo, object_store):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")

    key = repo.store_canonical_document(paper.paper_id, version.version, b'{"blocks": []}')
    assert key == canonical_json_key(paper.paper_id, version.version)
    assert object_store.get_bytes(key) == b'{"blocks": []}'


def test_blocks_carry_the_provenance_chain_paper_version_page_block(repo):
    """This is what makes EvidenceSpan (Step 1) resolvable: given a block_id
    you can walk back to page, version, and paper."""
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")

    blocks = repo.add_blocks(
        version.version_id,
        [
            dict(block_type="paragraph", page_number=4, section="Experimental",
                 text="heated at 80 \u00b0C for 2 h"),
            dict(block_type="table", page_number=5, section="Results",
                 text="", bbox={"x0": 10, "y0": 20, "x1": 300, "y1": 400}),
        ],
    )
    assert len(blocks) == 2
    fetched = repo.get_blocks(version.version_id)
    assert [b.page_number for b in fetched] == [4, 5]
    assert fetched[1].bbox == {"x0": 10, "y0": 20, "x1": 300, "y1": 400}


def test_chunks_store_embeddings_and_round_trip(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")

    chunks = repo.add_chunks(
        version.version_id,
        [
            dict(
                text="Silver nitrate was dissolved in ethanol...",
                section="Experimental",
                page_start=4,
                page_end=4,
                token_count=12,
                embedding_model="bge-small-en",
                embedding=[0.1, 0.2, 0.3],
            )
        ],
    )
    fetched = repo.get_chunks(version.version_id)
    assert fetched[0].embedding == pytest.approx([0.1, 0.2, 0.3])
    assert fetched[0].embedding_model == "bge-small-en"


def test_duplicate_paper_version_pair_is_rejected_at_db_level(session, object_store):
    """Belt-and-suspenders: even if repository logic were bypassed, the
    UNIQUE(paper_id, version) constraint from the LLD schema still holds."""
    from storage.models import DocumentVersionORM, PaperORM

    paper = PaperORM(title="P1")
    session.add(paper)
    session.flush()

    session.add(DocumentVersionORM(paper_id=paper.paper_id, version=1, content_hash="a" * 64))
    session.flush()

    session.add(DocumentVersionORM(paper_id=paper.paper_id, version=1, content_hash="b" * 64))
    with pytest.raises(Exception):  # IntegrityError, dialect-specific
        session.flush()


def test_submit_annotation_stores_full_payload(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")

    payload = {"paper_id": "P001", "temperature": {"value": 80.0, "unit": "C"}}
    record = repo.submit_annotation(version.version_id, "annotator_a", payload)

    assert record.annotator_id == "annotator_a"
    assert record.payload == payload


def test_resubmitting_annotation_overwrites_not_duplicates(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")

    repo.submit_annotation(version.version_id, "annotator_a", {"solvent": "ethanol"})
    repo.submit_annotation(version.version_id, "annotator_a", {"solvent": "methanol"})

    annotations = repo.get_annotations(version.version_id)
    assert len(annotations) == 1
    assert annotations[0].payload == {"solvent": "methanol"}


def test_two_different_annotators_produce_two_annotation_rows(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")

    repo.submit_annotation(version.version_id, "annotator_a", {"solvent": "ethanol"})
    repo.submit_annotation(version.version_id, "annotator_b", {"solvent": "ethanol"})

    annotations = repo.get_annotations(version.version_id)
    assert {a.annotator_id for a in annotations} == {"annotator_a", "annotator_b"}


def test_submit_and_fetch_gold(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")
    ann_a = repo.submit_annotation(version.version_id, "annotator_a", {"solvent": "ethanol"})

    gold = repo.submit_gold(
        version.version_id,
        payload={"paper_id": "P001", "solvent": "ethanol"},
        adjudicated_by="lead_annotator",
        source_annotation_ids=[str(ann_a.annotation_id)],
    )
    assert repo.get_gold(version.version_id).payload == gold.payload
    assert repo.get_gold(version.version_id).adjudicated_by == "lead_annotator"


def test_get_gold_returns_none_when_not_yet_adjudicated(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")
    assert repo.get_gold(version.version_id) is None


def test_store_extraction_result_is_idempotent(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")
    payload = {"paper_id": "P001", "temperature": {"value": 80.0, "unit": "C"}}

    first = repo.store_extraction_result(version.version_id, "RULE", "rule@0.1.0", "v1", payload)
    second = repo.store_extraction_result(version.version_id, "RULE", "rule@0.1.0", "v1", payload)

    assert first.extraction_id == second.extraction_id
    assert len(repo.get_extraction_results(version.version_id)) == 1


def test_store_extraction_result_different_extractor_versions_coexist(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")

    repo.store_extraction_result(version.version_id, "RULE", "rule@0.1.0", "v1", {"paper_id": "P001"})
    repo.store_extraction_result(version.version_id, "RULE", "rule@0.2.0", "v1", {"paper_id": "P001"})

    assert len(repo.get_extraction_results(version.version_id, extractor="RULE")) == 2


def test_job_lifecycle_success(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")

    job = repo.create_job(version.version_id, job_type="extract", extractor="RULE")
    assert job.status == "QUEUED"

    repo.mark_job_running(job.job_id)
    assert repo.get_job(job.job_id).status == "RUNNING"

    repo.mark_job_success(job.job_id)
    final = repo.get_job(job.job_id)
    assert final.status == "SUCCESS"
    assert final.error_message is None


def test_job_retries_then_dead_letters(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")
    job = repo.create_job(version.version_id, job_type="process")

    repo.mark_job_failed(job.job_id, "timeout", max_retries=3)
    assert repo.get_job(job.job_id).status == "QUEUED"  # retry 1/3, still retryable

    repo.mark_job_failed(job.job_id, "timeout", max_retries=3)
    repo.mark_job_failed(job.job_id, "timeout", max_retries=3)
    final = repo.get_job(job.job_id)
    assert final.status == "DEAD_LETTER"
    assert final.retry_count == 3


def test_get_jobs_lists_all_jobs_for_a_version(repo):
    paper = repo.create_paper(title="P1")
    version = repo.add_document_version(paper.paper_id, b"%PDF-1.4 raw")
    repo.create_job(version.version_id, job_type="process")
    repo.create_job(version.version_id, job_type="extract", extractor="NER")

    jobs = repo.get_jobs(version.version_id)
    assert len(jobs) == 2
    assert {j.job_type for j in jobs} == {"process", "extract"}
