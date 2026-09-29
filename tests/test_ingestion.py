from __future__ import annotations

import io

import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

from ingestion import ocr as ocr_module
from ingestion.grobid_client import GrobidClient, GrobidUnavailableError, HttpGrobidClient
from ingestion.grobid_tei import parse_tei
from ingestion.pipeline import DocumentIngestionPipeline
from ingestion.pymupdf_parser import extract_blocks, page_count, total_extractable_chars

INTRO_SENTENCE = (
    "Silver nanoparticles have attracted attention due to their antibacterial "
    "and catalytic properties across biomedical applications."
)
EXPERIMENTAL_SENTENCE = "Silver nitrate was dissolved in ethanol and heated at 80 C for 2 h."

FAKE_TEI = f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title level="a" type="main">Green Synthesis of Silver Nanoparticles Using Plant Extract</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author><persName><forename type="first">Aditi</forename><surname>Sharma</surname></persName></author>
            <author><persName><forename type="first">Rohan</forename><surname>Verma</surname></persName></author>
          </analytic>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract><p>This study reports a green synthesis route for silver nanoparticles.</p></abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div><head>Introduction</head><p>{INTRO_SENTENCE}</p></div>
      <div><head>Experimental</head><p>{EXPERIMENTAL_SENTENCE}</p></div>
    </body>
    <back>
      <div type="references">
        <listBibl>
          <biblStruct xml:id="b0"><analytic><title level="article">Synthesis of metal nanoparticles</title></analytic></biblStruct>
          <biblStruct xml:id="b1"><analytic><title level="article">Antibacterial activity of AgNPs</title></analytic></biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>"""


def make_text_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    return doc.tobytes()


def make_multi_block_pdf(paragraphs: list[str]) -> bytes:
    """Each paragraph is placed with a vertical gap so PyMuPDF's layout
    analysis treats them as separate blocks — needed for the section-tagging
    tests, which check a specific block's `.section`, not the whole page."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for para in paragraphs:
        page.insert_text((72, y), para, fontsize=11)
        y += 60
    return doc.tobytes()


def make_scanned_pdf(word: str) -> bytes:
    """A PDF containing only an image of text — no native text layer at all,
    the way a photographed/scanned page behaves."""
    img = Image.new("RGB", (600, 200), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=60)
    draw.text((20, 60), word, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=200)
    page.insert_image(pymupdf.Rect(0, 0, 600, 200), stream=buf.getvalue())
    return doc.tobytes()


class FakeGrobidClient(GrobidClient):
    def __init__(self, tei_xml: str | None = None, raise_error: bool = False):
        self.tei_xml = tei_xml
        self.raise_error = raise_error

    def is_alive(self) -> bool:
        return not self.raise_error

    def process_fulltext(self, pdf_bytes: bytes) -> str:
        if self.raise_error:
            raise GrobidUnavailableError("simulated GROBID outage")
        return self.tei_xml


# ---- PyMuPDF -----------------------------------------------------------------


def test_extract_blocks_finds_text_and_bbox():
    pdf_bytes = make_text_pdf(EXPERIMENTAL_SENTENCE)
    blocks = extract_blocks(pdf_bytes)
    assert len(blocks) >= 1
    assert any(EXPERIMENTAL_SENTENCE in b.text for b in blocks)
    assert blocks[0].page_number == 1
    assert blocks[0].bbox is not None


def test_page_count_and_char_count():
    pdf_bytes = make_text_pdf(EXPERIMENTAL_SENTENCE)
    assert page_count(pdf_bytes) == 1
    assert total_extractable_chars(pdf_bytes) >= len(EXPERIMENTAL_SENTENCE)


# ---- OCR decision (pure function) ---------------------------------------------


@pytest.mark.parametrize(
    "pages,chars,expected",
    [
        (5, 50, True),      # 10 chars/page -> clearly scanned
        (5, 5000, False),   # 1000 chars/page -> real text
        (1, 0, True),       # totally empty page
        (0, 0, False),      # no pages at all: nothing to OCR
    ],
)
def test_needs_ocr_decision(pages, chars, expected):
    assert ocr_module.needs_ocr(pages, chars) is expected


# ---- OCR end-to-end (real ocrmypdf/tesseract subprocess) ----------------------


@pytest.mark.skipif(not ocr_module.ocr_is_available(), reason="ocrmypdf not installed")
def test_run_ocr_recovers_text_from_image_only_pdf():
    scanned_pdf = make_scanned_pdf("EXPERIMENTAL")
    assert total_extractable_chars(scanned_pdf) == 0  # confirm it's genuinely image-only

    ocred_pdf = ocr_module.run_ocr(scanned_pdf)
    recovered_text = "".join(b.text for b in extract_blocks(ocred_pdf)).upper()
    assert "EXPERIMENTAL" in recovered_text


# ---- GROBID TEI parsing --------------------------------------------------------


def test_parse_tei_extracts_title_authors_abstract_sections_references():
    metadata = parse_tei(FAKE_TEI)
    assert metadata.title == "Green Synthesis of Silver Nanoparticles Using Plant Extract"
    assert metadata.authors == ["Aditi Sharma", "Rohan Verma"]
    assert "green synthesis route" in metadata.abstract
    assert [s.heading for s in metadata.sections] == ["Introduction", "Experimental"]
    assert metadata.sections[1].text == EXPERIMENTAL_SENTENCE
    assert len(metadata.references) == 2


# ---- GROBID HTTP client (transport only, requests.post faked) -----------------


class _FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def test_http_grobid_client_posts_pdf_and_returns_tei(monkeypatch):
    captured = {}

    def fake_post(url, files=None, timeout=None):
        captured["url"] = url
        captured["has_input_file"] = "input" in files
        return _FakeResponse(FAKE_TEI)

    monkeypatch.setattr("ingestion.grobid_client.requests.post", fake_post)

    client = HttpGrobidClient(base_url="http://localhost:8070")
    result = client.process_fulltext(b"%PDF-1.4 fake")

    assert result == FAKE_TEI
    assert captured["url"] == "http://localhost:8070/api/processFulltextDocument"
    assert captured["has_input_file"]


def test_http_grobid_client_wraps_connection_errors(monkeypatch):
    import requests

    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr("ingestion.grobid_client.requests.post", fake_post)
    client = HttpGrobidClient()

    with pytest.raises(GrobidUnavailableError):
        client.process_fulltext(b"%PDF-1.4 fake")


# ---- Full pipeline --------------------------------------------------------------


def test_pipeline_without_grobid_still_produces_canonical_document():
    pdf_bytes = make_multi_block_pdf([INTRO_SENTENCE, EXPERIMENTAL_SENTENCE])
    pipeline = DocumentIngestionPipeline(grobid_client=None)

    doc = pipeline.ingest(paper_id="P001", version=1, pdf_bytes=pdf_bytes)

    assert doc.used_grobid is False
    assert doc.title is None
    assert doc.was_ocred is False  # plenty of native text — OCR must not trigger
    assert any(EXPERIMENTAL_SENTENCE in b.text for b in doc.blocks)


def test_pipeline_with_working_grobid_merges_metadata_and_tags_sections():
    pdf_bytes = make_multi_block_pdf([INTRO_SENTENCE, EXPERIMENTAL_SENTENCE])
    pipeline = DocumentIngestionPipeline(grobid_client=FakeGrobidClient(tei_xml=FAKE_TEI))

    doc = pipeline.ingest(paper_id="P001", version=1, pdf_bytes=pdf_bytes)

    assert doc.used_grobid is True
    assert doc.title == "Green Synthesis of Silver Nanoparticles Using Plant Extract"
    assert [a.full_name for a in doc.authors] == ["Aditi Sharma", "Rohan Verma"]
    assert len(doc.references) == 2

    matched = [b for b in doc.blocks if EXPERIMENTAL_SENTENCE in b.text]
    assert matched and matched[0].section == "Experimental"


def test_pipeline_falls_back_gracefully_when_grobid_is_down():
    """If GROBID is unreachable, ingestion must still succeed — GROBID is an
    enrichment layer, not a hard dependency (LLD section 8)."""
    pdf_bytes = make_multi_block_pdf([INTRO_SENTENCE, EXPERIMENTAL_SENTENCE])
    pipeline = DocumentIngestionPipeline(grobid_client=FakeGrobidClient(raise_error=True))

    doc = pipeline.ingest(paper_id="P001", version=1, pdf_bytes=pdf_bytes)

    assert doc.used_grobid is False
    assert doc.title is None
    assert any(EXPERIMENTAL_SENTENCE in b.text for b in doc.blocks)


def test_pipeline_runs_ocr_when_needed_and_still_returns_canonical_document():
    scanned_pdf = make_scanned_pdf("EXPERIMENTAL")
    pipeline = DocumentIngestionPipeline(grobid_client=None)

    doc = pipeline.ingest(paper_id="P002", version=1, pdf_bytes=scanned_pdf)

    if ocr_module.ocr_is_available():
        assert doc.was_ocred is True
        assert "EXPERIMENTAL" in doc.full_text().upper()
    else:
        assert doc.was_ocred is False  # graceful degradation path
