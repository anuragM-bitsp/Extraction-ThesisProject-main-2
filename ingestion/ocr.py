"""
Conditional OCR.

LLD section 9: "OCR is expensive and can introduce recognition errors, so
I'd invoke it conditionally when the native PDF text layer is missing or
inadequate... Don't OCR every paper unnecessarily."

Split into two pieces on purpose:
  - `needs_ocr()` is a pure function of (page_count, extracted_chars) — no
    subprocess, no I/O, fully unit-testable, and it's the actual decision
    logic worth getting right.
  - `run_ocr()` shells out to the real `ocrmypdf` CLI (which wraps
    Tesseract). It's an integration point, not where the interesting logic
    lives, so it's kept thin and tested with an actually-scanned-looking
    PDF rather than mocked.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

# Below this many characters per page, we treat a PDF as "no usable text
# layer" rather than "a paper that happens to have short pages". A typical
# scientific-paper page has 2,000+ characters of body text; a scanned page
# with zero OCR already applied reports 0.
MIN_CHARS_PER_PAGE = 100


class OcrNotAvailableError(RuntimeError):
    """Raised when ocrmypdf/tesseract aren't installed on this machine."""


def needs_ocr(page_count: int, extracted_chars: int, min_chars_per_page: int = MIN_CHARS_PER_PAGE) -> bool:
    if page_count <= 0:
        return False
    return (extracted_chars / page_count) < min_chars_per_page


def ocr_is_available() -> bool:
    return shutil.which("ocrmypdf") is not None


def run_ocr(pdf_bytes: bytes, *, timeout_seconds: int = 120) -> bytes:
    """
    Run OCRmyPDF (Tesseract under the hood) over `pdf_bytes` and return a
    new PDF with a searchable text layer burned in. Raises OcrNotAvailableError
    if the `ocrmypdf` binary isn't on PATH, and subprocess.CalledProcessError
    if OCR itself fails (e.g. corrupt PDF) — callers should treat that as a
    permanent failure for this document, not something to retry blindly
    (LLD section 26: don't retry permanent errors indefinitely).
    """
    if not ocr_is_available():
        raise OcrNotAvailableError(
            "ocrmypdf is not installed. On Debian/Ubuntu: "
            "apt-get install ocrmypdf tesseract-ocr"
        )

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "input.pdf"
        out_path = Path(tmp) / "output.pdf"
        in_path.write_bytes(pdf_bytes)

        subprocess.run(
            [
                "ocrmypdf",
                "--force-ocr",       # this PDF has no/insufficient text layer, so nothing to skip
                "--quiet",
                str(in_path),
                str(out_path),
            ],
            check=True,
            timeout=timeout_seconds,
            capture_output=True,
        )
        return out_path.read_bytes()
