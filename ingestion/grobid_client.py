"""
GROBID client.

GROBID runs as its own service (typically a Docker container exposing
localhost:8070) — it is not a Python library, and there's no instance
running in this environment to integration-test against. So, as with
ObjectStore in Step 2: an abstract interface, one real implementation
(`HttpGrobidClient`, talking to the actual GROBID REST API — correctness of
the *parsing* is covered separately in grobid_tei.py, and the HTTP plumbing
here is covered by tests that fake `requests.post`), and callers (the
ingestion pipeline) depend only on `GrobidClient` so a fake can stand in
whenever a real server isn't reachable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import requests


class GrobidUnavailableError(RuntimeError):
    pass


class GrobidClient(ABC):
    @abstractmethod
    def process_fulltext(self, pdf_bytes: bytes) -> str:
        """Returns TEI XML as a string."""
        ...

    @abstractmethod
    def is_alive(self) -> bool: ...


class HttpGrobidClient(GrobidClient):
    def __init__(self, base_url: str = "http://localhost:8070", timeout_seconds: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def is_alive(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/isalive", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def process_fulltext(self, pdf_bytes: bytes) -> str:
        try:
            resp = requests.post(
                f"{self.base_url}/api/processFulltextDocument",
                files={"input": ("document.pdf", pdf_bytes, "application/pdf")},
                timeout=self.timeout_seconds,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise GrobidUnavailableError(f"GROBID request failed: {exc}") from exc
        return resp.text
