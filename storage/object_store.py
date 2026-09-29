"""
Object storage for immutable artifacts (raw PDFs, canonical JSON).

LLD section 3: "Raw PDFs shouldn't go into PostgreSQL... I'd store immutable
source artifacts in object storage and keep only metadata and references in
Postgres." This module is that boundary: `ObjectStore` is the interface,
`LocalObjectStore` is what tests and local dev run against, and
`S3ObjectStore` is the same interface pointed at S3 or a MinIO endpoint in
production. Nothing that calls an ObjectStore needs to know which one it has.

Key layout (matches the LLD exactly):

    papers/{paper_id}/v{version}/original.pdf
    processed/{paper_id}/v{version}/canonical.json
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path


def raw_pdf_key(paper_id: uuid.UUID | str, version: int) -> str:
    return f"papers/{paper_id}/v{version}/original.pdf"


def canonical_json_key(paper_id: uuid.UUID | str, version: int) -> str:
    return f"processed/{paper_id}/v{version}/canonical.json"


class ObjectStore(ABC):
    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...


class LocalObjectStore(ObjectStore):
    """Filesystem-backed store. Used for local dev and the test suite so the
    whole pipeline is runnable without provisioning S3/MinIO first."""

    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def put_bytes(self, key: str, data: bytes) -> None:
        self._path(key).write_bytes(data)

    def get_bytes(self, key: str) -> bytes:
        path = self.root / key
        if not path.exists():
            raise FileNotFoundError(key)
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()

    def delete(self, key: str) -> None:
        path = self.root / key
        if path.exists():
            path.unlink()


class S3ObjectStore(ObjectStore):
    """
    S3- or MinIO-backed store (same API — MinIO is S3-compatible, so this
    only needs an `endpoint_url` override to point at a local MinIO instance
    instead of AWS).

    Not exercised by the test suite (no S3/MinIO endpoint available in this
    environment) — swap this in for `LocalObjectStore` once infra exists;
    the repository layer in repository.py depends only on the `ObjectStore`
    interface, so no other code changes.
    """

    def __init__(self, bucket: str, endpoint_url: str | None = None, **client_kwargs):
        import boto3  # local import: keep boto3 optional for pure-local dev

        self.bucket = bucket
        self._client = boto3.client("s3", endpoint_url=endpoint_url, **client_kwargs)

    def put_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get_bytes(self, key: str) -> bytes:
        obj = self._client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)
