"""
Dialect-aware column types.

The LLD calls for PostgreSQL + JSONB (+ pgvector for embeddings). But this
repo needs to run its test suite without a live Postgres instance available
(no DB server in this sandbox), and a thesis project benefits from being
runnable on a laptop with `sqlite:///` before anyone provisions Postgres.

Rather than fork the schema into "real" vs "test" versions, each type below
picks the Postgres-native implementation when talking to Postgres, and falls
back to a portable equivalent otherwise. The ORM models in models.py never
know which one they got.
"""

from __future__ import annotations

import json
import uuid

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.types import CHAR, JSON, TypeDecorator


class GUID(TypeDecorator):
    """UUID primary/foreign keys. Native `uuid` type on Postgres, CHAR(32)
    hex elsewhere (e.g. SQLite for local dev/tests)."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return str(value)
        if isinstance(value, uuid.UUID):
            return value.hex
        return uuid.UUID(str(value)).hex

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


class JSONVariant(TypeDecorator):
    """JSONB on Postgres (indexable, binary-stored); plain JSON elsewhere.
    Used for `bbox` and any other semi-structured column, matching the
    `document_blocks.bbox JSONB` column in the LLD."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class EmbeddingType(TypeDecorator):
    """
    Chunk embedding vector.

    Stored as a JSON array of floats for now. Step 4 (chunking + retrieval)
    is where the embedding model — and therefore the vector dimension — gets
    picked; only at that point does it make sense to switch this to
    `pgvector.sqlalchemy.Vector(dim)` for real ANN indexing on Postgres.
    Wiring pgvector before that would mean hard-coding a dimension nothing
    else in the project depends on yet.
    """

    impl = JSON
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.dumps(list(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return json.loads(value) if isinstance(value, str) else value
