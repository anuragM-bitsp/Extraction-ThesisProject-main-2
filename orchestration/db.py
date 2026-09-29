"""
DB engine, session, and object-store construction from environment
variables — used identically by the FastAPI app (per-request session) and
Celery tasks (per-task session), so both sides of "web enqueues, worker
processes" agree on where data actually lives without sharing memory.

Deliberately reads env vars fresh on every call rather than caching a
module-level engine: this is what lets tests point `DATABASE_URL` at a
temp-file SQLite database per test (`monkeypatch.setenv`) and have both the
API's dependency and a Celery task's own session — which, in eager mode,
runs in the same process but must still open its own session, exactly like
a real separate worker would — see the same data. A production deployment
would cache/share the engine (e.g. via FastAPI's lifespan) rather than
reconstruct it on every call; noted here rather than built, since this
project's scale (50-100 papers) doesn't need connection-pool tuning yet.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from storage.models import Base
from storage.object_store import LocalObjectStore, ObjectStore


def get_engine():
    database_url = os.environ.get("DATABASE_URL", "sqlite:///./sci_extract.db")
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)  # no-op if tables already exist
    return engine


def get_session() -> Session:
    return Session(get_engine())


def get_object_store() -> ObjectStore:
    root = os.environ.get("OBJECT_STORE_ROOT", "./object_store")
    return LocalObjectStore(root)
