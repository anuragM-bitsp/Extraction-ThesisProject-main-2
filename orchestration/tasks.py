"""
Celery tasks: the async wrappers around `PipelineService` (LLD sections
24-26). Each task constructs its own DB session and object store rather
than receiving one — exactly what a real, separate worker process would
have to do, since it shares no memory with whatever enqueued the task.

Retry/DLQ flow (LLD section 26's diagram, implemented literally):

    job -> RUNNING -> exception -> mark_job_failed()
                                       |
                          retry_count < max_retries?
                             /                  \\
                          yes                    no
                           |                      |
                    self.retry()               leave as
                    asks Celery to             DEAD_LETTER —
                    reschedule the             a permanent
                    task                       failure, not
                                                retried forever

A real gotcha worth knowing about `CELERY_TASK_ALWAYS_EAGER` (this
project's default, since there's no Redis broker in this sandbox):
`self.retry()` does NOT automatically re-run the task in eager mode. In
production, a real broker+worker catches the `Retry` exception and
reschedules the task after `default_retry_delay` — that's what "Celery
handles retries" actually means. In eager mode there is no scheduler, so
`self.retry()` just raises `celery.exceptions.Retry` synchronously back to
whatever called `.delay()`. This is genuine, documented Celery behavior,
not a bug — but it means testing the full retry-to-DEAD_LETTER flow
offline requires a small loop that plays the part of the missing scheduler
(see `tests/test_orchestration.py`'s `_run_with_simulated_retries`), rather
than a single `.delay()` call resolving the whole thing.
"""

from __future__ import annotations

import uuid

from orchestration.celery_app import celery_app
from orchestration.db import get_object_store, get_session
from orchestration.extractor_registry import get_extractor
from orchestration.service import PipelineService
from storage.repository import PaperRepository


@celery_app.task(bind=True, max_retries=3, default_retry_delay=2)
def process_document_task(self, job_id: str, version_id: str) -> None:
    session = get_session()
    try:
        repo = PaperRepository(session, get_object_store())
        repo.mark_job_running(uuid.UUID(job_id))
        session.commit()

        PipelineService(repo).ingest_document(uuid.UUID(version_id))

        repo.mark_job_success(uuid.UUID(job_id))
        session.commit()
    except Exception as exc:
        session.rollback()
        repo = PaperRepository(session, get_object_store())
        job = repo.mark_job_failed(uuid.UUID(job_id), str(exc))
        session.commit()
        if job.status == "QUEUED":
            raise self.retry(exc=exc)
        # else DEAD_LETTER: the job row is the permanent record; don't loop.
    finally:
        session.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=2)
def extract_task(
    self,
    job_id: str,
    version_id: str,
    extractor_name: str,
    prompt: str | None = None,
    model: str | None = None,
) -> None:
    session = get_session()
    try:
        repo = PaperRepository(session, get_object_store())
        repo.mark_job_running(uuid.UUID(job_id))
        session.commit()

        extractor = get_extractor(extractor_name, prompt=prompt, model=model)
        PipelineService(repo).run_extraction(uuid.UUID(version_id), extractor)

        repo.mark_job_success(uuid.UUID(job_id))
        session.commit()
    except Exception as exc:
        session.rollback()
        repo = PaperRepository(session, get_object_store())
        job = repo.mark_job_failed(uuid.UUID(job_id), str(exc))
        session.commit()
        if job.status == "QUEUED":
            raise self.retry(exc=exc)
    finally:
        session.close()
