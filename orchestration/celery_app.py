"""
Celery application (LLD sections 24-27).

`task_always_eager` defaults to True in this environment: there's no Redis
broker running here, so `.delay()` executes the task body synchronously in
the calling thread instead of dispatching it to a separate worker process.
Set `CELERY_TASK_ALWAYS_EAGER=false` and point `CELERY_BROKER_URL` at a
real Redis instance to actually distribute work across workers — no task
body changes either way. This is the same "each paper is independently
processable" property Step 12's `ExperimentRunner` expresses as a thread
pool, expressed here as a task queue instead (LLD section 24: "For 50-100
papers, Celery + Redis is more than sufficient... you don't need
Kubernetes, Kafka and Spark just because this is called 'system design'").
"""

from __future__ import annotations

import os

from celery import Celery

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
ALWAYS_EAGER = os.environ.get("CELERY_TASK_ALWAYS_EAGER", "true").lower() != "false"

celery_app = Celery("sci_extract", broker=BROKER_URL, backend=BROKER_URL)
celery_app.conf.task_always_eager = ALWAYS_EAGER
celery_app.conf.task_eager_propagates = True  # eager-mode failures raise, not vanish silently
