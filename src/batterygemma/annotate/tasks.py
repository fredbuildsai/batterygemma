"""A small resumable job-queue helper backed by `gen_tasks`, shared by every annotate/generate stage.

Each unit of generation work (e.g. "extract facts from this chunk") gets one row keyed by a caller-chosen
string. Re-running the same key is a no-op unless `force=True`, so a bulk `bg annotate ...` run can be
interrupted and resumed without re-billing the teacher LLM for chunks already done.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from batterygemma.db.models import GenTask


def get_or_create_task(session: Session, task_type: str, key: str) -> GenTask:
    task = session.scalars(select(GenTask).where(GenTask.key == key)).first()
    if task is None:
        task = GenTask(task_type=task_type, key=key, status="pending")
        session.add(task)
        session.flush()
    return task


def mark_done(task: GenTask, payload: dict[str, Any]) -> None:
    task.status, task.payload, task.last_error = "done", payload, None


def mark_failed(task: GenTask, error: str) -> None:
    task.status, task.last_error = "failed", error[:2000]
