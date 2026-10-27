"""Celery worker (optional; the API runs the pipeline inline when REDIS_URL is
not set). Kept behind a lazy import so the API never requires celery.
"""

from __future__ import annotations

import os


def celery_app():
    from celery import Celery

    app = Celery("prguardian", broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    app.conf.task_serializer = "json"
    return app


def run_review_task(payload: dict) -> dict:
    """Shared implementation so the worker and tests exercise the same code."""
    from backend.app.pipeline import review_files

    files = payload["files"]
    return review_files(files, repo_index=payload.get("repo_index"))


if os.getenv("REDIS_URL"):

    _app = celery_app()

    @_app.task(name="prguardian.review")
    def review_task(payload: dict) -> dict:
        return run_review_task(payload)
