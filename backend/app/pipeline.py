"""Review orchestration shared by the API (inline), the CLI, and the worker:

per-file -> optional RAG context -> inference -> confidence-based filter.
"""

from __future__ import annotations

import os
import time


def review_files(
    files: list[dict],
    repo_index: str | None = None,
    min_confidence: float | None = None,
    use_context: bool = True,
) -> dict:
    """files: [{"path": str, "diff": str}] -> review result dict."""
    from backend.app.inference import get_inference

    if min_confidence is None:
        min_confidence = float(os.getenv("MIN_CONFIDENCE", "0.55"))

    inference = get_inference()
    comments: list[dict] = []
    started = time.perf_counter()
    for f in files:
        context = None
        if use_context and repo_index and inference.name != "mock":
            from rag.retrieve import retrieve_context

            context = retrieve_context(repo_index, f["path"], f["diff"])
        rc = inference.review(f["diff"], f["path"], context)
        keep = rc.verdict == "BUG" and (rc.confidence is None or rc.confidence >= min_confidence)
        if keep:
            comments.append(rc.model_dump(exclude_none=True))
    return {
        "comments": comments,
        "verdict": "BUG" if comments else "CLEAN",
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "model_mode": inference.name,
        "n_files": len(files),
    }
