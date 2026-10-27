"""PR Guardian API.

POST /review   {"diff": "..."} or {"pr_url": "..."} (needs GITHUB_TOKEN)
GET  /reviews/{id}
GET  /stats
POST /feedback {"review_id": "...", "helpful": true}
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:  # optional; .env support is a convenience, not a requirement
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from backend.app.github_client import fetch_pr_files, parse_pr_url
from backend.app.pipeline import review_files
from mining.diff_utils import split_multi_file_diff

app = FastAPI(title="PR Guardian", version="0.1.0")

_REVIEWS: dict[str, dict] = {}
_FEEDBACK: list[dict] = []


class ReviewRequest(BaseModel):
    diff: str | None = Field(default=None, description="unified diff, one or more files")
    pr_url: str | None = None
    repo_index: str | None = None
    use_context: bool = True


class ReviewResponse(BaseModel):
    id: str
    verdict: str
    comments: list[dict]
    latency_ms: int
    model_mode: str
    n_files: int


class FeedbackRequest(BaseModel):
    review_id: str
    helpful: bool
    comment: str | None = None


def _files_from_request(req: ReviewRequest) -> list[dict]:
    if req.pr_url:
        parsed = parse_pr_url(req.pr_url)
        if not parsed:
            raise HTTPException(400, "pr_url must look like https://github.com/owner/repo/pull/N")
        if not os.getenv("GITHUB_TOKEN"):
            raise HTTPException(400, "GITHUB_TOKEN is not configured — paste the diff instead")
        owner, repo, number = parsed
        try:
            return fetch_pr_files(owner, repo, number)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"GitHub API error: {e}")
    if req.diff and req.diff.strip():
        return [{"path": p, "diff": d} for p, d in split_multi_file_diff(req.diff)]
    raise HTTPException(400, "provide either `diff` or `pr_url`")


@app.post("/review", response_model=ReviewResponse)
def review(req: ReviewRequest) -> ReviewResponse:
    files = _files_from_request(req)
    if not files:
        raise HTTPException(422, "no reviewable files in input")
    result = review_files(files, repo_index=req.repo_index, use_context=req.use_context)
    review_id = uuid.uuid4().hex[:12]
    record = {
        "id": review_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    _REVIEWS[review_id] = record
    return ReviewResponse(
        id=review_id,
        verdict=record["verdict"],
        comments=record["comments"],
        latency_ms=record["latency_ms"],
        model_mode=record["model_mode"],
        n_files=record["n_files"],
    )


@app.get("/reviews/{review_id}")
def get_review(review_id: str):
    if review_id not in _REVIEWS:
        raise HTTPException(404, "review not found")
    return _REVIEWS[review_id]


@app.get("/stats")
def stats():
    n_bug = sum(1 for r in _REVIEWS.values() if r["verdict"] == "BUG")
    lat = [r["latency_ms"] for r in _REVIEWS.values()] or [0]
    return {
        "total_reviews": len(_REVIEWS),
        "bug_reviews": n_bug,
        "clean_reviews": len(_REVIEWS) - n_bug,
        "avg_latency_ms": round(sum(lat) / len(lat)),
        "feedback_count": len(_FEEDBACK),
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.review_id not in _REVIEWS:
        raise HTTPException(404, "review not found")
    _FEEDBACK.append({**req.model_dump(), "at": datetime.now(timezone.utc).isoformat()})
    return {"ok": True, "total_feedback": len(_FEEDBACK)}


@app.get("/health")
def health():
    return {"status": "ok", "mock_mode": os.getenv("PRGUARDIAN_MOCK", "1") != "0"}
