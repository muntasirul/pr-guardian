"""GitHub API client: fetch PR file diffs, post inline review comments.

All HTTP goes through _request() so tests can monkeypatch it. Needs a
GITHUB_TOKEN with repo scope; read-only for fetching, write for commenting.
"""

from __future__ import annotations

import os
import re

import requests

GITHUB_API = "https://api.github.com"

PR_URL_RE = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)")


def parse_pr_url(url: str) -> tuple[str, str, int] | None:
    m = PR_URL_RE.search(url)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def _request(method: str, url: str, token: str | None = None, json_body: dict | None = None):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return requests.request(method, url, headers=headers, json=json_body, timeout=30)


def fetch_pr_files(owner: str, repo: str, number: int, token: str | None = None) -> list[dict]:
    """Return [{"path": ..., "patch": ...}] for a PR (first 100 files)."""
    token = token or os.getenv("GITHUB_TOKEN")
    r = _request("GET", f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}/files?per_page=100", token)
    r.raise_for_status()
    return [
        {"path": f["filename"], "patch": f.get("patch") or ""}
        for f in r.json()
        if f.get("patch")
    ]


def post_inline_comment(
    owner: str,
    repo: str,
    number: int,
    commit_id: str,
    path: str,
    line: int,
    body: str,
    token: str | None = None,
) -> dict:
    """Post a single inline PR comment anchored to a diff line (best effort:
    uses the modern `line` anchor API)."""
    token = token or os.getenv("GITHUB_TOKEN")
    payload = {
        "commit_id": commit_id,
        "path": path,
        "line": line,
        "body": body,
        "subject_type": "line",
    }
    r = _request(
        "POST", f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}/comments", token, payload
    )
    r.raise_for_status()
    return r.json()


def get_pr_head(owner: str, repo: str, number: int, token: str | None = None) -> str:
    token = token or os.getenv("GITHUB_TOKEN")
    r = _request("GET", f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}", token)
    r.raise_for_status()
    return r.json()["head"]["sha"]
