"""Deterministic synthetic git repo used to test the mining pipeline end-to-end.

History (backdated so survival windows are testable):
    c1 (2020-01-01) adds app.py (safe) and stable.py
    c2 (2020-01-02) introduces a bug in app.py (floor division)
    c3 (2021-06-01) fixes it (message scores as a fix commit)
    c4 (2024-06-01) adds README only  -> repo "now" is 2024
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

GIT_ENV_BASE = {
    "GIT_AUTHOR_NAME": "Tester",
    "GIT_AUTHOR_EMAIL": "tester@example.com",
    "GIT_COMMITTER_NAME": "Tester",
    "GIT_COMMITTER_EMAIL": "tester@example.com",
}


def _git(repo: Path, *args: str, date: str | None = None, check: bool = True) -> str:
    env = {**os.environ, **GIT_ENV_BASE}
    if date:
        env["GIT_AUTHOR_DATE"] = date
        env["GIT_COMMITTER_DATE"] = date
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {args} failed: {proc.stderr.strip()}")
    return proc.stdout


def head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").strip()


def commit(repo: Path, msg: str, date: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", msg, "--quiet", date=date)
    return head(repo)


def build_scenario(root: Path) -> dict[str, Path | str]:
    repo = root / "toyproj"
    repo.mkdir(parents=True)
    _git(repo, "init", "--quiet")

    app = repo / "app.py"
    app.write_text("def divide(a, b):\n    return a / b\n")
    (repo / "stable.py").write_text("VALUE = 1\n\ndef compute(x):\n    return x * VALUE\n")
    c1 = commit(repo, "add app module and stable helper", "2020-01-01 12:00:00 +0000")

    app.write_text("def divide(a, b):\n    return a // b\n")
    c2 = commit(repo, "use floor division", "2020-01-02 12:00:00 +0000")

    app.write_text("def divide(a, b):\n    return a / b\n")
    c3 = commit(
        repo,
        "Fix bug: floor division broke negative operands, fixes #12",
        "2021-06-01 12:00:00 +0000",
    )

    (repo / "README.md").write_text("# toy\n")
    c4 = commit(repo, "docs: add readme", "2024-06-01 12:00:00 +0000")

    return {"repo": repo, "c1": c1, "c2": c2, "c3": c3, "c4": c4}
