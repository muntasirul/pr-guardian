"""Thin subprocess wrappers around git + blame parsing."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def run_git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()[:300]}")
    return proc.stdout


def parse_blame_porcelain(out: str) -> dict[int, str]:
    """Map final line number -> commit sha from ``git blame --porcelain`` output.

    Porcelain format: each group starts with a header line
    ``<sha> <orig_lineno> <final_lineno> <group_size>``; lines blamed to the
    same commit as the previous group omit the header. Content lines are
    tab-prefixed and correspond to consecutive final line numbers 1..N, so we
    can count them instead of trusting group sizes.
    """
    mapping: dict[int, str] = {}
    cur_sha: str | None = None
    n = 0
    for line in out.splitlines():
        if line.startswith("\t"):
            n += 1
            if cur_sha is not None:
                mapping[n] = cur_sha
            continue
        parts = line.split(" ")
        if (
            len(parts) >= 4
            and _SHA_RE.match(parts[0])
            and parts[1].isdigit()
            and parts[2].isdigit()
        ):
            cur_sha = parts[0]
    return mapping


def blame_lines(repo: Path, rev: str, path: str) -> dict[int, str]:
    """Blame the version of ``path`` as it existed at ``rev``."""
    out = run_git(repo, "blame", "--porcelain", rev, "--", path, check=False)
    return parse_blame_porcelain(out)


def show_file_diff(repo: Path, rev: str, path: str) -> str:
    """Unified diff of ``rev`` for ``path`` only (empty string if none)."""
    return run_git(repo, "show", rev, "--format=", "--unified=3", "--", path, check=False)


def commit_message(repo: Path, rev: str) -> str:
    return run_git(repo, "log", "-1", "--format=%s", rev, check=False).strip()


def commit_date_iso(repo: Path, rev: str) -> str:
    return run_git(repo, "log", "-1", "--format=%aI", rev, check=False).strip()


def clone(url: str, dest: Path) -> bool:
    """Clone ``url`` into ``dest`` if not already present. Returns True if newly cloned."""
    if (dest / ".git").exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", url, str(dest)],
        check=False,
        capture_output=True,
    )
    return (dest / ".git").exists()


def repo_url(slug: str) -> str:
    return f"https://github.com/{slug}.git"
