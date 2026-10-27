"""Detect likely bug-fix commits in a local git repo and extract their diffs.

Heuristic (standard in defect-prediction literature): a commit is a "fix" if
its message matches fix/bug/resolve/close patterns, references an issue, or
uses bug vocabulary. Merge commits and fixup!/squash! leftovers are excluded.

The score threshold (>= 2) trades recall for precision — for training data we
would rather miss fixes than label feature commits as fixes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from mining.diff_utils import diff_stats

FIX_START = re.compile(
    r"^(fix|fixes|fixed|bugfix|bug|hotfix|patch|repair|resolve|resolves|resolved|close|closes|closed)\b",
    re.IGNORECASE,
)
ISSUE_REF = re.compile(r"\b(closes?|fixes?|resolves?|refs?)\s*(issue\s*)?#\d+", re.IGNORECASE)
BUG_WORDS = re.compile(
    r"\b(bug|regression|crash|error|leak|failure|incorrect|wrong|broken|npe|null)\b",
    re.IGNORECASE,
)
EXCLUDE = re.compile(r"^(fixup!|squash!|revert|amend)", re.IGNORECASE)

PY_EXT = ".py"
MAX_DIFF_LINES = 400
MIN_DIFF_LINES = 1  # one-line bug fixes are the classic case; don't filter them out
MAX_FILE_BYTES = 200_000


@dataclass
class FixCommit:
    sha: str
    msg: str
    date: str
    files: list[dict] = field(default_factory=list)  # {"path","diff","added","removed"}


def fix_score(msg: str) -> int:
    """Score a commit message; >= 2 counts as a fix commit."""
    if not msg or EXCLUDE.match(msg):
        return 0
    score = 0
    if FIX_START.match(msg):
        score += 3
    if ISSUE_REF.search(msg):
        score += 2
    if BUG_WORDS.search(msg):
        score += 1
    return score


def _load_repository_class():
    try:
        from pydriller import Repository  # pydriller >= 2.0
    except ImportError:
        from pydriller import RepositoryMining as Repository  # pydriller 1.x
    return Repository


def mine_fix_commits(
    repo_path: Path,
    max_fixes: int = 200,
    max_files_per_fix: int = 3,
) -> list[FixCommit]:
    """Return likely bug-fix commits (capped) with their Python file diffs."""
    Repository = _load_repository_class()
    fixes: list[FixCommit] = []
    for commit in Repository(str(repo_path)).traverse_commits():
        if getattr(commit, "merge", False):
            continue
        if len(fixes) >= max_fixes:
            break
        if fix_score(commit.msg or "") < 2:
            continue
        files: list[dict] = []
        # pydriller renamed Commit.modifications -> Commit.modified_files (2.x)
        mods = getattr(commit, "modified_files", None)
        if mods is None:
            mods = getattr(commit, "modifications", [])
        for mod in mods:
            path = getattr(mod, "new_path", None) or getattr(mod, "old_path", None)
            diff = getattr(mod, "diff", None) or ""
            if not path or not path.endswith(PY_EXT):
                continue
            if not diff or len(diff.encode("utf-8", "ignore")) > MAX_FILE_BYTES:
                continue
            stats = diff_stats(diff)
            if not (MIN_DIFF_LINES <= stats["changed"] <= MAX_DIFF_LINES):
                continue
            files.append({"path": path, "diff": diff, **stats})
            if len(files) >= max_files_per_fix:
                break
        if files:
            fixes.append(
                FixCommit(
                    sha=commit.hash,
                    msg=(commit.msg or "").strip(),
                    date=commit.author_date.isoformat(),
                    files=files,
                )
            )
    return fixes
