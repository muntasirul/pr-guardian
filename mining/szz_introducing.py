"""SZZ-lite: for each fix commit, blame the lines it *removed* (as they were in
the parent revision) to find the commit that introduced them.

This is the classic SZZ heuristic (Sliwerski et al., 2005) in a cheap form:
no issue-tracker validation, no refactor filtering — which is exactly why the
project plan includes a manual label-noise audit on a sampled 100 diffs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mining.diff_utils import parse_unified_diff
from mining.git_utils import blame_lines, commit_date_iso, commit_message, show_file_diff


@dataclass
class IntroducedBug:
    fix_sha: str
    fix_msg: str
    fix_date: str
    path: str
    introducing_sha: str
    introducing_msg: str
    introducing_date: str
    introducing_diff: str
    n_evidence_lines: int


def find_introducing_commits(repo: Path, fix) -> list[IntroducedBug]:
    """For one FixCommit, blame removed lines at the parent revision and map
    them back to the commits that introduced them."""
    out: list[IntroducedBug] = []
    for f in fix.files:
        hunks = parse_unified_diff(f["diff"])
        removed_lines = sorted(ln for h in hunks for ln in h.removed)
        if not removed_lines:
            continue
        # blame the parent version: lines the fix removed lived there
        blame = blame_lines(repo, f"{fix.sha}~1", f["path"])
        if not blame:
            continue  # file added by the fix (or renamed) -> no parent version
        shas = [blame[ln] for ln in removed_lines if ln in blame]
        if not shas:
            continue
        intro_sha, votes = Counter(shas).most_common(1)[0]
        if intro_sha.startswith(fix.sha[:7]):
            continue  # degenerate mapping; skip rather than self-label
        diff = show_file_diff(repo, intro_sha, f["path"])
        if not diff.strip():
            continue
        out.append(
            IntroducedBug(
                fix_sha=fix.sha,
                fix_msg=fix.msg,
                fix_date=fix.date,
                path=f["path"],
                introducing_sha=intro_sha,
                introducing_msg=commit_message(repo, intro_sha),
                introducing_date=commit_date_iso(repo, intro_sha),
                introducing_diff=diff,
                n_evidence_lines=votes,
            )
        )
    return out
