"""Survival-heuristic negative sampling: diffs to code that stayed untouched for
a long window afterwards. A change to a file nobody had to touch again for
12+ months is a cheap-but-defensible "probably clean" example.

Caveat (documented, not hidden): surviving code can still contain latent bugs;
the audit notebook quantifies how often that happens on a sample.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mining.git_utils import run_git, show_file_diff

MIN_SURVIVAL_DAYS = 365
MAX_DIFF_LINES = 400
MIN_DIFF_LINES = 3


@dataclass
class CleanDiff:
    sha: str
    msg: str
    date: str
    path: str
    diff: str


def _parse_iso(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except ValueError:
        # old repos can carry malformed git dates (e.g. offset '+518:00' in
        # psf/requests, 2011); day-granularity survival math doesn't need the
        # timezone, so fall back to the naive timestamp as UTC
        return datetime.fromisoformat(s[:19]).replace(tzinfo=timezone.utc)


def _commit_log(repo: Path) -> list[dict]:
    """Parse `git log --format=... --name-only` into events (newest first).

    Actual output shape (one blank line AFTER each header, before the files):
        <header>\n\n<file1>\n<file2>\n<header2>\n\n<file1>\n...
    """
    out = run_git(repo, "log", "--format=%H%x1f%aI%x1f%s", "--name-only", check=False)
    events: list[dict] = []
    cur: dict | None = None
    skip_blank = False
    for line in out.splitlines():
        if "\x1f" in line:
            sha, date, msg = line.split("\x1f", 2)
            cur = {"sha": sha, "date": date, "msg": msg, "paths": []}
            events.append(cur)
            skip_blank = True
            continue
        if skip_blank:
            # the single blank separator right after the header
            skip_blank = False
            if line.strip():
                cur["paths"].append(line)  # no separator present; tolerate
            continue
        if cur is not None and line.strip():
            cur["paths"].append(line)
    return events


def mine_clean_diffs(
    repo: Path,
    contaminated_paths: set[str],
    max_clean: int = 100,
    min_survival_days: int = MIN_SURVIVAL_DAYS,
    seed: int = 7,
) -> list[CleanDiff]:
    """Sample diffs that survived >= min_survival_days untouched."""
    events = _commit_log(repo)
    if not events:
        return []
    now = _parse_iso(events[0]["date"])

    last_touch: dict[str, str] = {}
    for ev in events:  # newest first
        for p in ev["paths"]:
            last_touch.setdefault(p, ev["date"])

    candidates: list[tuple[dict, str]] = []
    for ev in events:
        ev_date = _parse_iso(ev["date"])
        if (now - ev_date).days < min_survival_days:
            continue  # too recent to judge survival
        for p in ev["paths"]:
            if not p.endswith(".py") or p in contaminated_paths:
                continue
            lt = _parse_iso(last_touch[p])
            # file never touched again after this event? it survived until repo HEAD
            end = now if lt <= ev_date else lt
            survived = (end - ev_date).days
            if survived >= min_survival_days:
                candidates.append((ev, p))

    rng = random.Random(seed)
    rng.shuffle(candidates)

    cleans: list[CleanDiff] = []
    for ev, p in candidates:
        if len(cleans) >= max_clean:
            break
        diff = show_file_diff(repo, ev["sha"], p)
        if not diff.strip():
            continue
        n_lines = diff.count("\n") + 1
        if not (MIN_DIFF_LINES <= n_lines <= MAX_DIFF_LINES):
            continue
        cleans.append(CleanDiff(sha=ev["sha"], msg=ev["msg"], date=ev["date"], path=p, diff=diff))
    return cleans
