"""Unified-diff parsing helpers (zero dependencies).

We parse diffs ourselves instead of relying on library internals so behavior
is stable across PyDriller versions and testable in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Hunk:
    old_start: int
    new_start: int
    removed: dict[int, str] = field(default_factory=dict)  # old-file line no -> text
    added: dict[int, str] = field(default_factory=dict)  # new-file line no -> text


def parse_unified_diff(diff_text: str) -> list[Hunk]:
    """Parse a unified diff into hunks with old/new line numbers.

    File headers (``diff --git``, ``---``, ``+++``) are only valid *before* the
    first hunk; inside a hunk any '-'/'+-prefixed line is content, which avoids
    the classic ``--- a/...`` header-vs-content ambiguity.
    """
    hunks: list[Hunk] = []
    cur: Hunk | None = None
    old_ln = 0
    new_ln = 0
    for line in diff_text.splitlines():
        m = _HUNK_RE.match(line)
        if m:
            cur = Hunk(old_start=int(m.group(1)), new_start=int(m.group(3)))
            hunks.append(cur)
            old_ln, new_ln = cur.old_start, cur.new_start
            continue
        if cur is None:
            continue  # pre-hunk header lines
        if line.startswith("-"):
            cur.removed[old_ln] = line[1:]
            old_ln += 1
        elif line.startswith("+"):
            cur.added[new_ln] = line[1:]
            new_ln += 1
        elif not line.startswith("\\"):
            # context line; "\ No newline at end of file" is not one
            old_ln += 1
            new_ln += 1
    return hunks


def diff_stats(diff_text: str) -> dict[str, int]:
    hunks = parse_unified_diff(diff_text)
    removed = sum(len(h.removed) for h in hunks)
    added = sum(len(h.added) for h in hunks)
    return {"added": added, "removed": removed, "changed": added + removed}


def split_multi_file_diff(diff_text: str) -> list[tuple[str, str]]:
    """Split a pasted/multi-file unified diff into (path, single-file diff).

    Handles diffs produced by ``git diff`` (``diff --git a/x b/x`` headers) and
    by the GitHub API (``diff --git ... --- a/x +++ b/x`` blocks).
    """
    if not diff_text.strip():
        return []
    blocks: list[list[str]] = []
    cur: list[str] | None = None
    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            if cur:
                blocks.append(cur)
            cur = [line]
        elif cur is not None:
            cur.append(line)
    if cur:
        blocks.append(cur)
    if not blocks:  # pasted single-file diff without git headers
        return [("unknown", diff_text)]

    out: list[tuple[str, str]] = []
    for block in blocks:
        path = ""
        for line in block:
            if line.startswith("+++ b/"):
                path = line[len("+++ b/"):].strip()
                break
            if line.startswith("diff --git") and " b/" in line:
                path = line.split(" b/", 1)[1].strip()
        if not path:
            path = "unknown"
        out.append((path, "\n".join(block)))
    return out
