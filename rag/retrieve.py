"""Retrieve repo context for a changed file: embed the diff's added lines,
search the chunk index, and assemble a compact context block for the prompt.
"""

from __future__ import annotations

import re
from pathlib import Path

from mining.diff_utils import parse_unified_diff
from rag.indexer import LocalIndex

_CODE_BLOCK = re.compile(r"^@@|^(diff --git|index )", re.MULTILINE)


def query_text_from_diff(diff_text: str) -> str:
    added = []
    for hunk in parse_unified_diff(diff_text):
        for ln in sorted(hunk.added):
            added.append(hunk.added[ln])
    return "\n".join(added)[:2000] or diff_text[:2000]


def retrieve_context(
    index_path: str | Path,
    target_path: str,
    diff_text: str,
    k: int = 3,
) -> str | None:
    """Return top-k related chunks (excluding the changed file itself), or None."""
    index_path = Path(index_path)
    if not index_path.with_suffix(".npz").exists():
        return None
    index = LocalIndex(index_path)
    query = query_text_from_diff(diff_text)
    hits = index.search(query, k=k + 2, exclude_path=target_path)
    if not hits:
        return None
    # prioritize tests and same-package chunks
    def boost(m: dict) -> float:
        s = 0.0
        if "test" in m["path"].lower():
            s += 0.05
        if Path(m["path"]).parent == Path(target_path).parent:
            s += 0.05
        return -s

    hits = sorted(hits, key=lambda h: (-h[1] + boost(h[0]),))[:k]
    blocks = [f"# {h['path']} ({h['kind']} {h['name']})\n{h['code']}" for h, _ in hits]
    return "\n\n---\n\n".join(blocks)[:4000]
