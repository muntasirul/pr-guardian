"""semgrep baseline: pattern-based static analysis on the added lines.

Serves as the "why not just use a linter?" comparison. Requires the semgrep
binary; skipped gracefully when unavailable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from mining.diff_utils import parse_unified_diff
from training.eval_verdict import compute_metrics
from training.schemas import ReviewComment


def _added_text(diff: str) -> str:
    lines = []
    for hunk in parse_unified_diff(diff):
        lines.extend(hunk.added[ln] for ln in sorted(hunk.added))
    return "\n".join(lines)


def _findings_for(diff_text: str, tmpdir: Path, config: str) -> list[dict]:
    added = _added_text(diff_text)
    if not added.strip():
        return []
    f = tmpdir / "snippet.py"
    f.write_text(added)
    proc = subprocess.run(
        ["semgrep", "--json", "--config", config, str(f)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        return []
    try:
        return json.loads(proc.stdout).get("results", [])
    except json.JSONDecodeError:
        return []


def run(records: list[dict], limit: int | None = None) -> dict:
    if shutil.which("semgrep") is None:
        return {"system": "semgrep", "skipped": "semgrep binary not installed"}
    config = os.getenv("SEMGREP_CONFIG", "auto")
    if limit:
        records = records[:limit]

    parsed: list[ReviewComment | None] = []
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for rec in records:
            diff = _extract_diff(rec)
            findings = _findings_for(diff, tmpdir, config)
            if findings:
                parsed.append(
                    ReviewComment(
                        verdict="BUG",
                        type="security",
                        severity="medium",
                        comment=findings[0].get("extra", {}).get("message", "semgrep finding"),
                        location=rec["meta"].get("path"),
                        confidence=0.7,
                    )
                )
            else:
                parsed.append(ReviewComment(verdict="CLEAN", confidence=0.5))

    y_true = ["BUG" if r["meta"]["kind"] == "bug" else "CLEAN" for r in records]
    y_pred = [p.verdict.value for p in parsed]
    conf = [p.confidence for p in parsed]
    valid = [True] * len(parsed)
    results = compute_metrics(records, y_true, y_pred, conf, valid, parsed, [], None, f"semgrep:{config}")
    results["system"] = "semgrep"
    results["est_cost_per_1k_diffs"] = 0.0
    return results


def _extract_diff(rec: dict) -> str:
    import re

    user = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
    m = re.search(r"```diff\n(.*?)\n```", user, re.DOTALL)
    return m.group(1) if m else user
