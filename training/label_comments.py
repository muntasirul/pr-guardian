"""One-time offline pass: turn (fix_msg + fix diff) into a natural reviewer-style
comment that becomes the generative training target.

Uses any OpenAI-compatible API when a key is configured; falls back to a
cleaned fix message so the pipeline works with zero API spend.

Usage:
    PRGUARDIAN_API_KEY=sk-... python -m training.label_comments \
        --in data/raw/positives.jsonl --out data/raw/positives_labeled.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

from training.prompts import _clean_fix_msg

MAX_FIX_DIFF_CHARS = 4000

SYSTEM = (
    "You write code-review comments. You are given (a) the commit message of a "
    "bug fix and (b) the diff that fixed the bug. Write the review comment that "
    "a reviewer could have left on the ORIGINAL bug-introducing change. "
    "1-3 sentences, plain technical English, no preamble, no quotes. State the "
    "concrete failure and, when possible, the input that triggers it."
)


def _api_creds() -> tuple[str | None, str, str]:
    key = os.getenv("PRGUARDIAN_API_KEY") or os.getenv("OPENAI_API_KEY")
    base = os.getenv("PRGUARDIAN_API_BASE") or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
    model = os.getenv("PRGUARDIAN_API_MODEL") or "gpt-4o-mini"
    return key, base.rstrip("/"), model


def _call_api(msg: str, fix_diff: str, base: str, key: str, model: str, retries: int = 2) -> str:
    payload = {
        "model": model,
        "temperature": 0.3,
        "max_tokens": 160,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Fix commit message:\n{msg}\n\nFix diff:\n{fix_diff[:MAX_FIX_DIFF_CHARS]}",
            },
        ],
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:  # noqa: BLE001
            if attempt == retries:
                return ""
            time.sleep(2 * (attempt + 1))
    return ""


def _usable(comment: str) -> bool:
    if not comment or len(comment) < 20:
        return False
    if re.search(r"^(as an ai|i cannot|sorry)", comment.lower()):
        return False
    return True


def label_records(records: list[dict], use_api: bool = True, limit: int | None = None) -> list[dict]:
    key, base, model = _api_creds()
    if use_api and not key:
        use_api = False
        print("[label_comments] no API key configured -> using fix-message fallback")
    if use_api:
        print(f"[label_comments] API: {model} @ {base}")

    out = []
    n_api = n_fallback = 0
    bugs = [r for r in records if r.get("kind") == "bug"]
    if limit:
        bugs = bugs[:limit]
    for rec in records:
        if rec.get("kind") != "bug" or (limit is not None and rec.get("review_comment")):
            out.append(rec)
            continue
        if rec in bugs or limit is None:
            comment = ""
            if use_api:
                comment = _call_api(rec.get("fix_msg") or "", rec.get("diff") or "", base, key, model)
                if not _usable(comment):
                    comment = ""
            if comment:
                n_api += 1
            else:
                comment = _clean_fix_msg(rec.get("fix_msg"))
                n_fallback += 1
            rec = {**rec, "review_comment": comment}
        out.append(rec)
    print(f"[label_comments] api={n_api} fallback={n_fallback}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-api", action="store_true", help="force fix-message fallback")
    ap.add_argument("--limit", type=int, default=None, help="label at most N bug records")
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    labeled = label_records(records, use_api=not args.no_api, limit=args.limit)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in labeled:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(labeled)} records -> {out}")


if __name__ == "__main__":
    main()
