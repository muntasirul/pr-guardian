"""Build train/val/test chat datasets from mined JSONL records.

Leakage guards (the whole point of this module):
- splits are per REPO when there are >= 3 repos (a repo never straddles splits)
- exact duplicate diffs are removed
- near-duplicate test diffs (shingle Jaccard >= 0.8 vs any train diff) are dropped

Usage:
    python -m training.build_dataset --in data/raw/sample.jsonl --task verdict
    python -m training.build_dataset --in data/raw/diffs_v1.jsonl data/raw/devign_records.jsonl --task review
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from training.common import write_jsonl
from training.prompts import review_messages, target_answer, verdict_messages

MAX_DIFF_CHARS = 6000
NEAR_DUP_THRESHOLD = 0.8
_SHINGLE_K = 3


def normalize_diff(diff: str) -> str:
    """Lowercase, strip whitespace, drop hunk headers — identity for dup checks."""
    lines = []
    for ln in diff.lower().splitlines():
        if ln.startswith(("+++", "---", "@@", "diff --git", "index ")):
            continue
        lines.append(ln.strip())
    return "\n".join(lines)


def diff_hash(diff: str) -> str:
    return hashlib.sha1(normalize_diff(diff).encode()).hexdigest()


def shingles(diff: str, k: int = _SHINGLE_K) -> frozenset[int]:
    lines = normalize_diff(diff).splitlines()
    if len(lines) < k:
        return frozenset((hash("\n".join(lines)),))
    return frozenset(
        hash("\n".join(lines[i : i + k])) for i in range(len(lines) - k + 1)
    )


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def split_by_repo(repos: list[str], test_frac: float = 0.1, val_frac: float = 0.1) -> dict[str, str]:
    """Deterministic repo->split assignment (sorted by md5 for stability)."""
    ordered = sorted(repos, key=lambda r: hashlib.md5(r.encode()).hexdigest())
    n = len(ordered)
    n_test = max(1, round(n * test_frac)) if n >= 3 else 0
    n_val = max(1, round(n * val_frac)) if n >= 3 else 0
    out = {}
    for i, repo in enumerate(ordered):
        if i < n_test:
            out[repo] = "test"
        elif i < n_test + n_val:
            out[repo] = "val"
        else:
            out[repo] = "train"
    return out


def build_records(raw_records: list[dict], task: str, max_diff_chars: int = MAX_DIFF_CHARS) -> list[dict]:
    """Map mined records to chat-format records with prompts + targets."""
    builder = verdict_messages if task == "verdict" else review_messages
    out = []
    for rec in raw_records:
        diff = (rec.get("diff") or "")[:max_diff_chars]
        if not diff.strip():
            continue
        messages = builder(diff)
        messages.append({"role": "assistant", "content": target_answer(rec, task)})
        out.append(
            {
                "messages": messages,
                "meta": {
                    "id": rec.get("id"),
                    "repo": rec.get("repo"),
                    "kind": rec.get("kind"),
                    "path": rec.get("path"),
                    "diff_preview": diff[:400],
                },
            }
        )
    return out


def dedup_exact(records: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    out = []
    for rec in records:
        diff = _extract_diff(rec)
        h = diff_hash(diff)
        if h in seen:
            continue
        seen.add(h)
        out.append(rec)
    return out, len(records) - len(out)


def _extract_diff(rec: dict) -> str:
    user = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
    m = re.search(r"```diff\n(.*?)\n```", user, re.DOTALL)
    return m.group(1) if m else user


def decontaminate_test(train: list[dict], test: list[dict]) -> tuple[list[dict], int]:
    """Drop test records that are near-duplicates of train records."""
    index: dict[int, list[int]] = defaultdict(list)
    train_shingles = [shingles(_extract_diff(r)) for r in train]
    for ti, sh in enumerate(train_shingles):
        for s in sh:
            index[s].append(ti)

    kept: list[dict] = []
    dropped = 0
    for rec in test:
        sh = shingles(_extract_diff(rec))
        cands: set[int] = set()
        for s in sh:
            for ti in index.get(s, []):
                cands.add(ti)
        is_dup = any(jaccard(sh, train_shingles[ti]) >= NEAR_DUP_THRESHOLD for ti in cands)
        if is_dup:
            dropped += 1
        else:
            kept.append(rec)
    return kept, dropped


def build_dataset(raw_records: list[dict], task: str, seed: int = 7) -> dict:
    records = build_records(raw_records, task)
    records, n_exact_dup = dedup_exact(records)

    repos = sorted({r["meta"]["repo"] for r in records})
    if len(repos) >= 3:
        assignment = split_by_repo(repos)
        splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for rec in records:
            splits[assignment[rec["meta"]["repo"]]].append(rec)
    else:
        # single/few repos: record-level split + near-dup decontamination is
        # the honest fallback (warning recorded in stats)
        rng = random.Random(seed)
        rng.shuffle(records)
        n = len(records)
        n_test = max(1, round(n * 0.1))
        n_val = max(1, round(n * 0.1))
        splits = {
            "test": records[:n_test],
            "val": records[n_test : n_test + n_val],
            "train": records[n_test + n_val :],
        }

    test_before = len(splits["test"])
    splits["test"], n_near_dup = decontaminate_test(splits["train"], splits["test"])

    stats = {
        "task": task,
        "total": sum(len(v) for v in splits.values()),
        "splits": {k: len(v) for k, v in splits.items()},
        "repos": len(repos),
        "split_mode": "repo" if len(repos) >= 3 else "record (few repos)",
        "labels": {
            k: {
                "bug": sum(1 for r in v if r["meta"]["kind"] == "bug"),
                "clean": sum(1 for r in v if r["meta"]["kind"] == "clean"),
            }
            for k, v in splits.items()
        },
        "dropped_exact_duplicates": n_exact_dup,
        "dropped_test_near_duplicates": n_near_dup,
        "test_size_before_decontamination": test_before,
    }
    return {"splits": splits, "stats": stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--task", choices=["verdict", "review"], default="verdict")
    ap.add_argument("--out-dir", default=None, help="default data/processed/<task>")
    ap.add_argument("--max-diff-chars", type=int, default=MAX_DIFF_CHARS)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    raw: list[dict] = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw.append(json.loads(line))

    out_dir = Path(args.out_dir) if args.out_dir else Path("data/processed") / args.task
    result = build_dataset(raw, args.task, seed=args.seed)
    for name, recs in result["splits"].items():
        write_jsonl(recs, out_dir / f"{name}.jsonl")
    with (out_dir / "stats.json").open("w") as f:
        json.dump(result["stats"], f, indent=2)

    s = result["stats"]
    print(f"task={args.task} splits={s['splits']} split_mode={s['split_mode']}")
    print(f"labels={s['labels']}")
    print(f"dropped: exact_dup={s['dropped_exact_duplicates']} test_near_dup={s['dropped_test_near_duplicates']}")
    print(f"-> {out_dir}/{{train,val,test}}.jsonl + stats.json")


if __name__ == "__main__":
    main()
