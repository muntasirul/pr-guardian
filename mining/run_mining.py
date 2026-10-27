"""End-to-end mining orchestrator.

Clone repos -> mine fix commits -> SZZ-lite introducing diffs -> survival clean
diffs -> one JSONL of training records + a stats summary.

Usage:
    python -m mining.run_mining --repos psf/requests --out data/raw/sample.jsonl
    python -m mining.run_mining --repos data/repolist.csv --out data/raw/diffs_v1.jsonl \
        --max-fixes 200 --max-clean 150 --append
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mining.git_utils import clone, repo_url
from mining.mine_clean_diffs import mine_clean_diffs
from mining.mine_fix_commits import mine_fix_commits
from mining.szz_introducing import find_introducing_commits


def _rec_id(repo: str, sha: str, path: str) -> str:
    h = hashlib.sha1(f"{repo}::{sha}::{path}".encode()).hexdigest()[:12]
    slug = repo.replace("/", "__")
    return f"{slug}__{sha[:8]}__{h}"


def record_from_bug(repo: str, bug) -> dict:
    return {
        "id": _rec_id(repo, bug.introducing_sha, bug.path),
        "repo": repo,
        "kind": "bug",
        "source": "szz",
        "path": bug.path,
        "lang": "Python",
        "diff": bug.introducing_diff,
        "commit_msg": bug.introducing_msg,
        "commit_date": bug.introducing_date,
        "fix_sha": bug.fix_sha,
        "fix_msg": bug.fix_msg,
        "n_evidence_lines": bug.n_evidence_lines,
    }


def record_from_clean(repo: str, clean) -> dict:
    return {
        "id": _rec_id(repo, clean.sha, clean.path),
        "repo": repo,
        "kind": "clean",
        "source": "survival",
        "path": clean.path,
        "lang": "Python",
        "diff": clean.diff,
        "commit_msg": clean.msg,
        "commit_date": clean.date,
        "fix_sha": None,
        "fix_msg": None,
        "n_evidence_lines": None,
    }


def load_repos_arg(repos: list[str]) -> list[str]:
    out: list[str] = []
    for r in repos:
        if r.endswith(".csv"):
            import csv as _csv

            with open(r) as f:
                out.extend(row["repo"] for row in _csv.DictReader(f))
        else:
            out.append(r)
    return out


def mine_repo(
    slug: str, repos_dir: Path, max_fixes: int, max_clean: int, max_files_per_fix: int
) -> tuple[list[dict], dict]:
    local = repos_dir / slug.replace("/", "__")
    if not (local / ".git").exists():
        cloned = clone(repo_url(slug), local)
        if not (local / ".git").exists():
            return [], {"repo": slug, "error": "clone failed", "cloned": cloned}

    fixes = mine_fix_commits(local, max_fixes=max_fixes, max_files_per_fix=max_files_per_fix)
    positives: list[dict] = []
    contaminated: set[str] = set()
    for fix in fixes:
        for f in fix.files:
            contaminated.add(f["path"])
        for bug in find_introducing_commits(local, fix):
            positives.append(record_from_bug(slug, bug))
            contaminated.add(bug.path)

    cleans = mine_clean_diffs(local, contaminated, max_clean=max_clean)
    records = positives + [record_from_clean(slug, c) for c in cleans]

    stats = {
        "repo": slug,
        "fixes_seen": len(fixes),
        "positives": len(positives),
        "cleans": len(cleans),
        "contaminated_paths": len(contaminated),
    }
    return records, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine bug-introducing and clean diffs")
    ap.add_argument("--repos", nargs="+", required=True, help="slugs like psf/requests, or a CSV path")
    ap.add_argument("--repos-dir", default="data/repos")
    ap.add_argument("--out", default="data/raw/diffs_v1.jsonl")
    ap.add_argument("--max-fixes", type=int, default=200)
    ap.add_argument("--max-clean", type=int, default=100)
    ap.add_argument("--max-files-per-fix", type=int, default=3)
    ap.add_argument("--append", action="store_true", help="skip repos already present in --out")
    args = ap.parse_args()

    repos_dir = Path(args.repos_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.append and out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["repo"])
                except (json.JSONDecodeError, KeyError):
                    continue

    all_stats: list[dict] = []
    total = 0
    with out_path.open("a") as out:
        for slug in load_repos_arg(args.repos):
            if slug in done:
                print(f"[skip] {slug} (already in {out_path})")
                continue
            print(f"[mine] {slug} ...")
            try:
                records, stats = mine_repo(
                    slug, repos_dir, args.max_fixes, args.max_clean, args.max_files_per_fix
                )
            except Exception as e:  # noqa: BLE001 - keep the batch running
                print(f"[error] {slug}: {e}")
                stats = {"repo": slug, "error": str(e)[:200]}
                records = []
            for rec in records:
                out.write(json.dumps(rec) + "\n")
            total += len(records)
            all_stats.append(stats)
            print(f"        {stats}")

    stats_path = out_path.with_suffix(".stats.json")
    with stats_path.open("w") as f:
        json.dump({"total_records": total, "repos": all_stats}, f, indent=2)
    print(f"\nwrote {total} records -> {out_path}")
    print(f"per-repo stats -> {stats_path}")


if __name__ == "__main__":
    main()
