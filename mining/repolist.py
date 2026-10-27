"""Curated list of Python repos for bug-fix mining.

Medium-history repos by default so clones stay quick and blame stays cheap.
Big frameworks (django, pandas, ...) are opt-in via --include-big.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_REPOS = [
    "psf/requests",
    "pallets/flask",
    "pallets/werkzeug",
    "pallets/click",
    "pallets/jinja",
    "pallets/itsdangerous",
    "pallets/markupsafe",
    "encode/httpx",
    "encode/starlette",
    "tiangolo/fastapi",
    "benoitc/gunicorn",
    "urllib3/urllib3",
    "certifi/python-certifi",
    "python-attrs/attrs",
    "jd/tenacity",
    "marshmallow-code/marshmallow",
    "getsentry/responses",
    "cookiecutter/cookiecutter",
]

BIG_REPOS = [
    "django/django",
    "pandas-dev/pandas",
    "scikit-learn/scikit-learn",
    "pytest-dev/pytest",
    "sqlalchemy/sqlalchemy",
    "scipy/scipy",
    "celery/celery",
    "twisted/twisted",
]


def full_list(include_big: bool = False) -> list[str]:
    return DEFAULT_REPOS + (BIG_REPOS if include_big else [])


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the mining repo list to data/repolist.csv")
    ap.add_argument("--out", default="data/repolist.csv")
    ap.add_argument("--include-big", action="store_true", help="include large frameworks (slow clones)")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["repo", "url", "size_class"])
        for slug in full_list(args.include_big):
            w.writerow([slug, f"https://github.com/{slug}", "big" if slug in BIG_REPOS else "medium"])
    print(f"wrote {out} with {len(full_list(args.include_big))} repos")


if __name__ == "__main__":
    main()
