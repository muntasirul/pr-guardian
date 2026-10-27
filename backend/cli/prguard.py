"""PR Guardian CLI.

    python -m backend.cli.prguard review --diff changes.diff
    python -m backend.cli.prguard review --pr https://github.com/o/r/pull/12
    python -m backend.cli.prguard index --repo-path ../requests --name requests
    python -m backend.cli.prguard mine --repos psf/requests --out data/raw/sample.jsonl
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="PR Guardian — review diffs like a reviewer, not a linter.")

try:  # optional .env support
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from backend.app.github_client import fetch_pr_files, parse_pr_url
from backend.app.pipeline import review_files
from mining.diff_utils import split_multi_file_diff


def _print_result(result: dict) -> None:
    typer.echo(f"verdict:      {result['verdict']}  ({result['model_mode']} model, {result['latency_ms']} ms)")
    typer.echo(f"files:        {result['n_files']}")
    if result["comments"]:
        for c in result["comments"]:
            loc = c.get("location")
            conf = c.get("confidence")
            typer.echo(f"- [{c['severity']}] {c['type']}: {c.get('comment')}  ({loc}, conf={conf})")
    else:
        typer.echo("no comments (clean)")


def _load_files(diff: Path | None, pr: str | None) -> list[dict]:
    if pr:
        parsed = parse_pr_url(pr)
        if not parsed:
            raise typer.BadParameter("PR URL must look like https://github.com/owner/repo/pull/N")
        if not os.getenv("GITHUB_TOKEN"):
            raise typer.BadParameter("GITHUB_TOKEN is not configured — use --diff instead")
        owner, repo, number = parsed
        return fetch_pr_files(owner, repo, number)
    if diff:
        return [{"path": p, "diff": d} for p, d in split_multi_file_diff(diff.read_text())]
    raise typer.BadParameter("provide --diff FILE or --pr URL")


@app.command()
def review(
    diff: Path = typer.Option(None, help="unified diff file"),
    pr: str = typer.Option(None, help="GitHub PR URL (needs GITHUB_TOKEN)"),
    repo_index: str = typer.Option(None, help="RAG index path for repo context"),
    json_out: bool = typer.Option(False, "--json", help="print raw JSON result"),
):
    """Review a diff file or a GitHub PR and print verdict + comments."""
    files = _load_files(diff, pr)
    result = review_files(files, repo_index=repo_index)
    if json_out:
        typer.echo(json.dumps(result, indent=2))
    else:
        _print_result(result)
    raise typer.Exit(1 if result["verdict"] == "BUG" else 0)


@app.command()
def index(
    repo_path: Path = typer.Option(...),
    name: str = typer.Option(...),
    max_files: int = typer.Option(None),
):
    """Build a RAG index for a local repo clone."""
    from rag.indexer import build_index

    stats = build_index(repo_path, Path("data/index") / name, max_files=max_files)
    typer.echo(f"indexed {stats['chunks']} chunks -> data/index/{name}.*")


def _run_module(module: str, extra: list[str]) -> None:
    proc = subprocess.run([sys.executable, "-m", module, *extra], check=False)
    raise typer.Exit(proc.returncode)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def mine(ctx: typer.Context):
    """Passthrough to mining.run_mining (see its --help)."""
    _run_module("mining.run_mining", ctx.args)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def data(ctx: typer.Context):
    """Passthrough to training.build_dataset."""
    _run_module("training.build_dataset", ctx.args)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def fetch_data(ctx: typer.Context):
    """Passthrough to training.fetch_public_datasets."""
    _run_module("training.fetch_public_datasets", ctx.args)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def evalmodel(ctx: typer.Context, name: str = typer.Option("evalmodel")):
    """Passthrough to training.eval_verdict."""
    _run_module("training.eval_verdict", ctx.args)


if __name__ == "__main__":
    app()
