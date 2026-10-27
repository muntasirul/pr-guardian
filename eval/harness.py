"""Run all systems over the held-out test split and aggregate the results.

    python -m eval.harness --test data/processed/verdict/test.jsonl --systems mock api ollama semgrep
    python -m eval.harness --systems finetuned --adapter models/verdict-v1

Produces eval/results.json + eval/results.md (the headline cost/quality table).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval import report as report_mod
from training.common import load_jsonl


def run_mock(records, limit, adapter=None, test=None):
    from backend.app.inference import MockInference
    from training.eval_verdict import compute_metrics

    inf = MockInference()
    parsed, lat = [], []
    for rec in records[:limit]:
        diff = _extract_diff(rec)
        t0 = time.perf_counter()
        parsed.append(inf.review(diff, rec["meta"].get("path") or "", None))
        lat.append((time.perf_counter() - t0) * 1000)
    y_true = ["BUG" if r["meta"]["kind"] == "bug" else "CLEAN" for r in records[:limit]]
    y_pred = [p.verdict.value for p in parsed]
    results = compute_metrics(
        records[:limit], y_true, y_pred, [p.confidence for p in parsed], [True] * len(parsed),
        parsed, [], None, "mock:rules",
    )
    results["system"] = "mock_rules"
    results["avg_latency_ms"] = round(sum(lat) / max(1, len(lat)))
    results["est_cost_per_1k_diffs"] = 0.0
    return results


def _extract_diff(rec: dict) -> str:
    import re

    user = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
    m = re.search(r"```diff\n(.*?)\n```", user, re.DOTALL)
    return m.group(1) if m else user


def run_finetuned(records, limit, adapter, test):
    from training.eval_verdict import evaluate

    if not adapter:
        return {"system": "finetuned", "skipped": "no --adapter provided"}
    return evaluate(
        test_path=Path(test),
        adapter=adapter,
        model_name=None,
        limit=limit,
        max_new_tokens=96,
        batch_size=4,
    )


RUNNERS = {
    "mock": lambda records, limit, adapter, test: run_mock(records, limit),
    "api": lambda records, limit, adapter, test: __import__(
        "eval.baselines.api_zero_shot", fromlist=["run"]
    ).run(records, limit),
    "ollama": lambda records, limit, adapter, test: __import__(
        "eval.baselines.local_zero_shot", fromlist=["run"]
    ).run(records, limit),
    "semgrep": lambda records, limit, adapter, test: __import__(
        "eval.baselines.semgrep_runner", fromlist=["run"]
    ).run(records, limit),
    "finetuned": lambda records, limit, adapter, test: run_finetuned(records, limit, adapter, test),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", default="data/processed/verdict/test.jsonl")
    ap.add_argument("--systems", nargs="*", default=["mock", "finetuned"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out-dir", default="eval")
    args = ap.parse_args()

    records = load_jsonl(args.test)
    print(f"test records: {len(records)} (limit {args.limit})")

    results = []
    for system in args.systems:
        runner = RUNNERS.get(system)
        if runner is None:
            print(f"[skip] unknown system: {system}")
            continue
        print(f"[run] {system} ...")
        try:
            results.append(runner(records, args.limit, args.adapter, args.test))
        except Exception as e:  # noqa: BLE001 - one system failing must not kill the harness
            results.append({"system": system, "skipped": f"error: {str(e)[:200]}"})
        r = results[-1]
        print(f"      {'skipped: ' + r['skipped'] if r.get('skipped') else r.get('accuracy')}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w") as f:
        json.dump({"test": args.test, "limit": args.limit, "systems": results}, f, indent=2)
    md = report_mod.build_markdown(results)
    (out_dir / "results.md").write_text(md)
    print(f"-> {out_dir}/results.json\n-> {out_dir}/results.md")


if __name__ == "__main__":
    main()
