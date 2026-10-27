"""Evaluate a fine-tuned (or base) model on the held-out test split.

    python -m training.eval_verdict --adapter models/verdict-v1 --test data/processed/verdict/test.jsonl

Writes a results JSON with accuracy, macro-F1, PR-AUC, valid-output rate, and
per-bug-type recall, plus a small sample of mismatches for error analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.common import build_model_and_tokenizer, generate_texts, load_jsonl
from training.schemas import parse_model_output


def evaluate(
    test_path: Path,
    adapter: str | None,
    model_name: str | None,
    limit: int,
    max_new_tokens: int,
    batch_size: int,
) -> dict:
    records = load_jsonl(test_path)
    if limit:
        records = records[:limit]
    if not records:
        raise SystemExit("no test records found")

    from training.common import DEFAULT_BASE_MODEL

    model, tokenizer = build_model_and_tokenizer(
        model_name or DEFAULT_BASE_MODEL, adapter=adapter, quantize="auto"
    )

    prompts = [r["messages"][:-1] for r in records]
    outputs = generate_texts(model, tokenizer, prompts, max_new_tokens, batch_size)

    y_true, y_pred, confidences, valid, parsed_all = [], [], [], [], []
    mismatches = []
    for rec, out in zip(records, outputs):
        parsed = parse_model_output(out)
        parsed_all.append(parsed)
        true_verdict = "BUG" if rec["meta"]["kind"] == "bug" else "CLEAN"
        y_true.append(true_verdict)
        valid.append(parsed is not None)
        if parsed is None:
            y_pred.append("INVALID")
            confidences.append(None)
        else:
            y_pred.append(parsed.verdict.value)
            confidences.append(parsed.confidence)
        if (parsed is None or parsed.verdict.value != true_verdict) and len(mismatches) < 20:
            mismatches.append(
                {
                    "id": rec["meta"]["id"],
                    "true": true_verdict,
                    "pred": y_pred[-1],
                    "model_output": out[:300],
                    "diff_preview": rec["meta"]["diff_preview"],
                }
            )

    return compute_metrics(
        records, y_true, y_pred, confidences, valid, parsed_all, mismatches, adapter, model_name
    )


def compute_metrics(records, y_true, y_pred, confidences, valid, parsed, mismatches, adapter, model_name) -> dict:
    n = len(y_true)
    n_valid = sum(valid)
    correct = sum(t == p for t, p in zip(y_true, y_pred))

    results = {
        "system": "finetuned" if adapter else "base",
        "adapter": adapter,
        "model": model_name,
        "n": n,
        "valid_output_rate": round(n_valid / n, 4),
        "accuracy": round(correct / n, 4) if n else None,
    }

    labels = sorted(set(y_true + y_pred) - {"INVALID"})
    try:
        from sklearn.metrics import f1_score

        results["macro_f1"] = round(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4
        )
    except ValueError:
        results["macro_f1"] = None

    try:
        from sklearn.metrics import average_precision_score

        if len(set(y_true)) == 2 and all(c is not None for c in confidences):
            y_bin = [1 if t == "BUG" else 0 for t in y_true]
            results["pr_auc"] = round(average_precision_score(y_bin, confidences), 4)
    except Exception:  # noqa: BLE001
        results["pr_auc"] = None

    # per-bug-type recall (populated for the review task, where targets carry a type)
    true_types: list[str | None] = []
    for rec in records:
        try:
            tgt = json.loads(rec["messages"][-1]["content"])
            true_types.append(tgt.get("type"))
        except Exception:  # noqa: BLE001
            true_types.append(None)
    type_recall: dict = {}
    for t in sorted({x for x in true_types if x}):
        idx = [i for i, x in enumerate(true_types) if x == t]
        hits = sum(
            1
            for i in idx
            if parsed[i] and parsed[i].verdict.value == "BUG"
            and parsed[i].type and parsed[i].type.value == t
        )
        type_recall[t] = {"n": len(idx), "recall_as_bug_type": round(hits / len(idx), 3)}
    results["per_type_recall"] = type_recall

    results["mismatch_samples"] = mismatches
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=None, help="base model (default Qwen2.5-Coder-1.5B)")
    ap.add_argument("--test", default="data/processed/verdict/test.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="eval/results_finetuned.json")
    args = ap.parse_args()

    results = evaluate(
        Path(args.test), args.adapter, args.model, args.limit, args.max_new_tokens, args.batch_size
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "mismatch_samples"}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
