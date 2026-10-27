"""Zero-shot baseline against any OpenAI-compatible chat API.

Uses the SAME prompt records the fine-tuned model sees (build_dataset output),
so the comparison is apples-to-apples. Configure via:

    PRGUARDIAN_API_KEY / PRGUARDIAN_API_BASE / PRGUARDIAN_API_MODEL
"""

from __future__ import annotations

import os
import time

import requests

from training.eval_verdict import compute_metrics
from training.schemas import parse_model_output


def _creds() -> tuple[str | None, str, str]:
    key = os.getenv("PRGUARDIAN_API_KEY") or os.getenv("OPENAI_API_KEY")
    base = (os.getenv("PRGUARDIAN_API_BASE") or "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("PRGUARDIAN_API_MODEL") or "gpt-4o-mini"
    return key, base, model


def _chat(messages: list[dict], base: str, key: str, model: str, json_mode: bool = True) -> str:
    payload = {"model": model, "temperature": 0, "max_tokens": 128, "messages": messages}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        r = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=120,
        )
        if r.status_code == 400 and json_mode:
            return _chat(messages, base, key, model, json_mode=False)  # endpoint may not support json mode
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.RequestException:
        return ""


def run(records: list[dict], limit: int | None = None) -> dict:
    key, base, model = _creds()
    if not key:
        return {"system": "api_zero_shot", "skipped": "no PRGUARDIAN_API_KEY configured"}
    if limit:
        records = records[:limit]

    parsed, outputs, latencies, total_tokens = [], [], [], 0
    for rec in records:
        t0 = time.perf_counter()
        out = _chat(rec["messages"][:-1], base, key, model)
        latencies.append((time.perf_counter() - t0) * 1000)
        outputs.append(out)
        parsed.append(parse_model_output(out))
        total_tokens += sum(len(m["content"]) for m in rec["messages"][:-1]) // 4 + 48

    y_true = ["BUG" if r["meta"]["kind"] == "bug" else "CLEAN" for r in records]
    y_pred = [p.verdict.value if p else "INVALID" for p in parsed]
    conf = [p.confidence if p else None for p in parsed]
    valid = [p is not None for p in parsed]
    results = compute_metrics(records, y_true, y_pred, conf, valid, parsed, [], None, f"api:{model}")
    results["system"] = "api_zero_shot"
    results["avg_latency_ms"] = round(sum(latencies) / max(1, len(latencies)))
    results["est_input_tokens_total"] = total_tokens
    results["est_cost_per_1k_diffs"] = round(
        total_tokens * float(os.getenv("PRGUARDIAN_API_PRICE_IN", "0.00000015")) * 1000 / max(1, len(records)), 4
    )
    return results
