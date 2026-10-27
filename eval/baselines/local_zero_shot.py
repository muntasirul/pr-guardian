"""Zero-shot baseline against a local ollama server.

    OLLAMA_URL=http://localhost:11434 OLLAMA_MODEL=llama3.1:8b-instruct ...
"""

from __future__ import annotations

import os
import time

import requests

from training.eval_verdict import compute_metrics
from training.schemas import parse_model_output


def run(records: list[dict], limit: int | None = None) -> dict:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct")
    try:
        requests.get(f"{url}/api/tags", timeout=5)
    except requests.RequestException:
        return {"system": "ollama_zero_shot", "skipped": f"no ollama server at {url}"}
    if limit:
        records = records[:limit]

    parsed, latencies = [], []
    for rec in records:
        t0 = time.perf_counter()
        try:
            r = requests.post(
                f"{url}/api/chat",
                json={
                    "model": model,
                    "messages": rec["messages"][:-1],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                },
                timeout=300,
            )
            out = r.json().get("message", {}).get("content", "") if r.status_code == 200 else ""
        except requests.RequestException:
            out = ""
        latencies.append((time.perf_counter() - t0) * 1000)
        parsed.append(parse_model_output(out))

    y_true = ["BUG" if r["meta"]["kind"] == "bug" else "CLEAN" for r in records]
    y_pred = [p.verdict.value if p else "INVALID" for p in parsed]
    conf = [p.confidence if p else None for p in parsed]
    valid = [p is not None for p in parsed]
    results = compute_metrics(records, y_true, y_pred, conf, valid, parsed, [], None, f"ollama:{model}")
    results["system"] = "ollama_zero_shot"
    results["avg_latency_ms"] = round(sum(latencies) / max(1, len(latencies)))
    results["est_cost_per_1k_diffs"] = 0.0  # local hardware
    return results
