"""PR Guardian analytics dashboard.

    streamlit run app/dashboard.py

Tabs: dataset stats, eval results (the cost/quality table), mismatch samples,
and a monthly cost calculator (API model vs self-hosted).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="PR Guardian")

st.title("PR Guardian — Evaluation & Dataset Dashboard")


def _load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


tab_stats, tab_eval, tab_mismatch, tab_cost = st.tabs(
    ["Dataset", "Eval results", "Mismatches", "Cost calculator"]
)

with tab_stats:
    processed = sorted(Path("data/processed").glob("*/stats.json")) if Path("data/processed").exists() else []
    if not processed:
        st.info("No dataset stats yet. Run `make mine-sample && make data` to build a dataset.")
    else:
        task = st.selectbox("Task", [p.parent.name for p in processed])
        stats = _load_json(f"data/processed/{task}/stats.json")
        if stats:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total examples", stats["total"])
            col2.metric("Repos", stats["repos"])
            col3.metric("Split mode", stats["split_mode"])
            st.subheader("Split sizes & labels")
            rows = []
            for split, n in stats["splits"].items():
                labels = stats["labels"].get(split, {})
                rows.append(
                    {
                        "split": split,
                        "n": n,
                        "bug": labels.get("bug", 0),
                        "clean": labels.get("clean", 0),
                    }
                )
            st.table(pd.DataFrame(rows))
            st.caption(
                f"dropped exact duplicates: {stats['dropped_exact_duplicates']} · "
                f"test near-duplicates removed (leakage guard): {stats['dropped_test_near_duplicates']}"
            )

with tab_eval:
    results = _load_json("eval/results.json")
    if not results or not results.get("systems"):
        st.info("No eval results yet. Run `python -m eval.harness --test data/processed/verdict/test.jsonl`.")
    else:
        rows = []
        for r in results["systems"]:
            rows.append(
                {
                    "system": r.get("system"),
                    "n": r.get("n"),
                    "valid %": r.get("valid_output_rate"),
                    "accuracy": r.get("accuracy"),
                    "macro F1": r.get("macro_f1"),
                    "PR-AUC": r.get("pr_auc"),
                    "latency ms": r.get("avg_latency_ms"),
                    "$ / 1k diffs": r.get("est_cost_per_1k_diffs"),
                    "skipped": r.get("skipped"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("Full details: eval/results.json / results.md")

with tab_mismatch:
    results = _load_json("eval/results.json")
    mismatches = []
    for r in (results or {}).get("systems", []):
        mismatches.extend(r.get("mismatch_samples", []))
    if not mismatches:
        st.info("Mismatch samples appear here after running the harness with a model system.")
    else:
        for i, m in enumerate(mismatches[:20]):
            with st.expander(f"{m['id']} — true={m['true']} pred={m['pred']}"):
                st.code(m["diff_preview"], language="diff")
                st.text(m["model_output"])

with tab_cost:
    st.subheader("API model vs fine-tuned local model (monthly estimate)")
    diffs_per_month = st.number_input("Diffs reviewed / month", 100, 500_000, 5_000, step=100)
    price_in = st.number_input("API price $/1M input tokens", 0.01, 10.0, 0.15, step=0.01)
    tokens_per_diff = st.number_input("Avg tokens per diff (prompt)", 100, 20_000, 1500, step=100)
    gpu_hour = st.number_input("Self-hosted GPU $/hour", 0.1, 10.0, 0.5, step=0.1)
    sec_per_diff = st.number_input("Local inference sec/diff", 0.1, 60.0, 3.0, step=0.1)

    api_cost = diffs_per_month * tokens_per_diff / 1_000_000 * price_in
    local_cost = diffs_per_month * sec_per_diff / 3600 * gpu_hour
    c1, c2, c3 = st.columns(3)
    c1.metric("API model", f"${api_cost:,.0f} / mo")
    c2.metric("Fine-tuned (self-hosted)", f"${local_cost:,.0f} / mo")
    c3.metric("Savings", f"${api_cost - local_cost:,.0f} / mo" if api_cost > local_cost else "—")
    st.caption("Estimates only — run `python -m eval.harness` for measured numbers.")
