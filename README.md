# PR Guardian

**A fine-tuned code-review agent: LoRA-tuned Qwen2.5-Coder-1.5B trained on real
bug-fix diffs, with repo-aware RAG context, served behind a FastAPI pipeline
that posts inline GitHub PR comments — evaluated head-to-head against zero-shot
API and open-8B baselines at a fraction of the cost.**

Status: **work in progress** — the mining pipeline, dataset builder, trainer,
eval harness, API, and dashboard are implemented; model quality numbers land
after the first full Kaggle training run (see `kaggle/README.md`).

Full plan with phases, success criteria, and interview talking points:
[`PR-Guardian-Project-Plan.md`](PR-Guardian-Project-Plan.md).

## What it does

1. **Mines its own dataset** — clones popular Python repos, detects bug-fix
   commits, and uses the SZZ heuristic (`git blame`) to recover the diff that
   *introduced* each bug. Clean examples come from a survival heuristic
   (diffs to code untouched for 12+ months).
2. **Fine-tunes a small code LLM** (QLoRA, free Kaggle T4) to review diffs and
   emit schema-constrained JSON verdicts + review comments.
3. **Reviews PRs** — paste a diff or point it at a GitHub PR; the pipeline
   chunks the diff, optionally retrieves repo context (RAG), runs inference,
   filters by confidence, and posts inline comments via the GitHub API.
4. **Measures honestly** — a per-repo-split test set (never trained on),
   near-duplicate decontamination, and baselines: zero-shot API model,
   zero-shot open-8B, semgrep, and a rule-based mock.

## Quickstart

```bash
make setup-core          # venv + core deps (no torch needed for everything below)

# 1. mine a small real dataset (~2 min, clones psf/requests)
make mine-sample

# 2. build train/val/test chat datasets
make data

# 3. review a diff with the rule-based mock model (works without ML deps)
echo '@@ -1 +1 @@
-def run(u):
+def run(u):
+    return eval(u)' > /tmp/d.diff
.venv/bin/python -m backend.cli.prguard review --diff /tmp/d.diff

# 4. tests
make test-fast

# 5. API + dashboard
make api                # http://localhost:8000/docs
make dashboard
```

Public seed datasets: `make fetch-data` (Devign/CodeXGLUE defect detection,
SWE-bench Lite) — everything lands in `data/raw/` with a manifest.

## Training on Kaggle (free GPU)

See [`kaggle/README.md`](kaggle/README.md). In short: run mining locally,
upload the JSONL as a Kaggle dataset, open `kaggle/PR-Guardian-Training.ipynb`,
enable GPU, run — the notebook downloads the base model, QLoRA-fine-tunes it,
evaluates on the held-out split, and zips the adapter for download.

Local end-to-end trainer smoke test (tiny model, CPU, ~1 min):

```bash
make setup               # includes torch/transformers/peft
make train-smoke
```

## Repository map

| Path | What |
|---|---|
| `mining/` | fix-commit miner, SZZ-lite blame mapping, survival negatives, orchestrator |
| `training/` | schemas, prompts, dataset builder (repo splits + decontamination), comment labeler, QLoRA trainer, evaluator, public-dataset fetcher |
| `rag/` | AST chunker, UniXcoder/dummy embedders, local vector index, retriever |
| `backend/` | FastAPI app, review pipeline, mock + HF inference, GitHub client, Celery worker, Typer CLI |
| `eval/` | harness + baselines (API zero-shot, ollama zero-shot, semgrep) + results.md report |
| `app/` | Streamlit dashboard (dataset stats, eval tables, cost calculator) |
| `kaggle/` | Kaggle training notebook + dataset upload instructions |
| `tests/` | pytest suite incl. a synthetic git repo that verifies SZZ end-to-end |

## Design notes (the honest parts)

- **SZZ labels are noisy.** We quantify it: a 100-diff manual audit is part of
  Phase 2, and the noise rate goes into the model card.
- **Mock inference** (`PRGUARDIAN_MOCK=1`) keeps the full pipeline demoable
  without ML deps — it is a handful of high-precision regex rules, clearly
  labeled "mock" in every API response.
- **Anti-leakage is code, not vibes**: repo-level splits, exact-diff dedup, and
  shingle-based near-dup decontamination of the test split
  (`training/build_dataset.py`).

## License

MIT for this code. Public datasets used for training/eval carry their own
licenses (see `data/raw/fetch_manifest.json` after fetching); raw data is
never committed.
