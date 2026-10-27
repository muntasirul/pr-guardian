# PR Guardian — Project Plan

> A fine-tuned code-review agent: LoRA-tuned Qwen2.5-Coder-1.5B trained on mined
> bug-fix diffs, with repo-aware RAG context (tree-sitter + UniXcoder + Qdrant),
> served behind FastAPI/Celery and integrated with GitHub inline PR comments —
> evaluated head-to-head against zero-shot frontier and 8B baselines at a
> fraction of the inference cost.

**Document status:** Implementation plan. Features and results described below
are planned, not completed.

**Planning estimate:** About 7–8 weeks at 12–15 focused hours per week, assuming
familiarity with Python, basic PyTorch, and Docker. Fine-tuning runs on free
Colab/Kaggle GPUs; no paid infrastructure is required for v1.

---

## 1. Problem Statement

Code review is the slowest stage of most development workflows:

- PRs sit in review queues for days; reviewers batch-review under time pressure
  and miss subtle bugs (null derefs, off-by-one, resource leaks, security issues).
- LLM-based review bots exist (CodeRabbit etc.) but are closed, billed per
  usage, send noisy comments when they lack repo context, and code leaves the
  company for a third party.
- Static analysis (linters, semgrep) catches syntax and known patterns but
  misses *semantic* bugs — logic that is wrong but legal.

**PR Guardian solves this with a model you actually train:**

1. Mine real bug-fix commits from popular GitHub repos (the fix tells you which
   earlier diff introduced the bug — the classic SZZ heuristic).
2. Fine-tune a small code LLM (Qwen2.5-Coder-1.5B) with QLoRA on those diffs so
   it learns what defect-introducing changes *look like* — then writes short,
   schema-constrained review comments.
3. Give the model repo context via RAG: tree-sitter chunking, UniXcoder
   embeddings, Qdrant retrieval of related code (callers, similar functions, tests).
4. Post the results as inline GitHub PR comments via a CLI/API, with
   confidence-based filtering to keep comment noise low.
5. Prove the whole thing with a real eval harness: precision of flagged diffs
   and comment quality vs. zero-shot baselines (frontier-lite API model,
   open-8B model, semgrep) — plus a cost-per-1,000-diffs comparison that is the
   headline chart.

---

## 2. Goals & CV Value

| Goal | Role it supports on CV |
|---|---|
| Mine & label bug-fix diff dataset (PyDriller + SZZ-lite) | Data Engineer / ML Engineer |
| QLoRA fine-tune code LLM (defect verdict + review comments) | ML Engineer (GenAI / LLM fine-tuning) |
| Label-noise audit + dataset EDA (yield, dedup, leakage checks) | Data Analyst |
| Repo-aware RAG (tree-sitter + UniXcoder + Qdrant) with ablation study | ML Engineer / LLM systems |
| Eval harness: F1, PR-AUC, precision@k vs. 3 baselines | ML Engineer / Data Analyst |
| Cost & latency analysis (fine-tuned 1.5B vs. API models) | Data Analyst / MLOps |
| FastAPI + Celery + Postgres + Qdrant service, GitHub API integration, CLI | Software Engineer |
| pytest suite, CI, Docker, model card, demo video | Software Engineer |

**Headline CV bullet (end state):**

> Built an open-source code-review agent by QLoRA fine-tuning Qwen2.5-Coder-1.5B
> on ~10k bug-fix diffs mined via a PyDriller/SZZ pipeline; added repo-aware RAG
> (tree-sitter + UniXcoder + Qdrant) and shipped it as a FastAPI/Celery service
> with GitHub inline-review integration. Matched zero-shot GPT-4o-mini-class
> defect detection on a held-out per-repo test split at ~30x lower inference
> cost, with an eval harness covering F1, PR-AUC, precision@k, and per-bug-class
> breakdowns.

**Secondary bullet (data side):**

> Designed and audited a bug-fix diff mining pipeline over 50+ popular Python
> repositories; quantified label noise via a 100-sample manual audit and
> decontaminated train/test splits at the repository level to prevent leakage.

---

## 3. What Makes This Project Credible (vs. "I prompted GPT")

1. **You fine-tune a model** — QLoRA on a code LLM, not just prompt engineering.
2. **You build the dataset yourself** — mining + SZZ labeling is real data
   engineering; most candidates never build training data.
3. **You measure honestly** — frozen per-repo test split, decontamination,
   three baselines, cost curves. The claim is "parity at 30x lower cost",
   not a magic accuracy number.
4. **You integrate with real software** — GitHub API inline comments, queue,
   retries, rate limits: genuine SWE work, not a notebook.
5. **The demo is unforgettable** — run it on the interviewer's own PR.

---

## 4. System Architecture

```
                 ┌────────────────────────────────────────────┐
                 │              Interfaces                     │
                 │  CLI: prguard review --pr <URL>            │
                 │  (stretch: GitHub App webhook, CI action)  │
                 └───────────────┬────────────────────────────┘
                                 │ REST
                 ┌───────────────▼────────────────────────────┐
                 │              FastAPI backend                 │
                 │  POST /review   GET /reviews/{id}           │
                 │  POST /feedback GET  /stats                 │
                 └──────┬───────────────────────────┬──────────┘
                        │ enqueue                   │ read
                ┌───────▼────────┐         ┌───────▼────────┐
                │  Celery worker  │         │   PostgreSQL    │
                │                │         │ reviews │ runs   │
                │ 1. fetch diff  │         │ comments│ fbk   │
                │ 2. chunk diff  │         └──────────────────┘
                │ 3. build RAG   │
                │    context     │  ┌───────────────────────────────┐
                │ 4. inference ──┼─►│  Inference service             │
                │ 5. filter by   │  │  Qwen2.5-Coder-1.5B + QLoRA    │
                │    confidence  │  │  adapter, schema-constrained  │
                │ 6. post inline │  │  JSON output (verdict/        │
                │    comments    │  │  severity/comment/location)   │
                └───────┬────────┘  └───────────────────────────────┘
                        │
          ┌─────────────▼──────────────┐      ┌──────────────────────┐
          │  RAG context service        │      │  Qdrant vector DB     │
          │  tree-sitter chunking       │◄─────│  UniXcoder embeddings │
          │  changed fn -> related code │      │  of indexed repo      │
          │  (callers, tests, siblings) │      └──────────────────────┘
          └────────────────────────────┘

Training side (offline, Colab/Kaggle):
  GitHub repos ──► PyDriller miner ──► fix commits ──► SZZ-lite mapping
       │                                              │
       │        ┌─────────────────────────────────────┘
       │        ▼
       │   (bug-introducing diff, buggy label)
       │   (long-lived diff, clean label)     Devign/Big-Vul (seed, optional)
       │        │                                   │
       │        ▼                                   │
       │   LLM-as-labeler (one-time, offline):      │
       │   fix diff + fix message + issue title ──► natural reviewer comment
       │        │
       │        ▼
       │   build_dataset.py (dedup, per-repo split, decontamination)
       │        │
       │        ▼
       └──► QLoRA fine-tune (PEFT + bitsandbytes) ──► frozen test set eval
                        │                                │
                        ▼                                ▼
                 MLflow (runs, adapter)         promote only if metrics hold
                                                        │
                                                        ▼
                                              Inference service (versioned)
```

**Key components**

1. **Diff miner** — PyDriller over 50–100 popular Python repos; identify fix
   commits (message keywords, linked issues); SZZ-lite: `git blame` the lines the
   fix touched to find the commit that introduced them.
2. **Dataset builder** — positives = bug-introducing diffs; negatives = diffs to
   code that survived 12+ months unmodified (cheap "probably clean" heuristic);
   LLM-as-labeler converts fix metadata into reviewer-style target comments;
   dedup + repo-level splits + decontamination.
3. **Defect + comment model** — one fine-tuned instruct model, two-stage prompt:
   (a) verdict `BUG|CLEAN` with type + severity, (b) if BUG, a 1–3 sentence
   comment with a location hint; outputs validated by Pydantic schema.
4. **RAG context layer** — tree-sitter extracts function/class chunks; UniXcoder
   embeds; Qdrant stores; at review time, retrieve related chunks (callers,
   similar functions, tests) for the changed functions only.
5. **Confidence filter** — only post comments above a tuned threshold; clean
   verdicts post nothing (comment noise is the #1 reason review bots get muted).
6. **Eval harness** — frozen per-repo test set; baselines: zero-shot API model,
   zero-shot open-8B model, semgrep; precision@k, F1, PR-AUC, per-class recall.
7. **Analytics dashboard** — eval results, bug-class coverage, cost curves,
   caught-bug gallery, feedback stats (GitHub reactions API as feedback signal).

---

## 5. Data Strategy & Datasets

### Primary (self-mined) — the real asset

| Item | Detail |
|---|---|
| Source repos | 50–100 popular Python repos (Django, requests, pandas, flask, scikit-learn, httpx, etc.) |
| Miner | PyDriller (fix-commit detection via message keywords + linked issue refs) |
| Labeling | SZZ-lite: blame the lines a fix touched → the introducing commit's diff = positive example |
| Negative sampling | Diffs to files unchanged for 12+ months (survival heuristic) |
| Comment targets | One-time offline LLM pass: fix diff + commit message + issue title → 1–3 sentence reviewer-style comment (approx. $10–20 total for ~10k pairs) |
| Expected yield | 5k–20k usable pairs after filtering (fix touches ≥3 lines, English message, dedup) |
| Balance | Target ~1 buggy : 3–5 clean; report PR-AUC so imbalance choice is defensible |
| Split discipline | **Hold out entire repos** for test (never split by diff) + near-dup decontamination vs. training set |

### Public datasets (supporting / optional seed)

| Dataset | Use | Source |
|---|---|---|
| **Devign / CodeXGLUE defect detection** | Function-level buggy/clean seed task (C) — optional warmup for the classifier phase | CodeXGLUE benchmark page (search "CodeXGLUE defect detection") |
| **Big-Vul** | CVE-labeled functions (C/C++) — EDA + optional extra pretraining | Big-Vul (ASE 2020); mirrors on GitHub/Zenodo |
| **CVEfixes** | CVE-fixing commits across languages — optional augmentation for security classes | Zenodo (search "CVEfixes dataset") |
| **CodeReviewer dataset (Microsoft, FSE'21)** | Real (diff → review comment) pairs for Java, if the generative phase needs a larger reference corpus | Released with the CodeReviewer paper; search "CodeReviewer dataset" |
| **SWE-bench Lite (stretch)** | Real issue-fix pairs for a final "hard mode" eval slice | princeton-nlp SWE-bench |

**Label-noise audit (Data Analyst deliverable):** manually audit a random 100
mined "bug-introducing" diffs; report precision of the SZZ-lite heuristic
(typically imperfect — that's the point); document observed noise rate in the
EDA notebook and model card. Review dataset terms/licenses before downloading;
keep raw data out of Git and provide a fetch script.

---

## 6. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Base model | Qwen2.5-Coder-1.5B-Instruct | small, strong on code, QLoRA-able on a free T4 |
| Fine-tuning | HF Transformers + PEFT + bitsandbytes (QLoRA) | standard, reproducible, single-GPU |
| Fallback model | Qwen2.5-Coder-0.5B-Instruct | if VRAM is tight |
| Stretch model | Qwen2.5-Coder-7B-Instruct | on Kaggle P100 / rented GPU, compare |
| Baselines | GPT-4o-mini-class API model (or current equivalent), Llama-3.1-8B-Instruct (via ollama), semgrep | three different families of competitors |
| Mining | PyDriller | purpose-built for commit/fix mining |
| Chunking | tree-sitter | robust AST-level function/class extraction |
| Embeddings | UniXcoder (microsoft/unixcoder-base) | pretrained for code search; CodeBERT fallback |
| Vector DB | Qdrant (docker) | simple, fast, metadata filtering for path/lang |
| Serving | HF transformers + PEFT adapter loading (vLLM with LoRA as stretch) | 1.5B runs fine without heavyweight serving |
| Experiment tracking | MLflow | runs, metrics, adapter registry |
| Backend | FastAPI + Pydantic | async, typed, schema-validated model outputs |
| Task queue | Celery + Redis | review jobs are multi-stage and slow |
| DB | PostgreSQL | reviews, comments, feedback, eval runs |
| GitHub client | PyGithub / `gh api` (CLI) | fetch PR diff, post inline review comments |
| CLI | Typer | fast to build, good UX |
| Dashboard | Streamlit | eval results, cost curves, gallery |
| Infra | Docker + docker-compose | one-command reproducibility |
| CI | GitHub Actions | lint, unit tests, dataset build dry-run |
| Tests | pytest, ruff | core paths: mining, schema, filter, API |

---

## 7. Milestones (Step-by-Step)

### Phase 0 — Setup & Data Exploration (Week 1)

- [ ] Scaffold repo (see §8), `docker-compose.yml` (api, worker, redis, qdrant,
      postgres), CI skeleton
- [ ] Write `mining/fetch_repolist.py`: curate the 50–100 repo list (stars,
      language, active) and save as `data/repolist.csv`
- [ ] Run miner on 5 repos end-to-end; `notebooks/01_eda_mining.ipynb`:
      fix-commit yield, diff-size distribution, survival-heuristic stats
- [ ] Define bug taxonomy: `null/none-safety, off-by-one/index, resource leak,
      exception handling, security (injection/xss/secret), concurrency, api-misuse,
      other`
- [ ] **Deliverables:** repo scaffold, EDA notebook, taxonomy doc in `docs/`

### Phase 1 — Diff Mining Pipeline (Weeks 1–2) — data engineering asset

- [ ] `mining/mine_fix_commits.py`: PyDriller fix-commit detection (message
      keywords: fix, bug, resolve, close + issue refs)
- [ ] `mining/szz_introducing.py`: blame-based mapping fix → introducing commit;
      extract the introducing diff as positive example with metadata
- [ ] `mining/mine_clean_diffs.py`: survival heuristic negatives (file untouched
      12+ months)
- [ ] Filters: diff 3–200 lines, single file preferred, drop vendored/generated
      paths (`*_pb2.py`, `vendor/`, minified JS analogues)
- [ ] Dedup (diff hash + normalized text hash); store raw JSONL in `data/raw/`
- [ ] **Deliverable:** `data/diffs_v1.jsonl` (5k–20k pairs) + stats notebook

### Phase 2 — Dataset Build & Label Audit (Week 2)

- [ ] `training/build_dataset.py`: JSONL → chat-format pairs; repo-level
      train/val/test split (80/10/10 by repo, test repos never trained on)
- [ ] LLM-as-labeler pass (one-time, offline): fix diff + fix message + issue
      title → reviewer-style comment; schema-validate outputs; manual spot-check
      50 generated comments for quality (report % usable)
- [ ] **Label-noise audit:** manually grade 100 sampled "bug-introducing" diffs;
      record noise rate in `notebooks/02_label_audit.ipynb`
- [ ] **Deliverables:** versioned train/val/test JSONL + audit notebook

### Phase 3 — Defect Verdict Model (Week 3)

- [ ] Baseline A: zero-shot API model (GPT-4o-mini class) on a 200-diff pilot
      set — verdict prompt, fixed template
- [ ] Baseline B: zero-shot Llama-3.1-8B-Instruct via ollama, same prompt
- [ ] QLoRA fine-tune Qwen2.5-Coder-1.5B on the verdict task (BUG|CLEAN + type);
      seq len 2048, 4-bit, rank 16–32; ~1–3 h on T4
- [ ] Evaluate on frozen per-repo test split: accuracy, F1, **PR-AUC**,
      per-bug-class recall, confusion matrix; log all to MLflow
- [ ] **Deliverables:** `notebooks/03_verdict_model.ipynb`, MLflow runs,
      baseline numbers in a results table

### Phase 4 — RAG Context Layer (Week 4)

- [ ] `rag/chunker.py`: tree-sitter function/class extraction (Python first)
- [ ] `rag/embed.py` + `rag/indexer.py`: UniXcoder embeddings → Qdrant with
      path/lang metadata
- [ ] `rag/retrieve.py`: for each changed function, retrieve top-k related
      chunks (same-repo filter; callers/tests prioritized by path heuristics)
- [ ] **Retrieval quality metric:** "did the fix touch any retrieved file?"
      → context recall@5/@10 on 100 fix commits (measurable, honest)
- [ ] Prompt v2: diff + retrieved context → verdict/comment; A/B compare
      zero-context vs. RAG-context on the pilot set
- [ ] **Deliverables:** indexed sample repo, `notebooks/04_rag_ablation.ipynb`

### Phase 5 — Generative Review Comments (Week 5)

- [ ] Prompt design: two-stage (verdict first, then comment only if BUG);
      constrain output to JSON schema `{"verdict", "type", "severity", "comment",
      "location_hint"}`; Pydantic validation + retry-on-invalid
- [ ] QLoRA fine-tune v2 on comment targets (diff + context → JSON review)
- [ ] Comment quality rubric eval on ~100 samples: correct location? describes
      the actual problem? actionable? (self-graded with a fixed rubric; report
      the rubric, not vibes)
- [ ] Confidence filter: tune threshold on val set to hit a target
      precision@top-20% (the knob that controls comment noise)
- [ ] **Deliverables:** `training/train_reviewer.py`, rubric results notebook

### Phase 6 — Backend, CLI & GitHub Integration (Weeks 5–6)

- [ ] FastAPI: `POST /review` (PR URL or diff payload), `GET /reviews/{id}`,
      `GET /stats`, `POST /feedback`
- [ ] Celery pipeline: fetch PR files → chunk → retrieve context → inference →
      confidence filter → GitHub inline comments
      (`POST /repos/{owner}/{repo}/pulls/{n}/comments`, anchored to diff positions)
- [ ] CLI: `prguard review --pr <URL>` (Typer + GITHUB_TOKEN); GitHub App /
      webhook / CI action are stretch goals
- [ ] Engineering hygiene: retries + backoff, GitHub rate-limit handling,
      idempotency (don't double-post on PR pushes), p50/p95 latency logging
- [ ] pytest on: schema validation, confidence filter, miner unit tests with a
      tiny fixture repo, API contract tests; ruff + CI green
- [ ] **Deliverables:** working CLI demo on a real public PR, test suite
      (>70% on core paths), Swagger docs

### Phase 7 — Eval Harness & Experiments (Week 6–7) ⭐ the credibility core

- [ ] `eval/harness.py`: frozen per-repo test diffs → run all four systems
      (fine-tuned w/ and w/o RAG, API zero-shot, 8B zero-shot, semgrep)
- [ ] Metrics: F1, PR-AUC, **precision@k** (top 10%/20% by confidence),
      per-bug-class recall, verdict distribution vs. ground truth
- [ ] **Cost analysis:** tokens × price per 1,000 diffs for each system;
      include one-time training cost amortized; latency p50/p95 table
- [ ] Error analysis: 30 FP + 30 FN case studies, categorized; write
      `notebooks/05_error_analysis.ipynb` with the honest failure taxonomy
- [ ] **Deliverables:** `docs/results.md` with all tables; headline cost/quality
      chart for the README

### Phase 8 — Dashboard & Polish (Weeks 7–8)

- [ ] Streamlit dashboard tabs: (1) eval results, (2) bug-class coverage vs.
      baselines, (3) cost & latency curves, (4) caught-bug gallery (diff +
      comment screenshots), (5) feedback stats
- [ ] README: line 1 = business impact; demo GIF; Mermaid architecture diagram;
      quickstart (`docker compose up` + CLI); model card; dataset card
- [ ] Record a 2–3 min demo video (Loom): mine → train → review a live PR
- [ ] Dogfood: run PR Guardian on 10 real PRs across your own repos; publish
      a writeup of catches/misses (honesty = credibility)
- [ ] **Deliverables:** polished repo, demo video, blog-style writeup, model card

### Stretch Goals (after v1)

- [ ] GitHub App with webhook + PR reactions (:thumbsup/:thumbsdown) as a
      feedback signal → feedback-weighted retraining (data flywheel, echoes
      DocuTriage's active-learning loop)
- [ ] Qwen2.5-Coder-7B fine-tune; report the quality/cost frontier
- [ ] Multi-language support (JavaScript/TypeScript via tree-sitter grammars)
- [ ] vLLM + LoRA serving for higher throughput; deploy to a cheap GPU VPS
      or HF Inference Endpoint with a live badge
- [ ] SWE-bench Lite as a "hard mode" eval slice
- [ ] "Verify before commenting" self-check pass (second inference that must
      confirm the bug) to cut false positives further

---

## 8. Repository Structure

```
pr-guardian/
├── docker-compose.yml
├── .github/workflows/ci.yml
├── data/                        # gitignored; fetch scripts produce everything
│   ├── replist.csv
│   └── raw/                     # mined JSONL
├── mining/
│   ├── fetch_repolist.py
│   ├── mine_fix_commits.py
│   ├── mine_clean_diffs.py
│   └── szz_introducing.py
├── training/
│   ├── build_dataset.py         # JSONL -> chat pairs, repo-level splits
│   ├── label_comments.py        # one-time LLM-as-labeler pass
│   ├── train_verdict.py         # QLoRA classifier-style fine-tune
│   ├── train_reviewer.py        # QLoRA generative fine-tune
│   ├── prompts.py
│   └── schemas.py               # Pydantic models for model output
├── rag/
│   ├── chunker.py               # tree-sitter
│   ├── embed.py                 # UniXcoder
│   ├── indexer.py                # Qdrant
│   └── retrieve.py
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI
│   │   ├── routes/              # review, reviews, stats, feedback
│   │   ├── workers/             # Celery tasks (multi-stage pipeline)
│   │   ├── github/              # PR diff fetch, inline comments
│   │   ├── inference/           # adapter loading, schema validation, retry
│   │   └── db/                  # SQLAlchemy models + migrations
│   ├── cli/
│   │   └── prguard.py           # Typer CLI
│   └── tests/
├── eval/
│   ├── harness.py
│   ├── baselines/
│   │   ├── api_zero_shot.py
│   │   ├── local_zero_shot.py   # ollama 8B
│   │   └── semgrep_runner.py
│   └── report.py
├── notebooks/
│   ├── 01_eda_mining.ipynb
│   ├── 02_label_audit.ipynb
│   ├── 03_verdict_model.ipynb
│   ├── 04_rag_ablation.ipynb
│   └── 05_error_analysis.ipynb
├── app/
│   └── dashboard.py             # Streamlit
├── docs/
│   ├── architecture.md
│   ├── model_card.md
│   └── results.md
├── tests/
│   ├── fixtures/tiny_repo/      # synthetic repo for miner unit tests
│   └── ...
└── README.md
```

---

## 9. Evaluation Plan (what to report)

| Level | Metric | Notes |
|---|---|---|
| Verdict model | F1, accuracy, **PR-AUC** | on frozen per-repo test set; PR-AUC because classes are imbalanced |
| Verdict model | per-bug-class recall | shows where it wins (null-safety) vs. loses (concurrency) |
| Review ranking | **precision@top-10% / top-20%** | the review-app metric: reviewers only look at the top of a ranked queue |
| Comments | rubric score on ~100 samples | correct location? real problem? actionable? publish the rubric |
| RAG | context recall@5/@10 + quality delta | honest A/B: zero-context vs. RAG-context |
| Retrieval | context recall on 100 fix commits | did retrieval surface the file the fix touched? |
| System | p50/p95 latency per diff; GitHub API success rate | Software Engineer talking points |
| Cost | $ per 1,000 diffs per system; one-time training cost amortized | the headline table vs. API baselines |
| Robustness | % schema-valid outputs; retry rate; FP comment rate | guards against "hallucinated reviewer" criticism |
| Data | SZZ label-noise rate from the 100-diff audit | turns dataset weakness into a strength |

**Target outcomes (honest, defensible goals):**

- Verdict F1 within ±0.05 of the best zero-shot baseline, or better, at ~30x
  lower cost per diff — parity claims are more credible than big-number claims.
- Measurable lift from RAG context (or a documented null result — still publishable).
- Confidence-filtered comment stream with ≥80% precision@top-20% on val data.

**Anti-leakage rules (your defense in interviews):** test split is by *repo*,
never by diff; near-duplicate check between train and test diffs; the 200-diff
pilot set for baseline prompting is disjoint from training; frozen test set is
never fine-tuned on.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| T4 VRAM too small for 1.5B QLoRA | seq 1024–2048, batch 1 + grad accumulation; 0.5B fallback; Kaggle P100 |
| Mined labels are noisy (SZZ is heuristic) | quantify noise in the 100-diff audit; survival heuristic for negatives; treat labels as weak supervision |
| Generative comments hallucinate on clean diffs | two-stage verdict-then-comment prompt; confidence threshold; "verify before commenting" stretch; measure FP comment rate |
| Bug-fix mining yields too few pairs | widen repo list (100+), relax single-file filter, augment with CVEfixes/Big-Vul, accept smaller dataset + more epochs |
| Eval set contaminated by near-dupes | repo-level splits + normalized-diff near-dup scan between train/test |
| GitHub App setup is a time sink | CLI-first with a GITHUB_TOKEN; webhook/App as stretch only |
| LLM-as-labeler produces bad comments | spot-check 50, drop unusable ones, keep the raw fix evidence alongside each comment |
| Comment latency too high for good demo UX | cache per-diff reviews; short context (top-3 chunks); quantized serving; document p95 honestly |
| Crowded space (CodeRabbit etc.) | differentiators are the trained model, the mined dataset, the cost study, and honest eval — none of which the closed bots publish |

---

## 11. Interview Talking Points (prepared answers)

- **Why fine-tune instead of just prompting GPT?** Cost at volume, latency,
  privacy (code never leaves the machine), control over comment format and
  style, and the eval harness proves parity on this task — that's an
  engineering-economics argument, not hype.
- **Why not just use semgrep?** Pattern-based tools catch syntax-level and
  known-CWE issues; the model generalizes to semantic bugs. The eval shows
  which classes each system wins — the overlap is smaller than you'd think.
- **Your SZZ labels are noisy — doesn't that invalidate training?** Yes, they're
  noisy; that's measured (audit), mitigated (survival negatives, filters), and
  standard practice in defect-prediction research (SZZ is the field's baseline
  heuristic). Weak supervision beats no supervision when you quantify it.
- **How do you evaluate a *comment*, not a verdict?** A published rubric on a
  fixed sample: correct location, describes the real problem, actionable.
  Acknowledge subjectivity; small rubric evals are defensible, vibes are not.
- **Why RAG instead of a long context window?** Small model + 2048 tokens can't
  fit a repo; retrieval adds exactly the related code (callers, tests) that a
  reviewer reads first. Ablation quantifies the lift.
- **How do you prevent train/test leakage?** Repo-level splits, near-dup scans,
  pilot set disjoint from training, frozen test set.
- **What breaks at scale?** Inference throughput (→ vLLM/LoRA serving), queue
  depth, GitHub rate limits, multi-repo indexing freshness, cost per PR.
- **Biggest failure mode?** Confident-but-wrong comments on clean diffs —
  mitigated by confidence filtering and the verify pass; measured as FP comment
  rate, not hidden.

---

## 12. Success Criteria (definition of done)

- [ ] `docker compose up` + `prguard review --pr <URL>` works from a fresh clone
      and posts a review on a real public PR
- [ ] Dataset: ≥5k mined pairs; label-noise audit published with the noise rate
- [ ] Verdict model: F1 within ±0.05 of the best zero-shot baseline on the
      frozen per-repo test split
- [ ] Eval harness runs all four systems end-to-end and produces
      `docs/results.md` with quality, cost, and latency tables
- [ ] RAG ablation published (lift or honest null result)
- [ ] Comment stream: ≥80% precision@top-20% on val data under the tuned filter
- [ ] CI green; README with demo GIF, Mermaid architecture, model card; demo video recorded
- [ ] Dogfooding writeup: 10 real PRs reviewed, catches and misses documented
- [ ] You can walk through any file — mining, training, serving, eval — and
      defend every design trade-off
