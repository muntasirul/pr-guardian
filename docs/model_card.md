# Model Card — PR Guardian verdict/review v1

_To be filled in after the first full training run. Template per
https://arxiv.org/abs/1810.03993 conventions._

## Model details
- Base model: Qwen2.5-Coder-1.5B-Instruct
- Fine-tuning: QLoRA (r=16, NF4) on <dataset version / commit hash>
- Adapter artifact: `models/verdict-v1/` (see `run_meta.json` for hyperparams)

## Intended use
Reviewing Python diffs for defect introduction; flagging low-confidence changes
for human review. **Not** a general code-quality linter, security product, or
merge gate.

## Training data
- Self-mined via PyDriller + SZZ-lite from N repos (see `data/raw/*.stats.json`)
- Optional seed: Devign/CodeXGLUE function-level data
- Label-noise audit: SZZ precision **__%** (100-diff manual sample, Phase 2)

## Metrics (held-out per-repo test split)
| Metric | Verdict task | Review task |
|---|---|---|
| Accuracy | TBD | TBD |
| Macro F1 | TBD | TBD |
| PR-AUC | TBD | TBD |
| Valid-output rate | TBD | TBD |

Baselines (same test split): API zero-shot TBD, 8B zero-shot TBD, semgrep TBD.

## Known failure modes
- Confidence-filtered comment stream still false-positives on defensive code
  (error handling patterns flagged as "swallowed exceptions").
- Long diffs truncated at 6k chars; multi-file refactor diffs lose context.
- Trained on Python only; other languages unsupported in v1.

## Ethics / safety
- Trained exclusively on open-source code under permissive licenses; no
  proprietary code.
- Outputs a verdict + short comment; it does not auto-block merges —
  humans stay in the loop by design.
