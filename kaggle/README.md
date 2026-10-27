# Training on Kaggle (free GPU)

The QLoRA fine-tune runs on a free Kaggle T4 — no paid infrastructure.

## 1. Produce the dataset locally

```bash
cd pr-guardian
make setup-core
make mine-sample            # quick: psf/requests, ~40 fixes + 40 clean diffs
# or the fuller run:
.venv/bin/python -m mining.repolist --out data/repolist.csv
.venv/bin/python -m mining.run_mining --repos data/repolist.csv \
    --out data/raw/diffs_v1.jsonl --max-fixes 200 --max-clean 150 --append
```

## 2. Upload the JSONL as a Kaggle Dataset

```bash
pip install kaggle
# Get a token at kaggle.com -> Account Settings -> Create New API Token
# Save it to ~/.kaggle/kaggle.json and chmod 600 it.

mkdir -p /tmp/pr-guardian-diffs
cp data/raw/diffs_v1.jsonl /tmp/pr-guardian-diffs/
cp kaggle/dataset-metadata.json /tmp/pr-guardian-diffs/
# edit /tmp/pr-guardian-diffs/dataset-metadata.json:
#   replace YOUR_KAGGLE_USERNAME with your Kaggle username

kaggle datasets create -p /tmp/pr-guardian-diffs --dir-mode zip
```

## 3. Run the training session

1. kaggle.com → **Code** → **New Notebook**.
2. **Add Input** → your `pr-guardian-diffs` dataset.
3. Settings: **Accelerator = GPU T4**, **Internet = On**
   (Internet is needed for the base model; it also enables the Devign
   fallback if no dataset is attached).
4. Import this repo's `kaggle/PR-Guardian-Training.ipynb`
   (File → Import Notebook) and **Run All** (~30–90 min for a few thousand
   diffs on T4).
5. In the notebook's Output panel, download `pr-guardian-verdict-v1.zip`.

Alternative to the UI:

```bash
kaggle kernels push -p kaggle/   # requires kernel-metadata.json (see Kaggle docs)
kaggle kernels output <you>/pr-guardian-training -p out/
```

## 4. Use the adapter locally

```bash
unzip pr-guardian-verdict-v1.zip -d models/
make setup                                   # full setup incl. torch
PRGUARDIAN_MOCK=0 ADAPTER_PATH=models/pr-guardian-verdict-v1 \
    .venv/bin/python -m backend.cli.prguard review --diff my.diff
```

## 5. The eval that goes in the README

```bash
.venv/bin/python -m training.build_dataset --in data/raw/diffs_v1.jsonl --task verdict
.venv/bin/python -m eval.harness --test data/processed/verdict/test.jsonl \
    --systems mock api ollama semgrep finetuned --adapter models/pr-guardian-verdict-v1
# -> eval/results.md (the cost/quality table for your README)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA OOM | 0.5B model, `MAX_LEN=1024`, `LORA_R=8`, `GRAD_ACCUM=32` |
| No diffs found in `/kaggle/input` | the attached dataset has no `*.jsonl` with `kind`+`diff` keys — rerun step 2 |
| Devign fallback 404 | CodeXGLUE raw URL changed — attach your mined JSONL instead |
| Internet toggle disabled for GPU sessions on your account | attach the dataset; base model also needs Internet — request verification on Kaggle or run on Colab with the same notebook |
