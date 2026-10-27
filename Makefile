PY := .venv/bin/python
PIP := .venv/bin/pip

.DEFAULT_GOAL := help

setup: ## full setup (venv + all deps incl. ML stack)
	python3 -m venv .venv
	$(PIP) install -r requirements.txt -r requirements-ml.txt -r requirements-dev.txt

setup-core: ## light setup (venv + core + dev deps, no torch)
	python3 -m venv .venv
	$(PIP) install -r requirements.txt -r requirements-dev.txt

lint:
	.venv/bin/ruff check mining training rag backend eval app tests

test:
	.venv/bin/pytest

test-fast:
	.venv/bin/pytest -m "not slow"

fetch-data: ## download public datasets (Devign, SWE-bench Lite, ...)
	$(PY) -m training.fetch_public_datasets

mine-sample: ## small real mining demo on psf/requests
	$(PY) -m mining.run_mining --repos psf/requests --out data/raw/sample_requests.jsonl --max-fixes 40 --max-clean 40

data: ## build train/val/test chat datasets from mined JSONL
	$(PY) -m training.build_dataset --in data/raw/sample_requests.jsonl --task verdict
	$(PY) -m training.build_dataset --in data/raw/sample_requests.jsonl --task review

train-smoke: ## end-to-end trainer smoke test with a tiny model on CPU
	$(PY) -m training.train_verdict --task verdict --data-dir data/processed/verdict \
		--model hf-internal-testing/tiny-random-MistralForCausalLM --epochs 1 --max-examples 16 --smoke-test

api:
	.venv/bin/uvicorn backend.app.main:app --reload --port 8000

dashboard:
	.venv/bin/streamlit run app/dashboard.py

compose-up:
	docker compose up -d

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'
