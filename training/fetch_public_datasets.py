"""Download public datasets used by PR Guardian into data/raw/.

Sources (all public):
1. Devign / CodeXGLUE defect detection — function-level buggy/clean C code.
   Used as a seed/warmup task and to make the Kaggle notebook runnable even
   before local mining produces diffs.
2. SWE-bench Lite — real GitHub issue/patch pairs for the hard-mode eval slice.

Everything is best-effort: a failed source logs a warning and the script moves
on, so partial downloads still leave a usable repo. A manifest records what was
fetched, from where, and how many rows.

    python -m training.fetch_public_datasets
    python -m training.fetch_public_datasets --only devign swebench
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

RAW = Path("data/raw")

DEVIGN_HF_CANDIDATES = [
    "google/code_x_glue_cc_defect_detection",  # official CodeXGLUE mirror
    "DetectVul/devign",
    "claudios/code_x_glue_devign",
]
BIGVUL_HF_CANDIDATES = [
    "bstee615/bigvul",
    "DynaOuchebara/BigVul",
    "mahdin70/cleaned_bigvul",
]
SWEBENCH_HF = "princeton-nlp/SWE-bench_Lite"


def _http_get(url: str, dest: Path) -> bool:
    import requests

    try:
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] GET {url}: {e}")
        return False


def _pseudo_diff(code: str) -> str:
    """Wrap whole-function code as a '+'-only pseudo diff so downstream
    prompt-building (which expects a diff) stays uniform."""
    return "\n".join("+" + ln for ln in code.splitlines())


def _devign_record(row: dict, split: str, i: int) -> dict:
    code = row.get("func") or row.get("code") or ""
    target = row.get("target", row.get("label", 0))
    return {
        "id": f"devign__{split}__{i}",
        "repo": f"devign/{split}",
        "kind": "bug" if int(target) == 1 else "clean",
        "source": "devign",
        "path": f"function_{i}.c",
        "lang": "C",
        "diff": _pseudo_diff(code),
        "commit_msg": "",
        "commit_date": None,
        "fix_sha": None,
        "fix_msg": None,
        "n_evidence_lines": None,
    }


def fetch_devign() -> dict:
    """Try Hugging Face mirrors of Devign / CodeXGLUE defect detection."""
    print("[devign] trying Hugging Face datasets ...")
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi

        candidates = list(DEVIGN_HF_CANDIDATES)
        try:
            candidates += [d.id for d in HfApi().list_datasets(search="devign", limit=10)]
        except Exception:  # noqa: BLE001
            pass
        seen: set[str] = set()
        for ds_id in candidates:
            if ds_id in seen:
                continue
            seen.add(ds_id)
            try:
                ds = load_dataset(ds_id)
            except Exception as e:  # noqa: BLE001 - gated/private/broken all land here
                print(f"  [skip] {ds_id}: {str(e)[:120]}")
                continue
            records = []
            for split in ds.keys():
                cols = ds[split].column_names
                code_col = next((c for c in ("func", "code", "function") if c in cols), None)
                label_col = next((c for c in ("target", "label") if c in cols), None)
                if not code_col or label_col is None:
                    print(f"  [skip] {ds_id}: no func/target columns ({cols})")
                    break
                for i, row in enumerate(ds[split]):
                    records.append(_devign_record(row, split, i))
            if not records:
                continue
            dest = RAW / "devign_records.jsonl"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
            print(f"  [ok] {ds_id} -> {dest} ({len(records)} rows)")
            return {"name": "devign", "ok": True, "source": f"hf:{ds_id}", "rows": len(records)}
    except ImportError as e:
        print(f"  [warn] datasets/huggingface_hub not installed: {e}")
    return {"name": "devign", "ok": False, "source": None, "rows": 0}


def fetch_swebench() -> dict:
    print("[swebench] fetching SWE-bench Lite ...")
    try:
        from datasets import load_dataset

        ds = load_dataset(SWEBENCH_HF)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] {e}")
        return {"name": "swebench_lite", "ok": False, "source": SWEBENCH_HF, "rows": 0}
    dest_dir = RAW / "swebench_lite"
    dest_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in ds.keys():
        path = dest_dir / f"{split}.jsonl"
        with path.open("w") as f:
            for row in ds[split]:
                f.write(json.dumps(dict(row)) + "\n")
                total += 1
        print(f"  [ok] {split}: {len(ds[split])} rows -> {path}")
    return {"name": "swebench_lite", "ok": True, "source": SWEBENCH_HF, "rows": total}


def fetch_bigvul() -> dict:
    """Best-effort: find an accessible Big-Vul mirror on HF; skip gracefully."""
    print("[bigvul] searching Hugging Face for a Big-Vul mirror ...")
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi

        ids = list(BIGVUL_HF_CANDIDATES)
        ids += [d.id for d in HfApi().list_datasets(search="big-vul", limit=10)]
        ids += [d.id for d in HfApi().list_datasets(search="bigvul", limit=10)]
        seen: set[str] = set()
        for ds_id in ids:
            if ds_id in seen:
                continue
            seen.add(ds_id)
            try:
                ds = load_dataset(ds_id, split="train")
            except Exception:  # noqa: BLE001
                continue
            cols = ds.column_names
            code_col = next((c for c in ("func", "function", "code") if c in cols), None)
            label_col = next((c for c in ("vul", "label", "target") if c in cols), None)
            if not code_col or label_col is None:
                print(f"  [skip] {ds_id}: unexpected columns {cols[:8]}")
                continue
            out = RAW / "bigvul_records.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            n = 0
            with out.open("w") as f:
                for i, row in enumerate(ds):
                    if not row.get(code_col):
                        continue
                    f.write(
                        json.dumps(
                            {
                                "id": f"bigvul__{i}",
                                "repo": "bigvul",
                                "kind": "bug" if int(row[label_col]) == 1 else "clean",
                                "source": "bigvul",
                                "path": f"function_{i}.c",
                                "lang": "C",
                                "diff": _pseudo_diff(row[code_col]),
                                "commit_msg": row.get("commit_message", ""),
                                "commit_date": None,
                                "fix_sha": None,
                                "fix_msg": None,
                                "n_evidence_lines": None,
                            }
                        )
                        + "\n"
                    )
                    n += 1
            print(f"  [ok] {ds_id} -> {out} ({n} rows)")
            return {"name": "bigvul", "ok": True, "source": f"hf:{ds_id}", "rows": n}
        print("  [warn] no accessible mirror found; Big-Vul is optional — skipping")
    except ImportError as e:
        print(f"  [warn] {e}")
    return {"name": "bigvul", "ok": False, "source": None, "rows": 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None, help="subset of: devign swebench bigvul")
    args = ap.parse_args()

    wanted = args.only or ["devign", "swebench", "bigvul"]
    results = []
    if "devign" in wanted:
        results.append(fetch_devign())
    if "swebench" in wanted:
        results.append(fetch_swebench())
    if "bigvul" in wanted:
        results.append(fetch_bigvul())

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "license_notes": (
            "Devign/CodeXGLUE: check https://github.com/microsoft/CodeXGLUE license. "
            "SWE-bench: MIT (https://github.com/SWE-bench/SWE-bench). "
            "Keep raw data out of git; this script reproduces data/raw."
        ),
    }
    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / "fetch_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    ok = [f"{r['name']}={r['rows']}" for r in results if r["ok"]]
    print(f"\nmanifest -> {RAW / 'fetch_manifest.json'}")
    print(f"fetched: {', '.join(ok) if ok else 'nothing (all sources failed)'}")


if __name__ == "__main__":
    main()
