import json

from training.build_dataset import (
    build_dataset,
    decontaminate_test,
    diff_hash,
    normalize_diff,
    shingles,
    split_by_repo,
)


def _rec(repo: str, kind: str, body: str, i: int) -> dict:
    return {
        "id": f"{repo}__{i}",
        "repo": repo,
        "kind": kind,
        "source": "szz",
        "path": f"{repo}/f{i}.py",
        "lang": "Python",
        "diff": f"@@ -1,2 +1,2 @@\n-import os\n+import sys\n x = {body}",
        "commit_msg": f"change {i}",
        "fix_msg": f"Fix bug: {body}" if kind == "bug" else None,
        "fix_sha": "a" * 40 if kind == "bug" else None,
        "commit_date": "2022-01-01T00:00:00+00:00",
        "n_evidence_lines": 1 if kind == "bug" else None,
    }


def test_repo_split_no_overlap_and_deterministic():
    repos = [f"acme/r{i}" for i in range(20)]
    a = split_by_repo(repos)
    b = split_by_repo(repos)
    assert a == b
    assert set(a.values()) == {"train", "val", "test"}
    assert sum(1 for v in a.values() if v == "test") >= 1
    assert sum(1 for v in a.values() if v == "train") >= 10


def test_normalize_diff_ignores_hunk_headers():
    d1 = "@@ -1 +1 @@\n-a\n+b\n"
    d2 = "@@ -99 +99 @@\n-a\n+b\n"
    assert normalize_diff(d1) == normalize_diff(d2)
    assert diff_hash(d1) == diff_hash(d2)


def test_build_dataset_repo_split_and_labels():
    raw = (
        [_rec(f"acme/r{i}", "bug", f"bug {i}", j) for i in range(4) for j in range(6)]
        + [_rec(f"acme/r{i+10}", "clean", f"clean {i}", j) for i in range(4) for j in range(6)]
    )
    out = build_dataset(raw, "verdict")
    s = out["stats"]
    assert s["split_mode"] == "repo"
    repos_by_split: dict[str, set] = {}
    for split, recs in out["splits"].items():
        for r in recs:
            repos_by_split.setdefault(split, set()).add(r["meta"]["repo"])
    all_repos = set().union(*repos_by_split.values())
    for r_train in repos_by_split.get("train", set()):
        assert r_train not in repos_by_split.get("test", set())
    assert all_repos == {r["repo"] for r in raw}
    # messages structure: system, user, assistant
    m = out["splits"]["train"][0]["messages"]
    assert [x["role"] for x in m] == ["system", "user", "assistant"]
    assert json.loads(m[-1]["content"])["verdict"] in ("BUG", "CLEAN")


def test_build_dataset_dedups_exact_duplicates():
    raw = [_rec("acme/a", "bug", "same", 1), _rec("acme/a", "bug", "same", 1)]
    raw[1]["id"] = "different-id-same-diff"
    out = build_dataset(raw, "verdict")
    assert out["stats"]["dropped_exact_duplicates"] >= 1


def test_decontaminate_test_drops_near_dups():
    train = [{"messages": [], "meta": {"id": "t1"}, "_diff": "line one\nline two\nline three\n"}]
    near = [{"messages": [], "meta": {"id": "x1"}, "_diff": "line one\nline two\nline three\n"}]
    diff = [{"messages": [], "meta": {"id": "x2"}, "_diff": "totally\ndifferent\ncontent\nhere"}]

    import training.build_dataset as bd

    orig = bd._extract_diff

    def fake_extract(rec):
        return rec["_diff"]

    bd._extract_diff = fake_extract
    try:
        kept, dropped = decontaminate_test(train, near + diff)
    finally:
        bd._extract_diff = orig
    assert dropped == 1
    assert [r["meta"]["id"] for r in kept] == ["x2"]


def test_shingles_identical_texts():
    a = shingles("a\nb\nc\nd\n")
    b = shingles("a\nb\nc\nd\n")
    assert a and a == b
