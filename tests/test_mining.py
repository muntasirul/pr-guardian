import json
from pathlib import Path

import pytest

from mining.git_utils import blame_lines, run_git
from mining.mine_clean_diffs import _parse_iso, mine_clean_diffs
from mining.mine_fix_commits import fix_score, mine_fix_commits
from mining.run_mining import record_from_bug, record_from_clean
from mining.szz_introducing import find_introducing_commits
from tests.git_fixture import build_scenario


@pytest.fixture()
def scenario(tmp_path: Path):
    return build_scenario(tmp_path)


def test_fix_score():
    assert fix_score("Fix bug: floor division broke negative operands, fixes #12") >= 2
    assert fix_score("Add feature X") == 0
    assert fix_score("use floor division") == 0
    assert fix_score("fixup! fix typo") == 0


def test_mine_fix_commits_finds_only_the_fix(scenario):
    fixes = mine_fix_commits(Path(scenario["repo"]))
    assert len(fixes) == 1
    fix = fixes[0]
    assert fix.sha == scenario["c3"]
    assert fix.files[0]["path"] == "app.py"
    assert fix.files[0]["changed"] == 2  # one removed + one added line


def test_szz_maps_fix_to_introducing_commit(scenario):
    repo = Path(scenario["repo"])
    fixes = mine_fix_commits(repo)
    bugs = find_introducing_commits(repo, fixes[0])
    assert len(bugs) == 1
    bug = bugs[0]
    assert bug.introducing_sha == scenario["c2"]
    assert bug.fix_sha == scenario["c3"]
    assert bug.n_evidence_lines == 1
    assert "a // b" in bug.introducing_diff


def test_blame_lines_maps_parent_version(scenario):
    repo = Path(scenario["repo"])
    blame = blame_lines(repo, f"{scenario['c3']}~1", "app.py")
    assert blame[1] == scenario["c1"]
    assert blame[2] == scenario["c2"]


def test_mine_clean_diffs_survival(scenario):
    repo = Path(scenario["repo"])
    # app.py is contaminated by fixes; only stable.py should survive
    cleans = mine_clean_diffs(repo, contaminated_paths={"app.py"})
    assert len(cleans) == 1
    c = cleans[0]
    assert c.sha == scenario["c1"]
    assert c.path == "stable.py"
    assert "VALUE = 1" in c.diff


def test_mine_clean_diffs_respects_contamination(scenario):
    repo = Path(scenario["repo"])
    assert mine_clean_diffs(repo, contaminated_paths={"app.py", "stable.py"}) == []


def test_record_serialization(scenario):
    repo = Path(scenario["repo"])
    fixes = mine_fix_commits(repo)
    bug = find_introducing_commits(repo, fixes[0])[0]
    rec = record_from_bug("acme/toyproj", bug)
    assert rec["kind"] == "bug"
    assert rec["source"] == "szz"
    assert rec["fix_msg"].startswith("Fix bug")
    json.dumps(rec)  # must be JSON-serializable

    cleans = mine_clean_diffs(repo, {"app.py"})
    rec2 = record_from_clean("acme/toyproj", cleans[0])
    assert rec2["kind"] == "clean"
    assert rec2["fix_sha"] is None
    json.dumps(rec2)


def test_parse_iso_tolerates_malformed_git_date():
    # real-world case: psf/requests has a 2011 commit with offset '+518:00'
    d = _parse_iso("2011-09-08T02:38:50+518:00")
    assert d.year == 2011
    d2 = _parse_iso("2024-06-01T12:00:00+00:00")
    assert d2.year == 2024


def test_commit_message_helper(scenario):
    repo = Path(scenario["repo"])
    msg = run_git(repo, "log", "-1", "--format=%s", scenario["c3"]).strip()
    assert msg.startswith("Fix bug:")
