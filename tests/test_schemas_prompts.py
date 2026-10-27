import json

from training.prompts import _clean_fix_msg, build_user_prompt, target_answer, verdict_messages
from training.schemas import parse_model_output


def test_parse_clean_json():
    rc = parse_model_output('{"verdict": "BUG", "type": "null_safety", "confidence": 0.8}')
    assert rc is not None
    assert rc.verdict.value == "BUG"
    assert rc.type.value == "null_safety"
    assert rc.confidence == 0.8


def test_parse_json_with_surrounding_text():
    rc = parse_model_output('Sure! Here is the review:\n{"verdict": "CLEAN", "confidence": 0.7}\nDone.')
    assert rc is not None
    assert rc.verdict.value == "CLEAN"


def test_lenient_type_coercion():
    assert parse_model_output('{"verdict": "BUG", "type": "race"}').type.value == "concurrency"
    assert parse_model_output('{"verdict": "BUG", "type": "made_up_thing"}').type.value == "other"
    assert parse_model_output('{"verdict": "BUG", "type": null}').type is None


def test_lenient_severity_coercion():
    assert parse_model_output('{"verdict": "BUG", "severity": "high"}').severity.value == "high"
    assert parse_model_output('{"verdict": "BUG", "severity": 3}').severity is not None


def test_confidence_clamp():
    assert parse_model_output('{"verdict": "BUG", "confidence": 12}').confidence == 1.0
    assert parse_model_output('{"verdict": "BUG", "confidence": "not-a-number"}').confidence is None


def test_parse_invalid_returns_none():
    assert parse_model_output("no json here at all") is None
    assert parse_model_output('{"verdict": "MAYBE"}') is None
    assert parse_model_output("") is None


def test_target_answer_verdict_task():
    bug = {"kind": "bug", "fix_msg": "Fix bug: crash on empty list, fixes #5", "path": "a.py"}
    assert json.loads(target_answer(bug, "verdict")) == {"verdict": "BUG", "confidence": 0.9}
    clean = {"kind": "clean"}
    assert json.loads(target_answer(clean, "verdict")) == {"verdict": "CLEAN", "confidence": 0.9}


def test_target_answer_review_task_uses_review_comment():
    bug = {
        "kind": "bug",
        "fix_msg": "Fix bug: crash on empty list",
        "path": "a.py",
        "review_comment": "Calling items[-1] raises IndexError when items is empty.",
    }
    tgt = json.loads(target_answer(bug, "review"))
    assert tgt["verdict"] == "BUG"
    assert tgt["comment"].startswith("Calling items[-1]")
    assert tgt["location"] == "a.py"


def test_clean_fix_msg_fallback():
    assert _clean_fix_msg("Fix bug: crash on empty list, fixes #5") == "Crash on empty list."
    assert _clean_fix_msg(None) == "This change introduces a defect."
    assert _clean_fix_msg("fix: division by zero") == "Division by zero."


def test_verdict_messages_structure():
    msgs = verdict_messages("+ x = 1", context="def f(): pass")
    assert msgs[0]["role"] == "system"
    assert "```diff" in msgs[1]["content"]
    assert "Related repository code" in msgs[1]["content"]
    assert build_user_prompt("+ x = 1").endswith("```")
