"""Prompt templates shared by training, inference, and baselines.

Keeping one prompt builder means the zero-shot baselines and the fine-tuned
model see byte-identical inputs — a fair comparison (the eval harness depends
on this).
"""

from __future__ import annotations

import json

from training.schemas import BugType, Severity

SCHEMA_INSTRUCTION = (
    'Respond ONLY with a JSON object in this exact shape: {"verdict": "BUG" or "CLEAN", '
    '"type": one of null_safety, index_error, resource_leak, exception_handling, '
    "security, concurrency, api_misuse, logic, other, "
    '"severity": one of low, medium, high, critical, '
    '"comment": short description of the problem (null when CLEAN), '
    '"location": file path or null, "confidence": number 0.0-1.0}. '
    "Output no text other than the JSON object."
)

SYSTEM_VERDICT = (
    "You are a meticulous senior code reviewer. You examine code diffs for bugs "
    "that will break behavior at runtime: null/None hazards, index errors, "
    "resource leaks, unhandled exceptions, security issues, concurrency bugs, "
    "API misuse, and logic errors. Style, naming, and refactoring suggestions "
    "are out of scope. Be precise: report BUG only when the diff itself "
    "introduces a real defect. " + SCHEMA_INSTRUCTION
)

SYSTEM_REVIEW = (
    SYSTEM_VERDICT
    + " When you report BUG, the comment must state the concrete failure mode "
    "and, when possible, the input that triggers it."
)


def build_user_prompt(diff_text: str, context: str | None = None) -> str:
    parts = ["Review the following diff and decide whether it introduces a bug."]
    if context:
        parts.append("Related repository code for context:\n```\n" + context + "\n```")
    parts.append("Diff:\n```diff\n" + diff_text + "\n```")
    return "\n\n".join(parts)


def verdict_messages(diff_text: str, context: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_VERDICT},
        {"role": "user", "content": build_user_prompt(diff_text, context)},
    ]


def review_messages(diff_text: str, context: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_REVIEW},
        {"role": "user", "content": build_user_prompt(diff_text, context)},
    ]


def target_answer(record: dict, task: str = "verdict") -> str:
    """Build the assistant-side training target from a mined record.

    verdict task: minimal JSON (cheap inference).
    review task: full JSON with a comment (the generative goal).
    """
    if record.get("kind") == "bug":
        comment = record.get("review_comment") or _clean_fix_msg(record.get("fix_msg"))
        if task == "verdict":
            return json.dumps({"verdict": "BUG", "confidence": 0.9})
        return json.dumps(
            {
                "verdict": "BUG",
                "type": (record.get("bug_type") or "other").lower(),
                "severity": (record.get("severity") or "medium").lower(),
                "comment": comment,
                "location": record.get("path"),
                "confidence": 0.9,
            }
        )
    if task == "verdict":
        return json.dumps({"verdict": "CLEAN", "confidence": 0.9})
    return json.dumps(
        {"verdict": "CLEAN", "comment": None, "location": None, "confidence": 0.9}
    )


def _clean_fix_msg(msg: str | None) -> str:
    """Turn a fix commit subject into a reviewer-style comment (fallback when
    the LLM-as-labeler pass has not been run)."""
    if not msg:
        return "This change introduces a defect."
    s = msg.strip()
    for prefix in (
        "fix:",
        "fix bug:",
        "bugfix:",
        "fixes:",
        "fixed:",
        "hotfix:",
        "fix ",
        "bug ",
    ):
        if s.lower().startswith(prefix):
            s = s[len(prefix):].strip()
            break
    s = s.split(", fixes #")[0].split(" fixes #")[0].strip()
    s = s.rstrip(".")
    if not s:
        return "This change introduces a defect."
    return s[0].upper() + s[1:] + "."


def all_bug_types() -> list[str]:
    return [bt.value for bt in BugType]


def all_severities() -> list[str]:
    return [s.value for s in Severity]
