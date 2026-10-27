"""Model inference: heuristic mock (no torch) or the fine-tuned HF model.

The mock exists so the full review pipeline — API, filter, GitHub posting — is
demoable and testable without ML dependencies. It uses a handful of
high-precision regex rules on ADDED lines only, so it rarely false-positives.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

from mining.diff_utils import parse_unified_diff
from training.common import DEFAULT_BASE_MODEL
from training.prompts import review_messages
from training.schemas import ReviewComment, parse_model_output

MOCK_RULES: list[tuple[re.Pattern, tuple[str, str, str]]] = [
    (
        re.compile(r"\b(eval|exec)\s*\("),
        ("security", "high", "Use of eval/exec on dynamic input enables code injection."),
    ),
    (
        re.compile(r"except\s*[^:]*:\s*pass"),
        ("exception_handling", "medium", "Silently swallowed exception; failures will be invisible."),
    ),
    (
        re.compile(r"def\s+\w+\([^)]*=\s*(\[\]|\{\}|None\b[^)]*\))"),
        ("logic", "medium", "Default argument value is shared/mutable across calls or misused as None sentinel."),
    ),
    (
        re.compile(r"[=!]=\s*None"),
        ("null_safety", "low", "Identity comparison with None; use `is None` / `is not None`."),
    ),
    (
        re.compile(r"\bopen\s*\((?![^)]*\bwith\b)"),
        ("resource_leak", "medium", "open() outside a `with` block risks leaking the file handle."),
    ),
]


def _added_lines(diff_text: str) -> list[str]:
    lines = []
    for hunk in parse_unified_diff(diff_text):
        lines.extend(hunk.added[ln] for ln in sorted(hunk.added))
    return lines


class Inference(Protocol):
    name: str

    def review(self, diff_text: str, path: str, context: str | None) -> ReviewComment: ...


class MockInference:
    name = "mock"

    def review(self, diff_text: str, path: str, context: str | None) -> ReviewComment:
        added = "\n".join(_added_lines(diff_text))
        for pattern, (btype, severity, message) in MOCK_RULES:
            if pattern.search(added):
                return ReviewComment(
                    verdict="BUG",
                    type=btype,
                    severity=severity,
                    comment=message,
                    location=path,
                    confidence=0.9,
                )
        return ReviewComment(verdict="CLEAN", confidence=0.8)


class HfInference:
    """Fine-tuned (or base) HF model behind the same interface."""

    def __init__(self, model_name: str | None = None, adapter: str | None = None):
        from training.common import build_model_and_tokenizer, generate_texts

        self._generate_texts = generate_texts
        self.name = "finetuned" if adapter else "base"
        model_name = model_name or os.getenv("PRGUARDIAN_BASE_MODEL", DEFAULT_BASE_MODEL)
        adapter = adapter or os.getenv("ADAPTER_PATH") or None
        self.model, self.tokenizer = build_model_and_tokenizer(model_name, adapter=adapter)

    def review(self, diff_text: str, path: str, context: str | None) -> ReviewComment:
        texts = self._generate_texts(
            self.model, self.tokenizer, [review_messages(diff_text, context)], max_new_tokens=96
        )
        parsed = parse_model_output(texts[0]) if texts else None
        if parsed is None:
            return ReviewComment(verdict="CLEAN", confidence=0.0)  # fail-open, logged upstream
        if parsed.verdict == "BUG" and not parsed.location:
            parsed = parsed.model_copy(update={"location": path})
        return parsed


_inference_cache: dict[str, Inference] = {}


def get_inference(mock: bool | None = None) -> Inference:
    """Factory honoring PRGUARDIAN_MOCK; caches the (expensive) HF model."""
    if mock is None:
        mock = os.getenv("PRGUARDIAN_MOCK", "1") != "0"
    key = "mock" if mock else "hf"
    if key not in _inference_cache:
        _inference_cache[key] = MockInference() if mock else HfInference()
    return _inference_cache[key]
