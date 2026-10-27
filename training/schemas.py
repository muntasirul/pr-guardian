"""Pydantic schemas for schema-constrained model output."""

from __future__ import annotations

import json
import re
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Verdict(str, Enum):
    BUG = "BUG"
    CLEAN = "CLEAN"


class BugType(str, Enum):
    NULL_SAFETY = "null_safety"
    INDEX = "index_error"
    RESOURCE_LEAK = "resource_leak"
    EXCEPTION = "exception_handling"
    SECURITY = "security"
    CONCURRENCY = "concurrency"
    API_MISUSE = "api_misuse"
    LOGIC = "logic"
    OTHER = "other"


BUG_TYPE_ALIASES = {
    "null": "null_safety",
    "none": "null_safety",
    "index": "index_error",
    "off_by_one": "index_error",
    "bounds": "index_error",
    "leak": "resource_leak",
    "memory": "resource_leak",
    "fd": "resource_leak",
    "sec": "security",
    "injection": "security",
    "race": "concurrency",
    "deadlock": "concurrency",
    "api": "api_misuse",
    "misuse": "api_misuse",
}


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewComment(BaseModel):
    verdict: Verdict
    type: BugType | None = None
    severity: Severity | None = None
    comment: str | None = None
    location: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v):
        """Be forgiving about bug-type vocabulary; unknown -> other."""
        if v is None or isinstance(v, BugType):
            return v
        s = str(v).strip().lower()
        s = BUG_TYPE_ALIASES.get(s, s)
        for bt in BugType:
            if s in (bt.value, bt.name.lower()):
                return bt
        return BugType.OTHER

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, v):
        if v is None or isinstance(v, Severity):
            return v
        s = str(v).strip().lower()
        for sev in Severity:
            if s.startswith(sev.value[:3]):
                return sev
        return Severity.MEDIUM

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v):
        if v is None:
            return v
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, f))


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_model_output(text: str) -> ReviewComment | None:
    """Lenient extraction of the JSON object a model was asked to emit."""
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return ReviewComment.model_validate(data)
    except Exception:  # noqa: BLE001 - invalid schema is an expected failure mode
        return None
