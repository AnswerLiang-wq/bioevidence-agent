"""Deterministic scope-policy controls over declared synthetic metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping


SCOPE_DIMENSIONS = ("species", "population", "intervention", "endpoint", "timepoint")
EVIDENCE_ROLES = {"direct", "context_only", "unknown"}
TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def load_scope_controls(path: Path | None = None) -> list[dict[str, object]]:
    source = path or Path(__file__).with_name("fixtures") / "scope_negative_controls_v1.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("scope-control fixture must be a non-empty list")
    controls = [row for row in value if isinstance(row, dict)]
    if len(controls) != len(value):
        raise ValueError("every scope-control row must be an object")
    case_ids = [row.get("case_id") for row in controls]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("scope-control case IDs must be unique")
    return controls


def evaluate_declared_scope(case: Mapping[str, object]) -> dict[str, object]:
    """Flag exact declared-scope differences without interpreting scientific text."""

    case_id = _token(case.get("case_id"), "case_id")
    claim_scope = _scope(case.get("claim_scope"), "claim_scope")
    evidence_scope = _scope(case.get("evidence_scope"), "evidence_scope")
    evidence_role = case.get("evidence_role")
    if evidence_role not in EVIDENCE_ROLES:
        raise ValueError("evidence_role must be direct, context_only, or unknown")

    flags: list[str] = []
    for dimension in SCOPE_DIMENSIONS:
        claim_value = claim_scope[dimension]
        evidence_value = evidence_scope[dimension]
        if "unknown" in {claim_value, evidence_value}:
            flags.append(f"{dimension}_unknown")
        elif claim_value != evidence_value:
            flags.append(f"{dimension}_mismatch")
    if evidence_role != "direct":
        flags.append(f"evidence_role_{evidence_role}")

    return {
        "case_id": case_id,
        "basis": "declared_metadata_only",
        "flags": flags,
        "candidate_for_decisive_human_review": not flags,
        "scientific_judgment": "not_performed",
        "requires_human_review": True,
    }


def _scope(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(SCOPE_DIMENSIONS):
        raise ValueError(f"{label} must contain exactly the frozen scope dimensions")
    return {
        dimension: _token(value[dimension], f"{label}.{dimension}")
        for dimension in SCOPE_DIMENSIONS
    }


def _token(value: object, label: str) -> str:
    if not isinstance(value, str) or TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical snake-case token")
    return value
