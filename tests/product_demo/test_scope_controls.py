from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.product_demo.scope_controls import (
    SCOPE_DIMENSIONS,
    evaluate_declared_scope,
    load_scope_controls,
)
from scripts import run_product_scope_controls

CONTROLS = load_scope_controls()


@pytest.mark.parametrize("case", CONTROLS, ids=lambda row: str(row["case_id"]))
def test_declared_scope_controls_fail_closed(case: dict[str, object]) -> None:
    result = evaluate_declared_scope(case)
    assert result["flags"] == case["expected_flags"]
    assert (
        result["candidate_for_decisive_human_review"]
        is case["expected_candidate_for_decisive_human_review"]
    )
    assert result["basis"] == "declared_metadata_only"
    assert result["scientific_judgment"] == "not_performed"
    assert result["requires_human_review"] is True
    assert not {"verdict", "direction", "supported", "confidence"} & set(result)


def test_scope_fixture_is_synthetic_and_frozen() -> None:
    assert len(CONTROLS) == 8
    assert sum(
        bool(case["expected_candidate_for_decisive_human_review"])
        for case in CONTROLS
    ) == 1
    serialized = json.dumps(CONTROLS, sort_keys=True).lower()
    assert "pmid" not in serialized
    assert "doi" not in serialized
    assert {case["case_id"] for case in CONTROLS} == {
        "direct_scope_match",
        "population_mismatch",
        "species_mismatch",
        "intervention_mismatch",
        "endpoint_mismatch",
        "timepoint_mismatch",
        "context_only_not_decisive",
        "scope_unknown_not_decisive",
    }
    for case in CONTROLS:
        assert set(case["claim_scope"]) == set(SCOPE_DIMENSIONS)
        assert set(case["evidence_scope"]) == set(SCOPE_DIMENSIONS)


def test_invalid_or_ambiguous_declared_scope_is_rejected() -> None:
    invalid = dict(CONTROLS[0])
    invalid["evidence_role"] = "supports"
    with pytest.raises(ValueError, match="evidence_role"):
        evaluate_declared_scope(invalid)

    incomplete = dict(CONTROLS[0])
    incomplete["evidence_scope"] = {"species": "human"}
    with pytest.raises(ValueError, match="frozen scope dimensions"):
        evaluate_declared_scope(incomplete)


def test_report_command_uses_frozen_paths_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_json = tmp_path / "scope.json"
    output_markdown = tmp_path / "scope.md"
    monkeypatch.setattr(
        run_product_scope_controls, "DEFAULT_OUTPUT_JSON", output_json
    )
    monkeypatch.setattr(
        run_product_scope_controls, "DEFAULT_OUTPUT_MARKDOWN", output_markdown
    )

    assert run_product_scope_controls.main([]) == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["passed_count"] == 8
    assert "Controls passed: 8/8" in output_markdown.read_text(encoding="utf-8")
