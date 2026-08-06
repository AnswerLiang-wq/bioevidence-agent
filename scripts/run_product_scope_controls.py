"""Generate the frozen synthetic product scope-control report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.product_demo.scope_controls import (  # noqa: E402
    evaluate_declared_scope,
    load_scope_controls,
)


CLAIMS_NOT_MADE = [
    "No biomedical natural-language-inference accuracy is measured.",
    "No automatic population, species, intervention, endpoint, or time extraction is measured.",
    "No real-paper scope judgment is validated.",
    "No clinical effectiveness or safety conclusion is produced.",
    "No open-world mismatch-detection performance is claimed.",
]


def build_report() -> dict[str, object]:
    controls = load_scope_controls()
    results: list[dict[str, object]] = []
    for case in controls:
        result = evaluate_declared_scope(case)
        result["expected_flags"] = case["expected_flags"]
        result["expected_candidate_for_decisive_human_review"] = case[
            "expected_candidate_for_decisive_human_review"
        ]
        result["passed"] = (
            result["flags"] == result["expected_flags"]
            and result["candidate_for_decisive_human_review"]
            is result["expected_candidate_for_decisive_human_review"]
        )
        results.append(result)
    return {
        "report_version": "product-scope-controls-v1",
        "basis": "synthetic_declared_metadata",
        "control_count": len(results),
        "passed_count": sum(result["passed"] is True for result in results),
        "failed_count": sum(result["passed"] is not True for result in results),
        "scientific_judgment": "not_performed",
        "claims_not_made": CLAIMS_NOT_MADE,
        "results": results,
    }


def render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Product scope-control report",
        "",
        f"- Version: `{report['report_version']}`",
        f"- Controls passed: {report['passed_count']}/{report['control_count']}",
        "- Basis: synthetic, pre-declared metadata only",
        "- Scientific judgment: not performed",
        "",
        "| Control | Flags | Decisive-review candidate | Result |",
        "|---|---|---:|---:|",
    ]
    for result in report["results"]:
        flags = ", ".join(result["flags"]) or "none"
        candidate = str(result["candidate_for_decisive_human_review"]).lower()
        passed = "pass" if result["passed"] else "fail"
        lines.append(f"| {result['case_id']} | {flags} | {candidate} | {passed} |")
    lines.extend(["", "## Claims not made", ""])
    lines.extend(f"- {claim}" for claim in report["claims_not_made"])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "control_count": report["control_count"],
                "passed_count": report["passed_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
