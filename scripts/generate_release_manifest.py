"""Generate a version-bound release manifest for every Git-tracked file.

The manifest mirrors the v0.3.0 schema and freezes the current release facts.
Run it only after staging every file that should ship (the manifest must be
committed after generation so `scripts/verify_release.py` can validate it).

The `verification` block is transcribed from an evidence record produced by the
current run (`--evidence`).  Nothing in this script asserts a check passed on
its own; a check absent from the record is reported as not executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10 ships no tomllib
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]

#: Checks the manifest may report.  A check is only ever reported as executed
#: when the evidence record carries it with a recorded exit code.
KNOWN_CHECKS = (
    "ruff",
    "tests",
    "wheel_build",
    "fresh_wheel_install",
    "out_of_tree_demo",
    "public_boundary_scan",
    "examples",
    "demo_core_flow",
    "scope_controls",
    "screenshots",
)

#: Statuses a check may carry in the evidence record.
CHECK_STATUSES = ("executed", "not_executed", "historical")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked_relative_paths(root: Path = ROOT) -> list[str]:
    """Return sorted Git-tracked file paths relative to the repository."""

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    relative_paths = [
        raw.decode("utf-8")
        for raw in result.stdout.split(b"\0")
        if raw
    ]
    return sorted(relative_paths)


def release_version(root: Path = ROOT) -> str:
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = metadata["project"]["version"]
    if not isinstance(version, str) or not version.count(".") == 2:
        raise ValueError(f"invalid release version: {version!r}")
    return version


def default_output_path(root: Path = ROOT) -> Path:
    """Return the manifest path for the version declared by the project."""

    return root / "reports" / f"release_manifest_v{release_version(root)}.json"


def load_evidence(path: Path) -> dict[str, object]:
    """Read the verification record produced by the current run.

    The record is the only source for the manifest's verification block.  A
    check it does not carry is reported as not executed, never as passed, so a
    manifest cannot claim a verification this run did not perform.

    The record must also name the release it verified.  Binding that identity
    here is what stops a record made for an earlier version from being
    transcribed into a later version's manifest.
    """

    if not path.is_file():
        raise FileNotFoundError(f"verification evidence record is missing: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("verification evidence record must be a JSON object")
    release = record.get("release")
    if not isinstance(release, str) or not release.strip():
        raise ValueError(
            "verification evidence record has no release identity; refusing to "
            "attribute unversioned evidence to a release"
        )
    checks = record.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("verification evidence record has no checks")
    seen: dict[str, dict[str, object]] = {}
    for entry in checks:
        if not isinstance(entry, dict):
            raise ValueError("each evidence check must be a JSON object")
        check_id = entry.get("id")
        if check_id not in KNOWN_CHECKS:
            raise ValueError(f"unknown evidence check id: {check_id!r}")
        if check_id in seen:
            raise ValueError(f"duplicate evidence check id: {check_id!r}")
        status = entry.get("status")
        if status not in CHECK_STATUSES:
            raise ValueError(f"invalid status for {check_id!r}: {status!r}")
        if status == "executed" and not isinstance(entry.get("exit_code"), int):
            raise ValueError(f"executed check {check_id!r} must record an exit code")
        seen[check_id] = entry
    return {"record": record, "release": release, "checks": seen}


def build_verification(evidence: dict[str, object]) -> dict[str, object]:
    """Turn the evidence record into the manifest's verification block."""

    checks: dict[str, object] = evidence["checks"]  # type: ignore[assignment]
    executed: dict[str, object] = {}
    not_executed: list[str] = []
    historical: dict[str, object] = {}
    for check_id in KNOWN_CHECKS:
        entry = checks.get(check_id)
        if entry is None or entry.get("status") == "not_executed":
            not_executed.append(check_id)
        elif entry.get("status") == "historical":
            historical[check_id] = {
                key: entry[key]
                for key in ("note", "source", "as_of")
                if key in entry
            }
        else:
            executed[check_id] = {
                key: entry[key]
                for key in ("command", "exit_code", "passed", "failed", "skipped")
                if key in entry
            }
    return {
        "basis": (
            "transcribed from the run record captured before this manifest was "
            "generated; not an independent assertion by the generator"
        ),
        "checks_executed_this_round": executed,
        "checks_not_executed_this_round": not_executed,
        "historical_references": historical,
        "github_publication_status": "not_asserted_by_prepublication_manifest",
    }


def build_manifest(
    root: Path = ROOT,
    *,
    output_path: Path | None = None,
    evidence: dict[str, object],
) -> dict[str, object]:
    version = release_version(root)
    release_id = f"bioevidence-agent-v{version}"
    evidence_release = evidence.get("release")
    if evidence_release != release_id:
        raise ValueError(
            "verification evidence does not belong to this release: "
            f"record claims {evidence_release!r}, target is {release_id!r}"
        )
    wheel = f"artifacts/bioevidence_agent-{version}-py3-none-any.whl"
    wheel_path = root.joinpath(*PurePosixPath(wheel).parts)
    if not wheel_path.is_file():
        raise FileNotFoundError(f"release wheel is missing: {wheel_path}")

    # The manifest cannot hash itself (self-reference has no fixed point),
    # mirroring the v0.3.0 manifest, which also omitted its own entry.
    manifest_path = (output_path or default_output_path(root)).resolve()
    artifacts: dict[str, str] = {}
    for relative in tracked_relative_paths(root):
        path = root.joinpath(*PurePosixPath(relative).parts)
        if path.resolve() == manifest_path:
            continue
        artifacts[relative] = sha256_file(path)
    artifacts[wheel] = sha256_file(wheel_path)

    return {
        "manifest_version": "1.0.0",
        "release_id": release_id,
        "release_scope": (
            "solo public portfolio closeout release; local source-bound Product "
            "Demo, deterministic engineering diagnostics, and synthetic scope "
            "controls - not blind or clinical validation and not user-value "
            "evidence"
        ),
        "claims_not_made": [
            "private blind performance",
            "open-world retrieval completeness",
            "semantic hallucination rate",
            "clinical validity or medical advice",
            "validated user value, time saving, task completion, trust, reuse "
            "intent, or VEPS improvement",
        ],
        "excluded": [
            "Day27-63 logs and archival governance/ACL workspaces",
            "raw PubMedQA files and test labels",
            "model weights and caches",
            "dense embeddings and local run rankings",
            "secrets, temporary files, and local absolute paths",
            "raw PILOT01 participant records and row-level aggregates (kept "
            "under ignored private/)",
        ],
        "frozen_results": {
            "abstract_only": {
                "rank_drop_count": 12,
                "reranked_mrr": 0.982,
                "reranked_recall_at_10": 0.99,
                "test_cases": 500,
            },
            "evidence_controls": {
                "correct_context_accuracy": 0.536,
                "exact_mcnemar_p": 0.06614291,
                "non_test_oof_cases": 500,
                "shuffled_context_accuracy": 0.486,
            },
            "title_assisted": {
                "answer_accuracy": 0.558,
                "answer_macro_f1": 0.383221,
                "reranked_recall_at_10": 1.0,
                "test_cases": 500,
            },
        },
        "package": {
            "version": version,
            "wheel": wheel,
            "wheel_sha256": sha256_file(wheel_path),
        },
        "verification": build_verification(evidence),
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    default_output = default_output_path()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="manifest output path (default: %(default)s)",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help=(
            "JSON record of the checks this run actually executed; the sole "
            "source for the manifest verification block"
        ),
    )
    args = parser.parse_args(argv)
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    evidence_path = (
        args.evidence if args.evidence.is_absolute() else ROOT / args.evidence
    )
    try:
        evidence = load_evidence(evidence_path)
        manifest = build_manifest(output_path=output_path, evidence=evidence)
    except (FileNotFoundError, ValueError) as error:
        # Refuse before writing: a rejected record must not create or replace
        # an existing manifest.
        print(f"refusing to write manifest: {error}", file=sys.stderr)
        return 2
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "release_id": manifest["release_id"],
                "artifact_count": len(manifest["artifacts"]),
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
