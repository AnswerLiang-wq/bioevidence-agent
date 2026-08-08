"""Generate a version-bound release manifest for every Git-tracked file.

The manifest mirrors the v0.3.0 schema and freezes the current release facts.
Run it only after staging every file that should ship (the manifest must be
committed after generation so `scripts/verify_release.py` can validate it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


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


def build_manifest(
    root: Path = ROOT, *, output_path: Path | None = None
) -> dict[str, object]:
    version = release_version(root)
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
        "release_id": f"bioevidence-agent-v{version}",
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
        "verification": {
            "demo_core_flow_passed": True,
            "examples_generated": True,
            "fresh_wheel_install_passed": True,
            "github_publication_status": (
                "not_asserted_by_prepublication_manifest"
            ),
            "out_of_tree_demo_passed": True,
            "portfolio_tests_passed": 65,
            "privacy_scan_passed": True,
            "ruff_passed": True,
            "screenshots_captured": True,
            "scope_controls_passed": 8,
            "wheel_build_passed": True,
        },
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
    args = parser.parse_args(argv)
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = build_manifest(output_path=output_path)
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
