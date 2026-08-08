from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from apps.product_demo.service import PRODUCT_VERSION
from bioevidence import __version__
from bioevidence.pubmed import USER_AGENT
from scripts.verify_release import (
    validate_public_text,
    validate_relative_path,
    verify_release,
)


ROOT = Path(__file__).resolve().parents[1]


def test_public_version_is_consistent() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    page = (ROOT / "apps/product_demo/static/index.html").read_text(
        encoding="utf-8"
    )

    project_metadata = project.split("[project]", 1)[1].split("[", 1)[0]
    assert '\nversion = "0.4.1"\n' in project_metadata
    assert __version__ == PRODUCT_VERSION == "0.4.1"
    assert USER_AGENT.startswith("BioEvidenceAgent/0.4.1 ")
    assert "\nversion: 0.4.1\n" in citation
    assert "BioEvidence Agent v0.4.1" in page


def test_release_path_validator_accepts_public_relative_path() -> None:
    assert validate_relative_path("reports/example.json").as_posix() == (
        "reports/example.json"
    )


@pytest.mark.parametrize(
    "relative",
    (
        "private/pilot.json",
        "apps/product_demo/session_data/events.jsonl",
        "../outside.md",
        "/" + "tmp/outside.md",
    ),
)
def test_release_path_validator_rejects_nonpublic_paths(relative: str) -> None:
    with pytest.raises(ValueError):
        validate_relative_path(relative)


@pytest.mark.parametrize(
    "value",
    (
        "/" + "Users/example/project/file.md",
        "/" + "tmp/example.json",
        "C:" + "\\Users\\example\\file.md",
        "file:" + "///" + "Users/example/file.md",
    ),
)
def test_public_text_validator_rejects_local_absolute_paths(value: str) -> None:
    with pytest.raises(ValueError):
        validate_public_text("example.md", value)


@pytest.mark.parametrize(
    "value",
    (
        "<" * 7 + " ours",
        "=" * 7,
        ">" * 7 + " theirs",
    ),
)
def test_public_text_validator_rejects_merge_conflicts(value: str) -> None:
    with pytest.raises(ValueError):
        validate_public_text("example.py", value)


def test_release_verifier_rejects_unmanifested_tracked_file(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "artifact.whl"
    wheel.write_bytes(b"wheel")
    omitted = tmp_path / "omitted.txt"
    omitted.write_text("tracked but absent from manifest\n", encoding="utf-8")
    wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "release_id": "bioevidence-agent-v0.4.1",
                "package": {
                    "version": "0.4.1",
                    "wheel": "artifact.whl",
                    "wheel_sha256": wheel_hash,
                },
                "artifacts": {"artifact.whl": wheel_hash},
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match="artifact set does not match"):
        verify_release(manifest, root=tmp_path)
