from __future__ import annotations

from pathlib import Path

import pytest

from apps.product_demo.service import PRODUCT_VERSION
from bioevidence import __version__
from bioevidence.pubmed import USER_AGENT
from scripts.verify_release import validate_public_text, validate_relative_path


ROOT = Path(__file__).resolve().parents[1]


def test_public_version_is_consistent() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    page = (ROOT / "apps/product_demo/static/index.html").read_text(
        encoding="utf-8"
    )

    project_metadata = project.split("[project]", 1)[1].split("[", 1)[0]
    assert '\nversion = "0.4.0"\n' in project_metadata
    assert __version__ == PRODUCT_VERSION == "0.4.0"
    assert USER_AGENT.startswith("BioEvidenceAgent/0.4.0 ")
    assert "\nversion: 0.4.0\n" in citation
    assert "BioEvidence Agent v0.4.0" in page


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
