from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pytest

from apps.product_demo.service import PRODUCT_VERSION
from bioevidence import __version__
from bioevidence.pubmed import USER_AGENT
from scripts.generate_release_manifest import (
    build_manifest,
    load_evidence,
)
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
    assert '\nversion = "0.5.0"\n' in project_metadata
    assert __version__ == PRODUCT_VERSION == "0.5.0"
    assert USER_AGENT.startswith("BioEvidenceAgent/0.5.0 ")
    assert "\nversion: 0.5.0\n" in citation
    assert "BioEvidence Agent v0.5.0" in page


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


TEMP_VERSION = "9.9.9"
TEMP_RELEASE_ID = f"bioevidence-agent-v{TEMP_VERSION}"


def _release_repo(tmp_path: Path) -> Path:
    """A throwaway Git repo that looks like a release root."""

    repo = tmp_path / "release"
    (repo / "artifacts").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "bioevidence-agent"\nversion = "{TEMP_VERSION}"\n',
        encoding="utf-8",
    )
    (repo / "artifacts" / f"bioevidence_agent-{TEMP_VERSION}-py3-none-any.whl").write_bytes(
        b"wheel"
    )
    (repo / "content.txt").write_text("shipped\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    return repo


def _evidence_record(tmp_path: Path, *, release: object = TEMP_RELEASE_ID) -> Path:
    checks: list[dict[str, object]] = [
        {"id": "ruff", "status": "executed", "exit_code": 0},
        {"id": "tests", "status": "executed", "exit_code": 0, "passed": 12,
         "failed": 0, "skipped": 0},
        {"id": "examples", "status": "not_executed"},
        {"id": "screenshots", "status": "historical", "note": "earlier capture"},
    ]
    record: dict[str, object] = {"record_version": "1.0.0", "checks": checks}
    if release is not None:
        record["release"] = release
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_release_manifest_generator_accepts_matching_evidence(
    tmp_path: Path,
) -> None:
    repo = _release_repo(tmp_path)
    evidence = load_evidence(_evidence_record(tmp_path))

    manifest = build_manifest(root=repo, evidence=evidence)

    assert manifest["release_id"] == TEMP_RELEASE_ID
    assert manifest["package"]["version"] == TEMP_VERSION
    assert "content.txt" in manifest["artifacts"]


def test_release_manifest_generator_rejects_stale_evidence_release(
    tmp_path: Path,
) -> None:
    repo = _release_repo(tmp_path)
    stale = _evidence_record(tmp_path, release="bioevidence-agent-v0.4.1")

    with pytest.raises(ValueError, match="does not belong to this release"):
        build_manifest(root=repo, evidence=load_evidence(stale))


def test_release_manifest_generator_rejects_missing_evidence_identity(
    tmp_path: Path,
) -> None:
    missing = _evidence_record(tmp_path, release=None)

    with pytest.raises(ValueError, match="no release identity"):
        load_evidence(missing)


def test_release_manifest_generator_does_not_claim_unexecuted_checks(
    tmp_path: Path,
) -> None:
    repo = _release_repo(tmp_path)
    evidence = load_evidence(_evidence_record(tmp_path))

    verification = build_manifest(root=repo, evidence=evidence)["verification"]

    executed = verification["checks_executed_this_round"]
    assert set(executed) == {"ruff", "tests"}
    assert executed["tests"]["passed"] == 12
    assert "examples" in verification["checks_not_executed_this_round"]
    assert verification["historical_references"]["screenshots"]["note"] == (
        "earlier capture"
    )
    assert "screenshots" not in executed
    assert not any(
        value is True for value in verification.values() if not isinstance(value, dict)
    )


def test_release_manifest_cli_refuses_stale_evidence_without_writing(
    tmp_path: Path,
) -> None:
    stale = _evidence_record(tmp_path, release="bioevidence-agent-v0.4.1")
    output = tmp_path / "manifest.json"
    output.write_text("previous manifest\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_release_manifest.py"),
            "--evidence",
            str(stale),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "refusing to write manifest" in result.stderr
    assert output.read_text(encoding="utf-8") == "previous manifest\n"


def _release_wheel_path() -> Path:
    """The committed release wheel for the version declared by the project."""

    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    metadata = project.split("[project]", 1)[1].split("[", 1)[0]
    version = __version__
    assert f'\nversion = "{version}"\n' in metadata
    return ROOT / "artifacts" / f"bioevidence_agent-{version}-py3-none-any.whl"


def test_release_wheel_long_description_matches_readme() -> None:
    """The committed wheel must ship the README it was built from.

    A wheel built before the README was finalised embeds the older text in its
    METADATA long description, so installers would read a release status that
    disagrees with the repository.  Compare the distributed bytes, not the
    repository file alone.
    """

    wheel = _release_wheel_path()
    assert wheel.is_file(), f"release wheel is missing: {wheel}"
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = f"bioevidence_agent-{__version__}.dist-info/METADATA"
        raw = archive.read(metadata_name)
    message = BytesParser(policy=policy.compat32).parsebytes(raw)
    distributed = message.get_payload(decode=True)
    assert distributed is not None, "wheel METADATA has no decodable body"

    readme = (ROOT / "README.md").read_bytes()
    assert message.get("Version") == __version__
    assert distributed == readme, (
        "release wheel long description does not match README.md: "
        f"wheel={hashlib.sha256(distributed).hexdigest()} "
        f"readme={hashlib.sha256(readme).hexdigest()}; rebuild the wheel"
    )
