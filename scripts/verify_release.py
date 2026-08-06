"""Verify a release manifest and the tracked public repository boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "reports" / "release_manifest_v0.4.0.json"
PRIVATE_PARTS = {"private", "session_data"}
IGNORED_TRACKED_PARTS = {".git"}
FORBIDDEN_NAMES = {
    "ori_pqal.json",
    "test_ground_truth.json",
    "embeddings.npy",
}
LOCAL_ABSOLUTE_PREFIXES = tuple(
    "/" + prefix
    for prefix in (
        "Users/",
        "home/",
        "tmp/",
        "private/" + "tmp/",
        "private/var/",
        "var/folders/",
        "opt/",
    )
)
WINDOWS_ABSOLUTE_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:[\\/](?:Users|Temp)[\\/])"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_relative_path(relative: str) -> PurePosixPath:
    """Validate one manifest or Git path as a public repository path."""

    if not relative or "\\" in relative:
        raise ValueError(f"invalid repository path: {relative!r}")
    path = PurePosixPath(relative)
    folded_parts = {part.casefold() for part in path.parts}
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"absolute or escaping repository path: {relative}")
    if folded_parts.intersection(PRIVATE_PARTS):
        raise ValueError(f"private path is tracked or manifested: {relative}")
    if path.name in FORBIDDEN_NAMES or path.name.casefold().startswith("day"):
        raise ValueError(f"forbidden release file: {relative}")
    return path


def validate_public_text(relative: str, value: str) -> None:
    """Reject machine-local paths and unresolved merge markers."""

    if any(prefix in value for prefix in LOCAL_ABSOLUTE_PREFIXES):
        raise ValueError(f"local absolute path leaked: {relative}")
    if ("file:" + "///") in value or WINDOWS_ABSOLUTE_RE.search(value):
        raise ValueError(f"local absolute path leaked: {relative}")

    opening = "<" * 7
    separator = "=" * 7
    closing = ">" * 7
    for line in value.splitlines():
        stripped = line.lstrip()
        if (
            stripped.startswith(opening)
            or stripped == separator
            or stripped.startswith(closing)
        ):
            raise ValueError(f"merge conflict marker found: {relative}")


def tracked_public_files(root: Path = ROOT) -> list[Path]:
    """Return existing Git-tracked files after enforcing the public boundary."""

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    files: list[Path] = []
    for raw_relative in result.stdout.split(b"\0"):
        if not raw_relative:
            continue
        relative = raw_relative.decode("utf-8")
        public_path = validate_relative_path(relative)
        if set(public_path.parts).intersection(IGNORED_TRACKED_PARTS):
            continue
        path = root.joinpath(*public_path.parts)
        if not path.is_file():
            raise FileNotFoundError(f"tracked file is missing: {relative}")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"tracked path escapes repository: {relative}")
        files.append(path)
    if not files:
        raise ValueError("repository has no tracked files")
    return sorted(files)


def audit_tracked_tree(root: Path = ROOT) -> int:
    """Scan only Git-tracked public text files and return the file count."""

    paths = tracked_public_files(root)
    for path in paths:
        payload = path.read_bytes()
        if b"\0" in payload:
            continue
        try:
            value = payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        validate_public_text(path.relative_to(root).as_posix(), value)
    return len(paths)


def verify_release(manifest_path: Path, *, root: Path = ROOT) -> dict[str, object]:
    """Verify manifest identity, artifact hashes, and public-tree hygiene."""

    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not manifest_path.is_relative_to(root.resolve()):
        raise ValueError("release manifest must be inside the repository")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    package = manifest.get("package")
    if not isinstance(package, dict):
        raise ValueError("release manifest has no package metadata")
    version = package.get("version")
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("release manifest package version is invalid")
    expected_release_id = f"bioevidence-agent-v{version}"
    if manifest.get("release_id") != expected_release_id:
        raise ValueError("release ID and package version disagree")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("release manifest has no artifact hashes")
    tracked = {
        path.relative_to(root).as_posix() for path in tracked_public_files(root)
    }
    for relative, expected in artifacts.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("artifact paths and hashes must be strings")
        public_path = validate_relative_path(relative)
        if relative not in tracked:
            raise ValueError(f"release artifact is not Git-tracked: {relative}")
        if SHA256_RE.fullmatch(expected) is None:
            raise ValueError(f"invalid SHA-256 in manifest: {relative}")
        path = root.joinpath(*public_path.parts)
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"release hash mismatch: {relative}: "
                f"expected {expected}, observed {observed}"
            )

    wheel = package.get("wheel")
    wheel_hash = package.get("wheel_sha256")
    if not isinstance(wheel, str) or artifacts.get(wheel) != wheel_hash:
        raise ValueError("package wheel metadata is not bound to artifact hashes")

    tracked_count = audit_tracked_tree(root)
    return {
        "release_id": expected_release_id,
        "verified_artifact_count": len(artifacts),
        "scanned_tracked_file_count": tracked_count,
        "status": "pass",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="release manifest to verify (default: v0.4.0)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    print(
        json.dumps(
            verify_release(manifest_path),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
