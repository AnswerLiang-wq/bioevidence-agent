"""Verify the v0.3.0 release allowlist, hashes, and privacy boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports" / "release_manifest_v0.3.0.json"
IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
}
FORBIDDEN_NAMES = {
    "ori_pqal.json",
    "test_ground_truth.json",
    "embeddings.npy",
}
FORBIDDEN_TEXT = (
    "/" + "Users/",
    "/" + "private/tmp/",
    "未来规划/" + "BioEvidenceAgent",
)
TEXT_SUFFIXES = {
    ".cff",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("release_id") != "bioevidence-agent-v0.3.0":
        raise ValueError("unexpected release ID")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("release manifest has no artifact hashes")
    for relative, expected in artifacts.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"release hash mismatch: {relative}: "
                f"expected {expected}, observed {observed}"
            )

    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        if path.name in FORBIDDEN_NAMES or path.name.lower().startswith("day"):
            raise ValueError(f"forbidden release file: {path.relative_to(ROOT)}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            value = path.read_text(encoding="utf-8")
            if any(fragment in value for fragment in FORBIDDEN_TEXT):
                raise ValueError(
                    f"local absolute path leaked: {path.relative_to(ROOT)}"
                )
    print(
        json.dumps(
            {
                "release_id": manifest["release_id"],
                "verified_artifact_count": len(artifacts),
                "status": "pass",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
