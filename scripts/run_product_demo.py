"""Repository-local launcher for the BioEvidence Product Demo."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))
    sys.path.insert(0, str(repository_root / "src"))
    server_main = import_module("apps.product_demo.server").main
    return server_main()


if __name__ == "__main__":
    raise SystemExit(main())
