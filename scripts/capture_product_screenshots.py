"""Capture desensitized Product Demo screenshots for the v0.4.1 portfolio.

The capture runs the Standard-mode task with fixed public PMIDs only, so the
resulting images contain public PubMed-derived content and no participant or
private material. Requires a running demo server (scripts/run_product_demo.py)
and Playwright with its Chromium browser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "product_case" / "screenshots"
DEFAULT_BASE_URL = "http://127.0.0.1:8765"


def _screenshot_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="demo server base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="screenshot output directory (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    from playwright.sync_api import sync_playwright

    output_dir = _screenshot_dir(args.output_dir)
    cards_png = output_dir / "product_demo_standard_cards.png"
    export_png = output_dir / "product_demo_evidence_export.png"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(args.base_url, wait_until="load")
        page.wait_for_selector("#task-select", timeout=30_000)
        page.wait_for_function(
            "document.querySelectorAll('#task-select option').length > 0",
            timeout=30_000,
        )
        page.locator("#safety-confirm").check()
        page.locator("#search-button").click()
        page.wait_for_selector("#cards .evidence-card", timeout=120_000)
        page.wait_for_timeout(800)
        page.locator("#results-section").scroll_into_view_if_needed()
        page.screenshot(path=str(cards_png), full_page=False)

        first_card = page.locator(".evidence-card").first
        first_card.locator('.choice-button[data-value="accepted"]').click()
        first_card.locator('.choice-button[data-value="opposes"]').click()
        first_card.locator('textarea').fill("28-day mortality endpoint retained for full-text check.")
        page.locator("#pack-status").select_option("opposes")
        page.locator("#synthesis").fill(
            "Public demo example: abstract-level evidence only; full-text review required."
        )
        with page.expect_download(timeout=30_000) as json_download_info:
            page.locator("#export-json").click()
        json_download = json_download_info.value
        json_path = json_download.path()
        if json_path is None or not json_download.suggested_filename.endswith(".json"):
            raise RuntimeError("JSON evidence-pack download was not created")
        pack = json.loads(json_path.read_text(encoding="utf-8"))
        if pack.get("audit", {}).get("accepted_count") != 1:
            raise RuntimeError("downloaded JSON evidence pack failed its audit check")
        declared_hash = pack.pop("pack_sha256", None)
        canonical_json = json.dumps(
            pack,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if declared_hash != hashlib.sha256(canonical_json).hexdigest():
            raise RuntimeError("downloaded JSON evidence-pack hash is invalid")

        page.wait_for_function(
            "document.querySelector('#export-status').textContent.includes('已生成')",
            timeout=30_000,
        )
        page.locator("#export-section").scroll_into_view_if_needed()
        page.screenshot(path=str(export_png), full_page=False)
        browser.close()

    print(
        f"captured: {cards_png}\ncaptured: {export_png}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
