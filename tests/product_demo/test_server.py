from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from apps.product_demo.event_store import EventStore
from apps.product_demo.server import create_handler
from apps.product_demo.service import ProductDemoService
from bioevidence.pubmed import PubMedArticle

QUESTION = "A study measured mortality after treatment in hospitalized adults."


class FakePubMedClient:
    api_key_present = False

    def __init__(self) -> None:
        self.articles = {
            article.pmid: article for article in [_article(1), _article(2), _article(3)]
        }

    def search(self, query: str, *, retmax: int = 5) -> list[str]:
        return list(self.articles)[:retmax]

    def fetch(self, pmids: list[str]) -> list[PubMedArticle]:
        return [self.articles[pmid] for pmid in pmids if pmid in self.articles]


def _article(index: int) -> PubMedArticle:
    return PubMedArticle(
        pmid=str(9000 + index),
        title=f"Source-bound trial {index}",
        doi=f"10.1000/source-{index}",
        publication_types=("Randomized Controlled Trial",),
        journal="Evidence Test Journal",
        year=2020,
        first_author="Verifier",
        abstract=(
            "BACKGROUND: Hospitalized adults were enrolled. "
            f"RESULTS: Study {index} measured mortality after treatment. "
            "CONCLUSIONS: Full-text verification is still required."
        ),
    )


@contextmanager
def _running_server(event_root: Path):
    service = ProductDemoService(
        client=FakePubMedClient(),
        standard_tasks=[
            {
                "id": "http-test",
                "title": "HTTP test",
                "description": "Local integration test",
                "question": QUESTION,
                "pmids": ["9001", "9002", "9003"],
            }
        ],
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        create_handler(service, EventStore(event_root)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _get(url: str) -> tuple[int, dict[str, str], bytes]:
    with urllib.request.urlopen(url, timeout=3) as response:
        return response.status, dict(response.headers), response.read()


def _post(url: str, value: object) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(value).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_local_server_serves_hardened_static_assets(tmp_path) -> None:
    with _running_server(tmp_path) as base_url:
        status, headers, html = _get(f"{base_url}/")
        assert status == 200
        assert headers["Cache-Control"] == "no-store"
        assert "default-src 'self'" in headers["Content-Security-Policy"]
        assert b"BioEvidence Agent" in html

        _, _, javascript = _get(f"{base_url}/app.js")
        assert b"/api/search" in javascript
        assert b"/api/export" in javascript
        assert b"innerHTML" not in javascript
        assert b"https://" not in javascript
        assert b"live-research" in javascript
        assert b"participant-owned" not in javascript
        assert b"stopTimerAfterExport" in javascript
        assert b"exportStatus" in javascript
        assert b"Source-bound evidence workspace" in html
        assert b'id="export-status"' in html


def test_search_export_and_privacy_event_routes(tmp_path) -> None:
    with _running_server(tmp_path) as base_url:
        status, _, task_payload = _get(f"{base_url}/api/standard-tasks")
        tasks = json.loads(task_payload)
        assert status == 200
        assert tasks["tasks"][0]["id"] == "http-test"

        session_id = "session-http-01"
        status, search = _post(
            f"{base_url}/api/search",
            {
                "session_id": session_id,
                "mode": "standard",
                "task_id": "http-test",
            },
        )
        assert status == 200
        assert len(search["cards"]) == 3

        decisions = [
            {
                "card_id": card["card_id"],
                "action": "accepted",
                "user_direction": "unclear",
                "useful": True,
                "note": "Check the full text.",
            }
            for card in search["cards"]
        ]
        status, exported = _post(
            f"{base_url}/api/export",
            {
                "session_id": session_id,
                "decisions": decisions,
                "pack_status": "insufficient",
                "synthesis": "Abstract-level evidence requires full-text review.",
            },
        )
        assert status == 200
        assert exported["json"]["audit"]["accepted_count"] == 3

        status, event = _post(
            f"{base_url}/api/events",
            {
                "session_id": session_id,
                "event_type": "export_completed",
                "task_id": "http-test",
                "elapsed_ms": 1500,
                "metadata": {"format": "json", "accepted_count": 3},
            },
        )
        assert status == 201
        assert event["sequence"] == 1
        stored = (tmp_path / f"{session_id}.jsonl").read_text()
        assert "Abstract-level evidence" not in stored
        assert "Check the full text" not in stored

        status, error = _post(
            f"{base_url}/api/events",
            {
                "session_id": session_id,
                "event_type": "note_changed",
                "elapsed_ms": 1600,
                "metadata": {"note": "must never be logged"},
            },
        )
        assert status == 400
        assert error["error"]["code"] == "invalid_event"
        assert "must never be logged" not in stored


@pytest.mark.parametrize("path", ["/missing", "/api/missing"])
def test_unknown_routes_fail_closed(tmp_path, path) -> None:
    with _running_server(tmp_path) as base_url:
        if path.startswith("/api/"):
            status, payload = _post(f"{base_url}{path}", {"anything": "value"})
        else:
            try:
                _get(f"{base_url}{path}")
            except urllib.error.HTTPError as error:
                status = error.code
                payload = json.loads(error.read())
        assert status == 404
        assert payload["error"]["code"] == "not_found"


def test_repository_launcher_works_outside_repository(tmp_path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "run_product_demo.py"),
            "--help",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0
    assert "Run the local BioEvidence demo" in completed.stdout
