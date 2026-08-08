from __future__ import annotations

import hashlib
import json

import pytest

from apps.product_demo.event_store import EventStore, EventValidationError
from apps.product_demo.service import ProductDemoError, ProductDemoService
from bioevidence.pubmed import PubMedArticle, PubMedError


QUESTION = "Remdesivir reduces all-cause mortality in hospitalized adults."


class FakePubMedClient:
    api_key_present = False

    def __init__(self, articles: list[PubMedArticle]) -> None:
        self._articles = {article.pmid: article for article in articles}

    def search(self, query: str, *, retmax: int = 5) -> list[str]:
        assert query
        return list(self._articles)[:retmax]

    def fetch(self, pmids: list[str]) -> list[PubMedArticle]:
        return [self._articles[pmid] for pmid in pmids if pmid in self._articles]


class FailingPubMedClient(FakePubMedClient):
    def search(self, query: str, *, retmax: int = 5) -> list[str]:
        raise PubMedError("offline")


class EmptyPubMedClient(FakePubMedClient):
    def search(self, query: str, *, retmax: int = 5) -> list[str]:
        return []


def _article(index: int) -> PubMedArticle:
    pmid = str(1000 + index)
    return PubMedArticle(
        pmid=pmid,
        title=f"Trial {index} of remdesivir in hospitalized adults",
        doi=f"10.1000/trial-{index}",
        publication_types=("Randomized Controlled Trial",),
        journal="Test Journal",
        year=2020 + index,
        first_author=f"Author {index}",
        abstract=(
            "BACKGROUND: Hospitalized adults were enrolled. "
            f"RESULTS: Trial {index} measured all-cause mortality after remdesivir. "
            "CONCLUSIONS: Full text review remains necessary."
        ),
    )


def _service() -> ProductDemoService:
    articles = [_article(index) for index in range(1, 5)]
    task = {
        "id": "test-task",
        "title": "Test task",
        "description": "A deterministic test task.",
        "question": QUESTION,
        "pmids": [article.pmid for article in articles],
    }
    return ProductDemoService(
        client=FakePubMedClient(articles),
        standard_tasks=[task],
    )


def test_standard_search_builds_source_bound_human_review_cards() -> None:
    service = _service()
    result = service.search(
        session_id="session-0001",
        mode="standard",
        task_id="test-task",
    )
    assert result["status"] == "complete"
    assert result["retrieval"]["candidate_source"] == "fixed_verified_pmids"
    assert result["retrieval"]["displayed_card_count"] == 4
    assert len(result["cards"]) == 4
    for card in result["cards"]:
        assert card["source_url"].startswith("https://pubmed.ncbi.nlm.nih.gov/")
        assert card["system_suggestion"]["direction"] == "unclear"
        snippet = card["snippet"]
        assert hashlib.sha256(snippet["text"].encode()).hexdigest() == snippet[
            "snippet_sha256"
        ]
        assert len(card["record_sha256"]) == 64
        source_abstract = service.client._articles[card["pmid"]].abstract
        assert source_abstract[snippet["start_char"] : snippet["end_char"]] == snippet[
            "text"
        ]
    assert result["tool_trace"][0]["tool_name"] == "search_literature"


def test_scope_like_language_never_becomes_an_automatic_verdict() -> None:
    abstracts = (
        "A mouse model measured viral load after remdesivir.",
        "Hospitalized adults received interferon beta-1a; mortality was measured.",
        "Hospitalized adults received remdesivir; the endpoint was time to recovery.",
    )
    articles = [
        PubMedArticle(
            pmid=str(2000 + index),
            title=f"Synthetic scope signal {index}",
            doi=None,
            publication_types=("Journal Article",),
            journal="Synthetic Test Journal",
            year=2026,
            first_author="Test Author",
            abstract=abstract,
        )
        for index, abstract in enumerate(abstracts, start=1)
    ]
    service = ProductDemoService(
        client=FakePubMedClient(articles),
        standard_tasks=[
            {
                "id": "scope-language-test",
                "title": "Scope language test",
                "description": "Synthetic regression only.",
                "question": QUESTION,
                "pmids": [article.pmid for article in articles],
            }
        ],
    )
    result = service.search(
        session_id="session-scope-0001",
        mode="standard",
        task_id="scope-language-test",
    )
    assert result["cards"]
    assert {
        card["system_suggestion"]["direction"] for card in result["cards"]
    } == {"unclear"}
    assert all("人工确认" in card["system_suggestion"]["label"] for card in result["cards"])


def test_export_uses_only_trusted_session_cards_and_user_judgment() -> None:
    service = _service()
    search = service.search(
        session_id="session-0002",
        mode="standard",
        task_id="test-task",
    )
    decisions = [
        {
            "card_id": card["card_id"],
            "action": "accepted" if index < 3 else "excluded",
            "user_direction": "opposes" if index < 3 else "unclear",
            "useful": index < 2,
            "note": f"User note {index}",
        }
        for index, card in enumerate(search["cards"])
    ]
    exported = service.export_pack(
        session_id="session-0002",
        decisions=decisions,
        pack_status="opposes",
        synthesis="The accepted abstracts do not support the stated endpoint.",
    )
    pack = exported["json"]
    assert pack["pack_version"] == "product-evidence-pack-v0.2"
    assert pack["audit"] == {
        "accepted_count": 3,
        "excluded_count": 1,
        "useful_count": 2,
        "all_source_urls_pubmed_formatted": True,
    }
    assert "all_sources_pubmed" not in pack["audit"]
    assert len(pack["pack_sha256"]) == 64
    assert "## Accepted evidence" in exported["markdown"]
    assert f"Pack SHA-256: `{pack['pack_sha256']}`" in exported["markdown"]
    assert search["cards"][0]["title"] in exported["markdown"]

    decisions[0]["card_id"] = "forged-card"
    with pytest.raises(ProductDemoError, match="标识无效"):
        service.export_pack(
            session_id="session-0002",
            decisions=decisions,
            pack_status="opposes",
            synthesis="",
        )


def test_live_mode_rejects_patient_specific_advice_and_freezes_failures() -> None:
    service = _service()
    with pytest.raises(ProductDemoError) as error:
        service.search(
            session_id="session-0003",
            mode="live",
            question="What should I take for my symptoms today?",
        )
    assert error.value.code == "medical_advice_not_supported"

    live_result = service.search(
        session_id="session-live-01",
        mode="live",
        question=QUESTION,
        task_id="caller-supplied",
    )
    assert live_result["task_id"] == "live-research"

    failing = ProductDemoService(
        client=FailingPubMedClient([]),
        standard_tasks=[
            {
                "id": "unused-task",
                "title": "Unused",
                "description": "Unused",
                "question": QUESTION,
                "pmids": ["1", "2", "3"],
            }
        ],
    )
    with pytest.raises(ProductDemoError) as external:
        failing.search(
            session_id="session-0004",
            mode="live",
            question=QUESTION,
        )
    assert external.value.code == "external_service_failure"
    assert external.value.http_status == 503


def test_event_store_logs_metrics_without_questions_or_notes(tmp_path) -> None:
    store = EventStore(tmp_path)
    event = store.append(
        {
            "session_id": "session-0005",
            "event_type": "card_action",
            "task_id": "test-task",
            "card_id": "E1",
            "elapsed_ms": 1234,
            "metadata": {"field": "action", "value": "accepted"},
        }
    )
    assert event["sequence"] == 1
    path = tmp_path / "session-0005.jsonl"
    stored = json.loads(path.read_text())
    assert stored["metadata"] == {"field": "action", "value": "accepted"}

    with pytest.raises(EventValidationError, match="event contract"):
        store.append(
            {
                "session_id": "session-0005",
                "event_type": "note_changed",
                "elapsed_ms": 2000,
                "metadata": {"note": "private research content"},
            }
        )
    assert "private research content" not in path.read_text()


@pytest.mark.parametrize("unknown_field", ["user_question", "details"])
def test_event_store_rejects_unknown_metadata_fields(tmp_path, unknown_field) -> None:
    store = EventStore(tmp_path)
    sensitive_text = "private research content"

    with pytest.raises(EventValidationError, match="event contract"):
        store.append(
            {
                "session_id": "session-privacy-01",
                "event_type": "source_opened",
                "elapsed_ms": 100,
                "metadata": {unknown_field: sensitive_text},
            }
        )

    path = tmp_path / "session-privacy-01.jsonl"
    assert not path.exists()


def test_event_store_continues_sequence_after_restart(tmp_path) -> None:
    payload = {
        "session_id": "session-restart-01",
        "event_type": "session_started",
        "elapsed_ms": 0,
        "metadata": {"mode": "standard"},
    }
    assert EventStore(tmp_path).append(payload)["sequence"] == 1

    restarted_store = EventStore(tmp_path)
    payload["event_type"] = "search_started"
    payload["elapsed_ms"] = 1
    assert restarted_store.append(payload)["sequence"] == 2

    stored = [
        json.loads(line)
        for line in (tmp_path / "session-restart-01.jsonl").read_text().splitlines()
    ]
    assert [event["sequence"] for event in stored] == [1, 2]


@pytest.mark.parametrize(
    ("event_type", "metadata"),
    [
        ("session_started", {"mode": "standard"}),
        ("search_started", {"mode": "live"}),
        (
            "search_completed",
            {"mode": "standard", "card_count": 3, "result_status": "complete"},
        ),
        ("search_failed", {"error_code": "external_service_failure"}),
        ("card_action", {"field": "action", "value": "excluded"}),
        ("card_action", {"field": "user_direction", "value": "supports"}),
        ("card_useful", {"useful": True}),
        ("source_opened", {}),
        ("note_changed", {"has_content": True}),
        ("export_started", {"format": "json"}),
        ("export_completed", {"format": "markdown", "accepted_count": 1}),
        ("error_shown", {"error_code": "no_results"}),
        ("timeout", {"threshold_minutes": 12}),
    ],
)
def test_event_store_accepts_ui_event_contracts(tmp_path, event_type, metadata) -> None:
    event = EventStore(tmp_path).append(
        {
            "session_id": f"session-{event_type.replace('_', '-')}",
            "event_type": event_type,
            "elapsed_ms": 1,
            "metadata": metadata,
        }
    )
    assert event["metadata"] == metadata


def test_exact_normalized_title_duplicates_are_removed_conservatively() -> None:
    articles = [_article(index) for index in range(1, 5)]
    duplicate = articles[-1]
    articles[-1] = PubMedArticle(
        pmid=duplicate.pmid,
        title=f"  {articles[0].title.upper()}  ",
        doi=duplicate.doi,
        publication_types=duplicate.publication_types,
        journal=duplicate.journal,
        year=duplicate.year,
        first_author=duplicate.first_author,
        abstract=duplicate.abstract,
    )
    service = ProductDemoService(
        client=FakePubMedClient(articles),
        standard_tasks=[
            {
                "id": "dedup-test",
                "title": "Deduplication test",
                "description": "Exact normalized titles only.",
                "question": QUESTION,
                "pmids": [article.pmid for article in articles],
            }
        ],
    )
    result = service.search(
        session_id="session-0006",
        mode="standard",
        task_id="dedup-test",
    )
    assert result["retrieval"]["candidate_pmid_count"] == 4
    assert result["retrieval"]["abstract_document_count"] == 3
    assert len({card["title"].casefold().strip() for card in result["cards"]}) == 3


def test_no_results_and_no_abstracts_have_distinct_safe_errors() -> None:
    empty = ProductDemoService(
        client=EmptyPubMedClient([]),
        standard_tasks=[
            {
                "id": "empty-task",
                "title": "Empty task",
                "description": "No search results",
                "question": QUESTION,
                "pmids": ["1", "2", "3"],
            }
        ],
    )
    with pytest.raises(ProductDemoError) as no_results:
        empty.search(
            session_id="session-0007",
            mode="live",
            question=QUESTION,
        )
    assert no_results.value.code == "no_results"
    assert no_results.value.http_status == 404

    articles = [_article(index) for index in range(1, 4)]
    without_abstracts = [
        PubMedArticle(
            pmid=article.pmid,
            title=article.title,
            doi=article.doi,
            publication_types=article.publication_types,
            journal=article.journal,
            year=article.year,
            first_author=article.first_author,
            abstract="",
        )
        for article in articles
    ]
    no_abstract_service = ProductDemoService(
        client=FakePubMedClient(without_abstracts),
        standard_tasks=[
            {
                "id": "no-abstract-task",
                "title": "No abstract task",
                "description": "Records without abstracts",
                "question": QUESTION,
                "pmids": [article.pmid for article in without_abstracts],
            }
        ],
    )
    with pytest.raises(ProductDemoError) as no_abstracts:
        no_abstract_service.search(
            session_id="session-0008",
            mode="standard",
            task_id="no-abstract-task",
        )
    assert no_abstracts.value.code == "no_abstract_evidence"
    assert no_abstracts.value.http_status == 422


@pytest.mark.parametrize("question", ["unclear", "x" * 601])
def test_live_mode_rejects_ambiguous_or_oversized_questions(question) -> None:
    with pytest.raises(ProductDemoError) as error:
        _service().search(
            session_id="session-0009",
            mode="live",
            question=question,
        )
    assert error.value.code == "invalid_question"


def test_mixed_and_insufficient_are_user_owned_not_system_verdicts() -> None:
    service = _service()
    search = service.search(
        session_id="session-0010",
        mode="standard",
        task_id="test-task",
    )
    decisions = [
        {
            "card_id": card["card_id"],
            "action": "accepted",
            "user_direction": "supports" if index % 2 == 0 else "opposes",
            "useful": True,
            "note": "User-reviewed direction.",
        }
        for index, card in enumerate(search["cards"])
    ]
    mixed = service.export_pack(
        session_id="session-0010",
        decisions=decisions,
        pack_status="mixed",
        synthesis="Accepted abstracts point in different directions.",
    )["json"]
    assert mixed["pack_status"] == "mixed"
    assert {item["user_decision"]["user_direction"] for item in mixed["evidence"]} == {
        "supports",
        "opposes",
    }
    assert all(
        item["card"]["system_suggestion"]["direction"] == "unclear"
        for item in mixed["evidence"]
    )

    insufficient = service.export_pack(
        session_id="session-0010",
        decisions=decisions[:1],
        pack_status="insufficient",
        synthesis="Only one accepted abstract; this does not meet VEPS.",
    )["json"]
    assert insufficient["pack_status"] == "insufficient"
    assert insufficient["audit"]["accepted_count"] == 1
