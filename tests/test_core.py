from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from bioevidence.bm25 import BM25Index
from bioevidence.corpus import read_corpus
from bioevidence.portfolio_cli import main


def _fixture_path() -> Path:
    return Path(
        str(files("bioevidence").joinpath("fixtures/tiny_corpus.jsonl"))
    )


def test_packaged_fixture_is_hash_valid_and_bm25_retrievable() -> None:
    documents = read_corpus(_fixture_path())
    assert len(documents) == 3
    result = BM25Index(documents).search(
        "mitochondria programmed cell death",
        top_k=3,
    )
    assert result[0].pmid == "1001"
    assert result[0].document_sha256 == documents[0].content_sha256


def test_model_free_demo_emits_typed_trace_and_source_hash(
    capsys,
) -> None:
    assert (
        main(
            [
                "demo",
                "--question",
                "Do mitochondria participate in programmed cell death?",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["scope"] == "retrieval and provenance demo; no medical verdict"
    assert result["retrieved"]["pmid"] == "1001"
    assert len(result["retrieved"]["document_sha256"]) == 64
    assert len(result["retrieved"]["snippet_sha256"]) == 64
    assert [row["tool_name"] for row in result["tool_trace"]] == [
        "search_literature",
        "fetch_record",
        "inspect_evidence",
    ]
    assert [row["sequence"] for row in result["tool_trace"]] == [1, 2, 3]
    assert all(row["status"] == "success" for row in result["tool_trace"])
