from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bioevidence.corpus import CorpusDocument, sha256_file
from bioevidence.evidence_utilization import (
    SYSTEMS as CONTROL_SYSTEMS,
    _shuffled_donors,
    _stratified_folds,
)
from bioevidence.hybrid import reciprocal_rank_fusion
from bioevidence.pubmed import PubMedArticle
from bioevidence.pubmedqa_stress import (
    SYSTEMS as RETRIEVAL_SYSTEMS,
    AbstractOnlyBM25,
    AbstractOnlyVectorIndex,
    abstract_passages,
    rerank_abstract_candidates,
)
from scripts.verify_release import audit_tracked_tree


ROOT = Path(__file__).resolve().parents[1]
ABSTRACT_REPORT = ROOT / "reports/pubmedqa_abstract_only_retrieval_v1.json"
CONTROL_REPORT = ROOT / "reports/pubmedqa_evidence_utilization_cv_v1.json"


class _Encoder:
    model_id = "fake-e5"
    model_revision = "fake-revision"
    dimension = 2
    max_length = 32

    def __init__(self) -> None:
        self.passage_inputs: list[str] = []

    def encode(
        self,
        texts: list[str],
        *,
        input_type: str,
        batch_size: int = 8,
    ) -> np.ndarray:
        del batch_size
        if input_type == "passage":
            self.passage_inputs = list(texts)
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class _Scorer:
    model_id = "fake-reranker"
    model_revision = "fake-revision"
    max_length = 32
    batch_size = 2
    device = "cpu"

    def __init__(self) -> None:
        self.passage_inputs: list[str] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        assert query == "secret title token"
        self.passage_inputs = list(passages)
        return [1.0, 0.0]


def _document(pmid: str, title: str, abstract: str) -> CorpusDocument:
    return CorpusDocument.from_pubmed(
        PubMedArticle(
            pmid=pmid,
            title=title,
            doi=None,
            publication_types=("Test",),
            journal="",
            year=2024,
            first_author=None,
            abstract=abstract,
        )
    )


def test_title_is_absent_from_bm25_e5_and_reranker_passages() -> None:
    documents = [
        _document("1", "secret title token", "alpha abstract"),
        _document("2", "other title", "beta abstract"),
    ]
    passages = abstract_passages(documents)
    assert all(not hasattr(row, "title") for row in passages)
    lexical = AbstractOnlyBM25(passages).search(
        "secret title token",
        top_k=2,
    )
    assert [row.score for row in lexical] == [0.0, 0.0]
    encoder = _Encoder()
    dense = AbstractOnlyVectorIndex.build(
        passages,
        encoder=encoder,
        batch_size=2,
    ).search("secret title token", encoder=encoder, top_k=2)
    hybrid = reciprocal_rank_fusion(
        {"bm25": lexical, "vector": dense},
        top_k=2,
    )
    scorer = _Scorer()
    rerank_abstract_candidates(
        "secret title token",
        hybrid,
        passages_by_pmid={row.pmid: row for row in passages},
        scorer=scorer,
        top_k=2,
    )
    assert encoder.passage_inputs == ["alpha abstract", "beta abstract"]
    assert scorer.passage_inputs == ["alpha abstract", "beta abstract"]


def test_fold_mapping_is_complete_and_shuffle_is_a_derangement() -> None:
    labels = ("yes", "no", "maybe")
    rows = [
        {
            "pmid": str(index + 1),
            "question": f"q{index}",
            "abstract": f"a{index}",
            "label": labels[index % 3],
        }
        for index in range(30)
    ]
    folds = _stratified_folds(rows)
    donors = _shuffled_donors(rows, folds)
    assert len(folds) == len(donors) == 30
    assert set(folds) == set(range(5))
    assert all(index != donor for index, donor in enumerate(donors))
    assert all(folds[index] == folds[donor] for index, donor in enumerate(donors))
    assert folds == _stratified_folds(rows)
    assert donors == _shuffled_donors(rows, folds)


def test_frozen_reports_have_honest_complete_denominators() -> None:
    abstract = json.loads(ABSTRACT_REPORT.read_text(encoding="utf-8"))
    controls = json.loads(CONTROL_REPORT.read_text(encoding="utf-8"))
    assert abstract["case_count"] == 500
    assert set(abstract["retrieval_metrics"]) == set(RETRIEVAL_SYSTEMS)
    assert abstract["rank_drop_count"] >= len(abstract["failure_examples"]) >= 10
    assert all(
        abstract["latency_ms"][system]["count"] == 500
        for system in RETRIEVAL_SYSTEMS
    )
    assert controls["case_count"] == 500
    assert controls["fold_count"] == 5
    assert set(controls["metrics"]) == set(CONTROL_SYSTEMS)
    assert all(
        controls["metrics"][system]["total"] == 500
        for system in CONTROL_SYSTEMS
    )
    assert controls["model_policy"][
        "same_combined_model_for_correct_and_shuffled_validation"
    ]
    assert set(controls["scientific_questions"]) == {
        "is_current_answerer_significantly_better_than_majority",
        "does_context_provide_measurable_gain",
        "does_shuffling_context_clearly_reduce_results",
        "why_is_maybe_weak",
        "is_primary_bottleneck_retrieval_or_evidence_interpretation",
    }
    assert "not strong evidence" in controls["scientific_questions"][
        "does_shuffling_context_clearly_reduce_results"
    ]


def test_json_and_markdown_headline_numbers_match() -> None:
    for json_name, md_name, expected in (
        (
            "pubmedqa_abstract_only_retrieval_v1.json",
            "pubmedqa_abstract_only_retrieval_v1.md",
            ("0.9900", "0.9820"),
        ),
        (
            "pubmedqa_evidence_utilization_cv_v1.json",
            "pubmedqa_evidence_utilization_cv_v1.md",
            ("0.5360", "0.3557"),
        ),
    ):
        report = json.loads((ROOT / "reports" / json_name).read_text())
        markdown = (ROOT / "reports" / md_name).read_text()
        assert report["case_count"] == 500
        assert all(value in markdown for value in expected)


def test_release_tree_contains_no_local_absolute_paths() -> None:
    assert audit_tracked_tree(ROOT) > 0


def test_report_hashes_can_be_recomputed() -> None:
    for name in (
        "pubmedqa_abstract_only_retrieval_v1.json",
        "pubmedqa_evidence_utilization_cv_v1.json",
    ):
        assert len(sha256_file(ROOT / "reports" / name)) == 64
