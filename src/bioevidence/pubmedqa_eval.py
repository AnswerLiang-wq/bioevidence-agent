"""One-pass public PubMedQA retrieval, answer, grounding, and runtime report."""

from __future__ import annotations

import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

from .bm25 import BM25Index
from .corpus import CorpusDocument, read_corpus, sha256_file
from .hybrid import reciprocal_rank_fusion
from .product_contracts import validate_product_response
from .pubmedqa import (
    CORPUS_FILENAME,
    MANIFEST_FILENAME,
    PUBMEDQA_LABELS,
    TEST_GOLD_FILENAME,
    TEST_INPUTS_FILENAME,
    TRAIN_FILENAME,
    TfidfLogisticAnswerer,
    read_jsonl,
)
from .pubmedqa_agent import PubMedQAEvidenceAgent
from .reranker import PairScorer, rerank_candidates
from .retrievers import RetrievalHit
from .vector import INDEX_MANIFEST, TextEncoder, VectorIndex


FINAL_SYSTEM = "hybrid_reranked"


class FrozenQueryRetriever:
    """Expose already-computed rankings through the normal Retriever contract."""

    def __init__(
        self,
        *,
        query: str,
        hits: Sequence[RetrievalHit],
        metadata: dict[str, object],
    ) -> None:
        self._query = query
        self._hits = tuple(hits)
        self.method = "hybrid_rrf_cross_encoder"
        self.metadata = metadata

    def search(
        self,
        query: str,
        *,
        top_k: int,
        rerank_query: str | None = None,
    ) -> list[RetrievalHit]:
        if query != self._query or rerank_query not in {None, self._query}:
            raise ValueError("frozen PubMedQA ranking query mismatch")
        return list(self._hits[:top_k])


def evaluate_pubmedqa(
    *,
    benchmark_dir: Path,
    index_dir: Path,
    output_json: Path,
    output_markdown: Path,
    responses_path: Path,
    encoder: TextEncoder,
    reranker: PairScorer,
    reranker_candidate_k: int = 20,
) -> dict[str, object]:
    """Run the fixed public test once and write immutable report artifacts."""

    for path in (output_json, output_markdown, responses_path):
        if path.exists():
            raise FileExistsError(
                f"PubMedQA evaluation artifacts are immutable: {path}"
            )
    artifacts = _load_artifacts(benchmark_dir)
    documents = artifacts["documents"]
    documents_by_pmid = {document.pmid: document for document in documents}
    vector_index = VectorIndex.load(
        index_dir,
        documents=documents,
        corpus_path=benchmark_dir / CORPUS_FILENAME,
    )
    answerer = TfidfLogisticAnswerer()
    answerer.fit(artifacts["train"], documents_by_pmid)
    bm25 = BM25Index(documents)

    case_results: list[dict[str, object]] = []
    response_rows: list[dict[str, object]] = []
    latencies_ms: list[float] = []
    system_predictions: dict[str, list[str]] = {
        name: [] for name in ("bm25", "vector", "hybrid_rrf", FINAL_SYSTEM)
    }
    expected_labels: list[str] = []

    for input_row in artifacts["inputs"]:
        case_id = str(input_row["case_id"])
        question = str(input_row["question"])
        gold = artifacts["gold_by_id"][case_id]
        expected_pmid = str(gold["pmid"])
        expected_label = str(gold["label"])
        expected_labels.append(expected_label)

        started = time.perf_counter()
        lexical = bm25.search(question, top_k=len(documents))
        dense = vector_index.search(
            question,
            encoder=encoder,
            top_k=len(documents),
        )
        hybrid = reciprocal_rank_fusion(
            {"bm25": lexical, "vector": dense},
            top_k=len(documents),
        )
        reranked = rerank_candidates(
            question,
            hybrid[: min(reranker_candidate_k, len(hybrid))],
            documents_by_pmid=documents_by_pmid,
            scorer=reranker,
            top_k=min(reranker_candidate_k, len(hybrid)),
        )
        reranked_hits = [
            RetrievalHit(
                pmid=item.pmid,
                rank=item.rank,
                score=item.score,
                document_sha256=item.document_sha256,
                retrieval_method="hybrid_rrf_cross_encoder",
                component_ranks=item.component_ranks,
                candidate_rank=item.candidate_rank,
            )
            for item in reranked
        ]
        frozen_retriever = FrozenQueryRetriever(
            query=question,
            hits=reranked_hits,
            metadata={
                "method": "hybrid_rrf_cross_encoder",
                "candidate_k": reranker_candidate_k,
                "fusion": {"method": "reciprocal_rank_fusion", "rrf_k": 60},
                "vector": {
                    "model_id": vector_index.model_id,
                    "model_revision": vector_index.model_revision,
                    "max_length": vector_index.max_length,
                },
                "reranker": {
                    "model_id": reranker.model_id,
                    "model_revision": reranker.model_revision,
                    "max_length": reranker.max_length,
                    "batch_size": reranker.batch_size,
                    "device": reranker.device,
                },
            },
        )
        response = PubMedQAEvidenceAgent(
            documents,
            corpus_sha256=sha256_file(benchmark_dir / CORPUS_FILENAME),
            retriever=frozen_retriever,
            answerer=answerer,
        ).run(question=question, request_id=case_id)
        latencies_ms.append((time.perf_counter() - started) * 1000)

        rankings = {
            "bm25": [item.pmid for item in lexical],
            "vector": [item.pmid for item in dense],
            "hybrid_rrf": [item.pmid for item in hybrid],
            FINAL_SYSTEM: [item.pmid for item in reranked],
        }
        predictions: dict[str, str] = {}
        for name, ranked_pmids in rankings.items():
            predicted_label, _ = answerer.predict(
                question,
                documents_by_pmid[ranked_pmids[0]].abstract,
            )
            predictions[name] = predicted_label
            system_predictions[name].append(predicted_label)

        response_rows.append({"case_id": case_id, "response": response})
        case_results.append(
            _case_result(
                case_id=case_id,
                question=question,
                expected_pmid=expected_pmid,
                expected_label=expected_label,
                rankings=rankings,
                predictions=predictions,
                response=response,
                documents_by_pmid=documents_by_pmid,
            )
        )

    _write_jsonl(responses_path, response_rows)
    retrieval_metrics = {
        name: _retrieval_metrics(case_results, name)
        for name in ("bm25", "vector", "hybrid_rrf", FINAL_SYSTEM)
    }
    answer_metrics = {
        name: _classification_metrics(expected_labels, predictions)
        for name, predictions in system_predictions.items()
    }
    failures = _failure_examples(case_results, minimum=10)
    report = {
        "report_version": "1.0.0",
        "benchmark_id": artifacts["manifest"]["benchmark_id"],
        "baseline_id": (
            "bioevidence-pubmedqa-public-closed-tfidf-logreg-reranked-v1"
        ),
        "evaluation_scope": (
            "fixed 500-case official PubMedQA public test in a 1,000-document "
            "closed corpus; not blind or open-world generalization"
        ),
        "case_count": len(case_results),
        "corpus_document_count": len(documents),
        "inputs": {
            "manifest_sha256": sha256_file(
                benchmark_dir / MANIFEST_FILENAME
            ),
            "corpus_sha256": sha256_file(
                benchmark_dir / CORPUS_FILENAME
            ),
            "train_sha256": sha256_file(benchmark_dir / TRAIN_FILENAME),
            "test_inputs_sha256": sha256_file(
                benchmark_dir / TEST_INPUTS_FILENAME
            ),
            "test_gold_sha256": sha256_file(
                benchmark_dir / TEST_GOLD_FILENAME
            ),
            "vector_manifest_sha256": sha256_file(
                index_dir / INDEX_MANIFEST
            ),
            "responses_sha256": sha256_file(responses_path),
        },
        "models": {
            "answerer": answerer.metadata,
            "vector": {
                "model_id": vector_index.model_id,
                "model_revision": vector_index.model_revision,
                "max_length": vector_index.max_length,
                "embedding_dimension": vector_index.embeddings.shape[1],
            },
            "reranker": {
                "model_id": reranker.model_id,
                "model_revision": reranker.model_revision,
                "max_length": reranker.max_length,
                "candidate_k": reranker_candidate_k,
                "batch_size": reranker.batch_size,
                "device": reranker.device,
            },
        },
        "retrieval_metrics": retrieval_metrics,
        "answer_metrics": answer_metrics,
        "headline_system": FINAL_SYSTEM,
        "structural_grounding": _grounding_metrics(case_results),
        "runtime": {
            "end_to_end_latency_ms": _distribution(latencies_ms),
            "reranker": _runtime_summary(reranker),
            "model_api_cost_usd": {
                "mean": 0.0,
                "p95": 0.0,
                "total": 0.0,
            },
        },
        "failure_taxonomy": _failure_counts(case_results),
        "failure_examples": failures,
        "case_results": case_results,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "claims_not_made": [
            "private blind performance",
            "open-world literature-retrieval completeness",
            "new-disease or real-world generalization",
            "clinical validity, treatment efficacy, or medical advice",
            "independently confirmed semantic hallucination rate",
            "semantic citation precision without independent human review",
        ],
        "interpretation_limits": artifacts["manifest"]["limitations"]
        + [
            (
                "Answer labels are public expert annotations, but the fixed "
                "answerer is a small engineering baseline and its probability "
                "is not evidence strength."
            ),
            (
                "Exact-span and PMID-source metrics verify structure and source "
                "identity, not whether every generated sentence is semantically "
                "complete."
            ),
            (
                "No test-case error was used to tune this v1 baseline after "
                "the run; improvements require a new baseline ID."
            ),
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(_markdown_report(report), encoding="utf-8")
    return report


def _load_artifacts(benchmark_dir: Path) -> dict[str, object]:
    manifest = json.loads(
        (benchmark_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    if not isinstance(manifest, dict):
        raise ValueError("PubMedQA manifest must be an object")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("PubMedQA manifest is missing file hashes")
    for name in (
        CORPUS_FILENAME,
        TRAIN_FILENAME,
        TEST_INPUTS_FILENAME,
        TEST_GOLD_FILENAME,
    ):
        if sha256_file(benchmark_dir / name) != files.get(name):
            raise ValueError(f"PubMedQA artifact hash mismatch: {name}")
    documents = read_corpus(benchmark_dir / CORPUS_FILENAME)
    train = read_jsonl(benchmark_dir / TRAIN_FILENAME)
    inputs = read_jsonl(benchmark_dir / TEST_INPUTS_FILENAME)
    gold = read_jsonl(benchmark_dir / TEST_GOLD_FILENAME)
    if not (
        len(documents) == 1000
        and len(train) == len(inputs) == len(gold) == 500
    ):
        raise ValueError("PubMedQA artifact counts differ from frozen design")
    if any(set(row) != {"case_id", "question"} for row in inputs):
        raise ValueError("PubMedQA runner input contains non-contract fields")
    gold_by_id = {str(row.get("case_id")): row for row in gold}
    if len(gold_by_id) != 500 or set(gold_by_id) != {
        str(row["case_id"]) for row in inputs
    }:
        raise ValueError("PubMedQA input/gold case IDs differ")
    return {
        "manifest": manifest,
        "documents": documents,
        "train": train,
        "inputs": inputs,
        "gold_by_id": gold_by_id,
    }


def _case_result(
    *,
    case_id: str,
    question: str,
    expected_pmid: str,
    expected_label: str,
    rankings: dict[str, list[str]],
    predictions: dict[str, str],
    response: dict[str, object],
    documents_by_pmid: dict[str, CorpusDocument],
) -> dict[str, object]:
    citation = response["citations"][0]
    cited_pmid = str(citation["pmid"])
    document = documents_by_pmid[cited_pmid]
    start = int(citation["start_char"])
    end = int(citation["end_char"])
    snippet = str(citation["snippet"])
    exact_span = document.abstract[start:end] == snippet
    source_hash_valid = citation["source_sha256"] == document.content_sha256
    schema_valid = not validate_product_response(response)
    claim_coverage = all(
        isinstance(claim.get("citation_ids"), list)
        and bool(claim["citation_ids"])
        for claim in response["claims"]
    )
    retrieval: dict[str, object] = {}
    for name, ranking in rankings.items():
        retrieval[name] = {
            "top_10_pmids": ranking[:10],
            "recall_at_5": float(expected_pmid in ranking[:5]),
            "recall_at_10": float(expected_pmid in ranking[:10]),
            "reciprocal_rank": _reciprocal_rank(ranking, expected_pmid),
            "ndcg_at_10": _ndcg_at_10(ranking, expected_pmid),
        }
    categories: list[str] = []
    final_ranking = rankings[FINAL_SYSTEM]
    if expected_pmid not in final_ranking[:10]:
        categories.append("retrieval_miss_at_10")
    elif final_ranking[0] != expected_pmid:
        categories.append("retrieval_not_top_1")
    if predictions[FINAL_SYSTEM] != expected_label:
        categories.append(
            f"label_{expected_label}_to_{predictions[FINAL_SYSTEM]}"
        )
    if cited_pmid != expected_pmid:
        categories.append("citation_source_mismatch")
    if not exact_span:
        categories.append("exact_span_failure")
    if not source_hash_valid:
        categories.append("source_hash_failure")
    if not claim_coverage:
        categories.append("uncited_claim")
    if not schema_valid:
        categories.append("schema_failure")
    return {
        "case_id": case_id,
        "question": question,
        "expected_pmid": expected_pmid,
        "expected_label": expected_label,
        "predictions": predictions,
        "retrieval": retrieval,
        "citation_pmid": cited_pmid,
        "schema_valid": schema_valid,
        "citation_source_match": cited_pmid == expected_pmid,
        "exact_span_integrity": exact_span,
        "source_hash_integrity": source_hash_valid,
        "claim_citation_coverage": claim_coverage,
        "failure_categories": categories,
    }


def _retrieval_metrics(
    rows: Sequence[dict[str, object]],
    system: str,
) -> dict[str, float]:
    values = [row["retrieval"][system] for row in rows]
    return {
        "recall_at_5": _mean(values, "recall_at_5"),
        "recall_at_10": _mean(values, "recall_at_10"),
        "mrr": _mean(values, "reciprocal_rank"),
        "ndcg_at_10": _mean(values, "ndcg_at_10"),
    }


def _classification_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, object]:
    per_label: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in PUBMEDQA_LABELS:
        true_positive = sum(
            gold == label and prediction == label
            for gold, prediction in zip(expected, predicted, strict=True)
        )
        false_positive = sum(
            gold != label and prediction == label
            for gold, prediction in zip(expected, predicted, strict=True)
        )
        false_negative = sum(
            gold == label and prediction != label
            for gold, prediction in zip(expected, predicted, strict=True)
        )
        support = sum(gold == label for gold in expected)
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_label[label] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    correct = sum(
        gold == prediction
        for gold, prediction in zip(expected, predicted, strict=True)
    )
    return {
        "correct": correct,
        "total": len(expected),
        "accuracy": round(correct / len(expected), 6),
        "macro_f1": round(statistics.fmean(f1_values), 6),
        "per_label": per_label,
    }


def _grounding_metrics(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    fields = (
        "schema_valid",
        "citation_source_match",
        "exact_span_integrity",
        "source_hash_integrity",
        "claim_citation_coverage",
    )
    return {
        field: {
            "count": sum(bool(row[field]) for row in rows),
            "total": len(rows),
            "rate": round(
                sum(bool(row[field]) for row in rows) / len(rows),
                6,
            ),
        }
        for field in fields
    }


def _failure_counts(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    categories = sorted(
        {
            category
            for row in rows
            for category in row["failure_categories"]
        }
    )
    return {
        category: sum(
            category in row["failure_categories"] for row in rows
        )
        for category in categories
    }


def _failure_examples(
    rows: Sequence[dict[str, object]],
    *,
    minimum: int,
) -> list[dict[str, object]]:
    failures = [row for row in rows if row["failure_categories"]]
    failures.sort(
        key=lambda row: (
            -len(row["failure_categories"]),
            str(row["case_id"]),
        )
    )
    return [
        {
            "case_id": row["case_id"],
            "question": row["question"],
            "expected_pmid": row["expected_pmid"],
            "retrieved_pmid": row["citation_pmid"],
            "expected_label": row["expected_label"],
            "predicted_label": row["predictions"][FINAL_SYSTEM],
            "categories": row["failure_categories"],
        }
        for row in failures[: max(minimum, 10)]
    ]


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean": round(statistics.fmean(ordered), 3),
        "median": round(statistics.median(ordered), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "total": round(sum(ordered), 3),
    }


def _runtime_summary(reranker: PairScorer) -> dict[str, object]:
    function = getattr(reranker, "runtime_summary", None)
    if callable(function):
        return function()
    return {
        "scored_pair_count": getattr(reranker, "scored_pair_count", None),
        "latency_ms": None,
    }


def _markdown_report(report: dict[str, object]) -> str:
    retrieval = report["retrieval_metrics"]
    answers = report["answer_metrics"]
    grounding = report["structural_grounding"]
    latency = report["runtime"]["end_to_end_latency_ms"]
    lines = [
        "# BioEvidence Agent — PubMedQA public closed-corpus report",
        "",
        f"- Benchmark cases: {report['case_count']}",
        f"- Corpus documents: {report['corpus_document_count']}",
        f"- Headline system: `{report['headline_system']}`",
        "- Scope: public, fixed, closed-corpus benchmark; not blind.",
        "",
        "## Retrieval ablation",
        "",
        "| System | Recall@5 | Recall@10 | MRR | nDCG@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("bm25", "vector", "hybrid_rrf", FINAL_SYSTEM):
        metrics = retrieval[name]
        lines.append(
            f"| {name} | {metrics['recall_at_5']:.4f} | "
            f"{metrics['recall_at_10']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['ndcg_at_10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Answer metrics",
            "",
            "| Retrieval input | Accuracy | Macro-F1 |",
            "|---|---:|---:|",
        ]
    )
    for name in ("bm25", "vector", "hybrid_rrf", FINAL_SYSTEM):
        metrics = answers[name]
        lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Structural grounding and runtime",
            "",
            f"- Schema-valid rate: {grounding['schema_valid']['rate']:.4f}",
            (
                "- Target-PMID citation rate: "
                f"{grounding['citation_source_match']['rate']:.4f}"
            ),
            (
                "- Exact-span integrity: "
                f"{grounding['exact_span_integrity']['rate']:.4f}"
            ),
            (
                "- Source-Hash integrity: "
                f"{grounding['source_hash_integrity']['rate']:.4f}"
            ),
            (
                "- Claim citation coverage: "
                f"{grounding['claim_citation_coverage']['rate']:.4f}"
            ),
            f"- Median latency: {latency['median']:.3f} ms/query",
            f"- P95 latency: {latency['p95']:.3f} ms/query",
            "- Mean and p95 model/API cost: USD 0.00.",
            "",
            "These are structural grounding measurements, not independently "
            "reviewed semantic citation precision or hallucination rate.",
            "",
            "## Failure taxonomy",
            "",
        ]
    )
    for category, count in report["failure_taxonomy"].items():
        lines.append(f"- `{category}`: {count}")
    lines.extend(["", "## Representative public failures", ""])
    for example in report["failure_examples"]:
        lines.append(
            f"- `{example['case_id']}` — gold `{example['expected_label']}`, "
            f"predicted `{example['predicted_label']}`; "
            f"{', '.join(example['categories'])}. "
            f"Question: {example['question']}"
        )
    lines.extend(["", "## Interpretation limits", ""])
    for limitation in report["interpretation_limits"]:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _mean(rows: Sequence[dict[str, object]], key: str) -> float:
    return round(
        statistics.fmean(float(row[key]) for row in rows),
        6,
    )


def _reciprocal_rank(ranking: Sequence[str], expected: str) -> float:
    for rank, pmid in enumerate(ranking, start=1):
        if pmid == expected:
            return 1 / rank
    return 0.0


def _ndcg_at_10(ranking: Sequence[str], expected: str) -> float:
    for rank, pmid in enumerate(ranking[:10], start=1):
        if pmid == expected:
            return 1 / math.log2(rank + 1)
    return 0.0


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
