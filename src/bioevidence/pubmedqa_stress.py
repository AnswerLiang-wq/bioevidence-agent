"""Title-free PubMedQA retrieval stress test with frozen pre-gold rankings."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import tokenize
from .corpus import CorpusDocument, read_corpus, sha256_file
from .hybrid import reciprocal_rank_fusion
from .pubmedqa import (
    CORPUS_FILENAME,
    MANIFEST_FILENAME,
    TEST_GOLD_FILENAME,
    TEST_INPUTS_FILENAME,
    read_jsonl,
)
from .reranker import PairScorer, RerankResult
from .vector import TextEncoder

ABSTRACT_ONLY_BASELINE_ID = "bioevidence-pubmedqa-abstract-only-retrieval-v1"
ABSTRACT_INDEX_MANIFEST = "manifest.json"
ABSTRACT_INDEX_EMBEDDINGS = "embeddings.npy"
SYSTEMS = ("bm25", "vector", "hybrid_rrf", "hybrid_reranked")


@dataclass(frozen=True)
class AbstractPassage:
    """The complete retrieval-side document view; deliberately has no title."""

    pmid: str
    abstract: str
    document_sha256: str


@dataclass(frozen=True)
class AbstractSearchResult:
    rank: int
    pmid: str
    score: float
    document_sha256: str


class AbstractOnlyBM25:
    """Okapi BM25 whose indexable view contains abstracts and nothing else."""

    def __init__(
        self,
        passages: Sequence[AbstractPassage],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not passages:
            raise ValueError("abstract-only BM25 requires passages")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("invalid BM25 parameters")
        self.passages = tuple(passages)
        self.k1 = k1
        self.b = b
        self._term_frequencies = [
            Counter(tokenize(passage.abstract)) for passage in passages
        ]
        self._lengths = [sum(row.values()) for row in self._term_frequencies]
        self._average_length = sum(self._lengths) / len(self._lengths)
        document_frequencies: Counter[str] = Counter()
        for row in self._term_frequencies:
            document_frequencies.update(row)
        self._idf = {
            term: math.log(
                1
                + (len(passages) - frequency + 0.5)
                / (frequency + 0.5)
            )
            for term, frequency in document_frequencies.items()
        }

    def search(self, query: str, *, top_k: int) -> list[AbstractSearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_terms = set(tokenize(query))
        scored: list[tuple[float, AbstractPassage]] = []
        for passage, frequencies, length in zip(
            self.passages,
            self._term_frequencies,
            self._lengths,
            strict=True,
        ):
            score = 0.0
            length_factor = 1 - self.b + self.b * length / self._average_length
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                numerator = term_frequency * (self.k1 + 1)
                denominator = term_frequency + self.k1 * length_factor
                score += self._idf[term] * numerator / denominator
            scored.append((score, passage))
        scored.sort(key=lambda item: (-item[0], int(item[1].pmid)))
        return [
            AbstractSearchResult(
                rank=rank,
                pmid=passage.pmid,
                score=round(score, 8),
                document_sha256=passage.document_sha256,
            )
            for rank, (score, passage) in enumerate(scored[:top_k], start=1)
        ]


class AbstractOnlyVectorIndex:
    """Dense index binding E5 embeddings to the title-free passage view."""

    def __init__(
        self,
        *,
        passages: Sequence[AbstractPassage],
        embeddings: np.ndarray,
        model_id: str,
        model_revision: str,
        max_length: int,
    ) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(passages):
            raise ValueError("embedding rows must match abstract passages")
        if not passages:
            raise ValueError("abstract-only vector index requires passages")
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4):
            raise ValueError("document embeddings must be L2-normalized")
        self.passages = tuple(passages)
        self.embeddings = matrix
        self.model_id = model_id
        self.model_revision = model_revision
        self.max_length = max_length

    @classmethod
    def build(
        cls,
        passages: Sequence[AbstractPassage],
        *,
        encoder: TextEncoder,
        batch_size: int,
    ) -> AbstractOnlyVectorIndex:
        embeddings = encoder.encode(
            [passage.abstract for passage in passages],
            input_type="passage",
            batch_size=batch_size,
        )
        return cls(
            passages=passages,
            embeddings=embeddings,
            model_id=encoder.model_id,
            model_revision=encoder.model_revision,
            max_length=encoder.max_length,
        )

    def search(
        self,
        query: str,
        *,
        encoder: TextEncoder,
        top_k: int,
    ) -> list[AbstractSearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if (
            encoder.model_id != self.model_id
            or encoder.model_revision != self.model_revision
        ):
            raise ValueError("query encoder does not match abstract-only index")
        query_vector = encoder.encode([query], input_type="query")
        if query_vector.shape != (1, self.embeddings.shape[1]):
            raise ValueError("query embedding dimension does not match index")
        scores = self.embeddings @ query_vector[0]
        order = sorted(
            range(len(self.passages)),
            key=lambda index: (
                -float(scores[index]),
                int(self.passages[index].pmid),
            ),
        )
        return [
            AbstractSearchResult(
                rank=rank,
                pmid=self.passages[index].pmid,
                score=round(float(scores[index]), 8),
                document_sha256=self.passages[index].document_sha256,
            )
            for rank, index in enumerate(order[:top_k], start=1)
        ]

    def save(
        self,
        index_dir: Path,
        *,
        corpus_path: Path,
        batch_size: int,
    ) -> dict[str, object]:
        if index_dir.exists():
            raise FileExistsError(f"abstract-only index is immutable: {index_dir}")
        index_dir.mkdir(parents=True)
        embeddings_path = index_dir / ABSTRACT_INDEX_EMBEDDINGS
        np.save(embeddings_path, self.embeddings, allow_pickle=False)
        manifest = {
            "manifest_version": "1.0.0",
            "index_id": "bioevidence-pubmedqa-abstract-only-e5-v1",
            "baseline_id": ABSTRACT_ONLY_BASELINE_ID,
            "retrieval_document_view": ["abstract"],
            "excluded_document_fields": [
                "pmid as model text",
                "title",
                "label",
                "LONG_ANSWER",
                "final_decision",
                "source target",
            ],
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "max_length": self.max_length,
            "batch_size": batch_size,
            "embedding_dimension": int(self.embeddings.shape[1]),
            "normalized": True,
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "pooling": "attention-mask mean pooling",
            "similarity": "cosine via normalized dot product",
            "parent_corpus_sha256": sha256_file(corpus_path),
            "embeddings_sha256": sha256_file(embeddings_path),
            "document_order": [passage.pmid for passage in self.passages],
            "abstract_sha256": {
                passage.pmid: _sha256_text(passage.abstract)
                for passage in self.passages
            },
        }
        _write_json(index_dir / ABSTRACT_INDEX_MANIFEST, manifest)
        return manifest

    @classmethod
    def load(
        cls,
        index_dir: Path,
        *,
        passages: Sequence[AbstractPassage],
        corpus_path: Path,
    ) -> AbstractOnlyVectorIndex:
        manifest = json.loads(
            (index_dir / ABSTRACT_INDEX_MANIFEST).read_text(encoding="utf-8")
        )
        embeddings_path = index_dir / ABSTRACT_INDEX_EMBEDDINGS
        if manifest.get("retrieval_document_view") != ["abstract"]:
            raise ValueError("abstract-only index view is not frozen to abstract")
        if manifest.get("parent_corpus_sha256") != sha256_file(corpus_path):
            raise ValueError("abstract-only index corpus hash mismatch")
        if manifest.get("embeddings_sha256") != sha256_file(embeddings_path):
            raise ValueError("abstract-only embedding hash mismatch")
        if manifest.get("document_order") != [
            passage.pmid for passage in passages
        ]:
            raise ValueError("abstract-only document order mismatch")
        if manifest.get("abstract_sha256") != {
            passage.pmid: _sha256_text(passage.abstract)
            for passage in passages
        }:
            raise ValueError("abstract-only passage hash mismatch")
        embeddings = np.load(embeddings_path, allow_pickle=False)
        return cls(
            passages=passages,
            embeddings=embeddings,
            model_id=_required_manifest_text(manifest, "model_id"),
            model_revision=_required_manifest_text(manifest, "model_revision"),
            max_length=int(manifest["max_length"]),
        )


def abstract_passages(
    documents: Sequence[CorpusDocument],
) -> list[AbstractPassage]:
    """Project corpus records into the only view allowed to retrieval models."""

    return [
        AbstractPassage(
            pmid=document.pmid,
            abstract=document.abstract,
            document_sha256=document.content_sha256,
        )
        for document in documents
    ]


def rerank_abstract_candidates(
    query: str,
    candidates: Sequence[object],
    *,
    passages_by_pmid: dict[str, AbstractPassage],
    scorer: PairScorer,
    top_k: int,
) -> list[RerankResult]:
    """Rerank using raw abstract strings, never document titles."""

    values: list[tuple[int, str, str, dict[str, int]]] = []
    model_passages: list[str] = []
    seen: set[str] = set()
    for candidate_rank, candidate in enumerate(candidates, start=1):
        pmid = getattr(candidate, "pmid", None)
        document_hash = getattr(candidate, "document_sha256", None)
        component_ranks = getattr(candidate, "component_ranks", None)
        if not isinstance(pmid, str) or pmid in seen:
            raise ValueError("reranker candidates must contain unique PMIDs")
        passage = passages_by_pmid.get(pmid)
        if passage is None or passage.document_sha256 != document_hash:
            raise ValueError(f"abstract-only candidate identity mismatch: {pmid}")
        if not isinstance(component_ranks, dict):
            raise ValueError("candidate component ranks are invalid")
        seen.add(pmid)
        model_passages.append(passage.abstract)
        values.append(
            (candidate_rank, pmid, document_hash, dict(component_ranks))
        )
    scores = scorer.score(query, model_passages)
    if len(scores) != len(values):
        raise ValueError("reranker score count does not match candidate count")
    ranked = sorted(
        zip(scores, values, strict=True),
        key=lambda item: (-item[0], item[1][0], int(item[1][1])),
    )
    return [
        RerankResult(
            rank=rank,
            pmid=pmid,
            score=round(float(score), 8),
            candidate_rank=candidate_rank,
            document_sha256=document_hash,
            component_ranks=component_ranks,
        )
        for rank, (
            score,
            (candidate_rank, pmid, document_hash, component_ranks),
        ) in enumerate(ranked[:top_k], start=1)
    ]


def run_abstract_only_rankings(
    *,
    benchmark_dir: Path,
    index_dir: Path,
    rankings_path: Path,
    ranking_manifest_path: Path,
    encoder: TextEncoder,
    reranker: PairScorer,
    reranker_candidate_k: int = 20,
) -> dict[str, object]:
    """Freeze rankings from gold-free inputs before any scorer sees targets."""

    for path in (rankings_path, ranking_manifest_path):
        if path.exists():
            raise FileExistsError(f"abstract-only ranking is immutable: {path}")
    benchmark_manifest = _verified_benchmark_manifest(benchmark_dir)
    documents = read_corpus(benchmark_dir / CORPUS_FILENAME)
    passages = abstract_passages(documents)
    passages_by_pmid = {passage.pmid: passage for passage in passages}
    inputs = read_jsonl(benchmark_dir / TEST_INPUTS_FILENAME)
    if len(inputs) != 500 or any(
        set(row) != {"case_id", "question"} for row in inputs
    ):
        raise ValueError("abstract-only runner requires 500 gold-free inputs")
    vector_index = AbstractOnlyVectorIndex.load(
        index_dir,
        passages=passages,
        corpus_path=benchmark_dir / CORPUS_FILENAME,
    )
    bm25 = AbstractOnlyBM25(passages)
    rows: list[dict[str, object]] = []
    for input_row in inputs:
        question = str(input_row["question"])
        bm25_started = time.perf_counter()
        lexical = bm25.search(question, top_k=len(passages))
        bm25_ms = (time.perf_counter() - bm25_started) * 1000

        vector_started = time.perf_counter()
        dense = vector_index.search(
            question,
            encoder=encoder,
            top_k=len(passages),
        )
        vector_ms = (time.perf_counter() - vector_started) * 1000

        fusion_started = time.perf_counter()
        hybrid = reciprocal_rank_fusion(
            {"bm25": lexical, "vector": dense},
            top_k=len(passages),
        )
        fusion_ms = (time.perf_counter() - fusion_started) * 1000

        reranker_started = time.perf_counter()
        reranked_head = rerank_abstract_candidates(
            question,
            hybrid[:reranker_candidate_k],
            passages_by_pmid=passages_by_pmid,
            scorer=reranker,
            top_k=reranker_candidate_k,
        )
        reranker_ms = (time.perf_counter() - reranker_started) * 1000
        reranked_pmids = [item.pmid for item in reranked_head]
        reranked_pmids.extend(
            item.pmid for item in hybrid if item.pmid not in set(reranked_pmids)
        )
        hybrid_ms = bm25_ms + vector_ms + fusion_ms
        rows.append(
            {
                "case_id": input_row["case_id"],
                "rankings": {
                    "bm25": [item.pmid for item in lexical],
                    "vector": [item.pmid for item in dense],
                    "hybrid_rrf": [item.pmid for item in hybrid],
                    "hybrid_reranked": reranked_pmids,
                },
                "latency_ms": {
                    "bm25": bm25_ms,
                    "vector": vector_ms,
                    "hybrid_rrf": hybrid_ms,
                    "hybrid_reranked": hybrid_ms + reranker_ms,
                },
            }
        )
    _write_jsonl(rankings_path, rows)
    manifest = {
        "manifest_version": "1.0.0",
        "baseline_id": ABSTRACT_ONLY_BASELINE_ID,
        "phase": "rankings_frozen_before_target_scoring",
        "case_count": len(rows),
        "runner_fields": ["case_id", "question"],
        "retrieval_document_view": ["abstract"],
        "forbidden_model_inputs": [
            "PMID",
            "title",
            "gold label",
            "LONG_ANSWER",
            "final_decision",
            "source target",
        ],
        "inputs": {
            "benchmark_manifest_sha256": sha256_file(
                benchmark_dir / MANIFEST_FILENAME
            ),
            "parent_corpus_sha256": sha256_file(
                benchmark_dir / CORPUS_FILENAME
            ),
            "test_inputs_sha256": sha256_file(
                benchmark_dir / TEST_INPUTS_FILENAME
            ),
            "abstract_index_manifest_sha256": sha256_file(
                index_dir / ABSTRACT_INDEX_MANIFEST
            ),
            "rankings_sha256": sha256_file(rankings_path),
        },
        "config": {
            "bm25": {"k1": bm25.k1, "b": bm25.b},
            "fusion": {"method": "RRF", "rrf_k": 60, "equal_weight": True},
            "vector": {
                "model_id": vector_index.model_id,
                "model_revision": vector_index.model_revision,
                "max_length": vector_index.max_length,
                "batch_size": int(
                    json.loads(
                        (index_dir / ABSTRACT_INDEX_MANIFEST).read_text(
                            encoding="utf-8"
                        )
                    )["batch_size"]
                ),
            },
            "reranker": {
                "model_id": reranker.model_id,
                "model_revision": reranker.model_revision,
                "max_length": reranker.max_length,
                "batch_size": reranker.batch_size,
                "device": reranker.device,
                "candidate_k": reranker_candidate_k,
            },
        },
        "source_benchmark_id": benchmark_manifest["benchmark_id"],
    }
    _write_json(ranking_manifest_path, manifest)
    return manifest


def score_abstract_only_rankings(
    *,
    benchmark_dir: Path,
    rankings_path: Path,
    ranking_manifest_path: Path,
    title_assisted_report_path: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, object]:
    """Score already-frozen rankings; target PMIDs enter only in this phase."""

    for path in (output_json, output_markdown):
        if path.exists():
            raise FileExistsError(f"abstract-only report is immutable: {path}")
    manifest = json.loads(ranking_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("phase") != "rankings_frozen_before_target_scoring":
        raise ValueError("abstract-only ranking manifest has invalid phase")
    if manifest["inputs"]["rankings_sha256"] != sha256_file(rankings_path):
        raise ValueError("abstract-only rankings changed after freeze")
    ranking_rows = read_jsonl(rankings_path)
    gold_rows = read_jsonl(benchmark_dir / TEST_GOLD_FILENAME)
    if len(ranking_rows) != 500 or len(gold_rows) != 500:
        raise ValueError("abstract-only scoring requires 500 rankings and gold rows")
    rankings_by_id = {str(row["case_id"]): row for row in ranking_rows}
    gold_by_id = {str(row["case_id"]): row for row in gold_rows}
    if set(rankings_by_id) != set(gold_by_id) or len(gold_by_id) != 500:
        raise ValueError("ranking and target case IDs differ")

    case_results: list[dict[str, object]] = []
    for case_id in sorted(rankings_by_id):
        row = rankings_by_id[case_id]
        expected_pmid = str(gold_by_id[case_id]["pmid"])
        per_system: dict[str, object] = {}
        for system in SYSTEMS:
            ranking = row["rankings"][system]
            if not isinstance(ranking, list) or len(ranking) != 1000:
                raise ValueError(f"{case_id}: incomplete {system} ranking")
            rank = ranking.index(expected_pmid) + 1
            per_system[system] = {
                "rank": rank,
                "recall_at_5": float(rank <= 5),
                "recall_at_10": float(rank <= 10),
                "reciprocal_rank": 1 / rank,
                "ndcg_at_10": 1 / math.log2(rank + 1) if rank <= 10 else 0.0,
            }
        case_results.append(
            {
                "case_id": case_id,
                "expected_pmid": expected_pmid,
                "retrieval": per_system,
            }
        )
    metrics = {
        system: _retrieval_metrics(case_results, system) for system in SYSTEMS
    }
    latency = {
        system: _distribution(
            [float(row["latency_ms"][system]) for row in ranking_rows]
        )
        for system in SYSTEMS
    }
    title_assisted = json.loads(
        title_assisted_report_path.read_text(encoding="utf-8")
    )
    comparison = {
        system: {
            "title_assisted": title_assisted["retrieval_metrics"][system],
            "abstract_only": metrics[system],
            "mrr_delta_abstract_minus_title": round(
                metrics[system]["mrr"]
                - title_assisted["retrieval_metrics"][system]["mrr"],
                6,
            ),
        }
        for system in SYSTEMS
    }
    drops = sorted(
        (
            {
                "case_id": row["case_id"],
                "expected_pmid": row["expected_pmid"],
                "abstract_only_rank": row["retrieval"]["hybrid_reranked"]["rank"],
                "title_assisted_rank": 1,
                "rank_drop": row["retrieval"]["hybrid_reranked"]["rank"] - 1,
                "category": (
                    "miss_at_10"
                    if row["retrieval"]["hybrid_reranked"]["rank"] > 10
                    else "rank_drop"
                ),
            }
            for row in case_results
            if row["retrieval"]["hybrid_reranked"]["rank"] > 1
        ),
        key=lambda row: (-int(row["rank_drop"]), str(row["case_id"])),
    )
    report = {
        "report_version": "1.0.0",
        "baseline_id": ABSTRACT_ONLY_BASELINE_ID,
        "benchmark_id": manifest["source_benchmark_id"],
        "evaluation_scope": (
            "500 official public PubMedQA test questions over the frozen "
            "1,000-document corpus; model-side passages contain abstracts only"
        ),
        "case_count": 500,
        "retrieval_metrics": metrics,
        "latency_ms": latency,
        "latency_scope": (
            "model load and one-time index build excluded; hybrid totals add "
            "per-query BM25, vector and fusion time; reranked totals also add "
            "cross-encoder time"
        ),
        "comparison_with_title_assisted_v1": comparison,
        "difficulty_statement": (
            "The v1 title-assisted and v1 abstract-only conditions are "
            "different retrieval tasks. The former exposes the title-derived "
            "question inside the indexed document; the latter does not."
        ),
        "rank_drop_count": len(drops),
        "failure_examples": drops[: max(10, min(len(drops), 20))],
        "inputs": {
            "ranking_manifest_sha256": sha256_file(ranking_manifest_path),
            "rankings_sha256": sha256_file(rankings_path),
            "test_gold_sha256": sha256_file(
                benchmark_dir / TEST_GOLD_FILENAME
            ),
            "title_assisted_report_sha256": sha256_file(
                title_assisted_report_path
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "claims_not_made": [
            "blind or independent generalization performance",
            "open-world literature retrieval completeness",
            "clinical validity",
        ],
    }
    _write_json(output_json, report)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_markdown_report(report), encoding="utf-8")
    return report


def _verified_benchmark_manifest(
    benchmark_dir: Path,
) -> dict[str, object]:
    manifest = json.loads(
        (benchmark_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("PubMedQA manifest is missing file hashes")
    for name in (CORPUS_FILENAME, TEST_INPUTS_FILENAME):
        if files.get(name) != sha256_file(benchmark_dir / name):
            raise ValueError(f"PubMedQA artifact hash mismatch: {name}")
    return manifest


def _retrieval_metrics(
    rows: Sequence[dict[str, object]],
    system: str,
) -> dict[str, float]:
    values = [row["retrieval"][system] for row in rows]
    return {
        "recall_at_5": round(
            statistics.fmean(float(row["recall_at_5"]) for row in values),
            6,
        ),
        "recall_at_10": round(
            statistics.fmean(float(row["recall_at_10"]) for row in values),
            6,
        ),
        "mrr": round(
            statistics.fmean(float(row["reciprocal_rank"]) for row in values),
            6,
        ),
        "ndcg_at_10": round(
            statistics.fmean(float(row["ndcg_at_10"]) for row in values),
            6,
        ),
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "median": round(statistics.median(ordered), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# PubMedQA abstract-only retrieval stress test",
        "",
        f"- Baseline: `{report['baseline_id']}`",
        f"- Public test cases: {report['case_count']}",
        "- Passage view: abstract only; titles are unavailable to every retriever.",
        "",
        "## Results",
        "",
        "| System | Recall@5 | Recall@10 | MRR | nDCG@10 | Median ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        metrics = report["retrieval_metrics"][system]
        latency = report["latency_ms"][system]
        lines.append(
            f"| {system} | {metrics['recall_at_5']:.4f} | "
            f"{metrics['recall_at_10']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['ndcg_at_10']:.4f} | {latency['median']:.3f} | "
            f"{latency['p95']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["difficulty_statement"],
            "",
            (
                f"The final reranked system had {report['rank_drop_count']} "
                "cases below rank 1. Poor results are retained, not tuned away."
            ),
            "",
            "## Representative rank drops",
            "",
        ]
    )
    for row in report["failure_examples"]:
        lines.append(
            f"- `{row['case_id']}` / PMID `{row['expected_pmid']}`: "
            f"rank {row['abstract_only_rank']} ({row['category']})."
        )
    lines.extend(
        [
            "",
            "This remains a public, fixed, closed-corpus diagnostic. It is not "
            "a blind estimate or an open-world search-completeness claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _required_manifest_text(
    manifest: dict[str, object],
    key: str,
) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"abstract-only manifest missing {key}")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
