"""Multilingual cross-encoder reranking over hybrid retrieval candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from .corpus import CorpusDocument


DEFAULT_RERANKER_MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_RERANKER_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"


class PairScorer(Protocol):
    model_id: str
    model_revision: str
    max_length: int
    batch_size: int
    device: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderScorer:
    """Transformers scorer for a single-logit cross-encoder."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_RERANKER_MODEL_ID,
        revision: str,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "reranking requires the 'retrieval' optional dependencies"
            ) from exc
        self._torch = torch
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=revision,
        )
        self._model.to(device)
        self._model.eval()
        if int(self._model.config.num_labels) != 1:
            raise ValueError("reranker must expose exactly one relevance logit")
        self.model_revision = (
            getattr(self._model.config, "_commit_hash", None) or revision
        )
        self.latencies_ms: list[float] = []
        self.scored_pair_count = 0

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        started = time.perf_counter()
        scores: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            encoded = self._tokenizer(
                [query] * len(batch),
                list(batch),
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.no_grad():
                logits = self._model(**encoded).logits
            scores.extend(float(value) for value in logits[:, 0].cpu().tolist())
        self.latencies_ms.append((time.perf_counter() - started) * 1000)
        self.scored_pair_count += len(passages)
        return scores

    def runtime_summary(self) -> dict[str, object]:
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "call_count": len(self.latencies_ms),
            "scored_pair_count": self.scored_pair_count,
            "median_latency_ms": _percentile(self.latencies_ms, 50),
            "p95_latency_ms": _percentile(self.latencies_ms, 95),
            "latency_scope": "reranker scoring only; model load excluded",
        }


@dataclass(frozen=True)
class RerankResult:
    rank: int
    pmid: str
    score: float
    candidate_rank: int
    document_sha256: str
    component_ranks: dict[str, int]


def rerank_candidates(
    query: str,
    candidates: Sequence[object],
    *,
    documents_by_pmid: dict[str, CorpusDocument],
    scorer: PairScorer,
    top_k: int = 10,
) -> list[RerankResult]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    passages: list[str] = []
    candidate_values: list[tuple[int, str, str, dict[str, int]]] = []
    seen: set[str] = set()
    for candidate_rank, candidate in enumerate(candidates, start=1):
        pmid = getattr(candidate, "pmid", None)
        document_hash = getattr(candidate, "document_sha256", None)
        component_ranks = getattr(candidate, "component_ranks", {})
        if not isinstance(pmid, str) or pmid in seen:
            raise ValueError("reranker candidates must have unique PMID values")
        document = documents_by_pmid.get(pmid)
        if document is None:
            raise ValueError(f"reranker candidate PMID is not in corpus: {pmid}")
        if document_hash != document.content_sha256:
            raise ValueError(f"reranker candidate hash mismatch for PMID {pmid}")
        if not isinstance(component_ranks, dict):
            raise ValueError("reranker candidate component ranks are invalid")
        seen.add(pmid)
        passages.append(f"Title: {document.title}\nAbstract: {document.abstract}")
        candidate_values.append(
            (candidate_rank, pmid, document_hash, dict(component_ranks))
        )
    scores = scorer.score(query, passages)
    if len(scores) != len(candidate_values):
        raise ValueError("reranker score count does not match candidate count")
    ranked = sorted(
        zip(scores, candidate_values, strict=True),
        key=lambda item: (-item[0], item[1][0], int(item[1][1]), item[1][1]),
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
        for rank, (score, (candidate_rank, pmid, document_hash, component_ranks))
        in enumerate(ranked[:top_k], start=1)
    ]


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return round(value, 3)
