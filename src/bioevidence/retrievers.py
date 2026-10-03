"""Uniform retriever adapters for BM25, dense, hybrid, and reranked search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .bm25 import BM25Index
from .corpus import CorpusDocument
from .hybrid import reciprocal_rank_fusion
from .reranker import PairScorer, rerank_candidates
from .vector import TextEncoder, VectorIndex


@dataclass(frozen=True)
class RetrievalHit:
    pmid: str
    rank: int
    score: float
    document_sha256: str
    retrieval_method: str
    component_ranks: dict[str, int]
    candidate_rank: int | None


class Retriever(Protocol):
    method: str
    metadata: dict[str, object]

    def search(
        self,
        query: str,
        *,
        top_k: int,
        rerank_query: str | None = None,
    ) -> list[RetrievalHit]: ...


class BM25Retriever:
    method = "bm25"

    def __init__(self, documents: Sequence[CorpusDocument]) -> None:
        self._index = BM25Index(documents)
        self.metadata = {
            "method": self.method,
            "k1": self._index.k1,
            "b": self._index.b,
        }

    def search(
        self,
        query: str,
        *,
        top_k: int,
        rerank_query: str | None = None,
    ) -> list[RetrievalHit]:
        del rerank_query
        return [
            RetrievalHit(
                pmid=result.pmid,
                rank=result.rank,
                score=result.score,
                document_sha256=result.document_sha256,
                retrieval_method=self.method,
                component_ranks={"bm25": result.rank},
                candidate_rank=None,
            )
            for result in self._index.search(query, top_k=top_k)
        ]


class VectorRetriever:
    method = "vector_e5"

    def __init__(self, index: VectorIndex, encoder: TextEncoder) -> None:
        self._index = index
        self._encoder = encoder
        self.metadata = {
            "method": self.method,
            "model_id": index.model_id,
            "model_revision": index.model_revision,
            "max_length": index.max_length,
        }

    def search(
        self,
        query: str,
        *,
        top_k: int,
        rerank_query: str | None = None,
    ) -> list[RetrievalHit]:
        del rerank_query
        return [
            RetrievalHit(
                pmid=result.pmid,
                rank=result.rank,
                score=result.score,
                document_sha256=result.document_sha256,
                retrieval_method=self.method,
                component_ranks={"vector": result.rank},
                candidate_rank=None,
            )
            for result in self._index.search(
                query,
                encoder=self._encoder,
                top_k=top_k,
            )
        ]


class HybridRetriever:
    method = "hybrid_rrf"

    def __init__(
        self,
        documents: Sequence[CorpusDocument],
        vector_index: VectorIndex,
        encoder: TextEncoder,
        *,
        rrf_k: int = 60,
    ) -> None:
        self._documents = tuple(documents)
        self._bm25 = BM25Index(documents)
        self._vector_index = vector_index
        self._encoder = encoder
        self._rrf_k = rrf_k
        self.metadata = {
            "method": self.method,
            "rrf_k": rrf_k,
            "components": {
                "bm25": {"k1": self._bm25.k1, "b": self._bm25.b},
                "vector": {
                    "model_id": vector_index.model_id,
                    "model_revision": vector_index.model_revision,
                    "max_length": vector_index.max_length,
                },
            },
        }

    def search(
        self,
        query: str,
        *,
        top_k: int,
        rerank_query: str | None = None,
    ) -> list[RetrievalHit]:
        del rerank_query
        component_k = len(self._documents)
        lexical = self._bm25.search(query, top_k=component_k)
        dense = self._vector_index.search(
            query,
            encoder=self._encoder,
            top_k=component_k,
        )
        fused = reciprocal_rank_fusion(
            {"bm25": lexical, "vector": dense},
            top_k=top_k,
            rrf_k=self._rrf_k,
        )
        return [
            RetrievalHit(
                pmid=result.pmid,
                rank=result.rank,
                score=result.score,
                document_sha256=result.document_sha256,
                retrieval_method=self.method,
                component_ranks=result.component_ranks,
                candidate_rank=None,
            )
            for result in fused
        ]


class RerankedHybridRetriever:
    method = "hybrid_rrf_cross_encoder"

    def __init__(
        self,
        documents: Sequence[CorpusDocument],
        hybrid: HybridRetriever,
        scorer: PairScorer,
        *,
        candidate_k: int = 20,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        self._documents_by_pmid = {
            document.pmid: document for document in documents
        }
        self._hybrid = hybrid
        self._scorer = scorer
        self._candidate_k = min(candidate_k, len(documents))
        self.metadata = {
            "method": self.method,
            "candidate_k": self._candidate_k,
            "hybrid": hybrid.metadata,
            "reranker": {
                "model_id": scorer.model_id,
                "model_revision": scorer.model_revision,
                "max_length": scorer.max_length,
                "batch_size": scorer.batch_size,
                "device": scorer.device,
                "query_strategy": "explicit_rerank_query_or_retrieval_query",
            },
        }

    def search(
        self,
        query: str,
        *,
        top_k: int,
        rerank_query: str | None = None,
    ) -> list[RetrievalHit]:
        candidates = self._hybrid.search(query, top_k=self._candidate_k)
        reranked = rerank_candidates(
            rerank_query or query,
            candidates,
            documents_by_pmid=self._documents_by_pmid,
            scorer=self._scorer,
            top_k=top_k,
        )
        return [
            RetrievalHit(
                pmid=result.pmid,
                rank=result.rank,
                score=result.score,
                document_sha256=result.document_sha256,
                retrieval_method=self.method,
                component_ranks=result.component_ranks,
                candidate_rank=result.candidate_rank,
            )
            for result in reranked
        ]
