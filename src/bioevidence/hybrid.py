"""Deterministic reciprocal-rank fusion for lexical and dense retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class HybridSearchResult:
    rank: int
    pmid: str
    score: float
    document_sha256: str
    component_ranks: dict[str, int]


def reciprocal_rank_fusion(
    rankings: dict[str, Sequence[object]],
    *,
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[HybridSearchResult]:
    if top_k < 1 or rrf_k < 1:
        raise ValueError("top_k and rrf_k must be positive")
    if not rankings:
        raise ValueError("at least one ranking is required")
    scores: dict[str, float] = {}
    component_ranks: dict[str, dict[str, int]] = {}
    document_hashes: dict[str, str] = {}
    for component, results in rankings.items():
        seen: set[str] = set()
        for rank, result in enumerate(results, start=1):
            pmid = getattr(result, "pmid", None)
            document_hash = getattr(result, "document_sha256", None)
            if not isinstance(pmid, str) or not isinstance(document_hash, str):
                raise ValueError(f"{component} ranking has an invalid result")
            if pmid in seen:
                raise ValueError(f"{component} ranking contains duplicate PMID {pmid}")
            seen.add(pmid)
            if pmid in document_hashes and document_hashes[pmid] != document_hash:
                raise ValueError(f"document hash disagreement for PMID {pmid}")
            document_hashes[pmid] = document_hash
            scores[pmid] = scores.get(pmid, 0.0) + 1 / (rrf_k + rank)
            component_ranks.setdefault(pmid, {})[component] = rank
    ordered = sorted(scores, key=lambda pmid: (-scores[pmid], int(pmid), pmid))
    return [
        HybridSearchResult(
            rank=rank,
            pmid=pmid,
            score=round(scores[pmid], 10),
            document_sha256=document_hashes[pmid],
            component_ranks=component_ranks[pmid],
        )
        for rank, pmid in enumerate(ordered[:top_k], start=1)
    ]

