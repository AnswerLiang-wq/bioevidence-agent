"""Small deterministic BM25 implementation for the local PubMed corpus."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .corpus import CorpusDocument

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


@dataclass(frozen=True)
class SearchResult:
    rank: int
    pmid: str
    score: float
    document_sha256: str


class BM25Index:
    """Okapi BM25 over immutable in-memory corpus records."""

    def __init__(
        self,
        documents: list[CorpusDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25 requires at least one document")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("invalid BM25 parameters")
        self.documents = tuple(documents)
        self.k1 = k1
        self.b = b
        self._term_frequencies = [
            Counter(tokenize(document.retrieval_text)) for document in documents
        ]
        self._lengths = [sum(counts.values()) for counts in self._term_frequencies]
        self._average_length = sum(self._lengths) / len(self._lengths)
        document_frequencies: Counter[str] = Counter()
        for counts in self._term_frequencies:
            document_frequencies.update(counts.keys())
        self._idf = {
            term: math.log(
                1
                + (len(documents) - frequency + 0.5)
                / (frequency + 0.5)
            )
            for term, frequency in document_frequencies.items()
        }

    def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_terms = set(tokenize(query))
        scored: list[tuple[float, CorpusDocument]] = []
        for document, frequencies, length in zip(
            self.documents,
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
                score += self._idf.get(term, 0.0) * numerator / denominator
            scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], int(item[1].pmid), item[1].pmid))
        return [
            SearchResult(
                rank=rank,
                pmid=document.pmid,
                score=round(score, 8),
                document_sha256=document.content_sha256,
            )
            for rank, (score, document) in enumerate(scored[:top_k], start=1)
        ]
