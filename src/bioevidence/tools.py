"""Typed local literature tools with auditable execution traces."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from typing import TypeVar, cast

from .bm25 import tokenize
from .corpus import CorpusDocument
from .retrievers import BM25Retriever, Retriever

T = TypeVar("T")


@dataclass(frozen=True)
class SearchLiteratureInput:
    query: str
    top_k: int = 5
    rerank_query: str | None = None


@dataclass(frozen=True)
class SearchHit:
    pmid: str
    rank: int
    score: float
    document_sha256: str
    retrieval_method: str
    component_ranks: dict[str, int]
    candidate_rank: int | None


@dataclass(frozen=True)
class SearchLiteratureOutput:
    hits: tuple[SearchHit, ...]


@dataclass(frozen=True)
class FetchRecordInput:
    pmid: str


@dataclass(frozen=True)
class RecordOutput:
    pmid: str
    title: str
    abstract: str
    doi: str | None
    publication_types: tuple[str, ...]
    journal: str
    year: int | None
    first_author: str | None
    source_url: str
    document_sha256: str


@dataclass(frozen=True)
class ResolveIdentifierInput:
    identifier: str


@dataclass(frozen=True)
class ResolveIdentifierOutput:
    found: bool
    pmid: str | None
    doi: str | None
    source_url: str | None
    document_sha256: str | None


@dataclass(frozen=True)
class InspectEvidenceInput:
    pmid: str
    query: str
    max_snippets: int = 2


@dataclass(frozen=True)
class EvidenceSnippet:
    text: str
    start_char: int
    end_char: int
    section: str
    snippet_sha256: str
    relevance_score: float


@dataclass(frozen=True)
class InspectEvidenceOutput:
    pmid: str
    document_sha256: str
    snippets: tuple[EvidenceSnippet, ...]


@dataclass(frozen=True)
class ToolCallTrace:
    sequence: int
    call_id: str
    tool_name: str
    input_payload: dict[str, object]
    input_sha256: str
    output_sha256: str | None
    duration_ms: float
    status: str
    retry_count: int
    error: str | None


class LiteratureTools:
    """Local PubMed snapshot tools; no network or gold access."""

    def __init__(
        self,
        documents: list[CorpusDocument],
        *,
        retriever: Retriever | None = None,
    ) -> None:
        self._documents = {document.pmid: document for document in documents}
        self._doi_to_pmid = {
            _normalize_doi(document.doi): document.pmid
            for document in documents
            if document.doi
        }
        self._retriever = retriever or BM25Retriever(documents)

    def search_literature(
        self,
        request: SearchLiteratureInput,
    ) -> SearchLiteratureOutput:
        results = self._retriever.search(
            request.query,
            top_k=request.top_k,
            rerank_query=request.rerank_query,
        )
        return SearchLiteratureOutput(
            hits=tuple(
                SearchHit(
                    pmid=result.pmid,
                    rank=result.rank,
                    score=result.score,
                    document_sha256=result.document_sha256,
                    retrieval_method=result.retrieval_method,
                    component_ranks=result.component_ranks,
                    candidate_rank=result.candidate_rank,
                )
                for result in results
            )
        )

    def fetch_record(self, request: FetchRecordInput) -> RecordOutput:
        document = self._documents.get(request.pmid)
        if document is None:
            raise ValueError(f"PMID is not present in local corpus: {request.pmid}")
        return RecordOutput(
            pmid=document.pmid,
            title=document.title,
            abstract=document.abstract,
            doi=document.doi,
            publication_types=document.publication_types,
            journal=document.journal,
            year=document.year,
            first_author=document.first_author,
            source_url=document.source_url,
            document_sha256=document.content_sha256,
        )

    def resolve_identifier(
        self,
        request: ResolveIdentifierInput,
    ) -> ResolveIdentifierOutput:
        normalized = request.identifier.strip()
        pmid = normalized if normalized in self._documents else None
        if pmid is None:
            pmid = self._doi_to_pmid.get(_normalize_doi(normalized))
        document = self._documents.get(pmid) if pmid else None
        return ResolveIdentifierOutput(
            found=document is not None,
            pmid=document.pmid if document else None,
            doi=document.doi if document else None,
            source_url=document.source_url if document else None,
            document_sha256=document.content_sha256 if document else None,
        )

    def inspect_evidence(
        self,
        request: InspectEvidenceInput,
    ) -> InspectEvidenceOutput:
        if request.max_snippets < 1:
            raise ValueError("max_snippets must be positive")
        document = self._documents.get(request.pmid)
        if document is None:
            raise ValueError(f"PMID is not present in local corpus: {request.pmid}")
        query_tokens = set(tokenize(request.query))
        candidates: list[tuple[int, EvidenceSnippet]] = []
        for start, end, text in _sentence_spans(document.abstract):
            sentence_tokens = set(tokenize(text))
            overlap = len(query_tokens & sentence_tokens)
            if overlap == 0:
                continue
            relevance = overlap / max(1, len(query_tokens))
            candidates.append(
                (
                    _evidence_priority(text),
                    EvidenceSnippet(
                        text=text,
                        start_char=start,
                        end_char=end,
                        section="abstract",
                        snippet_sha256=_sha256_text(text),
                        relevance_score=round(relevance, 8),
                    ),
                )
            )
        candidates.sort(
            key=lambda item: (
                -item[0],
                -item[1].relevance_score,
                item[1].start_char,
                item[1].text,
            )
        )
        return InspectEvidenceOutput(
            pmid=document.pmid,
            document_sha256=document.content_sha256,
            snippets=tuple(
                snippet
                for _, snippet in candidates[: request.max_snippets]
            ),
        )


class ToolExecutor:
    """Bounded typed dispatcher that records every local tool attempt."""

    def __init__(self, tools: LiteratureTools, *, max_calls: int = 8) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self._tools = tools
        self.max_calls = max_calls
        self.trace: list[ToolCallTrace] = []

    def search(self, request: SearchLiteratureInput) -> SearchLiteratureOutput:
        return cast(
            SearchLiteratureOutput,
            self._call("search_literature", request, self._tools.search_literature),
        )

    def fetch(self, request: FetchRecordInput) -> RecordOutput:
        return cast(
            RecordOutput,
            self._call("fetch_record", request, self._tools.fetch_record),
        )

    def resolve(self, request: ResolveIdentifierInput) -> ResolveIdentifierOutput:
        return cast(
            ResolveIdentifierOutput,
            self._call("resolve_identifier", request, self._tools.resolve_identifier),
        )

    def inspect(self, request: InspectEvidenceInput) -> InspectEvidenceOutput:
        return cast(
            InspectEvidenceOutput,
            self._call("inspect_evidence", request, self._tools.inspect_evidence),
        )

    def _call(
        self,
        tool_name: str,
        request: object,
        function: Callable[[object], T],
    ) -> T:
        if len(self.trace) >= self.max_calls:
            raise RuntimeError("tool-call budget exhausted")
        if not is_dataclass(request):
            raise TypeError("tool input must be a dataclass contract")
        sequence = len(self.trace) + 1
        input_payload = cast(dict[str, object], asdict(request))
        started = time.perf_counter()
        try:
            output = function(request)
            output_sha256 = _sha256_json(output)
            status = "success"
            error = None
        except Exception as exc:
            output = None
            output_sha256 = None
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        self.trace.append(
            ToolCallTrace(
                sequence=sequence,
                call_id=f"call-{sequence:03d}",
                tool_name=tool_name,
                input_payload=input_payload,
                input_sha256=_sha256_json(request),
                output_sha256=output_sha256,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                status=status,
                retry_count=0,
                error=error,
            )
        )
        if status == "error":
            raise RuntimeError(error)
        return cast(T, output)


def trace_as_dict(trace: list[ToolCallTrace]) -> list[dict[str, object]]:
    return [asdict(item) for item in trace]


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for boundary in re.finditer(r"(?<=[.!?])\s+|\n+", text):
        end = boundary.start()
        sentence = text[start:end].strip()
        if sentence:
            left_trim = len(text[start:end]) - len(text[start:end].lstrip())
            actual_start = start + left_trim
            spans.append((actual_start, actual_start + len(sentence), sentence))
        start = boundary.end()
    sentence = text[start:].strip()
    if sentence:
        left_trim = len(text[start:]) - len(text[start:].lstrip())
        actual_start = start + left_trim
        spans.append((actual_start, actual_start + len(sentence), sentence))
    return spans


def _normalize_doi(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        normalized = normalized.removeprefix(prefix)
    return normalized


def _evidence_priority(text: str) -> int:
    lowered = text.lower()
    priority = 0
    if re.match(r"^(results?|conclusions?)\s*:", lowered):
        priority += 3
    if re.search(
        r"\b(?:confidence interval|hazard ratio|odds ratio|rate ratio|"
        r"risk ratio)\b|p\s*[<=>]|\b\d+(?:\.\d+)?%",
        lowered,
    ):
        priority += 2
    if re.search(
        r"\b(?:did not|no significant|lower|higher|reduc(?:e|ed|tion)|"
        r"increas(?:e|ed)|improv(?:e|ed|ement)|worse|benefit|harm)\b",
        lowered,
    ):
        priority += 1
    if re.match(r"^(?:background|methods?|objective)\s*:", lowered):
        priority -= 3
    if "primary outcome was" in lowered:
        priority -= 1
    return priority


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: object) -> str:
    if is_dataclass(value):
        value = asdict(value)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
