"""Deterministic local corpus records for product-oriented retrieval."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .pubmed import PubMedArticle, PubMedClient

CORPUS_SCHEMA_VERSION = "1.0.0"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


@dataclass(frozen=True)
class CorpusDocument:
    """Normalized PubMed document with a verifiable content hash."""

    pmid: str
    title: str
    abstract: str
    doi: str | None
    publication_types: tuple[str, ...]
    journal: str
    year: int | None
    first_author: str | None
    source_url: str
    content_sha256: str

    @classmethod
    def from_pubmed(cls, article: PubMedArticle) -> CorpusDocument:
        payload = {
            "pmid": article.pmid.strip(),
            "title": _clean_text(article.title),
            "abstract": _clean_text(article.abstract),
            "doi": article.doi.strip() if article.doi else None,
            "publication_types": tuple(
                _clean_text(value) for value in article.publication_types if value.strip()
            ),
            "journal": _clean_text(article.journal),
            "year": article.year,
            "first_author": (
                _clean_text(article.first_author) if article.first_author else None
            ),
            "source_url": article.source_url,
        }
        content_sha256 = sha256_bytes(_canonical_json(payload))
        return cls(**payload, content_sha256=content_sha256)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> CorpusDocument:
        publication_types = value.get("publication_types")
        if not isinstance(publication_types, list):
            raise ValueError("publication_types must be a list")
        document = cls(
            pmid=_required_text(value, "pmid"),
            title=_required_text(value, "title"),
            abstract=_optional_text(value, "abstract") or "",
            doi=_optional_text(value, "doi"),
            publication_types=tuple(
                item
                for item in publication_types
                if isinstance(item, str) and item.strip()
            ),
            journal=_optional_text(value, "journal") or "",
            year=_optional_year(value.get("year")),
            first_author=_optional_text(value, "first_author"),
            source_url=_required_text(value, "source_url"),
            content_sha256=_required_text(value, "content_sha256"),
        )
        if document.content_sha256 != document.compute_content_sha256():
            raise ValueError(f"content hash mismatch for PMID {document.pmid}")
        return document

    def compute_content_sha256(self) -> str:
        payload = asdict(self)
        payload.pop("content_sha256")
        return sha256_bytes(_canonical_json(payload))

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["publication_types"] = list(self.publication_types)
        return value

    @property
    def retrieval_text(self) -> str:
        return " ".join(
            value
            for value in (
                self.title,
                self.title,
                self.abstract,
                self.first_author or "",
                self.journal,
                " ".join(self.publication_types),
                str(self.year or ""),
            )
            if value
        )


def read_corpus(path: Path) -> list[CorpusDocument]:
    documents: list[CorpusDocument] = []
    seen_pmids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            document = CorpusDocument.from_dict(value)
            if document.pmid in seen_pmids:
                raise ValueError(f"duplicate PMID in corpus: {document.pmid}")
            seen_pmids.add(document.pmid)
            documents.append(document)
    if not documents:
        raise ValueError("corpus is empty")
    return documents


def write_corpus(path: Path, documents: Iterable[CorpusDocument]) -> None:
    ordered = sorted(documents, key=lambda item: (int(item.pmid), item.pmid))
    if not ordered:
        raise ValueError("corpus is empty")
    if len({item.pmid for item in ordered}) != len(ordered):
        raise ValueError("corpus contains duplicate PMIDs")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True)
        for item in ordered
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_document_sets(
    document_sets: Sequence[Sequence[CorpusDocument]],
) -> list[CorpusDocument]:
    """Deduplicate identical PMIDs and reject conflicting corpus records."""

    if not document_sets:
        raise ValueError("at least one corpus is required")
    by_pmid: dict[str, CorpusDocument] = {}
    for documents in document_sets:
        for document in documents:
            existing = by_pmid.get(document.pmid)
            if existing is None:
                by_pmid[document.pmid] = document
            elif existing.content_sha256 != document.content_sha256:
                raise ValueError(
                    f"conflicting content for duplicate PMID {document.pmid}"
                )
    if not by_pmid:
        raise ValueError("merged corpus is empty")
    return sorted(by_pmid.values(), key=lambda item: (int(item.pmid), item.pmid))


def resolve_pubmed_queries(
    *,
    client: PubMedClient,
    queries: Sequence[str],
    retmax_per_query: int,
) -> dict[str, object]:
    """Resolve discovery queries once into a frozen corpus-fetch request."""

    cleaned = [" ".join(query.split()) for query in queries if query.strip()]
    if not cleaned or len(set(cleaned)) != len(cleaned):
        raise ValueError("queries must be a non-empty unique list")
    if not 1 <= retmax_per_query <= 1000:
        raise ValueError("retmax_per_query must be between 1 and 1000")
    query_results: dict[str, list[str]] = {}
    all_pmids: set[str] = set()
    for query in sorted(cleaned):
        pmids = client.search(query, retmax=retmax_per_query)
        if any(not pmid.isdigit() or int(pmid) < 1 for pmid in pmids):
            raise ValueError(f"PubMed query returned an invalid PMID: {query}")
        unique_pmids = sorted(set(pmids), key=int)
        query_results[query] = unique_pmids
        all_pmids.update(unique_pmids)
    ordered_pmids = sorted(all_pmids, key=int)
    if not ordered_pmids:
        raise ValueError("PubMed discovery queries returned no PMIDs")
    return {
        "resolution_version": "1.0.0",
        "retmax_per_query": retmax_per_query,
        "query_results": query_results,
        "unique_pmid_count": len(ordered_pmids),
        "sources": [{"pmid": pmid} for pmid in ordered_pmids],
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _required_text(value: dict[str, object], key: str) -> str:
    result = _optional_text(value, key)
    if result is None:
        raise ValueError(f"{key} must be a non-empty string")
    return result


def _optional_text(value: dict[str, object], key: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str):
        raise ValueError(f"{key} must be a string or null")
    cleaned = _clean_text(item)
    return cleaned or None


def _optional_year(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or not 1800 <= value <= 2200:
        raise ValueError("year must be an integer between 1800 and 2200")
    return value
