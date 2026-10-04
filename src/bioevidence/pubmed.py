"""A polite, zero-dependency PubMed E-utilities client."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT = "BioEvidenceAgent/0.5.1 (research; PubMed metadata lookup)"


class PubMedError(RuntimeError):
    """Raised when PubMed cannot return a parseable E-utilities response."""

    def __init__(self, message: str, *, status: str = "retrieval_failed") -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class PubMedArticle:
    """A selected subset of PubMed metadata plus in-memory abstract text."""

    pmid: str
    title: str
    doi: str | None
    publication_types: tuple[str, ...]
    journal: str
    year: int | None
    first_author: str | None
    abstract: str

    @property
    def source_url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    @property
    def study_type(self) -> str:
        return "; ".join(self.publication_types) or "not reported in PubMed metadata"


class PubMedClient:
    """Small NCBI client with a conservative no-key request rate."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        min_interval_seconds: float | None = None,
        email: str | None = None,
        api_key: str | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.email = email or os.environ.get("BIOEVIDENCE_CONTACT_EMAIL") or None
        self._api_key = api_key or os.environ.get("NCBI_API_KEY") or None
        self.min_interval_seconds = (
            min_interval_seconds
            if min_interval_seconds is not None
            else 0.12 if self._api_key else 0.40
        )
        self.max_attempts = max_attempts
        self._last_request_at = 0.0
        self._article_cache: dict[str, PubMedArticle] = {}

    @property
    def api_key_present(self) -> bool:
        """Expose only key presence for a run manifest; never expose its value."""

        return bool(self._api_key)

    def search(self, query: str, *, retmax: int = 5) -> list[str]:
        payload = self._request(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmax": str(retmax),
                "retmode": "json",
                "sort": "relevance",
                "tool": "bioevidence_agent",
            },
        )
        try:
            decoded = json.loads(payload.decode("utf-8"))
            identifiers = decoded["esearchresult"]["idlist"]
        except (UnicodeDecodeError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise PubMedError("PubMed ESearch returned an unexpected payload") from exc
        semantic_error = decoded.get("esearchresult", {}).get("ERROR")
        if semantic_error:
            raise PubMedError(f"PubMed ESearch semantic error: {semantic_error}")
        if not isinstance(identifiers, list):
            raise PubMedError("PubMed ESearch idlist was not a list")
        return [identifier for identifier in identifiers if isinstance(identifier, str)]

    def fetch(self, pmids: Iterable[str]) -> list[PubMedArticle]:
        requested = [pmid for pmid in pmids if pmid not in self._article_cache]
        if requested:
            for start in range(0, len(requested), 20):
                batch = requested[start : start + 20]
                payload = self._request(
                    "efetch.fcgi",
                    {
                        "db": "pubmed",
                        "id": ",".join(batch),
                        "retmode": "xml",
                        "tool": "bioevidence_agent",
                    },
                )
                parsed = self._parse_articles(payload)
                for article in parsed:
                    self._article_cache[article.pmid] = article

        return [
            self._article_cache[pmid]
            for pmid in pmids
            if pmid in self._article_cache
        ]

    def _request(self, endpoint: str, params: dict[str, str]) -> bytes:
        request_params = dict(params)
        if self.email:
            request_params["email"] = self.email
        if self._api_key:
            request_params["api_key"] = self._api_key
        url = f"{EUTILS_BASE}/{endpoint}?{urllib.parse.urlencode(request_params)}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        )
        last_error: BaseException | None = None
        for attempt in range(self.max_attempts):
            elapsed = time.monotonic() - self._last_request_at
            wait_seconds = self.min_interval_seconds - elapsed
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            retryable = False
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                last_error = exc
                retryable = exc.code == 429 or 500 <= exc.code <= 599
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                retryable = True
            finally:
                self._last_request_at = time.monotonic()

            if not retryable or attempt + 1 >= self.max_attempts:
                break
            time.sleep(0.5 * (2**attempt))
        raise PubMedError(
            f"PubMed request failed for {endpoint}: {last_error}",
            status="retrieval_failed",
        )

    @staticmethod
    def _element_text(element: ET.Element | None) -> str:
        if element is None:
            return ""
        return " ".join(part.strip() for part in element.itertext() if part.strip())

    def _parse_articles(self, payload: bytes) -> list[PubMedArticle]:
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise PubMedError("PubMed EFetch returned malformed XML") from exc
        semantic_errors = [
            self._element_text(node) for node in root.findall(".//ERROR")
        ]
        if semantic_errors:
            raise PubMedError(
                "PubMed EFetch semantic error: " + " | ".join(semantic_errors)
            )

        articles: list[PubMedArticle] = []
        for node in root.findall(".//PubmedArticle"):
            citation = node.find("./MedlineCitation")
            article_node = node.find("./MedlineCitation/Article")
            if citation is None or article_node is None:
                continue

            pmid = self._element_text(citation.find("./PMID"))
            title = self._element_text(article_node.find("./ArticleTitle"))
            if not pmid or not title:
                continue

            publication_types = tuple(
                self._element_text(item)
                for item in article_node.findall("./PublicationTypeList/PublicationType")
                if self._element_text(item)
            )
            abstract_parts: list[str] = []
            for section in article_node.findall("./Abstract/AbstractText"):
                label = section.attrib.get("Label", "").strip()
                text = self._element_text(section)
                if text:
                    abstract_parts.append(f"{label}: {text}" if label else text)
            abstract = "\n".join(abstract_parts)

            doi = None
            for identifier in node.findall("./PubmedData/ArticleIdList/ArticleId"):
                if identifier.attrib.get("IdType") == "doi":
                    candidate = self._element_text(identifier)
                    if candidate:
                        doi = candidate
                        break
            if doi is None:
                for identifier in article_node.findall("./ELocationID"):
                    if identifier.attrib.get("EIdType") == "doi":
                        candidate = self._element_text(identifier)
                        if candidate:
                            doi = candidate
                            break

            journal = self._element_text(article_node.find("./Journal/Title"))
            year = self._parse_year(article_node, citation)
            first_author = self._parse_first_author(article_node)
            articles.append(
                PubMedArticle(
                    pmid=pmid,
                    title=title,
                    doi=doi,
                    publication_types=publication_types,
                    journal=journal,
                    year=year,
                    first_author=first_author,
                    abstract=abstract,
                )
            )
        return articles

    def _parse_year(
        self,
        article_node: ET.Element,
        citation: ET.Element,
    ) -> int | None:
        candidates = (
            article_node.find("./Journal/JournalIssue/PubDate/Year"),
            article_node.find("./ArticleDate/Year"),
            citation.find("./DateCompleted/Year"),
        )
        for candidate in candidates:
            text = self._element_text(candidate)
            if text.isdigit() and len(text) == 4:
                return int(text)
        return None

    def _parse_first_author(self, article_node: ET.Element) -> str | None:
        author = article_node.find("./AuthorList/Author")
        if author is None:
            return None
        collective = self._element_text(author.find("./CollectiveName"))
        if collective:
            return collective
        last_name = self._element_text(author.find("./LastName"))
        initials = self._element_text(author.find("./Initials"))
        if last_name and initials:
            return f"{last_name} {initials}"
        return last_name or None
