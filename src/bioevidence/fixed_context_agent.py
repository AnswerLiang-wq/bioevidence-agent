"""Arm B: the fixed-context LLM baseline.

Same model and same retrieval index as the agent arm, but the comparison is
strategy-level, not loop-level: one BM25 search on the *original* question, the
top-k abstracts pasted into a single prompt as a static context block, and
exactly one model request.  There is no tool loop and no second chance — a
failure is recorded, never repaired with another request.

Because the two arms receive different evidence, an A-over-B result supports
"the agent strategy, including its evidence selection, outperforms direct
reading".  It does **not** isolate the loop's contribution.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .llm_agent import _estimate_cost_usd
from .tools import (
    FetchRecordInput,
    LiteratureTools,
    SearchLiteratureInput,
)

#: Median number of distinct records the agent fetched per completed v1 case.
#: Matching k to that keeps the two arms comparable in how much evidence they
#: see, without giving B a different retrieval budget than A typically used.
DEFAULT_TOP_K = 4

VERDICTS = ("supported", "contradicted", "mixed", "insufficient")

#: Verdicts that assert something about the evidence, and so must name at least
#: one record from the provided set.  Same set the agent arm's finish tool uses.
EVIDENCE_VERDICTS = ("supported", "contradicted", "mixed")

#: Fields the model must supply, and the type each must have.  Arm A's finish tool
#: rejects a call missing any of these; Arm B must apply the same bar, or its
#: "completed" cases would not be comparable to the agent's.
REQUIRED_FIELDS = {
    "verdict": str,
    "answer": str,
    "claim": str,
    "cited_pmids": list,
    "decisive_reason": str,
}

#: The task and the verdict meanings are worded to match the agent arm.  What is
#: removed is everything that only makes sense with tools: the search/fetch/
#: inspect workflow, the tool-call budget and the duplicate-call rule.
FIXED_CONTEXT_SYSTEM_PROMPT = (
    "You are a biomedical evidence agent. Your job is to answer a clinical "
    "question from a fixed set of PubMed abstracts that have already been "
    "retrieved for you.\n\n"
    "You have no search tools and must not ask for more evidence. The abstracts "
    "in the user message are the complete evidence available for this question.\n\n"
    "Rules:\n"
    "- Never make up citations. Only cite PMIDs that appear in the provided set.\n"
    "- verdict=supported, contradicted or mixed must cite at least one of those "
    "PMIDs; a verdict that names no evidence is not an answer.\n"
    "- Use verdict=insufficient when there is genuinely no relevant evidence; it "
    "triggers abstain=true and may be given with an empty cited_pmids.\n"
    "- Every field is required and must be filled in.\n\n"
    "Reply with exactly one JSON object and nothing else:\n"
    "{\n"
    '  "verdict": one of ["supported", "contradicted", "mixed", "insufficient"],\n'
    '  "answer": "one-paragraph natural-language answer citing evidence",\n'
    '  "claim": "single sentence summarising the key claim",\n'
    '  "cited_pmids": ["PMIDs of records that support the answer"],\n'
    '  "decisive_reason": "why this verdict was chosen over alternatives"\n'
    "}"
)

USER_TEMPLATE_HEADER = "Question: {question}\n\n"
USER_TEMPLATE_CONTEXT = "Retrieved abstracts ({n} records, BM25 on the original question):\n\n"
USER_TEMPLATE_FOOTER = "\nAnswer the question using only these abstracts."


class FixedContextResponseError(RuntimeError):
    """The single request produced nothing usable.  Never retried."""


def validate_payload(payload: Any, provided_pmids: list[str]) -> dict[str, Any]:
    """Check the single response against the same bar the agent's finish tool sets.

    A bare ``{"verdict": "supported"}`` is not a usable answer: it names no
    evidence and states nothing.  Arm A's ``finish`` rejects that, so accepting
    it here would let B report ``completed`` on responses the agent arm would
    have refused — and B's accuracy would then be measured against a weaker
    standard than A's.

    Raises ``FixedContextResponseError``; the caller records the case as failed
    and never re-asks.
    """
    if not isinstance(payload, dict):
        raise FixedContextResponseError("single response was not a JSON object")

    for field, expected in REQUIRED_FIELDS.items():
        if field not in payload:
            raise FixedContextResponseError(f"response is missing required field {field!r}")
        value = payload[field]
        if not isinstance(value, expected):
            raise FixedContextResponseError(
                f"field {field!r} must be {expected.__name__}, got "
                f"{type(value).__name__}"
            )
        if expected is str and not value.strip():
            raise FixedContextResponseError(f"field {field!r} must not be empty")

    verdict = payload["verdict"]
    if verdict not in VERDICTS:
        raise FixedContextResponseError(
            f"verdict {verdict!r} is not one of {sorted(VERDICTS)}"
        )

    cited = payload["cited_pmids"]
    if any(not isinstance(pmid, str) for pmid in cited):
        raise FixedContextResponseError("cited_pmids must contain only strings")

    unknown_citations = [pmid for pmid in cited if pmid not in provided_pmids]
    if unknown_citations:
        # The prompt restricts citations to the provided set; citing anything
        # else is a model error.  There is no retry budget to correct it, and
        # repairing it silently would misreport what the model did.
        raise FixedContextResponseError(
            f"cited PMIDs outside the provided set: {unknown_citations}"
        )

    if verdict in EVIDENCE_VERDICTS and not cited:
        raise FixedContextResponseError(
            f"verdict {verdict!r} needs at least one cited PMID from the provided "
            "set; with none it is not an evidence verdict"
        )
    return payload


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def build_user_message(question: str, records: list[dict[str, Any]]) -> str:
    """The single user turn: the original question plus the static context block."""
    blocks = []
    for record in records:
        blocks.append(
            f"[PMID {record['pmid']}] {record['title']}\n{record['abstract']}"
        )
    return (
        USER_TEMPLATE_HEADER.format(question=question)
        + USER_TEMPLATE_CONTEXT.format(n=len(records))
        + "\n\n".join(blocks)
        + USER_TEMPLATE_FOOTER
    )


class FixedContextLLMAgent:
    """One retrieval, one request, one verdict.  No loop, no retries."""

    AGENT_VERSION = "fixed-context-v1"
    RETRIEVAL_METHOD = "fixed_context_llm"

    def __init__(
        self,
        tools: LiteratureTools,
        *,
        api_key: str,
        model: str = "deepseek-flash",
        top_k: int = DEFAULT_TOP_K,
        max_tokens_per_call: int = 1024,
        base_url: str = "https://api.deepseek.com/v1",
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._tools = tools
        self._api_key = api_key
        self._model = model
        self._top_k = top_k
        self._max_tokens_per_call = max_tokens_per_call
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries

        try:
            from openai import OpenAI
            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "base_url": self._base_url,
            }
            if timeout is not None:
                client_kwargs["timeout"] = timeout
            if max_retries is not None:
                client_kwargs["max_retries"] = max_retries
            self._client = OpenAI(**client_kwargs)
        except ImportError as exc:
            raise ImportError(
                "openai package is required for FixedContextLLMAgent"
            ) from exc

    # ------------------------------------------------------------------

    def prompt_template(self) -> dict[str, Any]:
        """The static prompt parts, so a run can record what was actually sent."""
        return {
            "system": FIXED_CONTEXT_SYSTEM_PROMPT,
            "user_header": USER_TEMPLATE_HEADER,
            "user_context_header": USER_TEMPLATE_CONTEXT,
            "user_footer": USER_TEMPLATE_FOOTER,
            "top_k": self._top_k,
            "response_format": {"type": "json_object"},
        }

    def _retrieve(self, question: str) -> list[dict[str, Any]]:
        """One BM25 search on the original question; no query rewriting."""
        hits = self._tools.search_literature(
            SearchLiteratureInput(query=question, top_k=self._top_k)
        ).hits
        records = []
        for hit in hits:
            fetched = self._tools.fetch_record(FetchRecordInput(pmid=hit.pmid))
            records.append({
                "pmid": str(fetched.pmid),
                "title": fetched.title,
                "abstract": fetched.abstract,
            })
        return records

    # ------------------------------------------------------------------

    def run(self, *, question: str, request_id: str | None = None) -> dict[str, Any]:
        """Retrieve once, ask once, and return a response the runner can score.

        Raises on any failure — an unusable answer is a recorded failure, not
        something to be re-asked.  The caller's per-case handler turns that into
        an ``errored`` case.
        """
        rid = request_id or f"FIXED-{abs(hash(question)) % (10**8):08d}"
        records = self._retrieve(question)
        provided_pmids = [record["pmid"] for record in records]
        messages = [
            {"role": "system", "content": FIXED_CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, records)},
        ]

        started = time.time()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=self._max_tokens_per_call,
            temperature=0.0,
            extra_body={"thinking": {"type": "disabled"}},
        )
        elapsed_ms = (time.time() - started) * 1000

        usage = getattr(response, "usage", None)
        if usage is not None:
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            usage_coverage = "confirmed"
            cost = _estimate_cost_usd(self._model, input_tokens, output_tokens)
        else:
            input_tokens = output_tokens = 0
            usage_coverage = "unknown"
            cost = None

        raw = response.choices[0].message.content or ""
        try:
            decoded = json.loads(_strip_code_fence(raw))
        except (TypeError, ValueError) as exc:
            raise FixedContextResponseError(
                f"single response was not valid JSON: {exc}"
            ) from exc
        payload = validate_payload(decoded, provided_pmids)
        verdict = str(payload["verdict"])
        cited = [str(pmid) for pmid in payload["cited_pmids"]]

        by_pmid = {record["pmid"]: record for record in records}
        return {
            "response_version": "1.0.0",
            "baseline_id": f"{self.AGENT_VERSION}-{self._model.replace('/', '-')}",
            "request_id": rid,
            "verdict": verdict,
            "abstained": verdict == "insufficient",
            "answer": str(payload.get("answer") or ""),
            "claims": [{
                "text": str(payload.get("claim") or ""),
                "citation_ids": [f"C{i + 1}" for i in range(len(cited))],
            }] if cited else [],
            "citations": [
                {
                    "citation_id": f"C{i + 1}",
                    "pmid": pmid,
                    "title": by_pmid[pmid]["title"],
                }
                for i, pmid in enumerate(cited)
            ],
            "decisive_reason": str(payload.get("decisive_reason") or ""),
            "provenance": {
                "corpus_sha256": None,
                "retrieval_method": self.RETRIEVAL_METHOD,
                "retrieval_config": {
                    "method": self.RETRIEVAL_METHOD,
                    "model": self._model,
                    "top_k": self._top_k,
                    "steps_used": 1,
                    "retrieved_pmids": provided_pmids,
                },
                "answer_model": {
                    "type": "llm",
                    "model": self._model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "usage_coverage": usage_coverage,
                    "elapsed_ms": round(elapsed_ms, 1),
                },
                "answer_probabilities": {},
                "tool_call_count": 0,
                "tool_trace": [],
                "model_api_cost_usd": cost,
                "agent_run": {
                    "run_status": "completed",
                    "termination_reason": "single request completed.",
                    "requests_made": 1,
                },
            },
        }
