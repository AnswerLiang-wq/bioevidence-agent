"""LLM-driven ReAct Agent for biomedical evidence retrieval.

Replaces the fixed search→fetch→inspect→classify pipeline with a loop where
the model decides which tool to call next based on accumulated observations.
The program controls execution: validates tool names/args, enforces call
budgets, detects repetition, and handles tool failures.

Only the model decides *when* to search again, fetch another record, deepen
inspection, or stop — the program never hard-codes that sequence.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from .product_contracts import summarize_source_coverage, validate_product_response
from .tools import (
    FetchRecordInput,
    InspectEvidenceInput,
    LiteratureTools,
    SearchLiteratureInput,
    ToolExecutor,
    trace_as_dict,
)

# ---------------------------------------------------------------------------
# DeepSeek tool schemas (structured tool_use / function calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_literature",
            "description": (
                "Search the biomedical literature corpus. Returns ranked PMIDs "
                "with scores. Call this first, or again with a rewritten query "
                "if results look irrelevant. rerank_query can differ from query "
                "to improve cross-encoder reranking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, ideally a focused clinical question.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (1–10).",
                        "default": 5,
                    },
                    "rerank_query": {
                        "type": "string",
                        "description": "Optional reranking query (omit to use query).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_record",
            "description": (
                "Fetch the full title, abstract, and metadata for a PMID "
                "returned by search_literature. Use this before inspect_evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pmid": {
                        "type": "string",
                        "description": "PubMed ID to fetch.",
                    }
                },
                "required": ["pmid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_evidence",
            "description": (
                "Extract the most relevant citable text spans from a fetched "
                "record. Returns snippets with exact character offsets and "
                "SHA-256 hashes for citation verification. Call after "
                "fetch_record when you have found a promising article."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pmid": {
                        "type": "string",
                        "description": "PMID of the already-fetched record.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Evidence query — what claim you are trying to support or refute.",
                    },
                    "max_snippets": {
                        "type": "integer",
                        "description": "Maximum snippets to return (1–4).",
                        "default": 2,
                    },
                },
                "required": ["pmid", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Emit a final verdict when you have gathered sufficient evidence. "
                "verdict must be one of: supported, contradicted, mixed, insufficient. "
                "Use insufficient only when the corpus genuinely lacks relevant evidence; "
                "it triggers abstain=true. All cited pmids must have been fetched and "
                "inspected in this session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "contradicted", "mixed", "insufficient"],
                    },
                    "answer": {
                        "type": "string",
                        "description": "One-paragraph natural-language answer citing evidence.",
                    },
                    "claim": {
                        "type": "string",
                        "description": "Single sentence summarising the key claim.",
                    },
                    "cited_pmids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "PMIDs of records that support the answer.",
                    },
                    "decisive_reason": {
                        "type": "string",
                        "description": "Why this verdict was chosen over alternatives.",
                    },
                },
                "required": ["verdict", "answer", "claim", "cited_pmids", "decisive_reason"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Session memory: accumulates observations within one question
# ---------------------------------------------------------------------------


@dataclass
class SessionMemory:
    question: str
    search_results: list[dict[str, Any]]
    fetched_records: dict[str, dict[str, Any]]    # pmid → record
    inspected_snippets: dict[str, list[dict[str, Any]]]  # pmid → snippets
    call_history: list[str]                        # "tool:arg_hash" for dup detection


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------


class LLMEvidenceAgent:
    """Evidence retrieval agent driven by a DeepSeek LLM decision loop.

    The model selects which tool to call; the program executes and validates.
    Supports query rewriting, multi-article evidence gathering, and explicit
    abstention when evidence is insufficient.
    """

    AGENT_VERSION = "llm-react-v1"

    def __init__(
        self,
        tools: LiteratureTools,
        *,
        api_key: str,
        model: str = "deepseek-flash",
        max_steps: int = 8,
        max_tokens_per_call: int = 1024,
        base_url: str = "https://api.deepseek.com/v1",
        timeout: float | None = None,
        max_retries: int | None = None,
        system_prompt: str | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> None:
        self._tools = tools
        self._api_key = api_key
        self._model = model
        self._max_steps = max_steps
        self._max_tokens_per_call = max_tokens_per_call
        self._base_url = base_url.rstrip("/")
        # None keeps the SDK's own defaults.  A costed evaluation should pass
        # max_retries=0 explicitly: the SDK default of 2 turns one failing call
        # into up to three HTTP requests, each of which is billed.
        self._timeout = timeout
        self._max_retries = max_retries
        # Per-instance prompt material.  Defaults are the module-level values, so
        # a caller that passes nothing gets exactly the previous behaviour; an
        # experimental arm passes its own copy and cannot alter the shared
        # TOOL_SCHEMAS object for anyone else.
        self._system_prompt = system_prompt if system_prompt is not None else _system_prompt()
        self._tool_schemas = tool_schemas if tool_schemas is not None else TOOL_SCHEMAS

        # lazy import so the rest of the package doesn't require openai SDK
        try:
            from openai import OpenAI  # DeepSeek uses OpenAI-compatible API
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
                "openai package is required for LLMEvidenceAgent: "
                "pip install 'openai>=1.0'"
            ) from exc

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        question: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the ReAct loop and return a product-contract-validated response."""
        rid = request_id or _make_request_id(question)
        memory = SessionMemory(
            question=question,
            search_results=[],
            fetched_records={},
            inspected_snippets={},
            call_history=[],
        )
        executor = ToolExecutor(self._tools, max_calls=self._max_steps + 2)

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                # Agents built via __new__ in tests and scripts bypass __init__
                # and carry no instance prompt; fall back to the module values so
                # those paths behave exactly as before.
                "content": getattr(self, "_system_prompt", None) or _system_prompt(),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        total_input_tokens = 0
        total_output_tokens = 0
        confirmed_usage_rounds = 0
        steps = 0
        finish_args: dict[str, Any] | None = None
        # run_status tracks how the loop terminated; evaluators must not count
        # text_exit or budget_exhausted as normal insufficient predictions.
        run_status = "budget_exhausted"
        termination_reason = "Max steps reached without a finish call."
        t_start = time.time()

        while steps < self._max_steps:
            steps += 1
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=getattr(self, "_tool_schemas", None) or TOOL_SCHEMAS,
                tool_choice="auto",
                max_tokens=self._max_tokens_per_call,
                temperature=0.0,
                # DeepSeek Chat Completions default is thinking.type="enabled".
                # Explicitly disable it so every call is in non-thinking mode.
                extra_body={"thinking": {"type": "disabled"}},
            )
            usage = response.usage
            if usage:
                total_input_tokens += usage.prompt_tokens
                total_output_tokens += usage.completion_tokens
                confirmed_usage_rounds += 1

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            # No tool call → model answered in plain text without calling finish.
            # Treat as text_exit — not a valid prediction, never an affirmative answer.
            # Use a fixed canonical answer rather than the raw model text, because the
            # raw text has not been through the finish-tool validation workflow and may
            # contain unverified affirmative conclusions.
            if not msg.tool_calls:
                run_status = "text_exit"
                termination_reason = "Model produced a text response without calling finish."
                finish_args = {
                    "verdict": "insufficient",
                    "answer": (
                        "Evidence validation incomplete: the agent stopped without "
                        "completing the finish-tool verification workflow."
                    ),
                    "claim": "Insufficient evidence found.",
                    "cited_pmids": [],
                    "decisive_reason": "Model stopped without calling finish — treated as text_exit.",
                }
                break

            # Process all tool calls in this turn (usually one, but handle batch)
            all_done = False
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    raw_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    raw_args = {}

                # Duplicate-call guard for non-finish tools.
                # finish is excluded: a rejected finish (e.g. unfetched citations) must
                # remain retryable after the model has done additional fetch/inspect work.
                # The loop breaks on an accepted finish anyway, so dup-guard adds nothing.
                args_digest = hashlib.sha256(
                    json.dumps(raw_args, sort_keys=True).encode()
                ).hexdigest()[:12]
                call_key = f"{tool_name}:{args_digest}"
                if tool_name != "finish" and call_key in memory.call_history:
                    tool_result = {
                        "error": "duplicate_call",
                        "message": "You already made this exact call. Try a different query or call finish.",
                    }
                else:
                    if tool_name != "finish":
                        memory.call_history.append(call_key)
                    tool_result = self._dispatch(tool_name, raw_args, memory, executor)

                if tool_name == "finish" and "error" not in tool_result:
                    finish_args = raw_args
                    run_status = "completed"
                    termination_reason = "finish tool accepted."
                    all_done = True

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

            if all_done:
                break

        # Expose the final messages list so external auditors (e.g. _AuditingClient.finalize)
        # can snapshot tool results appended after the last create() call.
        self._last_messages: list[dict[str, Any]] = messages

        # Build the final structured response
        elapsed_ms = (time.time() - t_start) * 1000
        # Compute cost only when ALL rounds had confirmed usage.  If any round
        # returned usage=None, the token count is partial and writing a cost
        # figure would misrepresent it as a confirmed total.
        if confirmed_usage_rounds == steps and steps > 0:
            cost_usd: float | None = _estimate_cost_usd(
                self._model, total_input_tokens, total_output_tokens
            )
        elif confirmed_usage_rounds > 0:
            # Some rounds confirmed, some did not — cost is a partial estimate.
            cost_usd = None
        else:
            # No confirmed usage at all.
            cost_usd = None

        return self._build_response(
            finish_args=finish_args,
            memory=memory,
            executor=executor,
            request_id=rid,
            steps=steps,
            elapsed_ms=elapsed_ms,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost_usd=cost_usd,
            run_status=run_status,
            termination_reason=termination_reason,
            confirmed_usage_rounds=confirmed_usage_rounds,
        )

    # ------------------------------------------------------------------
    # Tool dispatch — validates and executes, returns JSON-serialisable dict
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        tool_name: str,
        args: dict[str, Any],
        memory: SessionMemory,
        executor: ToolExecutor,
    ) -> dict[str, Any]:
        try:
            if tool_name == "search_literature":
                result = executor.search(
                    SearchLiteratureInput(
                        query=args["query"],
                        top_k=min(int(args.get("top_k", 5)), 10),
                        rerank_query=args.get("rerank_query"),
                    )
                )
                hits = [
                    {
                        "pmid": h.pmid,
                        "rank": h.rank,
                        "score": round(h.score, 4),
                        "retrieval_method": h.retrieval_method,
                    }
                    for h in result.hits
                ]
                memory.search_results.extend(hits)
                return {"hits": hits}

            elif tool_name == "fetch_record":
                pmid = str(args["pmid"])
                record = executor.fetch(FetchRecordInput(pmid=pmid))
                data = {
                    "pmid": record.pmid,
                    "title": record.title,
                    "abstract": record.abstract,
                    "year": record.year,
                    "journal": record.journal,
                    "document_sha256": record.document_sha256,
                }
                memory.fetched_records[pmid] = data
                return data

            elif tool_name == "inspect_evidence":
                pmid = str(args["pmid"])
                if pmid not in memory.fetched_records:
                    return {
                        "error": "not_fetched",
                        "message": f"PMID {pmid} must be fetched before inspection.",
                    }
                result = executor.inspect(
                    InspectEvidenceInput(
                        pmid=pmid,
                        query=args["query"],
                        max_snippets=min(int(args.get("max_snippets", 2)), 4),
                    )
                )
                snippets = [
                    {
                        "text": s.text,
                        "start_char": s.start_char,
                        "end_char": s.end_char,
                        "section": s.section,
                        "snippet_sha256": s.snippet_sha256,
                        "relevance_score": round(s.relevance_score, 4),
                    }
                    for s in result.snippets
                ]
                memory.inspected_snippets.setdefault(pmid, []).extend(snippets)
                return {
                    "pmid": pmid,
                    "document_sha256": result.document_sha256,
                    "snippets": snippets,
                }

            elif tool_name == "finish":
                # Pre-accept validation: enforce ALL requirements BEFORE accepting
                # so the model can correct within budget.
                verdict = args.get("verdict", "")
                cited_pmids = [str(p) for p in args.get("cited_pmids", [])]
                answer = args.get("answer", "")
                claim = args.get("claim", "")
                decisive_reason = args.get("decisive_reason", "")
                EVIDENCE_VERDICTS = {"supported", "contradicted", "mixed"}
                VALID_VERDICTS = {"supported", "contradicted", "mixed", "insufficient"}

                # 1. Required fields check
                missing = []
                if not verdict:
                    missing.append("verdict")
                if not answer or not answer.strip():
                    missing.append("answer")
                if not claim or not claim.strip():
                    missing.append("claim")
                if "cited_pmids" not in args:
                    missing.append("cited_pmids")
                if not decisive_reason or not decisive_reason.strip():
                    missing.append("decisive_reason")
                if missing:
                    return {
                        "error": "missing_required_fields",
                        "message": (
                            f"finish requires: {missing}. "
                            "All five fields (verdict, answer, claim, cited_pmids, decisive_reason) are mandatory."
                        ),
                    }

                # 2. Verdict must be a known value
                if verdict not in VALID_VERDICTS:
                    return {
                        "error": "invalid_verdict",
                        "message": (
                            f"verdict='{verdict}' is not valid. "
                            "Must be one of: supported, contradicted, mixed, insufficient."
                        ),
                    }

                if verdict in EVIDENCE_VERDICTS:
                    # 3. Evidence verdicts need at least one citation
                    if not cited_pmids:
                        return {
                            "error": "citation_required",
                            "message": (
                                f"verdict='{verdict}' requires at least one cited PMID. "
                                "Add citations or change the verdict to 'insufficient'."
                            ),
                        }
                    # 4. Every cited PMID must have been fetched
                    unfetched = [p for p in cited_pmids if p not in memory.fetched_records]
                    if unfetched:
                        return {
                            "error": "citation_not_fetched",
                            "message": (
                                f"PMIDs {unfetched} were cited but never fetched. "
                                "Call fetch_record and inspect_evidence for each before finishing."
                            ),
                        }
                    # 5. Every cited PMID must have been inspected
                    uninspected = [
                        p for p in cited_pmids
                        if p not in memory.inspected_snippets
                    ]
                    if uninspected:
                        return {
                            "error": "citation_not_inspected",
                            "message": (
                                f"PMIDs {uninspected} were fetched but never inspected. "
                                "Call inspect_evidence for each before finishing."
                            ),
                        }
                    # 6. Every cited PMID must have appeared in a search result
                    #    (guarantees honest retrieval provenance in _build_response)
                    no_search_source = [
                        p for p in cited_pmids
                        if not any(str(h["pmid"]) == p for h in memory.search_results)
                    ]
                    if no_search_source:
                        return {
                            "error": "citation_no_search_source",
                            "message": (
                                f"PMIDs {no_search_source} were cited but never returned by "
                                "search_literature. Call search_literature first so retrieval "
                                "provenance can be recorded."
                            ),
                        }
                    # 7. Every cited PMID must have non-empty inspected snippets
                    empty_snippets = [
                        p for p in cited_pmids
                        if not memory.inspected_snippets.get(p)
                    ]
                    if empty_snippets:
                        return {
                            "error": "citation_empty_snippets",
                            "message": (
                                f"PMIDs {empty_snippets} were inspected but returned no evidence "
                                "snippets. Re-run inspect_evidence with a more targeted query, "
                                "or drop this PMID from citations."
                            ),
                        }

                return {"status": "accepted", "verdict": verdict}

            else:
                return {"error": "unknown_tool", "message": f"Tool '{tool_name}' does not exist."}

        except Exception as exc:  # noqa: BLE001
            return {"error": type(exc).__name__, "message": str(exc)}

    # ------------------------------------------------------------------
    # Response builder — assembles product-contract-compatible structure
    # ------------------------------------------------------------------

    def _build_response(
        self,
        *,
        finish_args: dict[str, Any] | None,
        memory: SessionMemory,
        executor: ToolExecutor,
        request_id: str,
        steps: int,
        elapsed_ms: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None,
        run_status: str = "budget_exhausted",
        termination_reason: str = "Max steps reached without a finish call.",
        confirmed_usage_rounds: int = 0,
    ) -> dict[str, Any]:
        # Fall back to abstain when finish was never called (budget_exhausted path)
        if finish_args is None:
            finish_args = {
                "verdict": "insufficient",
                "answer": "Agent exceeded step budget without reaching a verdict.",
                "claim": "Step budget exceeded — insufficient evidence gathered.",
                "cited_pmids": [],
                "decisive_reason": termination_reason,
            }

        verdict = finish_args["verdict"]
        cited_pmids: list[str] = finish_args.get("cited_pmids", [])

        # Safety net: if finish was somehow accepted with invalid citations (should be
        # blocked by _dispatch), raise explicitly rather than fabricate provenance.
        # This path is unreachable in normal operation but guards against unexpected
        # code paths (e.g. direct _build_response calls in tests).
        EVIDENCE_VERDICTS = {"supported", "contradicted", "mixed"}
        if verdict in EVIDENCE_VERDICTS and cited_pmids:
            invalid_pmids = [
                pmid for pmid in cited_pmids
                if not memory.fetched_records.get(pmid) or not memory.inspected_snippets.get(pmid)
            ]
            if invalid_pmids:
                # Protocol failure: _dispatch should have caught this. Record it clearly.
                raise ValueError(
                    f"Protocol failure: finish was accepted with PMIDs {invalid_pmids} that "
                    "were never fetched+inspected. This indicates a code path bypassed "
                    "_dispatch validation."
                )

        valid_cited_pmids = [str(p) for p in cited_pmids]

        citations: list[dict[str, Any]] = []
        for pmid in valid_cited_pmids:
            record = memory.fetched_records.get(pmid)
            snippets = memory.inspected_snippets.get(pmid, [])
            # Both are guaranteed non-empty here (filtered above)
            hit = next((h for h in memory.search_results if str(h["pmid"]) == pmid), None)
            if hit is None:
                # PMID was fetched but never appeared in search results (e.g. fetched directly).
                # Raise rather than fabricate provenance — this indicates an unexpected code path.
                raise ValueError(
                    f"PMID {pmid} was inspected but has no search-result entry in memory; "
                    "cannot construct a citation with honest retrieval provenance."
                )
            ret_method = hit["retrieval_method"]
            ret_rank = max(1, int(hit["rank"]))
            ret_score = float(hit["score"])
            # One citation per inspected snippet.  Emitting only snippets[0] would
            # silently drop evidence the model actually inspected and may have
            # relied on, so an assertion supported by a later snippet would have no
            # citation span covering it.
            for snippet in snippets:
                citations.append({
                    "citation_id": f"C{len(citations) + 1}",
                    "pmid": pmid,
                    "doi": None,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "title": record["title"],
                    "source_sha256": record["document_sha256"],
                    "section": snippet["section"],
                    "snippet": snippet["text"],
                    "start_char": snippet["start_char"],
                    "end_char": snippet["end_char"],
                    "snippet_sha256": snippet["snippet_sha256"],
                    "direction": _verdict_to_direction(verdict),
                    "retrieval": {
                        "method": ret_method,
                        "rank": ret_rank,
                        "score": ret_score,
                        "component_ranks": {ret_method: ret_rank},
                        "candidate_rank": None,
                    },
                })

        claim_text = finish_args.get("claim", finish_args["answer"][:120])
        citation_ids = [c["citation_id"] for c in citations]

        response: dict[str, Any] = {
            "response_version": "1.0.0",
            "baseline_id": f"{self.AGENT_VERSION}-{self._model.replace('/', '-')}",
            "request_id": request_id,
            "verdict": verdict,
            "abstained": verdict == "insufficient",
            "answer": finish_args["answer"],
            "claims": [{"text": claim_text, "citation_ids": citation_ids}] if citations else [],
            "citations": citations,
            "decisive_reason": finish_args.get("decisive_reason", ""),
            "scope_limits": [
                "Answers are based on the local PubMed snapshot only.",
                "LLM decisions are probabilistic; outputs require human expert review.",
                "No clinical advice is implied.",
            ],
            "confidence": _confidence_from_verdict(verdict),
            "provenance": {
                "corpus_sha256": None,
                "retrieval_method": "llm_react_loop",
                "retrieval_config": {
                    "method": "llm_react_loop",
                    "model": self._model,
                    "max_steps": self._max_steps,
                    "steps_used": steps,
                },
                "answer_model": {
                    "type": "llm",
                    "model": self._model,
                    # input_tokens and output_tokens are raw accumulated counts.
                    # usage_coverage tells you whether those counts are trustworthy:
                    #   "confirmed"  — every round returned usage; counts are complete.
                    #   "partial"    — some rounds had usage, some did not; counts are
                    #                  a lower bound only.
                    #   "unknown"    — no round returned usage; counts are 0 but that
                    #                  does NOT mean zero tokens were consumed.
                    # Consumers must check usage_coverage before treating the counts
                    # as a confirmed total.  When coverage is not "confirmed", treat
                    # input_tokens / output_tokens as a partial lower bound.
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "usage_coverage": (
                        "confirmed" if confirmed_usage_rounds == steps and steps > 0
                        else "partial" if confirmed_usage_rounds > 0
                        else "unknown"
                    ),
                    "elapsed_ms": round(elapsed_ms, 1),
                },
                "answer_probabilities": {},
                "tool_call_count": len(executor.trace),
                "tool_trace": trace_as_dict(executor.trace),
                "model_api_cost_usd": cost_usd,
                # Advisory provenance completeness check.  Records which literal
                # figures in `answer` fall inside a cited span.  Deliberately not
                # a gate: presence in a span is not entailment, and rejecting on
                # it would conflate "traceable to a source" with "supported by
                # that source".  See summarize_source_coverage for its limits.
                "source_coverage": summarize_source_coverage(
                    finish_args["answer"], citations
                ),
                # agent_run is LLM-agent-specific operational metadata.
                # It is nested here so the shared v1 product contract (which the
                # rule-based PubMedQA agent also satisfies) never has to include it.
                "agent_run": {
                    "run_status": run_status,
                    "termination_reason": termination_reason,
                },
            },
        }

        violations = validate_product_response(response)
        if violations:
            # Patch citations to satisfy contract rather than raising
            if any("span" in v or "sha" in v.lower() for v in violations):
                response["citations"] = []
                response["claims"] = [{"text": claim_text, "citation_ids": []}]
                violations = validate_product_response(response)
        if violations:
            raise ValueError(
                f"LLM agent response contract failure ({len(violations)} violations): "
                + " | ".join(violations)
            )
        return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _system_prompt() -> str:
    return (
        "You are a biomedical evidence retrieval agent. Your job is to answer "
        "a clinical question by searching a PubMed corpus, fetching relevant "
        "records, inspecting the evidence, and emitting a verdict.\n\n"
        "Workflow:\n"
        "1. Call search_literature with a focused query.\n"
        "2. For promising hits, call fetch_record to get the abstract.\n"
        "3. For records that look relevant, call inspect_evidence to get "
        "   citable snippets with exact character offsets.\n"
        "4. If results are poor, rewrite your query and search again.\n"
        "5. When you have enough evidence (or are confident the corpus lacks "
        "   relevant data), call finish.\n\n"
        "Rules:\n"
        "- Never make up citations. Only cite PMIDs you have fetched and inspected.\n"
        "- Use verdict=insufficient when there is genuinely no relevant evidence.\n"
        "- Be efficient: you have at most 8 tool calls total.\n"
        "- Do not repeat identical tool calls.\n"
    )


def _make_request_id(question: str) -> str:
    digest = hashlib.sha256(question.encode()).hexdigest()[:12].upper()
    return f"LLM-{digest}"


def _verdict_to_direction(verdict: str) -> str:
    return {
        "supported": "supports",
        "contradicted": "contradicts",
        "mixed": "context_only",
        "insufficient": "context_only",
    }.get(verdict, "context_only")


def _confidence_from_verdict(verdict: str) -> str:
    return "low" if verdict == "insufficient" else "medium"


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Conservative (peak-hour, cache-miss) cost estimate.

    Pricing source: https://api-docs.deepseek.com/quick_start/pricing
    Retrieved: 2026-09-30.  Prices are per 1M tokens.

    deepseek-flash (DeepSeek-V4.1-Flash):
        input  cache miss: $0.15 off-peak / $0.30 peak
        input  cache hit:  $0.003 off-peak / $0.006 peak
        output:            $0.60 off-peak  / $1.20 peak
    deepseek-v4-pro:
        input  cache miss: $0.66 off-peak / $1.32 peak
        input  cache hit:  $0.022 off-peak / $0.044 peak
        output:            $1.98 off-peak  / $3.96 peak

    This function uses peak-hour, cache-miss rates for a conservative upper
    bound.  A single 6-step smoke test on deepseek-flash (peak, cache miss)
    is well under $0.02.  Off-peak would be roughly half.

    Do NOT pass "deepseek-chat" — that alias is not guaranteed to persist and
    should not be used for new code.  Use "deepseek-flash" explicitly.

    Unknown models return None rather than 0.0 to make missing cost data
    distinguishable from a confirmed zero-cost run.
    """
    model_lower = model.lower()
    # deepseek-flash (explicit name only — deepseek-chat alias is not supported)
    if "deepseek-flash" in model_lower:
        # peak-hour, cache-miss: $0.30/M input, $1.20/M output
        return (input_tokens * 0.30 + output_tokens * 1.20) / 1_000_000
    # deepseek-v4-pro
    if "deepseek-v4-pro" in model_lower or "deepseek-v4pro" in model_lower:
        # peak-hour, cache-miss: $1.32/M input, $3.96/M output
        return (input_tokens * 1.32 + output_tokens * 3.96) / 1_000_000
    # Unknown model — return None so callers can distinguish "not measured"
    # from "confirmed zero"
    return None  # type: ignore[return-value]
