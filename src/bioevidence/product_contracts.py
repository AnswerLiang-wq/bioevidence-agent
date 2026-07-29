"""Runtime checks for the product response contract."""

from __future__ import annotations

import hashlib
import re


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VERDICTS = {"supported", "contradicted", "mixed", "insufficient"}


def validate_product_response(response: object) -> list[str]:
    if not isinstance(response, dict):
        return ["response must be an object"]
    required = {
        "response_version",
        "baseline_id",
        "request_id",
        "verdict",
        "abstained",
        "answer",
        "claims",
        "citations",
        "decisive_reason",
        "scope_limits",
        "confidence",
        "provenance",
    }
    errors: list[str] = []
    if set(response) != required:
        errors.append("response top-level fields differ from product contract")
    if response.get("verdict") not in VERDICTS:
        errors.append("invalid verdict")
    if not isinstance(response.get("abstained"), bool):
        errors.append("abstained must be boolean")
    if response.get("verdict") == "insufficient" and response.get("abstained") is not True:
        errors.append("insufficient verdict must abstain")
    citations = response.get("citations")
    claims = response.get("claims")
    if not isinstance(citations, list) or not isinstance(claims, list):
        errors.append("claims and citations must be lists")
        return errors
    citation_ids: set[str] = set()
    for citation in citations:
        if not isinstance(citation, dict):
            errors.append("citation must be an object")
            continue
        citation_id = citation.get("citation_id")
        if not isinstance(citation_id, str) or citation_id in citation_ids:
            errors.append("citation IDs must be unique strings")
        else:
            citation_ids.add(citation_id)
        for key in ("source_sha256", "snippet_sha256"):
            value = citation.get(key)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                errors.append(f"citation {key} must be SHA-256")
        start = citation.get("start_char")
        end = citation.get("end_char")
        snippet = citation.get("snippet")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(snippet, str)
            or start < 0
            or end <= start
        ):
            errors.append("citation span is invalid")
        elif end - start != len(snippet):
            errors.append("citation span length does not match snippet")
        elif hashlib.sha256(snippet.encode("utf-8")).hexdigest() != citation.get(
            "snippet_sha256"
        ):
            errors.append("citation snippet hash does not match snippet")
        retrieval = citation.get("retrieval")
        if not isinstance(retrieval, dict) or set(retrieval) != {
            "method",
            "rank",
            "score",
            "component_ranks",
            "candidate_rank",
        }:
            errors.append("citation retrieval lineage is invalid")
        else:
            component_ranks = retrieval.get("component_ranks")
            if (
                not isinstance(retrieval.get("method"), str)
                or not isinstance(retrieval.get("rank"), int)
                or retrieval["rank"] < 1
                or not isinstance(retrieval.get("score"), (int, float))
                or not isinstance(component_ranks, dict)
                or not component_ranks
                or any(
                    not isinstance(rank, int) or rank < 1
                    for rank in component_ranks.values()
                )
                or (
                    retrieval.get("candidate_rank") is not None
                    and (
                        not isinstance(retrieval["candidate_rank"], int)
                        or retrieval["candidate_rank"] < 1
                    )
                )
            ):
                errors.append("citation retrieval lineage values are invalid")
    for claim in claims:
        if not isinstance(claim, dict):
            errors.append("claim must be an object")
            continue
        references = claim.get("citation_ids")
        if not isinstance(references, list) or not references:
            errors.append("every answer claim must have at least one citation")
        elif any(reference not in citation_ids for reference in references):
            errors.append("answer claim references unknown citation")
    provenance = response.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
    elif not isinstance(provenance.get("tool_trace"), list):
        errors.append("provenance.tool_trace must be a list")
    else:
        retrieval_config = provenance.get("retrieval_config")
        if (
            not isinstance(retrieval_config, dict)
            or not retrieval_config
            or retrieval_config.get("method") != provenance.get("retrieval_method")
        ):
            errors.append("provenance retrieval configuration is invalid")
        trace = provenance["tool_trace"]
        if provenance.get("tool_call_count") != len(trace):
            errors.append("tool_call_count does not match tool_trace")
        if [
            item.get("sequence") for item in trace if isinstance(item, dict)
        ] != list(range(1, len(trace) + 1)):
            errors.append("tool trace sequence is not contiguous")
    return errors
