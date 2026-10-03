"""Runtime checks for the product response contract."""

from __future__ import annotations

import hashlib
import re
from typing import Any

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
        # agent_run is optional — present only in LLM-agent responses, absent in
        # rule-based PubMedQA responses.  When present it must be well-formed.
        agent_run = provenance.get("agent_run")
        if agent_run is not None:
            _VALID_RUN_STATUSES = {"completed", "text_exit", "budget_exhausted", "errored"}
            if not isinstance(agent_run, dict):
                errors.append("provenance.agent_run must be an object")
            else:
                if agent_run.get("run_status") not in _VALID_RUN_STATUSES:
                    errors.append(
                        "provenance.agent_run.run_status must be one of "
                        + str(sorted(_VALID_RUN_STATUSES))
                    )
                if not isinstance(agent_run.get("termination_reason"), str):
                    errors.append(
                        "provenance.agent_run.termination_reason must be a string"
                    )
    return errors


# ---------------------------------------------------------------------------
# Source-span coverage — advisory provenance check, NOT semantic validation
# ---------------------------------------------------------------------------
#
# This reports whether each literal figure the answer asserts (an effect
# estimate, a confidence interval, a percentage, a p-value, a sample size) can
# be found inside the character span of some citation.  It answers exactly one
# question: *is this figure traceable to a cited span?*
#
# It deliberately does NOT answer: does the cited span support the claim?  A
# figure being present inside a cited span does not establish that it was read
# in the right context, that it refers to the same comparison, that its
# direction is correct, or that the claim follows from the evidence.  Those are
# entailment judgements that this check cannot make, and its output says so.
#
# Normalisation is limited to cosmetic equivalences — whitespace, digit-group
# commas, dash variants, spacing around "%" and "=".  There is no synonym
# substitution, no numeric tolerance and no inference, so a paraphrase that
# changes the wording of a figure will be reported as uncovered rather than
# silently matched.

_ASSERTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "effect_estimate",
        re.compile(
            r"\b(?:a?HR|a?OR|a?RR|hazard\s+ratio|odds\s+ratio|risk\s+ratio|"
            r"relative\s+risk)\s*[=:]?\s*\d+(?:\.\d+)?",
            re.IGNORECASE,
        ),
    ),
    (
        "confidence_interval",
        re.compile(
            r"\b(?:95\s?%?\s*CI|CI)\s*[=:]?\s*\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?",
            re.IGNORECASE,
        ),
    ),
    ("p_value", re.compile(r"\bp\s*[=<>]\s*0?\.\d+", re.IGNORECASE)),
    (
        "sample_size",
        re.compile(
            r"\b(?:n\s*=\s*)?\d[\d,]*(?:\.\d+)?\s+"
            r"(?:patients|participants|subjects|adults|children|women|men|individuals)\b",
            re.IGNORECASE,
        ),
    ),
    ("percentage", re.compile(r"\d+(?:[.,]\d+)?\s?%")),
)


def _normalise_fragment(text: str) -> str:
    """Fold cosmetic-only differences so span containment is not defeated by them.

    Only rendering variants are folded: whitespace runs, digit-group commas,
    dash glyphs, and spacing around "%", "=" and ":".  A figure rewritten in a
    different notation (for example "HR 0.72" versus "hazard ratio 0.72") is
    deliberately left uncovered so it surfaces for review rather than passing
    silently.
    """
    folded = text.lower()
    for dash in ("–", "—", "−"):
        folded = folded.replace(dash, "-")
    folded = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", folded)  # 1,200 -> 1200
    folded = re.sub(r"\s+", " ", folded)
    folded = re.sub(r"\s+%", "%", folded)
    folded = re.sub(r"\s*([=:])\s*", r"\1", folded)
    folded = re.sub(r"\s*-\s*", "-", folded)
    return folded.strip()


def _extract_assertions(answer: str) -> list[tuple[int, int, str, str]]:
    """Return (start, end, kind, literal), longest-match-first so a CI beats its '95%'."""
    found: list[tuple[int, int, str, str]] = []
    for kind, pattern in _ASSERTION_PATTERNS:
        for match in pattern.finditer(answer):
            found.append((match.start(), match.end(), kind, match.group(0)))
    # Longer spans win; drop any span fully contained in an already-kept one.
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    kept: list[tuple[int, int, str, str]] = []
    for start, end, kind, literal in found:
        if any(start >= ks and end <= ke for ks, ke, _, _ in kept):
            continue
        kept.append((start, end, kind, literal))
    kept.sort(key=lambda item: item[0])
    return kept


def _containment_pattern(needle: str) -> re.Pattern[str]:
    """Literal match that refuses to land inside a longer numeric token.

    Plain substring lookup reports "28%" as covered by a span holding "128%",
    and "HR 0.72" as covered by "HR 0.729" — a false pass on exactly the figures
    an audit most needs to catch.

    The two edges are guarded differently, on purpose:

    * Leading edge, ``(?<![\\d.,])``.  A preceding dot or comma means the digits
      are the tail of a longer token, so "200" must not match the "200" inside
      "1,200", and "72" must not match the "72" inside "0.72".  Both characters
      are excluded for that reason.
    * Trailing edge, ``(?!\\d)`` — digits only.  A trailing dot or comma is
      ordinary punctuation: "HR 0.72, 95% CI" and "reduced by 28%." are normal
      prose, so excluding them would reject valid neighbours instead of
      preventing a false pass.  Only a directly following digit continues the
      number ("0.72" inside "0.729").

    Each guard applies only when that edge of the needle is itself a digit.  A
    needle ending in "%" or in "patients" needs no trailing guard, and adding one
    there would reject punctuation that legitimately follows.
    """
    escaped = re.escape(needle)
    prefix = r"(?<![\d.,])" if needle[:1].isdigit() else ""
    suffix = r"(?!\d)" if needle[-1:].isdigit() else ""
    return re.compile(prefix + escaped + suffix)


# Deliberately does not absorb trailing whitespace.  A match running one
# character past the number would fall outside the classified span, so a figure
# that *was* classified would be reported as unclassified whenever a space or a
# newline followed it.
_NUMERIC_LITERAL_RE = re.compile(r"\d+(?:[.,]\d+)*%?")


def _unclassified_numeric_literals(
    answer: str, classified: list[tuple[int, int]]
) -> list[str]:
    """Numeric literals in the answer that no extraction pattern classified.

    Reported so that a quiet result cannot be read as "every figure was checked".
    Nothing is asserted about these: they are neither covered nor uncovered, they
    are simply outside the check's scope until a pattern is added for them.
    """
    seen: list[str] = []
    for match in _NUMERIC_LITERAL_RE.finditer(answer):
        if any(match.start() >= start and match.end() <= end for start, end in classified):
            continue
        literal = match.group(0).strip()
        if literal and literal not in seen:
            seen.append(literal)
    return seen


_SEMANTIC_SCOPE_LIMITS = (
    "Presence inside a cited span is not entailment: this does not verify that the "
    "figure was read in the correct context, that it refers to the same comparison, "
    "that its direction is right, or that the claim follows from the evidence.",
    "Matching is literal after cosmetic normalisation only. A figure rewritten in "
    "different words (for example 'hazard ratio 0.72' against a source reading "
    "'HR 0.72') is reported as uncovered even when the source supports it; the "
    "check never treats one notation as equivalent to another.",
    "Numeric matching is boundary-aware, so '28%' is not accepted against a span "
    "holding '128%', and 'HR 0.72' is not accepted against 'HR 0.729'. The guard "
    "applies only to numeric edges, so ordinary neighbours still match.",
    "Only the extracted figure classes are checked. Numeric literals outside them "
    "are listed in unclassified_numeric_literals and are neither covered nor "
    "uncovered — the check makes no statement about them at all.",
    "Absence of uncovered items does not mean the answer is correct.",
)


def summarize_source_coverage(answer: object, citations: object) -> dict[str, Any]:
    """Report which literal answer figures fall inside a cited span.

    Advisory only: the result is recorded for audit and never gates a verdict.
    """
    report: dict[str, Any] = {
        "check": "cited_span_coverage",
        "is_semantic_validation": False,
        "scope": "literal traceability of quantitative figures to a cited span",
        "limitations": list(_SEMANTIC_SCOPE_LIMITS),
        "assertions": [],
        "counts": {"total": 0, "covered": 0, "uncovered": 0},
        "uncovered": [],
        "unclassified_numeric_literals": [],
        "skipped_reason": None,
    }
    if not isinstance(answer, str) or not answer.strip():
        report["skipped_reason"] = "answer is empty or not a string"
        return report
    if not isinstance(citations, list) or not citations:
        report["skipped_reason"] = "no citations to check against"
        return report

    spans: list[tuple[str, str]] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        snippet = citation.get("snippet")
        citation_id = citation.get("citation_id")
        if isinstance(snippet, str) and isinstance(citation_id, str):
            spans.append((citation_id, _normalise_fragment(snippet)))
    if not spans:
        report["skipped_reason"] = "no citation carried a usable snippet"
        return report

    assertions: list[dict[str, Any]] = []
    uncovered: list[str] = []
    classified_spans: list[tuple[int, int]] = []
    for start, end, kind, literal in _extract_assertions(answer):
        classified_spans.append((start, end))
        needle = _normalise_fragment(literal)
        pattern = _containment_pattern(needle) if needle else None
        matched = [
            cid for cid, span in spans if pattern is not None and pattern.search(span)
        ]
        assertions.append({
            "kind": kind,
            "text": literal,
            "covered": bool(matched),
            "citation_ids": matched,
        })
        if not matched:
            uncovered.append(literal)

    report["assertions"] = assertions
    report["counts"] = {
        "total": len(assertions),
        "covered": len(assertions) - len(uncovered),
        "uncovered": len(uncovered),
    }
    report["uncovered"] = uncovered
    report["unclassified_numeric_literals"] = _unclassified_numeric_literals(
        answer, classified_spans
    )
    return report
