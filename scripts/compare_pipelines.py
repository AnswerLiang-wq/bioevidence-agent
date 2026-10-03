"""Compare rule-based PubMedQAEvidenceAgent vs LLMEvidenceAgent.

Usage
-----
# Dry-run: tiny fixture corpus, mocked LLM loop (no API key needed)
python scripts/compare_pipelines.py --dry-run

# Full run: real PubMedQA benchmark + DeepSeek API
python scripts/compare_pipelines.py \\
    --benchmark-dir path/to/pubmedqa_benchmark \\
    --deepseek-key sk-... \\
    --n-cases 50

Metrics reported
----------------
  accuracy        fraction where predicted verdict matches gold label
  abstain_rate    fraction of questions answered with verdict=insufficient
  recall_at_1     gold PMID is rank-1 hit returned by retrieval (rule-based only)
  citation_hit    gold PMID appears in final cited_pmids (LLM agent only)
  avg_latency_ms  mean wall-clock time per question
  cost_usd_total  estimated DeepSeek cost for all LLM-agent questions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── make sure src/ is on the path when run as a script ──────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from bioevidence.audit import AuditingClient, summarize_rounds  # noqa: E402
from bioevidence.prompt_variants import (  # noqa: E402
    arm_c_intervention_diff,
    arm_c_system_prompt,
    arm_c_tool_schemas,
)
from bioevidence.bm25 import BM25Index  # noqa: E402, F401  kept for dry-run corpus build
from bioevidence.corpus import CorpusDocument, read_corpus  # noqa: E402
from bioevidence.fixed_context_agent import (  # noqa: E402
    DEFAULT_TOP_K as BASELINE_TOP_K,
)
from bioevidence.fixed_context_agent import (  # noqa: E402
    FIXED_CONTEXT_SYSTEM_PROMPT,
    FixedContextLLMAgent,
)
from bioevidence.llm_agent import LLMEvidenceAgent, _estimate_cost_usd  # noqa: E402
from bioevidence.pubmedqa import (  # noqa: E402
    CORPUS_FILENAME,
    PUBMEDQA_LABELS,
    PUBMEDQA_VERDICT_MAP,
    TEST_GOLD_FILENAME,
    TEST_INPUTS_FILENAME,
    TRAIN_FILENAME,
    TfidfLogisticAnswerer,
    read_jsonl,
)
from bioevidence.pubmedqa_agent import PubMedQAEvidenceAgent  # noqa: E402
from bioevidence.pubmedqa_sample import select_cases  # noqa: E402
from bioevidence.retrievers import BM25Retriever, RetrievalHit  # noqa: E402, F401
from bioevidence.tools import (  # noqa: E402
    FetchRecordInput,
    InspectEvidenceInput,
    LiteratureTools,
    SearchLiteratureInput,
    ToolExecutor,
)

# ── fixture corpus / questions for dry-run ──────────────────────────────────

FIXTURE_CORPUS = REPO_ROOT / "src" / "bioevidence" / "fixtures" / "tiny_corpus.jsonl"

FIXTURE_QUESTIONS = [
    {
        "case_id": "fix-001",
        "question": "Do mitochondria change shape before programmed cell death?",
        "gold_pmid": "1001",
        "gold_label": "yes",
    },
    {
        "case_id": "fix-002",
        "question": "Can refrigerators cause vaccine temperature problems?",
        "gold_pmid": "1002",
        "gold_label": "yes",
    },
    {
        "case_id": "fix-003",
        "question": "Does serum inhibin decline after molar pregnancy evacuation?",
        "gold_pmid": "1003",
        "gold_label": "yes",
    },
]


# ── mock LLM agent for dry-run (scripted 3-step loop, no API call) ───────────

class MockLLMEvidenceAgent:
    """Deterministic stand-in that mimics LLMEvidenceAgent's output shape.

    Runs a fixed search → fetch → finish loop so that the comparison harness
    can be tested end-to-end without a DeepSeek API key.
    """

    AGENT_VERSION = "mock-react-v1"

    def __init__(self, tools: LiteratureTools, *, model: str = "mock") -> None:
        self._tools = tools
        self._model = model

    def run(self, *, question: str, request_id: str | None = None) -> dict[str, Any]:
        from bioevidence.product_contracts import validate_product_response

        rid = request_id or hashlib.md5(question.encode()).hexdigest()[:8]
        executor = ToolExecutor(self._tools, max_calls=6)
        t_start = time.time()

        # step 1 — search
        search_result = executor.search(
            SearchLiteratureInput(query=question, top_k=3, rerank_query=question)
        )
        hits = search_result.hits
        cited_pmids: list[str] = []
        records: dict[str, Any] = {}
        snippets_map: dict[str, list[Any]] = {}

        if hits:
            top_hit = hits[0]
            pmid = top_hit.pmid

            # step 2 — fetch
            record = executor.fetch(FetchRecordInput(pmid=pmid))
            records[pmid] = record

            # step 3 — inspect
            inspect_result = executor.inspect(
                InspectEvidenceInput(pmid=pmid, query=question, max_snippets=2)
            )
            snippets_map[pmid] = inspect_result.snippets
            cited_pmids = [pmid]

        elapsed_ms = (time.time() - t_start) * 1000

        # assemble citations
        citations: list[dict[str, Any]] = []
        for i, pmid in enumerate(cited_pmids):
            rec = records.get(pmid)
            snips = snippets_map.get(pmid, [])
            if not rec or not snips:
                continue
            snip = snips[0]
            hit_info = next((h for h in hits if h.pmid == pmid), None)
            ret_rank = max(1, int(hit_info.rank)) if hit_info else 1
            ret_score = float(hit_info.score) if hit_info else 1.0
            ret_method = hit_info.retrieval_method if hit_info else "bm25"
            citations.append({
                "citation_id": f"C{i + 1}",
                "pmid": pmid,
                "doi": None,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "title": rec.title,
                "source_sha256": rec.document_sha256,
                "section": snip.section,
                "snippet": snip.text,
                "start_char": snip.start_char,
                "end_char": snip.end_char,
                "snippet_sha256": snip.snippet_sha256,
                "direction": "supporting",
                "retrieval": {
                    "method": ret_method,
                    "rank": ret_rank,
                    "score": ret_score,
                    "component_ranks": {ret_method: ret_rank},
                    "candidate_rank": None,
                },
            })

        verdict = "supported" if citations else "insufficient"
        response = {
            "response_version": "1.0.0",
            "baseline_id": f"{self.AGENT_VERSION}-{self._model}",
            "request_id": rid,
            "verdict": verdict,
            "abstained": verdict == "insufficient",
            "answer": "Mock answer based on top retrieval result." if citations else "Insufficient evidence.",
            "claims": [{"text": "Mock claim.", "citation_ids": ["C1"]}] if citations else [],
            "citations": citations,
            "decisive_reason": "Mock decisive reason.",
            "scope_limits": [
                "Answers are based on the local PubMed snapshot only.",
                "LLM decisions are probabilistic; outputs require human expert review.",
                "No clinical advice is implied.",
            ],
            "confidence": 0.7 if citations else 0.0,
            "provenance": {
                "corpus_sha256": None,
                "retrieval_method": "llm_react_loop",
                "retrieval_config": {
                    "method": "llm_react_loop",
                    "model": self._model,
                    "max_steps": 8,
                    "steps_used": 3 if citations else 1,
                },
                "answer_model": {
                    "type": "mock",
                    "model": self._model,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "elapsed_ms": round(elapsed_ms, 1),
                },
                "answer_probabilities": {},
                "tool_call_count": len(executor.trace),
                "tool_trace": [
                    {"sequence": i + 1, "tool": e.tool_name, "args": {}, "result": {}}
                    for i, e in enumerate(executor.trace)
                ],
                "model_api_cost_usd": 0.0,
                # agent_run is LLM-agent-specific; present here so the evaluator can
                # read run_status from provenance.agent_run on the standard path.
                "agent_run": {
                    "run_status": "completed",
                    "termination_reason": "finish tool accepted.",
                },
            },
        }

        errors = validate_product_response(response)
        if errors:
            raise ValueError(f"Mock agent contract failure: {errors}")
        return response


# ── Helpers ──────────────────────────────────────────────────────────────────

VERDICT_TO_LABEL = {v: k for k, v in PUBMEDQA_VERDICT_MAP.items()}


def _load_dry_run_data() -> tuple[list[CorpusDocument], list[dict[str, Any]]]:
    docs = list(read_corpus(FIXTURE_CORPUS))
    return docs, FIXTURE_QUESTIONS


def _load_benchmark_data(
    benchmark_dir: Path,
    n_cases: int | None,
) -> tuple[list[CorpusDocument], list[dict[str, Any]]]:
    corpus_path = benchmark_dir / "corpus.jsonl"
    docs = list(read_corpus(corpus_path))
    inputs_path = benchmark_dir / "test_inputs.jsonl"
    gold_path = benchmark_dir / "test_gold.jsonl"
    inputs = [json.loads(line) for line in inputs_path.read_text().splitlines() if line.strip()]
    gold_rows = [json.loads(line) for line in gold_path.read_text().splitlines() if line.strip()]
    gold_by_id = {str(r["case_id"]): r for r in gold_rows}
    cases = [
        {
            "case_id": str(row["case_id"]),
            "question": row["question"],
            "gold_pmid": str(gold_by_id[str(row["case_id"])]["pmid"]),
            "gold_label": gold_by_id[str(row["case_id"])]["label"],
        }
        for row in inputs
        if str(row["case_id"]) in gold_by_id
    ]
    if n_cases:
        cases = cases[:n_cases]
    return docs, cases


def _build_rule_based_agent(
    docs: list[CorpusDocument],
    corpus_sha: str | None,
    *,
    benchmark_dir: Path | None = None,
) -> PubMedQAEvidenceAgent:
    retriever = BM25Retriever(docs)
    answerer = TfidfLogisticAnswerer()
    by_pmid = {d.pmid: d for d in docs}

    if benchmark_dir is not None:
        # Load the official train split — strict train/test separation means
        # we never use test PMIDs here.  Training failure is a hard error;
        # silently swallowing it would let predict() fail mid-run with a
        # cryptic RuntimeError and potentially produce misleading metrics.
        train_path = benchmark_dir / "train.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(
                f"Official train split not found: {train_path}. "
                "Run the benchmark preparation step first."
            )
        train_rows = [
            json.loads(line)
            for line in train_path.read_text().splitlines()
            if line.strip()
        ]
        # This raises naturally (RuntimeError / ValueError) if fitting fails;
        # the caller should let it propagate rather than masking it.
        answerer.fit(train_rows, by_pmid)
    # In dry-run mode (benchmark_dir is None) the fixture corpus is too small
    # and homogeneous to train the classifier.  We leave the answerer unfitted;
    # any predict() call will raise RuntimeError("must be fitted") which will
    # surface as a logged per-case error below — not silently counted as correct.

    return PubMedQAEvidenceAgent(
        docs,
        corpus_sha256=corpus_sha or "unknown",
        retriever=retriever,
        answerer=answerer,
    )


def _verdict_matches_label(verdict: str, gold_label: str) -> bool:
    expected_verdict = PUBMEDQA_VERDICT_MAP.get(gold_label)
    return verdict == expected_verdict


def _predicted_label(verdict: str) -> str | None:
    """Map a product verdict onto a PubMedQA gold label, or None if it is none.

    ``mixed`` is a valid product verdict but the benchmark has no mixed gold
    label.  Returning None keeps it out of every per-class count, so it can never
    be silently folded into "maybe"; the caller reports how often it happened.
    """
    for label in PUBMEDQA_LABELS:
        if verdict == PUBMEDQA_VERDICT_MAP[label]:
            return label
    return None


#: What the cost figures are, stated once so no reader has to paraphrase it.
#: The agent prices tokens at peak-hour cache-miss rates.  For a run whose usage
#: was fully reported that makes the number a conservative ceiling, never a bill
#: and never a lower bound — real charges fall to roughly half off-peak and far
#: lower still on cache hits, and there is no invoice to confirm either.
COST_ESTIMATE_SEMANTICS = (
    "estimate, not a bill — tokens are priced at peak-hour cache-miss rates, so a "
    "fully reported run yields a conservative ceiling rather than a lower bound; "
    "off-peak is about half and cache hits are far cheaper, but no invoice is "
    "available to confirm the true charge"
)

#: The same caveat for a subtotal that covers only part of a run.  Here the
#: direction is not merely conservative — it is unknown, because rounds that
#: failed or reported no usage contribute nothing to the figure while still
#: having cost money.
KNOWN_SUBTOTAL_SEMANTICS = (
    "estimate, not a bill and not a lower bound — covers only the rounds whose "
    "usage the server reported, so rounds that failed or reported nothing are "
    "absent from it; the true charge for the whole run is unknown"
)


#: Scoring policies.  `legacy` is the v1/v2 mapping and is the default so those
#: paths keep producing exactly the numbers they produced before; `v3` is the
#: shared A/C mapping declared for the H3 experiment.  They are separate
#: implementations on purpose — a global change to the legacy mapping would
#: silently rewrite v1/v2 results.
SCORING_POLICIES = ("legacy", "v3")
DEFAULT_SCORING_POLICY = "legacy"

#: v3 mapping: both `mixed` and `insufficient` are counted as a *maybe*
#: prediction.  That is what makes it possible to test whether C's prompt change
#: moves verdicts toward the maybe class, and applying it to A as well keeps the
#: A/C comparison from being confounded by scoring.
_V3_VERDICT_TO_LABEL = {
    "supported": "yes",
    "contradicted": "no",
    "mixed": "maybe",
    "insufficient": "maybe",
}


def _predicted_label_v3(verdict: str) -> str | None:
    return _V3_VERDICT_TO_LABEL.get(verdict)


def _is_unscorable(policy: str, verdict: str) -> bool:
    """True when a policy's mapping gives the verdict no gold label at all.

    Under legacy, ``mixed`` maps to nothing and is therefore unscorable — that is
    the established v1/v2 meaning and it is unchanged.  Under v3, ``mixed`` maps
    to maybe, so it is an ordinary prediction and must not be flagged as
    unscorable.
    """
    return _label_of_for(policy)(verdict) is None


def _label_of_for(policy: str):
    """The verdict→label function a policy uses."""
    if policy == "v3":
        return _predicted_label_v3
    return _predicted_label


def _matches_label_for(policy: str):
    """The verdict/gold comparison a policy uses."""
    if policy == "v3":
        def _v3_matches(verdict: str, gold_label: str) -> bool:
            return _predicted_label_v3(verdict) == gold_label
        return _v3_matches
    return _verdict_matches_label


def _macro_f1_v3(
    results: list[CaseResult],
    *,
    verdict_attr: str,
    errored_attr: str,
    status_attr: str | None,
) -> dict[str, Any]:
    """v3 macro-F1 over **every attempted case**, including confirmed failures.

    Two rules distinguish it from the legacy metric:

    * A case that failed to complete contributes an FN to whichever gold label it
      carries and nothing else.  The fallback verdict such a case returned is
      never read: a failure is not a prediction, so it can never be a true
      positive or a false positive — not even when the fallback happens to be
      ``insufficient``.
    * Failed cases stay in the denominator, because the metric is defined over
      all attempted cases rather than over those that completed.

    The three classes are always averaged, and a class whose denominator
    (2*TP + FP + FN) is zero scores 0.
    """
    per_class: dict[str, Any] = {}
    scores: list[float] = []
    for label in PUBMEDQA_LABELS:
        tp = fp = fn = 0
        for result in results:
            gold = result.gold_label
            if status_attr is None:
                failed = bool(getattr(result, errored_attr))
            else:
                failed = _failure_status(
                    bool(getattr(result, errored_attr)),
                    str(getattr(result, status_attr) or ""),
                ) is not None
            if failed:
                if gold == label:
                    fn += 1
                continue
            predicted = _predicted_label_v3(str(getattr(result, verdict_attr) or ""))
            if gold == label and predicted == label:
                tp += 1
            elif gold != label and predicted == label:
                fp += 1
            elif gold == label:
                fn += 1
        denominator = 2 * tp + fp + fn
        f1 = (2 * tp / denominator) if denominator else 0.0
        scores.append(f1)
        per_class[label] = {
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1": round(f1, 4),
        }
    exact = sum(scores) / len(scores) if results else None
    return {
        "macro_f1": round(exact, 4) if exact is not None else None,
        # Kept unrounded: the acceptance threshold is compared against this, not
        # against the 4-decimal display value.
        "macro_f1_exact": exact,
        "n_classes_averaged": len(PUBMEDQA_LABELS) if results else 0,
        "per_class": per_class,
        "denominator": len(results),
        "scoring_policy": "v3",
        "denominator_note": (
            f"macro-F1 is computed over all {len(results)} attempted case(s), "
            "including confirmed failures; a failed case contributes only an FN to "
            "its gold class and is never scored from its fallback verdict"
        ),
    }


def _macro_f1(completed: list[CaseResult], verdict_attr: str) -> dict[str, Any]:
    """Macro-F1 over completed cases, always averaging all three gold labels.

    Each class scores ``2*TP / (2*TP + FP + FN)`` — the standard F1 written so the
    zero case is explicit.  Consequences worth stating because they are the point:

    * A class that occurs in gold but is never predicted scores 0 (TP=0, FN>0).
    * A class that never occurs but is wrongly predicted also scores 0 (TP=0, FP>0).
    * A class with no TP, FP or FN at all has denominator 0 and is defined as 0.

    No class is ever dropped from the average.  Excluding a class because a small
    sample happens not to contain it would let the macro average rise by deleting
    the very miss it exists to expose.

    With no completed cases there is nothing to average, so the result is None —
    not 0.0 and not 1.0.

    ``verdict_attr`` selects the arm's verdict field, so the rule-based arm is
    scored by exactly the same formula as the LLM arm.
    """
    per_class: dict[str, Any] = {}
    scores: list[float] = []
    for label in PUBMEDQA_LABELS:
        tp = fp = fn = 0
        for result in completed:
            predicted = _predicted_label(getattr(result, verdict_attr))
            if result.gold_label == label and predicted == label:
                tp += 1
            elif result.gold_label != label and predicted == label:
                fp += 1
            elif result.gold_label == label:
                fn += 1
        denominator = 2 * tp + fp + fn
        f1 = (2 * tp / denominator) if denominator else 0.0
        scores.append(f1)
        per_class[label] = {
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1": round(f1, 4),
        }
    return {
        "macro_f1": round(sum(scores) / len(scores), 4) if completed else None,
        "n_classes_averaged": len(scores) if completed else 0,
        "per_class": per_class,
        "denominator": len(completed),
        "denominator_note": (
            f"macro-F1 is computed over {len(completed)} completed case(s) and always "
            "averages all three gold labels; cases that failed to complete produced "
            "no prediction and are excluded entirely"
        ),
    }


def _mean_or_none(values: list[float | None]) -> tuple[float | None, int]:
    """Mean over the recorded values, plus how many those were.

    A merged report can hold cases whose segment did not record a latency.
    Averaging only what exists and stating how many that was is honest;
    substituting zero for the absent ones would not be.
    """
    known = [value for value in values if value is not None]
    if not known:
        return None, 0
    return round(sum(known) / len(known), 1), len(known)


def _arm_metrics(
    completed: list[CaseResult], verdict_attr: str, correct_of: Any
) -> dict[str, Any]:
    """Macro-F1 plus accuracy-over-completed, shared by both arms.

    Both arms are scored by the same formula so their numbers are comparable;
    only the verdict field and the notion of "correct" differ.
    """
    detail = _macro_f1(completed, verdict_attr)
    correct = sum(1 for r in completed if correct_of(r))
    return {
        "macro_f1": detail["macro_f1"],
        "macro_f1_detail": detail,
        "n_completed": len(completed),
        "n_correct": correct,
        "accuracy_on_completed": (
            round(correct / len(completed), 4) if completed else None
        ),
    }


def _citation_hit(response: dict[str, Any], gold_pmid: str) -> bool:
    citations = response.get("citations", [])
    return any(str(c.get("pmid")) == gold_pmid for c in citations)


# ── Core comparison function ──────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    question: str
    gold_label: str
    gold_pmid: str
    rule_verdict: str
    rule_abstained: bool
    rule_errored: bool          # True when an exception was raised — exclude from accuracy
    rule_error_msg: str | None
    rule_recall_at_1: bool
    rule_latency_ms: float
    llm_verdict: str
    llm_abstained: bool
    llm_errored: bool           # True when an exception was raised — exclude from accuracy
    llm_error_msg: str | None
    llm_citation_hit: bool
    llm_latency_ms: float
    llm_steps_used: int
    # run_status distinguishes how the LLM loop ended:
    #   "completed"        — finish accepted; a real prediction
    #   "text_exit"        — model answered in plain text without calling finish
    #   "budget_exhausted" — max_steps reached without an accepted finish call
    #   "errored"          — the agent raised; no verdict exists at all
    # Evaluators must not count text_exit, budget_exhausted or errored as correct
    # predictions, and must not count them as legitimate "insufficient" answers.
    llm_run_status: str
    # None when the response carried no usable cost.  Never 0.0: a missing cost
    # summed as zero would understate the total while looking confirmed.
    llm_cost_usd: float | None = None
    # provenance.answer_model.usage_coverage — "confirmed" | "partial" | "unknown".
    # Carries the three-way cost distinction per case so the aggregate can say
    # whether the total it prints is complete.
    llm_usage_coverage: str = "unknown"
    # Estimate for exactly those rounds whose usage the server reported.  Unlike
    # llm_cost_usd this survives a mid-run failure, so a case that died on round 2
    # still accounts for round 1's tokens.  None when no round reported usage.
    llm_cost_known_subtotal_usd: float | None = None
    # Compact per-case record of the round-trips the model actually made:
    # request count, the model identifier the server reported, token usage, the
    # tool calls emitted, and why the loop ended.
    llm_audit: dict[str, Any] | None = None
    # Where llm_steps_used came from.  "agent_provenance" is the loop's own count;
    # "audit_request_count" means the agent raised before reporting, so the number
    # is the round-trips that were actually attempted; "unknown" means neither.
    llm_steps_source: str = "unknown"
    # ── Arm B: fixed-context LLM baseline ────────────────────────────────────
    # Mirrors the agent arm's fields so both are scored by identical code.
    # Absent (defaults) when no baseline arm was run.
    baseline_verdict: str = ""
    baseline_abstained: bool = False
    baseline_errored: bool = False
    baseline_error_msg: str | None = None
    baseline_citation_hit: bool = False
    baseline_latency_ms: float | None = None
    baseline_steps_used: int = 0
    baseline_steps_source: str = "unknown"
    baseline_run_status: str = ""
    baseline_cost_usd: float | None = None
    baseline_usage_coverage: str = "unknown"
    baseline_cost_known_subtotal_usd: float | None = None
    baseline_audit: dict[str, Any] | None = None
    # ── Arm C: the prompt-variant agent (v3/H3) ──────────────────────────────
    # Same shape as the other model arms, so all three are scored by one code
    # path and C's only difference from A is the prompt material recorded in
    # the report's config and prompt fingerprint.
    arm_c_verdict: str = ""
    arm_c_abstained: bool = False
    arm_c_errored: bool = False
    arm_c_error_msg: str | None = None
    arm_c_citation_hit: bool = False
    arm_c_latency_ms: float | None = None
    arm_c_steps_used: int = 0
    arm_c_steps_source: str = "unknown"
    arm_c_run_status: str = ""
    arm_c_cost_usd: float | None = None
    arm_c_usage_coverage: str = "unknown"
    arm_c_cost_known_subtotal_usd: float | None = None
    arm_c_audit: dict[str, Any] | None = None
    # ── diagnostic material (v3) ─────────────────────────────────────────────
    # Enough to check an evidence chain later, not just the verdict.
    llm_answer: str = ""
    llm_claim: str = ""
    llm_decisive_reason: str = ""
    llm_tool_returns: list[dict[str, Any]] | None = None
    arm_c_answer: str = ""
    arm_c_claim: str = ""
    arm_c_decisive_reason: str = ""
    arm_c_tool_returns: list[dict[str, Any]] | None = None


#: The coverage vocabulary, ordered least-to-most complete.
_COVERAGE_RANK = {"unknown": 0, "partial": 1, "confirmed": 2}


def _merge_usage_coverage(reported: str | None, audited: str | None) -> str:
    """One coverage label per case, agreed by the audit, the case and the summary.

    The audit wins whenever it exists: it is computed from the round-trips that
    actually happened, whereas the agent's own label is absent entirely on a run
    that raised.  Treating that absence as an observation and letting it outrank
    the audit would record a first-round success as "unknown" — the opposite of
    what was seen.

    The reported label stands in only when there is nothing to audit (an agent
    with no client to wrap, such as the dry-run mock).
    """
    if isinstance(audited, str) and audited in _COVERAGE_RANK:
        return audited
    if isinstance(reported, str) and reported in _COVERAGE_RANK:
        return reported
    return "unknown"


def _attach_audit(llm_agent: Any) -> AuditingClient | None:
    """Wrap the agent's client exactly once so round-trips are captured.

    Detection alone was not enough: on the normal construction path the agent
    holds a plain OpenAI client, so nothing was ever wrapped and every case
    reported ``llm_audit=None``.  That in turn made a fully confirmed case look
    unaccountable to the spending cap, which stopped runs that had no reason to
    stop.

    Idempotent — a client that is already an AuditingClient is returned as-is, so
    repeated calls, or a caller that pre-wrapped deliberately, cannot double-wrap.
    Returns None when the agent has no client to audit (the dry-run mock), which
    is reported honestly rather than papered over.
    """
    client = getattr(llm_agent, "_client", None)
    if isinstance(client, AuditingClient):
        return client
    if client is None or not hasattr(client, "chat"):
        return None
    wrapper = AuditingClient(client)
    llm_agent._client = wrapper
    return wrapper


def _case_known_subtotal(case: dict[str, Any]) -> float:
    """Known cost estimate for **every paid arm this case actually ran**.

    Any arm that was billed has to be in here.  A carry-over or a merge that
    summed only the agent arm would understate spend by however much the other
    arms cost — and the next segment could then start cases the experiment budget
    had already consumed.  All three model arms are covered: the agent arm, the
    fixed-context baseline (v2), and the prompt-variant arm (v3).

    A report that never ran an arm carries none of that arm's cost fields and is
    summed without it; that is a missing *arm*, not missing data.  An arm that ran
    but reported no usage contributes nothing to the sum — there is no number to
    add — and is flagged as incomplete by `_budget_state` and by the per-arm cost
    blocks rather than being imputed here.
    """
    total = 0.0
    for key in (
        "llm_cost_known_subtotal_usd",
        "baseline_cost_known_subtotal_usd",
        "arm_c_cost_known_subtotal_usd",
    ):
        value = case.get(key)
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def _report_known_subtotal(report: dict[str, Any]) -> float:
    """Known subtotal across every case in one stored segment report."""
    return sum(_case_known_subtotal(case) for case in report.get("cases") or [])


#: The H3 acceptance threshold, a **pre-declared engineering criterion**.  It is
#: not a measured noise floor and a single 50-case run cannot establish general
#: validity; the report says so wherever it appears.
V3_DELTA_THRESHOLD = 0.04


def _maybe_misroutes(results: list[CaseResult], verdict_attr: str,
                     errored_attr: str, status_attr: str) -> dict[str, int]:
    """How often an arm spent a maybe prediction on a yes/no gold case.

    Reported next to any maybe gain, because a shift toward the maybe class can
    be bought by hedging on cases that are not condition-dependent at all.
    """
    counted = {"yes_to_maybe": 0, "no_to_maybe": 0}
    for result in results:
        if _failure_status(
            bool(getattr(result, errored_attr)), str(getattr(result, status_attr) or "")
        ) is not None:
            continue
        verdict = str(getattr(result, verdict_attr) or "")
        if verdict not in ("mixed", "insufficient"):
            continue
        if result.gold_label == "yes":
            counted["yes_to_maybe"] += 1
        elif result.gold_label == "no":
            counted["no_to_maybe"] += 1
    return counted


def _v3_acceptance(
    results: list[CaseResult],
    metrics: dict[str, Any],
    *,
    frozen_total: int,
    arm_c_enabled: bool,
) -> dict[str, Any]:
    """The pre-declared H3 acceptance test, or a statement that it cannot be run.

    Both conditions must hold: the macro-F1 gain reaches the threshold **and**
    C's correct rate on gold=maybe cases exceeds A's.  Compared on unrounded
    values; the rounded figures in the report are for display only.

    Completeness is judged against the **frozen sample**, not against however
    many cases this segment happened to be asked to run.  A `--n-cases 2` trial
    or a continuation segment covering only the remainder has not attempted the
    frozen set, so neither can carry an acceptance conclusion — only a merge
    whose union covers the whole sample can.

    A case counts as attempted only when **both** arms reached a definite run
    status on it.  A case that ran and failed is attempted (and is scored by the
    pre-declared rules); a case that never started is not.
    """
    a = metrics["llm_agent"]
    c = metrics.get("arm_c")
    attempted = sum(
        1 for r in results
        if r.llm_run_status and r.arm_c_run_status
    )
    not_run = frozen_total - attempted
    block: dict[str, Any] = {
        "design": "v3/H3",
        "scoring_policy": metrics.get("scoring_policy"),
        "delta_threshold": V3_DELTA_THRESHOLD,
        "delta_threshold_note": (
            "pre-declared engineering criterion, not a demonstrated noise floor; "
            "one 50-case run cannot establish general validity"
        ),
        "threshold_basis": "unrounded macro-F1 values are compared, not the rounded display",
        "frozen_total": frozen_total,
        "attempted": attempted,
        "not_run": not_run,
        "arm_c_enabled": arm_c_enabled,
    }
    if not arm_c_enabled:
        block["status"] = "not_applicable"
        block["reason"] = "no Arm C was run, so there is no A/C comparison to make"
        return block
    if attempted < frozen_total:
        block["status"] = "experiment_incomplete"
        block["reason"] = (
            f"{attempted} of {frozen_total} frozen case(s) have a definite run status "
            f"on both arms; {not_run} were never attempted. The primary metric is "
            "defined over the full frozen sample, so no acceptance conclusion is "
            "drawn. Cases that ran and failed are scored by the pre-declared rules "
            "and do not by themselves make the run incomplete."
        )
        block["criteria_met"] = None
        return block
    if c is None:
        # Arm C was requested but no case reached it (for example a budget stop
        # before the first case).  That is an incomplete experiment, not an
        # inapplicable comparison, and the policy must still be reported.
        block["status"] = "experiment_incomplete"
        block["reason"] = (
            "Arm C was enabled but no case produced a run status for it — "
            "typically a spending stop before the first case started"
        )
        block["criteria_met"] = None
        return block

    a_exact = a["macro_f1_detail"].get("macro_f1_exact")
    c_exact = c["macro_f1_detail"].get("macro_f1_exact")
    delta = None if a_exact is None or c_exact is None else c_exact - a_exact
    a_maybe = a["macro_f1_detail"]["per_class"]["maybe"]
    c_maybe = c["macro_f1_detail"]["per_class"]["maybe"]
    a_rate = (a_maybe["tp"] / a_maybe["support"]) if a_maybe["support"] else None
    c_rate = (c_maybe["tp"] / c_maybe["support"]) if c_maybe["support"] else None

    gain_met = delta is not None and delta >= V3_DELTA_THRESHOLD
    maybe_met = (
        a_rate is not None and c_rate is not None and c_rate > a_rate
    )
    block.update({
        "status": "complete",
        "macro_f1_a_exact": a_exact,
        "macro_f1_c_exact": c_exact,
        "delta_macro_f1_c_minus_a": delta,
        "macro_f1_gain_met": gain_met,
        "maybe_correct_rate": {
            "a": a_rate, "c": c_rate,
            "support": a_maybe["support"],
            "a_tp": a_maybe["tp"], "c_tp": c_maybe["tp"],
            "c_exceeds_a": maybe_met,
        },
        "criteria_met": bool(gain_met and maybe_met),
        "conclusion_strength": (
            "exploratory engineering criterion met" if (gain_met and maybe_met)
            else "exploratory engineering criterion not met"
        ),
        "per_class_f1": {
            label: {"a": a["macro_f1_detail"]["per_class"][label]["f1"],
                    "c": c["macro_f1_detail"]["per_class"][label]["f1"],
                    "delta": round(
                        c["macro_f1_detail"]["per_class"][label]["f1"]
                        - a["macro_f1_detail"]["per_class"][label]["f1"], 4)
                    }
            for label in PUBMEDQA_LABELS
        },
        "maybe_misroutes": {
            "a": _maybe_misroutes(results, "llm_verdict", "llm_errored", "llm_run_status"),
            "c": _maybe_misroutes(results, "arm_c_verdict", "arm_c_errored",
                                  "arm_c_run_status"),
            "meaning": (
                "yes/no gold cases on which the arm spent a mixed or insufficient "
                "prediction; a maybe gain accompanied by a rise here may be hedging "
                "rather than condition-dependent recognition"
            ),
        },
        "caveats": [
            "single 50-case run; no confidence interval or significance test is claimed",
            "C and A differ only in prompt material, but the comparison is still "
            "observational: one sample, one model, one corpus",
            "A yes/no F1 cost is reported above; a macro-F1 gain driven only by "
            "yes/no shifts is not evidence for H3",
        ],
    })
    return block


def _budget_state(
    results: list[CaseResult], *, include_baseline: bool = False,
    include_arm_c: bool = False,
) -> dict[str, Any]:
    """Where the run stands against its spending cap, across every paid arm.

    Arm B is billed too.  Leaving it out of the total would understate spend by
    whatever the baseline cost, and would let a case whose *baseline* usage was
    unconfirmed look fully accounted for — so the same conservative stop applies
    to any arm that actually ran.

    When the baseline arm was not run at all it is neither counted nor treated
    as unknown; an arm that does not exist cannot have missing data.

    No imputation: a case whose usage was not confirmed is **not** charged at the
    mean of the others.  Doing that would let a run with unusable cost data look
    cheap and sail past the cap it is supposed to respect.  Instead any
    unconfirmed usage marks the accounting as incomplete, and the caller stops.
    """
    subtotals: list[float] = []
    incomplete_arms: list[str] = []
    for result in results:
        arms: list[tuple[str, float | None, str]] = [
            ("agent", result.llm_cost_known_subtotal_usd, result.llm_usage_coverage)
        ]
        if include_baseline:
            arms.append(
                ("baseline", result.baseline_cost_known_subtotal_usd,
                 result.baseline_usage_coverage)
            )
        if include_arm_c:
            arms.append(
                ("arm_c", result.arm_c_cost_known_subtotal_usd,
                 result.arm_c_usage_coverage)
            )
        for arm, subtotal, coverage in arms:
            if subtotal is not None:
                subtotals.append(subtotal)
            if subtotal is None or coverage != "confirmed":
                incomplete_arms.append(f"{result.case_id}:{arm}")
    return {
        "known_subtotal_usd": round(sum(subtotals), 6),
        "n_known": len(subtotals),
        "n_incomplete": len(incomplete_arms),
        "incomplete_arms": incomplete_arms,
        "usage_incomplete": bool(incomplete_arms),
    }


@dataclass
class _ArmRun:
    """Everything the report needs from one model arm on one case.

    Shared by the agent arm and the fixed-context baseline so both are extracted,
    costed and audited by identical code rather than by two drifting copies.
    """

    verdict: str
    abstained: bool
    errored: bool
    error_msg: str | None
    citation_hit: bool
    latency_ms: float
    steps_used: int
    steps_source: str
    run_status: str
    cost_usd: float | None
    usage_coverage: str
    cost_known_subtotal_usd: float | None
    audit: dict[str, Any] | None
    # Diagnostic material, kept so a later audit can check the evidence chain
    # rather than only the verdict: the public answer text, the claim and the
    # stated reason, plus what the tools actually returned.
    answer: str = ""
    claim: str = ""
    decisive_reason: str = ""
    tool_returns: list[dict[str, Any]] | None = None


def _run_model_arm(
    agent: Any,
    *,
    audit: AuditingClient | None,
    question: str,
    request_id: str,
    gold_pmid: str,
    agent_model: str,
    label: str,
) -> _ArmRun:
    """Run one model arm on one case; a raised exception becomes a recorded failure."""
    if audit is not None:
        audit.reset()
    started = time.time()
    error_msg: str | None = None
    response: dict[str, Any] = {}
    try:
        response = agent.run(question=question, request_id=request_id)
    except Exception as exc:  # noqa: BLE001 - recorded, never repaired
        error_msg = f"{type(exc).__name__}: {exc}"
        print(f"    {label} ERROR [{request_id}]: {error_msg}")
        # Do NOT fabricate a verdict — the case is excluded from accuracy and
        # counted as a run failure.
    finally:
        # Read the audit in a finally block: the rounds already performed are
        # evidence whether or not the arm reached its end.
        rounds = list(audit.rounds) if audit is not None else []
    latency_ms = (time.time() - started) * 1000

    arm_audit = summarize_rounds(rounds) if audit is not None else None

    raw_prov = response.get("provenance") if isinstance(response, dict) else None
    prov: dict[str, Any] = raw_prov if isinstance(raw_prov, dict) else {}
    # model_api_cost_usd is None whenever the run did not obtain confirmed usage
    # for every round.  Calling float() on it used to raise TypeError here —
    # outside the try block — aborting the whole comparison instead of recording
    # the case.
    raw_cost = prov.get("model_api_cost_usd")
    cost_usd = float(raw_cost) if isinstance(raw_cost, (int, float)) else None
    retrieval_config = prov.get("retrieval_config")
    answer_model = prov.get("answer_model")
    reported_coverage = (
        str(answer_model.get("usage_coverage", "unknown"))
        if isinstance(answer_model, dict)
        else "unknown"
    )
    audit_coverage = arm_audit["usage"]["coverage"] if arm_audit else None
    usage_coverage = _merge_usage_coverage(reported_coverage, audit_coverage)

    if isinstance(retrieval_config, dict) and "steps_used" in retrieval_config:
        steps_used = int(retrieval_config.get("steps_used", 0))
        steps_source = "agent_provenance"
    elif arm_audit is not None:
        steps_used = int(arm_audit["request_count"])
        steps_source = "audit_request_count"
    else:
        steps_used = 0
        steps_source = "unknown"

    agent_run = prov.get("agent_run")
    run_status = (
        (agent_run.get("run_status") if isinstance(agent_run, dict) else None)
        or ("errored" if error_msg else "budget_exhausted")
    )

    # Subtotal for the rounds whose usage the server did report, priced by the
    # agent's own estimator so it is comparable with model_api_cost_usd.  Present
    # even when the run failed: the tokens already spent are real whether or not
    # the arm reached its end.
    known_subtotal: float | None = None
    if arm_audit is not None and arm_audit["usage"]["rounds_with_usage"]:
        known_subtotal = _estimate_cost_usd(
            agent_model,
            int(arm_audit["usage"]["prompt_tokens"]),
            int(arm_audit["usage"]["completion_tokens"]),
        )
    if known_subtotal is None:
        # No audit to price from, but the arm's own figure still covers every
        # round it confirmed.  Treating an absent audit as "cost unknown" is what
        # made a fully confirmed case look unaccountable to the spending cap.
        known_subtotal = cost_usd

    # Tool returns are read from the agent's own message list: the response holds
    # only the verdict and its rationale, not the evidence the verdict rests on.
    tool_returns: list[dict[str, Any]] = []
    for message in getattr(agent, "_last_messages", None) or []:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        content = message.get("content")
        parsed: Any = content
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except (TypeError, ValueError):
                parsed = content
        tool_returns.append({"tool_call_id": message.get("tool_call_id"),
                             "content": parsed})

    claims = response.get("claims") if isinstance(response, dict) else None
    claim_text = ""
    if isinstance(claims, list) and claims and isinstance(claims[0], dict):
        claim_text = str(claims[0].get("text") or "")

    return _ArmRun(
        verdict=response.get("verdict", "") if not error_msg else "",
        abstained=bool(response.get("abstained", False)),
        errored=error_msg is not None,
        error_msg=error_msg,
        citation_hit=_citation_hit(response, gold_pmid),
        latency_ms=latency_ms,
        steps_used=steps_used,
        steps_source=steps_source,
        run_status=run_status,
        cost_usd=cost_usd,
        usage_coverage=usage_coverage,
        cost_known_subtotal_usd=known_subtotal,
        audit=arm_audit,
        answer=str(response.get("answer") or "") if not error_msg else "",
        claim=claim_text if not error_msg else "",
        decisive_reason=(
            str(response.get("decisive_reason") or "") if not error_msg else ""
        ),
        tool_returns=tool_returns,
    )


def run_comparison(
    *,
    docs: list[CorpusDocument],
    cases: list[dict[str, Any]],
    llm_agent,
    baseline_agent: Any | None = None,
    arm_c_agent: Any | None = None,
    benchmark_dir: Path | None = None,
    verbose: bool = True,
    max_cost_usd: float | None = None,
    carry_over_cost_usd: float = 0.0,
    stop_state: dict[str, Any] | None = None,
) -> list[CaseResult]:
    """Run both arms over ``cases``, returning one CaseResult per case *started*.

    The spending cap is a **programmatic soft threshold**, checked before each
    case starts, and it cannot bound a bill: the provider bills what it bills,
    and one in-flight case can overshoot by its own cost before the check fires.
    When it stops, ``stop_state`` records which of the two conditions tripped —
    reaching the threshold, or losing the ability to account for spend at all.

    Cases never started are absent from the return value; the caller reports them
    as "not run", which is a different fact from "ran and failed".
    """
    corpus_sha = None
    rule_agent = _build_rule_based_agent(docs, corpus_sha, benchmark_dir=benchmark_dir)
    # Shared BM25Retriever for recall@1 measurement
    bm25_retriever = BM25Retriever(docs)
    results: list[CaseResult] = []

    # Wrap the agent's client once so every round-trip is captured, including the
    # rounds that precede a failure.  Without this a case that dies on round 2
    # reports nothing at all, and its round-1 tokens vanish from the accounting.
    audit = _attach_audit(llm_agent)
    agent_model = str(getattr(llm_agent, "_model", "") or "")
    # The baseline is audited the same way, so one missing tool call is provable
    # from its per-case record rather than merely asserted.
    baseline_audit = (
        _attach_audit(baseline_agent) if baseline_agent is not None else None
    )
    baseline_model = str(getattr(baseline_agent, "_model", "") or "")
    arm_c_audit = _attach_audit(arm_c_agent) if arm_c_agent is not None else None
    arm_c_model = str(getattr(arm_c_agent, "_model", "") or "")

    for i, case in enumerate(cases):
        if max_cost_usd is not None:
            # Checked before EVERY case, including the first: money an earlier
            # segment already spent counts against the same budget, so a
            # continuation can be over its cap before it makes a single request.
            # `_budget_state([])` reports zero segment spend, leaving the
            # carried-over figure to decide on its own.
            state = _budget_state(
                results,
                include_baseline=baseline_agent is not None,
                include_arm_c=arm_c_agent is not None,
            )
            experiment_spend = carry_over_cost_usd + state["known_subtotal_usd"]
            if state["usage_incomplete"]:
                reason, detail = "incomplete_usage", state
            elif experiment_spend >= max_cost_usd:
                reason, detail = "cost_threshold", state
            else:
                reason, detail = None, state
            if reason is not None:
                if verbose:
                    print(
                        f"  BUDGET STOP ({reason}) after {len(results)}/{len(cases)} "
                        f"cases: segment ${state['known_subtotal_usd']:.6f} + carried "
                        f"${carry_over_cost_usd:.6f} = ${experiment_spend:.6f}; "
                        f"{state['n_incomplete']} case(s) with unconfirmed usage"
                    )
                if stop_state is not None:
                    stop_state.update({
                        "reason": reason,
                        "stopped_before_index": i,
                        "cases_completed_before_stop": len(results),
                        "carry_over_cost_usd": round(carry_over_cost_usd, 6),
                        "experiment_subtotal_usd": round(experiment_spend, 6),
                        **detail,
                    })
                break
        qid = case["case_id"]
        question = case["question"]
        gold_label = case["gold_label"]
        gold_pmid = case["gold_pmid"]

        if verbose:
            print(f"  [{i+1}/{len(cases)}] {qid}: {question[:60]}...")

        # Rule-based agent
        t0 = time.time()
        rb_error_msg: str | None = None
        rb_resp: dict[str, Any] = {}
        try:
            rb_resp = rule_agent.run(question=question, request_id=qid + "-rb")
        except Exception as exc:
            rb_error_msg = f"{type(exc).__name__}: {exc}"
            print(f"    RULE ERROR [{qid}]: {rb_error_msg}")
            # Do NOT fabricate a verdict here — the case will be excluded from
            # accuracy metrics via rule_errored=True.
        rb_latency = (time.time() - t0) * 1000

        top_hits = bm25_retriever.search(question, top_k=1)
        rb_recall_at_1 = bool(top_hits and str(top_hits[0].pmid) == gold_pmid)

        # Arm A: the agent, unchanged.
        arm_a = _run_model_arm(
            llm_agent, audit=audit, question=question, request_id=qid + "-llm",
            gold_pmid=gold_pmid, agent_model=agent_model, label="LLM",
        )

        # Arm B: fixed-context baseline, when one was supplied.  Extracted and
        # costed by the same helper, so the two arms cannot diverge in how they
        # are measured.
        arm_b = None
        if baseline_agent is not None:
            arm_b = _run_model_arm(
                baseline_agent, audit=baseline_audit, question=question,
                request_id=qid + "-baseline", gold_pmid=gold_pmid,
                agent_model=baseline_model, label="BASE",
            )

        # Arm C: same loop as A, different prompt material.
        arm_c = None
        if arm_c_agent is not None:
            arm_c = _run_model_arm(
                arm_c_agent, audit=arm_c_audit, question=question,
                request_id=qid + "-armc", gold_pmid=gold_pmid,
                agent_model=arm_c_model, label="ARMC",
            )

        results.append(CaseResult(
            case_id=qid,
            question=question,
            gold_label=gold_label,
            gold_pmid=gold_pmid,
            rule_verdict=rb_resp.get("verdict", "") if not rb_error_msg else "",
            rule_abstained=bool(rb_resp.get("abstained", False)),
            rule_errored=rb_error_msg is not None,
            rule_error_msg=rb_error_msg,
            rule_recall_at_1=rb_recall_at_1,
            rule_latency_ms=rb_latency,
            llm_verdict=arm_a.verdict,
            llm_abstained=arm_a.abstained,
            llm_errored=arm_a.errored,
            llm_error_msg=arm_a.error_msg,
            llm_citation_hit=arm_a.citation_hit,
            llm_latency_ms=arm_a.latency_ms,
            llm_steps_used=arm_a.steps_used,
            llm_run_status=arm_a.run_status,
            llm_cost_usd=arm_a.cost_usd,
            llm_usage_coverage=arm_a.usage_coverage,
            llm_cost_known_subtotal_usd=arm_a.cost_known_subtotal_usd,
            llm_audit=arm_a.audit,
            llm_steps_source=arm_a.steps_source,
            llm_answer=arm_a.answer,
            llm_claim=arm_a.claim,
            llm_decisive_reason=arm_a.decisive_reason,
            llm_tool_returns=arm_a.tool_returns,
            baseline_verdict=arm_b.verdict if arm_b else "",
            baseline_abstained=arm_b.abstained if arm_b else False,
            baseline_errored=arm_b.errored if arm_b else False,
            baseline_error_msg=arm_b.error_msg if arm_b else None,
            baseline_citation_hit=arm_b.citation_hit if arm_b else False,
            baseline_latency_ms=arm_b.latency_ms if arm_b else None,
            baseline_steps_used=arm_b.steps_used if arm_b else 0,
            baseline_steps_source=arm_b.steps_source if arm_b else "unknown",
            baseline_run_status=arm_b.run_status if arm_b else "",
            baseline_cost_usd=arm_b.cost_usd if arm_b else None,
            baseline_usage_coverage=arm_b.usage_coverage if arm_b else "unknown",
            baseline_cost_known_subtotal_usd=(
                arm_b.cost_known_subtotal_usd if arm_b else None
            ),
            baseline_audit=arm_b.audit if arm_b else None,
            arm_c_verdict=arm_c.verdict if arm_c else "",
            arm_c_abstained=arm_c.abstained if arm_c else False,
            arm_c_errored=arm_c.errored if arm_c else False,
            arm_c_error_msg=arm_c.error_msg if arm_c else None,
            arm_c_citation_hit=arm_c.citation_hit if arm_c else False,
            arm_c_latency_ms=arm_c.latency_ms if arm_c else None,
            arm_c_steps_used=arm_c.steps_used if arm_c else 0,
            arm_c_steps_source=arm_c.steps_source if arm_c else "unknown",
            arm_c_run_status=arm_c.run_status if arm_c else "",
            arm_c_cost_usd=arm_c.cost_usd if arm_c else None,
            arm_c_usage_coverage=arm_c.usage_coverage if arm_c else "unknown",
            arm_c_cost_known_subtotal_usd=(
                arm_c.cost_known_subtotal_usd if arm_c else None
            ),
            arm_c_audit=arm_c.audit if arm_c else None,
            arm_c_answer=arm_c.answer if arm_c else "",
            arm_c_claim=arm_c.claim if arm_c else "",
            arm_c_decisive_reason=arm_c.decisive_reason if arm_c else "",
            arm_c_tool_returns=arm_c.tool_returns if arm_c else None,
        ))

    return results


# ── Report generation ─────────────────────────────────────────────────────────

def _failure_status(errored: bool, run_status: str) -> str | None:
    """None when the arm completed; otherwise the failure bucket it belongs to."""
    if errored:
        return "errored"
    if run_status == "completed":
        return None
    return run_status or "errored"


def _failure_breakdown(
    results: list[CaseResult], errored_attr: str, status_attr: str
) -> dict[str, int]:
    statuses = ("errored", "budget_exhausted", "text_exit")
    counts: dict[str, int] = {}
    for result in results:
        status = _failure_status(
            bool(getattr(result, errored_attr)), str(getattr(result, status_attr) or "")
        )
        if status is not None:
            counts[status] = counts.get(status, 0) + 1
    return {
        "errored": counts.get("errored", 0),
        "budget_exhausted": counts.get("budget_exhausted", 0),
        "text_exit": counts.get("text_exit", 0),
        # Anything outside the known taxonomy is surfaced rather than folded into
        # one of the three buckets, which would misreport why cases failed.
        "other": sum(v for k, v in counts.items() if k not in statuses),
    }


def _cost_block(
    results: list[CaseResult], subtotal_attr: str, coverage_attr: str
) -> dict[str, Any]:
    """Cost accounting shared by both model arms.

    Unknown cost is not zero cost: only known subtotals are summed, and the count
    of cases that contributed nothing travels with the total.
    """
    subtotals = [
        getattr(r, subtotal_attr) for r in results
        if getattr(r, subtotal_attr) is not None
    ]
    n_cost_unknown = sum(
        1 for r in results if getattr(r, subtotal_attr) is None
    )
    n_cost_partial_runs = sum(
        1 for r in results
        if getattr(r, subtotal_attr) is not None
        and getattr(r, coverage_attr) != "confirmed"
    )
    usage_coverage_counts = {"confirmed": 0, "partial": 0, "unknown": 0}
    for result in results:
        key = str(getattr(result, coverage_attr))
        usage_coverage_counts[key if key in usage_coverage_counts else "unknown"] += 1

    return {
        "total_cost_usd_known": round(sum(subtotals), 6),
        "n_cost_known": len(subtotals),
        "n_cost_unknown": n_cost_unknown,
        "n_cost_from_partial_runs": n_cost_partial_runs,
        "cost_complete": n_cost_unknown == 0 and n_cost_partial_runs == 0,
        "usage_coverage_counts": usage_coverage_counts,
        "semantics": COST_ESTIMATE_SEMANTICS,
        # Always present: the partial-run caveat is the one a reader most needs,
        # so it travels with the total even when the run happened to be complete.
        "known_subtotal_semantics": KNOWN_SUBTOTAL_SEMANTICS,
        "caveat": (
            f"cost is unknown for {n_cost_unknown} case(s) with no reported usage, and "
            f"{n_cost_partial_runs} case(s) contributed only a partial subtotal; the "
            "total covers the reported rounds alone, so it is neither an upper nor a "
            "lower bound on what was actually charged"
            if (n_cost_unknown or n_cost_partial_runs)
            else "every case reported confirmed usage for every round, so the total "
                 "covers the whole run"
        ),
    }


def _arm_block(
    results: list[CaseResult],
    *,
    verdict_attr: str,
    errored_attr: str,
    status_attr: str | None,
    correct_of: Any,
    n: int,
    scoring_policy: str = DEFAULT_SCORING_POLICY,
) -> dict[str, Any]:
    """Completion rate, accuracy on completed, end-to-end rate and macro-F1.

    All arms go through this, so the columns of the report are produced by one
    implementation rather than by several that can drift.

    ``status_attr=None`` is for an arm with no decision loop: not-errored is then
    its whole completion bar, and errored is its only failure mode.
    """
    if status_attr is None:
        completed = [r for r in results if not getattr(r, errored_attr)]
        breakdown = {"errored": n - len(completed)}
    else:
        completed = [
            r for r in results
            if str(getattr(r, status_attr)) == "completed"
            and not getattr(r, errored_attr)
        ]
        breakdown = _failure_breakdown(results, errored_attr, status_attr)
    arm = _arm_metrics(completed, verdict_attr, correct_of)
    # Under v3 the headline macro-F1 is defined over every attempted case, so it
    # is computed separately from the completed-only accuracy figures above.
    macro_detail = arm["macro_f1_detail"]
    if scoring_policy == "v3":
        macro_detail = _macro_f1_v3(
            results, verdict_attr=verdict_attr, errored_attr=errored_attr,
            status_attr=status_attr,
        )
    return {
        "n_completed": arm["n_completed"],
        "n_failed": sum(breakdown.values()),
        "failure_breakdown": breakdown,
        "n_mixed_verdicts": sum(
            1 for r in results if getattr(r, verdict_attr) == "mixed"
        ),
        "completion_rate": round(len(completed) / n, 4) if n else None,
        "accuracy_on_completed": arm["accuracy_on_completed"],
        "accuracy": arm["accuracy_on_completed"],  # alias, same value
        "end_to_end_success_rate": round(arm["n_correct"] / n, 4) if n else None,
        "macro_f1": macro_detail["macro_f1"],
        "macro_f1_exact": macro_detail.get("macro_f1_exact"),
        "macro_f1_detail": macro_detail,
    }


def _compact_arm_block(block: dict[str, Any]) -> dict[str, Any]:
    """The same block without the per-class detail, for a secondary arm."""
    return {k: v for k, v in block.items() if k != "macro_f1_detail"}


def _compute_metrics(
    results: list[CaseResult], *, scoring_policy: str = DEFAULT_SCORING_POLICY
) -> dict[str, Any]:
    if scoring_policy not in SCORING_POLICIES:
        raise ValueError(f"unknown scoring policy {scoring_policy!r}")
    matches = _matches_label_for(scoring_policy)
    n = len(results)
    if n == 0:
        # A run that evaluated nothing is a real outcome — a spending cap can stop
        # before the first case.  Returning the same shape keeps the report
        # readable instead of handing downstream consumers an empty object.
        return {
            "n_cases": 0,
            "rule_based": {
                "n_valid": 0, "n_errors": 0, "n_failed": 0, "n_completed": 0,
                "completion_rate": None,
                "end_to_end_success_rate": None,
                "accuracy_on_completed": None, "accuracy": None,
                "macro_f1": None,
                "macro_f1_detail": {
                    "macro_f1": None, "n_classes_averaged": 0, "per_class": {},
                    "denominator": 0, "denominator_note": "no cases were evaluated",
                },
                "failure_breakdown": {"errored": 0},
                "n_mixed_verdicts": 0,
                "abstain_rate": None,
                "recall_at_1": None, "avg_latency_ms": None,
            },
            "llm_agent": {
                "n_valid": 0, "n_errors": 0, "n_completed": 0, "n_failed": 0,
                "failure_breakdown": {
                    "errored": 0, "budget_exhausted": 0, "text_exit": 0, "other": 0,
                },
                "n_mixed_verdicts": 0,
                "completion_rate": None,
                "accuracy_on_completed": None,
                "accuracy": None,
                "end_to_end_success_rate": None,
                "macro_f1": None,
                "macro_f1_detail": {
                    "macro_f1": None,
                    "n_classes_averaged": 0,
                    "per_class": {},
                    "denominator": 0,
                    "denominator_note": "no cases were evaluated",
                },
                "abstain_rate": None, "citation_hit_rate": None,
                "avg_latency_ms": None,
                "cost": {
                    "total_cost_usd_known": 0.0,
                    "n_cost_known": 0, "n_cost_unknown": 0,
                    "n_cost_unknown_before_failure": 0,
                    "known_subtotal_semantics": COST_ESTIMATE_SEMANTICS,
                    "cost_complete": True,
                    "usage_coverage_counts": {"confirmed": 0, "partial": 0, "unknown": 0},
                    "caveat": "no cases were evaluated, so no cost was incurred",
                },
                "avg_steps_used": None,
            },
            # No baseline arm was run, which is not the same as one that failed.
            "baseline": None,
            "arm_c": None,
            # The policy is reported even when nothing ran: a zero-case v3 run is
            # an incomplete v3 experiment, not an absent one.
            "scoring_policy": scoring_policy,
        }

    # Only evaluate on cases that completed without an exception.
    # Errors must be surfaced separately — never folded into the denominator or
    # treated as a predicted verdict, which would silently distort accuracy.
    #
    # "Completed" means the same thing for both arms: the arm produced a real
    # response and did not raise.  The LLM arm additionally requires an accepted
    # finish (run_status == "completed"), because text_exit and budget_exhausted
    # are operational failures and must not be counted as correct insufficient
    # predictions.  The rule arm has no decision loop, so not-errored is its bar.
    rb_completed = [r for r in results if not r.rule_errored]
    llm_valid = [r for r in results if not r.llm_errored]
    rb_n = len(rb_completed)
    llm_n = len(llm_valid)

    # Every arm goes through the same helpers, so the columns of the report are
    # comparable by construction rather than by inspection.
    llm_arm = _arm_block(
        results, verdict_attr="llm_verdict", errored_attr="llm_errored",
        status_attr="llm_run_status",
        correct_of=lambda r: matches(r.llm_verdict, r.gold_label), n=n,
        scoring_policy=scoring_policy,
    )
    # The rule arm has no decision loop, so not-errored is its whole completion bar.
    rb_arm = _arm_block(
        results, verdict_attr="rule_verdict", errored_attr="rule_errored",
        status_attr=None,
        correct_of=lambda r: matches(r.rule_verdict, r.gold_label), n=n,
        scoring_policy=scoring_policy,
    )

    # Arm B is reported only when one was actually run; an absent arm must not
    # appear as a column of nulls that could be misread as a failed run.
    baseline_present = any(r.baseline_run_status for r in results)
    baseline_arm: dict[str, Any] | None = None
    if baseline_present:
        baseline_arm = _arm_block(
            results, verdict_attr="baseline_verdict", errored_attr="baseline_errored",
            status_attr="baseline_run_status",
            correct_of=lambda r: matches(r.baseline_verdict, r.gold_label),
            n=n,
            scoring_policy=scoring_policy,
        )
        baseline_valid = [r for r in results if not r.baseline_errored]
        baseline_arm["n_valid"] = len(baseline_valid)
        baseline_arm["n_errors"] = n - len(baseline_valid)
        baseline_arm["abstain_rate"] = (
            round(sum(r.baseline_abstained for r in baseline_valid) / len(baseline_valid), 4)
            if baseline_valid else None
        )
        baseline_arm["citation_hit_rate"] = (
            round(sum(r.baseline_citation_hit for r in baseline_valid) / len(baseline_valid), 4)
            if baseline_valid else None
        )
        baseline_arm["avg_latency_ms"], baseline_arm["n_latency_recorded"] = _mean_or_none(
            [r.baseline_latency_ms for r in results]
        )
        baseline_arm["avg_steps_used"] = round(
            sum(r.baseline_steps_used for r in results) / n, 2
        )
        baseline_arm["cost"] = _cost_block(
            results, "baseline_cost_known_subtotal_usd", "baseline_usage_coverage"
        )

    # Arm C (v3/H3): the prompt-variant agent.  Built by the same helpers as A,
    # so any difference between the two columns is the prompt, not the scoring.
    arm_c_present = any(r.arm_c_run_status for r in results)
    arm_c_block: dict[str, Any] | None = None
    if arm_c_present:
        arm_c_block = _arm_block(
            results, verdict_attr="arm_c_verdict", errored_attr="arm_c_errored",
            status_attr="arm_c_run_status",
            correct_of=lambda r: matches(r.arm_c_verdict, r.gold_label),
            n=n, scoring_policy=scoring_policy,
        )
        arm_c_valid = [r for r in results if not r.arm_c_errored]
        arm_c_block["n_valid"] = len(arm_c_valid)
        arm_c_block["n_errors"] = n - len(arm_c_valid)
        arm_c_block["abstain_rate"] = (
            round(sum(r.arm_c_abstained for r in arm_c_valid) / len(arm_c_valid), 4)
            if arm_c_valid else None
        )
        arm_c_block["citation_hit_rate"] = (
            round(sum(r.arm_c_citation_hit for r in arm_c_valid) / len(arm_c_valid), 4)
            if arm_c_valid else None
        )
        arm_c_block["avg_latency_ms"], arm_c_block["n_latency_recorded"] = _mean_or_none(
            [r.arm_c_latency_ms for r in results]
        )
        arm_c_block["avg_steps_used"] = round(
            sum(r.arm_c_steps_used for r in results) / n, 2
        )
        arm_c_block["cost"] = _cost_block(
            results, "arm_c_cost_known_subtotal_usd", "arm_c_usage_coverage"
        )

    llm_cost = _cost_block(
        results, "llm_cost_known_subtotal_usd", "llm_usage_coverage"
    )
    rb_latency, rb_latency_n = _mean_or_none([r.rule_latency_ms for r in results])
    llm_latency, llm_latency_n = _mean_or_none([r.llm_latency_ms for r in results])

    return {
        "n_cases": n,
        # All three arms are built by the same helpers, so their numbers are
        # comparable by construction.  The baseline key is null when no baseline
        # arm was supplied, which is different from an arm that ran and failed.
        "rule_based": {
            "n_valid": rb_n,
            "n_errors": n - rb_n,
            **rb_arm,
            "abstain_rate": (
                round(sum(r.rule_abstained for r in rb_completed) / rb_n, 4)
                if rb_n else None
            ),
            "recall_at_1": round(sum(r.rule_recall_at_1 for r in results) / n, 4),
            "avg_latency_ms": rb_latency,
            "n_latency_recorded": rb_latency_n,
        },
        "llm_agent": {
            "n_valid": llm_n,
            "n_errors": n - llm_n,
            **llm_arm,
            "abstain_rate": (
                round(sum(r.llm_abstained for r in llm_valid) / llm_n, 4)
                if llm_n else None
            ),
            "citation_hit_rate": (
                round(sum(r.llm_citation_hit for r in llm_valid) / llm_n, 4)
                if llm_n else None
            ),
            "avg_latency_ms": llm_latency,
            "n_latency_recorded": llm_latency_n,
            "cost": llm_cost,
            "avg_steps_used": round(sum(r.llm_steps_used for r in results) / n, 2),
        },
        "baseline": baseline_arm,
        "arm_c": arm_c_block,
        # Recorded with the numbers so a reader can tell which mapping produced
        # them; v3 differs from v1/v2 and the two are never mixed in one report.
        "scoring_policy": scoring_policy,
    }


def _print_summary(metrics: dict[str, Any]) -> None:
    n = metrics["n_cases"]
    rb = metrics["rule_based"]
    llm = metrics["llm_agent"]
    base = metrics.get("baseline")

    def pct(value: Any) -> str:
        return f"{value:.1%}" if value is not None else "n/a"

    def num(value: Any, fmt: str = ".0f") -> str:
        return format(value, fmt) if value is not None else "n/a"

    def mean_or_na(block: dict[str, Any], key: str, recorded: str) -> str:
        if block.get(key) is None:
            return "n/a"
        count = block.get(recorded)
        suffix = "" if count is None or count == n else f" ({count})"
        return f"{block[key]:.0f}{suffix}"

    print(f"\n{'='*60}")
    print(f"  Pipeline Comparison — {n} cases")
    print(f"{'='*60}")
    arm_c = metrics.get("arm_c")
    headers = ["Metric", "Rule-Based", "Agent A"]
    if base is not None:
        headers.append("Base B")
    if arm_c is not None:
        headers.append("Arm C")
    width = 24 if len(headers) > 3 else 28
    col = 12
    print("".join(h.ljust(width) if i == 0 else h.rjust(col) for i, h in enumerate(headers)))
    print("-" * (width + col * (len(headers) - 1)))

    def row(label: str, *values: str) -> None:
        cells = "".join(v.rjust(col) for v in values)
        print(f"{label:<{width}}{cells}")

    def base_cells(fn: Any) -> list[str]:
        return [fn(base)] if base is not None else []

    def arm_c_cells(fn: Any) -> list[str]:
        return [fn(arm_c)] if arm_c is not None else []

    row("Errors / n_valid", f"{rb['n_errors']}/{rb['n_valid']}",
        f"{llm['n_errors']}/{llm['n_valid']}",
        *[f"{base['n_errors']}/{base['n_valid']}"] if base is not None else [],
        *arm_c_cells(lambda b: f"{b['n_errors']}/{b['n_valid']}"))
    row("Completion rate", pct(rb["completion_rate"]), pct(llm["completion_rate"]),
        *base_cells(lambda b: pct(b["completion_rate"])),
        *arm_c_cells(lambda b: pct(b["completion_rate"])))
    row("Accuracy on completed", pct(rb["accuracy_on_completed"]),
        pct(llm["accuracy_on_completed"]),
        *base_cells(lambda b: pct(b["accuracy_on_completed"])),
        *arm_c_cells(lambda b: pct(b["accuracy_on_completed"])))
    row("End-to-end success rate", pct(rb["end_to_end_success_rate"]),
        pct(llm["end_to_end_success_rate"]),
        *base_cells(lambda b: pct(b["end_to_end_success_rate"])),
        *arm_c_cells(lambda b: pct(b["end_to_end_success_rate"])))
    f1_label = ("Macro-F1 (v3, all attempted)" if metrics.get("scoring_policy") == "v3"
                else "Macro-F1 (completed only)")
    row(f1_label, pct(rb["macro_f1"]), pct(llm["macro_f1"]),
        *base_cells(lambda b: pct(b["macro_f1"])),
        *arm_c_cells(lambda b: pct(b["macro_f1"])))
    row("Abstain rate", pct(rb["abstain_rate"]), pct(llm["abstain_rate"]),
        *base_cells(lambda b: pct(b.get("abstain_rate"))),
        *arm_c_cells(lambda b: pct(b.get("abstain_rate"))))
    row("Citation hit rate", "n/a", pct(llm["citation_hit_rate"]),
        *base_cells(lambda b: pct(b.get("citation_hit_rate"))),
        *arm_c_cells(lambda b: pct(b.get("citation_hit_rate"))))
    mixed_label = ("Mixed verdicts (v3: scored as maybe)"
                   if metrics.get("scoring_policy") == "v3"
                   else "Mixed verdicts (unscorable)")
    row(mixed_label, str(rb["n_mixed_verdicts"]), str(llm["n_mixed_verdicts"]),
        *base_cells(lambda b: str(b["n_mixed_verdicts"])),
        *arm_c_cells(lambda b: str(b["n_mixed_verdicts"])))

    def failure_cell(block: dict[str, Any]) -> str:
        fb = block["failure_breakdown"]
        return (f"{fb.get('errored', 0)}/{fb.get('budget_exhausted', 0)}/"
                f"{fb.get('text_exit', 0)}/{fb.get('other', 0)}")

    row("Failures err/budg/text/oth", failure_cell(rb), failure_cell(llm),
        *base_cells(failure_cell), *arm_c_cells(failure_cell))

    rb_lat_n = rb.get("n_latency_recorded")
    llm_lat_n = llm.get("n_latency_recorded")
    base_lat_n = base.get("n_latency_recorded") if base is not None else None
    c_lat_n0 = arm_c.get("n_latency_recorded") if arm_c is not None else None
    latency_partial = any(
        c is not None and c < n for c in (rb_lat_n, llm_lat_n, base_lat_n, c_lat_n0)
    )
    # A merged report can hold cases whose segment never recorded a latency.  The
    # mean then covers only the recorded cases, and saying "avg latency" without
    # that qualifier would read as an average over the whole experiment.
    label = "Avg latency (recorded only)" if latency_partial else "Avg latency (ms)"
    row(label, mean_or_na(rb, "avg_latency_ms", "n_latency_recorded"),
        mean_or_na(llm, "avg_latency_ms", "n_latency_recorded"),
        *([] if base is None else [mean_or_na(base, "avg_latency_ms", "n_latency_recorded")]),
        *arm_c_cells(lambda b: mean_or_na(b, "avg_latency_ms", "n_latency_recorded")))
    if latency_partial:
        c_lat_n = arm_c.get("n_latency_recorded") if arm_c is not None else None
        row("  of cases",
            str(rb_lat_n) if rb_lat_n is not None else "n/a",
            str(llm_lat_n) if llm_lat_n is not None else "n/a",
            *([] if base is None else [
                str(base_lat_n) if base_lat_n is not None else "n/a"
            ]),
            *([] if arm_c is None else [
                str(c_lat_n) if c_lat_n is not None else "n/a"
            ]))

    def cost_cell(block: dict[str, Any]) -> str:
        cost = block.get("cost")
        if not cost:
            return "n/a"
        note = "" if cost["cost_complete"] else " (incomplete)"
        return f"{cost['total_cost_usd_known']:.6f}{note}"

    row("Cost estimate (USD)", "n/a", cost_cell(llm), *base_cells(cost_cell),
        *arm_c_cells(cost_cell))
    # The rule arm has no decision loop, so it has no step count — shown as a
    # dash rather than 0, which would read as "ran and used no steps".
    row("Avg steps used", "—", num(llm["avg_steps_used"], ".2f"),
        *base_cells(lambda b: num(b["avg_steps_used"], ".2f")),
        *arm_c_cells(lambda b: num(b["avg_steps_used"], ".2f")))
    print(f"{'='*60}\n")


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _benchmark_hashes(benchmark_dir: Path) -> dict[str, str]:
    """sha256 of every input the run reads, so a report is tied to its data."""
    names = (CORPUS_FILENAME, TRAIN_FILENAME, TEST_INPUTS_FILENAME, TEST_GOLD_FILENAME)
    return {
        name: _sha256_file(benchmark_dir / name)
        for name in names
        if (benchmark_dir / name).is_file()
    }


# ── Segment merging ──────────────────────────────────────────────────────────

#: Config fields that define the experiment.  A continuation or merge whose
#: values differ is not the same experiment, so it is refused rather than
#: silently combined.  Budget settings are deliberately absent: each segment may
#: legitimately carry its own cap, and requiring them to match would forbid the
#: very thing a carried-over budget is for.
CONFIG_IDENTITY_FIELDS = (
    "model",
    "base_url",
    "thinking",
    "temperature",
    "max_steps",
    "max_tokens_per_call",
    "request_timeout_seconds",
    "sdk_max_retries",
)

#: Per-case keys the report writer emits.  A stored record missing any of these
#: is reported as such; nothing is reconstructed by estimation.
_REPORT_CASE_FIELDS = (
    "case_id", "question", "gold_label", "gold_pmid",
    "rule_verdict", "rule_abstained", "rule_latency_ms", "rule_recall_at_1",
    "rule_error_flag", "rule_error_message",
    "llm_verdict", "llm_abstained", "llm_latency_ms", "llm_citation_hit",
    "llm_steps", "llm_steps_source", "llm_run_status",
    "llm_error_flag", "llm_error_message",
    "llm_cost_usd", "llm_cost_status", "llm_cost_known_subtotal_usd", "llm_audit",
    # Arm B, so a segmented v2 run can be merged without losing the baseline column.
    "baseline_verdict", "baseline_run_status", "baseline_steps",
    "baseline_steps_source", "baseline_citation_hit", "baseline_latency_ms",
    "baseline_error_flag", "baseline_error_message",
    "baseline_cost_usd", "baseline_cost_status", "baseline_cost_known_subtotal_usd",
    "baseline_audit",
    # Arm C and the v3 diagnostic material, so a segmented v3 run merges without
    # losing the second arm or the evidence needed to audit it later.
    "arm_c_verdict", "arm_c_run_status", "arm_c_steps", "arm_c_steps_source",
    "arm_c_citation_hit", "arm_c_latency_ms", "arm_c_error_flag",
    "arm_c_error_message", "arm_c_cost_usd", "arm_c_cost_status",
    "arm_c_cost_known_subtotal_usd", "arm_c_audit",
    "llm_answer", "llm_claim", "llm_decisive_reason", "llm_tool_returns",
    "arm_c_answer", "arm_c_claim", "arm_c_decisive_reason", "arm_c_tool_returns",
)

_ABSTAIN_VERDICT = "insufficient"

#: Arm B's output cap.  Matches the agent arm's per-call cap so neither arm is
#: advantaged by a different ceiling.
DEFAULT_BASELINE_MAX_TOKENS = 1024


def _values_differ(left: Any, right: Any) -> bool:
    """Numeric equality that does not trip on 30 vs 30.0; None never conflicts."""
    if left is None or right is None:
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) != float(right)
    return left != right


def _case_ids_sha256(case_ids: Any) -> str:
    """Order-sensitive checksum of the sample's case IDs.

    The order matters: the same 50 IDs in a different order is a different
    experiment, and a checksum over the joined sequence catches that where a set
    comparison would not.
    """
    return hashlib.sha256(
        "\n".join(str(case_id) for case_id in case_ids).encode("utf-8")
    ).hexdigest()


def _sample_identity(sample_record: dict[str, Any]) -> dict[str, Any]:
    """The four sample fields that define which cases were drawn, and how."""
    return {
        "seed": sample_record.get("seed"),
        "total": sample_record.get("total"),
        "case_ids_sha256": _case_ids_sha256(sample_record.get("case_ids") or []),
        "gold_sha256": sample_record.get("gold_sha256"),
    }


#: Identity sections introduced with Arm B.  A report predating them (the whole
#: v1 series) carries neither key, and two such reports are compatible by
#: definition; inventing a value for them would fabricate history.  Once either
#: side records one, both must agree.
_OPTIONAL_IDENTITY_SECTIONS = (
    "baseline_arm",
    "prompt_fingerprint",
    # Which verdict->label mapping produced the numbers.  Reports predating v3
    # carry no policy field and are implicitly legacy, so two of them agree;
    # once either side records one, both must agree.
    "scoring_policy",
    # Whether Arm C ran, and with exactly what prompt intervention.
    "arm_c",
)


def _experiment_identity(report: dict[str, Any]) -> dict[str, Any]:
    sample = report.get("sample") or {}
    config = report.get("config") or {}
    identity: dict[str, Any] = {
        "sample": {
            key: sample.get(key)
            for key in ("seed", "total", "case_ids_sha256", "gold_sha256")
        },
        "input_hashes": report.get("input_hashes") or {},
        "config": {key: config.get(key) for key in CONFIG_IDENTITY_FIELDS},
    }
    # Whether Arm B ran, and with what settings, is part of the experiment: a
    # continuation that silently changes top_k is not the same experiment.
    if "baseline_arm" in config:
        identity["baseline_arm"] = config["baseline_arm"]
    # Frozen prompt and tool hashes, so a later segment cannot change the prompts
    # and still be merged as if nothing had moved.
    if "prompt_fingerprint" in report:
        identity["prompt_fingerprint"] = report["prompt_fingerprint"]
    if "scoring_policy" in config:
        identity["scoring_policy"] = config["scoring_policy"]
    if "arm_c" in config:
        identity["arm_c"] = config["arm_c"]
    return identity


def _compare_identity(
    earlier: dict[str, Any], earlier_label: str,
    current: dict[str, Any], current_label: str,
) -> list[str]:
    """Return every way two experiment identities fail to match.

    A field that either side did not record is a **conflict**, not a match.  An
    identity that cannot be checked cannot be relied on, and treating silence as
    agreement is exactly how a continuation ends up running against a different
    sample than the one it claims to extend.
    """
    conflicts: list[str] = []

    def _check(section: str, key: str, left: Any, right: Any) -> None:
        if left is None or right is None:
            missing_on = earlier_label if left is None else current_label
            conflicts.append(
                f"{section}.{key}: not recorded on {missing_on} "
                f"({earlier_label}={left!r}, {current_label}={right!r})"
            )
        elif _values_differ(left, right):
            conflicts.append(
                f"{section}.{key}: {earlier_label} has {left!r}, "
                f"{current_label} has {right!r}"
            )

    for key, left in (earlier.get("sample") or {}).items():
        _check("sample", key, left, (current.get("sample") or {}).get(key))

    left_files = (earlier.get("input_hashes") or {}).get("files") or {}
    right_files = (current.get("input_hashes") or {}).get("files") or {}
    for name in sorted(set(left_files) | set(right_files)):
        _check("input_hashes.files", name, left_files.get(name), right_files.get(name))
    _check(
        "input_hashes", "sample_file_sha256",
        (earlier.get("input_hashes") or {}).get("sample_file_sha256"),
        (current.get("input_hashes") or {}).get("sample_file_sha256"),
    )

    for key, left in (earlier.get("config") or {}).items():
        _check("config", key, left, (current.get("config") or {}).get(key))

    # Arm B's configuration and the prompt/tool fingerprints.  Deep equality, so
    # a changed top_k or a changed prompt hash is caught; two reports that both
    # predate these fields are compatible rather than in conflict.
    for section in _OPTIONAL_IDENTITY_SECTIONS:
        if section not in earlier and section not in current:
            continue
        left, right = earlier.get(section), current.get(section)
        if left != right:
            conflicts.append(
                f"{section}: {earlier_label} has {left!r}, {current_label} has {right!r}"
            )

    return conflicts


def _load_segment_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("cases"), list):
        raise ValueError(f"{path}: not a comparison report (no cases array)")
    return report


def _case_result_from_report_case(
    case: dict[str, Any]
) -> tuple[CaseResult, list[str], list[str]]:
    """Rebuild a CaseResult from a stored case record for metric recomputation.

    Returns the result, the fields that were absent and left absent, and the
    fields that were absent but are exactly determined by one the record did
    carry.  The two are kept apart on purpose: ``abstained`` is defined by the
    product contract as ``verdict == "insufficient"``, so recomputing it invents
    nothing, whereas a latency that was never recorded is simply unknown and
    stays None rather than being filled with an average.
    """
    absent: list[str] = []
    derived: list[str] = []
    rule_verdict = str(case.get("rule_verdict") or "")
    llm_verdict = str(case.get("llm_verdict") or "")

    def _value(name: str, fallback: Any = None) -> Any:
        value = case.get(name)
        if value is not None:
            return value
        if fallback is not None:
            derived.append(name)
            return fallback
        absent.append(name)
        return None

    result = CaseResult(
        case_id=str(case.get("case_id") or ""),
        question=str(case.get("question") or ""),
        gold_label=str(case.get("gold_label") or ""),
        gold_pmid=str(case.get("gold_pmid") or ""),
        rule_verdict=rule_verdict,
        rule_abstained=bool(_value("rule_abstained", rule_verdict == _ABSTAIN_VERDICT)),
        rule_errored=bool(_value("rule_error_flag", False)),
        rule_error_msg=case.get("rule_error_message"),
        rule_recall_at_1=bool(_value("rule_recall_at_1", False)),
        rule_latency_ms=_value("rule_latency_ms"),
        llm_verdict=llm_verdict,
        llm_abstained=bool(_value("llm_abstained", llm_verdict == _ABSTAIN_VERDICT)),
        llm_errored=bool(_value("llm_error_flag", False)),
        llm_error_msg=case.get("llm_error_message"),
        llm_citation_hit=bool(_value("llm_citation_hit", False)),
        llm_latency_ms=_value("llm_latency_ms"),
        llm_steps_used=int(_value("llm_steps") or 0),
        llm_run_status=str(case.get("llm_run_status") or ""),
        llm_cost_usd=case.get("llm_cost_usd"),
        llm_usage_coverage=str(case.get("llm_cost_status") or "unknown"),
        llm_cost_known_subtotal_usd=case.get("llm_cost_known_subtotal_usd"),
        llm_audit=case.get("llm_audit"),
        llm_steps_source=str(case.get("llm_steps_source") or "unknown"),
        # Arm B.  Rebuilt the same way, so a merged v2 report keeps the baseline
        # column instead of silently reporting it as "not run".
        baseline_verdict=str(case.get("baseline_verdict") or ""),
        baseline_abstained=bool(_value("baseline_abstained",
                                       str(case.get("baseline_verdict") or "") == _ABSTAIN_VERDICT)),
        baseline_errored=bool(_value("baseline_error_flag", False)),
        baseline_error_msg=case.get("baseline_error_message"),
        baseline_citation_hit=bool(_value("baseline_citation_hit", False)),
        baseline_latency_ms=_value("baseline_latency_ms"),
        baseline_steps_used=int(_value("baseline_steps") or 0),
        baseline_steps_source=str(case.get("baseline_steps_source") or "unknown"),
        baseline_run_status=str(case.get("baseline_run_status") or ""),
        baseline_cost_usd=case.get("baseline_cost_usd"),
        baseline_usage_coverage=str(case.get("baseline_cost_status") or "unknown"),
        baseline_cost_known_subtotal_usd=case.get("baseline_cost_known_subtotal_usd"),
        baseline_audit=case.get("baseline_audit"),
        # Arm C and diagnostics, rebuilt the same way.
        arm_c_verdict=str(case.get("arm_c_verdict") or ""),
        arm_c_abstained=bool(_value("arm_c_abstained",
                                    str(case.get("arm_c_verdict") or "") == _ABSTAIN_VERDICT)),
        arm_c_errored=bool(_value("arm_c_error_flag", False)),
        arm_c_error_msg=case.get("arm_c_error_message"),
        arm_c_citation_hit=bool(_value("arm_c_citation_hit", False)),
        arm_c_latency_ms=_value("arm_c_latency_ms"),
        arm_c_steps_used=int(_value("arm_c_steps") or 0),
        arm_c_steps_source=str(case.get("arm_c_steps_source") or "unknown"),
        arm_c_run_status=str(case.get("arm_c_run_status") or ""),
        arm_c_cost_usd=case.get("arm_c_cost_usd"),
        arm_c_usage_coverage=str(case.get("arm_c_cost_status") or "unknown"),
        arm_c_cost_known_subtotal_usd=case.get("arm_c_cost_known_subtotal_usd"),
        arm_c_audit=case.get("arm_c_audit"),
        llm_answer=str(case.get("llm_answer") or ""),
        llm_claim=str(case.get("llm_claim") or ""),
        llm_decisive_reason=str(case.get("llm_decisive_reason") or ""),
        llm_tool_returns=case.get("llm_tool_returns"),
        arm_c_answer=str(case.get("arm_c_answer") or ""),
        arm_c_claim=str(case.get("arm_c_claim") or ""),
        arm_c_decisive_reason=str(case.get("arm_c_decisive_reason") or ""),
        arm_c_tool_returns=case.get("arm_c_tool_returns"),
    )
    # A field the writer never emitted at all is absent too, even when a default
    # above happened to cover it.  On a v2 report Arm B was genuinely run, so its
    # fields are present; on a v1 report they were never written and stay absent.
    for name in _REPORT_CASE_FIELDS:
        if name not in case and name not in absent and name not in derived:
            absent.append(name)
    return result, sorted(set(absent)), sorted(set(derived))


def _collect_segment_cases(
    segments: list[tuple[Path, dict[str, Any]]],
    sample_record: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Gather stored cases in frozen-sample order, rejecting duplicates/strangers."""
    frozen = [str(case_id) for case_id in sample_record["case_ids"]]
    frozen_set = set(frozen)
    label_by_case = sample_record.get("label_by_case") or {}

    by_case: dict[str, dict[str, Any]] = {}
    for path, report in segments:
        seen_in_segment: set[str] = set()
        for case in report["cases"]:
            case_id = str(case.get("case_id") or "")
            if not case_id:
                raise ValueError(f"{path}: a case record has no case_id")
            if case_id in seen_in_segment:
                raise ValueError(f"{path}: duplicate case_id {case_id} within one segment")
            seen_in_segment.add(case_id)
            if case_id in by_case:
                raise ValueError(
                    f"duplicate case_id {case_id} across segments — refusing to merge, "
                    "which would count the same question twice"
                )
            if case_id not in frozen_set:
                raise ValueError(f"{path}: case_id {case_id} is not in the frozen sample")
            expected_label = label_by_case.get(case_id)
            if expected_label and str(case.get("gold_label")) != str(expected_label):
                raise ValueError(
                    f"{path}: {case_id} carries gold {case.get('gold_label')!r} but the "
                    f"frozen sample says {expected_label!r}"
                )
            by_case[case_id] = case

    ordered = [by_case[case_id] for case_id in frozen if case_id in by_case]
    not_covered = [case_id for case_id in frozen if case_id not in by_case]
    return ordered, not_covered


def _merge_segment_reports(
    segment_paths: list[Path],
    sample_record: dict[str, Any],
    sample_path: Path | None,
) -> dict[str, Any]:
    """Combine segment reports into one, recomputing every aggregate from cases.

    Metrics are never averaged across segments: they are recomputed from the
    concatenated case records, so a 5-case segment and a 45-case one cannot be
    weighted equally by accident.
    """
    if not segment_paths:
        raise ValueError("no segment reports supplied")
    segments = [(path, _load_segment_report(path)) for path in segment_paths]

    # Every segment must have been drawn from the sample file actually passed in,
    # not merely from a sample file with the same name.  Checked before the
    # segments are compared with each other, so a shared mistake cannot pass as
    # agreement.
    #
    # Only the sample identity and the sample file's own hash are in scope here:
    # a sample file does not define the run configuration, so comparing config
    # against it would report every field as unrecorded.
    sample_file_sha256 = _sha256_file(sample_path) if sample_path else None
    actual = {
        "sample": _sample_identity(sample_record),
        "input_hashes": {"sample_file_sha256": sample_file_sha256},
    }
    for path, report in segments:
        mine = _experiment_identity(report)
        comparable = {
            "sample": mine["sample"],
            "input_hashes": {
                "sample_file_sha256": (mine["input_hashes"] or {}).get(
                    "sample_file_sha256"
                )
            },
        }
        conflicts = _compare_identity(
            comparable, path.name, actual, "the supplied --sample-file"
        )
        if conflicts:
            raise ValueError(
                f"{path.name} was not produced from the supplied --sample-file: "
                + "; ".join(conflicts)
            )

    base_path, base = segments[0]
    identity = _experiment_identity(base)
    for path, report in segments[1:]:
        conflicts = _compare_identity(
            identity, base_path.name, _experiment_identity(report), path.name
        )
        if conflicts:
            raise ValueError(
                f"{path.name} does not describe the same experiment as {base_path.name}: "
                + "; ".join(conflicts)
            )

    ordered, not_covered = _collect_segment_cases(segments, sample_record)

    results: list[CaseResult] = []
    cases_with_absent_fields: dict[str, list[str]] = {}
    cases_with_derived_fields: dict[str, list[str]] = {}
    for case in ordered:
        result, absent, derived = _case_result_from_report_case(case)
        results.append(result)
        if absent:
            cases_with_absent_fields[result.case_id] = absent
        if derived:
            cases_with_derived_fields[result.case_id] = derived

    # All segments already agreed on the policy during the identity check.
    merged_policy = (base.get("config") or {}).get("scoring_policy") or DEFAULT_SCORING_POLICY
    metrics = _compute_metrics(results, scoring_policy=merged_policy)
    attempted = len(results)
    planned = len(sample_record["case_ids"])
    not_run = planned - attempted

    segment_summaries = []
    for path, report in segments:
        plan = report.get("plan") or {}
        # Every paid arm this segment ran, so the per-segment figure matches the
        # experiment-level one rather than covering the agent arm alone.
        subtotal = _report_known_subtotal(report)
        config = report.get("config") or {}
        segment_summaries.append({
            "file": path.name,
            "run_date": report.get("run_date"),
            "planned": plan.get("planned"),
            "attempted": plan.get("attempted"),
            "partial": plan.get("partial"),
            "stop_reason": plan.get("stop_reason"),
            "known_subtotal_usd": round(subtotal, 6),
            # Recorded per segment, because each may set its own cap.
            "max_cost_usd": config.get("max_cost_usd"),
            "experiment_budget_usd": config.get("experiment_budget_usd"),
            "carry_over_cost_usd": config.get("carry_over_cost_usd"),
        })

    # Summed from the merged cases' own per-arm subtotals — not from the
    # segments' carry-over figures, which would count earlier segments twice.
    total_known = sum(_case_known_subtotal(case) for case in ordered)

    return {
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "merged": True,
        "dry_run": False,
        "model": base.get("model"),
        "config": base.get("config"),
        "sample": base.get("sample"),
        "input_hashes": base.get("input_hashes"),
        # Carried through from the segments, which agreed on it during the
        # identity check; dropping it would leave the merged report unverifiable.
        "prompt_fingerprint": base.get("prompt_fingerprint"),
        "segments": segment_summaries,
        "plan": {
            "planned": planned,
            "attempted": attempted,
            "completed": metrics["llm_agent"]["n_completed"],
            "failed": metrics["llm_agent"]["n_failed"],
            "not_run": not_run,
            "partial": not_run > 0,
            "stop_reason": (
                f"{not_run} frozen case(s) have no segment report yet"
                if not_run > 0 else None
            ),
            "not_covered_case_ids": not_covered,
        },
        "metrics_notes": {
            "metrics_recomputed_from_cases": True,
            "avg_latency_ms_coverage": (
                "averaged over the "
                f"{metrics['llm_agent'].get('n_latency_recorded')} case(s) whose latency "
                "was recorded — this is NOT an average over all "
                f"{planned} frozen case(s)"
                if (metrics["llm_agent"].get("n_latency_recorded") or 0) < attempted
                else f"averaged over all {attempted} attempted case(s)"
            ),
            "cost_basis": (
                "the known subtotal sums only cases whose usage was reported; it is "
                "an estimate at peak cache-miss rates, not a bill, and not a bound "
                "on the true charge when any case reported nothing"
            ),
        },
        # Recomputed from the merged per-case records, the frozen sample and the
        # policy — never copied or averaged from the segments.  A merge that
        # covers the whole frozen sample can carry a conclusion; one that does
        # not must leave criteria_met null.
        "acceptance": _v3_acceptance(
            results, metrics, frozen_total=planned,
            arm_c_enabled=(base.get("config") or {}).get("arm_c") is not None,
        ),
        "merge_notes": {
            "metrics_recomputed_from_cases": True,
            "segments_combined": [path.name for path in segment_paths],
            "known_subtotal_usd": round(total_known, 6),
            "identity_check": (
                "every sample, input-hash and config field was compared across "
                "segments; a field either side did not record is treated as a "
                "conflict, not as agreement"
            ),
            "cases_with_absent_fields": cases_with_absent_fields,
            "cases_with_derived_fields": cases_with_derived_fields,
            "absent_field_policy": (
                "fields a stored case record did not carry are left null and listed "
                "in cases_with_absent_fields; none is filled in from a segment or "
                "experiment average"
            ),
            "derived_field_policy": (
                "cases_with_derived_fields lists fields that were not stored but are "
                "exactly determined by one that was — abstained is defined by the "
                "product contract as verdict == \"insufficient\" — so recomputing them "
                "adds no information the report did not already carry"
            ),
        },
        "metrics": metrics,
        "cases": ordered,
    }


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _prompt_fingerprint(
    llm_agent: Any, baseline_agent: Any | None, arm_c_agent: Any | None = None
) -> dict[str, Any]:
    """Hashes of the prompt material each arm sends, frozen with the results.

    Recorded so a later run can be shown to have used the same prompts — or
    shown not to have.  Historical v1 reports recorded none of this, so no
    byte-level claim about them is possible and none is made.
    """
    from bioevidence.llm_agent import TOOL_SCHEMAS, _system_prompt

    system = _system_prompt()
    fingerprint: dict[str, Any] = {
        "agent": {
            "system_prompt_sha256": _canonical_sha256(system),
            "tool_schemas_sha256": _canonical_sha256(TOOL_SCHEMAS),
            "system_prompt_chars": len(system),
            "tool_names": [t["function"]["name"] for t in TOOL_SCHEMAS],
        }
    }
    if baseline_agent is not None:
        template = baseline_agent.prompt_template()
        fingerprint["baseline"] = {
            "system_prompt_sha256": _canonical_sha256(FIXED_CONTEXT_SYSTEM_PROMPT),
            "user_template_sha256": _canonical_sha256({
                "header": template["user_header"],
                "context_header": template["user_context_header"],
                "footer": template["user_footer"],
            }),
            "system_prompt_chars": len(FIXED_CONTEXT_SYSTEM_PROMPT),
            "top_k": template["top_k"],
            "response_format": template["response_format"],
            "tool_names": [],
        }
    if arm_c_agent is not None:
        c_prompt = getattr(arm_c_agent, "_system_prompt", None) or arm_c_system_prompt()
        c_tools = getattr(arm_c_agent, "_tool_schemas", None) or arm_c_tool_schemas()
        fingerprint["arm_c"] = {
            "system_prompt_sha256": _canonical_sha256(c_prompt),
            "tool_schemas_sha256": _canonical_sha256(c_tools),
            "system_prompt_chars": len(c_prompt),
            "tool_names": [s["function"]["name"] for s in c_tools],
            "intervention": arm_c_intervention_diff(),
        }
    fingerprint["historical_v1"] = (
        "not recorded — the v1 reports carry no prompt or tool-schema hash, so no "
        "byte-level comparison against them is possible"
    )
    return fingerprint


def _effective_config(
    args: argparse.Namespace,
    llm_agent: Any,
    baseline_agent: Any | None = None,
    arm_c_agent: Any | None = None,
) -> dict[str, Any]:
    """The run configuration, built once and used for both the report and the
    cross-segment identity check, so the two can never drift apart."""
    return {
        "model": args.model if not args.dry_run else "mock",
        # Read from the agent that actually ran, so the recorded value cannot
        # drift from the one in effect.
        "base_url": (
            getattr(llm_agent, "_base_url", None) or "not used (mock agent, no HTTP)"
        ),
        "thinking": "disabled (extra_body={'thinking': {'type': 'disabled'}})",
        "temperature": 0.0,
        "max_steps": args.max_steps,
        "max_tokens_per_call": getattr(llm_agent, "_max_tokens_per_call", None),
        "request_timeout_seconds": args.timeout,
        "sdk_max_retries": args.max_retries,
        "max_cost_usd": args.max_cost_usd,
        "experiment_budget_usd": args.experiment_budget_usd,
        "scoring_policy": args.scoring_policy,
        "carry_over_cost_usd": getattr(args, "_carry_over_cost_usd", 0.0),
        "budget_enforcement": (
            "programmatic soft threshold checked before each case starts; it "
            "cannot bound a provider bill and one in-flight case can overshoot "
            "by its own cost"
        ) if (args.max_cost_usd is not None or args.experiment_budget_usd is not None)
        else None,
        "credential_source": (
            "not used (dry run)" if args.dry_run
            else "--deepseek-key or DEEPSEEK_API_KEY environment variable"
        ),
        # Arm B's own configuration, recorded honestly: it is a single-request
        # arm, so "max_steps" would be misleading and is recorded as such.
        "arm_c": (
            {
                "arm": "prompt_variant_agent",
                "model": arm_c_agent._model,
                "base_url": arm_c_agent._base_url,
                "max_steps": args.max_steps,
                "temperature": 0.0,
                "max_tokens_per_call": arm_c_agent._max_tokens_per_call,
                "request_timeout_seconds": arm_c_agent._timeout,
                "sdk_max_retries": arm_c_agent._max_retries,
                "intervention": arm_c_intervention_diff(),
                "note": (
                    "C is an experimental configuration; the product definition of "
                    "insufficient and the contract coupling are unchanged"
                ),
            }
            if arm_c_agent is not None
            else None
        ),
        "baseline_arm": (
            {
                "arm": "fixed_context_llm",
                "model": baseline_agent._model,
                "base_url": baseline_agent._base_url,
                "top_k": baseline_agent._top_k,
                "temperature": 0.0,
                "max_tokens_per_call": baseline_agent._max_tokens_per_call,
                "request_timeout_seconds": baseline_agent._timeout,
                "sdk_max_retries": baseline_agent._max_retries,
                "requests_per_case": 1,
                "tool_loop": False,
                "response_format": {"type": "json_object"},
                "retrieval": "one BM25 search on the original question, same index as Arm A",
            }
            if baseline_agent is not None
            else None
        ),
    }


def _verify_frozen_sample(
    sample_record: dict[str, Any], benchmark_dir: Path
) -> dict[str, Any]:
    """Check the frozen record still describes the data on disk.

    Runs before the first model request.  A mismatch means the run would not be
    the experiment that was declared, so it fails instead of adapting — the
    frozen sample is never rewritten to match whatever happens to be on disk.
    """
    gold_path = benchmark_dir / TEST_GOLD_FILENAME
    rows = {str(r["case_id"]): str(r["label"]) for r in read_jsonl(gold_path)}
    observed = _sha256_file(gold_path)
    expected = str(sample_record.get("gold_sha256") or "")

    if expected and observed != expected:
        raise ValueError(
            "frozen sample was drawn from a different gold file: "
            f"record says {expected}, {gold_path.name} hashes to {observed}. "
            "Refusing to run — regenerate the sample deliberately instead of "
            "letting the run silently evaluate a different set."
        )

    case_ids = [str(case_id) for case_id in sample_record["case_ids"]]
    duplicates = sorted({c for c in case_ids if case_ids.count(c) > 1})
    if duplicates:
        raise ValueError(f"frozen sample contains duplicate case IDs: {duplicates[:5]}")

    missing = [c for c in case_ids if c not in rows]
    if missing:
        raise ValueError(
            f"frozen sample references case IDs absent from {gold_path.name}: {missing[:5]}"
        )

    label_by_case = sample_record.get("label_by_case") or {}
    mismatched = [
        case_id for case_id in case_ids
        if case_id in label_by_case and str(label_by_case[case_id]) != rows[case_id]
    ]
    if mismatched:
        raise ValueError(
            "frozen sample labels disagree with the gold file for: "
            f"{mismatched[:5]}"
        )

    return {
        "gold_sha256_expected": expected or None,
        "gold_sha256_observed": observed,
        "hash_matches": (not expected) or observed == expected,
        "n_case_ids": len(case_ids),
        "n_duplicate_ids": 0,
        "n_label_mismatches": 0,
        "verified_before_model_requests": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Use tiny fixture corpus; mock LLM loop (no API key needed)")
    parser.add_argument("--benchmark-dir", type=Path,
                        help="Path to prepared PubMedQA benchmark directory")
    parser.add_argument("--deepseek-key", default=os.environ.get("DEEPSEEK_API_KEY"),
                        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--model", default="deepseek-flash",
                        help="DeepSeek model ID (default: deepseek-flash — the "
                             "'deepseek-chat' alias may be rerouted without notice)")
    parser.add_argument("--n-cases", type=int, default=None,
                        help="Limit to first N cases (default: all)")
    parser.add_argument("--max-steps", type=int, default=8,
                        help="Max ReAct steps per question (default: 8)")
    parser.add_argument("--sample-file", type=Path, default=None,
                        help="Frozen sample JSON; restricts the run to exactly the "
                             "case IDs it lists, in the order it lists them")
    parser.add_argument("--max-cost-usd", type=float, default=None,
                        help="Stop before starting a case once the estimated spend "
                             "reaches this cap; the report is marked partial")
    parser.add_argument("--experiment-budget-usd", type=float, default=None,
                        help="Cap for the WHOLE experiment, including spend already "
                             "known from --exclude-from segments")
    parser.add_argument("--exclude-from", type=Path, action="append", default=None,
                        metavar="REPORT",
                        help="Earlier segment report; its case IDs are dropped from "
                             "the frozen sample and its known spend is carried over. "
                             "Repeatable.")
    parser.add_argument("--scoring-policy", choices=SCORING_POLICIES,
                        default=DEFAULT_SCORING_POLICY,
                        help="Verdict->label mapping used for scoring (default: "
                             "legacy = the v1/v2 mapping; v3 = the H3 A/C mapping "
                             "where mixed also counts as a maybe prediction)")
    parser.add_argument("--arm-c", action="store_true",
                        help="Also run Arm C: the same agent with the mixed-usage "
                             "prompt clarification (v3/H3). A is unchanged.")
    parser.add_argument("--baseline-arm", action="store_true",
                        help="Also run Arm B: the fixed-context LLM baseline "
                             "(same model and index, one static top-k retrieval, "
                             "one request, no tool loop)")
    parser.add_argument("--baseline-top-k", type=int, default=None,
                        help="Records placed in Arm B's context block (default: "
                             "the v1 median of 4 distinct fetched records per case)")
    parser.add_argument("--merge-from", type=Path, action="append", default=None,
                        metavar="REPORT",
                        help="Merge segment reports instead of running: cases are "
                             "restored to frozen order and every metric is "
                             "recomputed from the cases. Repeatable.")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Per-request timeout in seconds (default: openai SDK default)")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="SDK auto-retries; pass 0 for a costed run so a failing "
                             "call bills one request instead of up to three")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON report to this path")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if args.merge_from:
        # Merge mode: no corpus, no agent, no model request.  Segments are
        # validated against each other, restored to frozen order, and every
        # aggregate is recomputed from the stored cases.
        if not args.sample_file:
            parser.error("--merge-from requires --sample-file")
        sample_record = json.loads(args.sample_file.read_text(encoding="utf-8"))
        merged = _merge_segment_reports(args.merge_from, sample_record, args.sample_file)
        _print_summary(merged["metrics"])
        output_path = args.output or (
            REPO_ROOT / "reports" / "pipeline_comparison_merged.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        print(f"Merged report saved to: {output_path}")
        plan = merged["plan"]
        if plan["not_run"]:
            print(
                f"INCOMPLETE COVERAGE: {plan['not_run']} of {plan['planned']} frozen "
                "case(s) have no segment report yet"
            )
            return 3
        return 0

    if args.dry_run:
        print("DRY RUN — using fixture corpus + mock LLM agent")
        docs, cases = _load_dry_run_data()
    elif args.benchmark_dir:
        print(f"Loading benchmark from: {args.benchmark_dir}")
        # Load the full case list; --n-cases is applied below so it can be
        # resolved against the frozen sample rather than against the corpus.
        docs, cases = _load_benchmark_data(args.benchmark_dir, None)
    else:
        parser.error("Specify --dry-run or --benchmark-dir")

    sample_record: dict[str, Any] | None = None
    sample_verification: dict[str, Any] | None = None
    if args.sample_file:
        sample_record = json.loads(args.sample_file.read_text(encoding="utf-8"))
        if args.benchmark_dir:
            # Before any model request: a sample drawn from a different gold file,
            # or one carrying a duplicate/contradictory ID, would make this a
            # different experiment than the one declared.
            sample_verification = _verify_frozen_sample(sample_record, args.benchmark_dir)
            print(
                f"Frozen sample verified against {TEST_GOLD_FILENAME}: "
                f"sha256 {'matches' if sample_verification['hash_matches'] else 'NOT RECORDED'}, "
                f"{sample_verification['n_case_ids']} unique case IDs"
            )
        wanted = [str(case_id) for case_id in sample_record["case_ids"]]
        cases = select_cases(cases, wanted)
        print(
            f"Frozen sample: {args.sample_file} "
            f"({len(cases)} cases, seed={sample_record.get('seed')})"
        )

    print(f"Corpus: {len(docs)} docs | sample cases: {len(cases)}")

    # Build shared retriever and tools
    retriever = BM25Retriever(docs)
    lit_tools = LiteratureTools(docs, retriever=retriever)

    if args.dry_run:
        llm_agent = MockLLMEvidenceAgent(lit_tools)
    else:
        if not args.deepseek_key:
            parser.error("--deepseek-key or DEEPSEEK_API_KEY required for real run")
        llm_agent = LLMEvidenceAgent(
            lit_tools,
            api_key=args.deepseek_key,
            model=args.model,
            max_steps=args.max_steps,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )

    arm_c_agent = None
    if args.arm_c:
        if args.dry_run:
            parser.error("--arm-c requires a real run, not --dry-run")
        if not args.deepseek_key:
            parser.error("--deepseek-key or DEEPSEEK_API_KEY required for a real run")
        # Identical non-prompt settings to A.  max_tokens_per_call is left at the
        # class default precisely because A does the same.
        arm_c_agent = LLMEvidenceAgent(
            lit_tools,
            api_key=args.deepseek_key,
            model=args.model,
            max_steps=args.max_steps,
            timeout=args.timeout,
            max_retries=args.max_retries,
            system_prompt=arm_c_system_prompt(),
            tool_schemas=arm_c_tool_schemas(),
        )

    baseline_agent = None
    if args.baseline_arm:
        if args.dry_run:
            parser.error("--baseline-arm requires a real run, not --dry-run")
        if not args.deepseek_key:
            parser.error("--deepseek-key or DEEPSEEK_API_KEY required for a real run")
        baseline_agent = FixedContextLLMAgent(
            lit_tools,
            api_key=args.deepseek_key,
            model=args.model,
            top_k=args.baseline_top_k or BASELINE_TOP_K,
            max_tokens_per_call=DEFAULT_BASELINE_MAX_TOKENS,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )

    # ── continuation: drop cases an earlier segment already attempted ────────
    # Done before any model request, and after the agent exists only so the
    # identity check can compare against the config this run will actually use.
    carry_over_cost_usd = 0.0
    exclusion: dict[str, Any] | None = None
    args._carry_over_cost_usd = 0.0
    if args.exclude_from:
        if sample_record is None:
            parser.error("--exclude-from requires --sample-file")
        config_block = _effective_config(args, llm_agent, baseline_agent, arm_c_agent)
        # The identity of the sample file actually passed in, not merely of a file
        # with the same name: seed, total, the order-sensitive checksum of the
        # case IDs, the gold hash, and the file's own sha256.
        # Built through the same helper the reports use, so the fields compared
        # are exactly the fields recorded — including whether Arm B ran, its
        # settings, and the prompt/tool fingerprints.
        candidates: dict[str, Any] = _experiment_identity({
            # _sample_identity produces the same four keys a report's sample
            # block carries, including the computed case-IDs checksum.
            "sample": _sample_identity(sample_record),
            "input_hashes": {
                "files": (
                    _benchmark_hashes(args.benchmark_dir) if args.benchmark_dir else {}
                ),
                "sample_file_sha256": _sha256_file(args.sample_file),
            },
            "config": config_block,
            "prompt_fingerprint": _prompt_fingerprint(llm_agent, baseline_agent, arm_c_agent),
        })

        attempted: list[str] = []
        for prior_path in args.exclude_from:
            prior = _load_segment_report(prior_path)
            conflicts = _compare_identity(
                _experiment_identity(prior), prior_path.name, candidates, "this run"
            )
            if conflicts:
                raise ValueError(
                    f"{prior_path.name} does not describe the same experiment as this "
                    "run, so it cannot be continued: " + "; ".join(conflicts)
                )
            for case in prior["cases"]:
                attempted.append(str(case.get("case_id") or ""))
            # Every paid arm counts, and the carry-over is the sum of per-case
            # known subtotals — never a previously carried-over figure, so
            # chained segments cannot count the same spend twice.
            carry_over_cost_usd += _report_known_subtotal(prior)

        blocked = set(attempted)
        cases = [case for case in cases if case["case_id"] not in blocked]
        exclusion = {
            "excluded_case_ids": sorted(blocked),
            "n_excluded": len(blocked),
            "carry_over_cost_usd": round(carry_over_cost_usd, 6),
            "identity_verified_before_model_requests": True,
        }
        args._carry_over_cost_usd = round(carry_over_cost_usd, 6)
        print(
            f"Continuation: {len(blocked)} case(s) excluded from earlier segment(s), "
            f"{len(cases)} remaining; known spend carried over ${carry_over_cost_usd:.6f}"
        )
        if not cases:
            print("Nothing left to run — every frozen case already has a segment report.")
            return 0

    # Applied last, so --n-cases means "the first N of what is actually left to
    # run".  Slicing earlier would take the first N of the frozen sample, which
    # for a continuation is exactly the cases an earlier segment already did.
    if args.n_cases:
        cases = cases[:args.n_cases]

    planned = len(cases)
    print(f"Cases to run: {planned}")

    print("\nRunning comparison...")
    stop_state: dict[str, Any] = {}
    # One cap applies at a time.  An experiment budget supersedes a segment cap
    # because it already accounts for what earlier segments spent.
    effective_cap = (
        args.experiment_budget_usd if args.experiment_budget_usd is not None
        else args.max_cost_usd
    )
    results = run_comparison(
        docs=docs,
        cases=cases,
        llm_agent=llm_agent,
        baseline_agent=baseline_agent,
        arm_c_agent=arm_c_agent,
        benchmark_dir=args.benchmark_dir if not args.dry_run else None,
        verbose=args.verbose,
        max_cost_usd=effective_cap,
        carry_over_cost_usd=carry_over_cost_usd,
        stop_state=stop_state,
    )
    attempted = len(results)
    # Cases never started are "not run" — a different fact from "ran and failed".
    # The gap is reported whenever it exists, but the reason distinguishes a
    # deliberate spending stop from any other early exit, so a cap is never
    # blamed for a shortfall it did not cause.
    not_run = planned - attempted
    if not_run <= 0:
        stop_reason = None
    elif stop_state.get("reason") == "cost_threshold":
        stop_reason = (
            "experiment budget already exhausted by earlier segments before any "
            "new case was started"
            if attempted == 0
            else "max_cost_usd reached before the next case started"
        )
    elif stop_state.get("reason") == "incomplete_usage":
        stop_reason = (
            "usage could not be accounted for after a case (unknown or partial "
            "usage); stopped conservatively rather than continuing unaccounted"
        )
    else:
        stop_reason = "runner returned fewer cases than planned (no spending cap was set)"
    metrics = _compute_metrics(results, scoring_policy=args.scoring_policy)
    _print_summary(metrics)

    report = {
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": args.dry_run,
        "model": args.model if not args.dry_run else "mock",
        # Effective run configuration, recorded so a reader can tell what produced
        # these numbers without reading the CLI invocation.
        "config": _effective_config(args, llm_agent, baseline_agent, arm_c_agent),
        # Present only for a continuation run: which cases an earlier segment had
        # already attempted, and how much known spend was carried forward.
        "continuation": exclusion,
        # Frozen hashes of the prompt material each arm sends, so a later run can
        # be shown to match this one — or shown not to.
        "prompt_fingerprint": _prompt_fingerprint(llm_agent, baseline_agent, arm_c_agent),
        "sample": (
            {
                "file": args.sample_file.name,
                "seed": sample_record.get("seed"),
                "total": sample_record.get("total"),
                "case_ids_sha256": _case_ids_sha256(sample_record["case_ids"]),
                "gold_sha256": sample_record.get("gold_sha256"),
                "apportionment": sample_record.get("apportionment"),
                "verification": sample_verification,
            }
            if sample_record
            else None
        ),
        # Every input the run read, hashed, so the numbers can be tied to the data
        # that produced them rather than to whatever is on disk later.
        "input_hashes": {
            "benchmark_dir_name": (
                args.benchmark_dir.name if args.benchmark_dir else None
            ),
            "files": (
                _benchmark_hashes(args.benchmark_dir) if args.benchmark_dir else {}
            ),
            "sample_file_sha256": (
                _sha256_file(args.sample_file) if args.sample_file else None
            ),
        },
        # Three different facts, kept apart on purpose: planned is what the run
        # set out to do, attempted is what it started, completed is what produced
        # a real prediction, failed is what ran and did not finish, and not_run
        # is what a spending stop kept from starting at all.
        "plan": {
            "planned": planned,
            "attempted": attempted,
            "completed": metrics.get("llm_agent", {}).get("n_completed", 0),
            "failed": metrics.get("llm_agent", {}).get("n_failed", 0),
            "not_run": not_run,
            "partial": not_run > 0,
            "stop_reason": stop_reason,
            "stop_detail": stop_state or None,
        },
        # The pre-declared H3 acceptance test, or an explicit statement that a
        # partial run cannot support one.  Computed here because it needs the
        # planned/attempted split, which the metrics alone do not carry.
        "acceptance": _v3_acceptance(
            results, metrics,
            # Completeness is judged against the frozen sample, not against the
            # slice this run happened to be asked for.
            frozen_total=(
                len(sample_record["case_ids"]) if sample_record else planned
            ),
            arm_c_enabled=arm_c_agent is not None,
        ),
        "metrics": metrics,
        "cases": [
            {
                "case_id": r.case_id,
                # Recorded so a merged report is self-contained and a reader can
                # see what was actually asked.
                "question": r.question,
                "gold_label": r.gold_label,
                "gold_pmid": r.gold_pmid,
                # Rule-based result
                "rule_verdict": r.rule_verdict,
                "rule_correct": (
                    _matches_label_for(args.scoring_policy)(r.rule_verdict, r.gold_label)
                    if not r.rule_errored else None
                ),
                "rule_recall_at_1": r.rule_recall_at_1,
                # Recorded so a later segment can be merged without inventing
                # anything: a missing value stays missing rather than being
                # reconstructed from a summary.
                "rule_abstained": r.rule_abstained,
                "rule_latency_ms": r.rule_latency_ms,
                "rule_error_flag": r.rule_errored,
                "rule_error_type": (
                    r.rule_error_msg.split(":")[0] if r.rule_error_msg else None
                ),
                "rule_error_message": r.rule_error_msg,
                # LLM result
                "llm_verdict": r.llm_verdict,
                # llm_correct is null for any non-completed run: text_exit and
                # budget_exhausted are operational failures, not legitimate predictions,
                # and must not be counted as correct even when the verdict happens to match.
                "llm_correct": (
                    _matches_label_for(args.scoring_policy)(r.llm_verdict, r.gold_label)
                    if r.llm_run_status == "completed" and not r.llm_errored else None
                ),
                # Explicitly unscorable: "mixed" is a valid product verdict that no
                # gold label matches, so it is incorrect without being a wrong label.
                "llm_verdict_unscorable": _is_unscorable(args.scoring_policy, r.llm_verdict),
                "llm_abstained": r.llm_abstained,
                "llm_latency_ms": r.llm_latency_ms,
                "llm_citation_hit": r.llm_citation_hit,
                "llm_steps": r.llm_steps_used,
                # Says where llm_steps came from, so a count derived from audited
                # requests is not mistaken for the loop's own step counter.
                "llm_steps_source": r.llm_steps_source,
                # null means the cost could not be determined — it does not mean zero.
                "llm_cost_usd": r.llm_cost_usd,
                "llm_cost_unknown": r.llm_cost_usd is None,
                "llm_cost_status": r.llm_usage_coverage,
                # Survives a mid-run failure: the rounds that did complete still
                # contribute, so a case that died on round 2 is not invisible.
                "llm_cost_known_subtotal_usd": r.llm_cost_known_subtotal_usd,
                # Per-case evidence trail: request count, the model identifier the
                # server actually reported, token usage, emitted tool calls, and
                # why the loop ended.  Present on failed cases too.
                "llm_audit": r.llm_audit,
                # run_status distinguishes completed / text_exit / budget_exhausted / errored
                # — evaluators must not count non-completed cases as correct insufficient preds.
                "llm_run_status": r.llm_run_status,
                "llm_error_flag": r.llm_errored,
                "llm_error_type": (
                    r.llm_error_msg.split(":")[0] if r.llm_error_msg else None
                ),
                "llm_error_message": r.llm_error_msg,
                # ── Arm B: fixed-context LLM baseline ────────────────────────
                # Same shape as the agent arm, so a reader can compare the two
                # field by field.  Empty/null when no baseline arm was run.
                "baseline_verdict": r.baseline_verdict,
                "baseline_correct": (
                    _matches_label_for(args.scoring_policy)(r.baseline_verdict, r.gold_label)
                    if r.baseline_run_status == "completed" and not r.baseline_errored
                    else None
                ),
                "baseline_verdict_unscorable": _is_unscorable(args.scoring_policy, r.baseline_verdict),
                "baseline_run_status": r.baseline_run_status or None,
                "baseline_requests": (
                    r.baseline_audit["request_count"] if r.baseline_audit else None
                ),
                "baseline_tool_calls": (
                    len(r.baseline_audit["tool_calls"]) if r.baseline_audit else None
                ),
                "baseline_steps": r.baseline_steps_used,
                "baseline_steps_source": r.baseline_steps_source,
                "baseline_latency_ms": r.baseline_latency_ms,
                "baseline_citation_hit": r.baseline_citation_hit,
                "baseline_cost_usd": r.baseline_cost_usd,
                "baseline_cost_status": r.baseline_usage_coverage,
                "baseline_cost_known_subtotal_usd": r.baseline_cost_known_subtotal_usd,
                "baseline_audit": r.baseline_audit,
                "baseline_error_flag": r.baseline_errored,
                "baseline_error_type": (
                    r.baseline_error_msg.split(":")[0] if r.baseline_error_msg else None
                ),
                "baseline_error_message": r.baseline_error_msg,
                # ── diagnostic material: enough to re-check an evidence chain ──
                # The public answer text and the stated reasoning, not just the
                # verdict, plus what the tools actually returned.  A verdict with
                # no retrievable evidence behind it cannot be audited later.
                "llm_answer": r.llm_answer,
                "llm_claim": r.llm_claim,
                "llm_decisive_reason": r.llm_decisive_reason,
                "llm_tool_returns": r.llm_tool_returns,
                # ── Arm C (v3/H3) ────────────────────────────────────────────
                "arm_c_verdict": r.arm_c_verdict,
                "arm_c_correct": (
                    _matches_label_for(args.scoring_policy)(r.arm_c_verdict, r.gold_label)
                    if r.arm_c_run_status == "completed" and not r.arm_c_errored
                    else None
                ),
                "arm_c_verdict_unscorable": _is_unscorable(args.scoring_policy, r.arm_c_verdict),
                "arm_c_run_status": r.arm_c_run_status or None,
                "arm_c_requests": (
                    r.arm_c_audit["request_count"] if r.arm_c_audit else None
                ),
                "arm_c_steps": r.arm_c_steps_used,
                "arm_c_steps_source": r.arm_c_steps_source,
                "arm_c_latency_ms": r.arm_c_latency_ms,
                "arm_c_citation_hit": r.arm_c_citation_hit,
                "arm_c_cost_usd": r.arm_c_cost_usd,
                "arm_c_cost_status": r.arm_c_usage_coverage,
                "arm_c_cost_known_subtotal_usd": r.arm_c_cost_known_subtotal_usd,
                "arm_c_audit": r.arm_c_audit,
                "arm_c_answer": r.arm_c_answer,
                "arm_c_claim": r.arm_c_claim,
                "arm_c_decisive_reason": r.arm_c_decisive_reason,
                "arm_c_tool_returns": r.arm_c_tool_returns,
                "arm_c_error_flag": r.arm_c_errored,
                "arm_c_error_type": (
                    r.arm_c_error_msg.split(":")[0] if r.arm_c_error_msg else None
                ),
                "arm_c_error_message": r.arm_c_error_msg,
            }
            for r in results
        ],
    }

    output_path = args.output
    if output_path is None:
        tag = "dry_run" if args.dry_run else f"n{len(cases)}"
        output_path = REPO_ROOT / "reports" / f"pipeline_comparison_{tag}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved to: {output_path}")

    if not_run > 0:
        # The partial report is already written; the non-zero status makes the
        # shortfall visible to anything scripting this run.
        print(
            f"PARTIAL RUN: {not_run} of {planned} planned case(s) were never started. "
            f"Reason: {stop_reason}"
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
