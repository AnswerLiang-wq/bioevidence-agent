"""Tests for LLMEvidenceAgent and compare_pipelines metric logic.

All LLMEvidenceAgent tests inject a scripted fake client so no DeepSeek API
calls are made.  Metric tests exercise _compute_metrics directly with
hand-crafted CaseResult lists.

Coverage
--------
LLMEvidenceAgent:
  - Normal finish: search → fetch → inspect → finish with real citation
  - Missing required fields in finish → tool error (citation_required / missing_required_fields)
  - Empty cited_pmids with evidence verdict → tool error, loop continues
  - All-fake cited_pmids (never fetched) → tool error, loop continues
  - Mixed citations (one real, one fake) → rejected then retried with valid PMID only
  - Plain-text exit → run_status=text_exit, verdict=insufficient, abstained=True
  - Affirmative plain-text exit must NOT become a positive prediction
  - No prior search source for a cited PMID → tool error (citation_no_search_source)
  - PMID fetched but no search hit → _build_response raises ValueError
  - Step budget exhaustion → run_status=budget_exhausted, verdict=insufficient, NOT counted as correct
  - Failed finish does NOT permanently block retry (dup-guard exempts finish)
  - Same-args finish retry succeeds after completing fetch+inspect
  - Every inspected snippet becomes its own citation (no snippets[0] narrowing)
  - Source-span coverage reports figures with no covering cited span, and is
    labelled as provenance checking rather than semantic validation
  - Each finish rejection path names its own cause (fetched / inspected /
    empty-snippets / required-fields / verdict / search-source)
  - A rejected finish leaves no citation trace in the accepted response

Metric logic (_compute_metrics):
  - 1 correct + 1 failed case → completion_rate=50%, accuracy_on_completed=100%, e2e=50%
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bioevidence.corpus import CorpusDocument
from bioevidence.llm_agent import LLMEvidenceAgent
from bioevidence.product_contracts import summarize_source_coverage
from bioevidence.tools import LiteratureTools

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_doc(pmid: str, title: str = "", abstract: str = "") -> CorpusDocument:
    import hashlib
    text = f"{title} {abstract}".strip() or "placeholder"
    sha = hashlib.sha256(text.encode()).hexdigest()
    return CorpusDocument(
        pmid=pmid,
        title=title or f"Title {pmid}",
        abstract=abstract or f"The intervention significantly reduced outcomes in trial {pmid}.",
        doi=None,
        publication_types=("Journal Article",),
        journal="Test Journal",
        year=2020,
        first_author="Author A",
        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        content_sha256=sha,
    )


@pytest.fixture()
def tiny_corpus():
    return [
        _make_doc(
            "1001", "Mitochondria and apoptosis",
            "Results: mitochondria change shape before programmed cell death.",
        ),
        _make_doc(
            "1002", "Vaccine refrigeration failures",
            "Results: refrigerator failures caused temperature excursions affecting 40% of vaccines.",
        ),
        _make_doc(
            "1003", "Inhibin after molar pregnancy",
            "Results: serum inhibin declined significantly after molar pregnancy evacuation.",
        ),
    ]


@pytest.fixture()
def lit_tools(tiny_corpus):
    return LiteratureTools(tiny_corpus)


def _make_agent(lit_tools: LiteratureTools, fake_client, max_steps: int = 8) -> LLMEvidenceAgent:
    """Build an LLMEvidenceAgent that uses fake_client instead of OpenAI."""
    agent = LLMEvidenceAgent.__new__(LLMEvidenceAgent)
    agent._tools = lit_tools
    agent._model = "test-model"
    agent._max_steps = max_steps
    agent._max_tokens_per_call = 1024
    agent._base_url = "https://api.deepseek.com/v1"
    agent._client = fake_client
    return agent


# ---------------------------------------------------------------------------
# Fake OpenAI client helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction


@dataclass
class _FakeMessage:
    content: str | None
    tool_calls: list[_FakeToolCall] | None

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {"role": "assistant"}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        return result


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeCompletion:
    choices: list[_FakeChoice]
    usage: _FakeUsage = field(default_factory=_FakeUsage)


def _tool_call(name: str, args: dict[str, Any], call_id: str = "tc1") -> _FakeToolCall:
    return _FakeToolCall(
        id=call_id,
        function=_FakeFunction(name=name, arguments=json.dumps(args)),
    )


def _response(*tool_calls: _FakeToolCall, content: str | None = None) -> _FakeCompletion:
    return _FakeCompletion(
        choices=[_FakeChoice(message=_FakeMessage(content=content, tool_calls=list(tool_calls) or None))],
    )


def _text_response(content: str) -> _FakeCompletion:
    return _FakeCompletion(
        choices=[_FakeChoice(message=_FakeMessage(content=content, tool_calls=None))],
    )


def _scripted_client(responses: list[_FakeCompletion]) -> MagicMock:
    """Return a mock whose chat.completions.create() yields responses in order."""
    client = MagicMock()
    client.chat.completions.create.side_effect = responses
    return client


# ---------------------------------------------------------------------------
# Helper: standard valid-finish args for PMID 1001 (requires prior search+fetch+inspect)
# ---------------------------------------------------------------------------

def _finish_args_1001(call_id: str = "tc_finish") -> _FakeToolCall:
    return _tool_call("finish", {
        "verdict": "supported",
        "answer": "Yes, mitochondria change shape before programmed cell death.",
        "claim": "Mitochondria change shape before apoptosis.",
        "cited_pmids": ["1001"],
        "decisive_reason": "PMID 1001 directly reports this.",
    }, call_id)


# ---------------------------------------------------------------------------
# Tests: LLMEvidenceAgent
# ---------------------------------------------------------------------------

class TestNormalFinish:
    """Happy path: search → fetch → inspect → finish with a valid citation."""

    def test_supported_verdict_with_real_citation(self, lit_tools):
        responses = [
            _response(_tool_call("search_literature", {"query": "mitochondria apoptosis shape"}, "tc1")),
            _response(_tool_call("fetch_record", {"pmid": "1001"}, "tc2")),
            _response(_tool_call("inspect_evidence", {"pmid": "1001", "query": "mitochondria shape apoptosis"}, "tc3")),
            _response(_finish_args_1001("tc4")),
        ]
        agent = _make_agent(lit_tools, _scripted_client(responses))
        result = agent.run(question="Do mitochondria change shape before programmed cell death?")

        assert result["verdict"] == "supported"
        assert result["abstained"] is False
        assert result["provenance"]["agent_run"]["run_status"] == "completed"
        assert len(result["citations"]) == 1
        assert result["citations"][0]["pmid"] == "1001"
        assert result["citations"][0]["retrieval"]["method"] == "bm25"
        assert result["citations"][0]["retrieval"]["rank"] >= 1


class TestRequiredFieldsValidation:
    """finish with missing required fields → tool error."""

    def test_missing_decisive_reason_returns_tool_error(self, lit_tools):
        responses = [
            _response(_tool_call("search_literature", {"query": "mitochondria"}, "tc1")),
            _response(_tool_call("fetch_record", {"pmid": "1001"}, "tc2")),
            _response(_tool_call("inspect_evidence", {"pmid": "1001", "query": "mitochondria"}, "tc3")),
            # finish without decisive_reason
            _response(_tool_call("finish", {
                "verdict": "supported",
                "answer": "Yes.",
                "claim": "Some claim.",
                "cited_pmids": ["1001"],
                # decisive_reason omitted
            }, "tc4")),
            # After error, give up with text exit
            _text_response("Cannot finish."),
        ]
        agent = _make_agent(lit_tools, _scripted_client(responses))
        result = agent.run(question="Do mitochondria change shape?")
        assert result["provenance"]["agent_run"]["run_status"] == "text_exit"
        assert result["verdict"] == "insufficient"

    def test_invalid_verdict_value_returns_tool_error(self, lit_tools):
        responses = [
            _response(_tool_call("finish", {
                "verdict": "maybe",  # not in valid set
                "answer": "Maybe.",
                "claim": "Some claim.",
                "cited_pmids": [],
                "decisive_reason": "Unclear.",
            }, "tc1")),
            _text_response("I don't know."),
        ]
        agent = _make_agent(lit_tools, _scripted_client(responses))
        result = agent.run(question="Question?")
        assert result["verdict"] == "insufficient"
        assert result["provenance"]["agent_run"]["run_status"] == "text_exit"


class TestEmptyCitedPmids:
    """finish with verdict=supported but cited_pmids=[] → tool error, loop continues."""

    def test_empty_citations_returns_tool_error_then_text_exit(self, lit_tools):
        responses = [
            _response(_tool_call("finish", {
                "verdict": "supported",
                "answer": "Yes.",
                "claim": "Claim without evidence.",
                "cited_pmids": [],
                "decisive_reason": "Trust me.",
            }, "tc1")),
            _text_response("I cannot provide citations."),
        ]
        agent = _make_agent(lit_tools, _scripted_client(responses))
        result = agent.run(question="Do mitochondria change shape?")

        assert result["verdict"] == "insufficient"
        assert result["abstained"] is True
        assert result["provenance"]["agent_run"]["run_status"] == "text_exit"
        assert result["citations"] == []


class TestAllFakeCitations:
    """finish citing only PMIDs that were never fetched → tool error, loop continues."""

    def test_all_fake_pmids_returns_tool_error(self, lit_tools):
        responses = [
            _response(_tool_call("finish", {
                "verdict": "supported",
                "answer": "Supported.",
                "claim": "Evidence from PMID 9999.",
                "cited_pmids": ["9999"],
                "decisive_reason": "PMID 9999 says so.",
            }, "tc1")),
            _text_response("Cannot find the PMID."),
        ]
        agent = _make_agent(lit_tools, _scripted_client(responses))
        result = agent.run(question="Does serum inhibin decline?")

        assert result["verdict"] == "insufficient"
        assert result["citations"] == []


class TestMixedCitations:
    """finish citing one real (fetched+inspected) PMID plus one fake one → rejected then accepted."""

    def test_mixed_citations_rejected_then_retry_succeeds(self, lit_tools):
        responses = [
            _response(_tool_call("search_literature", {"query": "vaccine temperature"}, "tc1")),
            _response(_tool_call("fetch_record", {"pmid": "1002"}, "tc2")),
            _response(_tool_call(
                "inspect_evidence",
                {"pmid": "1002", "query": "vaccine temperature refrigerator"},
                "tc3",
            )),
            # Cite real PMID 1002 AND fake PMID 8888 → rejected
            _response(_tool_call("finish", {
                "verdict": "supported",
                "answer": "Supported.",
                "claim": "Vaccine temperature problems documented.",
                "cited_pmids": ["1002", "8888"],
                "decisive_reason": "Both PMIDs confirm.",
            }, "tc4")),
            # After rejection, try again with only the real PMID → accepted
            _response(_tool_call("finish", {
                "verdict": "supported",
                "answer": "Supported by PMID 1002 only.",
                "claim": "Vaccine temperature problems documented.",
                "cited_pmids": ["1002"],
                "decisive_reason": "PMID 1002 confirms.",
            }, "tc5")),
        ]
        agent = _make_agent(lit_tools, _scripted_client(responses))
        result = agent.run(question="Can refrigerators cause vaccine temperature problems?")

        assert result["verdict"] == "supported"
        assert result["provenance"]["agent_run"]["run_status"] == "completed"
        assert len(result["citations"]) == 1
        assert result["citations"][0]["pmid"] == "1002"


class TestPlainTextExit:
    """Model produces a plain text response with no tool_call → text_exit."""

    def test_text_only_response_becomes_insufficient_text_exit(self, lit_tools):
        responses = [
            _text_response("I don't have enough information to answer this question."),
        ]
        agent = _make_agent(lit_tools, _scripted_client(responses))
        result = agent.run(question="Does serum inhibin decline after molar pregnancy?")

        assert result["verdict"] == "insufficient"
        assert result["abstained"] is True
        assert result["provenance"]["agent_run"]["run_status"] == "text_exit"
        assert result["citations"] == []

    def test_affirmative_text_exit_is_still_insufficient(self, lit_tools):
        """An affirmative plain-text exit must NOT be kept as a positive answer."""
        responses = [
            _text_response("Yes, serum inhibin definitely declines after molar pregnancy."),
        ]
        agent = _make_agent(lit_tools, _scripted_client(responses))
        result = agent.run(question="Does serum inhibin decline after molar pregnancy?")

        # Even though the model said "yes", it didn't call finish — must be insufficient.
        assert result["verdict"] == "insufficient"
        assert result["provenance"]["agent_run"]["run_status"] == "text_exit"
        assert result["citations"] == []
        # The raw affirmative model text must NOT appear in the product answer field —
        # only a fixed canonical "incomplete" string should be there.
        assert "definitely declines" not in result["answer"]
        assert "incomplete" in result["answer"].lower() or "did not complete" in result["answer"].lower()


class TestNoSearchSource:
    """finish citing a PMID that was fetched directly (no search) → tool error."""

    def test_pmid_with_no_search_entry_returns_tool_error(self, lit_tools):
        responses = [
            # Fetch without ever searching — inspect will succeed but finish must fail
            _response(_tool_call("fetch_record", {"pmid": "1003"}, "tc1")),
            _response(_tool_call("inspect_evidence", {"pmid": "1003", "query": "inhibin molar pregnancy"}, "tc2")),
            _response(_tool_call("finish", {
                "verdict": "supported",
                "answer": "Serum inhibin declines.",
                "claim": "Inhibin declines post-evacuation.",
                "cited_pmids": ["1003"],
                "decisive_reason": "PMID 1003 reports this.",
            }, "tc3")),
            _text_response("No search source available."),
        ]
        agent = _make_agent(lit_tools, _scripted_client(responses))
        result = agent.run(question="Does serum inhibin decline after molar pregnancy evacuation?")

        # finish should be rejected because 1003 was never in search results
        assert result["verdict"] == "insufficient"
        assert result["provenance"]["agent_run"]["run_status"] == "text_exit"
        assert result["citations"] == []


class TestNoSearchSourceRaisesInBuildResponse:
    """_build_response raises if a PMID was inspected but bypassed _dispatch validation."""

    def test_no_search_hit_raises_in_build_response(self, lit_tools):
        """Directly populate memory to bypass _dispatch, then call _build_response."""
        from bioevidence.llm_agent import SessionMemory
        from bioevidence.tools import ToolExecutor

        agent = _make_agent(lit_tools, MagicMock())

        # Manually inject a fetched + inspected record with NO search-result entry
        memory = SessionMemory(
            question="test",
            search_results=[],   # empty — no search was ever run
            fetched_records={"1003": {
                "title": "T", "abstract": "A", "year": 2020,
                "journal": "J", "document_sha256": "abc", "pmid": "1003",
            }},
            inspected_snippets={"1003": [{
                "text": "snippet", "start_char": 0, "end_char": 8,
                "section": "abstract", "snippet_sha256": "def",
                "relevance_score": 0.9,
            }]},
            call_history=[],
        )
        executor = ToolExecutor(agent._tools, max_calls=10)

        with pytest.raises(ValueError, match="no search-result entry in memory"):
            agent._build_response(
                finish_args={
                    "verdict": "supported",
                    "answer": "A.",
                    "claim": "C.",
                    "cited_pmids": ["1003"],
                    "decisive_reason": "R.",
                },
                memory=memory,
                executor=executor,
                request_id="test-req",
                steps=3,
                elapsed_ms=10.0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                run_status="completed",
                termination_reason="finish tool accepted.",
            )


class TestBudgetExhaustion:
    """Step budget exhausted without a finish call → budget_exhausted, insufficient."""

    def test_max_steps_reached_returns_insufficient(self, lit_tools):
        responses = [
            _response(_tool_call("search_literature", {"query": f"query {i}"}, f"tc{i}"))
            for i in range(20)
        ]
        agent = _make_agent(lit_tools, _scripted_client(responses), max_steps=3)
        result = agent.run(question="What is the effect of X on Y?")

        assert result["verdict"] == "insufficient"
        assert result["abstained"] is True
        assert result["provenance"]["agent_run"]["run_status"] == "budget_exhausted"

    def test_budget_exhausted_is_not_counted_correct_for_yes_gold(self, lit_tools):
        """Budget exhaustion → insufficient; must not match a gold_label='yes' verdict."""
        from bioevidence.pubmedqa import PUBMEDQA_VERDICT_MAP
        responses = [
            _response(_tool_call("search_literature", {"query": f"q{i}"}, f"tc{i}"))
            for i in range(20)
        ]
        agent = _make_agent(lit_tools, _scripted_client(responses), max_steps=2)
        result = agent.run(question="Anything?")

        assert result["provenance"]["agent_run"]["run_status"] == "budget_exhausted"
        # Gold label 'yes' maps to 'supported'; budget_exhausted gives 'insufficient' — not a match
        gold_verdict = PUBMEDQA_VERDICT_MAP.get("yes")
        assert result["verdict"] != gold_verdict


class TestFinishRetryAfterRejection:
    """A rejected finish must NOT permanently block retry with corrected args."""

    def test_same_finish_args_retryable_after_completing_fetch_inspect(self, lit_tools):
        """Dup-guard must exempt finish; same args should work after model fixes citations."""
        # First finish is missing citations (rejected).
        # Model then does search → fetch → inspect.
        # Second finish has identical structure but valid citations → must be accepted.
        responses = [
            # Step 1: finish attempt without any work done → citation_required error
            _response(_tool_call("finish", {
                "verdict": "supported",
                "answer": "Yes.",
                "claim": "Mitochondria change shape.",
                "cited_pmids": [],
                "decisive_reason": "I believe so.",
            }, "tc1")),
            # Step 2: model recovers — searches, fetches, inspects
            _response(_tool_call("search_literature", {"query": "mitochondria apoptosis"}, "tc2")),
            _response(_tool_call("fetch_record", {"pmid": "1001"}, "tc3")),
            _response(_tool_call("inspect_evidence", {"pmid": "1001", "query": "mitochondria apoptosis"}, "tc4")),
            # Step 3: retry finish with valid citation
            _response(_finish_args_1001("tc5")),
        ]
        agent = _make_agent(lit_tools, _scripted_client(responses))
        result = agent.run(question="Do mitochondria change shape before apoptosis?")

        assert result["verdict"] == "supported"
        assert result["provenance"]["agent_run"]["run_status"] == "completed"
        assert len(result["citations"]) == 1


class TestEmptySearchResults:
    """search returns no hits; model should still be able to call finish=insufficient."""

    def test_empty_search_results_finish_insufficient(self, lit_tools):
        responses = [
            _response(_tool_call("search_literature", {"query": "nonexistent topic xyzzy"}, "tc1")),
            _response(_tool_call("finish", {
                "verdict": "insufficient",
                "answer": "No relevant literature found.",
                "claim": "Insufficient evidence.",
                "cited_pmids": [],
                "decisive_reason": "Search returned no usable results.",
            }, "tc2")),
        ]
        agent = _make_agent(lit_tools, _scripted_client(responses))
        result = agent.run(question="Does xyzzy protein cause disease?")

        assert result["verdict"] == "insufficient"
        assert result["provenance"]["agent_run"]["run_status"] == "completed"
        assert result["citations"] == []


# ---------------------------------------------------------------------------
# Tests: multi-snippet citation emission and source-span coverage
# ---------------------------------------------------------------------------

_MULTI_SENTENCE_DOC = _make_doc(
    "2001",
    "Statin therapy and cardiovascular event reduction: a randomized controlled trial",
    "BACKGROUND: Statins lower LDL cholesterol. "
    "RESULTS: In this randomized controlled trial of 1200 patients, statin therapy "
    "reduced major cardiovascular events by 28% over 5 years compared to placebo "
    "(HR 0.72, 95% CI 0.61-0.85). "
    "The absolute risk reduction was 4.2%. "
    "No significant increase in myopathy was observed.",
)

_STATIN_QUESTION = (
    "Do statins reduce major cardiovascular events in randomized controlled trials?"
)
_STATIN_INSPECT_QUERY = (
    "statin therapy reduces major cardiovascular events randomized trial myopathy"
)


def _statin_responses(max_snippets: int, answer: str) -> list[Any]:
    """Scripted loop: search → fetch → inspect(max_snippets) → finish."""
    return [
        _response(_tool_call(
            "search_literature",
            {"query": "statin therapy cardiovascular events randomized trial", "top_k": 3},
            "tc1",
        )),
        _response(_tool_call("fetch_record", {"pmid": "2001"}, "tc2")),
        _response(_tool_call(
            "inspect_evidence",
            {"pmid": "2001", "query": _STATIN_INSPECT_QUERY, "max_snippets": max_snippets},
            "tc3",
        )),
        _response(_tool_call("finish", {
            "verdict": "supported",
            "answer": answer,
            "claim": "Statin therapy reduces major cardiovascular events in RCTs.",
            "cited_pmids": ["2001"],
            "decisive_reason": "PMID 2001 reports a randomized comparison against placebo.",
        }, "tc4")),
    ]


@pytest.fixture()
def multi_sentence_tools():
    return LiteratureTools([_MULTI_SENTENCE_DOC])


class TestMultiSnippetCitations:
    """Evidence the model inspected must not be narrowed to its first snippet."""

    def test_every_inspected_snippet_gets_its_own_citation(self, multi_sentence_tools):
        agent = _make_agent(
            multi_sentence_tools,
            _scripted_client(_statin_responses(4, "Statin therapy reduced events.")),
        )
        result = agent.run(question=_STATIN_QUESTION)

        assert result["provenance"]["agent_run"]["run_status"] == "completed"
        citations = result["citations"]
        assert len(citations) > 1, (
            "Expected one citation per inspected snippet; a query with several "
            f"matching sentences produced only {len(citations)} citation(s)."
        )
        assert {c["pmid"] for c in citations} == {"2001"}
        assert len({c["citation_id"] for c in citations}) == len(citations)
        spans = [(c["start_char"], c["end_char"]) for c in citations]
        assert len(set(spans)) == len(spans), "citation spans must be distinct"
        for citation in citations:
            assert json.loads(json.dumps(citation)) == citation, (
                "citation must survive a JSON round-trip to be auditable"
            )
            assert _MULTI_SENTENCE_DOC.abstract[
                citation["start_char"]:citation["end_char"]
            ] == citation["snippet"], "citation span must map onto the source abstract"

    def test_claim_references_every_emitted_citation(self, multi_sentence_tools):
        agent = _make_agent(
            multi_sentence_tools,
            _scripted_client(_statin_responses(4, "Statin therapy reduced events.")),
        )
        result = agent.run(question=_STATIN_QUESTION)

        emitted = {c["citation_id"] for c in result["citations"]}
        referenced = {
            cid for claim in result["claims"] for cid in claim["citation_ids"]
        }
        assert referenced == emitted


class TestSourceSpanCoverage:
    """Coverage is a provenance check, not an entailment judgement."""

    def _citation(self, snippet: str, citation_id: str = "C1") -> dict[str, Any]:
        return {"citation_id": citation_id, "snippet": snippet}

    def test_figure_inside_a_cited_span_is_covered(self):
        report = summarize_source_coverage(
            "Events fell by 28% (HR 0.72).",
            [self._citation("RESULTS: events fell by 28% (HR 0.72) overall.")],
        )
        assert report["counts"]["total"] >= 2
        assert report["uncovered"] == []

    def test_figure_absent_from_every_cited_span_is_surfaced(self):
        """Reproduces the observed defect: a figure sourced only from the fetched
        abstract, with no cited span covering it, must be reported, not silently
        accepted."""
        report = summarize_source_coverage(
            "Events fell by 28%, an absolute risk reduction of 4.2%.",
            [self._citation("RESULTS: events fell by 28% over 5 years.")],
        )
        assert "4.2%" in report["uncovered"]
        covered = {a["text"] for a in report["assertions"] if a["covered"]}
        assert "28%" in covered

    def test_reports_which_citation_covers_each_figure(self):
        report = summarize_source_coverage(
            "Events fell by 28%.",
            [self._citation("unrelated sentence", "C1"),
             self._citation("events fell by 28%", "C2")],
        )
        entry = next(a for a in report["assertions"] if a["text"] == "28%")
        assert entry["covered"] is True
        assert entry["citation_ids"] == ["C2"]

    def test_cosmetic_variants_do_not_produce_false_uncovered(self):
        report = summarize_source_coverage(
            "The trial enrolled 1,200 patients (95% CI 0.61 – 0.85).",
            [self._citation("In 1200 patients, 95% CI 0.61-0.85 was reported.")],
        )
        assert report["uncovered"] == [], report["uncovered"]

    # ── numeric boundary (supervisor review, three reproduced defects) ───────

    def test_percentage_is_not_covered_by_a_longer_number(self):
        """28% must not be accepted against a span holding 128%."""
        report = summarize_source_coverage(
            "Events fell by 28%.", [self._citation("An unrelated 128% figure.")]
        )
        assert report["uncovered"] == ["28%"]
        assert report["assertions"][0]["covered"] is False

    def test_effect_estimate_is_not_covered_by_a_longer_decimal(self):
        """HR 0.72 must not be accepted against a span holding HR 0.729."""
        report = summarize_source_coverage(
            "Effect was HR 0.72.", [self._citation("Effect was HR 0.729 in the model.")]
        )
        assert report["uncovered"] == ["HR 0.72"]

    def test_spelled_out_effect_label_is_extracted_not_silently_ignored(self):
        """'hazard ratio 0.72' must yield an assertion rather than an empty result."""
        report = summarize_source_coverage(
            "Effect was hazard ratio 0.72.", [self._citation("Effect was HR 0.72.")]
        )
        assert [a["text"] for a in report["assertions"]] == ["hazard ratio 0.72"]
        assert report["uncovered"] == ["hazard ratio 0.72"], (
            "A spelled-out label is not the same notation as 'HR'; it is reported "
            "for review rather than silently matched or silently dropped."
        )

    def test_boundary_guard_does_not_reject_ordinary_neighbours(self):
        """The guard applies to numeric edges only; brackets and commas still match."""
        report = summarize_source_coverage(
            "In 1200 patients the effect was HR 0.72.",
            [self._citation("(1200 patients) effect was HR 0.72, CI 0.61-0.85.")],
        )
        assert report["uncovered"] == [], report["uncovered"]

    def test_unclassified_numeric_literals_are_listed(self):
        report = summarize_source_coverage(
            "Over 5 years the effect was HR 0.72.",
            [self._citation("effect was HR 0.72")],
        )
        assert report["unclassified_numeric_literals"] == ["5"], (
            "Numbers no pattern classifies must stay visible, so a quiet result "
            "cannot be read as 'everything was checked'."
        )
        assert report["uncovered"] == []

    def test_classified_figures_are_not_also_reported_as_unclassified(self):
        report = summarize_source_coverage(
            "28% of 1200 patients.", [self._citation("28% of 1200 patients")]
        )
        assert report["unclassified_numeric_literals"] == []

    def test_classified_figure_is_not_repeated_when_a_space_follows(self):
        """The literal match must stop at the number.  Running one character past
        it pushed the match outside the classified span, so a figure that *was*
        classified reappeared as unclassified whenever whitespace followed."""
        report = summarize_source_coverage(
            "HR 0.72 was reported.", [self._citation("HR 0.72")]
        )
        assert report["unclassified_numeric_literals"] == []
        assert report["uncovered"] == []

    def test_classified_figure_is_not_repeated_when_a_newline_follows(self):
        report = summarize_source_coverage(
            "HR 0.72\nwas reported.", [self._citation("HR 0.72")]
        )
        assert report["unclassified_numeric_literals"] == []

    def test_classified_percentage_is_not_repeated_across_whitespace(self):
        for answer in ("28% was reported.", "28%\nwas reported."):
            report = summarize_source_coverage(answer, [self._citation("28%")])
            assert report["unclassified_numeric_literals"] == [], repr(answer)

    def test_unclassified_literals_are_still_reported_when_genuinely_so(self):
        """The fix must not silence real gaps: a duration is still unclassified."""
        report = summarize_source_coverage(
            "Over 5 years the effect was HR 0.72.",
            [self._citation("HR 0.72")],
        )
        assert report["unclassified_numeric_literals"] == ["5"]

    def test_leading_guard_blocks_a_comma_grouped_tail(self):
        """'200 patients' must not be accepted as the tail of '1,200 patients'."""
        report = summarize_source_coverage(
            "We enrolled 200 patients.", [self._citation("We enrolled 1,200 patients.")]
        )
        assert report["uncovered"] == ["200 patients"]

    def test_trailing_punctuation_does_not_break_a_match(self):
        """Only a following digit continues a number; punctuation does not."""
        for tail in (", 95% CI 0.61-0.85", ".", ")", " was reported"):
            report = summarize_source_coverage(
                "Effect was HR 0.72.", [self._citation(f"Effect was HR 0.72{tail}")]
            )
            assert report["uncovered"] == [], f"tail={tail!r}"

    def test_result_is_labeled_as_not_semantic_validation(self):
        report = summarize_source_coverage("Events fell by 28%.", [self._citation("28%")])
        assert report["check"] == "cited_span_coverage"
        assert report["is_semantic_validation"] is False
        assert report["limitations"], "limitations must be stated, not implied"

    def test_missing_citations_reports_why_it_was_skipped(self):
        report = summarize_source_coverage("Events fell by 28%.", [])
        assert report["skipped_reason"] == "no citations to check against"
        assert report["counts"]["total"] == 0

    def test_answer_without_literal_figures_yields_no_assertions(self):
        report = summarize_source_coverage(
            "The evidence is inconclusive.", [self._citation("some source text")]
        )
        assert report["assertions"] == []
        assert report["counts"] == {"total": 0, "covered": 0, "uncovered": 0}

    def test_coverage_is_recordable_on_the_agent_response(self, multi_sentence_tools):
        agent = _make_agent(
            multi_sentence_tools,
            _scripted_client(_statin_responses(
                4,
                "In 1200 patients, statin therapy reduced major cardiovascular events "
                "by 28% over 5 years (HR 0.72, 95% CI 0.61-0.85).",
            )),
        )
        result = agent.run(question=_STATIN_QUESTION)

        coverage = result["provenance"]["source_coverage"]
        assert coverage["check"] == "cited_span_coverage"
        assert coverage["is_semantic_validation"] is False
        assert coverage["uncovered"] == [], coverage["uncovered"]
        # Numeric material the patterns do not classify stays visible on the
        # response, so a clean `uncovered` list cannot be read as full coverage.
        # "5" comes from "over 5 years": a duration, outside the figure classes.
        assert coverage["unclassified_numeric_literals"] == ["5"], (
            f"unexpected unclassified literals: {coverage['unclassified_numeric_literals']}"
        )

    def test_figure_only_in_the_fetched_abstract_is_flagged(self, multi_sentence_tools):
        """With inspection limited to one snippet, a figure that lives only in the
        fetched abstract has no cited span covering it and must be reported."""
        agent = _make_agent(
            multi_sentence_tools,
            _scripted_client(_statin_responses(
                1,
                "Events fell by 28% with an absolute risk reduction of 4.2%.",
            )),
        )
        result = agent.run(question=_STATIN_QUESTION)

        coverage = result["provenance"]["source_coverage"]
        assert "4.2%" in coverage["uncovered"], (
            f"4.2% has no covering cited span; got {coverage!r}"
        )
        # The check is advisory: it must not have altered the verdict or contract.
        assert result["verdict"] == "supported"
        assert result["provenance"]["agent_run"]["run_status"] == "completed"


class TestRejectedFinishDoesNotLeakIntoCitations:
    """A rejected finish attempt must leave no citation trace."""

    def test_citations_reflect_only_the_accepted_finish(self, lit_tools):
        responses = [
            # First finish names a PMID that was never fetched → rejected.
            _response(_tool_call("finish", {
                "verdict": "supported",
                "answer": "Yes.",
                "claim": "Mitochondria change shape.",
                "cited_pmids": ["1003"],
                "decisive_reason": "I recall this.",
            }, "tc1")),
            # Model recovers and finishes against a properly retrieved PMID.
            _response(_tool_call("search_literature", {"query": "mitochondria apoptosis"}, "tc2")),
            _response(_tool_call("fetch_record", {"pmid": "1001"}, "tc3")),
            _response(_tool_call(
                "inspect_evidence", {"pmid": "1001", "query": "mitochondria apoptosis"}, "tc4"
            )),
            _response(_finish_args_1001("tc5")),
        ]
        agent = _make_agent(lit_tools, _scripted_client(responses))
        result = agent.run(question="Do mitochondria change shape before apoptosis?")

        assert result["provenance"]["agent_run"]["run_status"] == "completed"
        assert [c["pmid"] for c in result["citations"]] == ["1001"], (
            "The rejected finish named PMID 1003; no citation may reference it."
        )


def _tool_error_codes(agent: LLMEvidenceAgent) -> list[str]:
    """Error codes returned by rejected tool calls, in dispatch order."""
    codes: list[str] = []
    for message in getattr(agent, "_last_messages", []):
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content") or "")
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and "error" in payload:
            codes.append(str(payload["error"]))
    return codes


class TestFinishRejectionErrorCodes:
    """Each finish rejection must name its own cause, not fail generically."""

    def _run(self, lit_tools, responses):
        agent = _make_agent(lit_tools, _scripted_client(responses))
        result = agent.run(question="Do mitochondria change shape before apoptosis?")
        return agent, result

    def _finish(self, cited: list[str], call_id: str):
        return _response(_tool_call("finish", {
            "verdict": "supported",
            "answer": "Yes.",
            "claim": "Mitochondria change shape.",
            "cited_pmids": cited,
            "decisive_reason": "PMID 1001 reports it.",
        }, call_id))

    def test_unfetched_pmid_reports_citation_not_fetched(self, lit_tools):
        agent, result = self._run(lit_tools, [
            _response(_tool_call("search_literature", {"query": "mitochondria"}, "tc1")),
            self._finish(["1001"], "tc2"),
            _text_response("Giving up."),
        ])
        assert "citation_not_fetched" in _tool_error_codes(agent)
        assert result["citations"] == []

    def test_fetched_but_uninspected_pmid_reports_citation_not_inspected(self, lit_tools):
        agent, result = self._run(lit_tools, [
            _response(_tool_call("search_literature", {"query": "mitochondria apoptosis"}, "tc1")),
            _response(_tool_call("fetch_record", {"pmid": "1001"}, "tc2")),
            self._finish(["1001"], "tc3"),
            _text_response("Giving up."),
        ])
        assert "citation_not_inspected" in _tool_error_codes(agent)
        assert result["citations"] == []

    def test_inspection_yielding_no_snippets_reports_citation_empty_snippets(self, lit_tools):
        agent, result = self._run(lit_tools, [
            _response(_tool_call("search_literature", {"query": "mitochondria apoptosis"}, "tc1")),
            _response(_tool_call("fetch_record", {"pmid": "1001"}, "tc2")),
            # No token overlap with the abstract → the tool returns zero snippets.
            _response(_tool_call("inspect_evidence", {"pmid": "1001", "query": "zzz qqq"}, "tc3")),
            self._finish(["1001"], "tc4"),
            _text_response("Giving up."),
        ])
        assert "citation_empty_snippets" in _tool_error_codes(agent)
        assert result["citations"] == []


# ---------------------------------------------------------------------------
# Tests: _compute_metrics from compare_pipelines.py
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Verify completion_rate, accuracy_on_completed, end_to_end_success_rate arithmetic."""

    def _make_result(
        self,
        *,
        case_id: str,
        gold_label: str,
        llm_verdict: str,
        llm_run_status: str,
        llm_errored: bool = False,
    ):
        from scripts.compare_pipelines import CaseResult
        return CaseResult(
            case_id=case_id,
            question="Q",
            gold_label=gold_label,
            gold_pmid="0",
            rule_verdict="insufficient",
            rule_abstained=True,
            rule_errored=False,
            rule_error_msg=None,
            rule_recall_at_1=False,
            rule_latency_ms=10.0,
            llm_verdict=llm_verdict,
            llm_abstained=(llm_verdict == "insufficient"),
            llm_errored=llm_errored,
            llm_error_msg=None,
            llm_citation_hit=False,
            llm_latency_ms=50.0,
            llm_cost_usd=0.0,
            llm_steps_used=3,
            llm_run_status=llm_run_status,
        )

    def test_one_correct_one_failed_gives_expected_rates(self):
        """1 correct completed case + 1 budget_exhausted failed case → 50/100/50."""
        from bioevidence.pubmedqa import PUBMEDQA_VERDICT_MAP
        from scripts.compare_pipelines import _compute_metrics

        yes_verdict = PUBMEDQA_VERDICT_MAP["yes"]   # "supported"

        results = [
            # Case A: completed, correct (gold=yes, llm=supported)
            self._make_result(
                case_id="A",
                gold_label="yes",
                llm_verdict=yes_verdict,
                llm_run_status="completed",
            ),
            # Case B: budget_exhausted, verdict=insufficient — NOT correct for gold=yes
            self._make_result(
                case_id="B",
                gold_label="yes",
                llm_verdict="insufficient",
                llm_run_status="budget_exhausted",
            ),
        ]
        metrics = _compute_metrics(results)
        llm = metrics["llm_agent"]

        assert llm["completion_rate"] == pytest.approx(0.5)
        assert llm["accuracy_on_completed"] == pytest.approx(1.0)
        assert llm["end_to_end_success_rate"] == pytest.approx(0.5)

    def test_all_completed_correct(self):
        from bioevidence.pubmedqa import PUBMEDQA_VERDICT_MAP
        from scripts.compare_pipelines import _compute_metrics
        yes_verdict = PUBMEDQA_VERDICT_MAP["yes"]
        results = [
            self._make_result(case_id=str(i), gold_label="yes",
                              llm_verdict=yes_verdict, llm_run_status="completed")
            for i in range(4)
        ]
        metrics = _compute_metrics(results)
        llm = metrics["llm_agent"]
        assert llm["completion_rate"] == pytest.approx(1.0)
        assert llm["accuracy_on_completed"] == pytest.approx(1.0)
        assert llm["end_to_end_success_rate"] == pytest.approx(1.0)

    def test_text_exit_not_counted_as_correct_prediction(self):
        """text_exit case has verdict=insufficient but is NOT a legitimate prediction."""
        from bioevidence.pubmedqa import PUBMEDQA_VERDICT_MAP
        from scripts.compare_pipelines import _compute_metrics

        no_verdict = PUBMEDQA_VERDICT_MAP["no"]   # "contradicted"

        results = [
            # Case A: completed, correct
            self._make_result(case_id="A", gold_label="no", llm_verdict=no_verdict, llm_run_status="completed"),
            # Case B: text_exit — gold=maybe (insufficient), llm=insufficient
            # This should NOT count as a completed case for accuracy_on_completed
            self._make_result(case_id="B", gold_label="maybe", llm_verdict="insufficient", llm_run_status="text_exit"),
        ]
        metrics = _compute_metrics(results)
        llm = metrics["llm_agent"]

        # Only 1 of 2 cases completed
        assert llm["completion_rate"] == pytest.approx(0.5)
        # accuracy_on_completed is computed only over completed cases (case A: correct)
        assert llm["accuracy_on_completed"] == pytest.approx(1.0)
        # end_to_end: 1 correct out of 2 total
        assert llm["end_to_end_success_rate"] == pytest.approx(0.5)
        # n_completed should be 1
        assert llm["n_completed"] == 1

    def test_budget_exhausted_maybe_gold_all_null(self, tmp_path, monkeypatch):
        """gold=maybe + verdict=insufficient + run_status=budget_exhausted.

        Calls main() directly with a monkeypatched run_comparison so that the
        JSON report is produced by the real export code path — not a hand-rolled
        reproduction.  Asserts:
          - per-case llm_correct is null in the exported JSON
          - metrics.llm_agent.accuracy and accuracy_on_completed are null
          - metrics.llm_agent.completion_rate == 0
          - metrics.llm_agent.end_to_end_success_rate == 0

        Uses a mock client; no paid API is called.
        """
        import json as _json
        import sys
        from bioevidence.pubmedqa import PUBMEDQA_VERDICT_MAP
        from scripts.compare_pipelines import CaseResult, main

        maybe_verdict = PUBMEDQA_VERDICT_MAP["maybe"]  # "insufficient"

        budget_case = CaseResult(
            case_id="X",
            question="Q",
            gold_label="maybe",
            gold_pmid="0",
            rule_verdict="insufficient",
            rule_abstained=True,
            rule_errored=False,
            rule_error_msg=None,
            rule_recall_at_1=False,
            rule_latency_ms=10.0,
            llm_verdict=maybe_verdict,
            llm_abstained=True,
            llm_errored=False,
            llm_error_msg=None,
            llm_citation_hit=False,
            llm_latency_ms=50.0,
            llm_cost_usd=0.0,
            llm_steps_used=3,
            llm_run_status="budget_exhausted",
        )

        # Replace run_comparison with a stub that returns our scripted case.
        import scripts.compare_pipelines as _cp
        monkeypatch.setattr(_cp, "run_comparison", lambda **_kwargs: [budget_case])

        report_path = tmp_path / "report.json"
        monkeypatch.setattr(
            sys,
            "argv",
            ["compare_pipelines", "--dry-run", "--output", str(report_path)],
        )

        main()

        report = _json.loads(report_path.read_text())

        # ── per-case assertions ──────────────────────────────────────────────
        cases = report["cases"]
        assert len(cases) == 1, f"expected 1 case, got {len(cases)}"
        case = cases[0]
        assert case["llm_correct"] is None, (
            f"llm_correct must be null for budget_exhausted, got {case['llm_correct']!r}"
        )

        # ── metrics assertions ───────────────────────────────────────────────
        llm = report["metrics"]["llm_agent"]
        assert llm["accuracy_on_completed"] is None, (
            f"accuracy_on_completed must be null, got {llm['accuracy_on_completed']!r}"
        )
        assert llm["accuracy"] is None, (
            f"accuracy alias must be null, got {llm['accuracy']!r}"
        )
        assert llm["completion_rate"] == pytest.approx(0.0), (
            f"completion_rate must be 0.0, got {llm['completion_rate']!r}"
        )
        assert llm["end_to_end_success_rate"] == pytest.approx(0.0), (
            f"end_to_end_success_rate must be 0.0, got {llm['end_to_end_success_rate']!r}"
        )
