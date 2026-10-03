"""Offline regression tests for the PubMedQA two-arm comparison harness.

Covers the corrected scoring, cost and budget clauses: macro-F1 over completed
cases only, the three non-completion statuses, "mixed" as an unscorable verdict,
unknown cost that must not be summed as zero, the spending cap, and the frozen
stratified sample.

Every model interaction is a stub.  No API call is made and no API key is read.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bioevidence.audit import AuditingClient  # noqa: E402
from bioevidence.pubmedqa import PUBMEDQA_VERDICT_MAP  # noqa: E402
from bioevidence.pubmedqa_sample import (  # noqa: E402
    build_frozen_sample_record,
    frozen_sample,
    largest_remainder_quotas,
)

YES = PUBMEDQA_VERDICT_MAP["yes"]           # "supported"
NO = PUBMEDQA_VERDICT_MAP["no"]             # "contradicted"
MAYBE = PUBMEDQA_VERDICT_MAP["maybe"]       # "insufficient"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _gold_rows(counts: dict[str, int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    index = 0
    for label, count in counts.items():
        for _ in range(count):
            index += 1
            rows.append({
                "case_id": f"PUBMEDQA-TEST-{index:04d}",
                "pmid": str(1000000 + index),
                "label": label,
                "verdict": PUBMEDQA_VERDICT_MAP[label],
            })
    return rows


REAL_DISTRIBUTION = {"yes": 276, "no": 169, "maybe": 55}


def _case_result(**overrides: Any):
    from scripts.compare_pipelines import CaseResult

    base: dict[str, Any] = {
        "case_id": "C1",
        "question": "Q",
        "gold_label": "yes",
        "gold_pmid": "1",
        "rule_verdict": MAYBE,
        "rule_abstained": True,
        "rule_errored": False,
        "rule_error_msg": None,
        "rule_recall_at_1": False,
        "rule_latency_ms": 1.0,
        "llm_verdict": YES,
        "llm_abstained": False,
        "llm_errored": False,
        "llm_error_msg": None,
        "llm_citation_hit": False,
        "llm_latency_ms": 2.0,
        "llm_steps_used": 3,
        "llm_run_status": "completed",
        "llm_cost_usd": 0.001,
        "llm_usage_coverage": "confirmed",
        "llm_cost_known_subtotal_usd": 0.001,
    }
    # The audit-derived subtotal normally mirrors the agent's own figure for a
    # fully confirmed run; keeping them in step here avoids tests that disagree
    # with themselves about what a case cost.
    if "llm_cost_usd" in overrides and "llm_cost_known_subtotal_usd" not in overrides:
        base["llm_cost_known_subtotal_usd"] = overrides["llm_cost_usd"]
    base.update(overrides)
    return CaseResult(**base)


# ---------------------------------------------------------------------------
# Frozen stratified sample
# ---------------------------------------------------------------------------

class TestLargestRemainderQuotas:
    def test_real_distribution_apportions_28_17_5(self):
        apportionment = largest_remainder_quotas(REAL_DISTRIBUTION, 50)
        quotas = {row["label"]: row["quota"] for row in apportionment["strata"]}
        assert quotas == {"yes": 28, "no": 17, "maybe": 5}
        assert sum(quotas.values()) == 50

    def test_arithmetic_is_exposed_for_hand_checking(self):
        apportionment = largest_remainder_quotas(REAL_DISTRIBUTION, 50)
        by_label = {row["label"]: row for row in apportionment["strata"]}
        assert by_label["no"]["exact_quota"] == pytest.approx(16.9)
        assert by_label["no"]["floor"] == 16
        assert by_label["no"]["remainder"] == pytest.approx(0.9)
        # The two spare seats go to the two largest remainders: no (0.9), yes (0.6).
        assert by_label["no"]["quota"] == 17
        assert by_label["yes"]["quota"] == 28
        assert by_label["maybe"]["quota"] == 5

    def test_drawing_more_than_the_population_is_rejected(self):
        with pytest.raises(ValueError):
            largest_remainder_quotas({"yes": 2, "no": 1, "maybe": 1}, 50)


class TestFrozenSample:
    def test_same_seed_yields_identical_case_ids(self):
        rows = _gold_rows(REAL_DISTRIBUTION)
        first = frozen_sample(rows, total=50, seed=20260729)
        second = frozen_sample(rows, total=50, seed=20260729)
        assert first["case_ids"] == second["case_ids"]

    def test_input_row_order_does_not_change_the_draw(self):
        rows = _gold_rows(REAL_DISTRIBUTION)
        shuffled = list(reversed(rows))
        assert (
            frozen_sample(rows, total=50, seed=20260729)["case_ids"]
            == frozen_sample(shuffled, total=50, seed=20260729)["case_ids"]
        )

    def test_different_seed_changes_the_draw(self):
        rows = _gold_rows(REAL_DISTRIBUTION)
        assert (
            frozen_sample(rows, total=50, seed=20260729)["case_ids"]
            != frozen_sample(rows, total=50, seed=1)["case_ids"]
        )

    def test_stratum_quotas_are_respected(self):
        rows = _gold_rows(REAL_DISTRIBUTION)
        sample = frozen_sample(rows, total=50, seed=20260729)
        label_by_case = sample["label_by_case"]
        counts = {"yes": 0, "no": 0, "maybe": 0}
        for case_id in sample["case_ids"]:
            counts[label_by_case[case_id]] += 1
        assert counts == {"yes": 28, "no": 17, "maybe": 5}
        assert len(sample["case_ids"]) == 50

    def test_record_carries_seed_and_source_hash(self, tmp_path):
        gold = tmp_path / "test_gold.jsonl"
        gold.write_text(
            "".join(json.dumps(row) + "\n" for row in _gold_rows(REAL_DISTRIBUTION)),
            encoding="utf-8",
        )
        record = build_frozen_sample_record(gold, total=50, seed=20260729)
        assert record["seed"] == 20260729
        assert record["gold_rows"] == 500
        assert len(record["gold_sha256"]) == 64
        assert record["gold_used_for_model_input"] is False
        assert sum(row["quota"] for row in record["apportionment"]["strata"]) == 50

    def test_duplicate_case_ids_are_rejected(self):
        rows = _gold_rows({"yes": 2, "no": 1, "maybe": 1})
        rows.append(dict(rows[0]))
        with pytest.raises(ValueError, match="duplicate case_id"):
            frozen_sample(rows, total=3, seed=1)


# ---------------------------------------------------------------------------
# Macro-F1
# ---------------------------------------------------------------------------

class TestMacroF1:
    def _metrics(self, results):
        from scripts.compare_pipelines import _compute_metrics
        return _compute_metrics(results)

    def test_perfect_completed_run_scores_one(self):
        results = [
            _case_result(case_id="a", gold_label="yes", llm_verdict=YES),
            _case_result(case_id="b", gold_label="no", llm_verdict=NO),
            _case_result(case_id="c", gold_label="maybe", llm_verdict=MAYBE),
        ]
        assert self._metrics(results)["llm_agent"]["macro_f1"] == pytest.approx(1.0)

    def test_class_never_predicted_scores_zero_and_stays_in_the_average(self):
        """A label that occurs among completed cases but is never predicted is a
        real miss.  It scores 2*TP/(2*TP+FP+FN) = 0 and is NOT dropped — dropping
        it would raise the macro average by deleting the miss."""
        results = [
            _case_result(case_id="a", gold_label="yes", llm_verdict=YES),
            _case_result(case_id="b", gold_label="no", llm_verdict=MAYBE),
        ]
        detail = self._metrics(results)["llm_agent"]["macro_f1_detail"]
        assert detail["per_class"]["yes"]["f1"] == pytest.approx(1.0)
        assert detail["per_class"]["no"]["f1"] == 0.0          # FN=1, never predicted
        assert detail["per_class"]["maybe"]["f1"] == 0.0       # FP=1, never occurs
        assert detail["n_classes_averaged"] == 3
        assert self._metrics(results)["llm_agent"]["macro_f1"] == pytest.approx(0.3333)

    def test_wrong_direction_still_averages_all_three_classes(self):
        """Codex's reproduction: gold=yes predicted as no.  Every class scores 0
        and the average covers all three, rather than collapsing to one class."""
        results = [_case_result(case_id="a", gold_label="yes", llm_verdict=NO)]
        detail = self._metrics(results)["llm_agent"]["macro_f1_detail"]
        assert detail["per_class"]["yes"]["f1"] == 0.0         # FN
        assert detail["per_class"]["no"]["f1"] == 0.0          # FP
        assert detail["per_class"]["maybe"]["f1"] == 0.0       # denominator 0 -> 0
        assert detail["n_classes_averaged"] == 3
        assert self._metrics(results)["llm_agent"]["macro_f1"] == 0.0

    def test_class_with_no_activity_scores_zero_by_convention(self):
        results = [_case_result(case_id="a", gold_label="yes", llm_verdict=YES)]
        detail = self._metrics(results)["llm_agent"]["macro_f1_detail"]
        assert detail["per_class"]["maybe"] == {
            "support": 0, "tp": 0, "fp": 0, "fn": 0, "f1": 0.0,
        }
        assert detail["n_classes_averaged"] == 3
        assert self._metrics(results)["llm_agent"]["macro_f1"] == pytest.approx(0.3333)

    def test_rule_arm_uses_the_same_formula(self):
        """Both arms are scored identically, so their columns are comparable."""
        results = [
            _case_result(case_id="a", gold_label="yes", rule_verdict=YES),
            _case_result(case_id="b", gold_label="no", rule_verdict=MAYBE),
        ]
        llm_detail = self._metrics(results)["llm_agent"]["macro_f1_detail"]
        rb_detail = self._metrics(results)["rule_based"]["macro_f1_detail"]
        assert rb_detail["n_classes_averaged"] == 3
        assert set(rb_detail["per_class"]) == {"yes", "no", "maybe"}
        assert rb_detail["denominator"] == 2
        # Same arithmetic, different verdict field.
        assert rb_detail["per_class"]["no"]["f1"] == 0.0
        assert llm_detail["per_class"]["no"]["f1"] == 0.0

    def test_all_failed_gives_none_not_zero(self):
        results = [
            _case_result(case_id="a", llm_run_status="budget_exhausted", llm_verdict=MAYBE),
            _case_result(case_id="b", llm_run_status="text_exit", llm_verdict=MAYBE),
        ]
        llm = self._metrics(results)["llm_agent"]
        assert llm["macro_f1"] is None
        assert llm["accuracy_on_completed"] is None
        assert llm["end_to_end_success_rate"] == 0.0

    def test_denominator_is_completed_cases_only(self):
        results = [
            _case_result(case_id="a", gold_label="yes", llm_verdict=YES),
            _case_result(
                case_id="b", gold_label="no", llm_verdict=MAYBE,
                llm_run_status="budget_exhausted",
            ),
        ]
        detail = self._metrics(results)["llm_agent"]["macro_f1_detail"]
        assert detail["denominator"] == 1
        assert detail["per_class"]["no"]["support"] == 0

    def test_mixed_is_incorrect_and_not_remapped_to_maybe(self):
        results = [
            _case_result(case_id="a", gold_label="maybe", llm_verdict="mixed"),
        ]
        llm = self._metrics(results)["llm_agent"]
        assert llm["n_mixed_verdicts"] == 1
        # Scored as wrong: it matched neither maybe nor anything else.
        assert llm["accuracy_on_completed"] == 0.0
        detail = llm["macro_f1_detail"]
        assert detail["per_class"]["maybe"]["fn"] == 1
        # And it is not credited as a prediction of any class.
        assert all(
            detail["per_class"][label]["tp"] == 0
            for label in ("yes", "no", "maybe")
        )


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

class TestFailureTaxonomy:
    def _metrics(self, results):
        from scripts.compare_pipelines import _compute_metrics
        return _compute_metrics(results)

    @pytest.mark.parametrize("status", ["text_exit", "budget_exhausted"])
    def test_non_completion_statuses_are_failures(self, status):
        results = [_case_result(llm_run_status=status, llm_verdict=MAYBE, gold_label="maybe")]
        llm = self._metrics(results)["llm_agent"]
        assert llm["n_completed"] == 0
        assert llm["n_failed"] == 1
        assert llm["failure_breakdown"][status] == 1
        assert llm["completion_rate"] == 0.0
        assert llm["accuracy_on_completed"] is None
        assert llm["end_to_end_success_rate"] == 0.0

    def test_errored_is_a_failure_and_never_a_correct_prediction(self):
        """Even when a fallback verdict would match the gold label."""
        results = [
            _case_result(
                llm_errored=True, llm_error_msg="APIError: boom",
                llm_run_status="errored", llm_verdict="", gold_label="maybe",
            )
        ]
        llm = self._metrics(results)["llm_agent"]
        assert llm["failure_breakdown"]["errored"] == 1
        assert llm["n_completed"] == 0
        assert llm["end_to_end_success_rate"] == 0.0

    def test_all_three_statuses_are_counted_separately(self):
        results = [
            _case_result(case_id="a", llm_run_status="text_exit", llm_verdict=MAYBE),
            _case_result(case_id="b", llm_run_status="budget_exhausted", llm_verdict=MAYBE),
            _case_result(
                case_id="c", llm_run_status="errored", llm_errored=True,
                llm_error_msg="RateLimitError: slow down", llm_verdict="",
            ),
            _case_result(case_id="d", gold_label="yes", llm_verdict=YES),
        ]
        llm = self._metrics(results)["llm_agent"]
        assert llm["failure_breakdown"] == {
            "errored": 1, "budget_exhausted": 1, "text_exit": 1, "other": 0,
        }
        assert llm["n_failed"] == 3
        assert llm["n_completed"] == 1
        assert llm["completion_rate"] == pytest.approx(0.25)
        assert llm["end_to_end_success_rate"] == pytest.approx(0.25)

    def test_unknown_status_is_not_silently_folded_into_a_known_bucket(self):
        results = [_case_result(llm_run_status="something_new", llm_verdict=MAYBE)]
        breakdown = self._metrics(results)["llm_agent"]["failure_breakdown"]
        assert breakdown["other"] == 1
        assert breakdown["errored"] == 0


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------

class TestCostAccounting:
    def _cost(self, results):
        from scripts.compare_pipelines import _compute_metrics
        return _compute_metrics(results)["llm_agent"]["cost"]

    def test_unknown_cost_is_not_summed_as_zero(self):
        cost = self._cost([
            _case_result(case_id="a", llm_cost_usd=0.002),
            _case_result(case_id="b", llm_cost_usd=None, llm_usage_coverage="unknown"),
        ])
        assert cost["total_cost_usd_known"] == pytest.approx(0.002)
        assert cost["n_cost_known"] == 1
        assert cost["n_cost_unknown"] == 1
        assert cost["cost_complete"] is False

    def test_all_known_is_marked_complete(self):
        cost = self._cost([
            _case_result(case_id="a", llm_cost_usd=0.001),
            _case_result(case_id="b", llm_cost_usd=0.003),
        ])
        assert cost["total_cost_usd_known"] == pytest.approx(0.004)
        assert cost["cost_complete"] is True
        assert cost["n_cost_from_partial_runs"] == 0

    def test_partial_usage_is_tracked_separately_from_unknown(self):
        cost = self._cost([
            _case_result(case_id="a", llm_cost_usd=None, llm_usage_coverage="partial",
                         llm_cost_known_subtotal_usd=0.0005),
            _case_result(case_id="b", llm_cost_usd=None, llm_usage_coverage="unknown"),
            _case_result(case_id="c", llm_cost_usd=0.001),
        ])
        assert cost["usage_coverage_counts"] == {
            "confirmed": 1, "partial": 1, "unknown": 1,
        }
        # The partial run still contributed what it did report.
        assert cost["total_cost_usd_known"] == pytest.approx(0.0015)
        assert cost["n_cost_from_partial_runs"] == 1
        assert cost["n_cost_unknown"] == 1
        assert cost["cost_complete"] is False

    def test_partial_run_is_not_called_a_lower_bound(self):
        """A peak cache-miss subtotal covering only some rounds has an unknown
        direction — it must not be presented as a bound on the real charge."""
        cost = self._cost([
            _case_result(case_id="a", llm_cost_usd=None, llm_usage_coverage="partial",
                         llm_cost_known_subtotal_usd=0.0005),
        ])
        assert "not a bill" in cost["semantics"]
        assert "not a lower bound" in cost["known_subtotal_semantics"]
        assert "neither an upper nor a lower bound" in cost["caveat"]

    def test_fully_reported_run_is_described_as_a_ceiling(self):
        cost = self._cost([_case_result()])
        assert "conservative ceiling" in cost["semantics"]
        assert "not a bill" in cost["semantics"]
        assert "lower bound" not in cost["caveat"]


# ---------------------------------------------------------------------------
# Spending cap
# ---------------------------------------------------------------------------

class TestBudgetState:
    """The spending cap must not treat unaccountable spend as zero spend."""

    def _state(self, results):
        from scripts.compare_pipelines import _budget_state
        return _budget_state(results)

    def test_all_unknown_costs_do_not_look_like_zero_spend(self):
        """Codex's reproduction: five unknown-cost cases reported a subtotal of 0,
        so a positive cap could never fire and the run continued unchecked."""
        state = self._state([
            _case_result(case_id=str(i), llm_cost_usd=None,
                         llm_usage_coverage="unknown") for i in range(5)
        ])
        assert state["known_subtotal_usd"] == 0.0
        assert state["n_incomplete"] == 5
        assert state["usage_incomplete"] is True, (
            "unaccountable spend must mark the accounting incomplete, not free"
        )

    def test_partial_usage_also_marks_the_accounting_incomplete(self):
        state = self._state([
            _case_result(case_id="a", llm_cost_usd=None, llm_usage_coverage="partial",
                         llm_cost_known_subtotal_usd=0.0005),
        ])
        assert state["usage_incomplete"] is True
        assert state["known_subtotal_usd"] == pytest.approx(0.0005)

    def test_fully_confirmed_run_is_accounted(self):
        state = self._state([_case_result(llm_cost_usd=0.01)])
        assert state["usage_incomplete"] is False
        assert state["known_subtotal_usd"] == pytest.approx(0.01)

    def test_no_imputation_of_missing_costs(self):
        """A known case is never used to price an unknown one — that is exactly
        the substitution that lets a run slip past its cap."""
        state = self._state([
            _case_result(case_id="a", llm_cost_usd=0.10),
            _case_result(case_id="b", llm_cost_usd=None, llm_usage_coverage="unknown"),
        ])
        assert state["known_subtotal_usd"] == pytest.approx(0.10), (
            "the unknown case must not be charged the mean of the known ones"
        )
        assert "mean" not in state


class TestBudgetStop:
    def test_threshold_stop_is_reported_distinctly(self):
        from scripts.compare_pipelines import run_comparison

        docs, cases = _tiny_comparison_inputs(4)
        agent = _AuditedScriptedAgent([_llm_response()], calls_per_case=1)

        stop_state: dict[str, Any] = {}
        results = run_comparison(
            docs=docs, cases=cases, llm_agent=agent, verbose=False,
            max_cost_usd=1e-9, stop_state=stop_state,
        )
        # Cap checked before each case, so the first case always starts.
        assert len(results) == 1
        assert stop_state["reason"] == "cost_threshold"
        assert stop_state["n_incomplete"] == 0

    def test_incomplete_usage_stop_is_reported_distinctly(self):
        """An errored case makes the spend unaccountable.  With a cap enabled the
        run stops conservatively rather than continuing blind."""
        from scripts.compare_pipelines import run_comparison

        docs, cases = _tiny_comparison_inputs(4)
        agent = _AuditedScriptedAgent(
            [_llm_response(), RuntimeError("APIConnectionError: boom")],
            calls_per_case=2,
        )

        stop_state: dict[str, Any] = {}
        results = run_comparison(
            docs=docs, cases=cases, llm_agent=agent, verbose=False,
            max_cost_usd=100.0, stop_state=stop_state,
        )
        assert len(results) == 1
        assert stop_state["reason"] == "incomplete_usage"
        assert stop_state["reason"] != "cost_threshold", (
            "losing the ability to account for spend is a different stop reason "
            "from reaching the threshold"
        )
        assert stop_state["n_incomplete"] == 1

    def test_no_cap_never_stops_early(self):
        from scripts.compare_pipelines import run_comparison

        docs, cases = _tiny_comparison_inputs(4)
        agent = _AuditedScriptedAgent([RuntimeError("boom")], calls_per_case=1)

        results = run_comparison(
            docs=docs, cases=cases, llm_agent=agent, verbose=False, max_cost_usd=None,
        )
        assert len(results) == len(cases), (
            "without a cap every case runs, failures included"
        )


# ---------------------------------------------------------------------------
# run_comparison robustness (drives the real code path with a stub agent)
# ---------------------------------------------------------------------------

def _tiny_comparison_inputs(n_cases: int):
    """A two-document corpus plus n_cases questions, enough to drive the loop."""
    import dataclasses

    from bioevidence.corpus import CorpusDocument

    docs = []
    for pmid in ("1001", "1002"):
        abstract = f"RESULTS: intervention {pmid} changed the outcome."
        # The corpus validates its own content hash on read, so build the document
        # with the hash it will be checked against rather than an approximation.
        draft = CorpusDocument(
            pmid=pmid,
            title=f"Study {pmid}",
            abstract=abstract,
            doi=None,
            publication_types=("Journal Article",),
            journal="Test Journal",
            year=2020,
            first_author="Author A",
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            content_sha256="",
        )
        docs.append(dataclasses.replace(
            draft, content_sha256=draft.compute_content_sha256()
        ))
    cases = [
        {"case_id": f"C{i}", "question": f"Question {i}?",
         "gold_label": "yes", "gold_pmid": "1001"}
        for i in range(n_cases)
    ]
    return docs, cases


def _response(*, cost: float | None, status: str = "completed",
              coverage: str = "confirmed", verdict: str = YES) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "model_api_cost_usd": cost,
        "answer_model": {"usage_coverage": coverage},
        "retrieval_config": {"steps_used": 3},
        "agent_run": {"run_status": status, "termination_reason": "finish tool accepted."},
    }
    return {"verdict": verdict, "abstained": False, "citations": [], "provenance": provenance}


class _ScriptedAgent:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self._index = 0

    def run(self, *, question: str, request_id: str | None = None) -> dict[str, Any]:
        item = self._responses[self._index % len(self._responses)]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item


# ── an agent stub that really goes through the audit wrapper ─────────────────

def _llm_response(*, prompt: int = 1042, completion: int = 111,
                  model: str = "deepseek-flash", finish: str = "tool_calls") -> Any:
    """A duck-typed ChatCompletion carrying the counts the audit reads."""
    usage = types.SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )
    call = types.SimpleNamespace(
        id="tc1",
        function=types.SimpleNamespace(name="search_literature",
                                       arguments='{"query": "statins"}'),
    )
    message = types.SimpleNamespace(role="assistant", content=None, tool_calls=[call])

    def model_dump(*, exclude_none: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }],
        }
        return {k: v for k, v in payload.items() if v is not None} if exclude_none else payload

    message.model_dump = model_dump
    choice = types.SimpleNamespace(message=message, finish_reason=finish)
    return types.SimpleNamespace(id="r1", model=model, choices=[choice], usage=usage)


class _Transport:
    """Mimics the openai client shape that the audit wrapper delegates to."""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self._index = 0
        self.chat = self
        self.completions = self

    def create(self, **_kwargs: Any) -> Any:
        outcome = self._outcomes[min(self._index, len(self._outcomes) - 1)]
        self._index += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _AuditedScriptedAgent:
    """Stub agent whose round-trips pass through a real AuditingClient.

    Going through the genuine audit wrapper rather than returning canned dicts is
    what makes these tests cover the accounting that actually runs: the per-case
    request count, usage and known subtotal all come from captured round-trips.
    """

    def __init__(self, outcomes: list[Any], *, calls_per_case: int = 1,
                 verdict: str = YES) -> None:
        self._outcomes = list(outcomes)
        self._calls_per_case = calls_per_case
        self._verdict = verdict
        self._outcome_index = 0
        self._model = "deepseek-flash"
        self._max_tokens_per_call = 1024
        self._base_url = "https://api.deepseek.com/v1"
        self._client = AuditingClient(_Transport(self._outcomes))

    def run(self, *, question: str, request_id: str | None = None) -> dict[str, Any]:
        for _ in range(self._calls_per_case):
            outcome = self._outcomes[min(self._outcome_index, len(self._outcomes) - 1)]
            self._outcome_index += 1
            self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": question}],
                max_tokens=self._max_tokens_per_call,
                temperature=0.0,
                extra_body={"thinking": {"type": "disabled"}},
            )
            if isinstance(outcome, Exception):  # pragma: no cover - create() raises first
                raise outcome
        return {
            "verdict": self._verdict,
            "abstained": self._verdict == MAYBE,
            "citations": [],
            "provenance": {
                "model_api_cost_usd": None,
                "answer_model": {"usage_coverage": "confirmed"},
                "retrieval_config": {"steps_used": self._calls_per_case},
                "agent_run": {
                    "run_status": "completed",
                    "termination_reason": "finish tool accepted.",
                },
            },
        }


class TestRunComparisonRobustness:
    def test_unknown_cost_does_not_abort_the_run(self):
        """Regression: float(None) on model_api_cost_usd raised TypeError outside
        the per-case try block, killing the whole comparison on the first case
        whose usage was not confirmed."""
        from scripts.compare_pipelines import run_comparison
        docs, cases = _tiny_comparison_inputs(2)
        agent = _ScriptedAgent([
            _response(cost=None, coverage="unknown"),
            _response(cost=0.001),
        ])

        results = run_comparison(docs=docs, cases=cases, llm_agent=agent, verbose=False)

        assert len(results) == 2
        assert results[0].llm_cost_usd is None
        assert results[0].llm_usage_coverage == "unknown"
        assert results[1].llm_cost_usd == pytest.approx(0.001)

    def test_api_exception_is_recorded_and_the_run_continues(self):
        from scripts.compare_pipelines import run_comparison
        docs, cases = _tiny_comparison_inputs(3)
        agent = _ScriptedAgent([
            _response(cost=0.001),
            RuntimeError("APIConnectionError: boom"),
            _response(cost=0.002),
        ])

        results = run_comparison(docs=docs, cases=cases, llm_agent=agent, verbose=False)

        assert len(results) == 3
        assert results[1].llm_errored is True
        assert results[1].llm_run_status == "errored"
        assert results[1].llm_verdict == ""
        assert results[1].llm_cost_usd is None
        # Later cases still ran and were scored.
        assert results[2].llm_verdict == YES

    def test_rate_limit_error_is_recorded_as_errored_not_raised(self):
        import openai

        from scripts.compare_pipelines import run_comparison
        docs, cases = _tiny_comparison_inputs(1)
        error = openai.RateLimitError(
            "429", response=_FakeResponse(), body=None
        )
        agent = _ScriptedAgent([error])

        results = run_comparison(docs=docs, cases=cases, llm_agent=agent, verbose=False)

        assert results[0].llm_errored is True
        assert "RateLimitError" in (results[0].llm_error_msg or "")
        assert results[0].llm_run_status == "errored"


class _FakeResponse:
    status_code = 429
    headers: dict[str, str] = {}
    request = None

    def json(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Mid-run failure accounting — through the real agent loop
# ---------------------------------------------------------------------------

def _real_agent(outcomes: list[Any], docs: list[Any]):
    """A genuine LLMEvidenceAgent wired to a scripted transport.

    Used where the test must exercise the real ReAct loop: a stub that simply
    raises on its second call would never perform the successful first round,
    and so could not show whether that round's evidence survives.
    """
    from bioevidence.llm_agent import LLMEvidenceAgent
    from bioevidence.retrievers import BM25Retriever
    from bioevidence.tools import LiteratureTools

    agent = LLMEvidenceAgent.__new__(LLMEvidenceAgent)
    agent._tools = LiteratureTools(docs, retriever=BM25Retriever(docs))
    agent._model = "deepseek-flash"
    agent._max_steps = 8
    agent._max_tokens_per_call = 1024
    agent._base_url = "https://api.deepseek.com/v1"
    agent._client = AuditingClient(_Transport(outcomes))
    return agent


class TestMidRunFailureAccounting:
    """A case that dies part-way must still account for the rounds that ran."""

    def _run(self, outcomes, n_cases=1):
        from scripts.compare_pipelines import run_comparison
        docs, cases = _tiny_comparison_inputs(n_cases)
        agent = _real_agent(outcomes, docs)
        return run_comparison(docs=docs, cases=cases, llm_agent=agent, verbose=False)[0]

    def test_first_round_tokens_survive_a_second_round_failure(self):
        """Round 1 succeeds and reports usage; round 2 raises.  The first round's
        tokens, request record and cost estimate must all survive."""
        result = self._run([
            _llm_response(prompt=1042, completion=111),
            RuntimeError("APIConnectionError: connection reset"),
        ])

        assert result.llm_errored is True
        assert result.llm_run_status == "errored"
        # The complete cost is genuinely unknown.
        assert result.llm_cost_usd is None

        audit = result.llm_audit
        assert audit is not None
        assert audit["request_count"] == 2, "both round-trips must be recorded"
        assert audit["usage"]["prompt_tokens"] == 1042
        assert audit["usage"]["completion_tokens"] == 111
        assert audit["usage"]["coverage"] == "partial"
        assert audit["usage"]["rounds_with_usage"] == 1
        assert audit["usage"]["rounds_without_usage"] == 1
        assert len(audit["errors"]) == 1
        assert audit["errors"][0]["type"] == "RuntimeError"
        assert audit["tool_calls"][0]["name"] == "search_literature"

        # One coverage label everywhere: the audit, the case record and the
        # aggregate must agree that this run was only partly observed.
        assert result.llm_usage_coverage == "partial", (
            "a first round that succeeded must not be recorded as fully unknown"
        )
        from scripts.compare_pipelines import _compute_metrics
        counts = _compute_metrics([result])["llm_agent"]["cost"]["usage_coverage_counts"]
        assert counts == {"confirmed": 0, "partial": 1, "unknown": 0}

    def test_step_count_reflects_the_requests_actually_made(self):
        """The agent raised before reporting steps_used, so the count comes from
        the audit rather than being written as 'no steps ran'."""
        result = self._run([
            _llm_response(prompt=1042, completion=111),
            RuntimeError("APIConnectionError: connection reset"),
        ])
        assert result.llm_steps_used == 2
        assert result.llm_steps_source == "audit_request_count"

    def test_known_subtotal_is_reported_for_the_surviving_round(self):
        result = self._run([
            _llm_response(prompt=1042, completion=111),
            RuntimeError("APIConnectionError: connection reset"),
        ])
        subtotal = result.llm_cost_known_subtotal_usd
        assert subtotal is not None and subtotal > 0

    def test_failure_on_the_first_request_reports_no_known_subtotal(self):
        result = self._run([RuntimeError("APIConnectionError: refused")])
        assert result.llm_audit["request_count"] == 1
        assert result.llm_audit["usage"]["rounds_with_usage"] == 0
        assert result.llm_cost_known_subtotal_usd is None, (
            "no round reported usage, so there is nothing to subtotal"
        )
        # With nothing observed at all, every layer says unknown.
        assert result.llm_usage_coverage == "unknown"
        assert result.llm_audit["usage"]["coverage"] == "unknown"
        assert result.llm_steps_used == 1
        assert result.llm_steps_source == "audit_request_count"

    def test_later_cases_still_run_when_no_cap_is_set(self):
        """A failing case must not end the run when no cap is set."""
        from scripts.compare_pipelines import run_comparison
        docs, cases = _tiny_comparison_inputs(3)
        # Every request fails, so every case errors — but all three must still run.
        agent = _real_agent([RuntimeError("APIConnectionError: refused")], docs)
        results = run_comparison(docs=docs, cases=cases, llm_agent=agent, verbose=False)
        assert len(results) == 3
        assert all(r.llm_errored for r in results)
        assert all(r.llm_audit is not None for r in results)


# ---------------------------------------------------------------------------
# Per-case audit trail
# ---------------------------------------------------------------------------

class TestPerCaseAuditTrail:
    def _audit(self):
        from scripts.compare_pipelines import run_comparison
        docs, cases = _tiny_comparison_inputs(1)
        agent = _AuditedScriptedAgent([_llm_response()], calls_per_case=1)
        results = run_comparison(docs=docs, cases=cases, llm_agent=agent, verbose=False)
        return results[0].llm_audit

    def test_audit_carries_everything_needed_to_check_a_case(self):
        audit = self._audit()
        assert audit["request_count"] == 1
        assert audit["response_models"] == ["deepseek-flash"]
        assert audit["usage"]["prompt_tokens"] == 1042
        assert audit["usage"]["completion_tokens"] == 111
        assert audit["usage"]["coverage"] == "confirmed"
        assert [c["name"] for c in audit["tool_calls"]] == ["search_literature"]
        assert audit["finish_reasons"] == ["tool_calls"]

    def test_audit_records_the_request_configuration_actually_used(self):
        audit = self._audit()
        assert audit["request_max_tokens"] == 1024
        assert audit["request_temperature"] == 0.0
        assert audit["request_extra_body"] == {"thinking": {"type": "disabled"}}

    def test_unauditable_agent_falls_back_to_the_agents_own_cost(self):
        """With no client to wrap there is no per-round audit — but a confirmed
        cost from the agent is still a usable known subtotal.  Treating the
        missing audit as "cost unknown" is what stopped runs with no reason to
        stop, so the agent's own figure stands in."""
        from scripts.compare_pipelines import run_comparison
        docs, cases = _tiny_comparison_inputs(1)
        agent = _ScriptedAgent([_response(cost=0.001)])
        results = run_comparison(docs=docs, cases=cases, llm_agent=agent, verbose=False)

        assert results[0].llm_audit is None
        assert results[0].llm_cost_known_subtotal_usd == pytest.approx(0.001)
        assert results[0].llm_usage_coverage == "confirmed"
        assert results[0].llm_steps_source == "agent_provenance"


# ---------------------------------------------------------------------------
# Frozen sample verification
# ---------------------------------------------------------------------------

def _benchmark_dir(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    """A minimal but genuinely loadable benchmark directory."""
    from bioevidence.corpus import write_corpus

    docs, _ = _tiny_comparison_inputs(1)
    write_corpus(tmp_path / "corpus.jsonl", docs)
    (tmp_path / "train.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "test_inputs.jsonl").write_text(
        "".join(
            json.dumps({"case_id": r["case_id"], "question": "Q?"}) + "\n" for r in rows
        ),
        encoding="utf-8",
    )
    (tmp_path / "test_gold.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return tmp_path


class TestFrozenSampleVerification:
    def _rows(self) -> list[dict[str, str]]:
        return [
            {"case_id": "PUBMEDQA-TEST-0001", "pmid": "1", "label": "yes",
             "verdict": "supported"},
            {"case_id": "PUBMEDQA-TEST-0002", "pmid": "2", "label": "no",
             "verdict": "contradicted"},
        ]

    def test_matching_record_verifies(self, tmp_path):
        from scripts.compare_pipelines import _sha256_file, _verify_frozen_sample
        bdir = _benchmark_dir(tmp_path, self._rows())
        record = {
            "gold_sha256": _sha256_file(bdir / "test_gold.jsonl"),
            "case_ids": ["PUBMEDQA-TEST-0001", "PUBMEDQA-TEST-0002"],
            "label_by_case": {"PUBMEDQA-TEST-0001": "yes", "PUBMEDQA-TEST-0002": "no"},
        }
        result = _verify_frozen_sample(record, bdir)
        assert result["hash_matches"] is True
        assert result["n_case_ids"] == 2
        assert result["verified_before_model_requests"] is True

    def test_hash_mismatch_refuses_to_run(self, tmp_path):
        """The run must fail on the data, not adapt to it — the frozen sample is
        never rewritten to match whatever is on disk."""
        from scripts.compare_pipelines import _verify_frozen_sample
        bdir = _benchmark_dir(tmp_path, self._rows())
        record = {"gold_sha256": "0" * 64, "case_ids": ["PUBMEDQA-TEST-0001"]}
        with pytest.raises(ValueError, match="drawn from a different gold file"):
            _verify_frozen_sample(record, bdir)

    def test_duplicate_case_ids_are_rejected(self, tmp_path):
        from scripts.compare_pipelines import _verify_frozen_sample
        bdir = _benchmark_dir(tmp_path, self._rows())
        record = {"case_ids": ["PUBMEDQA-TEST-0001", "PUBMEDQA-TEST-0001"]}
        with pytest.raises(ValueError, match="duplicate case IDs"):
            _verify_frozen_sample(record, bdir)

    def test_label_disagreement_is_rejected(self, tmp_path):
        from scripts.compare_pipelines import _verify_frozen_sample
        bdir = _benchmark_dir(tmp_path, self._rows())
        record = {
            "case_ids": ["PUBMEDQA-TEST-0001"],
            "label_by_case": {"PUBMEDQA-TEST-0001": "no"},
        }
        with pytest.raises(ValueError, match="labels disagree"):
            _verify_frozen_sample(record, bdir)

    def test_unknown_case_id_is_rejected(self, tmp_path):
        from scripts.compare_pipelines import _verify_frozen_sample
        bdir = _benchmark_dir(tmp_path, self._rows())
        record = {"case_ids": ["PUBMEDQA-TEST-9999"]}
        with pytest.raises(ValueError, match="absent from test_gold"):
            _verify_frozen_sample(record, bdir)

    def test_main_verifies_before_building_the_agent(self, tmp_path, monkeypatch):
        """A bad sample must abort the run before any model request is possible."""
        import scripts.compare_pipelines as cp
        bdir = _benchmark_dir(tmp_path, self._rows())
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps({
            "case_ids": ["PUBMEDQA-TEST-0001"], "gold_sha256": "0" * 64,
        }), encoding="utf-8")

        reached: list[str] = []
        monkeypatch.setattr(cp, "LLMEvidenceAgent",
                            lambda *a, **k: reached.append("agent") or object())
        monkeypatch.setattr(cp, "run_comparison",
                            lambda **k: reached.append("run") or [])

        monkeypatch.setattr(sys, "argv", [
            "compare_pipelines", "--benchmark-dir", str(bdir),
            "--sample-file", str(sample_path), "--deepseek-key", "sk-test-offline",
            "--output", str(tmp_path / "r.json"),
        ])
        with pytest.raises(ValueError, match="drawn from a different gold file"):
            cp.main()
        assert reached == [], "no agent may be constructed before verification passes"


# ---------------------------------------------------------------------------
# Entry path: real main(), real agent construction, only the transport replaced
# ---------------------------------------------------------------------------

def _synthetic_benchmark(tmp_path: Path, case_ids: list[str]) -> Path:
    """A loadable benchmark dir with the fixed 500-row train split the answerer demands."""
    import dataclasses

    from bioevidence.corpus import CorpusDocument, write_corpus

    def _doc(pmid: str, topic: str):
        abstract = f"RESULTS: {topic} changed the outcome in study {pmid}."
        draft = CorpusDocument(
            pmid=pmid, title=f"Study of {topic}", abstract=abstract, doi=None,
            publication_types=("Journal Article",), journal="Test Journal", year=2020,
            first_author="Author A",
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", content_sha256="",
        )
        return dataclasses.replace(draft, content_sha256=draft.compute_content_sha256())

    docs = [_doc("1001", "statins"), _doc("1002", "aspirin"), _doc("1003", "metformin")]
    write_corpus(tmp_path / "corpus.jsonl", docs)
    labels = ["yes", "no", "maybe"]
    train = [
        {"pmid": docs[i % 3].pmid, "question": f"Does treatment {i} affect outcome {i % 7}?",
         "label": labels[i % 3]}
        for i in range(500)
    ]
    (tmp_path / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in train), encoding="utf-8"
    )
    (tmp_path / "test_inputs.jsonl").write_text(
        "".join(json.dumps({"case_id": c, "question": f"Does treatment affect outcome? {c}"}) + "\n"
                for c in case_ids), encoding="utf-8"
    )
    (tmp_path / "test_gold.jsonl").write_text(
        "".join(json.dumps({"case_id": c, "pmid": "1001", "label": "yes",
                            "verdict": "supported"}) + "\n" for c in case_ids),
        encoding="utf-8",
    )
    return tmp_path


def _tool_response(name: str, arguments: dict[str, Any], *,
                   prompt: int = 1000, completion: int = 100, with_usage: bool = True) -> Any:
    call = types.SimpleNamespace(
        id=f"tc-{name}", function=types.SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )
    message = types.SimpleNamespace(role="assistant", content=None, tool_calls=[call])
    message.model_dump = lambda **k: {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": call.id, "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)}}],
    }
    usage = (
        types.SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              total_tokens=prompt + completion)
        if with_usage else None
    )
    return types.SimpleNamespace(
        id="r", model="deepseek-flash",
        choices=[types.SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=usage,
    )


def _finish_call(pmid: str = "1001", *, with_usage: bool = True,
                 verdict: str = YES) -> Any:
    return _tool_response("finish", {
        "verdict": verdict, "answer": "Yes.", "claim": "It changed the outcome.",
        "cited_pmids": [pmid], "decisive_reason": f"PMID {pmid} reports it.",
    }, with_usage=with_usage)


def _completing_plan(case_id: str, pmid: str = "1001", *, with_usage: bool = True,
                     verdict: str = YES) -> list[Any]:
    """The four round-trips a case needs to reach an accepted finish."""
    return [
        _tool_response("search_literature", {"query": "treatment outcome", "top_k": 3},
                       with_usage=with_usage),
        _tool_response("fetch_record", {"pmid": pmid}, with_usage=with_usage),
        _tool_response("inspect_evidence", {"pmid": pmid, "query": "treatment outcome"},
                       with_usage=with_usage),
        _finish_call(pmid, with_usage=with_usage, verdict=verdict),
    ]


class _ScriptedTransport:
    """Serves a fixed queue of responses; stands in for the SDK transport only."""

    def __init__(self, queue: list[Any]) -> None:
        self.queue = queue
        self.chat = self
        self.completions = self

    def create(self, **_kwargs: Any) -> Any:
        if not self.queue:
            raise AssertionError("scripted transport ran out of responses")
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _install_transport(monkeypatch, queue, baseline_queue=None,
                       arm_c_queue=None) -> list[dict[str, Any]]:
    """Patch only the SDK client class; agent construction and the runner stay real.

    Routing is by what each arm sends:

    * Arm A and Arm C both send ``tools``, so they are told apart by the system
      message — C's carries the mixed clarification and A's does not.
    * Arm B sends no ``tools`` at all, which is also how the tests prove B really
      is the no-tool arm.
    """
    import openai

    from bioevidence.prompt_variants import ARM_C_MIXED_RULES

    captured: list[dict[str, Any]] = []
    agent_transport = _ScriptedTransport(queue)
    baseline_transport = _ScriptedTransport(baseline_queue or [])
    arm_c_transport = _ScriptedTransport(arm_c_queue or [])

    def _system_text(kwargs: dict[str, Any]) -> str:
        for message in kwargs.get("messages") or []:
            if isinstance(message, dict) and message.get("role") == "system":
                return str(message.get("content") or "")
        return ""

    class _RoutingTransport:
        def create(self, **kwargs: Any) -> Any:
            if "tools" not in kwargs:
                if baseline_queue is None:
                    raise AssertionError(
                        "a request without tool schemas was made but no baseline "
                        "queue was supplied — Arm B ran when it should not have"
                    )
                return baseline_transport.create(**kwargs)
            if ARM_C_MIXED_RULES in _system_text(kwargs):
                if arm_c_queue is None:
                    raise AssertionError(
                        "Arm C sent a request but no arm_c queue was supplied"
                    )
                return arm_c_transport.create(**kwargs)
            return agent_transport.create(**kwargs)

    routing = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_RoutingTransport()))

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)
            self.chat = routing.chat

    monkeypatch.setattr(openai, "OpenAI", _Client)
    return captured


def _default_sample(bdir: Path, case_ids: list[str]) -> dict[str, Any]:
    """The sample object the entry helpers write by default."""
    return {
        "seed": 20260729,
        "total": len(case_ids),
        "case_ids": list(case_ids),
        # The synthetic benchmark's gold is all "yes" with pmid 1001, so the
        # sample carries the same labels the gold file does.
        "label_by_case": {case_id: "yes" for case_id in case_ids},
        # The real frozen sample records this, and the identity check treats a
        # missing value as a conflict rather than as agreement.
        "gold_sha256": hashlib.sha256((bdir / "test_gold.jsonl").read_bytes()).hexdigest(),
    }


def _entry_setup(tmp_path: Path, case_ids: list[str], *, sample_name: str = "sample.json"):
    """Create the benchmark and sample without running anything.

    Separate from driving main() so a test can alter the sample file on disk and
    then run against it, instead of having its change overwritten.
    """
    bdir = _synthetic_benchmark(tmp_path, list(case_ids))
    sample_path = tmp_path / sample_name
    sample_path.write_text(
        json.dumps(_default_sample(bdir, case_ids)), encoding="utf-8"
    )
    return bdir, sample_path


def _drive_main(tmp_path: Path, monkeypatch, *, queue, sample_path, output_name,
                extra_argv=(), baseline_queue=None, arm_c_queue=None):
    """Run the real main() against whatever is currently on disk."""
    import scripts.compare_pipelines as cp

    output = tmp_path / output_name
    _install_transport(monkeypatch, queue, baseline_queue, arm_c_queue)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-offline")
    monkeypatch.setattr(sys, "argv", [
        "compare_pipelines", "--benchmark-dir", str(tmp_path),
        "--sample-file", str(sample_path), "--model", "deepseek-flash",
        "--timeout", "30", "--max-retries", "0", "--output", str(output),
        *extra_argv,
    ])
    code = cp.main()
    return code, output


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_entry(tmp_path: Path, monkeypatch, *, case_ids, queue, extra_argv=(),
               output_name: str = "report.json", sample_name: str = "sample.json",
               baseline_queue=None, arm_c_queue=None):
    """Drive the real main() and return (exit code, the JSON it actually wrote)."""
    _, sample_path = _entry_setup(tmp_path, case_ids, sample_name=sample_name)
    code, output = _drive_main(
        tmp_path, monkeypatch, queue=queue, sample_path=sample_path,
        output_name=output_name, extra_argv=extra_argv, baseline_queue=baseline_queue,
        arm_c_queue=arm_c_queue,
    )
    return code, _read_json(output)


def _run_merge(tmp_path: Path, monkeypatch, *, segments, sample_path, output_name):
    import scripts.compare_pipelines as cp

    argv = ["compare_pipelines", "--sample-file", str(sample_path)]
    for segment in segments:
        argv += ["--merge-from", str(segment)]
    output = tmp_path / output_name
    argv += ["--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    code = cp.main()
    return code, _read_json(output)


class TestEntryPathAuditWiring:
    """Only the transport is replaced; construction, runner and entry point are real."""

    def test_five_completed_cases_run_in_full_with_no_cap_stop(self, tmp_path, monkeypatch):
        """The omission Codex found: on the normal path nothing wrapped the client,
        so every case reported llm_audit=None and the spending cap then treated a
        fully confirmed case as unaccountable, stopping the run after one case."""
        case_ids = [f"C{i}" for i in range(5)]
        queue = [r for c in case_ids for r in _completing_plan(c)]

        code, report = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids, queue=queue,
            extra_argv=["--max-cost-usd", "0.05"],
        )

        assert code == 0, "a run under its cap must not be reported as partial"
        assert report["plan"]["attempted"] == 5
        assert report["plan"]["not_run"] == 0
        assert report["plan"]["partial"] is False
        assert report["plan"]["stop_reason"] is None

        for case in report["cases"]:
            assert case["llm_run_status"] == "completed", case["llm_run_status"]
            assert case["llm_audit"] is not None, (
                "the normal entry path must attach the audit wrapper itself"
            )
            assert case["llm_audit"]["request_count"] == 4
            assert case["llm_audit"]["response_models"] == ["deepseek-flash"]
            assert case["llm_cost_known_subtotal_usd"] is not None
            assert case["llm_cost_status"] == "confirmed"

        assert report["metrics"]["llm_agent"]["cost"]["cost_complete"] is True

    def test_recorded_config_reflects_the_real_construction(self, tmp_path, monkeypatch):
        case_ids = ["C0"]
        queue = _completing_plan("C0")
        code, report = _run_entry(tmp_path, monkeypatch, case_ids=case_ids, queue=queue)

        assert code == 0
        config = report["config"]
        assert config["base_url"] == "https://api.deepseek.com/v1"
        assert config["max_tokens_per_call"] == 1024
        assert config["request_timeout_seconds"] == 30
        assert config["sdk_max_retries"] == 0
        assert config["temperature"] == 0.0

    def test_exit_is_zero_and_report_written_when_nothing_stops(self, tmp_path, monkeypatch):
        case_ids = ["C0", "C1"]
        queue = [r for c in case_ids for r in _completing_plan(c)]
        code, report = _run_entry(tmp_path, monkeypatch, case_ids=case_ids, queue=queue)

        assert code == 0
        assert report["plan"]["completed"] == 2
        assert "input_hashes" in report and report["input_hashes"]["files"]


class TestEntryPathStops:
    """Both stop conditions, through the real entry point, with the JSON read back."""

    def test_first_case_without_usage_stops_conservatively(self, tmp_path, monkeypatch):
        case_ids = [f"C{i}" for i in range(3)]
        # Case 0 completes but reports no usage at all.
        queue = _completing_plan("C0", with_usage=False)

        code, report = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids, queue=queue,
            extra_argv=["--max-cost-usd", "0.05"],
        )

        assert code == 3, "a partial run must exit non-zero"
        assert report["plan"]["partial"] is True
        assert report["plan"]["attempted"] == 1
        assert report["plan"]["not_run"] == 2
        assert report["plan"]["stop_detail"]["reason"] == "incomplete_usage"
        assert "usage could not be accounted for" in report["plan"]["stop_reason"]
        assert report["cases"][0]["llm_cost_known_subtotal_usd"] is None
        assert report["metrics"]["llm_agent"]["cost"]["cost_complete"] is False

    def test_cost_threshold_stop_is_a_different_reason(self, tmp_path, monkeypatch):
        case_ids = [f"C{i}" for i in range(3)]
        queue = [r for c in case_ids for r in _completing_plan(c)]

        code, report = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids, queue=queue,
            extra_argv=["--max-cost-usd", "1e-9"],
        )

        assert code == 3
        assert report["plan"]["attempted"] == 1
        assert report["plan"]["stop_detail"]["reason"] == "cost_threshold"
        assert report["plan"]["stop_detail"]["n_incomplete"] == 0
        assert "max_cost_usd reached" in report["plan"]["stop_reason"]
        # Distinct from the incomplete-usage stop, even though both exit 3.
        assert "usage could not be accounted for" not in report["plan"]["stop_reason"]

    def test_partial_report_is_written_before_the_nonzero_exit(self, tmp_path, monkeypatch):
        case_ids = [f"C{i}" for i in range(3)]
        queue = _completing_plan("C0", with_usage=False)
        code, report = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids, queue=queue,
            extra_argv=["--max-cost-usd", "0.05"],
        )
        assert code == 3
        # The report exists and accounts for what did run.
        assert report["plan"]["planned"] == 3
        assert len(report["cases"]) == 1
        assert report["cases"][0]["llm_audit"] is not None


# ---------------------------------------------------------------------------
# Continuation and merge
# ---------------------------------------------------------------------------

class TestContinuation:
    """A second segment must run only what the first did not."""

    def test_continuation_excludes_the_attempted_cases(self, tmp_path, monkeypatch):
        case_ids = [f"C{i:02d}" for i in range(6)]
        first = case_ids[:2]
        code1, seg1 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in first for r in _completing_plan(c)],
            extra_argv=["--n-cases", "2"], output_name="seg1.json",
        )
        assert code1 == 0
        assert [c["case_id"] for c in seg1["cases"]] == first

        rest = case_ids[2:]
        code2, seg2 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in rest for r in _completing_plan(c)],
            extra_argv=["--exclude-from", str(tmp_path / "seg1.json")],
            output_name="seg2.json",
        )

        assert code2 == 0
        assert [c["case_id"] for c in seg2["cases"]] == rest, (
            "the continuation must run exactly the cases the first segment did not"
        )
        assert not ({c["case_id"] for c in seg1["cases"]}
                    & {c["case_id"] for c in seg2["cases"]})
        assert seg2["plan"]["planned"] == len(rest)
        assert seg2["continuation"]["n_excluded"] == len(first)
        assert seg2["continuation"]["carry_over_cost_usd"] > 0

    def test_n_cases_applies_after_exclusion_not_before(self, tmp_path, monkeypatch):
        """Slicing the frozen sample first would select exactly the cases the
        earlier segment already ran, leaving nothing to do."""
        case_ids = [f"C{i:02d}" for i in range(6)]
        first = case_ids[:2]
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in first for r in _completing_plan(c)],
            extra_argv=["--n-cases", "2"], output_name="seg1.json",
        )

        code, seg2 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids[2:4] for r in _completing_plan(c)],
            extra_argv=["--exclude-from", str(tmp_path / "seg1.json"), "--n-cases", "2"],
            output_name="seg2.json",
        )
        assert code == 0
        assert [c["case_id"] for c in seg2["cases"]] == case_ids[2:4]

    def test_nothing_left_to_run_exits_without_requests(self, tmp_path, monkeypatch):
        case_ids = [f"C{i:02d}" for i in range(3)]
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids for r in _completing_plan(c)],
            output_name="seg1.json",
        )
        # An empty queue would make the transport raise if any request were made.
        # Nothing is left to run, so no segment report is produced — main()
        # returns before the run, and nothing reads a file that was never written.
        import scripts.compare_pipelines as cp
        _install_transport(monkeypatch, [])
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-offline")
        monkeypatch.setattr(sys, "argv", [
            "compare_pipelines", "--benchmark-dir", str(tmp_path),
            "--sample-file", str(tmp_path / "sample.json"),
            "--exclude-from", str(tmp_path / "seg1.json"),
            # Same run configuration as the first segment, so the identity check
            # has nothing to object to and the case list is the only variable.
            "--model", "deepseek-flash", "--timeout", "30", "--max-retries", "0",
            "--output", str(tmp_path / "seg2.json"),
        ])
        assert cp.main() == 0
        assert not (tmp_path / "seg2.json").exists()


class TestMerge:
    def _two_segments(self, tmp_path, monkeypatch, total: int = 6, split: int = 2):
        case_ids = [f"C{i:02d}" for i in range(total)]
        first, second = case_ids[:split], case_ids[split:]
        _, seg1 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in first for r in _completing_plan(c)],
            extra_argv=["--n-cases", str(split)], output_name="seg1.json",
        )
        _, seg2 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in second for r in _completing_plan(c)],
            extra_argv=["--exclude-from", str(tmp_path / "seg1.json")],
            output_name="seg2.json",
        )
        return case_ids, tmp_path / "seg1.json", tmp_path / "seg2.json", seg1, seg2

    def test_full_continuation_covers_every_frozen_case_exactly_once(
        self, tmp_path, monkeypatch
    ):
        """The shape of the real run: 50 frozen cases taken in two segments."""
        case_ids = [f"C{i:02d}" for i in range(50)]
        first, second = case_ids[:5], case_ids[5:]
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in first for r in _completing_plan(c)],
            extra_argv=["--n-cases", "5"], output_name="seg1.json",
        )
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in second for r in _completing_plan(c)],
            extra_argv=["--exclude-from", str(tmp_path / "seg1.json")],
            output_name="seg2.json",
        )

        code, merged = _run_merge(
            tmp_path, monkeypatch,
            segments=[tmp_path / "seg1.json", tmp_path / "seg2.json"],
            sample_path=tmp_path / "sample.json", output_name="merged.json",
        )

        assert code == 0, "every frozen case is covered, so nothing is left unrun"
        assert merged["plan"]["planned"] == 50
        assert merged["plan"]["attempted"] == 50
        assert merged["plan"]["not_run"] == 0
        assert merged["plan"]["partial"] is False
        ids = [c["case_id"] for c in merged["cases"]]
        assert len(ids) == 50
        assert len(set(ids)) == 50, "no question may be counted twice"
        assert ids == case_ids, "cases are restored to frozen order"

    def test_metrics_are_recomputed_not_averaged(self, tmp_path, monkeypatch):
        case_ids, seg1_path, seg2_path, seg1, seg2 = self._two_segments(tmp_path, monkeypatch)

        _, merged = _run_merge(
            tmp_path, monkeypatch, segments=[seg1_path, seg2_path],
            sample_path=tmp_path / "sample.json", output_name="merged.json",
        )

        from scripts.compare_pipelines import _case_result_from_report_case, _compute_metrics
        rebuilt = [_case_result_from_report_case(c)[0] for c in merged["cases"]]
        expected = _compute_metrics(rebuilt)
        assert merged["metrics"] == expected
        assert merged["merge_notes"]["metrics_recomputed_from_cases"] is True

        # And it really is a recomputation, not an average of the two segments.
        seg1_f1 = seg1["metrics"]["llm_agent"]["macro_f1"]
        seg2_f1 = seg2["metrics"]["llm_agent"]["macro_f1"]
        if seg1_f1 is not None and seg2_f1 is not None and seg1_f1 != seg2_f1:
            assert merged["metrics"]["llm_agent"]["macro_f1"] != pytest.approx(
                (seg1_f1 + seg2_f1) / 2
            ), "a plain average of segment metrics would weight segments equally"

    def test_config_conflict_is_refused(self, tmp_path, monkeypatch):
        case_ids, seg1_path, seg2_path, _, _ = self._two_segments(tmp_path, monkeypatch)

        tampered = json.loads(seg2_path.read_text(encoding="utf-8"))
        tampered["config"]["model"] = "some-other-model"
        tampered_path = tmp_path / "tampered.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")

        with pytest.raises(ValueError, match="does not describe the same experiment"):
            _run_merge(
                tmp_path, monkeypatch, segments=[seg1_path, tampered_path],
                sample_path=tmp_path / "sample.json", output_name="merged.json",
            )

    def test_input_hash_conflict_is_refused(self, tmp_path, monkeypatch):
        case_ids, seg1_path, seg2_path, _, _ = self._two_segments(tmp_path, monkeypatch)

        tampered = json.loads(seg2_path.read_text(encoding="utf-8"))
        tampered["input_hashes"]["files"]["test_gold.jsonl"] = "0" * 64
        tampered_path = tmp_path / "tampered.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")

        with pytest.raises(ValueError, match="does not describe the same experiment"):
            _run_merge(
                tmp_path, monkeypatch, segments=[seg1_path, tampered_path],
                sample_path=tmp_path / "sample.json", output_name="merged.json",
            )

    def test_duplicate_case_across_segments_is_refused(self, tmp_path, monkeypatch):
        case_ids, seg1_path, _, _, _ = self._two_segments(tmp_path, monkeypatch)
        # Merge the same segment twice: every case would be counted twice.
        with pytest.raises(ValueError, match="duplicate case_id"):
            _run_merge(
                tmp_path, monkeypatch, segments=[seg1_path, seg1_path],
                sample_path=tmp_path / "sample.json", output_name="merged.json",
            )

    def test_case_outside_the_frozen_sample_is_refused(self, tmp_path, monkeypatch):
        case_ids, seg1_path, _, _, _ = self._two_segments(tmp_path, monkeypatch)

        tampered = json.loads(seg1_path.read_text(encoding="utf-8"))
        tampered["cases"][0]["case_id"] = "NOT-IN-THE-SAMPLE"
        tampered_path = tmp_path / "tampered.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")

        with pytest.raises(ValueError, match="not in the frozen sample"):
            _run_merge(
                tmp_path, monkeypatch, segments=[tampered_path],
                sample_path=tmp_path / "sample.json", output_name="merged.json",
            )

    def test_partial_coverage_reports_not_run_separately_from_failed(
        self, tmp_path, monkeypatch
    ):
        """A case that never ran is not a case that ran and failed."""
        case_ids = [f"C{i:02d}" for i in range(6)]
        # Segment one runs two cases, one of which errors.
        queue = _completing_plan("C00") + [RuntimeError("APIConnectionError: boom")]
        _, seg1 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids, queue=queue,
            extra_argv=["--n-cases", "2"], output_name="seg1.json",
        )
        assert seg1["plan"]["failed"] == 1

        code, merged = _run_merge(
            tmp_path, monkeypatch, segments=[tmp_path / "seg1.json"],
            sample_path=tmp_path / "sample.json", output_name="merged.json",
        )

        assert code == 3, "4 of 6 frozen cases have no segment report yet"
        plan = merged["plan"]
        assert plan["planned"] == 6
        assert plan["attempted"] == 2
        assert plan["failed"] == 1
        assert plan["not_run"] == 4
        assert plan["partial"] is True
        assert plan["not_covered_case_ids"] == case_ids[2:]

    def test_merged_report_keeps_the_baseline_arm(self, tmp_path, monkeypatch):
        """A segmented v2 run must not lose Arm B when the segments are merged."""
        case_ids, seg1_path, seg2_path, _, _ = self._two_segments(tmp_path, monkeypatch)

        for path in (seg1_path, seg2_path):
            report = json.loads(path.read_text(encoding="utf-8"))
            for case in report["cases"]:
                case.update({
                    "baseline_verdict": "contradicted",
                    "baseline_run_status": "completed",
                    "baseline_steps": 1,
                    "baseline_steps_source": "agent_provenance",
                    "baseline_citation_hit": False,
                    "baseline_latency_ms": 100.0,
                    "baseline_error_flag": False,
                    "baseline_cost_usd": 0.0007,
                    "baseline_cost_status": "confirmed",
                    "baseline_cost_known_subtotal_usd": 0.0007,
                    "baseline_audit": {"request_count": 1, "tool_calls": []},
                })
            path.write_text(json.dumps(report), encoding="utf-8")

        _, merged = _run_merge(
            tmp_path, monkeypatch, segments=[seg1_path, seg2_path],
            sample_path=tmp_path / "sample.json", output_name="merged.json",
        )

        base = merged["metrics"]["baseline"]
        assert base is not None, "Arm B was run; the merge must not drop it"
        assert base["n_completed"] == len(case_ids)
        assert base["completion_rate"] == 1.0
        assert base["avg_steps_used"] == 1.0
        assert all(c["baseline_verdict"] == "contradicted" for c in merged["cases"])

    def test_merged_report_without_a_baseline_arm_reports_it_as_null(
        self, tmp_path, monkeypatch
    ):
        case_ids, seg1_path, seg2_path, _, _ = self._two_segments(tmp_path, monkeypatch)
        _, merged = _run_merge(
            tmp_path, monkeypatch, segments=[seg1_path, seg2_path],
            sample_path=tmp_path / "sample.json", output_name="merged.json",
        )
        assert merged["metrics"]["baseline"] is None

    def test_absent_and_derived_fields_are_reported_separately(self, tmp_path, monkeypatch):
        case_ids, seg1_path, seg2_path, _, _ = self._two_segments(tmp_path, monkeypatch)

        # Strip a field a merged report must not invent.
        stripped = json.loads(seg1_path.read_text(encoding="utf-8"))
        for case in stripped["cases"]:
            case.pop("llm_latency_ms", None)
        stripped_path = tmp_path / "stripped.json"
        stripped_path.write_text(json.dumps(stripped), encoding="utf-8")

        _, merged = _run_merge(
            tmp_path, monkeypatch, segments=[stripped_path, seg2_path],
            sample_path=tmp_path / "sample.json", output_name="merged.json",
        )
        notes = merged["merge_notes"]
        for case in stripped["cases"]:
            assert "llm_latency_ms" in notes["cases_with_absent_fields"][case["case_id"]]
        # A latency that was never recorded is not invented from an average.
        latencies = [
            c.get("llm_latency_ms") for c in merged["cases"]
            if c["case_id"] in {x["case_id"] for x in stripped["cases"]}
        ]
        assert latencies and all(value is None for value in latencies)


class TestSampleFileIsBoundToTheExperiment:
    """The actual --sample-file must match what the earlier report recorded."""

    def _seg1(self, tmp_path, monkeypatch, case_ids):
        first = case_ids[:2]
        return _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in first for r in _completing_plan(c)],
            extra_argv=["--n-cases", "2"], output_name="seg1.json",
        )

    def _continue_with(self, tmp_path, monkeypatch, sample_obj, output_name="seg2.json"):
        """Continue using an altered sample file, written in place.

        main() is driven directly rather than through _run_entry, which would
        write its own sample and overwrite the alteration.
        """
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps(sample_obj), encoding="utf-8")
        return _drive_main(
            tmp_path, monkeypatch, queue=[], sample_path=sample_path,
            output_name=output_name,
            extra_argv=["--exclude-from", str(tmp_path / "seg1.json")],
        )

    def setup_method(self):
        self._case_ids = [f"C{i:02d}" for i in range(6)]

    def _sample(self, tmp_path) -> dict[str, Any]:
        return json.loads((tmp_path / "sample.json").read_text(encoding="utf-8"))

    def test_original_sample_still_continues_normally(self, tmp_path, monkeypatch):
        self._seg1(tmp_path, monkeypatch, self._case_ids)
        code, seg2 = _run_entry(
            tmp_path, monkeypatch, case_ids=self._case_ids,
            queue=[r for c in self._case_ids[2:] for r in _completing_plan(c)],
            extra_argv=["--exclude-from", str(tmp_path / "seg1.json")],
            output_name="seg2.json",
        )
        assert code == 0
        assert [c["case_id"] for c in seg2["cases"]] == self._case_ids[2:]

    def test_reordered_case_ids_are_rejected(self, tmp_path, monkeypatch):
        """Same 50 IDs in a different order is a different experiment."""
        self._seg1(tmp_path, monkeypatch, self._case_ids)
        tampered = self._sample(tmp_path)
        tampered["case_ids"] = list(reversed(tampered["case_ids"]))

        with pytest.raises(ValueError, match="case_ids_sha256"):
            self._continue_with(tmp_path, monkeypatch, tampered)

    def test_changed_seed_is_rejected(self, tmp_path, monkeypatch):
        self._seg1(tmp_path, monkeypatch, self._case_ids)
        tampered = self._sample(tmp_path)
        tampered["seed"] = 1

        with pytest.raises(ValueError, match="sample.seed"):
            self._continue_with(tmp_path, monkeypatch, tampered)

    def test_changed_total_is_rejected(self, tmp_path, monkeypatch):
        self._seg1(tmp_path, monkeypatch, self._case_ids)
        tampered = self._sample(tmp_path)
        tampered["total"] = 99

        with pytest.raises(ValueError, match="sample.total"):
            self._continue_with(tmp_path, monkeypatch, tampered)

    def test_rejection_happens_before_any_model_request(self, tmp_path, monkeypatch):
        """The queue is empty, so a request would raise; refusal must come first."""
        self._seg1(tmp_path, monkeypatch, self._case_ids)
        tampered = self._sample(tmp_path)
        tampered["seed"] = 1

        with pytest.raises(ValueError):
            self._continue_with(tmp_path, monkeypatch, tampered, output_name="never.json")
        assert not (tmp_path / "never.json").exists()

    def test_missing_identity_value_is_a_conflict_not_a_match(self, tmp_path, monkeypatch):
        """A field neither side recorded cannot be verified, so it cannot pass."""
        self._seg1(tmp_path, monkeypatch, self._case_ids)
        tampered = self._sample(tmp_path)
        del tampered["gold_sha256"]

        with pytest.raises(ValueError, match="gold_sha256"):
            self._continue_with(tmp_path, monkeypatch, tampered)

    def test_merge_rejects_a_segment_from_another_sample_file(self, tmp_path, monkeypatch):
        case_ids = self._case_ids
        _, _ = self._seg1(tmp_path, monkeypatch, case_ids)
        other = tmp_path / "other_sample.json"
        other.write_text(json.dumps({
            "seed": 999, "total": len(case_ids), "case_ids": case_ids,
            "label_by_case": {c: "yes" for c in case_ids},
            "gold_sha256": json.loads(
                (tmp_path / "sample.json").read_text(encoding="utf-8")
            )["gold_sha256"],
        }), encoding="utf-8")

        with pytest.raises(ValueError, match="supplied --sample-file"):
            _run_merge(
                tmp_path, monkeypatch, segments=[tmp_path / "seg1.json"],
                sample_path=other, output_name="merged.json",
            )


class TestExperimentBudgetPreflight:
    """Carried-over spend is checked before the first new case starts."""

    def test_carried_spend_at_the_cap_makes_no_requests(self, tmp_path, monkeypatch):
        case_ids = [f"C{i:02d}" for i in range(4)]
        _, seg1 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids[:2] for r in _completing_plan(c)],
            extra_argv=["--n-cases", "2"], output_name="seg1.json",
        )
        spent = sum(
            c["llm_cost_known_subtotal_usd"] for c in seg1["cases"]
            if c["llm_cost_known_subtotal_usd"] is not None
        )
        assert spent > 0

        # An empty queue: any model request at all would raise.
        code, seg2 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids, queue=[],
            extra_argv=["--exclude-from", str(tmp_path / "seg1.json"),
                        "--experiment-budget-usd", f"{spent / 2}"],
            output_name="seg2.json",
        )

        assert code == 3
        assert seg2["plan"]["attempted"] == 0, "no new case may start"
        assert seg2["plan"]["not_run"] == 2
        assert seg2["plan"]["partial"] is True
        assert seg2["plan"]["stop_detail"]["reason"] == "cost_threshold"
        # The stop happened before any new case started, and the wording says so
        # rather than implying the runner had begun and then halted.
        assert "before any new case was started" in seg2["plan"]["stop_reason"]
        assert seg2["plan"]["stop_detail"]["carry_over_cost_usd"] == pytest.approx(
            spent, abs=1e-6
        )
        assert seg2["continuation"]["carry_over_cost_usd"] == pytest.approx(spent, abs=1e-6)

    def test_a_budget_above_the_carried_spend_still_runs(self, tmp_path, monkeypatch):
        case_ids = [f"C{i:02d}" for i in range(4)]
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids[:2] for r in _completing_plan(c)],
            extra_argv=["--n-cases", "2"], output_name="seg1.json",
        )
        code, seg2 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids[2:] for r in _completing_plan(c)],
            extra_argv=["--exclude-from", str(tmp_path / "seg1.json"),
                        "--experiment-budget-usd", "1000"],
            output_name="seg2.json",
        )
        assert code == 0
        assert seg2["plan"]["attempted"] == 2

    def test_incomplete_usage_still_stops_conservatively(self, tmp_path, monkeypatch):
        """The carried-over budget does not replace the usage-integrity rule."""
        case_ids = [f"C{i:02d}" for i in range(4)]
        code, seg = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=_completing_plan("C00", with_usage=False),
            extra_argv=["--experiment-budget-usd", "1000"],
            output_name="seg.json",
        )
        assert code == 3
        assert seg["plan"]["attempted"] == 1
        assert seg["plan"]["stop_detail"]["reason"] == "incomplete_usage"


# ---------------------------------------------------------------------------
# Arm B: fixed-context LLM baseline
# ---------------------------------------------------------------------------

def _baseline_payload(**overrides: Any) -> str:
    """A complete, contract-satisfying baseline answer.

    ``cited_pmids`` defaults to a PMID the synthetic corpus contains: an
    evidence verdict with no citation is now rejected, exactly as the agent
    arm's finish tool rejects it.
    """
    payload = {
        "verdict": YES, "answer": "Yes.", "claim": "It changed the outcome.",
        "cited_pmids": ["1001"], "decisive_reason": "Direct evidence.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _baseline_completion(content: str, *, prompt: int = 2000, completion: int = 120,
                         with_usage: bool = True) -> Any:
    message = types.SimpleNamespace(content=content, tool_calls=None, role="assistant")
    usage = (
        types.SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              total_tokens=prompt + completion)
        if with_usage else None
    )
    return types.SimpleNamespace(
        id="r", model="deepseek-flash",
        choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
        usage=usage,
    )


def _baseline_agent(docs, outcomes, *, model="deepseek-flash", top_k=4):
    """A real FixedContextLLMAgent with only the SDK client replaced."""
    from bioevidence.fixed_context_agent import FixedContextLLMAgent
    from bioevidence.retrievers import BM25Retriever
    from bioevidence.tools import LiteratureTools

    agent = FixedContextLLMAgent(
        LiteratureTools(docs, retriever=BM25Retriever(docs)),
        api_key="sk-test-offline", model=model, top_k=top_k,
        max_tokens_per_call=1024, timeout=30, max_retries=0,
    )
    agent._client = AuditingClient(_Transport(list(outcomes)))
    return agent


class TestFixedContextBaseline:
    """Arm B must ask once, without tools, and never repair a failure by asking again."""

    def _docs_and_cases(self, n=1):
        return _tiny_comparison_inputs(n)

    def test_exactly_one_request_and_no_tool_parameters(self):
        docs, cases = self._docs_and_cases()
        transport_agent = _baseline_agent(docs, [_baseline_completion(_baseline_payload())])

        from scripts.compare_pipelines import run_comparison
        results = run_comparison(
            docs=docs, cases=cases, llm_agent=_AuditedScriptedAgent([_llm_response()]),
            baseline_agent=transport_agent, verbose=False,
        )

        base = results[0]
        assert base.baseline_run_status == "completed"
        assert base.baseline_steps_used == 1
        assert base.baseline_audit["request_count"] == 1
        assert base.baseline_audit["tool_calls"] == [], (
            "Arm B is a no-tool arm; any tool call would mean it is not the control"
        )
        assert base.baseline_verdict == YES

    def test_no_tools_parameter_is_sent(self):
        docs, _ = self._docs_and_cases()
        transport = _Transport([_baseline_completion(_baseline_payload())])
        from bioevidence.fixed_context_agent import FixedContextLLMAgent
        from bioevidence.retrievers import BM25Retriever
        from bioevidence.tools import LiteratureTools

        seen: list[dict[str, Any]] = []

        class _Recording:
            def create(self, **kwargs):
                seen.append(kwargs)
                return transport.create()

        agent = FixedContextLLMAgent(
            LiteratureTools(docs, retriever=BM25Retriever(docs)),
            api_key="sk-test-offline", model="deepseek-flash", top_k=4,
        )
        agent._client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Recording())
        )
        agent.run(question="Q?")

        assert len(seen) == 1
        assert "tools" not in seen[0] and "tool_choice" not in seen[0]
        assert seen[0]["response_format"] == {"type": "json_object"}
        assert seen[0]["temperature"] == 0.0
        assert seen[0]["max_tokens"] == 1024
        assert [m["role"] for m in seen[0]["messages"]] == ["system", "user"]

    def test_context_block_comes_from_one_search_with_top_k_records(self):
        docs, _ = self._docs_and_cases()
        agent = _baseline_agent(docs, [_baseline_completion(_baseline_payload())],
                                top_k=2)
        records = agent._retrieve("Question 0?")
        assert len(records) == 2
        assert all({"pmid", "title", "abstract"} <= set(r) for r in records)

        from bioevidence.fixed_context_agent import build_user_message
        message = build_user_message("Question 0?", records)
        for record in records:
            assert f"[PMID {record['pmid']}]" in message
            assert record["abstract"] in message

    @pytest.mark.parametrize("outcome,reason", [
        (_baseline_completion("not json at all"), "not valid JSON"),
        (_baseline_completion(_baseline_payload(verdict="maybe")), "not one of"),
        (_baseline_completion(_baseline_payload(cited_pmids=["99999999"])),
         "outside the provided set"),
    ])
    def test_a_bad_response_fails_the_case_without_a_second_request(self, outcome, reason):
        docs, cases = self._docs_and_cases()
        from scripts.compare_pipelines import run_comparison

        transport = _Transport([outcome])
        from bioevidence.fixed_context_agent import FixedContextLLMAgent
        from bioevidence.retrievers import BM25Retriever
        from bioevidence.tools import LiteratureTools

        agent = FixedContextLLMAgent(
            LiteratureTools(docs, retriever=BM25Retriever(docs)),
            api_key="sk-test-offline", model="deepseek-flash", top_k=4,
        )
        agent._client = AuditingClient(transport)

        results = run_comparison(
            docs=docs, cases=cases, llm_agent=_AuditedScriptedAgent([_llm_response()]),
            baseline_agent=agent, verbose=False,
        )
        base = results[0]
        assert base.baseline_errored is True
        assert reason in (base.baseline_error_msg or "")
        assert transport._index == 1, "a failure must not be repaired by asking again"
        assert base.baseline_audit["request_count"] == 1
        assert base.baseline_verdict == ""

    def test_a_failed_baseline_still_reports_the_usage_it_obtained(self):
        """The single request was billed even though its answer was unusable."""
        docs, cases = self._docs_and_cases()
        from scripts.compare_pipelines import run_comparison

        agent = _baseline_agent(
            docs,
            [_baseline_completion("not json", prompt=1800, completion=90)],
        )
        results = run_comparison(
            docs=docs, cases=cases, llm_agent=_AuditedScriptedAgent([_llm_response()]),
            baseline_agent=agent, verbose=False,
        )
        base = results[0]
        assert base.baseline_errored is True
        assert base.baseline_usage_coverage == "partial" or (
            base.baseline_usage_coverage == "confirmed"
        )
        assert base.baseline_audit["usage"]["prompt_tokens"] == 1800
        assert base.baseline_audit["usage"]["completion_tokens"] == 90
        assert base.baseline_cost_known_subtotal_usd is not None

    def test_without_a_baseline_agent_the_arm_is_null_not_failed(self):
        docs, cases = self._docs_and_cases()
        from scripts.compare_pipelines import run_comparison, _compute_metrics

        results = run_comparison(
            docs=docs, cases=cases,
            llm_agent=_AuditedScriptedAgent([_llm_response()]), verbose=False,
        )
        metrics = _compute_metrics(results)
        assert metrics["baseline"] is None, (
            "an arm that was not run must not appear as one that ran and failed"
        )
        assert results[0].baseline_run_status == ""


class TestArmsAreScoredIdentically:
    def _results(self, baseline_verdicts):
        results = [
            _case_result(case_id="a", gold_label="yes", llm_verdict=YES,
                         baseline_verdict=baseline_verdicts[0]),
            _case_result(case_id="b", gold_label="no", llm_verdict=MAYBE,
                         baseline_verdict=baseline_verdicts[1]),
            _case_result(case_id="c", gold_label="maybe", llm_verdict="mixed",
                         baseline_verdict=baseline_verdicts[2]),
        ]
        for r in results:
            r.baseline_run_status = "completed"
        return results

    def test_baseline_block_reports_the_same_keys(self):
        from scripts.compare_pipelines import _compute_metrics

        metrics = _compute_metrics(self._results([YES, NO, "mixed"]))
        base = metrics["baseline"]
        for key in (
            "completion_rate", "accuracy_on_completed", "accuracy",
            "end_to_end_success_rate", "macro_f1", "n_completed", "n_failed",
            "failure_breakdown", "n_mixed_verdicts",
        ):
            assert key in base, f"Arm B is missing {key}, which Arm A reports"

    def test_identical_verdicts_produce_identical_numbers(self):
        """Same verdicts in, same numbers out — the two arms share one scorer."""
        from scripts.compare_pipelines import _compute_metrics

        results = [
            _case_result(case_id="a", gold_label="yes", llm_verdict=YES,
                         baseline_verdict=YES),
            _case_result(case_id="b", gold_label="no", llm_verdict=NO,
                         baseline_verdict=NO),
            _case_result(case_id="c", gold_label="maybe", llm_verdict=MAYBE,
                         baseline_verdict=MAYBE),
        ]
        for r in results:
            r.baseline_run_status = "completed"
        metrics = _compute_metrics(results)

        llm, base = metrics["llm_agent"], metrics["baseline"]
        assert base["accuracy_on_completed"] == llm["accuracy_on_completed"] == 1.0
        assert base["macro_f1"] == llm["macro_f1"] == 1.0
        assert base["completion_rate"] == llm["completion_rate"] == 1.0
        assert base["end_to_end_success_rate"] == llm["end_to_end_success_rate"] == 1.0

    def test_differing_verdicts_are_scored_by_the_same_formula_not_copied(self):
        """Each arm's macro-F1 must match the shared formula applied to its own
        verdicts, so neither can inherit the other's number."""
        from scripts.compare_pipelines import _compute_metrics, _macro_f1

        results = self._results([YES, NO, "mixed"])
        metrics = _compute_metrics(results)
        llm, base = metrics["llm_agent"], metrics["baseline"]

        assert llm["macro_f1"] == _macro_f1(results, "llm_verdict")["macro_f1"]
        assert base["macro_f1"] == _macro_f1(results, "baseline_verdict")["macro_f1"]
        # Different verdicts, so genuinely different scores — not one value reused.
        assert llm["macro_f1"] != base["macro_f1"]
        assert llm["n_mixed_verdicts"] == base["n_mixed_verdicts"] == 1

    def test_a_failed_baseline_case_is_not_counted_as_a_prediction(self):
        from scripts.compare_pipelines import _compute_metrics

        results = [
            _case_result(case_id="a", gold_label="yes", llm_verdict=YES,
                         baseline_verdict=YES, baseline_run_status="completed"),
            _case_result(case_id="b", gold_label="yes", llm_verdict=YES,
                         baseline_verdict="", baseline_run_status="errored",
                         baseline_errored=True,
                         baseline_error_msg="FixedContextResponseError: not valid JSON"),
        ]
        base = _compute_metrics(results)["baseline"]
        assert base["n_completed"] == 1
        assert base["n_failed"] == 1
        assert base["failure_breakdown"]["errored"] == 1
        assert base["completion_rate"] == pytest.approx(0.5)
        assert base["end_to_end_success_rate"] == pytest.approx(0.5)


class TestV2Sample:
    """The v2 sample must be a fresh draw that reuses no v1 case."""

    def _gold(self) -> list[dict[str, str]]:
        rows = []
        index = 0
        for label, count in (("yes", 276), ("no", 169), ("maybe", 55)):
            for _ in range(count):
                index += 1
                rows.append({"case_id": f"PUBMEDQA-TEST-{index:04d}", "pmid": str(index),
                             "label": label, "verdict": PUBMEDQA_VERDICT_MAP[label]})
        return rows

    def _v1_ids(self, rows) -> list[str]:
        """The v1 sample as it really was: 28 yes / 17 no / 5 maybe, not the first 50."""
        chosen: dict[str, list[str]] = {"yes": [], "no": [], "maybe": []}
        for row in rows:
            if len(chosen[row["label"]]) < {"yes": 28, "no": 17, "maybe": 5}[row["label"]]:
                chosen[row["label"]].append(row["case_id"])
        return chosen["yes"] + chosen["no"] + chosen["maybe"]

    def test_v2_excludes_every_v1_case(self):
        rows = self._gold()
        v1_ids = self._v1_ids(rows)
        sample = frozen_sample(rows, total=50, seed=20261002, exclude_case_ids=v1_ids)
        assert not (set(sample["case_ids"]) & set(v1_ids))
        assert len(sample["case_ids"]) == 50
        assert sample["n_excluded"] == 50
        assert sample["pool_size"] == 450
        assert sample["pool_label_counts"] == {"yes": 248, "no": 152, "maybe": 50}

    def test_v2_quota_is_27_17_6(self):
        rows = self._gold()
        v1_ids = self._v1_ids(rows)
        sample = frozen_sample(rows, total=50, seed=20261002, exclude_case_ids=v1_ids)
        counts = {"yes": 0, "no": 0, "maybe": 0}
        for case_id in sample["case_ids"]:
            counts[sample["label_by_case"][case_id]] += 1
        assert counts == {"yes": 27, "no": 17, "maybe": 6}

    def test_excluding_only_attempted_cases_would_leak_v1_cases(self):
        """Documents the design conflict: excluding the 37 attempted rather than
        the 50 v1 cases leaves v1 cases eligible for v2."""
        rows = self._gold()
        v1_ids = self._v1_ids(rows)
        attempted = v1_ids[:37]
        sample = frozen_sample(rows, total=50, seed=20261002,
                               exclude_case_ids=attempted)
        assert set(sample["case_ids"]) & set(v1_ids), (
            "the narrower exclusion admits v1 cases, which is why the broader one "
            "was used"
        )
        # The quota is the same either way, so only the pool differs.
        counts = {"yes": 0, "no": 0, "maybe": 0}
        for case_id in sample["case_ids"]:
            counts[sample["label_by_case"][case_id]] += 1
        assert counts == {"yes": 27, "no": 17, "maybe": 6}

    def test_unknown_exclusions_are_rejected(self):
        rows = self._gold()
        with pytest.raises(ValueError, match="not present in the gold file"):
            frozen_sample(rows, total=50, seed=1, exclude_case_ids=["NOPE"])


# ---------------------------------------------------------------------------
# v2 gap regressions: budget across arms, baseline identity, baseline validation
# ---------------------------------------------------------------------------

CONFIG_FIELDS_FOR_TEST = (
    "model", "base_url", "thinking", "temperature", "max_steps",
    "max_tokens_per_call", "request_timeout_seconds", "sdk_max_retries",
)


def _baseline_queue_for(case_ids, *, with_usage=True, payload=None):
    return [
        _baseline_completion(payload or _baseline_payload(), with_usage=with_usage)
        for _ in case_ids
    ]


class TestBudgetCoversEveryPaidArm:
    """Arm B is billed too; leaving it out understates spend and hides gaps."""

    def _state(self, results, include_baseline=True):
        from scripts.compare_pipelines import _budget_state
        return _budget_state(results, include_baseline=include_baseline)

    def test_baseline_spend_is_added_to_the_subtotal(self):
        result = _case_result(
            llm_cost_usd=0.001, baseline_verdict=NO, baseline_run_status="completed",
            baseline_cost_usd=0.50, baseline_cost_known_subtotal_usd=0.50,
            baseline_usage_coverage="confirmed",
        )
        state = self._state([result])
        assert state["known_subtotal_usd"] == pytest.approx(0.501), (
            "the baseline arm's spend must count toward the experiment budget"
        )
        assert state["n_known"] == 2

    def test_baseline_not_run_is_not_counted_and_not_unknown(self):
        result = _case_result(llm_cost_usd=0.001)
        state = self._state([result], include_baseline=False)
        assert state["known_subtotal_usd"] == pytest.approx(0.001)
        assert state["usage_incomplete"] is False, (
            "an arm that never ran cannot have missing usage data"
        )

    def test_unknown_baseline_usage_marks_the_accounting_incomplete(self):
        result = _case_result(
            llm_cost_usd=0.001, baseline_verdict=NO, baseline_run_status="completed",
            baseline_cost_usd=None, baseline_cost_known_subtotal_usd=None,
            baseline_usage_coverage="unknown",
        )
        state = self._state([result])
        assert state["usage_incomplete"] is True
        assert state["incomplete_arms"] == ["C1:baseline"]

    def test_cap_trips_on_a_plus_b_but_not_on_a_alone(self, tmp_path, monkeypatch):
        """Through the real entry point: a cap that Arm A alone would not reach."""
        case_ids = [f"C{i:02d}" for i in range(3)]
        code, report = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids for r in _completing_plan(c)],
            baseline_queue=_baseline_queue_for(case_ids),
            extra_argv=["--baseline-arm", "--experiment-budget-usd", "0.002"],
        )
        assert code == 3
        assert report["plan"]["attempted"] == 1, (
            "A alone costs ~0.00168 and would not trip 0.002; A+B costs ~0.00242 and must"
        )
        assert report["plan"]["stop_detail"]["reason"] == "cost_threshold"
        cost = report["metrics"]["llm_agent"]["cost"]
        base_cost = report["metrics"]["baseline"]["cost"]
        assert cost["total_cost_usd_known"] > 0
        assert base_cost["total_cost_usd_known"] > 0

    def test_unknown_baseline_usage_stops_the_run_conservatively(
        self, tmp_path, monkeypatch
    ):
        case_ids = [f"C{i:02d}" for i in range(3)]
        code, report = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            # Arm A is fully confirmed; only Arm B reports no usage.
            queue=[r for c in case_ids for r in _completing_plan(c)],
            baseline_queue=[_baseline_completion(_baseline_payload(), with_usage=False)],
            extra_argv=["--baseline-arm", "--experiment-budget-usd", "1000"],
        )
        assert code == 3
        assert report["plan"]["attempted"] == 1
        assert report["plan"]["stop_detail"]["reason"] == "incomplete_usage"
        assert "C00:baseline" in report["plan"]["stop_detail"]["incomplete_arms"]
        assert report["cases"][0]["baseline_run_status"] == "completed"
        assert report["cases"][0]["baseline_cost_known_subtotal_usd"] is None


class TestBaselineIdentityIsBound:
    """Whether B ran, and how, is part of the experiment identity."""

    def _seg1(self, tmp_path, monkeypatch, case_ids, *, top_k):
        return _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids[:1] for r in _completing_plan(c)],
            baseline_queue=_baseline_queue_for(case_ids[:1]),
            extra_argv=["--baseline-arm", "--baseline-top-k", str(top_k),
                        "--n-cases", "1"],
            output_name="seg1.json",
        )

    def test_continuation_rejects_a_changed_baseline_top_k(self, tmp_path, monkeypatch):
        case_ids = [f"C{i:02d}" for i in range(4)]
        self._seg1(tmp_path, monkeypatch, case_ids, top_k=4)

        with pytest.raises(ValueError, match="baseline_arm"):
            _run_entry(
                tmp_path, monkeypatch, case_ids=case_ids, queue=[],
                baseline_queue=[],
                extra_argv=["--baseline-arm", "--baseline-top-k", "2",
                            "--exclude-from", str(tmp_path / "seg1.json")],
                output_name="seg2.json",
            )

    def test_continuation_rejects_enabling_the_baseline_later(
        self, tmp_path, monkeypatch
    ):
        """A segment run without Arm B is a different experiment from one with it."""
        case_ids = [f"C{i:02d}" for i in range(4)]
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids[:1] for r in _completing_plan(c)],
            extra_argv=["--n-cases", "1"], output_name="seg1.json",
        )
        with pytest.raises(ValueError, match="baseline_arm"):
            _run_entry(
                tmp_path, monkeypatch, case_ids=case_ids, queue=[],
                baseline_queue=[], extra_argv=["--baseline-arm",
                                               "--exclude-from", str(tmp_path / "seg1.json")],
                output_name="seg2.json",
            )

    def test_continuation_rejects_a_changed_prompt_fingerprint(
        self, tmp_path, monkeypatch
    ):
        case_ids = [f"C{i:02d}" for i in range(4)]
        self._seg1(tmp_path, monkeypatch, case_ids, top_k=4)

        seg1_path = tmp_path / "seg1.json"
        tampered = json.loads(seg1_path.read_text(encoding="utf-8"))
        tampered["prompt_fingerprint"]["agent"]["system_prompt_sha256"] = "f" * 64
        seg1_path.write_text(json.dumps(tampered), encoding="utf-8")

        with pytest.raises(ValueError, match="prompt_fingerprint"):
            _run_entry(
                tmp_path, monkeypatch, case_ids=case_ids, queue=[],
                baseline_queue=[],
                extra_argv=["--baseline-arm", "--baseline-top-k", "4",
                            "--exclude-from", str(seg1_path)],
                output_name="seg2.json",
            )

    def test_merge_rejects_a_changed_baseline_config(self, tmp_path, monkeypatch):
        case_ids = [f"C{i:02d}" for i in range(3)]
        for name, top_k in (("seg1.json", 4), ("seg2.json", 4)):
            _run_entry(
                tmp_path, monkeypatch, case_ids=case_ids,
                queue=[r for c in case_ids for r in _completing_plan(c)],
                baseline_queue=_baseline_queue_for(case_ids),
                extra_argv=["--baseline-arm", "--baseline-top-k", str(top_k)],
                output_name=name,
            )

        seg2_path = tmp_path / "seg2.json"
        tampered = json.loads(seg2_path.read_text(encoding="utf-8"))
        tampered["config"]["baseline_arm"]["top_k"] = 9
        seg2_path.write_text(json.dumps(tampered), encoding="utf-8")

        with pytest.raises(ValueError, match="baseline_arm"):
            _run_merge(
                tmp_path, monkeypatch,
                segments=[tmp_path / "seg1.json", seg2_path],
                sample_path=tmp_path / "sample.json", output_name="merged.json",
            )

    def test_reports_predating_these_sections_remain_mergeable(self):
        """v1 reports carry neither key; two of them agree by definition."""
        from scripts.compare_pipelines import _compare_identity, _experiment_identity

        v1 = {
            "sample": {"seed": 1, "total": 2, "case_ids_sha256": "a",
                       "gold_sha256": "b"},
            "input_hashes": {"sample_file_sha256": "c", "files": {"corpus.jsonl": "d"}},
            "config": dict.fromkeys(CONFIG_FIELDS_FOR_TEST, "x"),
        }
        conflicts = _compare_identity(
            _experiment_identity(v1), "v1a", _experiment_identity(v1), "v1b"
        )
        assert conflicts == [], (
            "the new sections must not make historical reports incompatible"
        )


class TestBaselineResponseValidation:
    """B must meet the same bar the agent's finish tool sets."""

    @pytest.mark.parametrize("missing", ["answer", "claim", "cited_pmids",
                                         "decisive_reason"])
    def test_a_response_missing_a_required_field_is_not_completed(
        self, tmp_path, monkeypatch, missing
    ):
        case_ids = ["C00"]
        payload = json.loads(_baseline_payload())
        del payload[missing]

        code, report = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=_completing_plan("C00"),
            baseline_queue=[_baseline_completion(json.dumps(payload))],
            extra_argv=["--baseline-arm"],
        )
        case = report["cases"][0]
        assert case["baseline_run_status"] == "errored", (
            f"a response without {missing!r} must not count as completed"
        )
        assert case["baseline_correct"] is None
        assert report["metrics"]["baseline"]["n_completed"] == 0

    def test_bare_verdict_is_rejected(self, tmp_path, monkeypatch):
        """Codex's exact reproduction: {"verdict": "supported"} alone."""
        code, report = _run_entry(
            tmp_path, monkeypatch, case_ids=["C00"],
            queue=_completing_plan("C00"),
            baseline_queue=[_baseline_completion(json.dumps({"verdict": "supported"}))],
            extra_argv=["--baseline-arm"],
        )
        case = report["cases"][0]
        assert case["baseline_run_status"] == "errored"
        assert "missing required field" in (case["baseline_error_message"] or "")
        assert case["baseline_requests"] == 1, "no repair request may be sent"

    def test_evidence_verdict_without_a_citation_is_rejected(self, tmp_path, monkeypatch):
        code, report = _run_entry(
            tmp_path, monkeypatch, case_ids=["C00"],
            queue=_completing_plan("C00"),
            baseline_queue=[_baseline_completion(
                _baseline_payload(verdict=NO, cited_pmids=[])
            )],
            extra_argv=["--baseline-arm"],
        )
        case = report["cases"][0]
        assert case["baseline_run_status"] == "errored"
        assert "at least one cited PMID" in (case["baseline_error_message"] or "")

    def test_abstention_may_cite_nothing(self, tmp_path, monkeypatch):
        """insufficient is the one verdict that need not name evidence."""
        code, report = _run_entry(
            tmp_path, monkeypatch, case_ids=["C00"],
            queue=_completing_plan("C00"),
            baseline_queue=[_baseline_completion(
                _baseline_payload(verdict=MAYBE, cited_pmids=[])
            )],
            extra_argv=["--baseline-arm"],
        )
        case = report["cases"][0]
        assert case["baseline_run_status"] == "completed"
        assert case["baseline_verdict"] == MAYBE

    def test_wrong_field_type_is_rejected(self, tmp_path, monkeypatch):
        code, report = _run_entry(
            tmp_path, monkeypatch, case_ids=["C00"],
            queue=_completing_plan("C00"),
            baseline_queue=[_baseline_completion(
                _baseline_payload(answer=123)
            )],
            extra_argv=["--baseline-arm"],
        )
        case = report["cases"][0]
        assert case["baseline_run_status"] == "errored"
        assert "must be str" in (case["baseline_error_message"] or "")

    def test_a_rejected_baseline_still_reports_its_obtained_usage(
        self, tmp_path, monkeypatch
    ):
        code, report = _run_entry(
            tmp_path, monkeypatch, case_ids=["C00"],
            queue=_completing_plan("C00"),
            baseline_queue=[_baseline_completion(
                json.dumps({"verdict": "supported"}), prompt=1800, completion=90
            )],
            extra_argv=["--baseline-arm"],
        )
        case = report["cases"][0]
        assert case["baseline_run_status"] == "errored"
        assert case["baseline_audit"]["usage"]["prompt_tokens"] == 1800
        assert case["baseline_cost_known_subtotal_usd"] is not None


# ---------------------------------------------------------------------------
# Cross-segment cost aggregation across every paid arm
# ---------------------------------------------------------------------------

#: Per-case known subtotals the scripted plans produce, at peak cache-miss rates.
_AGENT_CASE_COST = 0.001680    # 4 requests x (1000 in + 100 out)
_BASELINE_CASE_COST = 0.000744  # 1 request  x (2000 in + 120 out)
_BOTH_ARMS_CASE_COST = _AGENT_CASE_COST + _BASELINE_CASE_COST


class TestCrossSegmentCarryOver:
    """A continuation must carry over every paid arm, not just the agent."""

    def _seg1(self, tmp_path, monkeypatch, case_ids, *, n_cases=1, output="seg1.json"):
        return _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids[:n_cases] for r in _completing_plan(c)],
            baseline_queue=_baseline_queue_for(case_ids[:n_cases]),
            extra_argv=["--baseline-arm", "--n-cases", str(n_cases)],
            output_name=output,
        )

    def test_carry_over_includes_the_baseline_arm(self, tmp_path, monkeypatch):
        case_ids = [f"C{i:02d}" for i in range(3)]
        _, seg1 = self._seg1(tmp_path, monkeypatch, case_ids)
        assert seg1["cases"][0]["llm_cost_known_subtotal_usd"] == pytest.approx(
            _AGENT_CASE_COST, abs=1e-6
        )
        assert seg1["cases"][0]["baseline_cost_known_subtotal_usd"] == pytest.approx(
            _BASELINE_CASE_COST, abs=1e-6
        )

        code, seg2 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids[1:] for r in _completing_plan(c)],
            baseline_queue=_baseline_queue_for(case_ids[1:]),
            extra_argv=["--baseline-arm", "--exclude-from", str(tmp_path / "seg1.json")],
            output_name="seg2.json",
        )
        assert code == 0
        assert seg2["config"]["carry_over_cost_usd"] == pytest.approx(
            _BOTH_ARMS_CASE_COST, abs=1e-6
        ), "the carry-over must sum the agent and baseline arms"

    def test_a_reached_threshold_makes_no_new_requests(self, tmp_path, monkeypatch):
        """Codex's reproduction: carried spend alone exhausts the budget."""
        case_ids = [f"C{i:02d}" for i in range(3)]
        self._seg1(tmp_path, monkeypatch, case_ids)

        # Empty queues: any model request at all would raise.
        code, seg2 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids, queue=[], baseline_queue=[],
            extra_argv=["--baseline-arm", "--exclude-from", str(tmp_path / "seg1.json"),
                        "--experiment-budget-usd", "0.002"],
            output_name="seg2.json",
        )
        assert code == 3
        assert seg2["plan"]["attempted"] == 0, "no new case may start"
        assert seg2["plan"]["not_run"] == 2
        assert seg2["plan"]["partial"] is True
        assert seg2["plan"]["stop_detail"]["reason"] == "cost_threshold"
        assert seg2["config"]["carry_over_cost_usd"] == pytest.approx(
            _BOTH_ARMS_CASE_COST, abs=1e-6
        )

    def test_a_threshold_above_the_carried_spend_still_runs(self, tmp_path, monkeypatch):
        case_ids = [f"C{i:02d}" for i in range(3)]
        self._seg1(tmp_path, monkeypatch, case_ids)
        code, seg2 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids[1:] for r in _completing_plan(c)],
            baseline_queue=_baseline_queue_for(case_ids[1:]),
            extra_argv=["--baseline-arm", "--exclude-from", str(tmp_path / "seg1.json"),
                        "--experiment-budget-usd", "1000"],
            output_name="seg2.json",
        )
        assert code == 0
        assert seg2["plan"]["attempted"] == 2
        assert seg2["config"]["carry_over_cost_usd"] == pytest.approx(
            _BOTH_ARMS_CASE_COST, abs=1e-6
        )

    def test_merge_subtotals_equal_the_sum_of_original_case_costs(
        self, tmp_path, monkeypatch
    ):
        """Chained segments must not count an earlier segment's spend twice."""
        case_ids = [f"C{i:02d}" for i in range(3)]
        self._seg1(tmp_path, monkeypatch, case_ids, n_cases=1)
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids[1:2] for r in _completing_plan(c)],
            baseline_queue=_baseline_queue_for(case_ids[1:2]),
            extra_argv=["--baseline-arm", "--n-cases", "1",
                        "--exclude-from", str(tmp_path / "seg1.json")],
            output_name="seg2.json",
        )

        _, merged = _run_merge(
            tmp_path, monkeypatch,
            segments=[tmp_path / "seg1.json", tmp_path / "seg2.json"],
            sample_path=tmp_path / "sample.json", output_name="merged.json",
        )

        # Two cases were run in total, each paying both arms.
        assert merged["merge_notes"]["known_subtotal_usd"] == pytest.approx(
            2 * _BOTH_ARMS_CASE_COST, abs=1e-6
        ), "the merged subtotal must be the sum of the original per-case costs"
        # A naive sum of segment-level figures (including carry-over) would be
        # larger than this; the merge reads per-case values instead.
        naive = (merged["segments"][0]["known_subtotal_usd"]
                 + merged["segments"][0]["known_subtotal_usd"]
                 + merged["segments"][1]["known_subtotal_usd"])
        assert merged["merge_notes"]["known_subtotal_usd"] < naive

        # Per-segment figures now cover both arms too.
        for segment in merged["segments"]:
            assert segment["known_subtotal_usd"] == pytest.approx(
                _BOTH_ARMS_CASE_COST, abs=1e-6
            )

        a_cost = merged["metrics"]["llm_agent"]["cost"]["total_cost_usd_known"]
        b_cost = merged["metrics"]["baseline"]["cost"]["total_cost_usd_known"]
        assert a_cost == pytest.approx(2 * _AGENT_CASE_COST, abs=1e-6)
        assert b_cost == pytest.approx(2 * _BASELINE_CASE_COST, abs=1e-6)

    def test_v1_reports_without_a_baseline_are_summed_by_the_agent_arm_alone(
        self, tmp_path, monkeypatch
    ):
        """v1 never ran Arm B, so its subtotals must not change."""
        case_ids = [f"C{i:02d}" for i in range(4)]
        for name, subset in (("seg1.json", case_ids[:2]), ("seg2.json", case_ids[2:])):
            _run_entry(
                tmp_path, monkeypatch, case_ids=case_ids,
                queue=[r for c in subset for r in _completing_plan(c)],
                extra_argv=["--n-cases", str(len(subset))] if name == "seg1.json"
                else ["--exclude-from", str(tmp_path / "seg1.json")],
                output_name=name,
            )
        _, merged = _run_merge(
            tmp_path, monkeypatch,
            segments=[tmp_path / "seg1.json", tmp_path / "seg2.json"],
            sample_path=tmp_path / "sample.json", output_name="merged.json",
        )
        expected = sum(
            c["llm_cost_known_subtotal_usd"] for c in merged["cases"]
            if c["llm_cost_known_subtotal_usd"] is not None
        )
        assert merged["merge_notes"]["known_subtotal_usd"] == pytest.approx(
            round(expected, 6), abs=1e-6
        )
        assert merged["metrics"]["baseline"] is None, (
            "no baseline arm ran, so nothing may be added for it"
        )


class TestBaselineLatencyIsWritten:
    """The writer must *emit* baseline_latency_ms, not merely be able to rebuild it.

    An earlier regression asserted the reconstruction path only, so a writer that
    never emitted the field passed while every real report showed the baseline's
    average latency as n/a.  This one reads the exported JSON.
    """

    def test_written_by_main_and_preserved_by_merge(self, tmp_path, monkeypatch):
        case_ids = [f"C{i:02d}" for i in range(3)]
        _, segment = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids for r in _completing_plan(c)],
            baseline_queue=_baseline_queue_for(case_ids),
            extra_argv=["--baseline-arm"], output_name="seg.json",
        )

        for case in segment["cases"]:
            assert "baseline_latency_ms" in case, (
                "the writer must emit the field — a merge can only preserve what "
                "was recorded"
            )
            assert case["baseline_latency_ms"] is not None
        assert segment["metrics"]["baseline"]["n_latency_recorded"] == len(case_ids)
        assert segment["metrics"]["baseline"]["avg_latency_ms"] is not None

        _, merged = _run_merge(
            tmp_path, monkeypatch, segments=[tmp_path / "seg.json"],
            sample_path=tmp_path / "sample.json", output_name="merged.json",
        )

        # The per-case values survive the merge...
        for case in merged["cases"]:
            assert case["baseline_latency_ms"] is not None
        # ...and so does the aggregate derived from them.  Absent values are never
        # filled in from a segment mean, so equality here means nothing was.
        assert merged["metrics"]["baseline"]["n_latency_recorded"] == len(case_ids)
        assert merged["metrics"]["baseline"]["avg_latency_ms"] == pytest.approx(
            segment["metrics"]["baseline"]["avg_latency_ms"]
        )
        # Scoped to the field this test is about: a run that never ran Arm C
        # legitimately has no arm_c_* values, and those are reported separately.
        absent = merged["merge_notes"]["cases_with_absent_fields"]
        for case in merged["cases"]:
            assert "baseline_latency_ms" not in absent.get(case["case_id"], []), (
                "the writer emitted baseline_latency_ms, so the merge must not "
                "report it missing"
            )

    def test_no_field_the_merge_expects_is_missing_from_a_written_report(
        self, tmp_path, monkeypatch
    ):
        """Every field the merge knows how to rebuild is one the writer emitted."""
        import scripts.compare_pipelines as cp

        case_ids = ["C00"]
        _, report = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=_completing_plan("C00"),
            baseline_queue=_baseline_queue_for(case_ids),
            extra_argv=["--baseline-arm"],
        )
        written = set(report["cases"][0])
        for field in cp._REPORT_CASE_FIELDS:
            assert field in written, (
                f"{field} is in _REPORT_CASE_FIELDS but the writer never emits it"
            )


# ---------------------------------------------------------------------------
# v3 / H3: shared scoring, Arm C prompt intervention, sample and identity
# ---------------------------------------------------------------------------

def _v3_result(**overrides: Any):
    """A case carrying both model arms, for v3 arithmetic."""
    base = {
        "llm_run_status": "completed",
        "arm_c_run_status": "completed",
    }
    base.update(overrides)
    return _case_result(**base)


class TestV3Arithmetic:
    """The shared v3 mapping, counted the way the design table specifies."""

    def _detail(self, results):
        from scripts.compare_pipelines import _macro_f1_v3
        return _macro_f1_v3(
            results, verdict_attr="llm_verdict", errored_attr="llm_errored",
            status_attr="llm_run_status",
        )

    def test_mixed_on_a_yes_case_is_an_fn_for_yes_and_an_fp_for_maybe(self):
        detail = self._detail([_v3_result(gold_label="yes", llm_verdict="mixed")])
        assert detail["per_class"]["yes"] == {
            "support": 1, "tp": 0, "fp": 0, "fn": 1, "f1": 0.0,
        }
        assert detail["per_class"]["maybe"] == {
            "support": 0, "tp": 0, "fp": 1, "fn": 0, "f1": 0.0,
        }

    def test_mixed_on_a_no_case_does_the_same_for_no(self):
        detail = self._detail([_v3_result(gold_label="no", llm_verdict="mixed")])
        assert detail["per_class"]["no"] == {
            "support": 1, "tp": 0, "fp": 0, "fn": 1, "f1": 0.0,
        }
        assert detail["per_class"]["maybe"]["fp"] == 1

    def test_mixed_or_insufficient_on_a_maybe_case_is_a_true_positive(self):
        for verdict in ("mixed", "insufficient"):
            detail = self._detail([_v3_result(gold_label="maybe", llm_verdict=verdict)])
            assert detail["per_class"]["maybe"]["tp"] == 1, verdict
            assert detail["per_class"]["maybe"]["f1"] == 1.0

    def test_failure_status_gates_scoring_before_the_verdict(self):
        """A failed case contributes only an FN — never a TP, even if its fallback
        verdict would have matched the gold label under the v3 mapping."""
        results = [
            _v3_result(gold_label="maybe", llm_verdict="insufficient",
                       llm_run_status="budget_exhausted"),
            _v3_result(gold_label="maybe", llm_verdict="insufficient",
                       llm_run_status="text_exit"),
            _v3_result(gold_label="yes", llm_verdict="insufficient",
                       llm_run_status="errored", llm_errored=True),
        ]
        detail = self._detail(results)
        assert detail["per_class"]["maybe"] == {
            "support": 2, "tp": 0, "fp": 0, "fn": 2, "f1": 0.0,
        }, "a failed maybe case must not score as a correct maybe prediction"
        assert detail["per_class"]["yes"]["fn"] == 1

    def test_worked_example_matches_hand_counting(self):
        results = [
            _v3_result(case_id="a", gold_label="yes", llm_verdict="mixed"),
            _v3_result(case_id="b", gold_label="no", llm_verdict="mixed"),
            _v3_result(case_id="c", gold_label="maybe", llm_verdict="mixed"),
            _v3_result(case_id="d", gold_label="yes", llm_verdict="insufficient",
                       llm_run_status="errored", llm_errored=True),
            _v3_result(case_id="e", gold_label="maybe", llm_verdict="insufficient",
                       llm_run_status="budget_exhausted"),
        ]
        detail = self._detail(results)
        assert detail["per_class"]["yes"] == {
            "support": 2, "tp": 0, "fp": 0, "fn": 2, "f1": 0.0}
        assert detail["per_class"]["no"] == {
            "support": 1, "tp": 0, "fp": 0, "fn": 1, "f1": 0.0}
        assert detail["per_class"]["maybe"] == {
            "support": 2, "tp": 1, "fp": 2, "fn": 1, "f1": 0.4}
        assert detail["denominator"] == 5
        assert detail["macro_f1"] == pytest.approx(round((0.0 + 0.0 + 0.4) / 3, 4))

    def test_threshold_uses_the_unrounded_value(self):
        detail = self._detail([_v3_result(gold_label="yes", llm_verdict="yes")])
        assert detail["macro_f1_exact"] is not None
        assert detail["macro_f1"] == pytest.approx(detail["macro_f1_exact"], abs=5e-5)


class TestLegacyScoringUnchanged:
    """v1/v2 numbers must not move because v3 exists."""

    def test_default_policy_is_legacy_and_maps_mixed_to_no_label(self):
        from scripts.compare_pipelines import (
            DEFAULT_SCORING_POLICY, _matches_label_for, _predicted_label,
        )
        assert DEFAULT_SCORING_POLICY == "legacy"
        assert _predicted_label("mixed") is None, (
            "the legacy mapping must keep treating mixed as unscorable"
        )
        assert _predicted_label("insufficient") == "maybe"
        assert _matches_label_for("legacy") is _matches_label_for("legacy")
        # A legacy run scores a mixed prediction as wrong on every gold label.
        legacy = _matches_label_for("legacy")
        assert legacy("mixed", "maybe") is False
        assert _matches_label_for("v3")("mixed", "maybe") is True

    def test_legacy_macro_f1_still_excludes_failed_cases(self):
        """The v3 denominator rule must not leak into the legacy metric."""
        from scripts.compare_pipelines import _macro_f1
        results = [
            _case_result(case_id="a", gold_label="yes", llm_verdict="supported"),
            _case_result(case_id="b", gold_label="yes", llm_verdict="insufficient",
                         llm_run_status="budget_exhausted"),
        ]
        detail = _macro_f1([results[0]], "llm_verdict")
        assert detail["denominator"] == 1, "legacy macro-F1 is over completed cases"
        # One perfect yes prediction: yes=1.0, and the two empty classes score 0.
        assert detail["macro_f1"] == pytest.approx(0.3333)

    def test_unknown_policy_is_rejected(self):
        from scripts.compare_pipelines import _compute_metrics
        with pytest.raises(ValueError, match="unknown scoring policy"):
            _compute_metrics([], scoring_policy="v4")


class TestArmCPromptIntervention:
    """C differs from A in prompt material only, and cannot alter A's copies."""

    def test_a_prompt_and_tools_are_untouched(self):
        import hashlib
        import json as _json

        from bioevidence.llm_agent import TOOL_SCHEMAS, _system_prompt

        def canon(obj):
            return hashlib.sha256(_json.dumps(
                obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()

        assert canon(_system_prompt()) == (
            "84e8d97b91a0940698b93a25db09d821d5ab832e17776d62a114f99d468db880"
        ), "Arm A's system prompt must not change"
        assert canon(TOOL_SCHEMAS) == (
            "98e8916fc0a01162c7d4f178da0b4ff5f1069550cb963651e62a9a11146dcb16"
        ), "Arm A's tool schemas must not change"

    def test_the_diff_is_a_pure_insertion(self):
        from bioevidence.prompt_variants import (
            ARM_C_MIXED_RULES, arm_c_intervention_diff, arm_c_system_prompt,
        )
        from bioevidence.llm_agent import _system_prompt

        diff = arm_c_intervention_diff()
        assert diff["system_prompt_delta_is_insertion_only"] is True
        assert arm_c_system_prompt().replace(ARM_C_MIXED_RULES, "", 1) == _system_prompt()
        assert set(diff["changed"]) == {"system_prompt", "finish_tool_description"}

    def test_insufficient_wording_is_untouched_by_the_clarification(self):
        """The intervention must not redefine insufficient."""
        from bioevidence.prompt_variants import ARM_C_MIXED_RULES
        assert "insufficient" not in ARM_C_MIXED_RULES.lower(), (
            "the mixed clarification must not restate or alter the insufficient rule"
        )

    def test_c_schemas_are_an_independent_copy(self):
        from bioevidence.llm_agent import TOOL_SCHEMAS
        from bioevidence.prompt_variants import arm_c_tool_schemas

        before = TOOL_SCHEMAS[3]["function"]["description"]
        c_schemas = arm_c_tool_schemas()
        assert c_schemas is not TOOL_SCHEMAS
        assert c_schemas[3] is not TOOL_SCHEMAS[3]
        assert TOOL_SCHEMAS[3]["function"]["description"] == before, (
            "building C's schemas must not mutate the shared object A reads"
        )
        assert c_schemas[3]["function"]["description"] != before

    def test_c_and_a_share_every_non_prompt_setting(self, tmp_path, monkeypatch):
        case_ids = ["C00"]
        _, report = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=_completing_plan("C00"),
            baseline_queue=None,
            extra_argv=["--arm-c"], arm_c_queue=[r for c in case_ids for r in _completing_plan(c)],
        )
        config = report["config"]
        a, c = config, config["arm_c"]
        for key in ("model", "base_url", "max_steps", "max_tokens_per_call",
                    "request_timeout_seconds", "sdk_max_retries", "temperature"):
            assert c[key] == a[key], f"Arm C must share {key} with Arm A"
        assert c["intervention"]["changed"] == ["system_prompt",
                                                "finish_tool_description"]

    def test_c_runs_the_same_loop_and_does_not_change_a(self, tmp_path, monkeypatch):
        """A and C see the same question and produce independent records."""
        case_ids = ["C00"]
        _, report = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=_completing_plan("C00"),
            extra_argv=["--arm-c"], arm_c_queue=[r for c in case_ids for r in _completing_plan(c)],
        )
        case = report["cases"][0]
        assert case["llm_run_status"] == "completed"
        assert case["arm_c_run_status"] == "completed"
        assert case["arm_c_audit"]["request_count"] == 4, "C runs the same agent loop"


class TestV3Sample:
    def _gold(self):
        rows = []
        for label, count in (("yes", 276), ("no", 169), ("maybe", 55)):
            for i in range(count):
                rows.append({"case_id": f"{label}-{i}", "label": label})
        return rows

    def test_quota_is_28_17_5_from_the_400_case_pool(self):
        from bioevidence.pubmedqa_sample import largest_remainder_quotas
        apportionment = largest_remainder_quotas(
            {"yes": 221, "no": 135, "maybe": 44}, 50
        )
        assert {s["label"]: s["quota"] for s in apportionment["strata"]} == {
            "yes": 28, "no": 17, "maybe": 5,
        }

    def test_excluding_the_100_attempted_leaves_the_designed_pool(self, tmp_path):
        """Against the real frozen sample: 400 cases at 221/135/44, no overlap."""
        from collections import Counter

        from bioevidence.pubmedqa_sample import build_frozen_sample_record

        gold = REPO_ROOT / "data/pubmedqa/v1/test_gold.jsonl"
        if not gold.exists():  # pragma: no cover - prepared data is machine-local
            pytest.skip("prepared benchmark not present")
        attempted = set()
        for path in (REPO_ROOT / "private/env_precheck").glob("pipeline_comparison_*.json"):
            if path.name == "pipeline_comparison_50.json":
                continue
            attempted |= {c["case_id"] for c in json.loads(path.read_text())["cases"]}
        for path in (REPO_ROOT / "private/env_precheck").glob("v2_ab_*.json"):
            if path.name == "v2_ab_50_final.json":
                continue
            attempted |= {c["case_id"] for c in json.loads(path.read_text())["cases"]}
        assert len(attempted) == 100

        record = build_frozen_sample_record(gold, total=50, seed=20261002,
                                            exclude_case_ids=sorted(attempted))
        assert record["pool_size"] == 400
        assert record["pool_label_counts"] == {"yes": 221, "no": 135, "maybe": 44}
        assert Counter(record["label_by_case"].values()) == {
            "yes": 28, "no": 17, "maybe": 5}
        assert len(set(record["case_ids"])) == 50
        # Reproducible, and disjoint from both earlier samples.
        again = build_frozen_sample_record(gold, total=50, seed=20261002,
                                           exclude_case_ids=sorted(attempted))
        assert again["case_ids"] == record["case_ids"]
        for earlier in ("eval_sample_50_v1_seed20260729.json",
                        "eval_sample_v2_50_seed20261002.json"):
            other = json.loads((REPO_ROOT / "reports" / earlier).read_text())
            assert not (set(record["case_ids"]) & set(other["case_ids"]))

    def test_the_frozen_v3_sample_on_disk_matches_the_declared_rule(self):
        path = REPO_ROOT / "private/v3/eval_sample_v3_50_seed20261002.json"
        if not path.exists():  # pragma: no cover
            pytest.skip("v3 sample not present")
        record = json.loads(path.read_text())
        assert record["seed"] == 20261002
        assert record["total"] == 50
        assert record["n_excluded"] == 100
        assert len(record["case_ids"]) == 50
        counts = {"yes": 0, "no": 0, "maybe": 0}
        for label in record["label_by_case"].values():
            counts[label] += 1
        assert counts == {"yes": 28, "no": 17, "maybe": 5}


class TestV3AcceptanceIsWithheldOnPartialRuns:
    def _metrics(self, results, policy="v3"):
        from scripts.compare_pipelines import _compute_metrics
        return _compute_metrics(results, scoring_policy=policy)

    def test_partial_run_reports_no_acceptance_conclusion(self):
        from scripts.compare_pipelines import _v3_acceptance
        results = [
            _v3_result(case_id="a", gold_label="yes", llm_verdict="supported",
                       arm_c_verdict="mixed"),
        ]
        block = _v3_acceptance(results, self._metrics(results),
                               frozen_total=50, arm_c_enabled=True)
        assert block["status"] == "experiment_incomplete"
        assert block["criteria_met"] is None
        assert block["frozen_total"] == 50
        assert block["attempted"] == 1
        assert block["not_run"] == 49
        assert "never attempted" in block["reason"]

    def test_complete_run_produces_a_verdict_and_the_costs(self):
        from scripts.compare_pipelines import _v3_acceptance
        results = [
            _v3_result(case_id="a", gold_label="maybe", llm_verdict="supported",
                       arm_c_verdict="mixed"),
            _v3_result(case_id="b", gold_label="yes", llm_verdict="mixed",
                       arm_c_verdict="supported"),
        ]
        block = _v3_acceptance(results, self._metrics(results),
                               frozen_total=2, arm_c_enabled=True)
        assert block["status"] == "complete"
        assert block["maybe_correct_rate"]["a_tp"] == 0
        assert block["maybe_correct_rate"]["c_tp"] == 1
        assert block["maybe_correct_rate"]["c_exceeds_a"] is True
        assert block["maybe_misroutes"]["a"]["yes_to_maybe"] == 1
        assert block["maybe_misroutes"]["c"]["yes_to_maybe"] == 0
        assert "exploratory engineering criterion" in block["conclusion_strength"]
        assert block["per_class_f1"]["maybe"]["c"] > block["per_class_f1"]["maybe"]["a"]

    def test_no_arm_c_means_no_applicable_test(self):
        from scripts.compare_pipelines import _v3_acceptance
        # Explicitly no Arm C: the second arm's status is empty.
        results = [_v3_result(gold_label="yes", llm_verdict="supported",
                              arm_c_run_status="", arm_c_verdict="")]
        metrics = self._metrics(results)
        assert metrics["arm_c"] is None
        block = _v3_acceptance(results, metrics, frozen_total=1,
                               arm_c_enabled=False)
        assert block["status"] == "not_applicable"


class TestV3EvidenceAndIdentity:
    def test_exported_json_keeps_both_arms_audit_material(self, tmp_path, monkeypatch):
        """The report must carry the evidence chain, not only the verdict."""
        case_ids = ["C00"]
        _, report = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=_completing_plan("C00"), extra_argv=["--arm-c"], arm_c_queue=[r for c in case_ids for r in _completing_plan(c)],
        )
        case = report["cases"][0]
        for arm in ("llm", "arm_c"):
            assert case[f"{arm}_answer"], f"{arm} answer text missing"
            assert case[f"{arm}_claim"], f"{arm} claim missing"
            assert case[f"{arm}_decisive_reason"], f"{arm} reason missing"
            returns = case[f"{arm}_tool_returns"]
            assert returns, f"{arm} tool returns missing"
            assert any("abstract" in json.dumps(r) for r in returns), (
                f"{arm} tool returns must include the fetched evidence"
            )
            assert case[f"{arm}_audit"]["tool_calls"], f"{arm} tool inputs missing"
        assert report["config"]["arm_c"]["intervention"]["changed"]

    def test_budget_counts_arm_c(self, tmp_path, monkeypatch):
        from scripts.compare_pipelines import _budget_state
        result = _v3_result(
            llm_cost_usd=0.001, llm_cost_known_subtotal_usd=0.001,
            arm_c_verdict="supported", arm_c_cost_usd=0.25,
            arm_c_cost_known_subtotal_usd=0.25, arm_c_usage_coverage="confirmed",
        )
        state = _budget_state([result], include_baseline=False, include_arm_c=True)
        assert state["known_subtotal_usd"] == pytest.approx(0.251)
        without = _budget_state([result], include_baseline=False, include_arm_c=False)
        assert without["known_subtotal_usd"] == pytest.approx(0.001)

    def test_unknown_arm_c_usage_marks_the_accounting_incomplete(self):
        from scripts.compare_pipelines import _budget_state
        result = _v3_result(
            arm_c_verdict="supported", arm_c_cost_known_subtotal_usd=None,
            arm_c_usage_coverage="unknown",
        )
        state = _budget_state([result], include_baseline=False, include_arm_c=True)
        assert state["usage_incomplete"] is True
        assert state["incomplete_arms"] == ["C1:arm_c"]

    def test_continuation_rejects_a_different_scoring_policy(self, tmp_path, monkeypatch):
        case_ids = [f"C{i:02d}" for i in range(3)]
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids[:1] for r in _completing_plan(c)],
            extra_argv=["--arm-c", "--scoring-policy", "v3", "--n-cases", "1"],
            arm_c_queue=[r for c in case_ids[:1] for r in _completing_plan(c)],
            output_name="seg1.json",
        )
        with pytest.raises(ValueError, match="scoring_policy"):
            _run_entry(
                tmp_path, monkeypatch, case_ids=case_ids, queue=[],
                extra_argv=["--arm-c", "--scoring-policy", "legacy",
                            "--exclude-from", str(tmp_path / "seg1.json")],
                output_name="seg2.json",
            )

    def test_continuation_rejects_adding_arm_c_later(self, tmp_path, monkeypatch):
        case_ids = [f"C{i:02d}" for i in range(3)]
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids[:1] for r in _completing_plan(c)],
            extra_argv=["--scoring-policy", "v3", "--n-cases", "1"],
            output_name="seg1.json",
        )
        with pytest.raises(ValueError, match="arm_c"):
            _run_entry(
                tmp_path, monkeypatch, case_ids=case_ids, queue=[],
                extra_argv=["--arm-c", "--scoring-policy", "v3",
                            "--exclude-from", str(tmp_path / "seg1.json")],
                output_name="seg2.json",
            )

    def test_merge_preserves_the_policy_and_refuses_a_mismatch(
        self, tmp_path, monkeypatch
    ):
        case_ids = [f"C{i:02d}" for i in range(4)]
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids[:2] for r in _completing_plan(c)],
            extra_argv=["--arm-c", "--scoring-policy", "v3", "--n-cases", "2"],
            output_name="seg1.json", arm_c_queue=[r for c in case_ids[:2] for r in _completing_plan(c)],
        )
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[r for c in case_ids[2:] for r in _completing_plan(c)],
            extra_argv=["--arm-c", "--scoring-policy", "v3",
                        "--exclude-from", str(tmp_path / "seg1.json")],
            output_name="seg2.json", arm_c_queue=[r for c in case_ids[2:] for r in _completing_plan(c)],
        )
        _, merged = _run_merge(
            tmp_path, monkeypatch,
            segments=[tmp_path / "seg1.json", tmp_path / "seg2.json"],
            sample_path=tmp_path / "sample.json", output_name="merged.json",
        )
        assert merged["metrics"]["scoring_policy"] == "v3"
        assert merged["metrics"]["arm_c"] is not None
        assert merged["metrics"]["scoring_policy"] == merged["config"]["scoring_policy"]

    def test_legacy_reports_without_a_policy_field_still_merge(self):
        """v1/v2 carry no policy; two of them agree by definition."""
        from scripts.compare_pipelines import _compare_identity, _experiment_identity
        v1 = {
            "sample": {"seed": 1, "total": 2, "case_ids_sha256": "a",
                       "gold_sha256": "b"},
            "input_hashes": {"sample_file_sha256": "c", "files": {}},
            "config": dict.fromkeys(CONFIG_FIELDS_FOR_TEST, "x"),
        }
        assert _compare_identity(
            _experiment_identity(v1), "v1a", _experiment_identity(v1), "v1b"
        ) == []


class TestAcceptanceCompleteness:
    """Completeness is judged against the frozen sample, not the slice that ran."""

    def _ab_queue(self, subset):
        return [r for c in subset for r in _completing_plan(c)]

    def _run(self, tmp_path, monkeypatch, case_ids, subset, *, extra=(), out="r.json",
             queues=None):
        agent_q, c_q = queues if queues else (self._ab_queue(subset), self._ab_queue(subset))
        return _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids, queue=agent_q,
            arm_c_queue=c_q, extra_argv=["--arm-c", "--scoring-policy", "v3", *extra],
            output_name=out,
        )

    def test_trial_on_two_of_fifty_gives_no_acceptance_conclusion(
        self, tmp_path, monkeypatch
    ):
        """--n-cases 2 on a 50-case frozen sample is a trial, not the experiment."""
        case_ids = [f"C{i:02d}" for i in range(50)]
        code, report = self._run(
            tmp_path, monkeypatch, case_ids, case_ids[:2],
            extra=["--n-cases", "2"],
        )
        assert code == 0
        acc = report["acceptance"]
        assert acc["status"] == "experiment_incomplete"
        assert acc["criteria_met"] is None
        assert acc["frozen_total"] == 50
        assert acc["attempted"] == 2
        assert acc["not_run"] == 48
        assert acc["scoring_policy"] == "v3"

    def test_continuation_segment_alone_gives_no_acceptance_conclusion(
        self, tmp_path, monkeypatch
    ):
        """A segment that only covers the remainder has not attempted the sample."""
        case_ids = [f"C{i:02d}" for i in range(4)]
        self._run(tmp_path, monkeypatch, case_ids, case_ids[:2],
                  extra=["--n-cases", "2"], out="seg1.json")
        _, seg2 = self._run(
            tmp_path, monkeypatch, case_ids, case_ids[2:],
            extra=["--exclude-from", str(tmp_path / "seg1.json")], out="seg2.json",
        )
        assert seg2["acceptance"]["status"] == "experiment_incomplete"
        assert seg2["acceptance"]["criteria_met"] is None
        assert seg2["acceptance"]["not_run"] == 2

    def test_zero_case_start_reports_incomplete_not_inapplicable(
        self, tmp_path, monkeypatch
    ):
        """A budget stop before the first case is an incomplete v3 experiment."""
        case_ids = [f"C{i:02d}" for i in range(4)]
        code, report = self._run(
            tmp_path, monkeypatch, case_ids, case_ids,
            extra=["--experiment-budget-usd", "0"],
            queues=([], []),
        )
        assert code == 3
        acc = report["acceptance"]
        assert acc["status"] == "experiment_incomplete", (
            "Arm C was enabled; the comparison is incomplete, not inapplicable"
        )
        assert acc["criteria_met"] is None
        assert acc["attempted"] == 0
        assert acc["arm_c_enabled"] is True
        # The policy must survive a run in which nothing happened.
        assert acc["scoring_policy"] == "v3"
        assert report["metrics"]["scoring_policy"] == "v3"
        assert report["config"]["scoring_policy"] == "v3"

    def test_failed_cases_do_not_make_the_run_incomplete(self, tmp_path, monkeypatch):
        """A case that ran and failed is attempted and is scored by the rules."""
        case_ids = ["C00"]
        _, report = self._run(
            tmp_path, monkeypatch, case_ids, case_ids,
            queues=([_llm_response(), RuntimeError("APIConnectionError: boom")],
                    [_llm_response(), RuntimeError("APIConnectionError: boom")]),
        )
        acc = report["acceptance"]
        assert report["plan"]["attempted"] == 1
        assert acc["attempted"] == 1, "a failed case still reached a run status"
        assert acc["not_run"] == 0
        assert acc["status"] == "complete", (
            "incompleteness is about cases that never started, not about failures"
        )

    def test_partial_merge_keeps_criteria_met_null(self, tmp_path, monkeypatch):
        case_ids = [f"C{i:02d}" for i in range(6)]
        self._run(tmp_path, monkeypatch, case_ids, case_ids[:2],
                  extra=["--n-cases", "2"], out="seg1.json")
        _, merged = _run_merge(
            tmp_path, monkeypatch, segments=[tmp_path / "seg1.json"],
            sample_path=tmp_path / "sample.json", output_name="merged.json",
        )
        acc = merged["acceptance"]
        assert acc["status"] == "experiment_incomplete"
        assert acc["criteria_met"] is None
        assert acc["frozen_total"] == 6
        assert acc["attempted"] == 2
        assert acc["not_run"] == 4

    def test_full_merge_recomputes_acceptance_and_keeps_the_fingerprint(
        self, tmp_path, monkeypatch
    ):
        """Two segments covering the whole sample yield a conclusion — recomputed,
        not copied from either segment."""
        case_ids = [f"C{i:02d}" for i in range(4)]
        _, seg1 = self._run(tmp_path, monkeypatch, case_ids, case_ids[:2],
                            extra=["--n-cases", "2"], out="seg1.json")
        _, seg2 = self._run(
            tmp_path, monkeypatch, case_ids, case_ids[2:],
            extra=["--exclude-from", str(tmp_path / "seg1.json")], out="seg2.json",
        )
        for segment in (seg1, seg2):
            assert segment["acceptance"]["status"] == "experiment_incomplete"

        _, merged = _run_merge(
            tmp_path, monkeypatch,
            segments=[tmp_path / "seg1.json", tmp_path / "seg2.json"],
            sample_path=tmp_path / "sample.json", output_name="merged.json",
        )
        acc = merged["acceptance"]
        assert acc["status"] == "complete"
        assert acc["frozen_total"] == 4
        assert acc["attempted"] == 4
        assert acc["not_run"] == 0
        assert acc["criteria_met"] is not None
        # The fingerprint the segments agreed on is carried through.
        fp = merged["prompt_fingerprint"]
        assert fp is not None and "agent" in fp and "arm_c" in fp
        assert fp == seg1["prompt_fingerprint"] == seg2["prompt_fingerprint"]
        assert merged["metrics"]["scoring_policy"] == "v3"


class TestUnscorableFollowsThePolicy:
    def test_legacy_treats_mixed_as_unscorable(self):
        from scripts.compare_pipelines import _is_unscorable
        assert _is_unscorable("legacy", "mixed") is True
        assert _is_unscorable("legacy", "insufficient") is False
        assert _is_unscorable("legacy", "supported") is False

    def test_v3_treats_mixed_as_a_real_prediction(self):
        from scripts.compare_pipelines import _is_unscorable
        assert _is_unscorable("v3", "mixed") is False, (
            "v3 maps mixed to maybe, so it is scorable"
        )
        assert _is_unscorable("v3", "insufficient") is False
        assert _is_unscorable("v3", "nonsense") is True

    def test_exported_json_matches_the_policy(self, tmp_path, monkeypatch):
        case_ids = ["C00"]
        _, v3_report = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=_completing_plan("C00", verdict="mixed"),
            arm_c_queue=_completing_plan("C00", verdict="mixed"),
            extra_argv=["--arm-c", "--scoring-policy", "v3"], output_name="v3.json",
        )
        v3_case = v3_report["cases"][0]
        assert v3_case["llm_verdict"] == "mixed"
        assert v3_case["llm_verdict_unscorable"] is False
        assert v3_case["arm_c_verdict_unscorable"] is False

        _, legacy_report = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=_completing_plan("C00", verdict="mixed"),
            extra_argv=[], output_name="legacy.json",
        )
        legacy_case = legacy_report["cases"][0]
        assert legacy_case["llm_verdict"] == "mixed"
        assert legacy_case["llm_verdict_unscorable"] is True, (
            "legacy must keep its established meaning for mixed"
        )

    def test_merged_json_matches_the_policy(self, tmp_path, monkeypatch):
        case_ids = ["C00"]
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=_completing_plan("C00", verdict="mixed"),
            arm_c_queue=_completing_plan("C00", verdict="mixed"),
            extra_argv=["--arm-c", "--scoring-policy", "v3"], output_name="seg.json",
        )
        _, merged = _run_merge(
            tmp_path, monkeypatch, segments=[tmp_path / "seg.json"],
            sample_path=tmp_path / "sample.json", output_name="merged.json",
        )
        assert merged["cases"][0]["llm_verdict_unscorable"] is False
        assert merged["cases"][0]["arm_c_verdict_unscorable"] is False


# ---------------------------------------------------------------------------
# Cross-segment cost aggregation must cover every paid arm, including Arm C
# ---------------------------------------------------------------------------

_ARM_C_CASE_COST = _AGENT_CASE_COST          # C runs the same loop as A
_BOTH_AC_CASE_COST = _AGENT_CASE_COST + _ARM_C_CASE_COST


class TestArmCIsInTheCrossSegmentSubtotal:
    def _queue(self, subset, verdict=YES):
        return [r for c in subset for r in _completing_plan(c, verdict=verdict)]

    def test_case_subtotal_sums_all_three_arms(self):
        from scripts.compare_pipelines import _case_known_subtotal
        case = {
            "llm_cost_known_subtotal_usd": 0.001,
            "baseline_cost_known_subtotal_usd": 0.002,
            "arm_c_cost_known_subtotal_usd": 0.004,
        }
        assert _case_known_subtotal(case) == pytest.approx(0.007)

    def test_absent_arms_contribute_nothing_and_are_not_imputed(self):
        from scripts.compare_pipelines import _case_known_subtotal
        assert _case_known_subtotal({"llm_cost_known_subtotal_usd": 0.001}) == (
            pytest.approx(0.001)
        ), "a report with only the agent arm sums to the agent arm alone"
        assert _case_known_subtotal({
            "llm_cost_known_subtotal_usd": 0.001,
            "arm_c_cost_known_subtotal_usd": None,
        }) == pytest.approx(0.001), "an unknown arm contributes no invented number"

    def test_carry_over_includes_arm_c(self, tmp_path, monkeypatch):
        case_ids = [f"C{i:02d}" for i in range(4)]
        _, seg1 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=self._queue(case_ids[:1]), arm_c_queue=self._queue(case_ids[:1]),
            extra_argv=["--arm-c", "--scoring-policy", "v3", "--n-cases", "1"],
            output_name="seg1.json",
        )
        assert seg1["cases"][0]["llm_cost_known_subtotal_usd"] == pytest.approx(
            _AGENT_CASE_COST, abs=1e-6)
        assert seg1["cases"][0]["arm_c_cost_known_subtotal_usd"] == pytest.approx(
            _ARM_C_CASE_COST, abs=1e-6)

        code, seg2 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=self._queue(case_ids[1:]), arm_c_queue=self._queue(case_ids[1:]),
            extra_argv=["--arm-c", "--scoring-policy", "v3",
                        "--exclude-from", str(tmp_path / "seg1.json")],
            output_name="seg2.json",
        )
        assert code == 0
        assert seg2["config"]["carry_over_cost_usd"] == pytest.approx(
            _BOTH_AC_CASE_COST, abs=1e-6
        ), "the carry-over must sum Arm A and Arm C, not Arm A alone"

    def test_a_reached_threshold_makes_no_new_requests(self, tmp_path, monkeypatch):
        """Codex's reproduction: the carried A+C spend alone exhausts the budget."""
        case_ids = [f"C{i:02d}" for i in range(4)]
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=self._queue(case_ids[:1]), arm_c_queue=self._queue(case_ids[:1]),
            extra_argv=["--arm-c", "--scoring-policy", "v3", "--n-cases", "1"],
            output_name="seg1.json",
        )
        # 0.00252 sits between the buggy carry-over (A only) and the correct one.
        code, seg2 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=[], arm_c_queue=[],
            extra_argv=["--arm-c", "--scoring-policy", "v3",
                        "--exclude-from", str(tmp_path / "seg1.json"),
                        "--experiment-budget-usd", "0.00252"],
            output_name="seg2.json",
        )
        assert code == 3
        assert seg2["plan"]["attempted"] == 0, "no new case may start"
        assert seg2["plan"]["not_run"] == 3
        assert seg2["plan"]["stop_detail"]["reason"] == "cost_threshold"
        assert seg2["config"]["carry_over_cost_usd"] == pytest.approx(
            _BOTH_AC_CASE_COST, abs=1e-6
        )

    def test_a_threshold_above_the_carried_spend_still_runs(self, tmp_path, monkeypatch):
        case_ids = [f"C{i:02d}" for i in range(4)]
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=self._queue(case_ids[:1]), arm_c_queue=self._queue(case_ids[:1]),
            extra_argv=["--arm-c", "--scoring-policy", "v3", "--n-cases", "1"],
            output_name="seg1.json",
        )
        code, seg2 = _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=self._queue(case_ids[1:]), arm_c_queue=self._queue(case_ids[1:]),
            extra_argv=["--arm-c", "--scoring-policy", "v3",
                        "--exclude-from", str(tmp_path / "seg1.json"),
                        "--experiment-budget-usd", "1000"],
            output_name="seg2.json",
        )
        assert code == 0
        assert seg2["plan"]["attempted"] == 3
        assert seg2["config"]["carry_over_cost_usd"] == pytest.approx(
            _BOTH_AC_CASE_COST, abs=1e-6
        )

    def test_merge_subtotal_equals_the_original_per_case_a_plus_c(self, tmp_path, monkeypatch):
        """Chained segments must not count an earlier segment's spend twice."""
        case_ids = [f"C{i:02d}" for i in range(4)]
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=self._queue(case_ids[:2]), arm_c_queue=self._queue(case_ids[:2]),
            extra_argv=["--arm-c", "--scoring-policy", "v3", "--n-cases", "2"],
            output_name="seg1.json",
        )
        _run_entry(
            tmp_path, monkeypatch, case_ids=case_ids,
            queue=self._queue(case_ids[2:]), arm_c_queue=self._queue(case_ids[2:]),
            extra_argv=["--arm-c", "--scoring-policy", "v3",
                        "--exclude-from", str(tmp_path / "seg1.json")],
            output_name="seg2.json",
        )
        _, merged = _run_merge(
            tmp_path, monkeypatch,
            segments=[tmp_path / "seg1.json", tmp_path / "seg2.json"],
            sample_path=tmp_path / "sample.json", output_name="merged.json",
        )
        # Four cases ran in total, each paying both arms.
        assert merged["merge_notes"]["known_subtotal_usd"] == pytest.approx(
            4 * _BOTH_AC_CASE_COST, abs=1e-6
        ), "the merged subtotal must be the sum of the original per-case A+C costs"
        for segment in merged["segments"]:
            assert segment["known_subtotal_usd"] == pytest.approx(
                2 * _BOTH_AC_CASE_COST, abs=1e-6
            )
        # A naive sum of segment figures including carry-over would be larger.
        naive = (merged["segments"][0]["known_subtotal_usd"]
                 + merged["segments"][0]["known_subtotal_usd"]
                 + merged["segments"][1]["known_subtotal_usd"])
        assert merged["merge_notes"]["known_subtotal_usd"] < naive

    def test_legacy_and_v2_aggregates_are_unchanged(self):
        """v1 sums Arm A only and v2 sums A+B; neither gains an Arm C term."""
        from scripts.compare_pipelines import _case_known_subtotal
        v1_case = {"llm_cost_known_subtotal_usd": 0.001}
        v2_case = {"llm_cost_known_subtotal_usd": 0.001,
                   "baseline_cost_known_subtotal_usd": 0.002}
        assert _case_known_subtotal(v1_case) == pytest.approx(0.001)
        assert _case_known_subtotal(v2_case) == pytest.approx(0.003)

    def test_real_v1_and_v2_reports_aggregate_exactly_as_before(self):
        """Against the stored reports: the sums must not move."""
        from scripts.compare_pipelines import _case_known_subtotal

        v1_dir = REPO_ROOT / "private/env_precheck"
        expected_v1 = {
            "pipeline_comparison_trial5.json": 0.021089,
            "pipeline_comparison_remaining45.json": 0.115123,
            "pipeline_comparison_remaining13.json": 0.050318,
        }
        for name, expected in expected_v1.items():
            path = v1_dir / name
            if not path.exists():  # pragma: no cover - machine-local artifacts
                pytest.skip("stored v1 reports not present")
            report = json.loads(path.read_text())
            assert sum(_case_known_subtotal(c) for c in report["cases"]) == (
                pytest.approx(expected, abs=1e-6)
            ), f"{name} v1 subtotal moved"

        expected_v2 = {
            "v2_ab_trial5.json": 0.022142,
            "v2_ab_remaining45.json": 0.218375,
        }
        for name, expected in expected_v2.items():
            path = v1_dir / name
            if not path.exists():  # pragma: no cover
                pytest.skip("stored v2 reports not present")
            report = json.loads(path.read_text())
            assert sum(_case_known_subtotal(c) for c in report["cases"]) == (
                pytest.approx(expected, abs=1e-6)
            ), f"{name} v2 subtotal moved"


# ---------------------------------------------------------------------------
# Report produced by the real entry point
# ---------------------------------------------------------------------------

class TestReportFromRealEntrypoint:
    def test_dry_run_report_is_written_and_readable(self, tmp_path, monkeypatch):
        from scripts.compare_pipelines import main

        output = tmp_path / "report.json"
        monkeypatch.setattr(
            sys, "argv",
            ["compare_pipelines", "--dry-run", "--output", str(output)],
        )
        main()

        report = json.loads(output.read_text(encoding="utf-8"))
        assert set(report) >= {"run_date", "config", "sample", "plan", "metrics", "cases"}
        plan = report["plan"]
        assert plan["planned"] == len(report["cases"])
        assert plan["attempted"] == plan["planned"]
        assert plan["partial"] is False
        assert plan["stop_reason"] is None
        # Config travels with the numbers so a reader can see what produced them.
        assert report["config"]["temperature"] == 0.0
        assert "disabled" in report["config"]["thinking"]
        # Every case carries the cost provenance needed to read the total honestly.
        for case in report["cases"]:
            assert "llm_cost_unknown" in case
            assert "llm_cost_status" in case
            assert "llm_verdict_unscorable" in case

    def test_sample_file_restricts_the_run_to_exactly_the_sampled_ids(
        self, tmp_path, monkeypatch
    ):
        """The frozen sample must be the set actually evaluated — a run that
        silently widened it would not be the declared experiment."""
        import scripts.compare_pipelines as cp

        all_cases = [
            {"case_id": f"PUBMEDQA-TEST-{i:04d}", "question": f"Q{i}?",
             "gold_label": "yes", "gold_pmid": "1001"}
            for i in range(1, 6)
        ]
        docs, _ = _tiny_comparison_inputs(1)
        monkeypatch.setattr(cp, "_load_dry_run_data", lambda: (docs, all_cases))

        sampled = ["PUBMEDQA-TEST-0003", "PUBMEDQA-TEST-0001"]
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(json.dumps({
            "seed": 20260729,
            "total": 2,
            "case_ids": sampled,
            "gold_sha256": "a" * 64,
            "apportionment": {"strata": []},
        }), encoding="utf-8")

        captured: dict[str, Any] = {}

        def _capture(**kwargs):
            captured["case_ids"] = [c["case_id"] for c in kwargs["cases"]]
            return []

        monkeypatch.setattr(cp, "run_comparison", _capture)

        output = tmp_path / "report.json"
        monkeypatch.setattr(
            sys, "argv",
            ["compare_pipelines", "--dry-run", "--sample-file", str(sample_path),
             "--output", str(output)],
        )
        cp.main()

        assert captured["case_ids"] == sampled, (
            "the runner must evaluate the sampled case IDs in the sample's order"
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["sample"]["seed"] == 20260729
        assert report["sample"]["total"] == 2
        assert len(report["sample"]["case_ids_sha256"]) == 64
        assert report["plan"]["planned"] == 2
        # This stub evaluated nothing.  The report must still be written and must
        # say so plainly rather than crashing or implying the cases ran.
        assert report["plan"]["attempted"] == 0
        assert report["plan"]["not_run"] == 2
        assert report["plan"]["partial"] is True
        assert "fewer cases than planned" in report["plan"]["stop_reason"]
        assert report["metrics"]["n_cases"] == 0
        assert report["metrics"]["llm_agent"]["macro_f1"] is None

    def test_budget_cap_produces_a_partial_report_through_main(self, tmp_path, monkeypatch):
        """A spending stop must leave a readable report that separates the cases
        that ran from the cases that never started."""
        import scripts.compare_pipelines as cp

        docs, cases = _tiny_comparison_inputs(5)
        monkeypatch.setattr(cp, "_load_dry_run_data", lambda: (docs, cases))

        results = [
            _case_result(case_id="C0", gold_label="yes", llm_verdict=YES, llm_cost_usd=0.02),
            _case_result(case_id="C1", gold_label="yes", llm_verdict=YES, llm_cost_usd=0.02),
        ]

        def _stub(**kwargs):
            # Mirror what the real runner records when the cap trips.
            kwargs["stop_state"].update({
                "reason": "cost_threshold",
                "stopped_before_index": 2,
                "cases_completed_before_stop": 2,
                "known_subtotal_usd": 0.04,
                "n_incomplete": 0,
            })
            return results

        monkeypatch.setattr(cp, "run_comparison", _stub)

        output = tmp_path / "report.json"
        monkeypatch.setattr(
            sys, "argv",
            ["compare_pipelines", "--dry-run", "--max-cost-usd", "0.05",
             "--output", str(output)],
        )
        cp.main()

        report = json.loads(output.read_text(encoding="utf-8"))
        plan = report["plan"]
        assert plan["planned"] == 5
        assert plan["attempted"] == 2
        assert plan["completed"] == 2
        assert plan["failed"] == 0
        assert plan["not_run"] == 3
        assert plan["partial"] is True
        assert plan["stop_reason"] == "max_cost_usd reached before the next case started"
        assert report["config"]["max_cost_usd"] == 0.05
        # 3 of 5 cases never ran: the report must not read as a 5-case result.
        assert report["metrics"]["n_cases"] == 2

    def test_n_cases_slices_the_frozen_sample_not_the_corpus(self, tmp_path, monkeypatch):
        """--n-cases must mean "first N of the sample".  Slicing the corpus first
        would intersect two different sets and evaluate the wrong cases."""
        import scripts.compare_pipelines as cp

        docs, cases = _tiny_comparison_inputs(10)
        monkeypatch.setattr(cp, "_load_dry_run_data", lambda: (docs, cases))

        sample = ["C7", "C3", "C5", "C1", "C9"]
        sample_path = tmp_path / "s.json"
        sample_path.write_text(json.dumps({"case_ids": sample}), encoding="utf-8")

        captured: dict[str, Any] = {}

        def _capture(**kwargs):
            captured["ids"] = [c["case_id"] for c in kwargs["cases"]]
            return []

        monkeypatch.setattr(cp, "run_comparison", _capture)
        monkeypatch.setattr(
            sys, "argv",
            ["compare_pipelines", "--dry-run", "--sample-file", str(sample_path),
             "--n-cases", "3", "--output", str(tmp_path / "r.json")],
        )
        cp.main()

        assert captured["ids"] == ["C7", "C3", "C5"]

    def test_sample_naming_an_unknown_case_id_fails_loudly(self, tmp_path, monkeypatch):
        import scripts.compare_pipelines as cp

        sample_docs, _ = _tiny_comparison_inputs(1)
        monkeypatch.setattr(
            cp, "_load_dry_run_data",
            lambda: (sample_docs, [{"case_id": "PUBMEDQA-TEST-0001", "question": "Q?",
                                    "gold_label": "yes", "gold_pmid": "1001"}]),
        )
        sample_path = tmp_path / "sample.json"
        sample_path.write_text(
            json.dumps({"case_ids": ["PUBMEDQA-TEST-9999"]}), encoding="utf-8"
        )
        monkeypatch.setattr(
            sys, "argv",
            ["compare_pipelines", "--dry-run", "--sample-file", str(sample_path),
             "--output", str(tmp_path / "report.json")],
        )
        with pytest.raises(KeyError, match="PUBMEDQA-TEST-9999"):
            cp.main()
