"""Offline simulation tests for scripts/live_smoke_test.py.

These tests replace the real OpenAI client with lightweight mock objects so that
all four outcome paths — completed, budget_exhausted, text_exit, exception — and
two additional failure modes — corpus failure and missing usage — can be exercised
without any network activity.  No API key or paid call is made.

What is tested:
  configuration_error — missing DEEPSEEK_API_KEY returns immediately, no network
    call, no OpenAI constructor invoked.
  completed — search→fetch→inspect→finish mock sequence; report file written,
    outcome=="completed", raw_llm_rounds contains snapshotted request messages
    (with content), tool results are recoverable from the message sequence, cost
    is non-None and >0.
  budget_exhausted — repeated single-query mock hits max_steps=6 exactly;
    run_status=="budget_exhausted", 6 rounds recorded.
  text_exit — plain-text response (no tool_calls) terminates after 1 round;
    run_status=="text_exit".
  exception (APIError mid-run) — partial rounds preserved with error dict;
    written JSON is valid; the pre-error round has a response and the error round
    has error.type=="APIError".
  missing usage — when response.usage is None, round["response"]["usage"]
    is {"status": "unknown"}, NOT None or zero tokens.
  corpus failure — read_corpus raising FileNotFoundError produces
    outcome=="exception" with error.type=="FileNotFoundError" in the returned
    report (no crash, no unhandled exception).
  report file always written — main() always calls Path.write_text with a path
    ending in reports/live_smoke_test_v1.json on the completed path.

What is NOT tested here:
  Real model decisions — the mock replaces the model entirely.
  Actual DeepSeek API connectivity — live smoke test only.
  Citation quality or answer correctness.

Gap 3 note: DeepSeek Flash uses non-thinking mode when tool_choice is set.
  openai SDK v3.x does not expose reasoning_content on ChatCompletionMessage.
  No reasoning_content handling is needed or tested here.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

# ── project path bootstrap ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import importlib
import importlib.util

smoke_mod_path = str(PROJECT_ROOT / "scripts" / "live_smoke_test.py")
spec = importlib.util.spec_from_file_location("live_smoke_test", smoke_mod_path)
assert spec is not None and spec.loader is not None
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)  # type: ignore[union-attr]


# ── mock primitives ───────────────────────────────────────────────────────────

def _make_tool_call(name: str, arguments: dict[str, Any], tc_id: str = "tc0") -> Any:
    fn = types.SimpleNamespace(name=name, arguments=json.dumps(arguments))
    return types.SimpleNamespace(id=tc_id, function=fn)


def _make_msg(tool_calls: list[Any] | None, content: str | None = None) -> Any:
    msg = types.SimpleNamespace(
        tool_calls=tool_calls,
        content=content,
        role="assistant",
    )
    def model_dump(*, exclude_none: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.tool_calls is not None:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d
    msg.model_dump = model_dump
    return msg


def _make_usage(prompt: int = 300, completion: int = 100) -> Any:
    return types.SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def _make_response(
    msg: Any,
    usage: Any | None = None,
    model: str = "deepseek-flash",
    *,
    no_usage: bool = False,
) -> Any:
    """Build a minimal ChatCompletion-like object.

    Pass no_usage=True to simulate a response where .usage is None (usage absent).
    """
    choice = types.SimpleNamespace(
        message=msg,
        finish_reason="tool_calls" if msg.tool_calls else "stop",
    )
    actual_usage = None if no_usage else (usage or _make_usage())
    return types.SimpleNamespace(
        id="resp-mock",
        model=model,
        choices=[choice],
        usage=actual_usage,
    )


class _SequentialMockClient:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = iter(responses)
        self.chat = self
        self.completions = self

    def create(self, **_kwargs: Any) -> Any:
        return next(self._responses)


class _RaisingMockClient:
    def __init__(self, responses: list[Any], then_raise: Exception) -> None:
        self._responses = iter(responses)
        self._raise = then_raise
        self.chat = self
        self.completions = self

    def create(self, **_kwargs: Any) -> Any:
        try:
            return next(self._responses)
        except StopIteration:
            raise self._raise


# ── shared completed-path mock sequence ──────────────────────────────────────

def _build_completed_responses() -> list[Any]:
    """search → fetch → inspect → finish accepted by _dispatch."""
    return [
        _make_response(_make_msg([_make_tool_call(
            "search_literature", {"query": "statins cardiovascular", "top_k": 3}, "tc1"
        )])),
        _make_response(_make_msg([_make_tool_call(
            "fetch_record", {"pmid": "1004"}, "tc2"
        )])),
        _make_response(_make_msg([_make_tool_call(
            "inspect_evidence", {"pmid": "1004", "query": "statins cardiovascular events"}, "tc3"
        )])),
        _make_response(_make_msg([_make_tool_call("finish", {
            "verdict": "supported",
            "answer": "RCT evidence supports statin use for CV event reduction.",
            "claim": "Statins reduce major cardiovascular events in RCTs.",
            "cited_pmids": ["1004"],
            "decisive_reason": "PMID 1004 directly reports significant RCT results.",
        }, "tc4")])),
    ]


def _make_fake_openai(mock_client: Any) -> type:
    class _FakeOpenAI:
        def __init__(self, **_kw: Any) -> None:
            pass
        def __getattr__(self, name: str) -> Any:
            return getattr(mock_client, name)
    return _FakeOpenAI


def _run_with_mock(mock_client: Any, *, api_key: str = "sk-test-offline") -> dict[str, Any]:
    FakeOpenAI = _make_fake_openai(mock_client)
    with (
        patch.dict("os.environ", {"DEEPSEEK_API_KEY": api_key}),
        patch("openai.OpenAI", FakeOpenAI),
        patch.object(Path, "write_text", lambda *a, **k: None),
        patch.object(Path, "mkdir", lambda *a, **k: None),
    ):
        return smoke.run_smoke_test()


# ── tests ─────────────────────────────────────────────────────────────────────

class TestConfigurationError:
    def test_missing_api_key_returns_configuration_error(self) -> None:
        import os
        os.environ.pop("DEEPSEEK_API_KEY", None)
        with patch.dict("os.environ", {}, clear=True):
            report = smoke.run_smoke_test()
        assert report["outcome"] == "configuration_error"
        assert "DEEPSEEK_API_KEY" in report["error"]
        assert report["raw_llm_rounds"] == []

    def test_missing_api_key_no_network_attempted(self) -> None:
        import os
        os.environ.pop("DEEPSEEK_API_KEY", None)
        constructed: list[bool] = []

        class _TrackingOpenAI:
            def __init__(self, **_kw: Any) -> None:
                constructed.append(True)

        with patch("openai.OpenAI", _TrackingOpenAI), patch.dict("os.environ", {}, clear=True):
            smoke.run_smoke_test()

        assert constructed == [], "OpenAI constructor was called despite missing API key"


class TestCompletedPath:
    def test_completed_outcome_fields(self) -> None:
        mock_client = _SequentialMockClient(_build_completed_responses())
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "completed", (
            f"Expected 'completed', got {report['outcome']!r}. "
            f"contract_errors={report.get('agent_response_contract_errors')}"
        )
        assert report["agent_response_contract_valid"] is True
        assert report["agent_response_contract_errors"] == []

        agent_run = report["agent_response"]["provenance"]["agent_run"]
        assert agent_run["run_status"] == "completed"

    def test_audit_rounds_contain_request_message_content(self) -> None:
        """Each round must snapshot the actual messages sent, not just a count."""
        mock_client = _SequentialMockClient(_build_completed_responses())
        report = _run_with_mock(mock_client)

        assert len(report["raw_llm_rounds"]) >= 4

        round_0 = report["raw_llm_rounds"][0]
        # request_messages must be a non-empty list of dicts with 'role' and 'content'
        assert "request_messages" in round_0, (
            "round record must contain 'request_messages' snapshot"
        )
        msgs = round_0["request_messages"]
        assert isinstance(msgs, list) and len(msgs) >= 2, (
            f"Expected at least system+user messages, got {msgs!r}"
        )
        roles = [m.get("role") for m in msgs]
        assert "system" in roles, f"system message missing from round 0 snapshot: {roles}"
        assert "user" in roles, f"user message missing from round 0 snapshot: {roles}"

    def test_audit_rounds_snapshots_do_not_mutate_across_calls(self) -> None:
        """Round N's snapshot must not change when round N+1 appends tool results."""
        mock_client = _SequentialMockClient(_build_completed_responses())
        report = _run_with_mock(mock_client)

        rounds = report["raw_llm_rounds"]
        assert len(rounds) >= 2

        # Round 0 should have 2 messages (system + user).
        # Round 1 should have more (system + user + assistant tool-call + tool result).
        count_r0 = rounds[0]["request_messages_count"]
        count_r1 = rounds[1]["request_messages_count"]
        assert count_r0 < count_r1, (
            f"Round 0 message count ({count_r0}) should be less than round 1 ({count_r1}), "
            "indicating that round 0 is a snapshot not affected by later appends."
        )

    def test_tool_results_recoverable_from_message_sequence(self) -> None:
        """Tool results sent back to the model must be present in round N+1's messages."""
        mock_client = _SequentialMockClient(_build_completed_responses())
        report = _run_with_mock(mock_client)

        rounds = report["raw_llm_rounds"]
        assert len(rounds) >= 2

        # Round 1's request_messages should contain a tool-role message (the result
        # of the search_literature call from round 0).
        r1_messages = rounds[1]["request_messages"]
        tool_msgs = [m for m in r1_messages if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1, (
            f"Round 1 messages should include at least one tool-result message; "
            f"roles found: {[m.get('role') for m in r1_messages]}"
        )
        # The tool result content should be JSON with 'hits' key (search result)
        tool_content = tool_msgs[0].get("content", "")
        parsed = json.loads(tool_content)
        assert "hits" in parsed, (
            f"Tool result for search_literature should contain 'hits', got: {parsed}"
        )

    def test_usage_status_confirmed_when_present(self) -> None:
        """When usage is returned by the API, usage.status must be 'confirmed'."""
        mock_client = _SequentialMockClient(_build_completed_responses())
        report = _run_with_mock(mock_client)

        for rd in report["raw_llm_rounds"]:
            if rd["response"] is not None:
                usage = rd["response"]["usage"]
                assert usage["status"] == "confirmed", (
                    f"Round {rd['call_index']}: expected usage.status='confirmed', got {usage!r}"
                )
                assert "prompt_tokens" in usage
                assert "completion_tokens" in usage

    def test_cost_non_none_for_known_model(self) -> None:
        """model_api_cost_usd must be non-None and > 0 for deepseek-flash."""
        mock_client = _SequentialMockClient(_build_completed_responses())
        report = _run_with_mock(mock_client)
        provenance = report["agent_response"]["provenance"]
        cost = provenance.get("model_api_cost_usd")
        assert cost is not None, "model_api_cost_usd is None for deepseek-flash"
        assert cost > 0

    def test_main_exits_zero_on_completed(self) -> None:
        mock_client = _SequentialMockClient(_build_completed_responses())
        FakeOpenAI = _make_fake_openai(mock_client)
        with (
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test-offline"}),
            patch("openai.OpenAI", FakeOpenAI),
            patch.object(Path, "write_text", lambda *a, **k: None),
            patch.object(Path, "mkdir", lambda *a, **k: None),
        ):
            assert smoke.main() == 0


class TestMissingUsage:
    def test_missing_usage_recorded_as_unknown_not_zero(self) -> None:
        """When response.usage is None, audit record must show status='unknown', not zero tokens."""
        # One text-exit response with no usage
        no_usage_response = _make_response(
            _make_msg(tool_calls=None, content="I think statins help."),
            no_usage=True,
        )
        mock_client = _SequentialMockClient([no_usage_response])
        report = _run_with_mock(mock_client)

        assert len(report["raw_llm_rounds"]) == 1
        rd = report["raw_llm_rounds"][0]
        assert rd["response"] is not None
        usage = rd["response"]["usage"]
        assert usage["status"] == "unknown", (
            f"Expected usage status 'unknown', got {usage!r}. "
            "Zero tokens must not be written when usage data is absent."
        )
        # An absent usage object must not be described as a zero-token one: every
        # declared field has to read as absent, and no flat count may be invented.
        assert "prompt_tokens" not in usage, (
            f"Absent usage must not carry a flat token count, got {usage!r}"
        )
        states = {
            name: entry["state"] for name, entry in usage["fields"].items()
        }
        non_absent = {
            name: state for name, state in states.items() if state != "absent"
        }
        assert not non_absent, (
            "Every declared usage field must be marked absent when usage is None; "
            f"got {non_absent!r}"
        )
        assert usage["extension_fields"] == {}

    def test_missing_usage_does_not_crash_run(self) -> None:
        """A None-usage response must not cause run_smoke_test() to raise."""
        no_usage_response = _make_response(
            _make_msg(tool_calls=None, content="Some text."),
            no_usage=True,
        )
        mock_client = _SequentialMockClient([no_usage_response])
        report = _run_with_mock(mock_client)
        # Should produce text_exit, not exception
        assert report["outcome"] == "text_exit"


class TestBudgetExhaustedPath:
    def test_pure_budget_exhausted_exact_step_count(self) -> None:
        """Repeated identical search call hits duplicate-call guard; loop runs max_steps=6."""
        single_search_responses = [
            _make_response(_make_msg([_make_tool_call(
                "search_literature", {"query": "statins", "top_k": 3}, "tc0"
            )]))
            for _ in range(20)
        ]
        mock_client = _SequentialMockClient(single_search_responses)
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "budget_exhausted"
        assert len(report["raw_llm_rounds"]) == 6, (
            f"Expected exactly 6 rounds (max_steps), got {len(report['raw_llm_rounds'])}"
        )

    def test_budget_exhausted_rounds_all_have_responses(self) -> None:
        single_search_responses = [
            _make_response(_make_msg([_make_tool_call(
                "search_literature", {"query": "statins", "top_k": 3}, "tc0"
            )]))
            for _ in range(20)
        ]
        mock_client = _SequentialMockClient(single_search_responses)
        report = _run_with_mock(mock_client)

        for rd in report["raw_llm_rounds"]:
            assert rd["error"] is None, f"Round {rd['call_index']} has unexpected error"
            assert rd["response"] is not None


class TestTextExitPath:
    def test_text_exit_one_round_with_content(self) -> None:
        text_response = _make_response(
            _make_msg(tool_calls=None, content="Based on training, statins help.")
        )
        mock_client = _SequentialMockClient([text_response])
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "text_exit"
        agent_run = report["agent_response"]["provenance"]["agent_run"]
        assert agent_run["run_status"] == "text_exit"
        assert len(report["raw_llm_rounds"]) == 1

        rd = report["raw_llm_rounds"][0]
        # Response choice must record the text content and no tool_calls
        choice = rd["response"]["choices"][0]
        assert choice["message_content"] == "Based on training, statins help."
        assert choice["tool_calls"] is None

    def test_main_exits_nonzero_on_text_exit(self) -> None:
        text_response = _make_response(
            _make_msg(tool_calls=None, content="Just text.")
        )
        mock_client = _SequentialMockClient([text_response])
        FakeOpenAI = _make_fake_openai(mock_client)
        with (
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test-offline"}),
            patch("openai.OpenAI", FakeOpenAI),
            patch.object(Path, "write_text", lambda *a, **k: None),
            patch.object(Path, "mkdir", lambda *a, **k: None),
        ):
            assert smoke.main() != 0


class TestExceptionPath:
    def test_exception_preserves_partial_trace_and_error_dict(self) -> None:
        """APIError mid-run: one successful round then error; partial rounds preserved."""
        from openai import APIError
        fake_request = types.SimpleNamespace(method="POST", url="https://api.deepseek.com/v1")
        exc = APIError(message="Service unavailable", request=fake_request, body=None)

        first_response = _make_response(_make_msg([_make_tool_call(
            "search_literature", {"query": "statins", "top_k": 3}, "tc1"
        )]))
        mock_client = _RaisingMockClient([first_response], then_raise=exc)
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "exception"
        assert isinstance(report["error"], dict)
        assert report["error"]["type"] == "APIError"
        assert "traceback" in report["error"]

        # The successful round must be preserved
        rounds = report["raw_llm_rounds"]
        assert len(rounds) >= 1

        # First round: has response, no error
        first_round = rounds[0]
        assert first_round["response"] is not None
        assert first_round["error"] is None

        # Last round (the failing call): has error, no response
        last_round = rounds[-1]
        assert last_round["error"] is not None
        assert last_round["error"]["type"] == "APIError"
        assert last_round["response"] is None

    def test_exception_report_is_valid_json_when_written(self) -> None:
        from openai import APIError
        fake_request = types.SimpleNamespace(method="POST", url="https://api.deepseek.com/v1")
        exc = APIError(message="Boom", request=fake_request, body=None)
        mock_client = _RaisingMockClient([], then_raise=exc)
        FakeOpenAI = _make_fake_openai(mock_client)

        written_json: list[str] = []

        def _capture_write(self: Path, text: str, encoding: str = "utf-8") -> None:
            if "live_smoke_test" in self.name:
                written_json.append(text)

        with (
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test-offline"}),
            patch("openai.OpenAI", FakeOpenAI),
            patch.object(Path, "write_text", _capture_write),
            patch.object(Path, "mkdir", lambda *a, **k: None),
        ):
            smoke.main()

        assert len(written_json) == 1, "Expected exactly one write to the report file"
        parsed = json.loads(written_json[0])
        assert parsed["outcome"] == "exception"


class TestCorpusFailurePath:
    def test_corpus_read_failure_produces_structured_exception_report(self) -> None:
        """FileNotFoundError from read_corpus must yield outcome=='exception', not a crash."""
        # The smoke script imports read_corpus at module level; patch the name
        # in the smoke module's own namespace, not in bioevidence.corpus.
        with (
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}),
            patch.object(smoke, "read_corpus", side_effect=FileNotFoundError("corpus gone")),
            patch.object(Path, "write_text", lambda *a, **k: None),
            patch.object(Path, "mkdir", lambda *a, **k: None),
        ):
            report = smoke.run_smoke_test()

        assert report["outcome"] == "exception", (
            f"Expected 'exception' from corpus failure, got {report['outcome']!r}"
        )
        assert isinstance(report["error"], dict)
        assert report["error"]["type"] == "FileNotFoundError"
        assert "traceback" in report["error"]
        # No rounds: client was never built
        assert report["raw_llm_rounds"] == []

    def test_corpus_sha256_failure_produces_structured_report(self) -> None:
        """sha256_file raising also ends up in structured exception report."""
        with (
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}),
            patch.object(smoke, "sha256_file", side_effect=OSError("disk error")),
            patch.object(Path, "write_text", lambda *a, **k: None),
            patch.object(Path, "mkdir", lambda *a, **k: None),
        ):
            report = smoke.run_smoke_test()

        assert report["outcome"] == "exception"
        assert report["error"]["type"] == "OSError"


class TestReportFileAlwaysWritten:
    def test_report_file_written_on_success(self) -> None:
        mock_client = _SequentialMockClient(_build_completed_responses())
        FakeOpenAI = _make_fake_openai(mock_client)

        written_paths: list[Path] = []

        def _track(self: Path, text: str, encoding: str = "utf-8") -> None:
            written_paths.append(self)

        with (
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test-offline"}),
            patch("openai.OpenAI", FakeOpenAI),
            patch.object(Path, "write_text", _track),
            patch.object(Path, "mkdir", lambda *a, **k: None),
        ):
            smoke.main()

        target = next(
            (p for p in written_paths if "live_smoke_test_v1" in p.name), None
        )
        assert target is not None, (
            f"Expected live_smoke_test_v1.json to be written; paths were: {written_paths}"
        )
        assert target.parent.name == "reports"

    def test_budget_note_contains_estimate_not_upper_bound(self) -> None:
        """Verify the budget note uses 'estimate', not 'conservative upper bound'."""
        mock_client = _SequentialMockClient(_build_completed_responses())
        report = _run_with_mock(mock_client)
        budget_note = report.get("budget_note", "")
        assert "estimate" in budget_note.lower(), (
            f"budget_note should say 'estimate', not 'upper bound': {budget_note!r}"
        )
        assert "billing alert" not in budget_note.lower(), (
            "budget_note must not claim a billing alert provides a hard cap"
        )


class TestUsageSummarySemantics:
    """Issue 2: when usage is absent, cost must not be written as zero."""

    def test_all_unknown_usage_yields_none_cost(self) -> None:
        """All four rounds have no usage → model_api_cost_usd must be None, not 0."""
        responses = [
            _make_response(
                _make_msg([_make_tool_call(
                    "search_literature", {"query": "statins cardiovascular", "top_k": 3}, "tc1"
                )]),
                no_usage=True,
            ),
            _make_response(
                _make_msg([_make_tool_call("fetch_record", {"pmid": "1004"}, "tc2")]),
                no_usage=True,
            ),
            _make_response(
                _make_msg([_make_tool_call(
                    "inspect_evidence", {"pmid": "1004", "query": "statins cardiovascular events"}, "tc3"
                )]),
                no_usage=True,
            ),
            _make_response(
                _make_msg([_make_tool_call("finish", {
                    "verdict": "supported",
                    "answer": "RCT evidence supports statin use for CV event reduction.",
                    "claim": "Statins reduce major cardiovascular events in RCTs.",
                    "cited_pmids": ["1004"],
                    "decisive_reason": "PMID 1004 directly reports significant RCT results.",
                }, "tc4")]),
                no_usage=True,
            ),
        ]
        mock_client = _SequentialMockClient(responses)
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "completed", (
            f"Expected completed, got {report['outcome']!r}: "
            f"{report.get('agent_response_contract_errors')}"
        )
        provenance = report["agent_response"]["provenance"]
        cost = provenance.get("model_api_cost_usd")
        assert cost is None, (
            f"model_api_cost_usd must be None when all usage is unknown, got {cost!r}. "
            "Writing zero misrepresents absent data as a confirmed zero-cost run."
        )

    def test_partial_unknown_usage_yields_none_cost(self) -> None:
        """Some rounds have usage, some don't → cost must also be None (partial data)."""
        responses = [
            # Round 1: has usage
            _make_response(_make_msg([_make_tool_call(
                "search_literature", {"query": "statins cardiovascular", "top_k": 3}, "tc1"
            )])),
            # Round 2: no usage
            _make_response(
                _make_msg([_make_tool_call("fetch_record", {"pmid": "1004"}, "tc2")]),
                no_usage=True,
            ),
            _make_response(_make_msg([_make_tool_call(
                "inspect_evidence", {"pmid": "1004", "query": "statins cardiovascular events"}, "tc3"
            )])),
            _make_response(_make_msg([_make_tool_call("finish", {
                "verdict": "supported",
                "answer": "RCT evidence supports statin use for CV event reduction.",
                "claim": "Statins reduce major cardiovascular events in RCTs.",
                "cited_pmids": ["1004"],
                "decisive_reason": "PMID 1004 directly reports significant RCT results.",
            }, "tc4")])),
        ]
        mock_client = _SequentialMockClient(responses)
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "completed"
        provenance = report["agent_response"]["provenance"]
        cost = provenance.get("model_api_cost_usd")
        assert cost is None, (
            f"model_api_cost_usd must be None when any round has unknown usage, got {cost!r}."
        )


class TestTerminalRoundToolResults:
    """Issue 3: tool results from all rounds must be preserved after termination.

    Design note: ToolExecutor.trace records calls dispatched through the executor
    (search_literature, fetch_record, inspect_evidence, resolve_identifier).
    The finish tool is validated directly in _dispatch and intentionally not
    recorded in executor.trace — its acceptance is indicated by run_status==completed.

    Terminal-round tool results are recoverable via two mechanisms:
      1. executor.trace contains every non-finish tool call including the last one
         before termination (budget_exhausted, text_exit) or the accept (completed).
      2. The auditing client's request_messages snapshots for the final API call
         contain the tool-result messages from all prior rounds.
    """

    def test_finish_not_in_trace_but_all_prior_tools_are(self) -> None:
        """executor.trace covers search/fetch/inspect; finish is NOT in trace by design."""
        mock_client = _SequentialMockClient(_build_completed_responses())
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "completed"
        tool_trace = report["agent_response"]["provenance"]["tool_trace"]
        tool_names = [entry.get("tool_name") or entry.get("name") for entry in tool_trace]
        # All three non-finish tools must appear
        assert "search_literature" in tool_names, f"search_literature missing from trace: {tool_names}"
        assert "fetch_record" in tool_names, f"fetch_record missing from trace: {tool_names}"
        assert "inspect_evidence" in tool_names, f"inspect_evidence missing from trace: {tool_names}"
        # finish is intentionally not in trace; run_status captures its acceptance
        assert "finish" not in tool_names, (
            "finish must NOT be in executor.trace — its acceptance is recorded via run_status"
        )
        # run_status==completed confirms finish was accepted
        run_status = report["agent_response"]["provenance"]["agent_run"]["run_status"]
        assert run_status == "completed", f"Expected completed, got {run_status!r}"

    def test_budget_exhausted_all_tool_results_in_trace(self) -> None:
        """On budget_exhausted, all tool calls including the last must be in tool_trace."""
        responses = [
            _make_response(_make_msg([_make_tool_call(
                "search_literature", {"query": f"statins q{i}", "top_k": 3}, f"tc{i}"
            )]))
            for i in range(6)
        ]
        mock_client = _SequentialMockClient(responses)
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "budget_exhausted"
        tool_trace = report["agent_response"]["provenance"]["tool_trace"]
        search_calls = [e for e in tool_trace if (e.get("tool_name") or e.get("name")) == "search_literature"]
        assert len(search_calls) >= 1, (
            f"search_literature calls must appear in tool_trace on budget_exhausted; "
            f"trace: {tool_trace}"
        )

    def test_exception_mid_run_partial_trace_preserved(self) -> None:
        """On exception, the audit rounds before the error must be preserved in raw_llm_rounds.

        Note: agent_response is None on exception (run never completed).
        Tool results before the exception are recoverable from raw_llm_rounds[N].request_messages.
        """
        from openai import APIError
        fake_request = types.SimpleNamespace(method="POST", url="https://api.deepseek.com/v1")
        exc = APIError(message="Service unavailable", request=fake_request, body=None)

        first_response = _make_response(_make_msg([_make_tool_call(
            "search_literature", {"query": "statins", "top_k": 3}, "tc1"
        )]))
        mock_client = _RaisingMockClient([first_response], then_raise=exc)
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "exception"
        # agent_response is None on exception — _build_response never ran
        assert report["agent_response"] is None
        # The successful round must still be in raw_llm_rounds
        rounds = report["raw_llm_rounds"]
        assert len(rounds) >= 1, "Successful round before exception must be preserved"
        # Round 0: search tool call was dispatched; round 1: the raise
        first_round = rounds[0]
        assert first_round["response"] is not None, "Round 0 must have a response (search succeeded)"
        # The search tool result was appended to messages; it shows up in the next round's snapshot
        # OR we can verify via the error round's request_messages (if the raise was after dispatch)
        last_round = rounds[-1]
        assert last_round["error"] is not None
        assert last_round["error"]["type"] == "APIError"

    def test_final_round_messages_snapshot_captures_last_tool_result(self) -> None:
        """The last audit round's request_messages should include the tool result from N-1."""
        mock_client = _SequentialMockClient(_build_completed_responses())
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "completed"
        rounds = report["raw_llm_rounds"]
        # The finish round is round 3 (index 3). Its request_messages must include
        # the inspect_evidence tool result from round 2.
        assert len(rounds) >= 4
        finish_round = rounds[3]
        tool_msgs = [m for m in finish_round["request_messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) >= 3, (
            f"Finish round must include tool results from all prior rounds (search, fetch, inspect); "
            f"found {len(tool_msgs)} tool messages"
        )


class TestThinkingModeDisabled:
    """Issue 1: every create() call must explicitly pass the thinking-disable config."""

    def test_every_round_sends_thinking_disabled(self) -> None:
        """request_extra_body must contain thinking.type=disabled on every round."""
        mock_client = _SequentialMockClient(_build_completed_responses())
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "completed"
        rounds = report["raw_llm_rounds"]
        assert len(rounds) >= 1, "Expected at least one round"
        for rd in rounds:
            extra_body = rd.get("request_extra_body")
            assert extra_body is not None, (
                f"Round {rd['call_index']}: request_extra_body is None — "
                "thinking=disabled must be passed on every create() call"
            )
            thinking = extra_body.get("thinking", {})
            assert thinking.get("type") == "disabled", (
                f"Round {rd['call_index']}: expected thinking.type='disabled', "
                f"got {thinking!r}. DeepSeek default is enabled; must be explicitly disabled."
            )

    def test_budget_exhausted_all_rounds_have_thinking_disabled(self) -> None:
        """thinking-disable config must appear even on budget_exhausted runs."""
        responses = [
            _make_response(_make_msg([_make_tool_call(
                "search_literature", {"query": f"statins q{i}", "top_k": 3}, f"tc{i}"
            )]))
            for i in range(6)
        ]
        mock_client = _SequentialMockClient(responses)
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "budget_exhausted"
        for rd in report["raw_llm_rounds"]:
            extra_body = rd.get("request_extra_body")
            assert extra_body is not None
            assert extra_body.get("thinking", {}).get("type") == "disabled", (
                f"Round {rd['call_index']}: thinking not disabled in budget_exhausted run"
            )


class TestFinalMessagesSnapshot:
    """Issue 2: terminal-round tool results must appear in final_messages_snapshot."""

    def test_completed_final_messages_snapshot_contains_finish_tool_result(self) -> None:
        """After a completed run, final_messages_snapshot must contain the finish tool result."""
        mock_client = _SequentialMockClient(_build_completed_responses())
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "completed"
        snapshot = report.get("final_messages_snapshot")
        assert snapshot is not None, (
            "final_messages_snapshot must be present in the report after a completed run"
        )
        tool_msgs = [m for m in snapshot if m.get("role") == "tool"]
        assert len(tool_msgs) >= 4, (
            f"final_messages_snapshot must include all 4 tool results "
            f"(search, fetch, inspect, finish); found {len(tool_msgs)}: "
            f"{[m.get('content', '')[:60] for m in tool_msgs]}"
        )
        # The finish tool result should contain 'accepted'
        finish_results = [
            m for m in tool_msgs
            if "accepted" in m.get("content", "")
        ]
        assert len(finish_results) >= 1, (
            "final_messages_snapshot must contain the finish tool result with 'accepted'; "
            f"tool message contents: {[m.get('content', '')[:80] for m in tool_msgs]}"
        )

    def test_budget_exhausted_final_messages_snapshot_has_sixth_tool_result(self) -> None:
        """On budget_exhausted with 6 different searches, the 6th search result must be
        in final_messages_snapshot (it is never captured in any round's request_messages
        because there is no 7th call)."""
        responses = [
            _make_response(_make_msg([_make_tool_call(
                "search_literature", {"query": f"statins query{i}", "top_k": 3}, f"tc{i}"
            )]))
            for i in range(6)
        ]
        mock_client = _SequentialMockClient(responses)
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "budget_exhausted"
        snapshot = report.get("final_messages_snapshot")
        assert snapshot is not None, (
            "final_messages_snapshot must be present even on budget_exhausted"
        )
        tool_msgs = [m for m in snapshot if m.get("role") == "tool"]
        assert len(tool_msgs) >= 6, (
            f"All 6 search tool results must appear in final_messages_snapshot; "
            f"found only {len(tool_msgs)}"
        )
        # The 6th result content must contain search hits (or duplicate-call guard message)
        last_tool_content = tool_msgs[-1].get("content", "")
        assert last_tool_content, "Last tool message content must not be empty"


class TestUsageCoverageLabel:
    """Issue 3: provenance must carry usage_coverage confirmed/partial/unknown label."""

    def test_all_usage_present_yields_confirmed_coverage(self) -> None:
        """All rounds have usage → usage_coverage must be 'confirmed'."""
        mock_client = _SequentialMockClient(_build_completed_responses())
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "completed"
        answer_model = report["agent_response"]["provenance"]["answer_model"]
        coverage = answer_model.get("usage_coverage")
        assert coverage == "confirmed", (
            f"Expected usage_coverage='confirmed' when all rounds have usage, got {coverage!r}"
        )

    def test_all_unknown_usage_yields_unknown_coverage(self) -> None:
        """No round has usage → usage_coverage must be 'unknown'."""
        responses = [
            _make_response(
                _make_msg([_make_tool_call(
                    "search_literature", {"query": "statins cardiovascular", "top_k": 3}, "tc1"
                )]),
                no_usage=True,
            ),
            _make_response(
                _make_msg([_make_tool_call("fetch_record", {"pmid": "1004"}, "tc2")]),
                no_usage=True,
            ),
            _make_response(
                _make_msg([_make_tool_call(
                    "inspect_evidence", {"pmid": "1004", "query": "statins cardiovascular events"}, "tc3"
                )]),
                no_usage=True,
            ),
            _make_response(
                _make_msg([_make_tool_call("finish", {
                    "verdict": "supported",
                    "answer": "RCT evidence supports statin use for CV event reduction.",
                    "claim": "Statins reduce major cardiovascular events in RCTs.",
                    "cited_pmids": ["1004"],
                    "decisive_reason": "PMID 1004 directly reports significant RCT results.",
                }, "tc4")]),
                no_usage=True,
            ),
        ]
        mock_client = _SequentialMockClient(responses)
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "completed"
        answer_model = report["agent_response"]["provenance"]["answer_model"]
        coverage = answer_model.get("usage_coverage")
        assert coverage == "unknown", (
            f"Expected usage_coverage='unknown' when no round has usage, got {coverage!r}. "
            "input_tokens=0 with unknown coverage must not read as a confirmed zero."
        )

    def test_partial_usage_yields_partial_coverage(self) -> None:
        """Some rounds have usage, some don't → usage_coverage must be 'partial'."""
        responses = [
            _make_response(_make_msg([_make_tool_call(
                "search_literature", {"query": "statins cardiovascular", "top_k": 3}, "tc1"
            )])),
            _make_response(
                _make_msg([_make_tool_call("fetch_record", {"pmid": "1004"}, "tc2")]),
                no_usage=True,
            ),
            _make_response(_make_msg([_make_tool_call(
                "inspect_evidence", {"pmid": "1004", "query": "statins cardiovascular events"}, "tc3"
            )])),
            _make_response(_make_msg([_make_tool_call("finish", {
                "verdict": "supported",
                "answer": "RCT evidence supports statin use for CV event reduction.",
                "claim": "Statins reduce major cardiovascular events in RCTs.",
                "cited_pmids": ["1004"],
                "decisive_reason": "PMID 1004 directly reports significant RCT results.",
            }, "tc4")])),
        ]
        mock_client = _SequentialMockClient(responses)
        report = _run_with_mock(mock_client)

        assert report["outcome"] == "completed"
        answer_model = report["agent_response"]["provenance"]["answer_model"]
        coverage = answer_model.get("usage_coverage")
        assert coverage == "partial", (
            f"Expected usage_coverage='partial' when some rounds lack usage, got {coverage!r}"
        )


# ── SDK-shaped doubles for extension-field coverage ──────────────────────────
#
# The real SDK models declare extra="allow", so a server field the SDK does not
# model lands in model_extra, while model_fields_set records which declared
# fields the payload actually carried.  These doubles reproduce both, so the
# audit path can be exercised offline with the same field-presence semantics as
# a real response.

class _SdkLike:
    """Mimics a pydantic SDK model: declared fields plus extras."""

    def __init__(
        self,
        declared: dict[str, Any],
        *,
        unsent: tuple[str, ...] = (),
        extra: dict[str, Any] | None = None,
    ) -> None:
        skip = set(unsent)
        for name, value in declared.items():
            if name not in skip:
                setattr(self, name, value)
        self.model_fields_set = set(declared) - skip
        self.model_extra = dict(extra or {})


def _make_sdk_response(
    *,
    content: Any = "ok",
    tool_calls: list[Any] | None = None,
    usage_declared: dict[str, Any] | None = None,
    usage_unsent: tuple[str, ...] = (),
    usage_extra: dict[str, Any] | None = None,
    message_unsent: tuple[str, ...] = (),
    message_extra: dict[str, Any] | None = None,
) -> Any:
    msg = _SdkLike(
        {"role": "assistant", "content": content, "tool_calls": tool_calls},
        unsent=message_unsent,
        extra=message_extra,
    )
    usage = (
        None
        if usage_declared is None
        else _SdkLike(usage_declared, unsent=usage_unsent, extra=usage_extra)
    )
    choice = types.SimpleNamespace(message=msg, finish_reason="tool_calls")
    return types.SimpleNamespace(
        id="resp-sdk", model="deepseek-flash", choices=[choice], usage=usage
    )


def _completed_with_first_round(response: Any) -> list[Any]:
    responses = _build_completed_responses()
    responses[0] = response
    return responses


_SEARCH_CALL = _make_tool_call(
    "search_literature", {"query": "statins cardiovascular", "top_k": 3}, "tc1"
)
_FULL_COUNTS = {"prompt_tokens": 1042, "completion_tokens": 111, "total_tokens": 1153}


class TestUsageExtensionFields:
    """Server-side usage additions must survive into the audit record."""

    def test_cache_split_is_preserved(self) -> None:
        """DeepSeek reports the cache split outside the SDK schema; dropping it
        makes an honest cost estimate impossible."""
        response = _make_sdk_response(
            tool_calls=[_SEARCH_CALL],
            usage_declared=dict(_FULL_COUNTS),
            usage_extra={"prompt_cache_hit_tokens": 900, "prompt_cache_miss_tokens": 142},
        )
        report = _run_with_mock(_SequentialMockClient(_completed_with_first_round(response)))

        usage = report["raw_llm_rounds"][0]["response"]["usage"]
        assert usage["extension_fields"]["prompt_cache_hit_tokens"] == {
            "state": "present",
            "value": 900,
        }
        assert usage["extension_fields"]["prompt_cache_miss_tokens"]["value"] == 142
        # Flat counts must remain readable for existing consumers.
        assert usage["prompt_tokens"] == 1042
        assert usage["status"] == "confirmed"

    def test_field_never_sent_reads_as_absent(self) -> None:
        response = _make_sdk_response(
            tool_calls=[_SEARCH_CALL], usage_declared=dict(_FULL_COUNTS)
        )
        report = _run_with_mock(_SequentialMockClient(_completed_with_first_round(response)))

        usage = report["raw_llm_rounds"][0]["response"]["usage"]
        assert usage["fields"]["prompt_tokens_details"]["state"] == "absent"

    def test_explicit_null_is_distinct_from_absent(self) -> None:
        response = _make_sdk_response(
            tool_calls=[_SEARCH_CALL],
            usage_declared={**_FULL_COUNTS, "prompt_tokens_details": None},
        )
        report = _run_with_mock(_SequentialMockClient(_completed_with_first_round(response)))

        usage = report["raw_llm_rounds"][0]["response"]["usage"]
        states = {name: entry["state"] for name, entry in usage["fields"].items()}
        assert states["prompt_tokens_details"] == "null"
        # ...while a field the payload omitted stays a different fact entirely.
        assert states["completion_tokens_details"] == "absent"

    def test_empty_extension_value_is_empty_not_null(self) -> None:
        response = _make_sdk_response(
            tool_calls=[_SEARCH_CALL],
            usage_declared=dict(_FULL_COUNTS),
            usage_extra={"prompt_cache_hit_tokens": 0, "vendor_notes": []},
        )
        report = _run_with_mock(_SequentialMockClient(_completed_with_first_round(response)))

        extensions = report["raw_llm_rounds"][0]["response"]["usage"]["extension_fields"]
        assert extensions["vendor_notes"]["state"] == "empty"
        # A genuine zero is a value, not an absence — the two must not collapse.
        assert extensions["prompt_cache_hit_tokens"] == {"state": "present", "value": 0}


class TestMessageExtensionFields:
    """Server-side message additions must survive into the audit record."""

    def test_reasoning_content_is_preserved(self) -> None:
        response = _make_sdk_response(
            tool_calls=[_SEARCH_CALL],
            usage_declared=dict(_FULL_COUNTS),
            message_extra={"reasoning_content": "weighing the retrieval options"},
        )
        report = _run_with_mock(_SequentialMockClient(_completed_with_first_round(response)))

        choice = report["raw_llm_rounds"][0]["response"]["choices"][0]
        assert choice["message_extension_fields"]["reasoning_content"] == {
            "state": "present",
            "value": "weighing the retrieval options",
        }

    def test_null_content_and_unsent_field_are_distinguishable(self) -> None:
        response = _make_sdk_response(
            content=None,
            tool_calls=[_SEARCH_CALL],
            usage_declared=dict(_FULL_COUNTS),
        )
        report = _run_with_mock(_SequentialMockClient(_completed_with_first_round(response)))

        states = {
            name: entry["state"]
            for name, entry in report["raw_llm_rounds"][0]["response"]["choices"][0][
                "message_fields"
            ].items()
        }
        assert states["content"] == "null"       # the server said: no content
        assert states["refusal"] == "absent"     # the server said nothing at all

    def test_audit_record_stays_json_serialisable(self) -> None:
        response = _make_sdk_response(
            content=None,
            tool_calls=[_SEARCH_CALL],
            usage_declared=dict(_FULL_COUNTS),
            usage_extra={"prompt_cache_hit_tokens": 900},
            message_extra={"reasoning_content": "chain of thought"},
        )
        report = _run_with_mock(_SequentialMockClient(_completed_with_first_round(response)))

        round_record = report["raw_llm_rounds"][0]
        assert json.loads(json.dumps(round_record)) == round_record
        assert json.loads(json.dumps(report)) == report


# ── genuine SDK types (not doubles) ──────────────────────────────────────────
#
# The doubles above reproduce field-presence semantics by hand, which only shows
# the audit agrees with our assumptions about the SDK.  These tests build real
# openai ChatCompletion objects through the installed SDK, so extension capture
# and the four-state handling are verified against the actual pydantic types.

from openai.types.chat.chat_completion import ChatCompletion


def _real_completion(
    *,
    content: Any = None,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    message_extra: dict[str, Any] | None = None,
) -> Any:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    message.update(message_extra or {})
    payload: dict[str, Any] = {
        "id": "chatcmpl-real",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-flash",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
    }
    if usage is not None:
        payload["usage"] = usage
    return ChatCompletion.model_validate(payload)


def _real_tool_calls(name: str, arguments: dict[str, Any], call_id: str) -> list[dict[str, Any]]:
    return [{
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }]


# What DeepSeek adds beyond the SDK schema, plus the counts the SDK does declare.
_DEEPSEEK_USAGE = {
    "prompt_tokens": 1042,
    "completion_tokens": 111,
    "total_tokens": 1153,
    "prompt_cache_hit_tokens": 900,
    "prompt_cache_miss_tokens": 142,
}


class TestRealSdkTypeSerialisation:
    """Extension capture must hold for genuine SDK objects, not only doubles."""

    def test_cache_split_survives_a_real_completion(self) -> None:
        response = _real_completion(
            tool_calls=_real_tool_calls("search_literature", {"query": "statins"}, "tc1"),
            usage=dict(_DEEPSEEK_USAGE),
        )
        usage = smoke._response_to_dict(response)["usage"]

        assert usage["status"] == "confirmed"
        assert usage["extension_fields"]["prompt_cache_hit_tokens"] == {
            "state": "present",
            "value": 900,
        }
        assert usage["extension_fields"]["prompt_cache_miss_tokens"]["value"] == 142

    def test_field_the_server_omitted_reads_as_absent(self) -> None:
        response = _real_completion(
            tool_calls=_real_tool_calls("search_literature", {"query": "statins"}, "tc1"),
            usage=dict(_DEEPSEEK_USAGE),
        )
        fields = smoke._response_to_dict(response)["usage"]["fields"]
        assert fields["prompt_tokens_details"]["state"] == "absent"

    def test_explicit_null_is_distinct_from_absent(self) -> None:
        response = _real_completion(
            tool_calls=_real_tool_calls("search_literature", {"query": "statins"}, "tc1"),
            usage={**_DEEPSEEK_USAGE, "prompt_tokens_details": None},
        )
        states = {
            name: entry["state"]
            for name, entry in smoke._response_to_dict(response)["usage"]["fields"].items()
        }
        assert states["prompt_tokens_details"] == "null"
        assert states["completion_tokens_details"] == "absent"

    def test_reasoning_content_survives_a_real_completion(self) -> None:
        response = _real_completion(
            tool_calls=_real_tool_calls("search_literature", {"query": "statins"}, "tc1"),
            usage=dict(_DEEPSEEK_USAGE),
            message_extra={"reasoning_content": "weighing the retrieval options"},
        )
        choice = smoke._response_to_dict(response)["choices"][0]
        assert choice["message_extension_fields"]["reasoning_content"] == {
            "state": "present",
            "value": "weighing the retrieval options",
        }

    def test_null_content_and_unsent_field_differ_on_a_real_completion(self) -> None:
        response = _real_completion(
            tool_calls=_real_tool_calls("search_literature", {"query": "statins"}, "tc1"),
            usage=dict(_DEEPSEEK_USAGE),
        )
        fields = smoke._response_to_dict(response)["choices"][0]["message_fields"]
        assert fields["content"]["state"] == "null"
        assert fields["refusal"]["state"] == "absent"

    def test_real_completion_serialises_to_json(self) -> None:
        response = _real_completion(
            tool_calls=_real_tool_calls("search_literature", {"query": "statins"}, "tc1"),
            usage=dict(_DEEPSEEK_USAGE),
            message_extra={"reasoning_content": "chain of thought"},
        )
        record = smoke._response_to_dict(response)
        assert json.loads(json.dumps(record)) == record


class TestRealSdkTypesThroughTheFullRun:
    """The whole loop, driven by genuine SDK responses rather than doubles."""

    def _real_sequence(self) -> list[Any]:
        return [
            _real_completion(
                tool_calls=_real_tool_calls(
                    "search_literature", {"query": "statins cardiovascular", "top_k": 3}, "tc1"
                ),
                usage=dict(_DEEPSEEK_USAGE),
                message_extra={"reasoning_content": "search the corpus first"},
            ),
            _real_completion(
                tool_calls=_real_tool_calls("fetch_record", {"pmid": "1004"}, "tc2"),
                usage=dict(_DEEPSEEK_USAGE),
            ),
            _real_completion(
                tool_calls=_real_tool_calls(
                    "inspect_evidence",
                    {"pmid": "1004", "query": "statins cardiovascular events"},
                    "tc3",
                ),
                usage=dict(_DEEPSEEK_USAGE),
            ),
            _real_completion(
                tool_calls=_real_tool_calls("finish", {
                    "verdict": "supported",
                    "answer": "RCT evidence supports statin use for CV event reduction.",
                    "claim": "Statins reduce major cardiovascular events in RCTs.",
                    "cited_pmids": ["1004"],
                    "decisive_reason": "PMID 1004 directly reports significant RCT results.",
                }, "tc4"),
                usage=dict(_DEEPSEEK_USAGE),
            ),
        ]

    def test_real_objects_carry_through_the_whole_audit_path(self) -> None:
        report = _run_with_mock(_SequentialMockClient(self._real_sequence()))

        assert report["outcome"] == "completed", (
            f"{report['outcome']}: {report.get('agent_response_contract_errors')}"
        )
        assert len(report["raw_llm_rounds"]) == 4

        first = report["raw_llm_rounds"][0]["response"]
        assert first["usage"]["extension_fields"]["prompt_cache_hit_tokens"]["value"] == 900
        assert first["choices"][0]["message_extension_fields"]["reasoning_content"][
            "value"
        ] == "search the corpus first"

        answer_model = report["agent_response"]["provenance"]["answer_model"]
        assert answer_model["usage_coverage"] == "confirmed"
        assert answer_model["input_tokens"] == 4 * _DEEPSEEK_USAGE["prompt_tokens"]

    def test_full_report_from_real_objects_is_json_serialisable(self) -> None:
        report = _run_with_mock(_SequentialMockClient(self._real_sequence()))
        assert json.loads(json.dumps(report)) == report
