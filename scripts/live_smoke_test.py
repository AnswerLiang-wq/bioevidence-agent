"""Live smoke test for BioEvidenceAgent — one real API call sequence.

PURPOSE
-------
Verify that LLMEvidenceAgent can complete a real conversation with the DeepSeek
Flash API: the HTTP connection succeeds, the model emits tool calls, the ReAct
loop executes, and (if all goes well) finish is validated and accepted.

WHAT THIS PROVES AND DOES NOT PROVE
-------------------------------------
Communication success  — HTTP + auth OK; server returned a non-error response.
Tool call emitted      — Model produced a structurally valid tool_calls message.
Workflow completed     — Loop ran to finish with run_status == "completed" AND
                         validate_product_response returned no errors.  A finish
                         that was rejected by _dispatch is NOT completion.
Answer correctness     — NOT claimed.  One case on a 10-document synthetic corpus
                         is not a sample size.  Use the 500-case PubMedQA benchmark
                         for accuracy claims.

MODEL AND THINKING MODE (2026-09-30)
-------------------------------------
Model: deepseek-flash (explicit name; do NOT use the "deepseek-chat" alias — it
may be removed or rerouted without notice).

This test uses NON-THINKING MODE.  Per DeepSeek Chat Completions API docs, the
default is thinking.type="enabled".  To disable, every create() call passes
  extra_body={"thinking": {"type": "disabled"}}
LLMEvidenceAgent.run() now passes this parameter explicitly.  The audit record
captures it as request_extra_body so it can be verified post-run.

reasoning_content transparency (openai SDK 3.22.1, verified 2026-09-30):
  ChatCompletionMessage has model_config extra="allow", so the SDK round-trips any
  extra fields from the API response into msg.model_extra.  If DeepSeek ever
  returns reasoning_content in a response, it will appear as:
    msg.model_extra.get("reasoning_content")
  and model_dump(exclude_none=True) will include it automatically.  No special
  handling is needed for non-thinking mode (reasoning_content will not appear).

temperature is fixed at 0.0 inside LLMEvidenceAgent.run(); this script does not
override it.

PRICING (source: https://api-docs.deepseek.com/quick_start/pricing, 2026-09-30)
-----------------
  Input  cache miss  off-peak: $0.15/M   peak: $0.30/M
  Input  cache hit   off-peak: $0.003/M  peak: $0.006/M
  Output             off-peak: $0.60/M   peak: $1.20/M
Peak = Mon-Fri 01:00-04:00 UTC and 06:00-10:00 UTC (excl. Chinese public holidays).
All other hours (incl. weekends) are off-peak.

BUDGET ESTIMATE
---------------
There is no API-enforced hard cap per individual request.  The only spending
control in this script is the soft ceiling imposed by max_steps and
max_tokens_per_call.

Estimate (assuming ~4 000 input tokens per call, peak cache-miss rates):
  6 calls × 4 000 input tokens × $0.30/M  = $0.0072
  6 calls × 512 output tokens  × $1.20/M  = $0.0037
  Total estimate: ~$0.011

Actual cost depends on the true prompt length, whether responses hit the cache,
and the time of day.  The estimate uses the worst-case (peak, cache-miss, full
512 output tokens) per call.  Off-peak or cache-hit runs cost roughly half.

The openai SDK default max_retries=2 means a single failing call can produce up
to 3 HTTP requests before raising; each retry counts toward spend.  We override
to max_retries=0 for the smoke test so that a transient error produces exactly
one request and does not silently multiply cost.  Set timeout=30 to prevent
hanging; the SDK raises APITimeoutError on expiry.

STOP CONDITIONS
---------------
  1. max_steps=6 is reached without finish → run_status "budget_exhausted".
  2. finish is called and _dispatch accepts it → run_status "completed".
  3. Model emits text with no tool call → run_status "text_exit".
  4. Any openai.APIError, AuthenticationError, RateLimitError, or network
     exception → caught below, partial trace saved, script exits non-zero.

OUTPUT
------
  reports/live_smoke_test_v1.json   (written after the run; on init failure
                                     the file may not be written — see Gap 2
                                     in tests/test_live_smoke_test.py)
  stdout: same JSON

RUN
---
  export DEEPSEEK_API_KEY=sk-...
  cd /path/to/BioEvidenceAgent-Portfolio
  python -m scripts.live_smoke_test
  # OR: python scripts/live_smoke_test.py
"""

from __future__ import annotations

# ruff: noqa: E402  — project imports follow a sys.path bootstrap; E402 is expected.

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ── project path bootstrap ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from importlib.resources import files

from bioevidence.audit import (
    ABSENT,
    MESSAGE_FIELDS,
    USAGE_FIELDS,
    AuditingClient,
    declared_states,
    extension_states,
    field_state,
    json_safe,
    response_to_dict,
    usage_to_dict,
)
from bioevidence.corpus import read_corpus, sha256_file
from bioevidence.llm_agent import LLMEvidenceAgent
from bioevidence.product_contracts import validate_product_response
from bioevidence.tools import LiteratureTools


# ── audit-capturing client wrapper ───────────────────────────────────────────

# ── auditing primitives (shared with the comparison harness) ─────────────────
# Defined once in bioevidence.audit and re-exported under the private names this
# module and its tests already use, so there is a single implementation of
# lossless response capture rather than two drifting copies.

_AuditingClient = AuditingClient

_ABSENT = ABSENT
_json_safe = json_safe
_field_state = field_state
_declared_states = declared_states
_extension_states = extension_states
_usage_to_dict = usage_to_dict
_response_to_dict = response_to_dict

_USAGE_FIELDS = USAGE_FIELDS
_MESSAGE_FIELDS = MESSAGE_FIELDS


# ── main ──────────────────────────────────────────────────────────────────────

def run_smoke_test() -> dict[str, Any]:
    """Run the test and return the report dict.

    The report is always returned.  It is written to disk by main() after this
    function returns.  If initialization fails before the report dict is built
    (e.g. if the bioevidence package is not installed), a minimal error dict is
    returned but no file is written — the caller (main) is responsible for the
    write.
    """

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        # Return a structured failure immediately; do not attempt a network call.
        return {
            "outcome": "configuration_error",
            "error": "DEEPSEEK_API_KEY environment variable is not set.",
            "raw_llm_rounds": [],
        }

    # Build the report skeleton before any fallible work so the finally block
    # can always attach raw_llm_rounds even on an early exception.
    report: dict[str, Any] = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_interpreter": sys.executable,
        "model": "deepseek-flash",
        "corpus_path": None,
        "corpus_sha256": None,
        "corpus_size": None,
        "question": (
            "Do statins reduce major cardiovascular events in randomized controlled trials?"
        ),
        "max_steps": 6,
        "max_tokens_per_call": 512,
        "sdk_max_retries_override": 0,
        "sdk_timeout_seconds": 30,
        "pricing_note": (
            "deepseek-flash peak cache-miss: $0.30/M input, $1.20/M output. "
            "Off-peak (weekends + non-peak UTC hours) is roughly half. "
            "Source: https://api-docs.deepseek.com/quick_start/pricing 2026-09-30."
        ),
        "budget_note": (
            "No API-level hard cap. Soft ceiling: max_steps=6, max_tokens=512/call. "
            "Estimate (assuming ~4000 input tokens/call, peak cache-miss): ~$0.011. "
            "Actual cost depends on true prompt length, cache state, and time of day. "
            "SDK retries overridden to 0 to prevent silent cost multiplication."
        ),
        "elapsed_seconds": None,
        "outcome": None,
        "error": None,
        "agent_response": None,
        "agent_response_contract_valid": None,
        "agent_response_contract_errors": None,
        "raw_llm_rounds": [],
        # final_messages_snapshot: deep copy of the full messages list after the
        # ReAct loop ends.  Populated by _AuditingClient.finalize() to capture
        # tool results appended after the last create() call (which are never
        # snapshotted in any round's request_messages).
        "final_messages_snapshot": None,
    }

    model = report["model"]
    max_steps = report["max_steps"]
    max_tokens_per_call = report["max_tokens_per_call"]
    question = report["question"]
    auditing_client: _AuditingClient | None = None

    try:
        # All initialization is inside the try block so failures produce
        # a structured report rather than an unhandled exception.
        corpus_path = Path(str(files("bioevidence").joinpath("fixtures/tiny_corpus.jsonl")))
        report["corpus_path"] = str(corpus_path)
        report["corpus_sha256"] = sha256_file(corpus_path)

        documents = read_corpus(corpus_path)
        report["corpus_size"] = len(documents)
        tools_obj = LiteratureTools(documents)

        # Build the real client then wrap it for auditing.
        try:
            from openai import OpenAI
        except ImportError as exc:
            report["outcome"] = "configuration_error"
            report["error"] = f"openai package not importable: {exc}"
            return report

        real_client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            max_retries=0,   # no silent retries — each 429/5xx counts once
            timeout=30.0,    # seconds; raises APITimeoutError on expiry
        )
        auditing_client = _AuditingClient(real_client)

        # Inject the auditing wrapper using the same __new__ pattern as tests.
        agent = LLMEvidenceAgent.__new__(LLMEvidenceAgent)
        agent._tools = tools_obj
        agent._model = model
        agent._max_steps = max_steps
        agent._max_tokens_per_call = max_tokens_per_call
        agent._base_url = "https://api.deepseek.com/v1"
        agent._client = auditing_client

        t0 = time.time()
        agent_response = agent.run(question=question, request_id="live-smoke-v1")
        report["elapsed_seconds"] = round(time.time() - t0, 2)
        # Capture the final messages list (includes tool results appended after the
        # last create() call, which are never snapshotted by _AuditingClient alone).
        if auditing_client is not None and hasattr(agent, "_last_messages"):
            auditing_client.finalize(agent._last_messages)

        contract_errors = validate_product_response(agent_response)
        report["agent_response"] = agent_response
        report["agent_response_contract_valid"] = len(contract_errors) == 0
        report["agent_response_contract_errors"] = contract_errors

        run_status = (
            agent_response.get("provenance", {})
            .get("agent_run", {})
            .get("run_status", "unknown")
        )
        if run_status == "completed" and len(contract_errors) == 0:
            report["outcome"] = "completed"
        elif run_status == "completed" and contract_errors:
            report["outcome"] = "completed_contract_invalid"
        elif run_status == "budget_exhausted":
            report["outcome"] = "budget_exhausted"
        elif run_status == "text_exit":
            report["outcome"] = "text_exit"
        else:
            report["outcome"] = f"unknown_run_status:{run_status}"

    except Exception as exc:  # noqa: BLE001
        report["outcome"] = "exception"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    finally:
        # Capture whatever rounds were recorded before any exception.
        if auditing_client is not None:
            report["raw_llm_rounds"] = auditing_client.rounds
            if auditing_client.final_messages_snapshot is not None:
                report["final_messages_snapshot"] = auditing_client.final_messages_snapshot

    return report


def main() -> int:
    report = run_smoke_test()

    output_path = PROJECT_ROOT / "reports" / "live_smoke_test_v1.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

    outcome = report.get("outcome", "")
    # Exit 0 only on clean completion with valid contract.
    return 0 if outcome == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
