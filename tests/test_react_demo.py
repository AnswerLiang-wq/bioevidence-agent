"""Regression tests for the react-demo subcommand.

These tests call main() through the real CLI dispatch with a scripted fake
LLM client — no real API key or network access is involved.  They verify:
  - CLI exits 0 and emits valid JSON for every named scenario
  - demo_envelope declares scripted_mode and corpus_is_synthetic
  - corpus_note carries human-readable synthetic-fixture language
  - The product contract passes (agent_response_contract_valid is True)
  - The agent_response verdict matches the scenario's expected value
  - provenance.agent_run.run_status is "completed"
  - At least one citation references the scenario's target PMID
  - An unrecognised --scenario value exits with a non-zero code

What is NOT tested here (and why):
  - Real DeepSeek model decisions — the scripted client replaces the model
    step entirely; model behaviour must be validated against a live API key
  - Retrieval ranking quality — covered by test_diagnostics.py benchmarks
  - Corpus hash integrity — covered by test_core.py
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import pytest

from bioevidence.corpus import read_corpus
from bioevidence.portfolio_cli import main

# Ground-truth table for the four scripted scenarios.
# Any change to _REACT_DEMO_SCENARIOS in portfolio_cli.py that alters
# verdict or target PMID will surface here immediately.
_EXPECTED: dict[str, dict[str, str]] = {
    "statins": {"verdict": "supported", "target_pmid": "1004"},
    "metformin": {"verdict": "supported", "target_pmid": "1005"},
    "aspirin-primary": {"verdict": "contradicted", "target_pmid": "1006"},
    "pembrolizumab": {"verdict": "supported", "target_pmid": "1008"},
}


@pytest.mark.parametrize("scenario", list(_EXPECTED))
def test_react_demo_full_contract(
    scenario: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end: CLI output, product contract, provenance, citation PMID."""
    expected = _EXPECTED[scenario]

    exit_code = main(["react-demo", "--scenario", scenario])
    assert exit_code == 0, f"main() returned non-zero for scenario {scenario!r}"

    output = json.loads(capsys.readouterr().out)
    envelope = output["demo_envelope"]
    agent_response = output["agent_response"]

    # --- envelope: scripted-mode declarations ---
    assert envelope["scripted_mode"] is True
    assert envelope["scenario"] == scenario
    assert envelope["corpus_is_synthetic"] is True
    # The human-readable corpus_note must mention synthetic fixtures so
    # anyone reading the raw output understands the data provenance.
    assert "synthetic" in envelope["corpus_note"].lower()

    # --- envelope: product contract ---
    assert envelope["agent_response_contract_valid"] is True, (
        f"Contract errors for {scenario!r}: "
        f"{envelope['agent_response_contract_errors']}"
    )
    assert envelope["agent_response_contract_errors"] == []

    # --- agent_response: verdict ---
    assert agent_response["verdict"] == expected["verdict"]

    # Non-insufficient verdicts must not set abstained=True.
    if agent_response["verdict"] != "insufficient":
        assert agent_response["abstained"] is False

    # --- agent_response: agent_run provenance ---
    provenance = agent_response["provenance"]
    agent_run = provenance.get("agent_run")
    assert agent_run is not None, "provenance.agent_run is absent"
    assert agent_run["run_status"] == "completed", (
        f"Unexpected run_status for {scenario!r}: {agent_run['run_status']!r}"
    )

    # --- agent_response: citation traces back to the target PMID ---
    # citation_id values are opaque sequential strings ("C1", "C2", …); the
    # authoritative provenance link is source_sha256 == the document's
    # content_sha256.  Load the fixture corpus to get the expected hash.
    corpus_path = Path(
        str(files("bioevidence").joinpath("fixtures/tiny_corpus.jsonl"))
    )
    docs_by_pmid = {d.pmid: d for d in read_corpus(corpus_path)}
    target_pmid = expected["target_pmid"]
    assert target_pmid in docs_by_pmid, (
        f"Target PMID {target_pmid!r} not found in corpus"
    )
    expected_sha = docs_by_pmid[target_pmid].content_sha256

    citations = agent_response["citations"]
    assert len(citations) >= 1, f"No citations produced for {scenario!r}"
    source_hashes = [c["source_sha256"] for c in citations]
    assert expected_sha in source_hashes, (
        f"Target PMID {target_pmid!r} (sha256={expected_sha!r}) not found in "
        f"citation source_sha256 values {source_hashes!r} for scenario {scenario!r}"
    )


def test_react_demo_invalid_scenario_exits() -> None:
    """Argparse must reject an unrecognised --scenario value."""
    with pytest.raises(SystemExit) as exc_info:
        main(["react-demo", "--scenario", "not-a-real-scenario"])
    assert exc_info.value.code != 0
