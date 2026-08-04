# Experiment log

This log records engineering and product experiments with their scope. No
entry is a user study unless it explicitly identifies real, consented,
anonymous participants.

## E-000 — v0.3.0 baseline re-verification

- Date: 2026-08-05
- Type: engineering audit; no participants
- Commit: `46afee6ca5b5a4066cdcafa9532abcc7d6e805ef`
- Commands:
  - `python -m ruff check src tests scripts`
  - `python -m pytest -q`
  - `python scripts/verify_release.py`
  - install `artifacts/bioevidence_agent-0.3.0-py3-none-any.whl` into a new
    temporary virtual environment and run `bioevidence demo` from `/tmp`
- Results:
  - Ruff passed;
  - 8/8 lightweight tests passed;
  - 44/44 release-manifest artifact hashes passed;
  - fresh wheel installation passed;
  - out-of-tree demo returned PMID `1001`, an exact snippet hash, a source
    hash, and three successful typed-tool calls.
- Limitation: the fixture contains three synthetic documents and measures
  packaging/provenance behavior, not user value or open-world retrieval.

## E-001 — Live PubMed path smoke

- Date: 2026-08-05
- Type: external-service engineering audit; no participants
- Query: `remdesivir COVID-19 randomized trial`
- Configuration: official NCBI E-utilities, relevance sort, `retmax=3`, no
  NCBI API key, at most two attempts, 20-second request timeout.
- Results:
  - ESearch returned three PMIDs;
  - EFetch returned three records;
  - all three records contained abstracts;
  - returned records included primary remdesivir trial reports.
- Interpretation: the existing client can support a live candidate-discovery
  prototype. This single smoke does not establish relevance quality,
  availability, latency, or task success across user questions.
- Privacy: the query was a public generic test query, not participant data.

## User-study status

- Discovery interviews: 0
- Pilot participants: 0
- Main-study participants: 0
- VEPS observations: 0

No simulated or model-generated behavior is counted as participant evidence.
