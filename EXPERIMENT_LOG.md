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
- Formative pilot participants: 1
- Pilot attempted task rows: 2
- Pilot evaluable task rows: 0
- Main-study participants: 0
- VEPS observations: 0

No simulated or model-generated behavior is counted as participant evidence.

The pilot is formative only. `VEPS observations: 0` means there is no
evaluable denominator; it is not a 0% success result.

## E-003 — Product Demo internal workflow and fault audit

- Date: 2026-08-05
- Type: engineering/internal usability audit; no participants
- Surfaces: local HTTP API, desktop browser, 390 x 844 responsive browser,
  standard fixed-PMID mode, and live PubMed ESearch/EFetch mode.
- Public live question: `Does tocilizumab reduce mortality in adults
  hospitalized with severe COVID-19?`
- Deterministic checks: happy path, empty search, missing abstracts, fetch
  failure, mixed and insufficient packs, forged-card rejection, question
  bounds, medical-advice boundary, privacy logs, and original core regression.
- Result at this experiment snapshot: 25/25 Python tests passed, including an
  out-of-tree launcher regression; Ruff passed.
- Live observation: the first run exposed duplicate same-title PubMed records.
  Exact-title deduplication and visible publication types were implemented;
  the same query then displayed five distinct titles.
- Browser observation: both standard and live tasks rendered, card acceptance
  updated the pack counter, and the narrow layout had no horizontal overflow.
- Limitation: no real participant was involved. This experiment cannot supply
  VEPS, efficiency, trust, reuse, or product-value evidence.
- Detailed record: `docs/product_case/INTERNAL_TEST_REPORT.md`.

## E-004 — Formative pilot and additive audit

- Date: 2026-08-05 to 2026-08-06
- Type: one real, anonymous, consented formative pilot; not a product-effect
  study
- Attempted rows: one manual task and one Agent task
- Deterministic result:
  - 2 attempted rows;
  - 0 evaluable rows;
  - 0 VEPS observations;
  - 0 eligible paired time comparisons;
  - 0 main-study participants or tasks.
- Positive integrity checks: private records remained Git ignored; no
  recording or privacy incident was found; exported Agent PMIDs were
  resolvable and their stored snippets and hashes matched.
- Blocking findings: conflicting task times, a Standard-to-Live mode switch,
  inconsistent source-open/timeout records, invalid spreadsheet placeholders,
  and accepted evidence with intervention or endpoint mismatch.
- Decision: retain the original records and mark both rows not evaluable. Do
  not infer efficiency, success, trust, reuse, or product value.
- Data boundary: the public log contains aggregate facts only. Raw records and
  row-level evidence remain under ignored `private/` paths.

## E-005 — Synthetic evidence-scope controls and portfolio closeout

- Date: 2026-08-06
- Type: deterministic engineering regression; no participants and no real
  paper adjudication
- Fixture: eight frozen synthetic cases using declared species, population,
  intervention, endpoint, timepoint, and evidence-role metadata.
- Cases: direct match; population, species, intervention, endpoint, and
  timepoint mismatch; context-only evidence; unknown scope.
- Results:
  - 8/8 scope controls passed;
  - only direct match remained a candidate for decisive human review;
  - all cases retained `requires_human_review=true` and
    `scientific_judgment=not_performed`;
  - final candidate collected 46 tests;
  - 41/41 non-socket checks and Ruff passed in the restricted local audit;
  - the complete run in a localhost-capable environment remains a Release
    gate.
- Interpretation: declared-scope fail-closed policy is deterministic. This is
  not automatic PICO extraction, biomedical NLI accuracy, real-paper
  relevance, or a clinical claim.
- Product decision: stop additional recruitment and finish v0.4 through
  automated release gates. User value remains unmeasured.
- Reports: `reports/product_scope_controls_v1.json` and
  `reports/product_scope_controls_v1.md`.

## E-006 — v0.4.0 final release verification

- Date: 2026-08-06
- Type: engineering release verification; no participants
- Commands:
  - `python -m ruff check src apps tests scripts`
  - `python -m pytest -q`
  - `python scripts/generate_product_example.py --output-dir examples`
  - `python scripts/capture_product_screenshots.py` (against a running local
    demo server, Standard fixed-PMID task)
  - `python -m build`
  - install `artifacts/bioevidence_agent-0.4.0-py3-none-any.whl` into a new
    temporary virtual environment and run `bioevidence demo` from a
    directory outside the repository
  - `python scripts/generate_release_manifest.py` and
    `python scripts/verify_release.py`
- Results:
  - Ruff passed;
  - 46/46 Python tests passed, including the localhost-bind and launcher
    checks;
  - deterministic public example regenerated with 0 sensitive-text hits;
  - two desensitized Standard-task screenshots captured and committed;
  - wheel and sdist built; fresh out-of-tree wheel install passed and the CLI
    demo returned fixture PMID `1001` with `__version__ == "0.4.0"`;
  - release manifest generated and verified (every tracked file hashed, public
    boundary audit scanned all tracked text files);
  - Demo core flow (health, tasks, search, export) and privacy checks passed;
    the event log contained no question, abstract, snippet, or note text.
- Limitation: these are software-contract checks. They are not user research,
  an effectiveness measure, or evidence of product value.
- Related release assets: `reports/release_manifest_v0.4.0.json`,
  `docs/product_case/screenshots/`, `examples/`.
