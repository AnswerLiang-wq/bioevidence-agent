# Product Demo internal and automated closeout report

Historical audit date: 2026-08-06

Historical version under test: local `v0.4.0` release candidate. The current
`v0.4.1` integrity-maintenance verification is recorded at the end of this
report and supersedes release-status language, not the historical observations.

Human evidence: one separate formative pilot; 0 evaluable task rows. The
checks in this report are engineering checks, not user research or evidence of
product value.

## Result summary

- Python tests: 46 collected in the final candidate; 41/41 checks that do not
  bind a local socket passed in the restricted audit environment;
- local-server tests: the five-test module includes four loopback-bind checks
  and one launcher check; the complete 46-test run remains a release/CI gate;
- Python lint: passed for `src`, `apps`, `tests`, and `scripts`;
- synthetic scope cases: 8/8 passed;
- desktop browser: Standard PubMed task completed and rendered four cards;
- live browser: PubMed ESearch/EFetch completed and rendered five cards;
- responsive check: at a 390 x 844 override the document client width and
  scroll width were both 375 CSS pixels, with no horizontal overflow;
- client JavaScript: parsed and executed in the browser, including mode
  switching, search, card acceptance, timer, and privacy-event submission;
- independent `node --check`: not run because Node.js was not installed in
  the audited shell environment.

The test inventory and partial local result are deterministic repository
facts at this audit snapshot. A final Release must record a complete passing
run in an environment that permits localhost binding. These are not a user
completion rate, model accuracy, retrieval-quality estimate,
production-availability claim, or replacement for scientific review.

## Product scenario matrix

| Scenario | Method | Observed result | Remaining boundary |
|---|---|---|---|
| Standard happy path | real local server, browser, official PubMed EFetch | four fixed-candidate cards rendered with PMID links, exact snippets, and normalized-record hashes | one curated task does not establish general retrieval quality |
| Live retrieval | browser, official PubMed ESearch and EFetch | five cards rendered for a public tocilizumab question | local BM25 results include mixed designs and require human screening |
| No results | injected empty PubMed search | typed `no_results` 404 response | real-world frequency is unknown |
| No usable abstract | records with empty abstracts | typed `no_abstract_evidence` 422 response | full text is deliberately out of scope |
| Source/fetch failure | injected `PubMedError` | typed 503 response without internal exception details | external availability is not guaranteed |
| Conflicting evidence | deterministic accepted cards with opposing user labels | `mixed` pack preserved both directions | the system does not decide which study is scientifically decisive |
| Insufficient evidence | export with one accepted abstract | `insufficient` pack exported | exportability is not task success or a VEPS result |
| Citation mismatch/injection | forged card ID and exact-snippet hash checks | forged card rejected; export used server-held search results | source identity does not establish scientific relevance |
| Long or ambiguous question | seven-character and 601-character inputs | both rejected before retrieval | semantic specificity beyond length remains a user responsibility |
| Patient-specific advice | personal medication/symptom request | rejected with a research-only boundary message | pattern guard is not a clinical classifier |
| Privacy logging | API and file inspection | event log contained action/timing metadata but no query, abstract, or note text | users must still avoid sensitive input |
| Original Agent regression | full repository test suite and Ruff | passed | the v0.3 manifest remains version-bound and is not rewritten |

## Eight synthetic scope controls

The suite uses frozen synthetic metadata, not biomedical prose or real PMIDs.

| Control | Expected policy | Observed result |
|---|---|---|
| direct scope match | allow only as a candidate for decisive human review | pass |
| population mismatch | fail closed | pass |
| species mismatch | fail closed | pass |
| intervention mismatch | fail closed | pass |
| endpoint mismatch | fail closed | pass |
| timepoint mismatch | fail closed | pass |
| context-only evidence | fail closed | pass |
| unknown scope | fail closed | pass |

All eight results retain `requires_human_review=true` and
`scientific_judgment=not_performed`. The suite does not measure automatic
scope extraction, biomedical NLI, support/contradiction classification,
real-paper relevance, or clinical correctness.

Machine-readable and human-readable outputs are stored in
`reports/product_scope_controls_v1.json` and
`reports/product_scope_controls_v1.md`.

## Evidence-driven internal iteration

The first live QA run returned two PubMed records with the same normalized
title and made study design hard to see. The product layer was changed to:

1. remove exact normalized-title duplicates before local ranking;
2. avoid fuzzy semantic deduplication, which could hide distinct studies;
3. display up to two PubMed publication types on every card.

After the change, the same live question displayed five distinct titles and
made observational, trial, meta-analysis, and randomized-trial types visible.
This is internal QA evidence only.

## Formative pilot converted into regression risk

The separate pilot attempted two task rows, but both records were
non-evaluable. The audit found timing and task-mode conflicts and showed that
authentic citations can still be irrelevant to the decisive intervention or
endpoint. Consequently:

- pilot count is 1, attempted task rows are 2, evaluable rows are 0;
- VEPS observations are 0, not 0% and not a failed-user rate;
- no time-saving, trust, reuse, or unassisted-completion result is reported;
- no larger study was authorized;
- scope mismatch, context-only, and unknown-scope behavior was converted into
  synthetic deterministic controls.

The controls reduce regression risk in declared metadata. They do not turn the
pilot into validation and do not prove that the software detects those
conditions in arbitrary biomedical text.

## External-source accessibility note

The official NCBI E-utilities endpoints returned the tested PubMed records and
the product generated canonical `https://pubmed.ncbi.nlm.nih.gov/{PMID}/`
links. A separate automated visible-page check encountered an anti-bot/browser
challenge and could not certify end-user page accessibility. Link
accessibility is therefore not claimed as 100%.

## Visual, interaction, and security findings

- The desktop layout maintains a workflow rail, safety boundary, evidence-card
  hierarchy, exact snippet metadata, and human-review controls.
- The narrow layout stacks the workflow and form controls without horizontal
  overflow.
- PubMed-derived text is inserted with `textContent`; the product script does
  not use `innerHTML`, remote scripts, or remote styles.
- `record_sha256` binds the normalized local PubMed record, not the remote
  PubMed webpage bytes and not scientific truth.
- System direction remains `unclear / needs human review` in both modes.

## Closeout conclusion

The local source-bound workflow and its deterministic safeguards are suitable
for a portfolio demonstration. They do not establish user value or clinical
reliability. The product-management decision is to publish v0.4 after the
remaining release gates pass and to stop, rather than recruit more users for a
measurement design that has not produced an evaluable observation.

## Final release verification (2026-08-06)

Completed in the localhost-capable release environment against the v0.4.0
candidate tree, after the audit snapshot above. This section supersedes the
pending-gate statement in the Result summary:

- Python tests: 46/46 passed, including the four loopback-bind checks and the
  launcher check;
- Python lint: `ruff check src apps tests scripts` passed;
- package: `python -m build` produced the wheel and sdist; a fresh
  out-of-tree virtual environment installed the wheel and the `bioevidence
  demo` CLI returned the fixture PMID `1001` with `__version__ == "0.4.0"`;
- release verifier: `scripts/verify_release.py` passed for
  `reports/release_manifest_v0.4.0.json`; that manifest recorded 79 of 80
  tracked files, excluding itself to avoid self-reference, but its verifier did
  not yet enforce exact set equality;
- synthetic scope controls: 8/8 passed (deterministic regeneration);
- Demo core flow: health, two standard tasks, standard search (five cards,
  `candidate_source=fixed_verified_pmids`, `local_ranker=bm25`, every card
  `direction=unclear`), and evidence-pack export (accepted count, 64-hex pack
  SHA-256) passed against the local server;
- privacy checks: the event log contained no question, abstract, snippet, or
  note text; `git ls-files` contained no `private/`, `session_data`,
  participant, or pilot paths; both paths remain Git-ignored;
- GitHub Actions CI: the push/PR workflow runs lint, the conditional
  JavaScript syntax check, the 46-test suite, the release verifier, the wheel
  build, and the fresh-wheel smoke on Python 3.10/3.12/3.13; the merged-commit
  run result is recorded on the GitHub repository checks.

These results are engineering verification of the software contract. They are
not user research, an effectiveness measure, or evidence of product value.

## v0.4.1 integrity-maintenance verification (2026-08-08)

This additive maintenance pass preserved the v0.4.0 tag, manifest, wheel, and
history. It added no model, benchmark, deployment, performance claim, or human
study.

- Python tests: 65/65 passed, including loopback server tests, an omitted-file
  manifest regression, public-example determinism, strict per-event metadata
  allowlists, and event-sequence recovery after restart;
- Python lint: `ruff check src apps tests scripts` passed;
- synthetic scope controls: 8/8 passed and regenerated byte-identically;
- public example: JSON canonical Hash, JSON-declared Hash, and Markdown-displayed
  Hash matched (`27314556024b858f73d43e15798205b1bd4af62db9861cd66c1c7ce4ac9971ca`);
  the incompatible audit-key rename is explicitly versioned as
  `product-evidence-pack-v0.2`;
  the accepted RECOVERY card now retains a mortality result sentence rather
  than an endpoint-definition sentence;
- real export capture: the Standard fixed-PMID Demo downloaded a JSON pack,
  canonical Hash verification passed, and the 1440 x 900 export screenshot
  shows the in-product completed-download state;
- package: the isolated PEP 517 build produced the v0.4.1 wheel and sdist; a
  dependency-resolving, out-of-tree virtual environment installed the wheel
  and returned fixture PMID `1001` with `__version__ == "0.4.1"`;
- wheel SHA-256:
  `13d6351c49a073be5a11864a76fcc9ee866257e18714538bbc33457162d3edd2`;
- release manifest: 82 of 83 tracked files were hashed, with only the manifest
  itself excluded; exact tracked-set equality, every declared Hash, and the
  public-tree boundary passed;
- CI contract: Python 3.10/3.11/3.12/3.13, current Node-24 actions, lint,
  JavaScript syntax, 65 tests, release verification, build, and a dependency-
  resolving fresh-wheel smoke. Public GitHub run and Release state remain
  external publication receipts rather than prepublication manifest claims.

These checks establish package, byte-integrity, privacy-contract, and workflow
behavior only. User value, clinical correctness, open-world retrieval, and
semantic hallucination rates remain unmeasured.
