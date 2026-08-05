# Decision log

This log separates observed evidence, decisions, and pending hypotheses. It
does not contain participant data.

## D-001 — Productize the independent portfolio repository

- Date: 2026-08-05
- Status: decided
- Evidence: the historical `BioEvidenceAgent` directory is untracked inside
  an unrelated parent Git repository; `BioEvidenceAgent-Portfolio` is an
  independent clean repository synchronized with the public v0.3.0 release.
- Decision: use `BioEvidenceAgent-Portfolio` as the engineering baseline and
  productization worktree.
- Rejected alternative: adding product code to the historical workspace,
  which would mix archived governance files with an unrelated parent history.
- Baseline impact: none.

## D-002 — Preserve v0.3.0 and work on a local feature branch

- Date: 2026-08-05
- Status: decided
- Evidence: `main`, tag `v0.3.0`, GitHub Release, and hosted CI all point to
  commit `46afee6ca5b5a4066cdcafa9532abcc7d6e805ef`.
- Decision: use `codex/bioevidence-product-demo`; do not rewrite history,
  merge, push, tag, or release before the corresponding user gate.
- Rollback: discard or retain the feature branch; the baseline remains
  addressable by both `main` and `v0.3.0`.

## D-003 — Do not reuse the PubMedQA answerer as an open-world judge

- Date: 2026-08-05
- Status: decided for the first prototype; subject to G1 confirmation
- Evidence: public-test macro-F1 is 0.383, `maybe` F1 is 0.063, and the
  correct-versus-shuffled diagnostic has `p=0.066`. Training labels represent
  PubMedQA article-level yes/no/maybe questions, not arbitrary claim-level
  evidence direction.
- Decision: live evidence cards default to `unclear / needs human review`.
  The participant records their own final direction. No model probability is
  presented as evidence strength.
- Trade-off: this reduces apparent automation but avoids unsupported
  scientific judgment and gives the user a meaningful correction action.

## D-004 — Isolate product orchestration from Agent capability code

- Date: 2026-08-05
- Status: implemented after G1 confirmation
- Evidence: PubMed, corpus, retriever, typed-tool, exact-span, and hash
  components already expose reusable Python interfaces.
- Decision: place web UI and workflow orchestration in `apps/product_demo/`,
  tests in `tests/product_demo/`, public research templates in
  `research/templates/`, and raw records under ignored `private/` paths.
- Core-change rule: modify `src/bioevidence/` only for a demonstrated adapter
  blocker, with a regression test and rollback note.

## D-005 — Separate fixed-candidate and live-discovery task modes

- Date: 2026-08-05
- Status: implemented and internally tested
- Evidence: the packaged fixture is stable but not representative of a live
  research workflow; PubMed ESearch/EFetch works but depends on an external
  service and has variable retrieval results.
- Decision: provide a fixed-PMID standard-demo mode for repeatable candidate
  identity and a visibly labelled live-research mode for participant-owned
  safe questions. Both fetch current PubMed records and remain exposed to
  external-service change or failure. Metrics from different modes are never
  pooled without an explicit mode field and comparable task definition.

## D-007 — Preserve the v0.3.0 manifest as historical evidence

- Date: 2026-08-05
- Status: decided
- Evidence: Stage 2 adds explicit private-research ignore rules, so the current
  `.gitignore` no longer matches the SHA-256 frozen in the v0.3.0 release
  manifest. The original tagged checkout and GitHub Release remain unchanged.
- Decision: do not update `reports/release_manifest_v0.3.0.json`. Its full-tree
  verifier is expected to fail on the product branch once a v0.3.0-bound file
  changes. Create a separate product-version manifest at final release audit.
- Verification during development: run all tests, lint, privacy checks, and
  explicit frozen-report checks; rerun the v0.3.0 verifier only against the
  v0.3.0 tagged checkout.

## D-008 — Use a zero-new-dependency local web layer

- Date: 2026-08-05
- Status: implemented
- Evidence: the repository already depends on Python and exposes reusable
  PubMed, BM25, typed-tool, exact-snippet, and hashing interfaces. The first
  product iteration does not require accounts, cloud persistence, or a large
  frontend framework.
- Decision: use Python's local `ThreadingHTTPServer` plus static HTML, CSS, and
  JavaScript. Bind only to `127.0.0.1`, apply a restrictive CSP, keep session
  evidence in memory, and write only allowlisted privacy-minimized events.
- Trade-off: this is appropriate for moderated local research, not a
  production deployment architecture.

## D-009 — Deduplicate only exact normalized titles in live results

- Date: 2026-08-05
- Status: implemented after internal live QA
- Evidence: a real tocilizumab query returned two language/indexing records
  with the same normalized title. The duplicate displaced another candidate
  in a five-card interface.
- Decision: remove exact normalized-title duplicates before BM25 ranking and
  show PubMed publication types on each card. Do not apply fuzzy semantic
  deduplication without user evidence because similar titles can describe
  scientifically distinct studies.
- Baseline impact: none; the change is confined to `apps/product_demo/`.
