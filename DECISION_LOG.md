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
- Status: implemented; v0.4 closeout continues on
  `codex/bioevidence-portfolio-v0.4`
- Evidence: `main`, tag `v0.3.0`, GitHub Release, and hosted CI all point to
  commit `46afee6ca5b5a4066cdcafa9532abcc7d6e805ef`.
- Decision: begin on `codex/bioevidence-product-demo`, preserve its history,
  and complete the portfolio closeout on
  `codex/bioevidence-portfolio-v0.4`. Do not rewrite the v0.3.0 tag. Publish
  v0.4 only after its explicit release checklist passes.
- Rollback: discard or retain the feature branch; the baseline remains
  addressable by both `main` and `v0.3.0`.

## D-003 — Do not reuse the PubMedQA answerer as an open-world judge

- Date: 2026-08-05
- Status: decided and retained for v0.4
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
- Status: implemented
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

## D-010 — Treat PILOT01 as formative and non-evaluable

- Date: 2026-08-06
- Status: decided
- Evidence: one anonymous pilot produced two attempted task rows, but timing,
  mode, logging, and evidence-scope conflicts made both
  `metric_record_complete=false` and `veps=not_evaluable`.
- Decision: preserve the original private evidence and additive audit; report
  1 formative pilot, 2 attempted rows, 0 evaluable rows, and 0 VEPS
  observations.
- Claims excluded: VEPS percentage, time saving, trust, reuse,
  unassisted-completion, or product-value conclusions.
- Integrity rule: do not overwrite the first attempt or repair it into a
  successful observation.

## D-011 — Stop human-study scale-up for the portfolio release

- Date: 2026-08-06
- Status: decided
- Evidence: additional recruitment would not resolve the measurement defects
  in the consumed pilot, and the original engineering/product portfolio is
  otherwise demonstrable without a causal user-value claim.
- Decision: cancel the planned second pilot and larger main study as v0.4
  completion requirements. Close the project through automated engineering,
  safety, privacy, packaging, and documentation gates.
- Interpretation: user value remains unknown. This is a resource and evidence
  quality decision, not a positive or negative product-effect conclusion.

## D-012 — Convert evidence-scope risks into synthetic fail-closed controls

- Date: 2026-08-06
- Status: implemented
- Evidence: authentic PMIDs in the formative pilot still included decisive
  endpoint and intervention mismatches.
- Decision: freeze eight synthetic declared-metadata cases covering direct
  match; population, species, intervention, endpoint, and timepoint mismatch;
  context-only evidence; and unknown scope. A flagged case cannot become a
  candidate for decisive human review.
- Boundary: these controls do not extract metadata from prose, judge real
  papers, infer evidence direction, or measure biomedical NLI accuracy. Every
  case retains `requires_human_review=true`.

## D-013 — Define a finite v0.4 portfolio stop condition

- Date: 2026-08-06
- Status: decided
- Decision: publish and stop when lint, full tests, package build, release
  manifest, privacy scan, local search/review/export, eight scope controls,
  documentation consistency, and GitHub tag/Release all pass.
- Out of scope after closeout: new model training, benchmark expansion, cloud
  deployment, and additional human research.
