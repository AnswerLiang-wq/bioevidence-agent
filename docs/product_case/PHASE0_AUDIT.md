# Phase 0 audit and v0.4 productization boundary

Original baseline audit: 2026-08-05

Portfolio closeout update: 2026-08-06

Productization branch: `codex/bioevidence-portfolio-v0.4`

Engineering baseline: `v0.3.0`, commit
`46afee6ca5b5a4066cdcafa9532abcc7d6e805ef`

## Repository decision

Two local directories were inspected.

- `BioEvidenceAgent` is the historical development workspace. It remains
  evidence for the Day 27–63 work, but is not the public product-development
  Git root.
- `BioEvidenceAgent-Portfolio` is the independent public repository. Its
  `main` branch and `v0.3.0` tag preserve the engineering baseline.

Product work therefore proceeded in the independent repository on a feature
branch. Raw pilot records remained under Git-ignored `private/` paths. The
historical v0.3.0 manifest was not rewritten to describe later product files.

## Reverified engineering baseline

The following checks were rerun against the v0.3.0 baseline rather than
inferred from earlier notes.

| Evidence | Observed result |
|---|---|
| Ruff | pass |
| Lightweight tests | 8/8 pass |
| Release verifier | 44 hashed artifacts pass |
| Fresh wheel installation | pass |
| Out-of-tree CLI demo | pass; one PMID, exact snippet hash, and three typed-tool calls |
| Hosted CI | pass on Python 3.10, 3.12, and 3.13 at the baseline commit |
| Live PubMed smoke | ESearch returned 3 PMIDs and EFetch returned 3 abstracts without an API key |

The direct command `python -m bioevidence.portfolio_cli` is not a supported
uninstalled invocation because the project uses a `src/` layout. The installed
`bioevidence demo` entry point works from outside the repository.

## Frozen Agent Engineering results

All numbers below remain bound to public aggregate reports.

- Title-assisted closed-corpus answer baseline: 500 official public test
  cases, accuracy 0.558, macro-F1 0.383.
- Abstract-only final reranker: 500 official public test cases, Recall@10
  0.990, MRR 0.982, nDCG@10 0.984; 12 rank drops and 5 top-10 misses.
- Evidence-utilization diagnostic: 500 official non-test cases, fixed
  five-fold out-of-fold procedure; correct-context accuracy 0.536 versus
  shuffled-context accuracy 0.486; exact McNemar `p=0.066`.
- Structural grounding in the original public benchmark is byte-level
  integrity evidence. It is not an independently reviewed semantic
  hallucination or citation-correctness rate.
- Frozen title-assisted end-to-end latency was median 1.109 seconds and p95
  1.235 seconds per query on local CPU; model/API cost was USD 0.00.

These are Agent Engineering results. They are not Product Demo user metrics,
clinical claims, or estimates of open-world performance.

## Reusable engineering assets

- zero-dependency PubMed E-utilities search/fetch client with conservative
  rate limiting and bounded retry;
- normalized `CorpusDocument` with PMID, DOI, title, abstract, source URL,
  publication metadata, and content SHA-256;
- BM25, multilingual-E5, RRF, and cross-encoder retriever adapters;
- typed `search`, `fetch`, `resolve`, and `inspect` tools with bounded traces;
- exact abstract sentence offsets and snippet SHA-256;
- structured response validation and explicit scope limits;
- deterministic tests, dependency locks, CI, a release verifier, and a small
  model-free fixture.

## Productization delivered for v0.4

The portfolio branch adds a local source-bound workflow without rewriting the
Agent capability layer:

- local web interface bound to `127.0.0.1`;
- fixed-candidate Standard and live PubMed discovery modes;
- 3–5 evidence cards with publication types and provenance;
- human accept/exclude/note/direction controls;
- JSON and Markdown evidence-pack export;
- privacy-minimized local event logging;
- typed empty, partial, timeout, and external-service error states;
- exact-title duplicate removal in live results;
- deterministic scope-policy controls over synthetic declared metadata.

The web Demo uses local BM25 ranking. The E5/RRF/reranker stack remains a
separate evaluated engineering path and is not represented as the Demo's
runtime.

## Formative pilot audit

One anonymous, consented pilot attempted one manual and one Agent task. Its
private additive audit found no fabricated exported PMID and no recording or
privacy incident, but it also found conflicting timestamps, a task-mode
switch, incomplete source-open telemetry, invalid placeholder values, and
evidence-scope mismatches.

The deterministic outcome is:

- formative participants: 1;
- attempted task rows: 2;
- evaluable task rows: 0;
- VEPS observations: 0;
- eligible paired time comparisons: 0;
- main-study participants and tasks: 0 and 0.

Therefore the pilot supplies workflow-risk discovery only. It does not supply
efficiency, success-rate, trust, reuse, or user-value evidence. Raw records,
row-level observations, and private evidence remain outside the public tree.

## Product decision: stop recruitment and automate the lesson

The earlier plan would have repaired the protocol and recruited additional
participants. That would be appropriate for a product-validation program, but
it is disproportionate to the current portfolio objective and cannot rescue
the already non-evaluable observations.

The v0.4 decision is to preserve the failed measurement, stop further human
research, and encode its most generalizable safety lesson as eight synthetic
declared-metadata controls: direct scope match; population, species,
intervention, endpoint, and timepoint mismatches; context-only evidence; and
unknown scope. Seven fail closed before an item can be a candidate for
decisive human review.

These controls test deterministic policy wiring only. They do not perform
biomedical natural-language inference or validate real-paper scientific
judgments.

## Privacy, copyright, and safety boundary

- No raw PubMedQA data, model weights, caches, participant records, patient
  data, recordings, or transcripts are tracked in the public repository.
- Runtime code reads optional NCBI credentials from environment variables and
  reports key presence only, never the value.
- PubMed records are fetched from the official service; the repository stores
  metadata, source links, and derived reproducibility artifacts rather than a
  redistributed raw corpus.
- The product is research support. It rejects or redirects patient-specific
  advice and never presents an abstract as a full-text review.
- A hash proves byte identity, not scientific truth.

## Remaining limitations at closeout

1. Live PubMed relevance and availability remain query-dependent.
2. Abstracts can omit decisive methods, endpoints, harms, or subgroup details.
3. User-set direction can still be mistaken or anchored by card ordering.
4. The Product Demo has no validated time-saving or usability result.
5. Declared-metadata controls do not extract scope from natural language.
6. Neither the public benchmark nor the pilot supports a clinical-reliability
   or real-world generalization claim.

## Safe rollback and release rule

The v0.3.0 baseline remains addressable by its tag. v0.4 is released only if
the final lint, tests, package build, release-manifest verification, privacy
scan, Demo workflow, documentation consistency, and eight scope controls pass.
After that release, the portfolio project stops; no larger user study is a
v0.4 completion requirement.
