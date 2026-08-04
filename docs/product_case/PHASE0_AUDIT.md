# Phase 0 audit: Agent baseline and productization boundary

Audit date: 2026-08-05

Productization branch: `codex/bioevidence-product-demo`

Engineering baseline: `v0.3.0`, commit
`46afee6ca5b5a4066cdcafa9532abcc7d6e805ef`

## Repository decision

Two local directories were inspected.

- `BioEvidenceAgent` is the historical development workspace. It is an
  untracked directory inside an unrelated parent repository and contains the
  archived Day 27–63 governance work. It remains evidence for historical
  results, but is not a safe product-development Git root.
- `BioEvidenceAgent-Portfolio` is the independent public repository. Its
  `main` branch was clean and synchronized with
  `AnswerLiang-wq/bioevidence-agent`; `v0.3.0` and its GitHub Release both
  remain intact. This repository is the productization baseline.

Product work therefore proceeds only on the local feature branch. Nothing is
merged, pushed, tagged, or released without the final publication gate.

## Verified engineering baseline

The following checks were rerun rather than inferred from earlier notes.

| Evidence | Observed result |
|---|---|
| Ruff | pass |
| Lightweight tests | 8/8 pass |
| Release verifier | 44 hashed artifacts pass |
| Fresh wheel installation | pass |
| Out-of-tree CLI demo | pass; one PMID, exact snippet hash, and three typed-tool calls |
| Hosted CI | pass on Python 3.10, 3.12, and 3.13 at baseline commit |
| Live PubMed path | ESearch returned 3 PMIDs and EFetch returned 3 abstracts without an API key |

The direct command `python -m bioevidence.portfolio_cli` is not a supported
uninstalled invocation because the project uses a `src/` layout. The
documented installed `bioevidence demo` entry point works from outside the
repository.

## Reverified technical results

All numbers below are bound to frozen aggregate reports, not reconstructed
from memory.

- Title-assisted public closed-corpus answer baseline: 500 cases, accuracy
  0.558, macro-F1 0.383.
- Abstract-only final reranker: 500 cases, Recall@10 0.990, MRR 0.982,
  nDCG@10 0.984; 12 rank drops and 5 top-10 misses.
- Evidence-utilization diagnostic: 500 non-test cases, fixed 5-fold OOF;
  correct-context accuracy 0.536 versus shuffled-context 0.486; exact
  McNemar `p=0.066`.
- Structural grounding in the original public benchmark is byte-level
  integrity evidence. It is not an independently reviewed semantic
  hallucination or citation-correctness rate.

These remain Agent Engineering metrics. They are not Product Demo user
metrics and will not be used as evidence of user value.

## Reusable assets

- zero-dependency PubMed E-utilities search/fetch client with conservative
  rate limiting and bounded retry;
- normalized `CorpusDocument` with PMID, DOI, title, abstract, source URL,
  publication metadata, and content SHA-256;
- BM25, multilingual-E5, RRF, and cross-encoder retriever adapters;
- typed `search`, `fetch`, `resolve`, and `inspect` tools with bounded traces;
- exact abstract sentence offsets and snippet SHA-256;
- structured response validator and explicit scope limits;
- deterministic tests, dependency locks, CI, release verifier, and a small
  model-free fixture.

## Product gaps

The current CLI demo is an engineering smoke test, not a usable research
workflow. It retrieves one article from a three-document synthetic fixture.
It does not yet provide:

- a local web interface;
- 3–5 evidence cards;
- live-query evidence-pack creation;
- user accept/exclude/note actions;
- JSON/Markdown evidence-pack export;
- research-session event logging;
- understandable partial-failure states;
- user-study timing and VEPS computation;
- a validated open-world support/oppose classifier.

The last gap is deliberate. The existing answerer was trained for public
PubMedQA article-level labels and has macro-F1 0.383; applying it to arbitrary
research claims would exceed its evidence. For live tasks, the safe default is
an explicitly provisional `unclear / needs human review` suggestion while the
user makes the final evidence decision.

## Proposed code and data isolation

```text
src/bioevidence/                 unchanged Agent capability layer
apps/product_demo/               local web UI, service adapter, static assets
tests/product_demo/              product-only unit/integration tests
research/templates/              public, blank research protocols/templates
docs/product_case/               public aggregate case-study material
private/user_research/           raw anonymous records; Git-ignored
private/identity_map/             optional identity mapping; Git-ignored
```

The product adapter will import existing core types instead of copying
retrieval, PubMed, evidence-inspection, or hashing logic. A core change is
allowed only when the adapter cannot meet a tested requirement; such a change
must have a regression test and a documented rollback.

Proposed runtime modes:

1. `standard-demo`: a deterministic real-PubMed task fixture for unattended
   demonstration and repeatable user-test rehearsal;
2. `live-research`: PubMed ESearch/EFetch candidate discovery followed by
   local ranking and evidence extraction for participant-owned questions;
3. an explicit degraded BM25 mode if local transformer dependencies are not
   installed, visibly labelled so its measurements are not mixed with the
   full mode.

## Privacy, copyright, and safety audit

- No common GitHub/OpenAI token formats or private-key headers were found in
  the current tree or Git history.
- Runtime code reads optional NCBI credentials from environment variables and
  exposes only key presence, not values.
- No raw PubMedQA files, model weights, caches, participant records, patient
  data, or interview transcripts are tracked in the public repository.
- Dataset/model source and license notices exist. Raw PubMedQA data and model
  weights are downloaded by users rather than redistributed.
- The current `.gitignore` covers generic data, runs, `.env`, keys, and
  caches, but does not yet explicitly cover raw user-research directories,
  recordings, transcripts, or identity maps. Stage 2 must close this gap
  before any recruitment begins.
- The product remains research support. It must reject or redirect requests
  for patient-specific diagnosis/treatment and must never present an abstract
  as a full-text review.

## Existing resume narrative

The currently located resume and evidence-ledger materials describe only the
Agent Engineering baseline. Their metrics match the frozen reports and keep
the public/closed-corpus, `p=0.066`, and semantic-grounding limitations. No
verified product-user claim currently exists. A product-manager version must
not be written as completed experience until real participant data and the
G5 fact check exist.

## Risks to validate

1. Live PubMed relevance may be much worse than the closed benchmark.
2. Abstract-only evidence may be insufficient for claim-level decisions.
3. Unvalidated direction suggestions may reduce rather than improve trust.
4. Model download and CPU latency may harm unattended task completion.
5. Users may interpret source hashes as scientific correctness.
6. Real research questions may contain unpublished or identifying material.
7. Five participants provide exploratory product evidence, not population
   inference.

## Safe rollback

The baseline is unchanged on `main` and at `v0.3.0`. Abandoning the feature
branch restores the exact public engineering project; no migration or data
conversion is required.
