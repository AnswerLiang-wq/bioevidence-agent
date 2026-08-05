# BioEvidence Agent Product Demo — one-pager (G1 draft)

Status: hypothesis, not validated product evidence

Draft date: 2026-08-05

Baseline: BioEvidence Agent v0.3.0

Proposed product-demo line: v0.4 candidate; no tag or Release authorized

## Product statement

BioEvidence Agent Product Demo helps life-science researchers turn a specific
biomedical claim into a small, traceable evidence pack. It retrieves candidate
PubMed records, presents exact abstract evidence with stable identifiers and
source links, and lets the researcher accept, exclude, annotate, and export
the evidence without delegating the final scientific judgment to the model.

This is a second-stage productization of an existing Agent engineering
prototype. It is not presented as the project's original purpose.

## Target user

Primary target:

- master's students, doctoral students, and junior researchers in life
  science, pharmacy, bioinformatics, or medical research;
- currently search biomedical literature at least weekly;
- need to verify a concrete claim before experiment design, proposal writing,
  or manuscript argumentation;
- can independently judge whether an abstract is relevant to their research.

Excluded from the first study:

- patients or members of the public seeking personal medical advice;
- users asking the tool to make diagnosis or treatment decisions;
- systematic-review teams requiring exhaustive/full-text screening;
- tasks containing patient data, unpublished confidential results, or other
  information the participant cannot safely share;
- users who do not regularly perform biomedical literature work.

## Job to be done

When I need to verify a specific biomedical claim before designing or writing
research, help me find and organize a few inspectable source records so that I
can decide what the evidence says without losing the path back to the original
article.

## Current workflow hypothesis

The assumed manual workflow is:

1. translate a research question into search terms;
2. search PubMed or another literature database;
3. open many records and skim titles/abstracts;
4. decide which records are relevant;
5. copy identifiers, links, and notes into a document or spreadsheet;
6. revisit sources to check whether notes overstate the evidence.

This workflow and its pain points are hypotheses until discovery interviews
and observed tasks confirm, reject, or refine them.

## Core problem

General-purpose language models can produce fluent answers without a stable
mapping from each claim to inspectable source text. Manual search preserves
control but can require repetitive opening, screening, and copying. The
product hypothesis is that a structured, source-bound evidence-pack workflow
can reduce coordination effort while preserving human scientific judgment.

## MVP workflow

1. User enters one concrete, non-clinical biomedical claim or question.
2. System retrieves candidate PubMed records and ranks them.
3. System displays 3–5 evidence cards containing, when available:
   - title and year;
   - PMID and DOI;
   - PubMed source link;
   - exact abstract snippet;
   - normalized-record and snippet SHA-256;
   - provisional `supports / opposes / unclear` suggestion;
   - abstract-only and uncertainty boundaries.
4. User accepts or excludes each card and records a note and their own
   direction judgment.
5. System exports a JSON and Markdown evidence pack with source lineage and
   user decisions.
6. The system logs only the task events required for usability analysis.

Because the existing answerer is weak and benchmark-specific, live-mode
direction defaults to `unclear / needs human review` unless a later component
is separately validated. Model confidence is never displayed as evidence
strength.

## User value hypothesis

For suitable tasks, the workflow may shorten time to the first useful source,
reduce manual copying, and make later verification easier. It may fail when
PubMed retrieval is poor, the abstract omits decisive details, sources
conflict, or users need full-text/systematic-review coverage.

## North-star metric

**Verified Evidence Pack Success (VEPS)**

A session succeeds only if all conditions hold:

1. completed within 12 minutes of task start;
2. exported at least 3 distinct PMID-backed evidence cards;
3. participant marked at least 2 cards useful for the task;
4. every exported source link was accessible during the session;
5. every exported snippet matched the stored abstract bytes and offsets;
6. zero fabricated citations;
7. an evidence-insufficient task was not exported as a certain conclusion.

VEPS is computed per participant-task pair. Pilot and main-study results are
reported separately. With a small purposive sample, VEPS is exploratory and
has no population confidence claim.

## Supporting metrics and operational definitions

| Metric | Operational definition |
|---|---|
| Time to First Useful Evidence | seconds from task start to first card the participant marks useful |
| Evidence-pack completion time | seconds from task start to successful export, or timeout at 12 minutes |
| Manual-flow time change | `(Agent time - manual time) / manual time`, paired only within the same participant and comparable task |
| Top-5 acceptance rate | accepted cards divided by cards shown among the first five |
| Citation accessibility | source links successfully opened during audit divided by exported citations |
| Snippet consistency | byte/offset-valid exported snippets divided by exported snippets |
| Unsupported-conclusion rate | exported conclusion claims lacking an accepted supporting/contradicting card divided by conclusion claims, manually audited |
| Fabricated citations | count of exported identifiers not resolvable to the claimed source; target is zero |
| Unassisted completion | participant completed the defined flow without moderator procedural help |
| Trust | post-task 1–5 response to a fixed trust item, reported with individual values and median |
| Reuse intent | post-task 1–5 response to a fixed reuse item, reported with individual values and median |
| Failure types | pre-defined taxonomy plus new observed categories; report counts and denominators |

Manual and Agent timing comparisons require task-order recording. They are not
called causal efficiency gains in this small, non-randomized study.

## Product hypotheses

- H1: at least some target users can independently complete the evidence-pack
  flow without moderator help.
- H2: exact snippets and direct source links reduce re-verification friction.
- H3: accept/exclude/note controls are more trustworthy than an uneditable
  generated conclusion.
- H4: the standard task is sufficiently representative to reveal workflow
  problems but does not expose participant research secrets.
- H5: a live PubMed mode adds value for real tasks despite weaker and more
  variable retrieval than the closed benchmark.

## Risk hypotheses

- Users may mistake ranking or model confidence for evidence strength.
- A relevant abstract may not contain the decisive full-text result.
- Provisional direction may anchor user judgment.
- Live external-service failures may prevent task completion.
- Research questions or notes may contain identifying or confidential data.
- A fast but incomplete evidence pack may create false confidence.

## Scope for the first product-demo iteration

In scope:

- local web application;
- one standard real-PubMed task and a participant-owned safe task;
- 3–5 evidence cards;
- accept/exclude/note and user direction;
- JSON/Markdown export;
- minimal local event log for task metrics;
- clear empty, partial, timeout, and external-service error states;
- medical-advice and abstract-only boundary messages;
- a fixed-PMID standard-demo mode and a labelled live-research mode; both
  fetch current records from PubMed and therefore depend on external access.

Out of scope:

- diagnosis, treatment, or patient-specific recommendations;
- exhaustive systematic review or full-text claim;
- autonomous final scientific conclusions;
- login, cloud account, payment, sharing, or long-term memory;
- multi-Agent orchestration or a large backend framework;
- production deployment and analytics collection;
- any new model or feature not justified by observed user evidence.

## Engineering boundary

- Preserve `src/bioevidence/` as the Agent capability layer.
- Add the UI and orchestration adapter under `apps/product_demo/`.
- Import existing PubMed, retrieval, evidence inspection, hashing, and typed
  tool contracts; do not duplicate them.
- Keep public blank research templates separate from ignored raw participant
  records.
- Keep Agent benchmark numbers, Product Demo technical checks, and real-user
  metrics in separate tables with version, sample, method, and limitations.

## README dual-entry draft

The existing engineering narrative remains first-class.

1. **Agent Engineering**: architecture, retrieval/reranking, typed tools,
   provenance, frozen public benchmark, CI, reproducibility, and limitations.
2. **Product Demo / Case Study**: target user, evidence-pack workflow, local
   startup, study protocol, anonymized aggregate findings, evidence-driven
   iteration, product boundaries, and roadmap.

No README metric changes are made before real product evidence exists.

## G1 decisions requested

1. Confirm or revise the primary target user.
2. Confirm the claim-verification scenario as the single MVP scenario.
3. Confirm VEPS and the 12-minute threshold as an initial test definition.
4. Confirm the safe default: live direction is `unclear / needs human review`
   until separately validated.
5. Confirm the repository isolation and dual-entry README plan.
6. Confirm that product work remains local on the feature branch until the
   final publication gate.
