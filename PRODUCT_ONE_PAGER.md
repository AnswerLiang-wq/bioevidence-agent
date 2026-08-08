# BioEvidence Agent v0.4.1 — portfolio one-pager

Status: released portfolio artifact; engineering validation completed, user
value not validated

Closeout decision date: 2026-08-06

Engineering baseline: `v0.3.0`

Product release: `v0.4.1` integrity maintenance release; it supersedes v0.4.0
without rewriting the historical v0.4.0 tag, manifest, wheel, or Git history

## Product statement

BioEvidence Agent is designed to help a life-science researcher turn one
bounded biomedical claim into a small, inspectable PubMed evidence pack. It
retrieves candidate records, displays exact abstract snippets with stable
source identifiers, and lets the user accept, exclude, annotate, and export
evidence without asking the system to make a clinical decision. This intended
value remains a product hypothesis rather than a validated user outcome.

The product layer is a portfolio demonstration built on an existing Agent
engineering project. It is not a validated medical product, systematic-review
tool, or autonomous scientific judge.

## Target user and job to be done

Hypothesized primary user: a life-science graduate student or junior researcher
who searches biomedical literature regularly and can judge whether an abstract
is relevant.

Hypothesized job to be done:

> When I need to check a specific biomedical claim before research planning or
> writing, help me organize a few source-bound records so I can inspect the
> evidence and retain a path back to the original article.

Excluded uses include patient-specific advice, diagnosis or treatment
decisions, confidential or identifiable inputs, exhaustive systematic review,
and claims that require full-text evidence.

## Product workflow

1. Enter one concrete, non-clinical biomedical question.
2. Retrieve candidate PubMed records in a labelled Standard or Live mode.
3. Review 3–5 cards containing title, year, PMID/DOI, PubMed link, exact
   abstract snippet, offsets, and SHA-256 provenance.
4. Accept or exclude cards, add a note, and record a human evidence direction.
5. Export a JSON or Markdown evidence pack with source lineage and user
   decisions.

The web Product Demo uses PubMed retrieval plus a local BM25 ranking adapter.
The full multilingual-E5, RRF, and cross-encoder stack belongs to the separate
Agent Engineering evaluation path; it is not silently attributed to the web
workflow.

## What is actually validated

| Evidence layer | Frozen result | Defensible interpretation |
|---|---|---|
| Public title-assisted PubMedQA test | 500 cases; answer accuracy 0.558, macro-F1 0.383 | closed-corpus engineering baseline, not open-world accuracy |
| Abstract-only retrieval stress test | 500 cases; reranked Recall@10 0.990, MRR 0.982, nDCG@10 0.984; 12 rank drops and 5 top-10 misses | retrieval remains strong after removing indexed titles, with visible failures |
| Evidence-utilization negative control | 500 non-test OOF cases; correct context 0.536 vs shuffled 0.486; exact McNemar `p=0.066` | context-sensitive pattern, not strong evidence of reliable reasoning |
| Product engineering checks | 65/65 deterministic tests passed in the full localhost-capable run; Ruff passed | software behavior, not user success |
| Synthetic scope controls | 8/8 passed | deterministic policy over declared metadata, not biomedical NLI accuracy |
| Formative human pilot | 1 anonymous pilot, 2 attempted task rows, 0 evaluable tasks, 0 VEPS observations | workflow risks discovered; no efficiency, trust, reuse, or product-value conclusion |

All model inference in the frozen engineering experiments was local CPU, with
USD 0.00 model/API cost. The latency figures belong to that fixed benchmark,
not to general PubMed availability or end-user task time.

## Product decision after the formative pilot

The single pilot exposed conflicting timestamps, a Standard-to-Live mode
switch, incomplete source-open telemetry, invalid spreadsheet placeholders,
and two accepted citations that did not match the decisive intervention or
endpoint. Because neither attempted task produced a reliable metric record,
scaling recruitment would have generated more rows without repairing
measurement validity.

The product decision was therefore to:

- retain the original private pilot as an immutable formative failure;
- report 0 evaluable tasks and 0 VEPS observations rather than repair the data
  into a success;
- cancel the planned larger user study for this portfolio release;
- convert the highest-risk evidence-scope failures into deterministic,
  synthetic regression controls;
- stop after a reproducible, privacy-safe v0.4 portfolio release.

This decision does not show that users benefit or do not benefit. It shows
that the available pilot cannot answer that question and that further
recruitment was not justified for the current portfolio objective.

## Eight synthetic scope controls

The v0.4 suite covers one direct declared-scope match and seven fail-closed
conditions:

1. direct scope match;
2. population mismatch;
3. species mismatch;
4. intervention mismatch;
5. endpoint mismatch;
6. timepoint mismatch;
7. context-only evidence must not be treated as decisive;
8. an unknown scope dimension must not be treated as decisive.

The controls compare pre-declared synthetic metadata only. They do not extract
scope from prose, judge real papers, determine support or contradiction, or
measure clinical correctness. Every result still requires human review.

## Claims deliberately not made

- no validated time saving, VEPS rate, trust, reuse, or unassisted-completion
  claim;
- no open-world PubMed recall or semantic citation-correctness claim;
- no claim that the web Demo runs the full hybrid/reranker stack;
- no biomedical NLI or automatic endpoint-extraction accuracy claim;
- no zero-hallucination, clinical reliability, safety, or treatment claim;
- no population inference from one purposive formative pilot.

## Portfolio stop conditions

The v0.4.1 maintenance effort ends when lint and tests pass, the package and
release manifest verify, the local Demo completes search/review/export, all 8
scope controls pass, private pilot material remains outside Git, documentation
numbers agree, and the GitHub tag and Release are published. It does not expand
into another model-training cycle, a larger benchmark, cloud deployment, or
additional human research.

The full product reasoning is documented in
[`docs/product_case/PORTFOLIO_CASE_STUDY.md`](docs/product_case/PORTFOLIO_CASE_STUDY.md).
