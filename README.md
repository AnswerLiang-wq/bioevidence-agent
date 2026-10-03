# BioEvidence Agent

BioEvidence Agent is a portfolio project with two deliberately separate
entry points:

Release status: **v0.5.0 is a local candidate and has not been published.** No
tag, GitHub release, or deployment exists for it.

Linux CI has run for this candidate. Commit
`734277f67bfe31354c6da8813cadfce5c31cae30` passed the lightweight matrix on
Python 3.10–3.13
([run 37102392291](https://github.com/AnswerLiang-wq/bioevidence-agent/actions/runs/37102392291)):
each leg reported 339 passed and 3 skipped. The three skips are guarded
cross-checks against machine-local frozen artifacts — the prepared benchmark
under `data/` and records under the ignored `private/` — that a public checkout
does not carry, so they run only locally, where the full suite is 342 passing.
**That result belongs to `734277f` alone.** Any later commit, including the one
that updates this paragraph, carries its own CI result and is not covered by
that run.

What v0.5.0 adds over v0.4.1:

- a runnable LLM agent and its controls — `llm_agent`, `fixed_context_agent`,
  `prompt_variants`, `audit` — plus the product-response contract checks;
- evaluation tooling for the frozen-sample and pipeline comparisons
  (`pubmedqa_sample`, `scripts/compare_pipelines.py`) and offline tests for
  both, which script the model transport and call no API;
- two frozen public evaluation samples, a machine-readable experiment summary,
  and three documents: [claims registry](docs/claims_registry.md),
  [experiment log](docs/experiment_log.md) and
  [interview notes](docs/interview_prep.md);
- synthetic fixture records and a `react-demo` command that walks the ReAct
  loop against them with a scripted client, so the loop is inspectable without
  a key or a network call.

The published release remains **v0.4.1**, an integrity-only maintenance
release over the v0.4.0 portfolio closeout. It adds no model, benchmark,
deployment, or human study.

The exported evidence-pack contract is `product-evidence-pack-v0.2`. It
replaces the ambiguous v0.1 audit key `all_sources_pubmed` with the narrower
`all_source_urls_pubmed_formatted`; consumers should branch on `pack_version`.

| Entry point | What it demonstrates | Retrieval path |
|---|---|---|
| **Agent Engineering** | reproducible public-benchmark retrieval, answer diagnostics, typed tools, and byte-verifiable citation lineage | BM25 + pinned multilingual-E5 + equal-weight RRF + pinned cross-encoder reranker |
| **AI Product Case Study** | a local, human-in-the-loop PubMed evidence-card workflow | fixed PMIDs or PubMed ESearch/EFetch, followed by local BM25 |

Two reading paths run through this repository:

- **Engineering and evaluation.** Start with the
  [three API evaluation rounds](#three-api-evaluation-rounds) note and the
  [public-benchmark results](#real-public-benchmark-results) further down, then
  follow the [experiment log](docs/experiment_log.md) for full sample, scoring
  and reproduction detail.
- **Product reasoning.** Start with the
  [AI Product Case Study](docs/product_case/PORTFOLIO_CASE_STUDY.md), then come
  back to the numbers with the trade-offs already in mind.

The web Product Demo does **not** run the full hybrid benchmark stack. The two
paths reuse the same evidence types and provenance checks, but answer
different questions: the engineering path measures a frozen closed-corpus
pipeline; the product path demonstrates a cautious research workflow over
live PubMed abstracts.

The central engineering question is not merely “can the system find a paper?”
It is “does the apparent retrieval success survive removal of title leakage,
and does the answerer measurably use the retrieved abstract?”

```text
question
  ├─ BM25 ───────────┐
  ├─ multilingual-E5 ├─ RRF ─ cross-encoder ─ typed evidence tools
  └──────────────────┘                         ├─ PMID / PubMed URL
                                               ├─ exact abstract span
                                               ├─ source + snippet SHA-256
                                               └─ bounded tool trace
```

The implementation and the frozen benchmark results follow below; the
[AI Product Case Study](docs/product_case/PORTFOLIO_CASE_STUDY.md) covers the
product reasoning behind the same evidence types.

## Five-minute Product Demo

The web application runs from a repository checkout and needs network access
to official NCBI E-utilities:

```bash
git clone https://github.com/AnswerLiang-wq/bioevidence-agent.git
cd bioevidence-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python scripts/run_product_demo.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765), keep **Standard task**
selected, confirm the safety statement, and choose **生成证据卡片**. Then:

1. open at least one PubMed source;
2. mark cards as accepted or excluded and record a human direction;
3. write a bounded synthesis and export JSON or Markdown;
4. inspect the exported PMID, exact abstract span, record/snippet SHA-256, and
   scope limits.

A deterministic, abridged output is included as
[`examples/public_evidence_pack_example.md`](examples/public_evidence_pack_example.md).
It shows the export shape and source links without claiming to be a current
live-PubMed response; its record hashes bind the included abridged example
records.

Screenshots of the Standard-task workflow (fixed public PMIDs only):
[evidence cards](docs/product_case/screenshots/product_demo_standard_cards.png)
and [verified evidence-pack export](docs/product_case/screenshots/product_demo_evidence_export.png).

Use **Live research** only with a public, non-sensitive research question.
The server binds locally; the official UI sends only event-specific,
allowlisted action metadata, and the server rejects unknown metadata fields.
Question, abstract, and note text are not accepted by that event contract. The
Product Demo still requires human review and does not produce an autonomous
biomedical verdict.

Packaging boundary: the wheel installs the core `bioevidence` CLI and its
model-free fixture. The web app under `apps/product_demo/` and its launcher
under `scripts/` are repository assets, so run the web demo from a clone rather
than expecting a `bioevidence web` command from the wheel.

## Real public-benchmark results

All figures below are frozen public PubMedQA diagnostics. They are not a
private blind test and are not estimates of clinical validity.

### Retrieval: title-assisted versus abstract-only

PubMedQA questions are article titles or title-derived. In the original
condition the indexed document also contained its title, creating an
unusually easy source-localization task. The stress test removed title, PMID,
label, `LONG_ANSWER`, `final_decision`, and target identity from every
retrieval/reranking passage.

| Passage condition / system | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| title-assisted / all four systems | 1.000 | 1.000 | 1.000 | 1.000 |
| abstract-only / BM25 | 0.976 | 0.980 | 0.953 | 0.960 |
| abstract-only / multilingual-E5 | 0.990 | 0.990 | 0.979 | 0.982 |
| abstract-only / hybrid RRF | 0.984 | 0.988 | 0.978 | 0.980 |
| abstract-only / hybrid + reranker | 0.990 | 0.990 | 0.982 | 0.984 |

The reranked abstract-only system produced 12 real rank drops; five targets
fell outside top 10 and the worst fell from title-assisted rank 1 to
abstract-only rank 774. This is a useful stress result, not a failure to hide.

### Answering and evidence-utilization controls

The frozen v0.2 answer baseline scored **0.558 accuracy** and **0.383
macro-F1** on the 500 official public test cases. It was particularly weak on
`maybe` (F1 0.063).

To test whether context contributes information, v0.3 adds fixed stratified
five-fold out-of-fold controls on the separate 500 official non-test records
(seed `20260729`):

| OOF control | Accuracy | Macro-F1 |
|---|---:|---:|
| fold majority | 0.552 | 0.237 |
| question only | 0.512 | 0.381 |
| context only | 0.534 | 0.364 |
| question + correct context | 0.536 | 0.356 |
| question + shuffled context | 0.486 | 0.302 |

Correct context improved accuracy over question-only by 2.4 percentage points
but reduced macro-F1 by 2.6 points. It exceeded shuffled-context accuracy by
5.0 points; the paired exact McNemar p-value was 0.066. The defensible
conclusion is that the fixed linear model is context-sensitive, but this is
not strong evidence of reliable evidence use. The answerer—not source
localization—remains the bottleneck.

## Three API evaluation rounds

Separate from the local-CPU benchmark above, three further rounds ran the same
agent strategy against the **DeepSeek API** (`deepseek-flash`). They are kept
apart from the frozen local results deliberately: different execution model,
different samples, and — for v3 — a different scoring policy.

**Why three rounds.** v1 established a baseline on a fresh stratified sample.
v2 added a fixed-context arm, comparing the full agent strategy with reading a
single static context block. v3 asked whether clarifying *when* the `mixed`
verdict applies changes how the agent handles `maybe` cases. Each round was
reported using its own frozen sample and scoring policy. Pre-run frozen
protocols are archived for v2 and v3; the reviewed archive does not provide a
v1 frozen protocol.

| Round | Sample (pool → drawn) | Scoring policy | Arm A macro-F1 | Same-round rule baseline | Additional arm |
|---|---|---|---|---|---|
| v1† | 500 → 50, no exclusion | legacy | 0.5638 (covers 49 of 50) | 0.4196 (covers 50) | — |
| v2 | 450 → 50, excluding v1 | legacy | 0.6003 | 0.4216 | fixed-context arm B: 0.5388 |
| v3 | 400 → 50, excluding v1 ∪ v2 | v3 | 0.5700 | 0.3344 | prompt-variant arm C: 0.6074 |

† v1 uses the same planned 50-case sample for both arms. Agent A's legacy
macro-F1 covers 49 completed cases (one `errored` case excluded); the rule
baseline covers all 50.

Under **legacy** scoring, `mixed` matches no gold label and failed cases leave
the macro-F1 denominator; under the **v3** policy, `mixed` maps to `maybe` and
all 50 attempted cases enter the metric, a failed case contributing one false
negative to its own gold class. The agent arm scored above that round's rule
baseline in each of the three rounds. Those are **three separate per-round
observations**: the samples do not overlap, and the policy differs in v3, so
the figures are neither comparable nor a repeated verification of one result.

The v3 joint criterion was **not met**: Δmacro-F1 = 0.037372796459921864 against
a pre-declared 0.04 threshold, so `criteria_met = false`. The second condition
did hold (arm C was correct on 2 of 5 `maybe` cases; arm A on 1 of 5). A single
50-case run has no reported confidence interval and establishes no general validity.
Nothing here isolates the tool loop: the v2 arms differ in evidence selection as
well as loop presence, and the v3 arms differ only in prompt material.

Costs are **estimates** at recorded peak cache-miss rates, not bills; the v1
round includes one case whose usage was never fully reported.

Sample quotas, exclusion sets, seeds, per-class scoring and known gaps:
[experiment log](docs/experiment_log.md). Machine-readable aggregates:
[public summary](reports/agent_experiments_summary_v1.json). Claim-level
provenance: [claims registry](docs/claims_registry.md).

## Install and run the core CLI demo

Python 3.10–3.13 is supported and exercised by CI.

```bash
python -m pip install .
bioevidence demo \
  --question "Do mitochondria participate in programmed cell death?"
```

This CLI demo uses a packaged three-document synthetic fixture. It downloads no
models, reads no gold labels, and prints the top PMID, exact snippet,
document/snippet hashes, and all typed-tool calls. It deliberately does not
invent a medical verdict.

For development:

```bash
python -m pip install -r requirements/ci.lock
python -m pip install --no-deps -e .
ruff check src apps tests scripts
pytest -q
python -m build
```

## Reproduce the heavy public evaluation

The full 500-case runs are intentionally excluded from push CI because they
download two transformer models and execute 10,000 cross-encoder pairs per
retrieval condition. They are available as a manual GitHub Actions workflow
and as local commands.

```bash
python -m pip install -r requirements/heavy.lock
python -m pip install --no-deps -e .

bioevidence prepare --output-dir data/pubmedqa/v1

bioevidence vector-index-build \
  --benchmark-dir data/pubmedqa/v1 \
  --index-dir indexes/title_assisted_e5_v1

bioevidence evaluate \
  --benchmark-dir data/pubmedqa/v1 \
  --index-dir indexes/title_assisted_e5_v1 \
  --responses runs/title_assisted/responses.jsonl \
  --output-json runs/title_assisted/report.json \
  --output-markdown runs/title_assisted/report.md

bioevidence abstract-index-build \
  --benchmark-dir data/pubmedqa/v1 \
  --index-dir indexes/abstract_only_e5_v1

bioevidence abstract-stress \
  --benchmark-dir data/pubmedqa/v1 \
  --index-dir indexes/abstract_only_e5_v1 \
  --rankings runs/abstract_only/rankings.jsonl \
  --ranking-manifest runs/abstract_only/manifest.json \
  --title-assisted-report runs/title_assisted/report.json \
  --output-json runs/abstract_only/report.json \
  --output-markdown runs/abstract_only/report.md

bioevidence evidence-controls \
  --benchmark-dir data/pubmedqa/v1 \
  --run-dir runs/evidence_controls \
  --output-json runs/evidence_controls.json \
  --output-markdown runs/evidence_controls.md
```

Preparation downloads two files from PubMedQA commit
`1cbae8e92f72f20c8d3747cbb3bf5bc53554d997` and rejects bytes that do not
match the pre-registered SHA-256 values. Both model revisions, E5 pooling and
prefixes, RRF parameters, reranker candidate count, fold seed, and output
contracts are fixed in code and manifests.

## What this repository contains

- a minimal runnable import closure under `src/bioevidence`;
- deterministic download and source-hash validation;
- BM25, E5, RRF, reranker, answer baseline, typed tools, and citation checks;
- the abstract-only stress-test runner and the five-control CV runner;
- a repository-run local web workflow for fixed-PMID and live PubMed evidence
  cards;
- a standalone synthetic regression harness with eight declared-metadata scope
  controls covering population, species, intervention, endpoint, timepoint,
  context-only, and unknown scope;
- lightweight tests and a model-free smoke fixture;
- frozen aggregate JSON/Markdown reports and representative failures;
- exact CI/heavy dependency locks, GitHub Actions, and release metadata.

See:

- [architecture](docs/architecture.md)
- [methodology](docs/methodology.md)
- [limitations and claim boundaries](docs/limitations.md)
- [AI Product Case Study](docs/product_case/PORTFOLIO_CASE_STUDY.md)
- [synthetic scope-control report](reports/product_scope_controls_v1.md)
- [experiment log](docs/experiment_log.md) — frozen conditions for the three
  API evaluation rounds and their archival gaps: samples, exclusion sets,
  seeds, scoring policies
- [public summary (JSON)](reports/agent_experiments_summary_v1.json) —
  machine-readable aggregate for the same rounds
- [claims registry](docs/claims_registry.md) — every published claim with its
  strength grade and source field
- [interview notes for the engineering and product stages](docs/resume_and_interview.md)
  — covers the local-CPU retrieval benchmark, the negative controls and the
  pilot decision; predates the three API evaluation rounds
- [interview notes for the API evaluation rounds](docs/interview_prep.md) —
  covers v1/v2/v3 only; does not replace or restate the document above
- [v0.5.0 release manifest](reports/release_manifest_v0.5.0.json) — local
  candidate; its verification block transcribes only checks actually run
- [historical v0.4.1 release manifest](reports/release_manifest_v0.4.1.json)
- [historical v0.4.0 release manifest](reports/release_manifest_v0.4.0.json)

## Formative pilot status

Stopping the study was a product decision, and it followed a measurement
failure — not a result to be softened. One target-user formative pilot was run
to test the research protocol. Both task rows had conflicting or incomplete
measurement records, so the number of evaluable tasks is **0**. Because more
participants cannot repair an unstable measurement protocol, recruitment was
stopped rather than extended. The pilot is retained as a process lesson—not as
a success metric. This repository therefore makes no claim about user value,
time saved, task completion, trust, reuse intent, or VEPS improvement.

The pilot exposed mode, timing, record-consistency, and evidence-scope risks.
Rather than recruiting more participants against an unstable measurement
protocol, v0.4 closed the portfolio study and converted the reusable failure
patterns into deterministic synthetic scope controls. v0.4.1 only repairs
integrity contracts found by post-release audit. Those controls test software
behavior against declared metadata; they are not real-paper scientific
validation or biomedical NLI accuracy.

## What this is not

- not a clinical decision-support or medical-advice system;
- not a systematic review or open-world literature search;
- not a private blind benchmark;
- not evidence of performance on new diseases, papers, or real-world queries;
- not a semantic hallucination-rate estimate;
- not proof that exact citations entail every generated claim.
- not evidence of validated user value or workflow efficiency.
- not a combined result across the three API evaluation rounds: their samples
  differ and v3 uses a different scoring policy, so no cross-round total is
  claimed.
- not a statistical confirmation of any hypothesis tested in those rounds; a
  single 50-case run per round was reported without a confidence interval or
  significance test.
- not an attribution of performance to the tool loop, which these arms do not
  isolate.

The 500 official test labels are public. Structural metrics such as exact-span
and source-hash integrity establish provenance, not semantic correctness.
Model scores are not evidence strength.

## Data, model, and cost boundaries

- Dataset: public expert-labeled PubMedQA PQA-L, 1,000 records.
- Split: official 500 test IDs; the remaining 500 are used for training/CV.
- Corpus: frozen 1,000-document closed corpus.
- Monetary model/API cost for the **local-CPU retrieval and answering runs above**:
  USD 0.00; all inference was local CPU. This applies to the frozen
  title-assisted, abstract-only and evidence-control runs only. It does **not**
  cover the three later API evaluation rounds, which used a paid hosted model
  and whose costs are reported separately as estimates.
- Full v0.2 end-to-end latency: median 1.109 s, p95 1.235 s/query (local CPU,
  same scope as the line above).
- Abstract-only reranked retrieval latency: median 1.109 s, p95 1.252
  s/query, excluding model load and one-time indexing (local CPU).

## License

Project code is MIT-licensed. Dataset and model notices are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); raw PubMedQA files, model
weights, caches, and embeddings are not redistributed.
