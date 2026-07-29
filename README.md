# BioEvidence Agent

BioEvidence Agent is a reproducible biomedical evidence-retrieval and
structured-answer research project. It combines local BM25, pinned
multilingual-E5 embeddings, equal-weight reciprocal-rank fusion, a pinned
cross-encoder reranker, typed search/fetch/inspect tools, and byte-verifiable
citation lineage.

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

## Install and run the offline demo

Python 3.10+ is supported.

```bash
python -m pip install .
bioevidence demo \
  --question "Do mitochondria participate in programmed cell death?"
```

The demo uses a packaged three-document synthetic fixture. It downloads no
models, reads no gold labels, and prints the top PMID, exact snippet,
document/snippet hashes, and all typed-tool calls. It deliberately does not
invent a medical verdict.

For development:

```bash
python -m pip install -r requirements/ci.lock
python -m pip install --no-deps -e .
ruff check src tests
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
- lightweight tests and a model-free smoke fixture;
- frozen aggregate JSON/Markdown reports and representative failures;
- exact CI/heavy dependency locks, GitHub Actions, and release metadata.

See:

- [architecture](docs/architecture.md)
- [methodology](docs/methodology.md)
- [limitations and claim boundaries](docs/limitations.md)
- [resume and interview notes](docs/resume_and_interview.md)
- [release manifest](reports/release_manifest_v0.3.0.json)

## What this is not

- not a clinical decision-support or medical-advice system;
- not a systematic review or open-world literature search;
- not a private blind benchmark;
- not evidence of performance on new diseases, papers, or real-world queries;
- not a semantic hallucination-rate estimate;
- not proof that exact citations entail every generated claim.

The 500 official test labels are public. Structural metrics such as exact-span
and source-hash integrity establish provenance, not semantic correctness.
Model scores are not evidence strength.

## Data, model, and cost boundaries

- Dataset: public expert-labeled PubMedQA PQA-L, 1,000 records.
- Split: official 500 test IDs; the remaining 500 are used for training/CV.
- Corpus: frozen 1,000-document closed corpus.
- Monetary model/API cost: USD 0.00; all inference was local CPU.
- Full v0.2 end-to-end latency: median 1.109 s, p95 1.235 s/query.
- Abstract-only reranked retrieval latency: median 1.109 s, p95 1.252
  s/query, excluding model load and one-time indexing.

## License

Project code is MIT-licensed. Dataset and model notices are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); raw PubMedQA files, model
weights, caches, and embeddings are not redistributed.
