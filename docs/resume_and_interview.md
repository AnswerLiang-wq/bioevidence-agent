# Resume and interview notes

## Defensible resume bullets

- Built a reproducible biomedical evidence Agent combining Okapi BM25,
  multilingual-E5, reciprocal-rank fusion, a cross-encoder reranker, typed
  local tools, and exact-span/hash-bound PubMed citations.
- Evaluated 500 public expert-labeled PubMedQA test questions; reported
  retrieval, answer, grounding, latency, cost, and failure metrics with frozen
  data/model identities and immutable artifacts.
- Designed an abstract-only leakage stress test: reranked Recall@10 decreased
  from 1.000 in the title-assisted condition to 0.990, with 12 rank drops and
  five top-10 misses.
- Implemented five fixed out-of-fold evidence-utilization controls on 500
  non-test cases; found context-sensitive but inconclusive behavior
  (correct-vs-shuffled +5.0 accuracy points, exact McNemar p=0.066).
- Packaged the project as a Python 3.10/3.12/3.13-tested v0.3.0 release with
  exact dependency locks, CI, a model-free demo, reproducible heavy workflows,
  and explicit scientific claim boundaries.

## 60-second project explanation

“I built BioEvidence Agent to separate three capabilities that are often
collapsed in RAG demos: locating a source, interpreting its evidence, and
proving where an answer came from. The first benchmark initially showed
perfect retrieval, but PubMedQA questions are title-derived and the title was
indexed. I therefore froze a title-free abstract-only test; the reranked
system stayed strong at 0.990 Recall@10 but exposed 12 rank drops. I also ran
five out-of-fold negative controls. Correct context beat shuffled context by
five accuracy points, but p was 0.066 and macro-F1 did not improve over
question-only, so I report context sensitivity rather than claiming robust
reasoning. Every citation includes PMID, exact offsets, and source/snippet
hashes, and all experiments are reproducible through a clean package and CI.” 

## Likely interview questions

### Why is perfect title-assisted retrieval not the headline?

Because the question is the article title or a close derivative and the same
title is in the indexed document. That measures a favorable identity match,
not general literature retrieval.

### Why use a shuffled-context control?

Question-only can exploit lexical and label priors. Comparing the same trained
combined model on correct versus deterministically wrong contexts tests
whether matching evidence changes predictions beneficially.

### Does p=0.066 mean context is useless?

No. The observed direction is compatible with context sensitivity, but it is
not conventionally strong evidence. The correct statement preserves both the
effect size and uncertainty.

### What is the main bottleneck?

The fixed linear answerer. Retrieval is strong even without titles, whereas
answer macro-F1 is 0.383 on the public test and the minority `maybe` class is
rarely recognized.

### What makes the Agent auditable?

Pinned data/model revisions, source and artifact hashes, typed bounded tools,
complete tool traces, exact abstract offsets, snippet hashes, retrieval
lineage, immutable baseline IDs, and explicit reporting limits.
