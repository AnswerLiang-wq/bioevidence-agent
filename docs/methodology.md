# Methodology

## Frozen public benchmark

The source is PubMedQA PQA-L at repository commit
`1cbae8e92f72f20c8d3747cbb3bf5bc53554d997`.

- `ori_pqal.json` SHA-256:
  `8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d`
- `test_ground_truth.json` SHA-256:
  `939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51`
- official test: 500 records;
- non-test train/CV: 500 records;
- corpus: all 1,000 abstract contexts;
- excluded from model inputs: `LONG_ANSWER`, `final_decision`, annotator
  predictions, and target PMID.

Public labels map `yes → supported`, `no → contradicted`, and
`maybe → insufficient`.

## Retrieval

- BM25: `k1=1.5`, `b=0.75`.
- Dense: `intfloat/multilingual-e5-small` at
  `fd1525a9fd15316a2d503bf26ab031a61d056e98`, attention-mask mean pooling,
  L2 normalization, `query:` / `passage:` prefixes, max length 512.
- Fusion: equal-weight reciprocal-rank fusion, `k=60`.
- Reranker:
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` at
  `1427fd652930e4ba29e8149678df786c240d8825`, top 20 candidates.
- Metrics: Recall@5, Recall@10, MRR over the complete 1,000-document ranking,
  and nDCG@10 for one relevant target.

The title-assisted and abstract-only conditions use the same corpus order,
questions, models, revisions, RRF, candidate count, batch size, and CPU
device. Only the passage view changes.

## Pre-gold ranking freeze

The abstract-only runner accepts only the 500 `case_id + question` rows.
It writes all four complete rankings and their SHA-256. A separate scorer then
loads the target PMIDs and rejects rankings whose hash changed. This prevents
selective reruns based on target performance.

## Answer baseline

The fixed baseline uses a word 1–2 gram and character 3–5 gram TF-IDF
FeatureUnion with balanced multinomial logistic regression. It is deliberately
small and transparent. No official test error was used to tune v0.2 or v0.3.

## Evidence-utilization controls

All five systems use the same stratified five-fold mapping over the 500
official non-test cases, with seed `20260729`. Every case has exactly one
out-of-fold prediction per system.

1. fold-training-set majority;
2. question only;
3. correct context only;
4. question plus correct context;
5. question plus shuffled context.

For control 5, contexts are deterministically rotated by one within each
PMID-sorted validation fold. No case receives its own abstract. The combined
model is trained once per fold on correct training pairs and reused unchanged
for both correct-context and shuffled-context validation.

Reported metrics include accuracy, macro-F1, per-label precision/recall/F1 and
support, confusion matrices, deltas, paired win/loss/tie counts, and exact
two-sided McNemar tests.

## Reproducibility

CI runs lint, lightweight tests, a wheel build, and a fresh out-of-tree
model-free demo on Python 3.10, 3.12, and 3.13. The transformer-heavy
evaluation is a separate manual workflow so normal pushes do not download
models or execute the full reranker workload.
