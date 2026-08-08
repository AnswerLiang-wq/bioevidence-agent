# Methodology

## Evaluation surfaces

Two evidence bases are reported separately:

1. **Agent Engineering:** deterministic experiments on public PubMedQA with a
   frozen corpus, split, model revisions, and scoring procedures.
2. **Product Case Study:** a repository-run PubMed evidence-card workflow,
   deterministic synthetic scope controls, and one formative pilot whose two
   task rows were not evaluable.

The Product Demo's fixed-PMID/ESearch + EFetch + BM25 path must not be described
as the BM25 + E5 + RRF + cross-encoder benchmark pipeline. Conversely, the
closed-corpus benchmark does not establish live PubMed retrieval quality or
user value.

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

## Product Demo retrieval and evidence packs

The local web workflow has two candidate sources:

- **Standard task:** use a pre-checked fixed PMID list, then fetch current
  PubMed records with EFetch.
- **Live research:** send a public, non-sensitive question to PubMed ESearch,
  take at most 15 returned PMIDs, then fetch them with EFetch.

Records without abstracts are removed and only exact normalized-title
duplicates are deduplicated. Local BM25 ranks the remaining abstracts and at
most five evidence cards are shown. Each card carries a canonical PubMed URL,
PMID, publication metadata, an exact abstract span, normalized-record hash,
snippet hash, and bounded tool trace. The user—not the system—accepts or
excludes a card and records its direction.

This method measures no open-world recall and does not use the benchmark
answerer to label real PubMed claims. ESearch order, PubMed availability, and
returned record bytes are external and may change.

## Synthetic scope controls

Eight versioned fixtures declare claim and evidence metadata in advance. They
cover a direct match plus population, species, intervention, endpoint,
timepoint, context-only, and unknown-scope cases. A control passes when the
software produces the pre-declared flags and only the direct match remains a
candidate for decisive human review.

The report is reproducible with:

```bash
python scripts/run_product_scope_controls.py
```

With no output flags, the command deterministically rewrites the frozen JSON
and Markdown reports under `reports/`. The frozen result is
[8/8 controls passed](../reports/product_scope_controls_v1.md).
Because the inputs contain declared metadata, this is an invariant test—not
automatic field extraction, real-paper scope classification, or biomedical
natural-language-inference accuracy.

## Formative pilot disposition

One real target-user formative pilot exercised one manual task and one Agent
task. The audit found conflicting or incomplete timing, mode, and task records;
both rows were therefore marked not evaluable. The valid quantitative status
is **1 pilot and 0 evaluable tasks**.

No missing value was repaired into a success and no second participant cohort
was added. The pilot informs product risks and the synthetic control design,
but it is excluded from product-effect estimates. Consequently there is no
reported user-value, time-savings, completion-rate, trust, reuse-intent, or
VEPS result. See the
[AI Product Case Study](product_case/PORTFOLIO_CASE_STUDY.md) for the decision
to stop the exploratory study and close the portfolio scope.

## Reproducibility

CI runs lint, lightweight tests, a wheel build, and a fresh out-of-tree
model-free demo on Python 3.10, 3.11, 3.12, and 3.13. The transformer-heavy
evaluation is a separate manual workflow so normal pushes do not download
models or execute the full reranker workload.

The wheel intentionally packages the core CLI only. Product Demo verification
runs from the repository because its `apps/product_demo/` UI and
`scripts/run_product_demo.py` launcher are not wheel entry points.
