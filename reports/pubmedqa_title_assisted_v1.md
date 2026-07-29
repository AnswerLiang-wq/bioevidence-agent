# PubMedQA title-assisted public benchmark

- Cases: 500
- Corpus: 1,000 documents
- Scope: public closed-corpus benchmark; not blind.

## Retrieval

All four systems reached Recall@5/10, MRR, and nDCG@10 of **1.000**. The questions are titles or title-derived and indexed documents contain those titles, so this is a favorable source-localization condition.

## Answer baseline

- Accuracy: **0.558** (279/500)
- Macro-F1: **0.383**
- yes F1: 0.675
- no F1: 0.412
- maybe F1: 0.063

## Provenance and operations

- Schema, target PMID, exact span, source hash, and claim coverage: 500/500 each.
- Local CPU latency: median 1.109 s, p95 1.235 s/query.
- Model/API cost: USD 0.00.

These structural checks are not semantic citation precision or a hallucination rate.
