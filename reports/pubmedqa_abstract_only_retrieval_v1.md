# PubMedQA abstract-only retrieval stress test

- Baseline: `bioevidence-pubmedqa-abstract-only-retrieval-v1`
- Public test cases: 500
- Passage view: abstract only; titles are unavailable to every retriever.

## Results

| System | Recall@5 | Recall@10 | MRR | nDCG@10 | Median ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.9760 | 0.9800 | 0.9532 | 0.9597 | 2.476 | 3.222 |
| vector | 0.9900 | 0.9900 | 0.9792 | 0.9817 | 14.173 | 16.984 |
| hybrid_rrf | 0.9840 | 0.9880 | 0.9777 | 0.9800 | 18.552 | 21.227 |
| hybrid_reranked | 0.9900 | 0.9900 | 0.9820 | 0.9839 | 1108.533 | 1251.850 |

## Interpretation

The v1 title-assisted and v1 abstract-only conditions are different retrieval tasks. The former exposes the title-derived question inside the indexed document; the latter does not.

The final reranked system had 12 cases below rank 1. Poor results are retained, not tuned away.

## Representative rank drops

- `PUBMEDQA-TEST-0189` / PMID `18359123`: rank 774 (miss_at_10).
- `PUBMEDQA-TEST-0061` / PMID `11570976`: rank 209 (miss_at_10).
- `PUBMEDQA-TEST-0216` / PMID `19106867`: rank 55 (miss_at_10).
- `PUBMEDQA-TEST-0241` / PMID `20064872`: rank 45 (miss_at_10).
- `PUBMEDQA-TEST-0376` / PMID `24139705`: rank 27 (miss_at_10).
- `PUBMEDQA-TEST-0208` / PMID `18719011`: rank 4 (rank_drop).
- `PUBMEDQA-TEST-0454` / PMID `26460153`: rank 3 (rank_drop).
- `PUBMEDQA-TEST-0500` / PMID `29112560`: rank 3 (rank_drop).
- `PUBMEDQA-TEST-0100` / PMID `14599616`: rank 2 (rank_drop).
- `PUBMEDQA-TEST-0128` / PMID `15995461`: rank 2 (rank_drop).
- `PUBMEDQA-TEST-0175` / PMID `18041059`: rank 2 (rank_drop).
- `PUBMEDQA-TEST-0452` / PMID `26418441`: rank 2 (rank_drop).

This remains a public, fixed, closed-corpus diagnostic. It is not a blind estimate or an open-world search-completeness claim.
