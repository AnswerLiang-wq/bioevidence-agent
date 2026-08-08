# BioEvidence Product Demo

This local web application turns a concrete biomedical research claim into a
small, source-bound abstract evidence pack. It reuses the repository's PubMed
client, BM25 retriever, typed evidence tools, exact character spans, and
normalized-record hashes.

It is a research workflow prototype. It is not a clinical decision tool, a
systematic review, or an autonomous scientific judge.

Exports use the versioned `product-evidence-pack-v0.2` contract. Compared with
v0.1, its audit object replaces `all_sources_pubmed` with the narrower
`all_source_urls_pubmed_formatted`; integrations should inspect `pack_version`.

This is the product entry point. It is intentionally different from the
closed-corpus engineering benchmark:

| Surface | Candidate discovery | Local ranking |
|---|---|---|
| Standard task | pre-checked fixed PMIDs, fetched with PubMed EFetch | BM25 |
| Live research | PubMed ESearch (up to 15 PMIDs), then EFetch | BM25 |
| Engineering benchmark | frozen 1,000-document corpus | BM25 + E5 + RRF + cross-encoder reranker |

The web UI never claims that it is running the full hybrid benchmark stack.

## Five-minute local path

From a fresh clone with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python scripts/run_product_demo.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The server refuses
non-local bind addresses.

For the shortest review path:

1. leave **Standard task** selected;
2. choose a task and confirm the non-sensitive research-use statement;
3. generate the cards and open at least one PubMed source;
4. accept or exclude cards, set a human direction, and add a short note;
5. choose an overall status and export JSON or Markdown;
6. verify that the export preserves PMID, source URL, exact abstract span,
   record/snippet SHA-256, the human decisions, and scope limits.

The Python wheel packages the core `bioevidence` CLI, not this web
application. `apps/product_demo/` and `scripts/run_product_demo.py` are
repository assets; run this UI from a cloned repository.

Both task modes require network access to official NCBI E-utilities:

- **Standard task** uses a small, pre-checked, fixed PMID candidate set, then
  fetches the current PubMed records. Candidate identity is stable; external
  availability and returned record bytes are not frozen.
- **Live research** asks PubMed ESearch for up to 15 candidates, fetches
  abstracts, removes only exact normalized-title duplicates, and ranks the
  remaining records with local BM25.

In both modes, system direction is `unclear / needs human review`. The user
accepts or excludes cards, supplies direction and notes, and exports JSON or
Markdown. The system does not invoke the weak benchmark answerer as an
open-world judge.

The standard task offers repeatability, not a frozen local corpus: PubMed
record availability and returned bytes can still change. Live mode is a
bounded discovery aid, not a systematic review; its recall is not measured.

## Local data boundary

By default, privacy-minimized events are written under the Git-ignored path:

```text
apps/product_demo/session_data/
```

The event contract permits task actions and elapsed time. It rejects question,
query, title, snippet, note, name, email, and phone fields. Evidence-pack
exports are downloaded by the browser and should be treated as research data.
Do not use participant-owned, unpublished, patient, or confidential content.

For a temporary event directory:

```bash
python scripts/run_product_demo.py \
  --event-dir apps/product_demo/session_data/temporary
```

## Verify

```bash
python -m pip install -r requirements/ci.lock
python -m pip install --no-deps -e .
ruff check src apps tests scripts
pytest -q
python scripts/run_product_scope_controls.py
```

With no output flags, this command regenerates the frozen JSON and Markdown
reports under `reports/`.

The eight deterministic
[synthetic scope controls](../../reports/product_scope_controls_v1.md) verify
declared population, species, intervention, endpoint, timepoint, context-only,
and unknown-scope rules. They do not evaluate automatic scientific inference
on real papers.

See the [internal engineering audit](../../docs/product_case/INTERNAL_TEST_REPORT.md)
and [AI Product Case Study](../../docs/product_case/PORTFOLIO_CASE_STUDY.md).
One formative target-user pilot was run, but record conflicts made both task
rows non-evaluable: **1 pilot, 0 evaluable tasks**. It supports no claim about
user value, time savings, completion, trust, reuse, or VEPS.
