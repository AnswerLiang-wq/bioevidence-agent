# BioEvidence Product Demo

This local web application turns a concrete biomedical research claim into a
small, source-bound abstract evidence pack. It reuses the repository's PubMed
client, BM25 retriever, typed evidence tools, exact character spans, and
normalized-record hashes.

It is a research workflow prototype. It is not a clinical decision tool, a
systematic review, or an autonomous scientific judge.

## Clean local start

From a fresh clone with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python scripts/run_product_demo.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The server refuses
non-local bind addresses.

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
  --event-dir /tmp/bioevidence-product-events
```

## Verify

```bash
python -m pip install -r requirements/ci.lock
python -m pip install --no-deps -e .
ruff check src apps tests scripts
pytest -q
```

The current internal audit is documented in
[`docs/product_case/INTERNAL_TEST_REPORT.md`](../../docs/product_case/INTERNAL_TEST_REPORT.md).
No internal test is counted as a real participant result.
