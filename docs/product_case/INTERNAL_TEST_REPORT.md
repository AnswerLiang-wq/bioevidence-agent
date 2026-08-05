# Product Demo internal test report

Date: 2026-08-05

Version under test: local `v0.4.0-dev` product branch

Participants: none. This is an engineering and internal usability audit, not
user research and not evidence of product value.

## Result summary

- Python tests: 25/25 passed, including the out-of-tree launcher check;
- Python lint: passed for `src`, `apps`, `tests`, and `scripts`;
- desktop browser: standard PubMed task completed and rendered four cards;
- live browser: PubMed ESearch/EFetch completed and rendered five cards;
- responsive check: at a 390 x 844 override the document client width and
  scroll width were both 375 CSS pixels, with no horizontal overflow;
- client JavaScript: parsed and executed in the real browser, including mode
  switching, search, card acceptance, timer, and privacy-event submission;
- independent `node --check`: not run because Node.js is not installed in the
  current shell environment.

The test count above is a deterministic repository result. It is not a user
completion rate, model accuracy, retrieval quality estimate, or production
availability claim.

## Scenario matrix

| Scenario | Method | Observed result | Remaining boundary |
|---|---|---|---|
| Happy path | real local server, browser, official PubMed EFetch | four standard-task cards rendered with PMID links, exact snippets and normalized-record hashes | one task does not establish general retrieval quality |
| Live retrieval | browser, official PubMed ESearch and EFetch | five cards rendered for a public tocilizumab question | results included mixed study designs and still require human screening |
| No results | injected empty PubMed search | typed `no_results` 404 response | actual frequency is unknown |
| No usable abstract | records with empty abstracts | typed `no_abstract_evidence` 422 response | full text is deliberately out of scope |
| Source/fetch failure | injected `PubMedError` | typed 503 response with no internal exception details | external availability is not guaranteed |
| Conflicting evidence | deterministic accepted cards with opposing user labels | `mixed` evidence pack preserved both directions | system does not decide which study is scientifically decisive |
| Insufficient evidence | export with one accepted abstract | `insufficient` pack exported; it does not meet VEPS | exportability must not be confused with task success |
| Citation mismatch/injection | forged card ID and exact-snippet hash checks | forged card rejected; exported cards came only from server-held search results | source identity still requires human audit for real tasks |
| Long or ambiguous question | seven-character and 601-character inputs | both rejected before retrieval | semantic specificity beyond length remains a user responsibility |
| Patient-specific advice | personal medication/symptom request | rejected with a research-only boundary message | pattern guard is a product boundary, not a clinical classifier |
| Privacy logging | API and file inspection | event log contained action/timing metadata but no query, abstract or note text | moderator must still prevent sensitive input at collection time |
| Original Agent regression | full repository test suite and Ruff | passed | v0.3 manifest is intentionally version-bound and not rewritten |

## Evidence-driven internal iteration

The first live run returned two PubMed records with the same normalized title,
representing duplicate language/indexing records. It also made study design
hard to see even though publication types were present in the API response.

The product layer was changed to:

1. remove only exact normalized-title duplicates before local ranking;
2. avoid fuzzy semantic deduplication, which could hide scientifically
   distinct studies;
3. display up to two PubMed publication types on every evidence card.

After the change, the same live question displayed five distinct titles and
made observational, clinical-trial, meta-analysis, and randomized-trial types
visible. This is an internal QA improvement, not the required
real-user-evidence-driven iteration from Stage 7.

## External-source accessibility note

The official NCBI E-utilities endpoints returned the tested PubMed records and
the product generated canonical `https://pubmed.ncbi.nlm.nih.gov/{PMID}/`
links. A separate automated visible-page check encountered an anti-bot/browser
challenge and could not certify end-user page accessibility. Citation-link
accessibility therefore remains a per-session user-study metric; it is not
claimed as 100% here.

## Visual and interaction findings

- The desktop layout maintained a persistent workflow rail, safety boundary,
  evidence-card hierarchy, exact snippet metadata, and visible human-review
  controls.
- The narrow layout stacked the workflow and form controls without horizontal
  overflow.
- All PubMed-derived text is inserted with `textContent`; the product script
  does not use `innerHTML`, remote scripts, or remote styles.
- `record_sha256` binds the normalized local PubMed record; it is not described
  as a hash of the remote PubMed webpage bytes.
- System direction remained `unclear / needs human review` in both standard
  and live modes.

## Current conclusion

Stage 3's minimum evidence-pack workflow is implemented and can proceed to a
single real target-user pilot after G2 recruitment is complete. The Demo has
not yet demonstrated VEPS, time savings, trust, reuse intent, or unassisted
completion for any real participant.
