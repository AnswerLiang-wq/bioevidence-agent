# Limitations and claim boundaries

## What the measurements establish

- Retrieval metrics describe source localization in a fixed public
  1,000-document corpus.
- The abstract-only contrast measures sensitivity to removal of indexed
  titles.
- Answer accuracy and F1 describe a fixed public PubMedQA label task.
- Exact spans and SHA-256 checks establish structural provenance and byte
  integrity.
- Evidence controls describe out-of-fold behavior on the fixed 500 non-test
  records.
- Product scope controls establish eight deterministic behaviors over
  synthetic, pre-declared metadata.
- Product Demo tests establish structural contracts for fixed-PMID and live
  branches using deterministic clients; they do not measure current PubMed
  availability or live retrieval quality.

## What they do not establish

- open-world PubMed search completeness;
- performance on new diseases, papers, time periods, or user populations;
- private blind or independent generalization performance;
- semantic entailment of every generated sentence;
- an independently confirmed semantic hallucination rate;
- clinical efficacy, safety, diagnosis, or treatment recommendation quality.
- automatic extraction of population, species, intervention, endpoint, or
  timepoint from real literature;
- validated product usefulness, time savings, completion, trust, reuse intent,
  or VEPS improvement.

PubMedQA questions are title-derived and public labels are available in the
dataset ecosystem. The abstract omits article conclusions in this benchmark.
The corpus is purposive and fixed, not a probability sample of future
biomedical questions.

The Product Demo uses a different path from the engineering benchmark. Its
standard mode fetches fixed PMIDs and ranks their abstracts with BM25; its live
mode uses PubMed ESearch/EFetch followed by BM25. It does not run E5, RRF, or
the cross-encoder, and the benchmark's strong closed-corpus retrieval metrics
must not be attributed to the web workflow. Live PubMed recall and ranking
quality have not been measured.

## Negative-control interpretation

Correct context was 5.0 accuracy points better than shuffled context, but the
paired exact McNemar p-value was 0.066. This is compatible with context
sensitivity but is not strong evidence that the model reliably interprets
evidence. Correct context also did not improve macro-F1 over question-only and
remained very weak on `maybe`.

## Synthetic controls and formative pilot

The eight [scope controls](../reports/product_scope_controls_v1.md) use fields
declared by their fixtures. Passing them proves that the code honors those
fields; it does not show that the system can infer scientific scope from an
abstract or judge a real paper correctly.

One target-user formative pilot was completed, but both task records had
measurement conflicts and were excluded from evaluation. The accurate count
is **1 pilot, 0 evaluable tasks**. It is evidence of protocol and product risks,
not evidence of user value. No completion rate, time comparison, trust score,
reuse-intent score, or VEPS effect is reported from that pilot.

The decision to stop additional recruitment limits empirical product claims,
but avoids converting incomplete records into favorable evidence. v0.4 is a
portfolio closeout, not a validated usability study or production launch.

## Packaging boundary

The distributable wheel contains the core CLI. The local web application and
launcher are repository assets and must be run from a clone. Installing the
wheel alone does not install or expose a web-demo command.

## Safety

The tool should be used as an auditable research-engineering demonstration,
not for patient care. Model confidence is not evidence strength. Users should
verify the source article and seek qualified clinical guidance for medical
decisions.
