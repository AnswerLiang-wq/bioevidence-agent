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

## What they do not establish

- open-world PubMed search completeness;
- performance on new diseases, papers, time periods, or user populations;
- private blind or independent generalization performance;
- semantic entailment of every generated sentence;
- an independently confirmed semantic hallucination rate;
- clinical efficacy, safety, diagnosis, or treatment recommendation quality.

PubMedQA questions are title-derived and public labels are available in the
dataset ecosystem. The abstract omits article conclusions in this benchmark.
The corpus is purposive and fixed, not a probability sample of future
biomedical questions.

## Negative-control interpretation

Correct context was 5.0 accuracy points better than shuffled context, but the
paired exact McNemar p-value was 0.066. This is compatible with context
sensitivity but is not strong evidence that the model reliably interprets
evidence. Correct context also did not improve macro-F1 over question-only and
remained very weak on `maybe`.

## Safety

The tool should be used as an auditable research-engineering demonstration,
not for patient care. Model confidence is not evidence strength. Users should
verify the source article and seek qualified clinical guidance for medical
decisions.
