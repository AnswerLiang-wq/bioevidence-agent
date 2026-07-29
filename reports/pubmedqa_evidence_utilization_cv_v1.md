# PubMedQA evidence-utilization negative controls

- Baseline: `bioevidence-pubmedqa-evidence-utilization-cv-v1`
- Data: 500 official non-test records only.
- CV: stratified 5-fold, seed `20260729`.
- Every metric is out-of-fold; the official test set is not read.

## Results

| System | Accuracy | Macro-F1 | yes F1 | no F1 | maybe F1 |
|---|---:|---:|---:|---:|---:|
| majority | 0.5520 | 0.2371 | 0.7113 | 0.0000 | 0.0000 |
| question_only | 0.5120 | 0.3815 | 0.6353 | 0.4308 | 0.0784 |
| context_only | 0.5340 | 0.3643 | 0.6572 | 0.3721 | 0.0635 |
| question_correct_context | 0.5360 | 0.3557 | 0.6646 | 0.3709 | 0.0317 |
| question_shuffled_context | 0.4860 | 0.3018 | 0.6283 | 0.2770 | 0.0000 |

## What the controls say

- **does context provide measurable gain:** Versus question-only, accuracy changes by +0.024 and macro-F1 by -0.026; paired accuracy p=0.363. The gain is mixed rather than robust.
- **does shuffling context clearly reduce results:** Correct-context minus shuffled-context accuracy is +0.050; the paired exact McNemar p-value is 0.066. This pattern is consistent with context sensitivity but is not strong evidence of reliable evidence use.
- **is current answerer significantly better than majority:** No on paired accuracy in this fixed development diagnostic: correct-context accuracy is 0.536 versus majority 0.552, with exact McNemar p=0.526. Its macro-F1 is higher because majority never predicts no/maybe.
- **is primary bottleneck retrieval or evidence interpretation:** The fixed linear answerer is context-sensitive but the shuffle contrast is not conventionally significant (p=0.066); it remains weak on the minority maybe class. Label imbalance and evidence interpretation are the main bottlenecks. Retrieval is not the dominant bottleneck in this closed corpus.
- **why is maybe weak:** Both class imbalance and evidence-use limitations contribute: maybe has only 55 cases, and correct-context maybe F1 is 0.032, a -0.047 change versus question-only.

The correct-context and shuffled-context validation predictions use the same fold model. Shuffling is a deterministic within-fold derangement, so no case receives its own abstract.

These controls diagnose this fixed public training set and model recipe. They do not establish blind generalization, causal reasoning, or clinical validity.
