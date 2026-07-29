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

- **does correct context improve over question only:** Accuracy delta +0.024; macro-F1 delta -0.026.
- **does shuffling damage performance:** Correct-context minus shuffled-context accuracy is +0.050; the paired exact McNemar p-value is 0.066. This pattern is consistent with context sensitivity but is not strong evidence of reliable evidence use.
- **does context only carry signal:** Context-only accuracy is 0.534 versus majority 0.552.
- **does correct context help maybe:** The maybe-label F1 delta versus question-only is -0.047.
- **what is the bottleneck:** The fixed linear answerer is context-sensitive but the shuffle contrast is not conventionally significant (p=0.066); it remains weak on the minority maybe class. Label imbalance and evidence interpretation are the main bottlenecks.

The correct-context and shuffled-context validation predictions use the same fold model. Shuffling is a deterministic within-fold derangement, so no case receives its own abstract.

These controls diagnose this fixed public training set and model recipe. They do not establish blind generalization, causal reasoning, or clinical validity.
