# Third-party notices

This repository contains integration code and small synthetic demo fixtures.
It does not redistribute model weights or the raw PubMedQA dataset.

## PubMedQA

- Project: <https://github.com/pubmedqa/pubmedqa>
- Frozen source commit used by the reports:
  `1cbae8e92f72f20c8d3747cbb3bf5bc53554d997`
- License: MIT

The heavy evaluation command downloads the two official files from that
commit and verifies their pre-registered SHA-256 values before use.

## multilingual-E5-small

- Model: <https://huggingface.co/intfloat/multilingual-e5-small>
- Frozen revision:
  `fd1525a9fd15316a2d503bf26ab031a61d056e98`
- License identified by the model card: MIT

Model weights are downloaded by the user through Hugging Face and are not
included in this repository.

## MMARCO cross-encoder

- Model:
  <https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1>
- Frozen revision:
  `1427fd652930e4ba29e8149678df786c240d8825`
- License identified by the model card: Apache-2.0

Model weights are downloaded by the user through Hugging Face and are not
included in this repository.

## Python libraries

The implementation uses NumPy, scikit-learn, PyTorch, Transformers, Pytest,
Ruff, and their transitive dependencies under their respective licenses. Exact
versions used by CI and the heavyweight workflow are listed in
`requirements/`.
