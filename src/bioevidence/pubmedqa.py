"""Pinned PubMedQA preparation and a fixed public-benchmark answer baseline."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .corpus import CorpusDocument, sha256_file, write_corpus
from .pubmed import PubMedArticle


PUBMEDQA_REPOSITORY = "https://github.com/pubmedqa/pubmedqa"
PUBMEDQA_COMMIT = "1cbae8e92f72f20c8d3747cbb3bf5bc53554d997"
PUBMEDQA_LICENSE = "MIT"
PUBMEDQA_RAW_NAME = "ori_pqal.json"
PUBMEDQA_TEST_GOLD_NAME = "test_ground_truth.json"
PUBMEDQA_RAW_SHA256 = (
    "8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d"
)
PUBMEDQA_TEST_GOLD_SHA256 = (
    "939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51"
)
PUBMEDQA_RAW_URL = (
    f"https://raw.githubusercontent.com/pubmedqa/pubmedqa/{PUBMEDQA_COMMIT}/"
    f"data/{PUBMEDQA_RAW_NAME}"
)
PUBMEDQA_TEST_GOLD_URL = (
    f"https://raw.githubusercontent.com/pubmedqa/pubmedqa/{PUBMEDQA_COMMIT}/"
    f"data/{PUBMEDQA_TEST_GOLD_NAME}"
)
PUBMEDQA_LABELS = ("yes", "no", "maybe")
PUBMEDQA_VERDICT_MAP = {
    "yes": "supported",
    "no": "contradicted",
    "maybe": "insufficient",
}

CORPUS_FILENAME = "corpus.jsonl"
TRAIN_FILENAME = "train.jsonl"
TEST_INPUTS_FILENAME = "test_inputs.jsonl"
TEST_GOLD_FILENAME = "test_gold.jsonl"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class PubMedQARecord:
    pmid: str
    question: str
    contexts: tuple[str, ...]
    section_labels: tuple[str, ...]
    year: int | None
    label: str

    @property
    def abstract(self) -> str:
        return " ".join(
            f"{section}: {context}"
            for section, context in zip(
                self.section_labels,
                self.contexts,
                strict=True,
            )
        )


def prepare_pubmedqa(
    output_dir: Path,
    *,
    raw_path: Path | None = None,
    test_ground_truth_path: Path | None = None,
) -> dict[str, object]:
    """Build immutable corpus/train/test projections from the pinned source."""

    if output_dir.exists():
        raise FileExistsError(
            f"PubMedQA output is immutable; choose a new directory: {output_dir}"
        )
    raw_bytes = _read_or_download(
        raw_path,
        url=PUBMEDQA_RAW_URL,
        expected_sha256=PUBMEDQA_RAW_SHA256,
    )
    test_bytes = _read_or_download(
        test_ground_truth_path,
        url=PUBMEDQA_TEST_GOLD_URL,
        expected_sha256=PUBMEDQA_TEST_GOLD_SHA256,
    )
    records, test_labels = load_pubmedqa(raw_bytes, test_bytes)
    test_ids = set(test_labels)
    train_records = sorted(
        (record for record in records if record.pmid not in test_ids),
        key=lambda item: int(item.pmid),
    )
    test_records = sorted(
        (record for record in records if record.pmid in test_ids),
        key=lambda item: int(item.pmid),
    )
    documents = [_as_corpus_document(record) for record in records]

    output_dir.mkdir(parents=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    (raw_dir / PUBMEDQA_RAW_NAME).write_bytes(raw_bytes)
    (raw_dir / PUBMEDQA_TEST_GOLD_NAME).write_bytes(test_bytes)
    corpus_path = output_dir / CORPUS_FILENAME
    train_path = output_dir / TRAIN_FILENAME
    test_inputs_path = output_dir / TEST_INPUTS_FILENAME
    test_gold_path = output_dir / TEST_GOLD_FILENAME
    write_corpus(corpus_path, documents)
    _write_jsonl(
        train_path,
        [
            {
                "pmid": record.pmid,
                "question": record.question,
                "label": record.label,
            }
            for record in train_records
        ],
    )
    case_ids = {
        record.pmid: f"PUBMEDQA-TEST-{index:04d}"
        for index, record in enumerate(test_records, start=1)
    }
    _write_jsonl(
        test_inputs_path,
        [
            {
                "case_id": case_ids[record.pmid],
                "question": record.question,
            }
            for record in test_records
        ],
    )
    _write_jsonl(
        test_gold_path,
        [
            {
                "case_id": case_ids[record.pmid],
                "pmid": record.pmid,
                "label": record.label,
                "verdict": PUBMEDQA_VERDICT_MAP[record.label],
            }
            for record in test_records
        ],
    )
    manifest = {
        "manifest_version": "1.0.0",
        "benchmark_id": "pubmedqa-pqal-public-closed-v1",
        "evaluation_scope": "public closed-corpus benchmark; not blind",
        "source": {
            "repository": PUBMEDQA_REPOSITORY,
            "commit": PUBMEDQA_COMMIT,
            "license": PUBMEDQA_LICENSE,
            "raw_url": PUBMEDQA_RAW_URL,
            "raw_sha256": PUBMEDQA_RAW_SHA256,
            "test_ground_truth_url": PUBMEDQA_TEST_GOLD_URL,
            "test_ground_truth_sha256": PUBMEDQA_TEST_GOLD_SHA256,
        },
        "counts": {
            "all": len(records),
            "train": len(train_records),
            "test": len(test_records),
            "test_labels": _label_counts(test_records),
        },
        "label_to_verdict": PUBMEDQA_VERDICT_MAP,
        "runner_excludes": [
            "LONG_ANSWER",
            "final_decision",
            "reasoning_required_pred",
            "reasoning_free_pred",
            "gold PMID",
        ],
        "files": {
            CORPUS_FILENAME: sha256_file(corpus_path),
            TRAIN_FILENAME: sha256_file(train_path),
            TEST_INPUTS_FILENAME: sha256_file(test_inputs_path),
            TEST_GOLD_FILENAME: sha256_file(test_gold_path),
        },
        "limitations": [
            (
                "Questions are article titles or title-derived and the corpus "
                "contains those titles, so retrieval is a favorable source-"
                "localization task."
            ),
            (
                "The official test labels are public; results are public "
                "benchmark performance, not a private blind estimate."
            ),
            (
                "PubMedQA contexts omit the conclusion and do not establish "
                "open-world literature-search completeness or clinical validity."
            ),
        ],
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_pubmedqa(
    raw_bytes: bytes,
    test_ground_truth_bytes: bytes,
) -> tuple[list[PubMedQARecord], dict[str, str]]:
    """Validate the pinned PQA-L payload and official 500-case test mapping."""

    _require_sha256(raw_bytes, PUBMEDQA_RAW_SHA256, PUBMEDQA_RAW_NAME)
    _require_sha256(
        test_ground_truth_bytes,
        PUBMEDQA_TEST_GOLD_SHA256,
        PUBMEDQA_TEST_GOLD_NAME,
    )
    raw = json.loads(raw_bytes)
    test_labels = json.loads(test_ground_truth_bytes)
    if not isinstance(raw, dict) or len(raw) != 1000:
        raise ValueError("PubMedQA PQA-L must contain exactly 1000 records")
    if not isinstance(test_labels, dict) or len(test_labels) != 500:
        raise ValueError("PubMedQA official test mapping must contain 500 records")
    records = [_parse_record(pmid, value) for pmid, value in raw.items()]
    by_pmid = {record.pmid: record for record in records}
    if set(test_labels) - set(by_pmid):
        raise ValueError("PubMedQA test mapping contains an unknown PMID")
    for pmid, label in test_labels.items():
        if label not in PUBMEDQA_LABELS:
            raise ValueError(f"PubMedQA test PMID {pmid} has an invalid label")
        if by_pmid[pmid].label != label:
            raise ValueError(f"PubMedQA test label mismatch for PMID {pmid}")
    return sorted(records, key=lambda item: int(item.pmid)), dict(test_labels)


class TfidfLogisticAnswerer:
    """Fixed, transparent answer baseline trained only on public train rows."""

    model_id = "tfidf-word-char-logistic-regression"
    baseline_id = "bioevidence-pubmedqa-tfidf-logreg-v1"

    def __init__(self) -> None:
        self._features: object | None = None
        self._classifier: object | None = None

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "baseline_id": self.baseline_id,
            "word_tfidf": {
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_df": 0.98,
                "max_features": 60000,
                "sublinear_tf": True,
            },
            "character_tfidf": {
                "analyzer": "char_wb",
                "ngram_range": [3, 5],
                "min_df": 3,
                "max_features": 40000,
                "sublinear_tf": True,
            },
            "classifier": {
                "type": "logistic_regression",
                "solver": "lbfgs",
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": 0,
            },
            "training_policy": (
                "fit on the 500 records outside official PubMedQA test IDs; "
                "no LONG_ANSWER or official-test labels used as model input"
            ),
        }

    def fit(
        self,
        rows: Sequence[dict[str, object]],
        documents_by_pmid: dict[str, CorpusDocument],
    ) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import FeatureUnion
        except ImportError as exc:
            raise RuntimeError(
                "PubMedQA answer evaluation requires the 'benchmark' extra"
            ) from exc
        texts: list[str] = []
        labels: list[str] = []
        for row in rows:
            pmid = _required_text(row, "pmid")
            question = _required_text(row, "question")
            label = _required_text(row, "label")
            if label not in PUBMEDQA_LABELS:
                raise ValueError(f"invalid PubMedQA train label: {label}")
            document = documents_by_pmid.get(pmid)
            if document is None:
                raise ValueError(f"training PMID is absent from corpus: {pmid}")
            texts.append(_answer_text(question, document.abstract))
            labels.append(label)
        if len(rows) != 500 or len(set(labels)) < 3:
            raise ValueError("PubMedQA answerer requires the fixed 500-row train split")
        self._features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.98,
                        sublinear_tf=True,
                        max_features=60000,
                    ),
                ),
                (
                    "character",
                    TfidfVectorizer(
                        lowercase=True,
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        sublinear_tf=True,
                        max_features=40000,
                    ),
                ),
            ]
        )
        self._classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=0,
        )
        matrix = self._features.fit_transform(texts)
        self._classifier.fit(matrix, labels)

    def predict(
        self,
        question: str,
        abstract: str,
    ) -> tuple[str, dict[str, float]]:
        if self._features is None or self._classifier is None:
            raise RuntimeError("PubMedQA answerer must be fitted before prediction")
        matrix = self._features.transform([_answer_text(question, abstract)])
        label = str(self._classifier.predict(matrix)[0])
        probabilities = self._classifier.predict_proba(matrix)[0]
        scores = {
            str(name): round(float(score), 8)
            for name, score in zip(
                self._classifier.classes_,
                probabilities,
                strict=True,
            )
        }
        return label, scores


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _read_or_download(
    path: Path | None,
    *,
    url: str,
    expected_sha256: str,
) -> bytes:
    if path is not None:
        value = path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            value = response.read()
    _require_sha256(value, expected_sha256, url)
    return value


def _require_sha256(value: bytes, expected: str, label: str) -> None:
    observed = hashlib.sha256(value).hexdigest()
    if observed != expected:
        raise ValueError(
            f"SHA-256 mismatch for {label}: expected {expected}, observed {observed}"
        )


def _parse_record(pmid: object, value: object) -> PubMedQARecord:
    if not isinstance(pmid, str) or not pmid.isdigit() or int(pmid) < 1:
        raise ValueError("PubMedQA keys must be valid PMID strings")
    if not isinstance(value, dict):
        raise ValueError(f"PubMedQA PMID {pmid} must be an object")
    question = _required_text(value, "QUESTION")
    contexts = value.get("CONTEXTS")
    section_labels = value.get("LABELS")
    if (
        not isinstance(contexts, list)
        or not contexts
        or any(not isinstance(item, str) or not item.strip() for item in contexts)
    ):
        raise ValueError(f"PubMedQA PMID {pmid} has invalid contexts")
    if (
        not isinstance(section_labels, list)
        or len(section_labels) != len(contexts)
        or any(
            not isinstance(item, str) or not item.strip()
            for item in section_labels
        )
    ):
        raise ValueError(f"PubMedQA PMID {pmid} has invalid section labels")
    label = _required_text(value, "final_decision")
    if label not in PUBMEDQA_LABELS:
        raise ValueError(f"PubMedQA PMID {pmid} has an invalid label")
    year_value = value.get("YEAR")
    if year_value is None:
        year = None
    elif isinstance(year_value, str) and year_value.isdigit():
        year = int(year_value)
    else:
        raise ValueError(f"PubMedQA PMID {pmid} has an invalid year")
    return PubMedQARecord(
        pmid=pmid,
        question=" ".join(question.split()),
        contexts=tuple(" ".join(item.split()) for item in contexts),
        section_labels=tuple(
            " ".join(item.split()).upper() for item in section_labels
        ),
        year=year,
        label=label,
    )


def _as_corpus_document(record: PubMedQARecord) -> CorpusDocument:
    return CorpusDocument.from_pubmed(
        PubMedArticle(
            pmid=record.pmid,
            title=record.question,
            doi=None,
            publication_types=("PubMedQA public article context",),
            journal="",
            year=record.year,
            first_author=None,
            abstract=record.abstract,
        )
    )


def _label_counts(records: Sequence[PubMedQARecord]) -> dict[str, int]:
    return {
        label: sum(record.label == label for record in records)
        for label in PUBMEDQA_LABELS
    }


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _required_text(value: dict[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return item.strip()


def _answer_text(question: str, abstract: str) -> str:
    return f"Question: {question}\nAbstract context: {abstract}"
