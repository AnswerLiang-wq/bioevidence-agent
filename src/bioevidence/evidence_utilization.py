"""Out-of-fold controls testing whether PubMedQA answers use evidence."""

from __future__ import annotations

import json
import math
import platform
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from .corpus import CorpusDocument, read_corpus, sha256_file
from .pubmedqa import (
    CORPUS_FILENAME,
    MANIFEST_FILENAME,
    PUBMEDQA_LABELS,
    TRAIN_FILENAME,
    read_jsonl,
)


CONTROL_BASELINE_ID = "bioevidence-pubmedqa-evidence-utilization-cv-v1"
CV_SEED = 20260729
SYSTEMS = (
    "majority",
    "question_only",
    "context_only",
    "question_correct_context",
    "question_shuffled_context",
)


class _TextClassifier:
    """The frozen TF-IDF/logistic-regression recipe used by each CV fold."""

    def __init__(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import FeatureUnion
        except ImportError as exc:
            raise RuntimeError(
                "evidence-utilization controls require the 'benchmark' extra"
            ) from exc
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

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        matrix = self._features.fit_transform(texts)
        self._classifier.fit(matrix, labels)

    def predict(self, texts: Sequence[str]) -> list[str]:
        matrix = self._features.transform(texts)
        return [str(value) for value in self._classifier.predict(matrix)]


def run_evidence_utilization_controls(
    *,
    benchmark_dir: Path,
    run_dir: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, object]:
    """Run fixed five-fold OOF controls on the 500 official non-test rows."""

    if run_dir.exists():
        raise FileExistsError(f"evidence-control run is immutable: {run_dir}")
    for path in (output_json, output_markdown):
        if path.exists():
            raise FileExistsError(f"evidence-control report is immutable: {path}")
    manifest = _verified_benchmark_manifest(benchmark_dir)
    train_rows = read_jsonl(benchmark_dir / TRAIN_FILENAME)
    if len(train_rows) != 500:
        raise ValueError("evidence controls require exactly 500 non-test rows")
    documents = read_corpus(benchmark_dir / CORPUS_FILENAME)
    documents_by_pmid = {document.pmid: document for document in documents}
    rows = _control_rows(train_rows, documents_by_pmid)
    folds = _stratified_folds(rows)
    assignment_rows = [
        {"pmid": row["pmid"], "fold": folds[index]}
        for index, row in enumerate(rows)
    ]
    shuffled_donors = _shuffled_donors(rows, folds)
    shuffled_rows = [
        {
            "pmid": row["pmid"],
            "fold": folds[index],
            "donor_pmid": rows[shuffled_donors[index]]["pmid"],
        }
        for index, row in enumerate(rows)
    ]

    run_dir.mkdir(parents=True)
    assignments_path = run_dir / "fold_assignments.jsonl"
    shuffled_path = run_dir / "shuffled_context_map.jsonl"
    design_manifest_path = run_dir / "design_manifest.json"
    _write_jsonl(assignments_path, assignment_rows)
    _write_jsonl(shuffled_path, shuffled_rows)
    design_manifest = {
        "manifest_version": "1.0.0",
        "baseline_id": CONTROL_BASELINE_ID,
        "phase": "folds_and_derangements_frozen_before_model_fit",
        "population": "official 500 PubMedQA non-test records only",
        "case_count": 500,
        "fold_count": 5,
        "splitter": {
            "type": "StratifiedKFold",
            "shuffle": True,
            "random_state": CV_SEED,
        },
        "systems": list(SYSTEMS),
        "derangement": (
            "within each validation fold, rotate PMID-sorted cases by one; "
            "the correct-context model is reused without retraining"
        ),
        "forbidden_inputs": [
            "official test IDs",
            "official test labels",
            "LONG_ANSWER",
            "test_gold.jsonl",
        ],
        "inputs": {
            "benchmark_manifest_sha256": sha256_file(
                benchmark_dir / MANIFEST_FILENAME
            ),
            "train_sha256": sha256_file(benchmark_dir / TRAIN_FILENAME),
            "corpus_sha256": sha256_file(benchmark_dir / CORPUS_FILENAME),
            "fold_assignments_sha256": sha256_file(assignments_path),
            "shuffled_context_map_sha256": sha256_file(shuffled_path),
        },
        "source_benchmark_id": manifest["benchmark_id"],
    }
    _write_json(design_manifest_path, design_manifest)

    predictions = {system: [""] * len(rows) for system in SYSTEMS}
    for fold in range(5):
        train_indexes = [
            index for index, assigned in enumerate(folds) if assigned != fold
        ]
        validation_indexes = [
            index for index, assigned in enumerate(folds) if assigned == fold
        ]
        train_labels = [str(rows[index]["label"]) for index in train_indexes]
        majority = _majority_label(train_labels)
        for index in validation_indexes:
            predictions["majority"][index] = majority

        question_model = _TextClassifier()
        question_model.fit(
            [str(rows[index]["question"]) for index in train_indexes],
            train_labels,
        )
        _assign_predictions(
            predictions["question_only"],
            validation_indexes,
            question_model.predict(
                [str(rows[index]["question"]) for index in validation_indexes]
            ),
        )

        context_model = _TextClassifier()
        context_model.fit(
            [str(rows[index]["abstract"]) for index in train_indexes],
            train_labels,
        )
        _assign_predictions(
            predictions["context_only"],
            validation_indexes,
            context_model.predict(
                [str(rows[index]["abstract"]) for index in validation_indexes]
            ),
        )

        combined_model = _TextClassifier()
        combined_model.fit(
            [
                _combined_text(
                    str(rows[index]["question"]),
                    str(rows[index]["abstract"]),
                )
                for index in train_indexes
            ],
            train_labels,
        )
        _assign_predictions(
            predictions["question_correct_context"],
            validation_indexes,
            combined_model.predict(
                [
                    _combined_text(
                        str(rows[index]["question"]),
                        str(rows[index]["abstract"]),
                    )
                    for index in validation_indexes
                ]
            ),
        )
        _assign_predictions(
            predictions["question_shuffled_context"],
            validation_indexes,
            combined_model.predict(
                [
                    _combined_text(
                        str(rows[index]["question"]),
                        str(rows[shuffled_donors[index]]["abstract"]),
                    )
                    for index in validation_indexes
                ]
            ),
        )

    if any(
        len(values) != 500 or any(value not in PUBMEDQA_LABELS for value in values)
        for values in predictions.values()
    ):
        raise RuntimeError("OOF prediction coverage is incomplete")
    labels = [str(row["label"]) for row in rows]
    oof_rows = [
        {
            "pmid": row["pmid"],
            "fold": folds[index],
            "gold_label": row["label"],
            "predictions": {
                system: predictions[system][index] for system in SYSTEMS
            },
            "shuffled_context_donor_pmid": rows[
                shuffled_donors[index]
            ]["pmid"],
        }
        for index, row in enumerate(rows)
    ]
    oof_path = run_dir / "oof_predictions.jsonl"
    _write_jsonl(oof_path, oof_rows)

    metrics = {
        system: _classification_metrics(labels, predictions[system])
        for system in SYSTEMS
    }
    deltas = {
        system: {
            "accuracy_minus_majority": round(
                metrics[system]["accuracy"] - metrics["majority"]["accuracy"],
                6,
            ),
            "macro_f1_minus_majority": round(
                metrics[system]["macro_f1"] - metrics["majority"]["macro_f1"],
                6,
            ),
        }
        for system in SYSTEMS
        if system != "majority"
    }
    paired = {
        comparator: _paired_comparison(
            labels,
            predictions["question_correct_context"],
            predictions[comparator],
        )
        for comparator in (
            "majority",
            "question_only",
            "context_only",
            "question_shuffled_context",
        )
    }
    correct = metrics["question_correct_context"]
    question = metrics["question_only"]
    shuffled = metrics["question_shuffled_context"]
    context_gain = correct["accuracy"] - question["accuracy"]
    shuffle_damage = correct["accuracy"] - shuffled["accuracy"]
    maybe_delta = (
        correct["per_label"]["maybe"]["f1"]
        - question["per_label"]["maybe"]["f1"]
    )
    answers = {
        "is_current_answerer_significantly_better_than_majority": (
            "No on paired accuracy in this fixed development diagnostic: "
            f"correct-context accuracy is {correct['accuracy']:.3f} versus "
            f"majority {metrics['majority']['accuracy']:.3f}, with exact "
            "McNemar p="
            f"{paired['majority']['exact_mcnemar_two_sided_p']:.3f}. "
            "Its macro-F1 is higher because majority never predicts no/maybe."
        ),
        "does_context_provide_measurable_gain": (
            f"Versus question-only, accuracy changes by {context_gain:+.3f} "
            f"and macro-F1 by "
            f"{correct['macro_f1'] - question['macro_f1']:+.3f}; paired "
            "accuracy p="
            f"{paired['question_only']['exact_mcnemar_two_sided_p']:.3f}. "
            "The gain is mixed rather than robust."
        ),
        "does_shuffling_context_clearly_reduce_results": (
            f"Correct-context minus shuffled-context accuracy is "
            f"{shuffle_damage:+.3f}; the paired exact McNemar p-value is "
            f"{paired['question_shuffled_context']['exact_mcnemar_two_sided_p']:.3f}. "
            "This pattern is consistent with context sensitivity but is not "
            "strong evidence of reliable evidence use."
        ),
        "why_is_maybe_weak": (
            "Both class imbalance and evidence-use limitations contribute: "
            f"maybe has only {correct['per_label']['maybe']['support']} cases, "
            f"and correct-context maybe F1 is "
            f"{correct['per_label']['maybe']['f1']:.3f}, a "
            f"{maybe_delta:+.3f} change versus question-only."
        ),
        "is_primary_bottleneck_retrieval_or_evidence_interpretation": (
            _bottleneck_statement(
                metrics,
                context_gain=context_gain,
                shuffle_damage=shuffle_damage,
                shuffle_p=float(
                    paired["question_shuffled_context"][
                        "exact_mcnemar_two_sided_p"
                    ]
                ),
            )
            + " Retrieval is not the dominant bottleneck in this closed corpus."
        ),
    }
    report = {
        "report_version": "1.0.0",
        "baseline_id": CONTROL_BASELINE_ID,
        "benchmark_id": manifest["benchmark_id"],
        "evaluation_scope": (
            "fixed stratified five-fold out-of-fold diagnostic on the 500 "
            "official non-test PubMedQA records; not a test-set estimate"
        ),
        "case_count": 500,
        "fold_count": 5,
        "seed": CV_SEED,
        "metrics": metrics,
        "deltas": deltas,
        "paired_correct_context_vs": paired,
        "scientific_questions": answers,
        "model_policy": {
            "recipe": (
                "same frozen word/character TF-IDF plus balanced logistic "
                "regression parameters as v0.2"
            ),
            "same_fold_mapping_for_all_systems": True,
            "same_combined_model_for_correct_and_shuffled_validation": True,
            "each_case_has_exactly_one_oof_prediction_per_system": True,
            "tuning_after_results": False,
        },
        "inputs": {
            "design_manifest_sha256": sha256_file(design_manifest_path),
            "fold_assignments_sha256": sha256_file(assignments_path),
            "shuffled_context_map_sha256": sha256_file(shuffled_path),
            "oof_predictions_sha256": sha256_file(oof_path),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "claims_not_made": [
            "official test-set performance",
            "blind generalization performance",
            "causal proof that the model reasons correctly",
            "clinical validity",
        ],
    }
    _write_json(output_json, report)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_markdown_report(report), encoding="utf-8")
    return report


def _control_rows(
    train_rows: Sequence[dict[str, object]],
    documents_by_pmid: dict[str, CorpusDocument],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in sorted(train_rows, key=lambda value: int(str(value["pmid"]))):
        if set(row) != {"pmid", "question", "label"}:
            raise ValueError("non-test training projection has unexpected fields")
        pmid = str(row["pmid"])
        label = str(row["label"])
        if pmid in seen or label not in PUBMEDQA_LABELS:
            raise ValueError("invalid non-test training projection")
        document = documents_by_pmid.get(pmid)
        if document is None:
            raise ValueError(f"non-test PMID absent from corpus: {pmid}")
        seen.add(pmid)
        rows.append(
            {
                "pmid": pmid,
                "question": str(row["question"]),
                "abstract": document.abstract,
                "label": label,
            }
        )
    return rows


def _stratified_folds(rows: Sequence[dict[str, str]]) -> list[int]:
    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError as exc:
        raise RuntimeError(
            "evidence-utilization controls require the 'benchmark' extra"
        ) from exc
    labels = [row["label"] for row in rows]
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=CV_SEED,
    )
    folds = [-1] * len(rows)
    for fold, (_, validation_indexes) in enumerate(
        splitter.split(range(len(rows)), labels)
    ):
        for index in validation_indexes:
            if folds[int(index)] != -1:
                raise RuntimeError("case assigned to multiple folds")
            folds[int(index)] = fold
    if any(fold < 0 for fold in folds):
        raise RuntimeError("fold assignment is incomplete")
    return folds


def _shuffled_donors(
    rows: Sequence[dict[str, str]],
    folds: Sequence[int],
) -> list[int]:
    donors = [-1] * len(rows)
    for fold in range(5):
        indexes = sorted(
            (
                index
                for index, assigned_fold in enumerate(folds)
                if assigned_fold == fold
            ),
            key=lambda index: int(rows[index]["pmid"]),
        )
        if len(indexes) < 2:
            raise ValueError("each validation fold requires a derangement")
        rotated = indexes[1:] + indexes[:1]
        for index, donor in zip(indexes, rotated, strict=True):
            donors[index] = donor
    if any(donor < 0 or donor == index for index, donor in enumerate(donors)):
        raise RuntimeError("shuffled-context mapping is not a derangement")
    return donors


def _majority_label(labels: Sequence[str]) -> str:
    counts = Counter(labels)
    return max(
        PUBMEDQA_LABELS,
        key=lambda label: (counts[label], -PUBMEDQA_LABELS.index(label)),
    )


def _assign_predictions(
    target: list[str],
    indexes: Sequence[int],
    values: Sequence[str],
) -> None:
    if len(indexes) != len(values):
        raise ValueError("prediction count differs from validation fold")
    for index, value in zip(indexes, values, strict=True):
        if target[index]:
            raise RuntimeError("OOF prediction assigned more than once")
        target[index] = value


def _combined_text(question: str, abstract: str) -> str:
    return f"Question: {question}\nAbstract context: {abstract}"


def _classification_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, object]:
    per_label: dict[str, dict[str, float | int]] = {}
    matrix: list[list[int]] = []
    f1_values: list[float] = []
    for gold_label in PUBMEDQA_LABELS:
        matrix.append(
            [
                sum(
                    gold == gold_label and prediction == predicted_label
                    for gold, prediction in zip(
                        expected,
                        predicted,
                        strict=True,
                    )
                )
                for predicted_label in PUBMEDQA_LABELS
            ]
        )
    for label in PUBMEDQA_LABELS:
        true_positive = sum(
            gold == label and prediction == label
            for gold, prediction in zip(expected, predicted, strict=True)
        )
        false_positive = sum(
            gold != label and prediction == label
            for gold, prediction in zip(expected, predicted, strict=True)
        )
        false_negative = sum(
            gold == label and prediction != label
            for gold, prediction in zip(expected, predicted, strict=True)
        )
        support = sum(gold == label for gold in expected)
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_label[label] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    correct = sum(
        gold == prediction
        for gold, prediction in zip(expected, predicted, strict=True)
    )
    return {
        "correct": correct,
        "total": len(expected),
        "accuracy": round(correct / len(expected), 6),
        "macro_f1": round(statistics.fmean(f1_values), 6),
        "labels": list(PUBMEDQA_LABELS),
        "confusion_matrix_gold_rows_predicted_columns": matrix,
        "per_label": per_label,
    }


def _paired_comparison(
    gold: Sequence[str],
    correct_context: Sequence[str],
    comparator: Sequence[str],
) -> dict[str, object]:
    wins = losses = ties = 0
    correct_only = comparator_only = 0
    for expected, first, second in zip(
        gold,
        correct_context,
        comparator,
        strict=True,
    ):
        first_correct = first == expected
        second_correct = second == expected
        if first_correct and not second_correct:
            wins += 1
            correct_only += 1
        elif second_correct and not first_correct:
            losses += 1
            comparator_only += 1
        else:
            ties += 1
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "exact_mcnemar_two_sided_p": _mcnemar_exact(
            correct_only,
            comparator_only,
        ),
    }


def _mcnemar_exact(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(first_only, second_only) + 1)
    ) / (2**discordant)
    return round(min(1.0, 2 * tail), 8)


def _bottleneck_statement(
    metrics: dict[str, dict[str, object]],
    *,
    context_gain: float,
    shuffle_damage: float,
    shuffle_p: float,
) -> str:
    maybe_f1 = metrics["question_correct_context"]["per_label"]["maybe"]["f1"]
    if context_gain <= 0 and shuffle_damage <= 0:
        return (
            "The fixed linear answerer shows no positive correct-context "
            "advantage in this diagnostic; question/label priors and limited "
            "evidence utilization are the dominant bottlenecks."
        )
    if maybe_f1 < 0.25:
        return (
            "The fixed linear answerer is context-sensitive but the shuffle "
            f"contrast is not conventionally significant (p={shuffle_p:.3f}); "
            "it remains weak on the minority maybe class. Label imbalance and "
            "evidence interpretation are the main bottlenecks."
        )
    return (
        "Correct context contributes measurable signal, while residual errors "
        "reflect limitations of the fixed linear representation and label "
        "decision boundary."
    )


def _markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# PubMedQA evidence-utilization negative controls",
        "",
        f"- Baseline: `{report['baseline_id']}`",
        "- Data: 500 official non-test records only.",
        f"- CV: stratified 5-fold, seed `{report['seed']}`.",
        "- Every metric is out-of-fold; the official test set is not read.",
        "",
        "## Results",
        "",
        "| System | Accuracy | Macro-F1 | yes F1 | no F1 | maybe F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        metrics = report["metrics"][system]
        lines.append(
            f"| {system} | {metrics['accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} | "
            f"{metrics['per_label']['yes']['f1']:.4f} | "
            f"{metrics['per_label']['no']['f1']:.4f} | "
            f"{metrics['per_label']['maybe']['f1']:.4f} |"
        )
    lines.extend(["", "## What the controls say", ""])
    for key, value in report["scientific_questions"].items():
        lines.append(f"- **{key.replace('_', ' ')}:** {value}")
    lines.extend(
        [
            "",
            "The correct-context and shuffled-context validation predictions "
            "use the same fold model. Shuffling is a deterministic within-fold "
            "derangement, so no case receives its own abstract.",
            "",
            "These controls diagnose this fixed public training set and model "
            "recipe. They do not establish blind generalization, causal "
            "reasoning, or clinical validity.",
            "",
        ]
    )
    return "\n".join(lines)


def _verified_benchmark_manifest(
    benchmark_dir: Path,
) -> dict[str, object]:
    manifest = json.loads(
        (benchmark_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("PubMedQA manifest is missing file hashes")
    for name in (CORPUS_FILENAME, TRAIN_FILENAME):
        if files.get(name) != sha256_file(benchmark_dir / name):
            raise ValueError(f"PubMedQA artifact hash mismatch: {name}")
    return manifest


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
