"""Deterministic stratified sampling for a PubMedQA evaluation slice.

The slice is frozen before any model call: the same gold file, the same seed and
the same rule must always yield the same case IDs, so a later run can be shown
to have evaluated exactly the sample that was declared.  Only gold *labels* are
used here to balance the strata; they never enter a model prompt.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .pubmedqa import PUBMEDQA_LABELS

DEFAULT_SEED = 20260729
DEFAULT_TOTAL = 50

#: How ties on the fractional remainder are broken.  Sorting by label keeps the
#: apportionment independent of dict ordering, and "larger remainder first, then
#: label ascending" is the whole rule — there is no hidden preference.
REMAINDER_TIE_BREAK = "larger remainder first; ties broken by label ascending"

#: Within a stratum the cases are sorted by case_id and then drawn with
#: random.Random(seed).sample, so the draw does not depend on file order.
WITHIN_STRATUM_RULE = (
    "cases sorted by case_id ascending, then random.Random(seed).sample without "
    "replacement; strata processed in PUBMEDQA_LABELS order on the same RNG instance"
)


def largest_remainder_quotas(counts: Mapping[str, int], total: int) -> dict[str, Any]:
    """Apportion ``total`` seats across strata by the largest remainder method.

    Returns the per-stratum final quota plus the intermediate arithmetic, so the
    apportionment can be checked by hand rather than trusted.
    """
    population = sum(counts.values())
    if population <= 0:
        raise ValueError("population is empty; cannot apportion seats")
    if total <= 0:
        raise ValueError("total must be positive")
    if total > population:
        raise ValueError(f"cannot draw {total} cases from a population of {population}")

    rows: list[dict[str, Any]] = []
    for label in PUBMEDQA_LABELS:
        count = counts.get(label, 0)
        exact = total * count / population
        floor = int(exact)
        rows.append({
            "label": label,
            "population": count,
            "exact_quota": round(exact, 6),
            "floor": floor,
            "remainder": round(exact - floor, 6),
            "quota": floor,
        })

    seats_left = total - sum(row["floor"] for row in rows)
    ranked = sorted(rows, key=lambda row: (-row["remainder"], row["label"]))
    for row in ranked[:seats_left]:
        row["quota"] += 1

    return {
        "method": "largest remainder (Hare quota)",
        "total": total,
        "population": population,
        "remainder_tie_break": REMAINDER_TIE_BREAK,
        "strata": rows,
    }


def frozen_sample(
    gold_rows: Sequence[Mapping[str, Any]],
    *,
    total: int = DEFAULT_TOTAL,
    seed: int = DEFAULT_SEED,
    exclude_case_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Draw the stratified slice and return it with its full provenance record.

    ``exclude_case_ids`` removes cases from the pool before apportionment, so a
    follow-up sample is drawn from the remaining distribution rather than from
    the whole gold file.  Excluded cases are recorded, not silently dropped.
    """
    excluded = {str(case_id) for case_id in exclude_case_ids}
    by_label: dict[str, list[str]] = {label: [] for label in PUBMEDQA_LABELS}
    label_of: dict[str, str] = {}
    seen_excluded: set[str] = set()
    for row in gold_rows:
        case_id = str(row["case_id"])
        label = str(row["label"])
        if label not in by_label:
            raise ValueError(f"unexpected gold label {label!r} for case {case_id}")
        if case_id in label_of:
            raise ValueError(f"duplicate case_id {case_id!r} in gold file")
        label_of[case_id] = label
        if case_id in excluded:
            seen_excluded.add(case_id)
            continue
        by_label[label].append(case_id)

    unknown_exclusions = sorted(excluded - seen_excluded)
    if unknown_exclusions:
        raise ValueError(
            f"excluded case IDs are not present in the gold file: {unknown_exclusions[:5]}"
        )

    counts = {label: len(ids) for label, ids in by_label.items()}
    apportionment = largest_remainder_quotas(counts, total)

    rng = random.Random(seed)
    picked: list[str] = []
    for stratum in apportionment["strata"]:
        label = stratum["label"]
        pool = sorted(by_label[label])
        chosen = rng.sample(pool, stratum["quota"])
        stratum["selected"] = sorted(chosen)
        picked.extend(chosen)

    return {
        "sample_version": "1.0.0",
        "seed": seed,
        "total": total,
        "labels": list(PUBMEDQA_LABELS),
        "within_stratum_rule": WITHIN_STRATUM_RULE,
        "apportionment": apportionment,
        "case_ids": sorted(picked),
        "label_by_case": {case_id: label_of[case_id] for case_id in sorted(picked)},
        "excluded_case_ids": sorted(excluded),
        "n_excluded": len(excluded),
        "pool_size": sum(counts.values()),
        "pool_label_counts": counts,
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_gold(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_frozen_sample_record(
    gold_path: Path,
    *,
    total: int = DEFAULT_TOTAL,
    seed: int = DEFAULT_SEED,
    exclude_case_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Read the gold file and produce the record that gets written to disk."""
    gold_rows = read_gold(gold_path)
    sample = frozen_sample(
        gold_rows, total=total, seed=seed, exclude_case_ids=exclude_case_ids
    )
    sample["gold_file"] = str(gold_path.name)
    sample["gold_sha256"] = sha256_path(gold_path)
    sample["gold_rows"] = len(gold_rows)
    sample["gold_verdicts"] = sorted({str(row.get("verdict")) for row in gold_rows})
    sample["gold_used_for_model_input"] = False
    return sample


def select_cases(
    cases: Iterable[Mapping[str, Any]], case_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Return the cases named by ``case_ids``, in the order the sample declares."""
    by_id = {str(case["case_id"]): dict(case) for case in cases}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise KeyError(f"sample references case IDs missing from the gold file: {missing[:5]}")
    return [by_id[case_id] for case_id in case_ids]
