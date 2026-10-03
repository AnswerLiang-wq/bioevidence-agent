"""Generate a deterministic, source-linked Product Demo export example.

The release artifact uses short, manually verified PubMed abstract excerpts so
it can be regenerated without network access or redistributing full abstracts.
It demonstrates the export contract, not a fresh PubMed fetch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from apps.product_demo.service import (  # noqa: E402
    ProductDemoService,
    render_markdown_pack,
)
from bioevidence.pubmed import PubMedArticle  # noqa: E402

ACCEPTED_PMIDS = {"33031652", "33264556"}
FIXED_CREATED_AT_UTC = "2026-08-06T00:00:00+00:00"


class CuratedExampleClient:
    """Offline client containing four short excerpts from public PubMed pages."""

    api_key_present = False

    def __init__(self) -> None:
        rows = (
            PubMedArticle(
                pmid="33031652",
                title="Effect of Hydroxychloroquine in Hospitalized Patients with Covid-19",
                doi="10.1056/NEJMoa2022926",
                publication_types=("Randomized Controlled Trial",),
                journal="The New England Journal of Medicine",
                year=2020,
                first_author="RECOVERY Collaborative Group",
                abstract=(
                    "Death within 28 days occurred in 421 patients (27.0%) in the "
                    "hydroxychloroquine group and in 790 (25.0%) in the usual-care "
                    "group."
                ),
            ),
            PubMedArticle(
                pmid="33264556",
                title=(
                    "Repurposed Antiviral Drugs for Covid-19 - Interim WHO "
                    "Solidarity Trial Results"
                ),
                doi="10.1056/NEJMoa2023184",
                publication_types=("Randomized Controlled Trial",),
                journal="The New England Journal of Medicine",
                year=2021,
                first_author="WHO Solidarity Trial Consortium",
                abstract=(
                    "No drug definitely reduced mortality, overall or in any subgroup, "
                    "or reduced initiation of ventilation or hospitalization duration."
                ),
            ),
            PubMedArticle(
                pmid="33165621",
                title=(
                    "Effect of Hydroxychloroquine on Clinical Status at 14 Days in "
                    "Hospitalized Patients With COVID-19"
                ),
                doi="10.1001/jama.2020.22240",
                publication_types=("Randomized Controlled Trial",),
                journal="JAMA",
                year=2020,
                first_author="Self WH",
                abstract=(
                    "The primary outcome was clinical status 14 days after "
                    "randomization."
                ),
            ),
            PubMedArticle(
                pmid="32706953",
                title=(
                    "Hydroxychloroquine with or without Azithromycin in "
                    "Mild-to-Moderate Covid-19"
                ),
                doi="10.1056/NEJMoa2019014",
                publication_types=("Randomized Controlled Trial",),
                journal="The New England Journal of Medicine",
                year=2020,
                first_author="Cavalcanti AB",
                abstract="The primary outcome was clinical status at 15 days.",
            ),
        )
        self._articles = {row.pmid: row for row in rows}

    def fetch(self, pmids: list[str]) -> list[PubMedArticle]:
        return [self._articles[pmid] for pmid in pmids if pmid in self._articles]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples")
    args = parser.parse_args(argv)

    client = CuratedExampleClient()
    service = ProductDemoService(
        client=client,
        standard_tasks=[
            {
                "id": "standard-hydroxychloroquine-mortality",
                "title": "Hydroxychloroquine and mortality",
                "description": "Deterministic public export example.",
                "question": (
                    "Hydroxychloroquine lowers 28-day mortality in adults "
                    "hospitalized with COVID-19."
                ),
                "pmids": list(client._articles),
            }
        ],
    )
    result = service.search(
        session_id="public-example-hcq-0001",
        mode="standard",
        task_id="standard-hydroxychloroquine-mortality",
    )
    decisions = []
    for card in result["cards"]:
        accepted = card["pmid"] in ACCEPTED_PMIDS
        decisions.append(
            {
                "card_id": card["card_id"],
                "action": "accepted" if accepted else "excluded",
                "user_direction": "opposes" if accepted else "unclear",
                "useful": accepted,
                "note": (
                    "Public example: mortality evidence retained for human review."
                    if accepted
                    else "Public example: non-decisive scope for the exact 28-day claim."
                ),
            }
        )
    exported = service.export_pack(
        session_id="public-example-hcq-0001",
        decisions=decisions,
        pack_status="opposes",
        synthesis=(
            "Illustrative user judgment: the retained randomized-trial abstracts do "
            "not show lower mortality with hydroxychloroquine. Full-text review and "
            "broader searching remain necessary."
        ),
    )
    pack = exported["json"]
    pack["created_at_utc"] = FIXED_CREATED_AT_UTC
    pack["example_context"] = {
        "artifact_type": "deterministic_abridged_public_example",
        "live_pubmed_fetch_performed": False,
        "record_hash_scope": (
            "Hashes bind the abridged local example records, not current PubMed bytes."
        ),
        "scientific_judgment_owner": "illustrative_human_decision",
        "source_note": (
            "Titles, identifiers, and short excerpts were checked against the linked "
            "public PubMed records; run the web demo to fetch current records."
        ),
    }
    pack.pop("pack_sha256", None)
    pack["pack_sha256"] = _canonical_sha256(pack)
    markdown = render_markdown_pack(pack)
    notice = (
        "> **Example boundary:** deterministic abridged public example; no live "
        "PubMed fetch was performed. Hashes bind these local example records.\n\n"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "public_evidence_pack_example.json"
    markdown_path = args.output_dir / "public_evidence_pack_example.md"
    json_path.write_text(
        json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(notice + markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted_count": pack["audit"]["accepted_count"],
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
