from __future__ import annotations

import hashlib
import json

from scripts.generate_product_example import CuratedExampleClient, main


DECISIVE_HCQ_SNIPPET = (
    "Death within 28 days occurred in 421 patients (27.0%) in the "
    "hydroxychloroquine group and in 790 (25.0%) in the usual-care group."
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_public_example_is_direction_grounded_and_byte_deterministic(tmp_path) -> None:
    assert main(["--output-dir", str(tmp_path)]) == 0
    json_path = tmp_path / "public_evidence_pack_example.json"
    markdown_path = tmp_path / "public_evidence_pack_example.md"
    first_json = json_path.read_bytes()
    first_markdown = markdown_path.read_bytes()

    pack = json.loads(first_json)
    claimed_hash = pack.pop("pack_sha256")
    assert _canonical_sha256(pack) == claimed_hash
    assert f"Pack SHA-256: `{claimed_hash}`" in first_markdown.decode("utf-8")

    hcq = next(
        item for item in pack["evidence"] if item["card"]["pmid"] == "33031652"
    )
    snippet = hcq["card"]["snippet"]
    assert hcq["user_decision"]["user_direction"] == "opposes"
    assert snippet["text"] == DECISIVE_HCQ_SNIPPET
    assert hashlib.sha256(DECISIVE_HCQ_SNIPPET.encode()).hexdigest() == snippet[
        "snippet_sha256"
    ]
    source_abstract = CuratedExampleClient().fetch(["33031652"])[0].abstract
    assert source_abstract[snippet["start_char"] : snippet["end_char"]] == snippet[
        "text"
    ]

    assert main(["--output-dir", str(tmp_path)]) == 0
    assert json_path.read_bytes() == first_json
    assert markdown_path.read_bytes() == first_markdown
