"""Product workflow adapter over the existing BioEvidence core."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from bioevidence.corpus import CorpusDocument
from bioevidence.pubmed import PubMedArticle, PubMedClient, PubMedError
from bioevidence.retrievers import BM25Retriever
from bioevidence.tools import (
    FetchRecordInput,
    InspectEvidenceInput,
    LiteratureTools,
    SearchLiteratureInput,
    ToolExecutor,
    trace_as_dict,
)


PRODUCT_VERSION = "0.4.0-dev"
MAX_CARDS = 5
LIVE_CANDIDATE_LIMIT = 15
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")
MEDICAL_ADVICE_PATTERNS = (
    re.compile(r"\bwhat should (?:i|we) (?:take|do)\b", re.IGNORECASE),
    re.compile(r"\bdiagnos(?:e|is) (?:me|my)\b", re.IGNORECASE),
    re.compile(r"\bmy (?:symptoms|medication|dose|treatment)\b", re.IGNORECASE),
    re.compile(r"我该(?:吃|用|停)|给我开药|诊断我|我的症状|患者姓名"),
)
USER_ACTIONS = {"accepted", "excluded", "undecided"}
USER_DIRECTIONS = {"supports", "opposes", "unclear"}
PACK_STATUSES = {"supports", "opposes", "mixed", "insufficient"}


class ProductDemoError(RuntimeError):
    """Safe, typed error returned by the local product API."""

    def __init__(self, code: str, message: str, *, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


class ProductDemoService:
    """Build evidence cards and trusted exports without making a verdict."""

    def __init__(
        self,
        *,
        client: PubMedClient | None = None,
        standard_tasks: Sequence[dict[str, object]] | None = None,
    ) -> None:
        self.client = client or PubMedClient()
        task_rows = list(standard_tasks or load_standard_tasks())
        self._tasks = {_task_id(row): row for row in task_rows}
        if len(self._tasks) != len(task_rows):
            raise ValueError("standard task IDs must be unique")
        self._sessions: dict[str, dict[str, object]] = {}

    def standard_tasks(self) -> list[dict[str, object]]:
        return [
            {
                "id": task_id,
                "title": row["title"],
                "question": row["question"],
                "description": row["description"],
                "candidate_count": len(_pmids(row)),
            }
            for task_id, row in sorted(self._tasks.items())
        ]

    def search(
        self,
        *,
        session_id: str,
        mode: str,
        question: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, object]:
        session_id = _session_id(session_id)
        if mode == "standard":
            task = self._tasks.get(task_id or "")
            if task is None:
                raise ProductDemoError("invalid_task", "请选择有效的标准任务。")
            resolved_question = _question(task.get("question"))
            candidate_pmids = _pmids(task)
            resolved_task_id = _task_id(task)
            candidate_source = "fixed_verified_pmids"
        elif mode == "live":
            resolved_question = _question(question)
            _reject_medical_advice(resolved_question)
            resolved_task_id = task_id or "participant-owned"
            candidate_source = "pubmed_esearch_relevance"
            try:
                candidate_pmids = self.client.search(
                    resolved_question,
                    retmax=LIVE_CANDIDATE_LIMIT,
                )
            except PubMedError as exc:
                raise ProductDemoError(
                    "external_service_failure",
                    "PubMed暂时无法完成检索，请保留本次失败记录后再试。",
                    http_status=503,
                ) from exc
        else:
            raise ProductDemoError("invalid_mode", "模式必须是standard或live。")

        if not candidate_pmids:
            raise ProductDemoError("no_results", "没有检索到候选PubMed记录。", http_status=404)
        try:
            articles = self.client.fetch(candidate_pmids)
        except PubMedError as exc:
            raise ProductDemoError(
                "external_service_failure",
                "PubMed候选记录抓取失败，请保留本次失败记录。",
                http_status=503,
            ) from exc
        documents = _documents_with_abstracts(articles)
        if not documents:
            raise ProductDemoError(
                "no_abstract_evidence",
                "候选记录没有可用摘要，不能生成摘要级证据卡。",
                http_status=422,
            )

        executor = ToolExecutor(
            LiteratureTools(documents, retriever=BM25Retriever(documents)),
            max_calls=20,
        )
        search_result = executor.search(
            SearchLiteratureInput(
                query=resolved_question,
                top_k=min(MAX_CARDS, len(documents)),
            )
        )
        cards: list[dict[str, object]] = []
        for hit in search_result.hits:
            record = executor.fetch(FetchRecordInput(pmid=hit.pmid))
            inspected = executor.inspect(
                InspectEvidenceInput(
                    pmid=hit.pmid,
                    query=resolved_question,
                    max_snippets=1,
                )
            )
            if not inspected.snippets:
                inspected = executor.inspect(
                    InspectEvidenceInput(
                        pmid=hit.pmid,
                        query=record.abstract[:300],
                        max_snippets=1,
                    )
                )
            if not inspected.snippets:
                continue
            snippet = inspected.snippets[0]
            cards.append(
                {
                    "card_id": f"E{len(cards) + 1}",
                    "display_rank": len(cards) + 1,
                    "pmid": record.pmid,
                    "doi": record.doi,
                    "title": record.title,
                    "year": record.year,
                    "journal": record.journal,
                    "publication_types": list(record.publication_types),
                    "source_url": record.source_url,
                    "record_sha256": record.document_sha256,
                    "snippet": {
                        "text": snippet.text,
                        "section": snippet.section,
                        "start_char": snippet.start_char,
                        "end_char": snippet.end_char,
                        "snippet_sha256": snippet.snippet_sha256,
                        "relevance_score": snippet.relevance_score,
                    },
                    "retrieval": {
                        "method": hit.retrieval_method,
                        "rank": hit.rank,
                        "score": hit.score,
                        "component_ranks": hit.component_ranks,
                    },
                    "system_suggestion": {
                        "direction": "unclear",
                        "label": "需人工确认",
                        "reason": (
                            "开放任务未使用PubMedQA弱回答器自动裁决；"
                            "请按人群、干预、终点和时间自行判断。"
                        ),
                    },
                    "limits": [
                        "仅展示PubMed摘要片段，不代表全文证据。",
                        "相关性排序和Hash不能证明科学结论正确。",
                    ],
                }
            )
        if not cards:
            raise ProductDemoError(
                "no_citable_snippets",
                "候选摘要中没有可生成精确片段的记录。",
                http_status=422,
            )

        result = {
            "response_version": "product-demo-search-v0.1",
            "product_version": PRODUCT_VERSION,
            "session_id": session_id,
            "task_id": resolved_task_id,
            "mode": mode,
            "question": resolved_question,
            "status": "complete" if len(cards) >= 3 else "partial",
            "retrieval": {
                "candidate_source": candidate_source,
                "candidate_pmid_count": len(candidate_pmids),
                "abstract_document_count": len(documents),
                "displayed_card_count": len(cards),
                "local_ranker": "bm25",
                "api_key_present": self.client.api_key_present,
            },
            "cards": cards,
            "scope_limits": [
                "研究文献信息工具，不提供诊断或治疗建议。",
                "摘要级结果不能替代全文审查或系统综述。",
                "系统方向默认为需人工确认，最终科学判断由用户完成。",
            ],
            "tool_trace": trace_as_dict(executor.trace),
        }
        self._sessions[session_id] = result
        return result

    def export_pack(
        self,
        *,
        session_id: str,
        decisions: object,
        pack_status: str,
        synthesis: str,
    ) -> dict[str, object]:
        session_id = _session_id(session_id)
        search_result = self._sessions.get(session_id)
        if search_result is None:
            raise ProductDemoError("unknown_session", "当前会话没有可导出的检索结果。")
        if pack_status not in PACK_STATUSES:
            raise ProductDemoError("invalid_pack_status", "请选择有效的证据包状态。")
        synthesis = _bounded_text(synthesis, "synthesis", max_length=1500)
        cards_by_id = {
            str(card["card_id"]): card
            for card in search_result["cards"]
            if isinstance(card, dict)
        }
        cleaned_decisions = _decisions(decisions, cards_by_id)
        accepted = [row for row in cleaned_decisions if row["action"] == "accepted"]
        if not accepted:
            raise ProductDemoError("no_accepted_evidence", "至少接受一张证据卡后才能导出。")

        evidence: list[dict[str, object]] = []
        exclusions: list[dict[str, object]] = []
        for decision in cleaned_decisions:
            card = cards_by_id[str(decision["card_id"])]
            row = {
                "card": card,
                "user_decision": decision,
            }
            if decision["action"] == "accepted":
                evidence.append(row)
            elif decision["action"] == "excluded":
                exclusions.append(row)

        pack = {
            "pack_version": "product-evidence-pack-v0.1",
            "product_version": PRODUCT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "task_id": search_result["task_id"],
            "mode": search_result["mode"],
            "question": search_result["question"],
            "pack_status": pack_status,
            "synthesis": synthesis,
            "evidence": evidence,
            "exclusions": exclusions,
            "scope_limits": search_result["scope_limits"],
            "audit": {
                "accepted_count": len(evidence),
                "excluded_count": len(exclusions),
                "useful_count": sum(bool(row["useful"]) for row in accepted),
                "all_sources_pubmed": all(
                    str(row["card"]["source_url"]).startswith(
                        "https://pubmed.ncbi.nlm.nih.gov/"
                    )
                    for row in evidence
                ),
            },
        }
        pack["pack_sha256"] = _sha256_json(pack)
        return {
            "json": pack,
            "markdown": _markdown_pack(pack),
        }


def load_standard_tasks() -> list[dict[str, object]]:
    path = Path(__file__).with_name("standard_tasks.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("standard tasks must be a non-empty list")
    return [row for row in value if isinstance(row, dict)]


def _documents_with_abstracts(articles: Sequence[PubMedArticle]) -> list[CorpusDocument]:
    documents: list[CorpusDocument] = []
    seen: set[str] = set()
    seen_titles: set[str] = set()
    for article in articles:
        normalized_title = re.sub(r"\W+", " ", article.title.casefold()).strip()
        if (
            article.pmid in seen
            or normalized_title in seen_titles
            or not article.abstract.strip()
        ):
            continue
        seen.add(article.pmid)
        seen_titles.add(normalized_title)
        documents.append(CorpusDocument.from_pubmed(article))
    return documents


def _question(value: object) -> str:
    if not isinstance(value, str):
        raise ProductDemoError("invalid_question", "请输入具体的生物医学研究问题。")
    cleaned = " ".join(value.split())
    if not 8 <= len(cleaned) <= 600:
        raise ProductDemoError("invalid_question", "问题长度应为8到600个字符。")
    return cleaned


def _reject_medical_advice(question: str) -> None:
    if any(pattern.search(question) for pattern in MEDICAL_ADVICE_PATTERNS):
        raise ProductDemoError(
            "medical_advice_not_supported",
            "该Demo不处理个人诊断或治疗请求。请改写为不含患者信息的科研证据问题。",
            http_status=422,
        )


def _session_id(value: object) -> str:
    if not isinstance(value, str) or SESSION_ID_RE.fullmatch(value) is None:
        raise ProductDemoError("invalid_session", "会话标识无效。")
    return value


def _task_id(row: dict[str, object]) -> str:
    value = row.get("id")
    if not isinstance(value, str) or re.fullmatch(r"[a-z0-9-]+", value) is None:
        raise ValueError("standard task id is invalid")
    return value


def _pmids(row: dict[str, object]) -> list[str]:
    value = row.get("pmids")
    if not isinstance(value, list) or not 3 <= len(value) <= 20:
        raise ValueError("standard task must contain 3 to 20 PMIDs")
    pmids = [item for item in value if isinstance(item, str) and item.isdigit()]
    if len(pmids) != len(value) or len(set(pmids)) != len(pmids):
        raise ValueError("standard task PMIDs must be unique digit strings")
    return pmids


def _bounded_text(value: object, label: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise ProductDemoError(f"invalid_{label}", f"{label}必须是文本。")
    cleaned = value.strip()
    if len(cleaned) > max_length:
        raise ProductDemoError(f"invalid_{label}", f"{label}内容过长。")
    return cleaned


def _decisions(
    value: object,
    cards_by_id: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ProductDemoError("invalid_decisions", "证据卡决定必须是列表。")
    cleaned: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, dict):
            raise ProductDemoError("invalid_decisions", "证据卡决定格式无效。")
        card_id = row.get("card_id")
        action = row.get("action")
        direction = row.get("user_direction")
        useful = row.get("useful")
        if not isinstance(card_id, str) or card_id not in cards_by_id or card_id in seen:
            raise ProductDemoError("invalid_decisions", "证据卡标识无效或重复。")
        if action not in USER_ACTIONS or direction not in USER_DIRECTIONS:
            raise ProductDemoError("invalid_decisions", "用户决定或方向无效。")
        if not isinstance(useful, bool):
            raise ProductDemoError("invalid_decisions", "useful必须是布尔值。")
        note = _bounded_text(row.get("note", ""), "note", max_length=1200)
        cleaned.append(
            {
                "card_id": card_id,
                "action": action,
                "user_direction": direction,
                "useful": useful,
                "note": note,
            }
        )
        seen.add(card_id)
    return cleaned


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _markdown_pack(pack: dict[str, object]) -> str:
    lines = [
        "# BioEvidence evidence pack",
        "",
        f"- Question: {pack['question']}",
        f"- User status: `{pack['pack_status']}`",
        f"- Mode: `{pack['mode']}`",
        f"- Pack SHA-256: `{pack['pack_sha256']}`",
        "",
        "## User synthesis",
        "",
        str(pack["synthesis"]) or "No synthesis provided.",
        "",
        "## Accepted evidence",
        "",
    ]
    for index, item in enumerate(pack["evidence"], start=1):
        card = item["card"]
        decision = item["user_decision"]
        snippet = card["snippet"]
        lines.extend(
            [
                f"### {index}. {card['title']}",
                "",
                f"- PMID: [{card['pmid']}]({card['source_url']})",
                f"- DOI: {card['doi'] or 'not reported'}",
                f"- Year: {card['year'] or 'not reported'}",
                f"- User direction: `{decision['user_direction']}`",
                f"- Useful: `{str(decision['useful']).lower()}`",
                f"- Note: {decision['note'] or 'none'}",
                f"- Exact abstract snippet: {snippet['text']}",
                f"- Normalized record SHA-256: `{card['record_sha256']}`",
                f"- Snippet SHA-256: `{snippet['snippet_sha256']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundaries",
            "",
            *[f"- {limit}" for limit in pack["scope_limits"]],
            "",
        ]
    )
    return "\n".join(lines)
