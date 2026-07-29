"""Structured PubMedQA Agent using the existing typed local literature tools."""

from __future__ import annotations

import hashlib

from .corpus import CorpusDocument
from .product_contracts import validate_product_response
from .pubmedqa import PUBMEDQA_VERDICT_MAP, TfidfLogisticAnswerer
from .retrievers import Retriever
from .tools import (
    FetchRecordInput,
    InspectEvidenceInput,
    LiteratureTools,
    SearchLiteratureInput,
    ToolExecutor,
    trace_as_dict,
)


class PubMedQAEvidenceAgent:
    """Retrieve one article, predict yes/no/maybe, and cite an exact span."""

    def __init__(
        self,
        documents: list[CorpusDocument],
        *,
        corpus_sha256: str,
        retriever: Retriever,
        answerer: TfidfLogisticAnswerer,
    ) -> None:
        self._documents = documents
        self.corpus_sha256 = corpus_sha256
        self._retriever = retriever
        self._answerer = answerer
        self.baseline_id = (
            f"{answerer.baseline_id}-{retriever.method.replace('_', '-')}"
        )

    def run(
        self,
        *,
        question: str,
        request_id: str | None = None,
    ) -> dict[str, object]:
        executor = ToolExecutor(
            LiteratureTools(self._documents, retriever=self._retriever),
            max_calls=4,
        )
        search = executor.search(
            SearchLiteratureInput(
                query=question,
                top_k=10,
                rerank_query=question,
            )
        )
        if not search.hits:
            raise RuntimeError("PubMedQA retriever returned no documents")
        hit = search.hits[0]
        record = executor.fetch(FetchRecordInput(pmid=hit.pmid))
        inspection = executor.inspect(
            InspectEvidenceInput(
                pmid=hit.pmid,
                query=question,
                max_snippets=1,
            )
        )
        if not inspection.snippets:
            inspection = executor.inspect(
                InspectEvidenceInput(
                    pmid=hit.pmid,
                    query=record.abstract[:240],
                    max_snippets=1,
                )
            )
        if not inspection.snippets:
            raise RuntimeError("retrieved PubMedQA context has no citable span")

        label, probabilities = self._answerer.predict(question, record.abstract)
        verdict = PUBMEDQA_VERDICT_MAP[label]
        snippet = inspection.snippets[0]
        citation_id = "C1"
        direction = {
            "yes": "supports",
            "no": "contradicts",
            "maybe": "context_only",
        }[label]
        confidence_score = max(probabilities.values())
        confidence = (
            "high"
            if confidence_score >= 0.75
            else "medium" if confidence_score >= 0.5 else "low"
        )
        response = {
            "response_version": "1.0.0",
            "baseline_id": self.baseline_id,
            "request_id": request_id or _request_id(question),
            "verdict": verdict,
            "abstained": verdict == "insufficient",
            "answer": (
                f"固定 PubMedQA 回答器预测为 {label}，映射为 "
                f"{verdict}。摘要证据片段：{snippet.text} [{citation_id}]"
            ),
            "claims": [
                {
                    "text": (
                        f"该公开基准样例的模型预测标签为 {label}；"
                        "这是一项摘要级模型输出，而非临床建议。"
                    ),
                    "citation_ids": [citation_id],
                }
            ],
            "citations": [
                {
                    "citation_id": citation_id,
                    "pmid": record.pmid,
                    "doi": record.doi,
                    "url": record.source_url,
                    "title": record.title,
                    "source_sha256": record.document_sha256,
                    "section": snippet.section,
                    "snippet": snippet.text,
                    "start_char": snippet.start_char,
                    "end_char": snippet.end_char,
                    "snippet_sha256": snippet.snippet_sha256,
                    "direction": direction,
                    "retrieval": {
                        "method": hit.retrieval_method,
                        "rank": hit.rank,
                        "score": hit.score,
                        "component_ranks": hit.component_ranks,
                        "candidate_rank": hit.candidate_rank,
                    },
                }
            ],
            "decisive_reason": (
                "verdict 来自仅在 500 条官方非测试记录上训练的固定 "
                "TF-IDF + logistic-regression 回答器；概率不是科学证据强度。"
            ),
            "scope_limits": [
                "只使用冻结的 PubMedQA 闭集语料和无结论摘要上下文。",
                "公开标签表现不代表私有盲测、开放世界检索或临床有效性。",
                "未进行独立人工语义复核，因此不报告确认的语义幻觉率。",
            ],
            "confidence": confidence,
            "provenance": {
                "corpus_sha256": self.corpus_sha256,
                "retrieval_method": self._retriever.method,
                "retrieval_config": self._retriever.metadata,
                "answer_model": self._answerer.metadata,
                "answer_probabilities": probabilities,
                "tool_call_count": len(executor.trace),
                "tool_trace": trace_as_dict(executor.trace),
                "model_api_cost_usd": 0.0,
            },
        }
        violations = validate_product_response(response)
        if violations:
            raise ValueError(
                "PubMedQA response contract failure: " + " | ".join(violations)
            )
        return response


def _request_id(question: str) -> str:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12].upper()
    return f"PQA-{digest}"
