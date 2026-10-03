"""Small public CLI for the BioEvidence Agent portfolio release."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from .corpus import read_corpus, sha256_file
from .evidence_utilization import run_evidence_utilization_controls
from .llm_agent import LLMEvidenceAgent
from .pubmedqa import CORPUS_FILENAME, prepare_pubmedqa
from .pubmedqa_eval import evaluate_pubmedqa
from .pubmedqa_stress import (
    ABSTRACT_INDEX_MANIFEST,
    AbstractOnlyVectorIndex,
    abstract_passages,
    run_abstract_only_rankings,
    score_abstract_only_rankings,
)
from .reranker import (
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_REVISION,
    CrossEncoderScorer,
)
from .tools import (
    FetchRecordInput,
    InspectEvidenceInput,
    LiteratureTools,
    SearchLiteratureInput,
    ToolExecutor,
    trace_as_dict,
)
from .vector import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    E5Encoder,
    VectorIndex,
    read_index_manifest,
)

# ---------------------------------------------------------------------------
# Scripted fake LLM client — used by react-demo to show the full ReAct loop
# without requiring a live API key.  The response sequence mirrors what a
# real DeepSeek model would do for the statin RCT question.
# ---------------------------------------------------------------------------


@dataclass
class _FakeUsage:
    prompt_tokens: int = 12
    completion_tokens: int = 6


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction


@dataclass
class _FakeMessage:
    content: str | None
    tool_calls: list[_FakeToolCall] | None

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {"role": "assistant"}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return result


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeCompletion:
    choices: list[_FakeChoice]
    usage: _FakeUsage = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = _FakeUsage()


def _tc(name: str, args: dict[str, Any], call_id: str) -> _FakeToolCall:
    return _FakeToolCall(
        id=call_id,
        function=_FakeFunction(name=name, arguments=json.dumps(args)),
    )


def _resp(*tool_calls: _FakeToolCall) -> _FakeCompletion:
    return _FakeCompletion(
        choices=[
            _FakeChoice(
                message=_FakeMessage(content=None, tool_calls=list(tool_calls))
            )
        ],
    )


# ---------------------------------------------------------------------------
# Scenario catalog — each entry has a fixed question and a pre-built scripted
# PMID sequence.  Only questions in this catalog are supported by react-demo;
# free-form input is intentionally not accepted.
# ---------------------------------------------------------------------------

_REACT_DEMO_SCENARIOS: dict[str, dict[str, object]] = {
    "statins": {
        "question": (
            "Do statins reduce major cardiovascular events in randomized controlled trials?"
        ),
        "target_pmid": "1004",
        "finish_verdict": "supported",
        "finish_answer": (
            "A synthetic fixture RCT (PMID 1004) found that statin therapy reduced major "
            "cardiovascular events by 28% over 5 years (HR 0.72, 95% CI 0.61–0.85) with "
            "no significant increase in myopathy. The fixture evidence supports "
            "cardiovascular benefit. NOTE: This document is a synthetic fixture for "
            "demonstration purposes and does not represent a real published trial."
        ),
        "finish_claim": (
            "Statins significantly reduce major cardiovascular events (synthetic fixture data)."
        ),
        "finish_reason": (
            "Single fixture RCT directly answering the question; scripted demo scenario."
        ),
    },
    "metformin": {
        "question": (
            "Does metformin reduce HbA1c compared to placebo in type 2 diabetes?"
        ),
        "target_pmid": "1005",
        "finish_verdict": "supported",
        "finish_answer": (
            "A synthetic fixture RCT (PMID 1005) found metformin reduced HbA1c by 1.5% "
            "versus 0.3% in placebo over 12 months (p < 0.001). Weight was stable in the "
            "metformin group. NOTE: This document is a synthetic fixture and does not "
            "represent a real published trial."
        ),
        "finish_claim": (
            "Metformin significantly reduces HbA1c versus placebo (synthetic fixture data)."
        ),
        "finish_reason": (
            "Single fixture RCT with clear primary endpoint; scripted demo scenario."
        ),
    },
    "aspirin-primary": {
        "question": (
            "Does aspirin provide net benefit for primary cardiovascular prevention "
            "in low-risk adults?"
        ),
        "target_pmid": "1006",
        "finish_verdict": "contradicted",
        "finish_answer": (
            "A synthetic fixture RCT (PMID 1006) found aspirin did not significantly reduce "
            "first myocardial infarction (HR 0.96) and increased major GI bleeding by 44% "
            "over 7 years. Net benefit was absent in the low-risk population. NOTE: This "
            "document is a synthetic fixture and does not represent a real published trial."
        ),
        "finish_claim": (
            "Aspirin shows net harm, not benefit, in low-risk primary prevention "
            "(synthetic fixture data)."
        ),
        "finish_reason": (
            "Fixture RCT shows net harm; verdict is contradicted for the stated claim; "
            "scripted demo scenario."
        ),
    },
    "pembrolizumab": {
        "question": (
            "Does pembrolizumab plus chemotherapy improve overall survival in "
            "metastatic non-small cell lung cancer?"
        ),
        "target_pmid": "1008",
        "finish_verdict": "supported",
        "finish_answer": (
            "A synthetic fixture RCT (PMID 1008) found pembrolizumab plus chemotherapy "
            "improved overall survival versus chemotherapy alone (median 21.9 vs 14.2 months, "
            "HR 0.62). NOTE: This document is a synthetic fixture and does not represent a "
            "real published trial."
        ),
        "finish_claim": (
            "Pembrolizumab plus chemotherapy improves OS in metastatic NSCLC "
            "(synthetic fixture data)."
        ),
        "finish_reason": (
            "Fixture RCT shows clear OS benefit; scripted demo scenario."
        ),
    },
}


def _build_scripted_responses(scenario: dict[str, object]) -> list[_FakeCompletion]:
    """Return the four-step ReAct script for the given scenario dict."""
    question = str(scenario["question"])
    pmid = str(scenario["target_pmid"])
    return [
        # Step 1 — search
        _resp(
            _tc(
                "search_literature",
                {"query": question, "top_k": 3},
                "tc_search",
            )
        ),
        # Step 2 — fetch the target PMID
        _resp(_tc("fetch_record", {"pmid": pmid}, "tc_fetch")),
        # Step 3 — inspect evidence
        _resp(
            _tc(
                "inspect_evidence",
                {"pmid": pmid, "query": question, "max_snippets": 2},
                "tc_inspect",
            )
        ),
        # Step 4 — finish
        _resp(
            _tc(
                "finish",
                {
                    "verdict": scenario["finish_verdict"],
                    "answer": scenario["finish_answer"],
                    "claim": scenario["finish_claim"],
                    "cited_pmids": [pmid],
                    "decisive_reason": scenario["finish_reason"],
                },
                "tc_finish",
            )
        ),
    ]


def _make_agent_with_fake_client(
    tools: LiteratureTools,
    scripted_responses: list[_FakeCompletion],
    max_steps: int = 8,
) -> LLMEvidenceAgent:
    """Build an LLMEvidenceAgent with a scripted fake client (no API key needed)."""

    class _ScriptedClient:
        def __init__(self, responses: list[_FakeCompletion]) -> None:
            self._responses = list(responses)
            self._index = 0
            self.chat = self  # support self.chat.completions.create(...)
            self.completions = self

        def create(self, **_kwargs: Any) -> _FakeCompletion:
            if self._index >= len(self._responses):
                raise RuntimeError(
                    "ScriptedClient exhausted: more LLM calls than scripted responses."
                )
            result = self._responses[self._index]
            self._index += 1
            return result

    agent = LLMEvidenceAgent.__new__(LLMEvidenceAgent)
    agent._tools = tools
    agent._model = "scripted-demo"
    agent._max_steps = max_steps
    agent._max_tokens_per_call = 1024
    agent._base_url = "https://api.deepseek.com/v1"
    agent._client = _ScriptedClient(scripted_responses)
    return agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bioevidence")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser(
        "demo",
        help="Run a lightweight BM25 + typed-tool provenance demo.",
    )
    demo.add_argument("--question", required=True)
    demo.add_argument("--corpus", type=Path)

    react_demo = commands.add_parser(
        "react-demo",
        help=(
            "Walk through the full ReAct tool-use loop with a scripted fake LLM "
            "(no API key required). Shows search → fetch → inspect → finish with "
            "complete provenance output. Uses synthetic fixture corpus only."
        ),
    )
    react_demo.add_argument(
        "--scenario",
        choices=list(_REACT_DEMO_SCENARIOS.keys()),
        default="statins",
        help=(
            "Which scripted scenario to run. Each scenario has a fixed question and "
            "pre-built scripted LLM responses targeting a specific fixture PMID. "
            f"Choices: {', '.join(_REACT_DEMO_SCENARIOS.keys())} (default: statins)."
        ),
    )
    react_demo.add_argument("--corpus", type=Path)
    react_demo.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum ReAct loop steps (default: 8).",
    )

    prepare = commands.add_parser(
        "prepare",
        help="Download and hash-verify the pinned PubMedQA benchmark.",
    )
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--raw", type=Path)
    prepare.add_argument("--test-ground-truth", type=Path)

    vector_build = commands.add_parser(
        "vector-index-build",
        help="Build the title-assisted multilingual-E5 index.",
    )
    vector_build.add_argument("--benchmark-dir", type=Path, required=True)
    vector_build.add_argument("--index-dir", type=Path, required=True)
    vector_build.add_argument("--device", default="cpu")
    vector_build.add_argument("--batch-size", type=int, default=16)

    evaluate = commands.add_parser(
        "evaluate",
        help="Run the 500-case title-assisted public benchmark.",
    )
    _add_heavy_retrieval_arguments(evaluate)
    evaluate.add_argument("--responses", type=Path, required=True)

    abstract_build = commands.add_parser(
        "abstract-index-build",
        help="Build the title-free E5 index from abstracts only.",
    )
    abstract_build.add_argument("--benchmark-dir", type=Path, required=True)
    abstract_build.add_argument("--index-dir", type=Path, required=True)
    abstract_build.add_argument("--device", default="cpu")
    abstract_build.add_argument("--batch-size", type=int, default=16)

    abstract_stress = commands.add_parser(
        "abstract-stress",
        help="Freeze and score the title-free retrieval stress test.",
    )
    _add_heavy_retrieval_arguments(abstract_stress)
    abstract_stress.add_argument("--rankings", type=Path, required=True)
    abstract_stress.add_argument(
        "--ranking-manifest",
        type=Path,
        required=True,
    )
    abstract_stress.add_argument(
        "--title-assisted-report",
        type=Path,
        required=True,
    )

    controls = commands.add_parser(
        "evidence-controls",
        help="Run five-fold evidence-utilization negative controls.",
    )
    controls.add_argument("--benchmark-dir", type=Path, required=True)
    controls.add_argument("--run-dir", type=Path, required=True)
    controls.add_argument("--output-json", type=Path, required=True)
    controls.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        return _demo(args)
    if args.command == "react-demo":
        return _react_demo(args)
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "vector-index-build":
        return _vector_index_build(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "abstract-index-build":
        return _abstract_index_build(args)
    if args.command == "abstract-stress":
        return _abstract_stress(args)
    if args.command == "evidence-controls":
        return _evidence_controls(args)
    raise AssertionError(f"unhandled command: {args.command}")


def _demo(args: argparse.Namespace) -> int:
    corpus_path = args.corpus or Path(
        str(files("bioevidence").joinpath("fixtures/tiny_corpus.jsonl"))
    )
    documents = read_corpus(corpus_path)
    executor = ToolExecutor(LiteratureTools(documents), max_calls=3)
    search = executor.search(
        SearchLiteratureInput(query=args.question, top_k=1)
    )
    hit = search.hits[0]
    record = executor.fetch(FetchRecordInput(pmid=hit.pmid))
    inspected = executor.inspect(
        InspectEvidenceInput(
            pmid=hit.pmid,
            query=args.question,
            max_snippets=1,
        )
    )
    snippet = inspected.snippets[0] if inspected.snippets else None
    result = {
        "demo_version": "1.0.0",
        "scope": "retrieval and provenance demo; no medical verdict",
        "question": args.question,
        "retrieved": {
            "pmid": record.pmid,
            "title": record.title,
            "url": record.source_url,
            "document_sha256": record.document_sha256,
            "rank": hit.rank,
            "score": hit.score,
            "snippet": snippet.text if snippet else None,
            "snippet_sha256": snippet.snippet_sha256 if snippet else None,
        },
        "corpus_sha256": sha256_file(corpus_path),
        "tool_trace": trace_as_dict(executor.trace),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _react_demo(args: argparse.Namespace) -> int:
    corpus_path = args.corpus or Path(
        str(files("bioevidence").joinpath("fixtures/tiny_corpus.jsonl"))
    )
    from .corpus import sha256_file as _sha256_file

    documents = read_corpus(corpus_path)
    tools = LiteratureTools(documents)
    scenario = _REACT_DEMO_SCENARIOS[args.scenario]
    question = str(scenario["question"])
    scripted = _build_scripted_responses(scenario)
    agent = _make_agent_with_fake_client(tools, scripted, max_steps=args.max_steps)
    agent_response = agent.run(question=question, request_id=f"react-demo-{args.scenario}")

    # Verify the agent response passes the product contract before printing.
    # If it has errors, they surface in the envelope rather than silently corrupting output.
    from .product_contracts import validate_product_response as _validate
    contract_errors = _validate(agent_response)

    # Wrap in a demo envelope so the output is clearly distinguished from a production
    # API response.  The envelope is NOT a product response; the agent_response within it
    # IS, and it is validated above.  The two layers have different purposes and different
    # consumers: the envelope is for human readers of the demo; the inner response is what
    # an integration would parse.
    envelope = {
        "demo_envelope": {
            "scenario": args.scenario,
            "question": question,
            "scripted_mode": True,
            "scripted_mode_note": (
                "This demo uses a scripted fake LLM client — no API key required. "
                "The tool-call sequence (search → fetch → inspect → finish) is identical "
                "to a live DeepSeek run; only the model decision step is scripted."
            ),
            "corpus_is_synthetic": True,
            "corpus_note": (
                "All documents in tiny_corpus.jsonl (PMIDs 1001-1010) are synthetic "
                "fixtures generated for demonstration purposes. They do NOT correspond "
                "to real published papers. PMIDs 1001-1003 are simple three-sentence "
                "demos; PMIDs 1004-1010 are LLM-generated RCT-style abstracts with "
                "plausible but invented statistics."
            ),
            "corpus_path": str(corpus_path),
            "corpus_sha256": _sha256_file(corpus_path),
            "corpus_size": len(documents),
            "agent_response_contract_valid": len(contract_errors) == 0,
            "agent_response_contract_errors": contract_errors,
        },
        "agent_response": agent_response,
    }
    print(json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not contract_errors else 1


def _prepare(args: argparse.Namespace) -> int:
    result = prepare_pubmedqa(
        args.output_dir,
        raw_path=args.raw,
        test_ground_truth_path=args.test_ground_truth,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _vector_index_build(args: argparse.Namespace) -> int:
    corpus_path = args.benchmark_dir / CORPUS_FILENAME
    encoder = E5Encoder(
        model_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_MODEL_REVISION,
        device=args.device,
    )
    result = VectorIndex.build(
        read_corpus(corpus_path),
        encoder=encoder,
        batch_size=args.batch_size,
    ).save(args.index_dir, corpus_path=corpus_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    index_manifest = read_index_manifest(args.index_dir)
    encoder = E5Encoder(
        model_id=str(index_manifest["model_id"]),
        revision=str(index_manifest["model_revision"]),
        device=args.device,
        max_length=int(index_manifest["max_length"]),
    )
    reranker = _reranker(args)
    report = evaluate_pubmedqa(
        benchmark_dir=args.benchmark_dir,
        index_dir=args.index_dir,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        responses_path=args.responses,
        encoder=encoder,
        reranker=reranker,
        reranker_candidate_k=args.reranker_candidates,
    )
    print(json.dumps(_retrieval_summary(report), indent=2, sort_keys=True))
    return 0


def _abstract_index_build(args: argparse.Namespace) -> int:
    corpus_path = args.benchmark_dir / CORPUS_FILENAME
    encoder = E5Encoder(
        model_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_MODEL_REVISION,
        device=args.device,
    )
    result = AbstractOnlyVectorIndex.build(
        abstract_passages(read_corpus(corpus_path)),
        encoder=encoder,
        batch_size=args.batch_size,
    ).save(
        args.index_dir,
        corpus_path=corpus_path,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _abstract_stress(args: argparse.Namespace) -> int:
    index_manifest = json.loads(
        (args.index_dir / ABSTRACT_INDEX_MANIFEST).read_text(encoding="utf-8")
    )
    encoder = E5Encoder(
        model_id=str(index_manifest["model_id"]),
        revision=str(index_manifest["model_revision"]),
        device=args.device,
        max_length=int(index_manifest["max_length"]),
    )
    run_abstract_only_rankings(
        benchmark_dir=args.benchmark_dir,
        index_dir=args.index_dir,
        rankings_path=args.rankings,
        ranking_manifest_path=args.ranking_manifest,
        encoder=encoder,
        reranker=_reranker(args),
        reranker_candidate_k=args.reranker_candidates,
    )
    report = score_abstract_only_rankings(
        benchmark_dir=args.benchmark_dir,
        rankings_path=args.rankings,
        ranking_manifest_path=args.ranking_manifest,
        title_assisted_report_path=args.title_assisted_report,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(json.dumps(_retrieval_summary(report), indent=2, sort_keys=True))
    return 0


def _evidence_controls(args: argparse.Namespace) -> int:
    report = run_evidence_utilization_controls(
        benchmark_dir=args.benchmark_dir,
        run_dir=args.run_dir,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(
        json.dumps(
            {
                name: {
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
                for name, metrics in report["metrics"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _reranker(args: argparse.Namespace) -> CrossEncoderScorer:
    return CrossEncoderScorer(
        model_id=args.reranker_model_id,
        revision=args.reranker_revision,
        device=args.device,
        batch_size=args.reranker_batch_size,
    )


def _retrieval_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "case_count": report["case_count"],
        "retrieval_metrics": report["retrieval_metrics"],
    }


def _add_heavy_retrieval_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reranker-model-id",
        default=DEFAULT_RERANKER_MODEL_ID,
    )
    parser.add_argument(
        "--reranker-revision",
        default=DEFAULT_RERANKER_REVISION,
    )
    parser.add_argument("--reranker-candidates", type=int, default=20)
    parser.add_argument("--reranker-batch-size", type=int, default=16)


if __name__ == "__main__":
    raise SystemExit(main())
