"""Small public CLI for the BioEvidence Agent portfolio release."""

from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path

from .corpus import read_corpus, sha256_file
from .evidence_utilization import run_evidence_utilization_controls
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bioevidence")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser(
        "demo",
        help="Run a lightweight BM25 + typed-tool provenance demo.",
    )
    demo.add_argument("--question", required=True)
    demo.add_argument("--corpus", type=Path)

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
