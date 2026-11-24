"""Validate the benchmark or score retrieval and answer comparisons."""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repoguard.evaluation.benchmark import (  # noqa: E402
    BenchmarkCase,
    load_benchmark,
    read_jsonl,
    validate_indexes,
)
from repoguard.evaluation.benchmark_scoring import (  # noqa: E402
    AnswerReview,
    CaseScore,
    Prediction,
    evaluate_predictions,
    summarize,
)
from repoguard.exceptions import RepoGuardError  # noqa: E402
from repoguard.retrieval import Retriever  # noqa: E402
from repoguard.retrieval.metrics import EvalReport, evaluate_cases  # noqa: E402
from repoguard.schemas import IndexArtifacts  # noqa: E402

DEFAULT_BENCHMARK = ROOT / "data/questions/phase2_benchmark_v1.jsonl"
DEFAULT_INDEXES = ROOT / "data/indexes"
DEFAULT_OUTPUT = ROOT / "experiments/results/phase2"


def _retrieval(
    cases: list[BenchmarkCase], indexes: dict[str, IndexArtifacts]
) -> tuple[dict, list[dict]]:
    by_repo = defaultdict(list)
    for case in cases:
        if case.answerable:
            by_repo[case.repository_id].append(
                {
                    "case_id": case.case_id,
                    "case_type": case.question_type,
                    "repository_id": case.repository_id,
                    "query": case.question,
                    "relevant_units": sorted({reference.unit for reference in case.evidence}),
                    "split_hint": case.split,
                }
            )
    summary = {}
    details = []
    retrievers = {key: Retriever(indexes[key]) for key in sorted(by_repo)}
    for variant, options in (
        ("hybrid", {}),
        ("dense_only", {"definition_boost": 0.0, "mention_boost": 0.0}),
    ):
        scores = EvalReport()
        for key, retriever in retrievers.items():
            repo_report = evaluate_cases(retriever, by_repo[key], ks=(1, 5), **options)
            scores.cases.extend(repo_report.cases)
            details.extend(
                {
                    "variant": variant,
                    "backend": retriever.backend,
                    "repository_id": key,
                    "case_id": row.case_id,
                    "split": row.split,
                    "question_type": row.case_type,
                    "relevant": row.relevant,
                    "ranked": row.ranked,
                    "metrics": row.metrics,
                    "latency_ms": row.latency_ms,
                }
                for row in repo_report.cases
            )
        summary[variant] = {
            "backends": {key: retriever.backend for key, retriever in retrievers.items()},
            "by_repository": scores.summary(("repository_id",)),
            "by_split": scores.summary(("split",)),
            "by_question_type": scores.summary(("case_type",)),
        }
    return summary, details


def _scores(scored: list[CaseScore]) -> dict:
    groups = defaultdict(list)
    for row in scored:
        groups[row.prediction.system].append(row)
    result = {}
    for system, rows in sorted(groups.items()):
        partitions = {}
        for name in ("repository_id", "split", "question_type"):
            slices = defaultdict(list)
            for row in rows:
                slices[getattr(row, name)].append(row)
            partitions[f"by_{name}"] = {
                key: summarize(items) for key, items in sorted(slices.items())
            }
        result[system] = {
            **summarize(rows),
            **partitions,
            "outcomes": dict(sorted(Counter(row.outcome for row in rows).items())),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--indexes", type=Path, default=DEFAULT_INDEXES)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="all")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--retrieval-only", action="store_true")
    mode.add_argument("--predictions", type=Path, nargs="+")
    parser.add_argument("--reviews", type=Path, help="Optional independently reviewed answers.")
    args = parser.parse_args()
    if args.reviews and not args.predictions:
        parser.error("--reviews requires --predictions.")

    try:
        all_cases = load_benchmark(args.benchmark)
        cases = [row for row in all_cases if args.split in ("all", row.split)]
        indexes = validate_indexes(cases, args.indexes)
        report = {
            "benchmark_version": all_cases[0].benchmark_version,
            "benchmark_sha256": hashlib.sha256(
                args.benchmark.read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest(),
            "split": args.split,
            "case_count": len(cases),
            "index_ids": {key: index.index_id for key, index in indexes.items()},
        }
        if args.validate_only:
            print(json.dumps(report, indent=2))
            return
        if args.retrieval_only:
            report["retrieval"], details = _retrieval(cases, indexes)
            filename = "retrieval"
        else:
            rows = [row for path in args.predictions for row in read_jsonl(path, Prediction)]
            reviews = read_jsonl(args.reviews, AnswerReview) if args.reviews else []
            scored = evaluate_predictions(cases, rows, indexes, reviews)
            report["generation"] = rows[0].generation.model_dump(mode="json")
            report["systems"] = _scores(scored)
            details = [row.model_dump(mode="json") for row in scored]
            filename = "comparison"
    except (OSError, RepoGuardError, ValueError) as exc:
        parser.error(str(exc))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_case = args.out_dir / f"{filename}_cases.jsonl"
    per_case.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details),
        encoding="utf-8",
    )
    summary = args.out_dir / f"{filename}_summary.json"
    summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(details)} scores to {per_case} and a summary to {summary}")


if __name__ == "__main__":
    main()
