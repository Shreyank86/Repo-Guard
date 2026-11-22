"""Audit labelled claims and replay the saved baselines against a pinned index."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repoguard.evaluation.metrics import ScoredAnswer, baseline_metrics, verification_metrics  # noqa: E402
from repoguard.repository import load_index  # noqa: E402
from repoguard.schemas import BaselineRunOutput, BenchmarkQuestion, IndexArtifacts  # noqa: E402
from repoguard.verification import Claim, extract_claims, verify  # noqa: E402


class AuditCase(BaseModel):
    id: str
    claim: Claim
    supported: bool
    source_check: str


class LabelledAudit(BaseModel):
    commit: str
    cases: list[AuditCase]


def audit_labels(index: IndexArtifacts, labels: Path) -> dict:
    audit = LabelledAudit.model_validate_json(labels.read_bytes())
    if audit.commit != index.repository.commit:
        raise ValueError("The manual labels point at a different repository commit.")
    cases = audit.cases
    result = verify([case.claim for case in cases], index)
    pairs = list(zip(cases, result.claims, strict=True))
    return {
        **verification_metrics((case.supported, item.status) for case, item in pairs),
        "cases": [
            {
                "id": case.id,
                "supported": case.supported,
                "source_check": case.source_check,
                **item.model_dump(mode="json"),
            }
            for case, item in pairs
        ],
    }


def replay(index: IndexArtifacts, questions_path: Path, runs_path: Path) -> dict:
    questions = {}
    with questions_path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                question = BenchmarkQuestion.model_validate_json(line)
                if question.question_id in questions:
                    raise ValueError(f"Duplicate question {question.question_id!r}.")
                questions[question.question_id] = question

    seen = set()
    strategies = defaultdict(list)
    with runs_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            run = BaselineRunOutput.model_validate_json(line)
            question = questions.get(run.question_id)
            if question is None or question.question != run.question:
                raise ValueError(f"Run has no matching question {run.question_id!r}.")
            if question.commit != index.repository.commit:
                raise ValueError(
                    f"Run {run.question_id!r} points at a different repository commit."
                )

            strategy = run.baseline.value
            key = (strategy, run.question_id)
            if key in seen:
                raise ValueError(f"Duplicate run for {strategy}/{run.question_id}.")
            seen.add(key)

            claims = [] if run.abstained else extract_claims(run.answer)
            result = verify(claims, index)
            strategies[strategy].append(ScoredAnswer(question.is_trap, run.abstained, result))
    return {name: baseline_metrics(answers) for name, answers in sorted(strategies.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--labels", type=Path, default=Path("tests/fixtures/verifier_audit.json"))
    parser.add_argument(
        "--questions", type=Path, default=Path("data/questions/baseline_questions.jsonl")
    )
    parser.add_argument("--runs", type=Path, default=Path("data/baseline/raw_outputs.jsonl"))
    parser.add_argument(
        "--output", type=Path, help="Optional JSON report path. Defaults to stdout."
    )
    args = parser.parse_args()

    index = load_index(args.index)
    try:
        baselines = replay(index, args.questions, args.runs)
        labelled = audit_labels(index, args.labels)
    except ValueError as exc:
        parser.error(str(exc))
    report = {
        "repository_commit": index.repository.commit,
        "labelled_audit": labelled,
        "baselines": baselines,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
