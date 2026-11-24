from pathlib import Path

import pytest

from repoguard.evaluation.benchmark import BenchmarkCase, SourceReference
from repoguard.evaluation.benchmark_scoring import (
    GenerationSettings,
    Prediction,
    evaluate_predictions,
    summarize,
)
from repoguard.repository import load_index
from repoguard.schemas import CallEdge, FileMetadata, FunctionMetadata, IndexArtifacts
from repoguard.verification import Claim, ClaimType

COMMIT = "1" * 40


@pytest.fixture
def index() -> IndexArtifacts:
    fixture = Path(__file__).resolve().parents[1] / "fixtures/indexes/indexer_repo"
    return load_index(fixture)


@pytest.fixture
def case() -> BenchmarkCase:
    file = "pkg/config.py"
    symbol = "pkg.config.load_config"
    return BenchmarkCase(
        benchmark_version="1.0",
        case_id="find-load-config",
        repository_id="indexer-repo",
        repository_url="https://github.com/example/indexer-repo",
        commit=COMMIT,
        licence="MIT",
        split="train",
        question_type="definition",
        question="Where is load_config defined?",
        answerable=True,
        expected_facts=[Claim(type=ClaimType.FUNCTION_EXISTS, symbol=symbol, file=file)],
        evidence=[SourceReference(file=file, start_line=9, end_line=9, symbol=symbol)],
        reference_answer="load_config is defined in pkg/config.py.",
    )


def _answer(case: BenchmarkCase, text: str, **fields: object) -> Prediction:
    return Prediction(
        system="test-system",
        case_id=case.case_id,
        commit=case.commit,
        generation=GenerationSettings(
            mode="mock",
            model="test-model",
            max_new_tokens=128,
            temperature=0,
            top_p=0.95,
            repetition_penalty=1,
        ),
        answer=text,
        abstained=False,
    ).model_copy(update=fields)


def test_scores_scoped_definitions_with_aliases_and_rejects_wrong_file(case, index) -> None:
    supported = _answer(case, "The function `load_config` is defined in `pkg/config.py`.")
    wrong_file = _answer(case, "The function `load_config` is defined in `pkg/app.py`.")

    [good] = evaluate_predictions([case], [supported], {case.repository_id: index})
    [bad] = evaluate_predictions([case], [wrong_file], {case.repository_id: index})

    assert good.structural_correct and good.matched_facts == 1
    assert summarize([good])["claim_precision"] == 1
    assert bad.outcome == "refuted_claim" and bad.matched_facts == 0


def test_skips_missing_telemetry_and_distinguishes_abstention_errors(case, index) -> None:
    trap = case.model_copy(
        update={
            "case_id": "missing",
            "answerable": False,
            "question_type": "near_miss",
            "expected_facts": [],
        }
    )
    predictions = [
        _answer(case, "ABSTAIN", abstained=True, steps=1),
        _answer(trap, "ABSTAIN", abstained=True, steps=1),
    ]
    scored = evaluate_predictions([case, trap], predictions, {case.repository_id: index})
    summary = summarize(scored)

    assert summary["structural_answer_accuracy"] == 0.5
    assert summary["coverage"] == 0
    assert summary["abstention_recall"] == 1
    assert summary["abstention_precision"] == 0.5
    assert summary["false_abstention_rate"] == 1
    assert summary["claim_extraction_coverage"] is None
    assert summary["mean_input_tokens"] is None
    assert summary["latency_p95_ms"] is None


def test_retrieval_score_requires_matching_index_and_uses_real_rank(case, index) -> None:
    [relevant] = case.evidence
    result = _answer(
        case,
        "The function `load_config` is defined in `pkg/config.py`.",
        index_id=index.index_id,
        retrieved_units=["pkg/app.py::pkg.app.main", relevant.unit],
    )
    [scored] = evaluate_predictions([case], [result], {case.repository_id: index})

    assert scored.retrieval["recall_at_5"] == 1
    assert scored.retrieval["mrr_at_5"] == 0.5
    with pytest.raises(ValueError, match="index ID mismatch"):
        evaluate_predictions(
            [case], [result.model_copy(update={"index_id": "stale"})], {case.repository_id: index}
        )


def test_rejects_incomplete_or_incomparable_predictions(case, index) -> None:
    another = case.model_copy(update={"case_id": "another"})
    run = _answer(case, "ABSTAIN", abstained=True)
    different_model = run.generation.model_copy(update={"model": "different"})
    indexes = {case.repository_id: index}

    with pytest.raises(ValueError, match="missing 1 selected"):
        evaluate_predictions([case, another], [run], indexes)
    with pytest.raises(ValueError, match="identical model"):
        evaluate_predictions(
            [case],
            [run, run.model_copy(update={"system": "two", "generation": different_model})],
            indexes,
        )


def test_verified_call_from_a_different_same_named_caller_is_not_gold(case, index) -> None:
    file = "pkg/other.py"
    other = "pkg.other.initialize_app"
    callee = "pkg.config.load_config"
    index.files.append(FileMetadata(path=file, module="pkg.other", line_count=5))
    index.symbols.append(
        FunctionMetadata(
            symbol_id=f"{file}::{other}",
            name="initialize_app",
            qualified_name=other,
            type="function",
            file=file,
            module="pkg.other",
            line=2,
            end_line=3,
        )
    )
    index.call_edges.append(
        CallEdge(
            caller=other,
            caller_name="initialize_app",
            callee="load_config",
            callee_expr="load_config",
            callee_qualified=callee,
            resolution="imported",
            file=file,
            line=3,
        )
    )
    case = case.model_copy(
        update={
            "question_type": "call",
            "expected_facts": [
                Claim(type=ClaimType.CALLS, caller="pkg.app.App.initialize_app", callee=callee)
            ],
        }
    )
    response = _answer(
        case,
        "The other initializer calls load_config.",
        claims=[Claim(type=ClaimType.CALLS, caller="initialize_app", callee=callee, file=file)],
    )
    [score] = evaluate_predictions([case], [response], {case.repository_id: index})

    assert score.outcome == "missing_required_fact"
    assert score.matched_facts == 0
