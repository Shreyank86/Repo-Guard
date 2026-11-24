"""Score comparable Phase 2 runs against independent, source-grounded labels."""

from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean, median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from repoguard.evaluation.benchmark import BenchmarkCase, Commit, Text
from repoguard.retrieval.metrics import ndcg_at_k, percentile, recall_at_k, reciprocal_rank
from repoguard.schemas import IndexArtifacts
from repoguard.verification import (
    Claim,
    ClaimType,
    ClaimVerdict,
    VerdictStatus,
    extract_claims,
    verify,
)

_DEFINITIONS = {ClaimType.CLASS_EXISTS, ClaimType.FUNCTION_EXISTS, ClaimType.SYMBOL_EXISTS}


class GenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["mock", "live"]
    model: Text
    revision: Commit | None = None
    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0, allow_inf_nan=False)
    top_p: float = Field(gt=0, le=1)
    repetition_penalty: float = Field(gt=0, allow_inf_nan=False)
    seed: int | None = None

    @model_validator(mode="after")
    def check_revision(self) -> "GenerationSettings":
        if self.mode == "live" and self.revision is None:
            raise ValueError("Live comparisons require an immutable model revision.")
        return self


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: Text
    case_id: Text
    commit: Commit
    generation: GenerationSettings
    answer: str
    abstained: bool
    claims: list[Claim] | None = None
    index_id: str | None = None
    retrieved_units: list[Text] | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    steps: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class AnswerReview(BaseModel):
    """Optional independent human judgment of a complete answer."""

    model_config = ConfigDict(extra="forbid")

    system: Text
    case_id: Text
    correct: bool
    rationale: Text


class CaseScore(BaseModel):
    prediction: Prediction
    repository_id: str
    split: str
    question_type: str
    answerable: bool
    required_facts: int
    matched_facts: int
    structural_claims: int
    verdicts: list[ClaimVerdict]
    structural_correct: bool
    outcome: str
    retrieval: dict[str, float] | None
    review: AnswerReview | None = None


def _matches_fact(expected: Claim, item: ClaimVerdict, index: IndexArtifacts) -> bool:
    claim = item.claim
    if claim is None or item.status != VerdictStatus.VERIFIED:
        return False
    if expected.file is not None and claim.file != expected.file:
        return False
    generic_definition = (
        expected.type in _DEFINITIONS
        and claim.type in _DEFINITIONS
        and ClaimType.SYMBOL_EXISTS in (expected.type, claim.type)
    )
    if expected.type != claim.type and not generic_definition:
        return False
    if expected.type == ClaimType.FILE_EXISTS:
        return True
    if expected.type == ClaimType.IMPORTS:
        return claim.module == expected.module

    identities = {evidence.qualified_name for evidence in item.evidence}
    if expected.type == ClaimType.CALLS:
        assert expected.callee is not None
        gold_sites = {
            (edge.file, edge.line)
            for edge in index.callers_of(expected.callee)
            if edge.caller == expected.caller and edge.callee_qualified == expected.callee
        }
        return identities == {expected.callee} and any(
            (evidence.file, evidence.line) in gold_sites for evidence in item.evidence
        )

    return identities == {expected.symbol} and claim.parameter == expected.parameter


def _outcome(
    case: BenchmarkCase, prediction: Prediction, verdicts: list[ClaimVerdict], matches: int
) -> str:
    if prediction.abstained:
        return "false_abstention" if case.answerable else "correct_abstention"
    if not case.answerable:
        return "answered_unanswerable"
    if not verdicts:
        return "no_extracted_claims"
    if any(item.status == VerdictStatus.REFUTED for item in verdicts):
        return "refuted_claim"
    if any(item.status == VerdictStatus.UNVERIFIABLE for item in verdicts):
        return "unverifiable_claim"
    if matches < len(case.expected_facts):
        return "missing_required_fact"
    return "supported_answer"


def _score_case(
    case: BenchmarkCase,
    prediction: Prediction,
    index: IndexArtifacts,
    review: AnswerReview | None,
) -> CaseScore:
    claims = (
        prediction.claims if prediction.claims is not None else extract_claims(prediction.answer)
    )
    claims = [] if prediction.abstained else list(dict.fromkeys(claims))
    verdicts = verify(claims, index).claims
    matched = set()
    for item in verdicts:
        for i, fact in enumerate(case.expected_facts):
            if i not in matched and _matches_fact(fact, item, index):
                matched.add(i)
                break

    retrieval = None
    if prediction.retrieved_units is not None and case.answerable:
        ranked = list(dict.fromkeys(prediction.retrieved_units))[:5]
        relevant = {evidence.unit for evidence in case.evidence}
        retrieval = {
            "recall_at_5": recall_at_k(ranked, relevant, 5),
            "mrr_at_5": reciprocal_rank(ranked, relevant),
            "ndcg_at_5": ndcg_at_k(ranked, relevant, 5),
        }
    outcome = _outcome(case, prediction, verdicts, len(matched))
    return CaseScore(
        prediction=prediction,
        repository_id=case.repository_id,
        split=case.split,
        question_type=case.question_type,
        answerable=case.answerable,
        required_facts=len(case.expected_facts),
        matched_facts=len(matched),
        structural_claims=sum(claim.type != ClaimType.SEMANTIC for claim in claims),
        verdicts=verdicts,
        structural_correct=outcome in {"supported_answer", "correct_abstention"},
        outcome=outcome,
        retrieval=retrieval,
        review=review,
    )


def evaluate_predictions(
    cases: list[BenchmarkCase],
    predictions: list[Prediction],
    indexes: dict[str, IndexArtifacts],
    reviews: Iterable[AnswerReview] = (),
) -> list[CaseScore]:
    """Reject mismatched snapshots, incomparable settings, and incomplete systems."""
    if not cases or not predictions:
        raise ValueError("Evaluation requires cases and predictions.")
    by_id = {case.case_id: case for case in cases}
    expected_ids = set(by_id)
    generation = predictions[0].generation
    review_map = {}
    for review in reviews:
        key = (review.system, review.case_id)
        if key in review_map:
            raise ValueError(f"Duplicate independent answer review for {key}.")
        review_map[key] = review

    seen: dict[str, set[str]] = defaultdict(set)
    units = {
        name: {passage.passage_id.rsplit("@", 1)[0] for passage in index.passages}
        for name, index in indexes.items()
    }
    scored = []
    for prediction in predictions:
        case = by_id.get(prediction.case_id)
        if case is None or prediction.commit != case.commit:
            raise ValueError(f"Incorrect benchmark case or commit for {prediction.case_id!r}.")
        if prediction.generation != generation:
            raise ValueError("Compare systems with identical model, mode, revision, and settings.")
        attempted = seen[prediction.system]
        if prediction.case_id in attempted:
            raise ValueError(f"Duplicate prediction for {prediction.system}/{prediction.case_id}.")
        attempted.add(prediction.case_id)

        index = indexes[case.repository_id]
        if prediction.retrieved_units is not None:
            if not index.index_id or prediction.index_id != index.index_id:
                raise ValueError(f"Retrieval index ID mismatch for {case.case_id!r}.")
            if set(prediction.retrieved_units) - units[case.repository_id]:
                raise ValueError(f"Unknown retrieved evidence for {case.case_id!r}.")
        review = review_map.pop((prediction.system, prediction.case_id), None)
        scored.append(_score_case(case, prediction, index, review))

    for system, attempted in seen.items():
        if attempted != expected_ids:
            raise ValueError(
                f"{system!r} is missing {len(expected_ids - attempted)} selected cases."
            )
    if review_map:
        raise ValueError("At least one independent review has no matching prediction.")
    return sorted(scored, key=lambda row: (row.prediction.system, row.prediction.case_id))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(rows: list[CaseScore]) -> dict:
    """Summarize a system or slice, preserving unavailable measurements as null."""
    answered = [row for row in rows if not row.prediction.abstained]
    abstained = len(rows) - len(answered)
    correct_abstentions = sum(not row.answerable and row.prediction.abstained for row in rows)
    answerable = sum(row.answerable for row in rows)
    unanswerable = len(rows) - answerable
    verdicts = [item.status for row in answered for item in row.verdicts]
    hits = sum(row.matched_facts for row in answered)
    known_reviews = [row.review for row in rows if row.review is not None]
    retrieval = [row.retrieval for row in rows if row.retrieval is not None]
    tokens_in = [p.input_tokens for row in rows if (p := row.prediction).input_tokens is not None]
    tokens_out = [
        p.output_tokens for row in rows if (p := row.prediction).output_tokens is not None
    ]
    steps = [p.steps for row in rows if (p := row.prediction).steps is not None]
    latencies = [p.latency_ms for row in rows if (p := row.prediction).latency_ms is not None]
    return {
        "cases": len(rows),
        "structural_answer_accuracy": _ratio(
            sum(row.structural_correct for row in rows), len(rows)
        ),
        "manual_answer_accuracy": _ratio(
            sum(review.correct for review in known_reviews), len(known_reviews)
        ),
        "manual_reviewed": len(known_reviews),
        "claim_precision": _ratio(hits, sum(row.structural_claims for row in answered)),
        "claim_recall": _ratio(hits, sum(row.required_facts for row in rows)),
        "unsupported_claim_rate": _ratio(
            verdicts.count(VerdictStatus.REFUTED) + verdicts.count(VerdictStatus.UNVERIFIABLE),
            len(verdicts),
        ),
        "refuted_claim_rate": _ratio(verdicts.count(VerdictStatus.REFUTED), len(verdicts)),
        "unknown_claim_rate": _ratio(verdicts.count(VerdictStatus.UNVERIFIABLE), len(verdicts)),
        "claim_extraction_coverage": _ratio(
            sum(bool(row.verdicts) for row in answered), len(answered)
        ),
        "coverage": _ratio(len(answered), len(rows)),
        "abstention_precision": _ratio(correct_abstentions, abstained),
        "abstention_recall": _ratio(correct_abstentions, unanswerable),
        "false_abstention_rate": _ratio(abstained - correct_abstentions, answerable),
        "retrieval_cases": len(retrieval),
        **{
            key: fmean(item[key] for item in retrieval) if retrieval else None
            for key in ("recall_at_5", "mrr_at_5", "ndcg_at_5")
        },
        "mean_input_tokens": fmean(tokens_in) if tokens_in else None,
        "mean_output_tokens": fmean(tokens_out) if tokens_out else None,
        "mean_steps": fmean(steps) if steps else None,
        "latency_p50_ms": median(latencies) if latencies else None,
        "latency_p95_ms": percentile(latencies, 95) if latencies else None,
    }
