"""Verification metrics for independently labelled claims and baseline replays."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from repoguard.verification.models import VerdictStatus, VerificationResult


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def verification_metrics(
    labels: Iterable[tuple[bool, VerdictStatus]],
) -> dict[str, int | float | None]:
    """Treat only VERIFIED as a positive prediction against the human label."""
    outcomes = Counter(
        (supported, verdict == VerdictStatus.VERIFIED) for supported, verdict in labels
    )
    tp = outcomes[True, True]
    fp = outcomes[False, True]
    fn = outcomes[True, False]
    tn = outcomes[False, False]
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "verification_precision": _rate(tp, tp + fp),
        "verification_recall": _rate(tp, tp + fn),
    }


@dataclass(frozen=True)
class ScoredAnswer:
    is_trap: bool
    abstained: bool
    verification: VerificationResult


def baseline_metrics(answers: Iterable[ScoredAnswer]) -> dict[str, int | float | None]:
    """Measure structural evidence and abstention for a single baseline."""
    counts: Counter[str] = Counter()
    for answer in answers:
        counts["runs"] += 1
        counts["traps"] += answer.is_trap
        counts["correct_decisions"] += answer.abstained == answer.is_trap
        if answer.abstained:
            counts["trap_abstentions"] += answer.is_trap
            counts["answerable_abstentions"] += not answer.is_trap
            continue
        counts["answered"] += 1
        statuses = {item.status for item in answer.verification.claims}
        counts["answers_with_claims"] += bool(statuses)
        counts["answers_with_all_claims_verified"] += statuses == {VerdictStatus.VERIFIED}
        counts["answers_with_refuted_claims"] += VerdictStatus.REFUTED in statuses
        counts["answers_with_unverifiable_claims"] += VerdictStatus.UNVERIFIABLE in statuses

    runs, traps, answered = counts["runs"], counts["traps"], counts["answered"]
    return {
        "runs": runs,
        "answered": answered,
        "answers_with_claims": counts["answers_with_claims"],
        "answers_with_all_claims_verified": counts["answers_with_all_claims_verified"],
        "answers_with_refuted_claims": counts["answers_with_refuted_claims"],
        "answers_with_unverifiable_claims": counts["answers_with_unverifiable_claims"],
        "coverage": _rate(answered, runs),
        "claim_extraction_coverage": _rate(counts["answers_with_claims"], answered),
        "structural_hallucination_rate": _rate(counts["answers_with_refuted_claims"], runs),
        "abstention_accuracy": _rate(counts["correct_decisions"], runs),
        "correct_abstention_rate": _rate(counts["trap_abstentions"], traps),
        "over_refusal_rate": _rate(counts["answerable_abstentions"], runs - traps),
    }
