from repoguard.evaluation.metrics import ScoredAnswer, baseline_metrics, verification_metrics
from repoguard.verification import Claim, ClaimVerdict, VerdictStatus, VerificationResult


def test_precision_and_recall_use_independent_labels() -> None:
    scores = verification_metrics(
        [
            (True, VerdictStatus.VERIFIED),
            (True, VerdictStatus.UNVERIFIABLE),
            (False, VerdictStatus.VERIFIED),
            (False, VerdictStatus.REFUTED),
        ]
    )

    assert scores == {
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "true_negatives": 1,
        "verification_precision": 0.5,
        "verification_recall": 0.5,
    }


def test_refutation_is_counted_separately_from_uncertainty_and_no_claims() -> None:
    empty = VerificationResult(verdict="unsupported", claims=[], unsupported_names=[])
    claim = Claim(symbol="Job", file="job.py")
    unknown = empty.model_copy(
        update={
            "claims": [
                ClaimVerdict(claim=claim, status=VerdictStatus.UNVERIFIABLE, reason="Not indexed")
            ]
        }
    )
    refuted = unknown.model_copy(
        update={
            "claims": [ClaimVerdict(claim=claim, status=VerdictStatus.REFUTED, reason="Wrong file")]
        }
    )
    scores = baseline_metrics(
        [
            ScoredAnswer(is_trap=True, abstained=True, verification=empty),
            ScoredAnswer(is_trap=False, abstained=False, verification=empty),
            ScoredAnswer(is_trap=False, abstained=False, verification=refuted),
            ScoredAnswer(is_trap=True, abstained=False, verification=unknown),
        ]
    )

    assert scores["coverage"] == 0.75
    assert scores["abstention_accuracy"] == 0.75
    assert scores["claim_extraction_coverage"] == 2 / 3
    assert scores["structural_hallucination_rate"] == 0.25
    assert scores["answers_with_unverifiable_claims"] == 1
    assert scores["answers_with_all_claims_verified"] == 0
