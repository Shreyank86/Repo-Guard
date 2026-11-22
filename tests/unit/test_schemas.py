import pytest
from repoguard.schemas import RLAction, StateVector, GenerationResult


def test_rl_action_enum_has_six_actions() -> None:
    actions = list(RLAction)
    assert len(actions) == 6
    assert [a.value for a in actions] == [
        "search",
        "inspect",
        "verify",
        "answer",
        "answer_with_evidence",
        "abstain",
    ]


def test_state_vector_has_eighteen_dimensions() -> None:
    vec = StateVector()
    tensor = vec.to_tensor()
    assert len(tensor) == 18
    assert all(isinstance(val, float) for val in tensor)
    assert vec.risk_lambda == 1.0


def test_generation_result_uncertainty_fields() -> None:
    res = GenerationResult(
        text="The Job class is defined in schedule/__init__.py.",
        entropy=0.35,
        min_token_prob=0.88,
    )
    assert res.entropy == 0.35
    assert res.min_token_prob == 0.88
