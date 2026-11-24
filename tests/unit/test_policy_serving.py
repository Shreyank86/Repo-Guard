"""Unit tests for RepoGuardPolicy in src/repoguard/agent/policy.py."""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from repoguard.agent.dqn import DQNAgent
from repoguard.agent.policy import RepoGuardPolicy
from repoguard.schemas import RLAction, StateVector


@pytest.fixture
def trained_checkpoint():
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "test_dqn_policy.pt"
        agent = DQNAgent(state_dim=18, action_dim=6)
        agent.save(ckpt_path, metadata={"algo": "dqn", "episodes": 10})
        yield ckpt_path


def test_policy_load_and_action_selection(trained_checkpoint: Path):
    policy = RepoGuardPolicy(trained_checkpoint)
    assert policy.algo == "dqn"

    state_vec = StateVector(risk_lambda=1.0)
    action = policy.select_action(state_vec)
    assert isinstance(action, RLAction)
    assert action in list(RLAction)


def test_policy_respects_valid_action_mask(trained_checkpoint: Path):
    policy = RepoGuardPolicy(trained_checkpoint)
    state = np.zeros(18, dtype=np.float32)

    # Allow only ABSTAIN (action 5)
    strict_mask = [False, False, False, False, False, True]
    action = policy.select_action(state, valid_mask=strict_mask)
    assert action == RLAction.ABSTAIN


def test_policy_action_probabilities(trained_checkpoint: Path):
    policy = RepoGuardPolicy(trained_checkpoint)
    state = np.zeros(18, dtype=np.float32)
    probs = policy.get_action_probabilities(state)

    assert len(probs) == 6
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-2
