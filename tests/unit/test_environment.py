"""Unit tests for RepoGuardEnv and ActionExecutor."""

from pathlib import Path
import pytest
import numpy as np

from repoguard.agent.environment import RepoGuardEnv, ACTION_LOOKUP
from repoguard.evaluation.benchmark import load_benchmark
from repoguard.repository.indexer import load_index
from repoguard.schemas import RLAction, StateVector


@pytest.fixture(scope="module")
def benchmark_data():
    bench_file = Path("data/questions/phase2_benchmark_v1.jsonl")
    indexes_dir = Path("data/indexes")
    cases = load_benchmark(bench_file)
    # Load all distinct indexes present in benchmark
    repo_names = {c.repository_id for c in cases}
    indexes = {name: load_index(indexes_dir / name) for name in repo_names}
    return cases, indexes


@pytest.fixture
def env(benchmark_data):
    cases, indexes = benchmark_data
    return RepoGuardEnv(
        benchmark_cases=cases,
        indexes=indexes,
        c_token=0.2,
        token_budget_ref=2000,
        c_step=0.01,
        max_steps=4,
    )


def test_env_observation_and_action_spaces(env: RepoGuardEnv):
    assert env.observation_space.shape == (18,)
    assert env.action_space.n == 6


def test_env_reset_returns_valid_state_and_mask(env: RepoGuardEnv):
    obs, info = env.reset(case_index=0, risk_lambda=2.0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (18,)
    assert info["risk_lambda"] == 2.0
    assert "valid_mask" in info
    assert len(info["valid_mask"]) == 6
    # Initially, cannot INSPECT or ANSWER_WITH_EVIDENCE because evidence pool is empty
    assert info["valid_mask"][1] is False  # INSPECT
    assert info["valid_mask"][4] is False  # ANSWER_WITH_EVIDENCE
    assert info["valid_mask"][0] is True   # SEARCH
    assert info["valid_mask"][3] is True   # ANSWER
    assert info["valid_mask"][5] is True   # ABSTAIN


def test_search_action_populates_evidence_and_enables_inspect(env: RepoGuardEnv):
    obs, info = env.reset(case_index=0)
    next_obs, reward, done, truncated, step_info = env.step(0)  # SEARCH

    assert done is False
    assert reward == -0.01  # -c_step
    # Check that did_search flag is set (dim index 12 in StateVector)
    assert env.state.did_search == 1.0

    mask = step_info["valid_mask"]
    assert mask[0] is False  # cannot search again
    if len(env.evidence_pool) > 0:
        assert mask[1] is True  # INSPECT now valid
        assert mask[4] is True  # ANSWER_WITH_EVIDENCE now valid


def test_abstain_on_trap_yields_correct_abstention_reward(env: RepoGuardEnv):
    # Find an unanswerable / trap case
    trap_idx = next(i for i, c in enumerate(env.benchmark_cases) if not c.answerable)
    obs, info = env.reset(case_index=trap_idx, risk_lambda=3.0)

    next_obs, reward, done, truncated, step_info = env.step(5)  # ABSTAIN
    assert done is True
    # Correct abstention: R_final = 0.0, step_penalty = 0.0, token_penalty = 0.0
    assert reward == 0.0
    assert step_info["final_outcome_reward"] == 0.0


def test_invalid_action_returns_penalty(env: RepoGuardEnv):
    env.reset(case_index=0)
    # Attempt INSPECT (action 1) when evidence pool is empty -> invalid action
    obs, reward, done, truncated, info = env.step(1)
    assert reward == -0.1
    assert info.get("error") == "invalid_action"
