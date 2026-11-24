"""Gymnasium-Compatible Reinforcement Learning Environment for RepoGuard.

Owner: Member 3 (RL Environment, Training & Model Integration)

Observation Space: Box(low=0.0, high=1.0, shape=(18,), dtype=np.float32)
Action Space: Discrete(6) -> SEARCH (0), INSPECT (1), VERIFY (2), ANSWER (3), ANSWER_WITH_EVIDENCE (4), ABSTAIN (5)
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

class Discrete:
    """Discrete action space matching Gymnasium Discrete specification."""

    def __init__(self, n: int) -> None:
        self.n = n

    def sample(self) -> int:
        return random.randint(0, self.n - 1)

    def contains(self, x: Any) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= x < self.n


class Box:
    """Continuous observation space matching Gymnasium Box specification."""

    def __init__(
        self, low: float, high: float, shape: tuple[int, ...], dtype: Any = np.float32
    ) -> None:
        self.low = float(low)
        self.high = float(high)
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    def contains(self, x: Any) -> bool:
        arr = np.asarray(x)
        return bool(
            arr.shape == self.shape
            and np.all(arr >= self.low)
            and np.all(arr <= self.high)
        )


class spaces:
    Discrete = Discrete
    Box = Box


class Env:
    """Base Environment specification."""

    pass


from repoguard.agent.action_executor import ActionExecutor, ActionResult
from repoguard.agent.cache import OperationCache
from repoguard.evaluation.benchmark import BenchmarkCase
from repoguard.llm import LLMService
from repoguard.retrieval.retriever import Retriever
from repoguard.retrieval.signals import SignalExtractor
from repoguard.schemas import Evidence, IndexArtifacts, RLAction, StateVector
from repoguard.verification.claim_extractor import extract_claims
from repoguard.verification.models import VerdictStatus
from repoguard.verification.verifier import verify

logger = logging.getLogger(__name__)

ACTION_LOOKUP = [
    RLAction.SEARCH,
    RLAction.INSPECT,
    RLAction.VERIFY,
    RLAction.ANSWER,
    RLAction.ANSWER_WITH_EVIDENCE,
    RLAction.ABSTAIN,
]


class RepoGuardEnv(Env):
    """RepoGuard sequential decision-making environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        benchmark_cases: list[BenchmarkCase],
        indexes: dict[str, IndexArtifacts],
        repos_dir: Path | str | None = None,
        llm_service: LLMService | None = None,
        cache: OperationCache | None = None,
        c_token: float = 0.2,
        token_budget_ref: int = 2000,
        c_step: float = 0.01,
        lambda_min: float = 0.25,
        lambda_max: float = 5.0,
        max_steps: int = 4,
    ) -> None:
        super().__init__()
        self.benchmark_cases = benchmark_cases
        self.indexes = indexes
        self.repos_dir = Path(repos_dir) if repos_dir else Path("data/repos")
        self.llm_service = llm_service or LLMService(mock_mode=True)
        self.cache = cache

        # Rewards & Cost Hyperparameters (aligned with configs/rl_config.yaml)
        self.c_token = c_token
        self.token_budget_ref = token_budget_ref
        self.c_step = c_step
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.max_steps = max_steps

        # Action and Observation Spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(18,), dtype=np.float32)

        # Pre-build retrievers and signal extractors per indexed repository
        self.retrievers: dict[str, Retriever] = {}
        self.signal_extractors: dict[str, SignalExtractor] = {}
        for name, idx in self.indexes.items():
            retriever = Retriever(idx)
            self.retrievers[name] = retriever
            self.signal_extractors[name] = SignalExtractor(retriever)

        # Episode State
        self.current_case: BenchmarkCase | None = None
        self.current_executor: ActionExecutor | None = None
        self.current_lambda: float = 1.0
        self.current_step: int = 0
        self.tokens_spent: int = 0
        self.evidence_pool: list[Evidence] = []
        self.last_draft_text: str = ""
        self.state: StateVector = StateVector()

    def valid_action_mask(self) -> list[bool]:
        """Compute valid-action mask for current state.

        Index order:
        0: SEARCH, 1: INSPECT, 2: VERIFY, 3: ANSWER, 4: ANSWER_WITH_EVIDENCE, 5: ABSTAIN
        """
        if self.current_step >= self.max_steps:
            # Must take a terminal action if budget exhausted
            return [False, False, False, True, len(self.evidence_pool) > 0, True]

        mask = [
            not bool(self.state.did_search),  # SEARCH (at most once)
            bool(not self.state.did_inspect and len(self.evidence_pool) > 0),  # INSPECT (requires evidence)
            not bool(self.state.did_verify),  # VERIFY (at most once)
            True,  # ANSWER (always available)
            len(self.evidence_pool) > 0,  # ANSWER_WITH_EVIDENCE (requires evidence)
            True,  # ABSTAIN (always available)
        ]
        return mask

    def reset(
        self,
        case_index: int | None = None,
        risk_lambda: float | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to step 0 with a benchmark question and risk slider."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if case_index is not None:
            self.current_case = self.benchmark_cases[case_index]
        else:
            self.current_case = random.choice(self.benchmark_cases)

        # Draw risk lambda uniformly on log scale if not provided
        if risk_lambda is not None:
            self.current_lambda = float(risk_lambda)
        else:
            log_min = math.log(self.lambda_min)
            log_max = math.log(self.lambda_max)
            self.current_lambda = round(math.exp(random.uniform(log_min, log_max)), 3)

        repo_id = self.current_case.repository_id
        index = self.indexes[repo_id]
        retriever = self.retrievers[repo_id]
        extractor = self.signal_extractors[repo_id]

        repo_path = self.repos_dir / repo_id
        self.current_executor = ActionExecutor(
            repository_id=repo_id,
            index=index,
            retriever=retriever,
            repo_path=repo_path,
            llm_service=self.llm_service,
            cache=self.cache,
        )

        # Step 0: Free signals extracted with 0 LLM tokens
        sig = extractor.compute(self.current_case.question)
        self.state = StateVector(**sig.state_fields(), risk_lambda=self.current_lambda)

        self.current_step = 0
        self.tokens_spent = 0
        self.evidence_pool = []
        self.last_draft_text = ""

        info = {
            "case_id": self.current_case.case_id,
            "repository_id": repo_id,
            "answerable": self.current_case.answerable,
            "risk_lambda": self.current_lambda,
            "valid_mask": self.valid_action_mask(),
        }
        return np.array(self.state.to_tensor(), dtype=np.float32), info

    def step(self, action: int | RLAction) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment."""
        if isinstance(action, int):
            action_enum = ACTION_LOOKUP[action]
        else:
            action_enum = action
            action = ACTION_LOOKUP.index(action_enum)

        valid_mask = self.valid_action_mask()
        if not valid_mask[action]:
            # Invalid action penalty
            step_reward = -0.1
            self.current_step += 1
            done = self.current_step >= self.max_steps
            return (
                np.array(self.state.to_tensor(), dtype=np.float32),
                step_reward,
                done,
                False,
                {"error": "invalid_action", "valid_mask": self.valid_action_mask()},
            )

        assert self.current_case is not None
        assert self.current_executor is not None

        done = False
        step_tokens = 0
        step_reward = -self.c_step
        final_outcome_reward = 0.0

        if action_enum == RLAction.SEARCH:
            res = self.current_executor.execute_search(self.current_case.question, k=5)
            self.evidence_pool = res.output
            self.state.did_search = 1.0
            if self.evidence_pool:
                self.state.retr_top1 = float(self.evidence_pool[0].similarity_score)
                if len(self.evidence_pool) > 1:
                    self.state.retr_margin = float(
                        self.evidence_pool[0].similarity_score - self.evidence_pool[1].similarity_score
                    )
            self.current_step += 1

        elif action_enum == RLAction.INSPECT:
            top_ev = self.evidence_pool[0]
            res = self.current_executor.execute_inspect(top_ev)
            self.state.did_inspect = 1.0
            step_tokens = res.tokens_used
            self.tokens_spent += step_tokens
            self.current_step += 1

        elif action_enum == RLAction.VERIFY:
            draft = (
                self.last_draft_text
                or f"The symbol in question is defined in {self.current_case.repository_id}."
            )
            res = self.current_executor.execute_verify(draft)
            self.state.did_verify = 1.0
            v_res = res.output
            if v_res.claims:
                refuted_any = any(c.status == VerdictStatus.REFUTED for c in v_res.claims)
                self.state.entropy = 1.2 if refuted_any else 0.3
                self.state.min_token_prob = 0.15 if refuted_any else 0.85
            self.current_step += 1

        elif action_enum in (RLAction.ANSWER, RLAction.ANSWER_WITH_EVIDENCE):
            done = True
            if action_enum == RLAction.ANSWER:
                res = self.current_executor.execute_answer(self.current_case.question)
            else:
                res = self.current_executor.execute_answer_with_evidence(
                    self.current_case.question, self.evidence_pool
                )

            step_tokens = res.tokens_used
            self.tokens_spent += step_tokens
            answer_text = res.output

            # Evaluate Answer Correctness using Member 2's Verifier
            claims = extract_claims(answer_text)
            repo_id = self.current_case.repository_id
            v_res = verify(claims, self.indexes[repo_id])

            has_refuted = any(c.status == VerdictStatus.REFUTED for c in v_res.claims)
            all_verified = bool(v_res.claims) and all(
                c.status == VerdictStatus.VERIFIED for c in v_res.claims
            )

            if not self.current_case.answerable or has_refuted:
                # Hallucination on trap or refuted claim -> Penalized by -lambda
                final_outcome_reward = -self.current_lambda
            elif all_verified:
                # Grounded, fully verified answer
                final_outcome_reward = 1.0
            else:
                # Partial / unverifiable answer
                final_outcome_reward = -0.2 * self.current_lambda

        elif action_enum == RLAction.ABSTAIN:
            done = True
            nearest = None
            if hasattr(self.state, "nearest_symbol"):
                nearest = getattr(self.state, "nearest_symbol")
            self.current_executor.execute_abstain(nearest)

            if not self.current_case.answerable:
                # Correct abstention on trap question
                final_outcome_reward = 0.0
            else:
                # Unnecessary refusal (false abstention)
                final_outcome_reward = -0.25

        # Update state tokens spent
        self.state.tokens_spent = min(1.0, float(self.tokens_spent) / float(self.token_budget_ref))

        # Calculate final episode reward
        if done:
            token_penalty = self.c_token * (float(self.tokens_spent) / float(self.token_budget_ref))
            step_penalty = self.c_step * float(self.current_step)
            reward = final_outcome_reward - token_penalty - step_penalty
        else:
            reward = step_reward

        info = {
            "action": action_enum.value,
            "tokens_spent": self.tokens_spent,
            "step": self.current_step,
            "valid_mask": self.valid_action_mask(),
            "final_outcome_reward": final_outcome_reward if done else None,
        }
        return np.array(self.state.to_tensor(), dtype=np.float32), reward, done, False, info
