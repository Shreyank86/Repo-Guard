"""Offline Transition and Episode Replay Collector for RepoGuard.

Owner: Member 3 (RL Environment, Training & Model Integration)

Generates environment transitions across benchmark questions and actions,
saving them to data/episodes/episode_log.jsonl for environment validation,
trace inspection, and operation-cache prewarming (Design Option C).
Note: Offline transitions are strictly NOT used to train policies directly;
both DQN and REINFORCE train via active online environment interaction.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from repoguard.agent.cache import OperationCache
from repoguard.agent.environment import RepoGuardEnv, ACTION_LOOKUP
from repoguard.evaluation.benchmark import load_benchmark
from repoguard.llm import LLMService
from repoguard.repository.indexer import load_index
from repoguard.schemas import IndexArtifacts

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("run_episodes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/questions/phase2_benchmark_v1.jsonl"),
        help="Path to benchmark JSONL",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
        help="Benchmark split to collect transitions for (strictly 'train' for prewarming training cache)",
    )
    parser.add_argument(
        "--indexes",
        type=Path,
        default=Path("data/indexes"),
        help="Path to repository indexes directory",
    )
    parser.add_argument(
        "--cache-db",
        type=Path,
        default=Path("data/cache/operations.db"),
        help="Path to persistent SQLite operation cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/episodes/episode_log.jsonl"),
        help="Output JSONL path for recorded episode transitions",
    )
    parser.add_argument(
        "--episodes-per-case",
        type=int,
        default=3,
        help="Number of diverse exploratory trajectories per benchmark case",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use MockLLM for instant generation",
    )

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading benchmark cases from %s (split: %s)", args.benchmark, args.split)
    all_cases = load_benchmark(args.benchmark)
    cases = [c for c in all_cases if getattr(c, "split", "") == args.split]

    assert len(cases) > 0, f"No cases found for split '{args.split}' in {args.benchmark}!"
    if args.split == "train":
        if len(all_cases) == 60:
            assert len(cases) == 36, f"Expected 36 train cases in benchmark v1, got {len(cases)}"
        assert all(c.split == "train" for c in cases), "All cases must belong to the train split!"
        assert not any(c.split in ("validation", "test") for c in cases), (
            "Data leakage: validation or test cases found in transition collection!"
        )
        logger.info("[VERIFIED] Offline transitions restricted strictly to %d train cases.", len(cases))

    repo_names = {c.repository_id for c in cases}
    logger.info("Loading indexes for %d repositories: %s", len(repo_names), sorted(repo_names))
    indexes: dict[str, IndexArtifacts] = {name: load_index(args.indexes / name) for name in repo_names}

    cache = OperationCache(db_path=args.cache_db)
    llm = LLMService(mock_mode=args.mock)

    env = RepoGuardEnv(
        benchmark_cases=cases,
        indexes=indexes,
        llm_service=llm,
        cache=cache,
    )

    total_transitions = 0
    with args.output.open("w", encoding="utf-8") as out_f:
        for case_idx, case in enumerate(cases):
            for traj in range(args.episodes_per_case):
                obs, info = env.reset(case_index=case_idx)
                done = False

                while not done:
                    valid_mask = env.valid_action_mask()
                    valid_actions = [i for i, v in enumerate(valid_mask) if v]
                    action = valid_actions[traj % len(valid_actions)]

                    next_obs, reward, done, _, step_info = env.step(action)

                    record = {
                        "case_id": case.case_id,
                        "repository_id": case.repository_id,
                        "risk_lambda": info["risk_lambda"],
                        "state": obs.tolist(),
                        "action": action,
                        "action_name": ACTION_LOOKUP[action].value,
                        "reward": reward,
                        "next_state": next_obs.tolist(),
                        "done": done,
                        "valid_mask": valid_mask,
                        "tokens_spent": step_info.get("tokens_spent", 0),
                    }
                    out_f.write(json.dumps(record) + "\n")
                    total_transitions += 1
                    obs = next_obs

    logger.info("Successfully recorded %d transitions to %s", total_transitions, args.output)


if __name__ == "__main__":
    main()
