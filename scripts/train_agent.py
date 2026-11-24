"""Training Runner for RepoGuard RL Agents (DQN and REINFORCE).

Owner: Member 3 (RL Environment, Training & Model Integration)

Supports:
- Local Windows execution on CPU/GPU
- Cloud execution on Google Colab GPU
- Risk conditioning with dynamic lambda ~ LogUniform(0.25, 5.0)
- Checkpoint saving and recovery
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

# Add src to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from repoguard.agent.cache import OperationCache
from repoguard.agent.dqn import DQNAgent
from repoguard.agent.environment import RepoGuardEnv, ACTION_LOOKUP
from repoguard.agent.reinforce import REINFORCEAgent
from repoguard.evaluation.benchmark import load_benchmark
from repoguard.llm import LLMService
from repoguard.repository.indexer import load_index
from repoguard.schemas import IndexArtifacts

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("train_agent")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algo",
        choices=["dqn", "reinforce"],
        default="dqn",
        help="RL algorithm to train",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data/questions/phase2_benchmark_v1.jsonl",
        help="Benchmark questions JSONL",
    )
    parser.add_argument(
        "--indexes",
        type=Path,
        default=ROOT / "data/indexes",
        help="Repository indexes directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory to save checkpoints (defaults to data/checkpoints/<algo>/)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for DQN updates",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--mock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use MockLLM for ultra-fast training execution (use --no-mock for live LLM)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print metrics every N episodes",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    ckpt_dir = args.checkpoint_dir or (ROOT / "data/checkpoints" / args.algo)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting %s training on %s (seed: %d)", args.algo.upper(), args.device, args.seed)
    logger.info("Loading benchmark cases from %s", args.benchmark)
    cases = load_benchmark(args.benchmark)

    # Filter to train cases if annotated
    train_cases = [c for c in cases if getattr(c, "split", "train") in ("train", "all")]
    if not train_cases:
        train_cases = cases

    repo_names = {c.repository_id for c in train_cases}
    indexes: dict[str, IndexArtifacts] = {name: load_index(args.indexes / name) for name in repo_names}

    cache = OperationCache()
    llm = LLMService(mock_mode=args.mock)

    env = RepoGuardEnv(
        benchmark_cases=train_cases,
        indexes=indexes,
        llm_service=llm,
        cache=cache,
    )

    # Initialize Agent
    if args.algo == "dqn":
        agent = DQNAgent(
            state_dim=18,
            action_dim=6,
            learning_rate=args.learning_rate,
            device=args.device,
        )
    else:
        agent = REINFORCEAgent(
            state_dim=18,
            action_dim=6,
            learning_rate=args.learning_rate,
            device=args.device,
        )

    # Exploration schedule
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    history: list[dict[str, Any]] = []
    recent_rewards: list[float] = []
    recent_lengths: list[int] = []

    start_time = time.perf_counter()

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        episode_reward = 0.0
        steps = 0
        last_loss = 0.0

        while not done:
            mask = env.valid_action_mask()
            if args.algo == "dqn":
                action = agent.select_action(obs, epsilon=epsilon, valid_mask=mask)
            else:
                action = agent.select_action(obs, valid_mask=mask)

            next_obs, reward, done, _, step_info = env.step(action)
            episode_reward += reward
            steps += 1

            if args.algo == "dqn":
                next_mask = env.valid_action_mask()
                agent.replay_buffer.push(obs, action, reward, next_obs, done, next_mask)
                loss = agent.update(batch_size=args.batch_size)
                if loss is not None:
                    last_loss = loss
            else:
                agent.store_reward(reward)

            obs = next_obs

        # REINFORCE policy update at episode end
        if args.algo == "reinforce":
            loss = agent.update()
            if loss is not None:
                last_loss = loss

        # Epsilon decay & target sync for DQN
        if args.algo == "dqn":
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            if episode % 50 == 0:
                agent.sync_target_network()

        recent_rewards.append(episode_reward)
        recent_lengths.append(steps)

        record = {
            "episode": episode,
            "reward": round(episode_reward, 4),
            "steps": steps,
            "lambda": info["risk_lambda"],
            "loss": round(last_loss, 4),
            "epsilon": round(epsilon, 4) if args.algo == "dqn" else None,
        }
        history.append(record)

        if episode % args.log_interval == 0 or episode == args.episodes:
            mean_r = sum(recent_rewards[-args.log_interval:]) / len(recent_rewards[-args.log_interval:])
            mean_l = sum(recent_lengths[-args.log_interval:]) / len(recent_lengths[-args.log_interval:])
            logger.info(
                "Episode %4d/%d | Mean Reward: %6.3f | Mean Steps: %4.2f | Loss: %6.4f | Epsilon: %5.3f",
                episode,
                args.episodes,
                mean_r,
                mean_l,
                last_loss,
                epsilon if args.algo == "dqn" else 0.0,
            )

        # Checkpoint every 50 episodes
        if episode % 50 == 0 or episode == args.episodes:
            ckpt_path = ckpt_dir / f"{args.algo}_checkpoint_ep{episode}.pt"
            agent.save(
                ckpt_path,
                metadata={
                    "algo": args.algo,
                    "episode": episode,
                    "seed": args.seed,
                    "recent_mean_reward": round(mean_r, 4),
                    "training_time_sec": round(time.perf_counter() - start_time, 2),
                },
            )

    # Save final policy
    final_path = ckpt_dir / f"{args.algo}_policy_latest.pt"
    agent.save(
        final_path,
        metadata={
            "algo": args.algo,
            "total_episodes": args.episodes,
            "final_reward": round(recent_rewards[-1], 4),
            "training_time_sec": round(time.perf_counter() - start_time, 2),
        },
    )
    logger.info("Saved latest policy checkpoint to %s", final_path)

    # Save training metrics history
    metrics_file = ckpt_dir / f"{args.algo}_training_history.json"
    metrics_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info("Saved training history to %s", metrics_file)


if __name__ == "__main__":
    train()
