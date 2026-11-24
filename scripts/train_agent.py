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
import re
import sys
import time
from pathlib import Path
from typing import Any

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
        "--split",
        choices=["train", "validation", "test"],
        default="train",
        help="Benchmark split to train on (strictly 'train' for training)",
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
        "--cache-db",
        type=Path,
        default=ROOT / "data/cache/operations.db",
        help="Path to SQLite operation cache file",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to specific checkpoint to resume training from",
    )
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically resume from latest checkpoint in checkpoint-dir if present",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Save checkpoint every N episodes",
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
        "--replay-capacity",
        type=int,
        default=20000,
        help="Replay buffer capacity for DQN",
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

    logger.info("Starting %s training on %s (seed: %d, split: %s)", args.algo.upper(), args.device, args.seed, args.split)
    logger.info("Loading benchmark cases from %s", args.benchmark)
    cases = load_benchmark(args.benchmark)

    # Strictly enforce split boundary
    train_cases = [c for c in cases if getattr(c, "split", "") == args.split]
    assert len(train_cases) > 0, f"No cases found for split '{args.split}' in {args.benchmark}!"

    if args.split == "train":
        if len(cases) == 60:
            assert len(train_cases) == 36, f"Expected exactly 36 train cases in benchmark v1, got {len(train_cases)}"
        # Strict boundary assertion: verify 0 validation or test cases are in train_cases
        assert all(c.split == "train" for c in train_cases), "Non-train cases found in training split!"
        assert not any(c.split in ("validation", "test") for c in train_cases), (
            "Data leakage: validation or test cases found in training split!"
        )
        logger.info("[VERIFIED] Training strictly restricted to %d train cases (0 val/test leakage).", len(train_cases))
    else:
        logger.warning("Training on non-train split '%s' (%d cases).", args.split, len(train_cases))

    repo_names = {c.repository_id for c in train_cases}
    indexes: dict[str, IndexArtifacts] = {name: load_index(args.indexes / name) for name in repo_names}

    cache = OperationCache(db_path=args.cache_db)
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
            buffer_capacity=args.replay_capacity,
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
    start_episode = 1

    history: list[dict[str, Any]] = []
    recent_rewards: list[float] = []
    recent_lengths: list[int] = []

    # Auto-resume or specific checkpoint resume
    resume_path = args.resume
    if not resume_path and args.auto_resume:
        ep_ckpts = list(ckpt_dir.glob(f"{args.algo}_checkpoint_ep*.pt"))
        if ep_ckpts:
            def _extract_ep(p: Path) -> int:
                m = re.search(r"ep(\d+)", p.name)
                return int(m.group(1)) if m else 0
            ep_ckpts.sort(key=_extract_ep)
            resume_path = ep_ckpts[-1]
        elif (ckpt_dir / f"{args.algo}_policy_latest.pt").exists():
            resume_path = ckpt_dir / f"{args.algo}_policy_latest.pt"

    if resume_path and Path(resume_path).exists():
        logger.info("Resuming %s from checkpoint: %s", args.algo.upper(), resume_path)
        meta = agent.load(resume_path, restore_rng=True)
        last_ep = meta.get("episode", 0)
        start_episode = last_ep + 1
        if "epsilon" in meta and meta["epsilon"] is not None and args.algo == "dqn":
            epsilon = float(meta["epsilon"])

        metrics_file = ckpt_dir / f"{args.algo}_training_history.json"
        if metrics_file.exists():
            try:
                history = json.loads(metrics_file.read_text(encoding="utf-8"))
                for h in history[-50:]:
                    if "reward" in h:
                        recent_rewards.append(float(h["reward"]))
                    if "steps" in h:
                        recent_lengths.append(int(h["steps"]))
            except Exception as e:
                logger.warning("Could not parse existing training history: %s", e)

        buffer_info = f", replay buffer: {len(agent.replay_buffer)}" if args.algo == "dqn" else ""
        logger.info(
            "Resumed state: starting at episode %d/%d (epsilon: %.4f, history records: %d%s)",
            start_episode,
            args.episodes,
            epsilon,
            len(history),
            buffer_info,
        )

    if start_episode > args.episodes:
        logger.info("Target episodes (%d) already reached (checkpoint has %d). Training complete.", args.episodes, start_episode - 1)
        return

    start_time = time.perf_counter()

    for episode in range(start_episode, args.episodes + 1):
        try:
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
                mean_r = sum(recent_rewards[-args.log_interval:]) / max(1, len(recent_rewards[-args.log_interval:]))
                mean_l = sum(recent_lengths[-args.log_interval:]) / max(1, len(recent_lengths[-args.log_interval:]))
                logger.info(
                    "Episode %4d/%d | Mean Reward: %6.3f | Mean Steps: %4.2f | Loss: %6.4f | Epsilon: %5.3f",
                    episode,
                    args.episodes,
                    mean_r,
                    mean_l,
                    last_loss,
                    epsilon if args.algo == "dqn" else 0.0,
                )

            # Atomic checkpoint every save_interval episodes
            if episode % args.save_interval == 0 or episode == args.episodes:
                ckpt_path = ckpt_dir / f"{args.algo}_checkpoint_ep{episode}.pt"
                latest_path = ckpt_dir / f"{args.algo}_policy_latest.pt"
                mean_r = sum(recent_rewards[-args.log_interval:]) / max(1, len(recent_rewards[-args.log_interval:]))
                meta_payload = {
                    "algo": args.algo,
                    "episode": episode,
                    "target_episodes": args.episodes,
                    "seed": args.seed,
                    "split": args.split,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size if args.algo == "dqn" else None,
                    "epsilon": epsilon if args.algo == "dqn" else None,
                    "recent_mean_reward": round(mean_r, 4),
                    "training_time_sec": round(time.perf_counter() - start_time, 2),
                }
                agent.save(ckpt_path, metadata=meta_payload)
                agent.save(latest_path, metadata=meta_payload)

                # Persist metrics JSON atomically
                metrics_file = ckpt_dir / f"{args.algo}_training_history.json"
                tmp_metrics = metrics_file.with_suffix(".json.tmp")
                tmp_metrics.write_text(json.dumps(history, indent=2), encoding="utf-8")
                tmp_metrics.replace(metrics_file)

        except Exception as exc:
            logger.error("Episode %d failed with error: %s. Skipping to next episode...", episode, exc)
            continue

    logger.info("Training successfully finished up to %d episodes.", args.episodes)


if __name__ == "__main__":
    train()
