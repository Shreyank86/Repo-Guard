"""Evaluation Runner for Trained RepoGuard RL Policies.

Owner: Member 3 (RL Environment, Training & Model Integration)

Evaluates trained policy across risk levels (lambda in [0.25, 5.0]), splits, and metrics:
- Coverage (%)
- Hallucination Rate (%)
- Accuracy on Answered (%)
- Correct Abstention Rate (%)
- Mean Tokens & Steps per Query
Saves comparative evaluation table to reports/rl_evaluation_results.csv.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from repoguard.agent.cache import OperationCache
from repoguard.agent.environment import RepoGuardEnv, ACTION_LOOKUP
from repoguard.agent.policy import RepoGuardPolicy
from repoguard.evaluation.benchmark import load_benchmark
from repoguard.llm import LLMService
from repoguard.repository.indexer import load_index
from repoguard.schemas import IndexArtifacts, RLAction

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("evaluate_rl")

LAMBDA_LEVELS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]


def evaluate() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="Path to trained policy checkpoint (.pt)",
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
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
        help="Benchmark split to evaluate on (validation for tuning, test for final held-out)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "reports/rl_evaluation_results.csv",
        help="Path to output summary CSV",
    )
    parser.add_argument(
        "--cache-db",
        type=Path,
        default=ROOT / "data/cache/operations.db",
        help="Path to SQLite operation cache file",
    )
    parser.add_argument(
        "--mock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use MockLLM for instant evaluation (use --no-mock for live LLM)",
    )

    args = parser.parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading policy checkpoint from %s", args.policy)
    policy = RepoGuardPolicy(args.policy)

    logger.info("Loading benchmark from %s (split: %s)", args.benchmark, args.split)
    all_cases = load_benchmark(args.benchmark)
    cases = [c for c in all_cases if getattr(c, "split", "") == args.split]

    assert len(cases) > 0, f"No cases found for split '{args.split}' in {args.benchmark}!"
    if len(all_cases) == 60:
        if args.split in ("validation", "test"):
            assert len(cases) == 12, f"Expected 12 cases for split '{args.split}', got {len(cases)}"
        elif args.split == "train":
            assert len(cases) == 36, f"Expected 36 cases for split 'train', got {len(cases)}"
    assert all(c.split == args.split for c in cases), f"All cases must belong to split '{args.split}'!"
    logger.info("[VERIFIED] Evaluating exclusively on %d %s cases (isolated split).", len(cases), args.split)

    repo_names = {c.repository_id for c in cases}
    indexes: dict[str, IndexArtifacts] = {name: load_index(args.indexes / name) for name in repo_names}

    cache = OperationCache(db_path=args.cache_db)
    llm = LLMService(mock_mode=args.mock)
    env = RepoGuardEnv(benchmark_cases=cases, indexes=indexes, llm_service=llm, cache=cache)

    results_rows = []

    for lam in LAMBDA_LEVELS:
        total = len(cases)
        answered = 0
        correct_answers = 0
        hallucinations = 0
        correct_abstentions = 0
        total_traps = sum(1 for c in cases if not c.answerable)
        tokens_list = []
        steps_list = []
        rewards_list = []
        latencies_list = []

        for idx, case in enumerate(cases):
            t0 = time.perf_counter()
            obs, info = env.reset(case_index=idx, risk_lambda=lam)
            done = False
            ep_steps = 0
            ep_tokens = 0
            ep_reward = 0.0
            final_action = None

            while not done:
                mask = env.valid_action_mask()
                action_enum = policy.select_action(obs, valid_mask=mask)
                action_idx = ACTION_LOOKUP.index(action_enum)

                next_obs, reward, done, _, step_info = env.step(action_idx)
                ep_steps += 1
                ep_tokens = step_info.get("tokens_spent", 0)
                ep_reward += reward
                final_action = action_enum
                obs = next_obs

            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies_list.append(latency_ms)
            rewards_list.append(ep_reward)
            tokens_list.append(ep_tokens)
            steps_list.append(ep_steps)

            if final_action in (RLAction.ANSWER, RLAction.ANSWER_WITH_EVIDENCE):
                answered += 1
                if case.answerable:
                    correct_answers += 1
                else:
                    hallucinations += 1
            elif final_action == RLAction.ABSTAIN:
                if not case.answerable:
                    correct_abstentions += 1

        coverage = round(answered / total * 100.0, 1) if total else 0.0
        halluc_rate = round(hallucinations / total * 100.0, 1) if total else 0.0
        acc_answered = round(correct_answers / answered * 100.0, 1) if answered else 0.0
        corr_abst = round(correct_abstentions / total_traps * 100.0, 1) if total_traps else 0.0
        mean_tok = round(sum(tokens_list) / len(tokens_list), 1) if tokens_list else 0.0
        mean_step = round(sum(steps_list) / len(steps_list), 2) if steps_list else 0.0
        mean_reward = round(sum(rewards_list) / len(rewards_list), 4) if rewards_list else 0.0
        mean_latency = round(sum(latencies_list) / len(latencies_list), 2) if latencies_list else 0.0

        row = {
            "Policy": policy.algo.upper(),
            "Split": args.split,
            "Lambda": lam,
            "Cases": total,
            "Coverage (%)": coverage,
            "Hallucination Rate (%)": halluc_rate,
            "Accuracy on Answered (%)": acc_answered,
            "Correct Abstention (%)": corr_abst,
            "Mean Reward": mean_reward,
            "Mean Latency (ms)": mean_latency,
            "Mean Tokens / Query": mean_tok,
            "Mean Steps": mean_step,
            # Lowercase aliases to avoid KeyError in plotting scripts
            "policy": policy.algo.upper(),
            "split": args.split,
            "lambda": lam,
            "cases": total,
            "coverage_pct": coverage,
            "hallucination_rate_pct": halluc_rate,
            "accuracy_answered_pct": acc_answered,
            "correct_abstention_pct": corr_abst,
            "mean_reward": mean_reward,
            "mean_latency_ms": mean_latency,
            "mean_tokens": mean_tok,
            "mean_steps": mean_step,
        }
        results_rows.append(row)
        logger.info(
            "Lambda %4.2f | Coverage: %5.1f%% | Hallucination: %5.1f%% | Acc: %5.1f%% | Abstain: %5.1f%% | Mean Tokens: %5.1f | Mean Reward: %6.3f | Latency: %5.1f ms",
            lam,
            coverage,
            halluc_rate,
            acc_answered,
            corr_abst,
            mean_tok,
            mean_reward,
            mean_latency,
        )

    # Write CSV summary
    fieldnames = list(results_rows[0].keys())
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    logger.info("Saved evaluation report to %s", args.output_csv)


if __name__ == "__main__":
    evaluate()
