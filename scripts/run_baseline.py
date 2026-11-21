"""
Baseline Execution Runner for RepoGuard.
Owner: Member 3 (LLM & Experimentation Engineer)

Runs the 100-question benchmark against candidate models and baselines:
1. always_answer
2. always_evidence
3. threshold_gate

Saves traces to data/baseline/raw_outputs.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Automatically add src to sys.path so the script runs without requiring PYTHONPATH
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repoguard.baselines import always_answer, always_use_evidence, threshold_gate
from repoguard.llm import LLMService
from repoguard.repository import load_index
from repoguard.schemas import BaselineRunOutput, BenchmarkQuestion

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("run_baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RepoGuard Phase 1 Baselines.")
    parser.add_argument(
        "--questions",
        type=str,
        default="data/questions/baseline_questions.jsonl",
        help="Path to questions JSONL file (default: full 100-question benchmark)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="data/indexes/schedule",
        help="Path to repository index directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/baseline/raw_outputs.jsonl",
        help="Path to output raw runs JSONL",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear previous raw outputs before running fresh batch",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["always_answer", "always_evidence", "threshold_gate", "all"],
        default="all",
        help="Baseline system to evaluate",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
        help="Hugging Face model identifier or local path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device (prefers cuda with automatic cpu fallback)",
    )
    parser.add_argument(
        "--quantization-bits",
        type=int,
        default=4,
        choices=[4, 8, 16],
        help="Quantization bits for GPU execution (4, 8, or 16)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode for fast testing/evaluation without loading heavy weights",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to run (e.g. for quick smoke testing)",
    )

    args = parser.parse_args()

    # Load questions
    q_path = Path(args.questions)
    if not q_path.exists():
        raise FileNotFoundError(f"Questions file not found: {q_path}")

    questions: list[BenchmarkQuestion] = []
    with open(q_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(BenchmarkQuestion.model_validate_json(line))

    if args.limit:
        questions = questions[: args.limit]

    logger.info("Loaded %d questions from %s", len(questions), q_path)

    # Load repository index
    index_path = Path(args.index)
    if not index_path.exists():
        raise FileNotFoundError(f"Index directory not found: {index_path}. Run build_index.py first.")

    index = load_index(str(index_path))
    logger.info("Loaded index for %s with %d symbols.", index.repository.name, len(index.symbols))

    # Initialize LLM Service
    llm = LLMService(
        model_name=args.model_name,
        device=args.device,
        quantization_bits=args.quantization_bits,
        mock_mode=args.mock,
    )
    logger.info("Initialized LLM Service (mock_mode=%s, device=%s)", args.mock, args.device)

    # Select baselines
    baselines_to_run = (
        ["always_answer", "always_evidence", "threshold_gate"]
        if args.baseline == "all"
        else [args.baseline]
    )

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if args.reset and out_file.exists():
        out_file.unlink()
        logger.info("Cleared previous raw outputs at %s due to --reset flag.", out_file)

    # Load already completed (baseline, question_id) pairs to enable seamless resumption
    completed_keys: set[tuple[str, str]] = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        b_val = record.get("baseline")
                        q_val = record.get("question_id")
                        if b_val and q_val:
                            completed_keys.add((b_val, q_val))
                    except Exception:
                        pass
        if completed_keys:
            logger.info("Found %d already completed results in %s. Resuming...", len(completed_keys), out_file)

    for b_name in baselines_to_run:
        logger.info("--- Executing Baseline: %s ---", b_name)
        for idx, q in enumerate(questions, 1):
            if (b_name, q.question_id) in completed_keys:
                continue

            if b_name == "always_answer":
                res = always_answer(q.question_id, q.question, llm)
            elif b_name == "always_evidence":
                res = always_use_evidence(q.question_id, q.question, index, llm)
            elif b_name == "threshold_gate":
                res = threshold_gate(q.question_id, q.question, index, llm)
            else:
                continue

            # Stream result to disk immediately so no progress is ever lost if stopped
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(res.model_dump()) + "\n")
                f.flush()

            completed_keys.add((b_name, q.question_id))
            logger.info(
                "[%s] [%d/%d] %s: '%s' -> %d tokens, %.0fms",
                b_name,
                idx,
                len(questions),
                q.question_id,
                q.question[:45] + "..." if len(q.question) > 45 else q.question,
                res.total_tokens,
                res.latency_ms,
            )

    logger.info("Baseline execution complete. Total results stored in %s: %d", out_file, len(completed_keys))


if __name__ == "__main__":
    main()
