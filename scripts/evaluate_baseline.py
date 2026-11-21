"""
Baseline Evaluation and Analysis for RepoGuard.
Owner: Member 3 (LLM & Experimentation Engineer)

Computes:
- Hallucination rate
- Coverage
- Accuracy on answered
- Correct-abstention rate on traps
- Over-refusal rate
- Mean tokens per query and per correct answer
- Latency (median, mean)

Outputs summary table to reports/baseline_results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Automatically add src to sys.path so the script runs without requiring PYTHONPATH
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repoguard.schemas import BaselineRunOutput, BenchmarkQuestion

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("evaluate_baseline")


def evaluate(
    questions_file: Path,
    runs_file: Path,
    output_csv: Path,
) -> None:
    # Load benchmark ground truth
    questions_map: dict[str, BenchmarkQuestion] = {}
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = BenchmarkQuestion.model_validate_json(line)
                questions_map[q.question_id] = q

    # Load runs by baseline
    runs_by_baseline: dict[str, list[BaselineRunOutput]] = defaultdict(list)
    with open(runs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = BaselineRunOutput.model_validate_json(line)
                runs_by_baseline[item.baseline.value].append(item)

    # Automatically scope evaluation to the questions present in the runs
    evaluated_ids = {item.question_id for items in runs_by_baseline.values() for item in items}
    eval_questions = {qid: questions_map[qid] for qid in evaluated_ids if qid in questions_map}

    total_questions = len(eval_questions)
    total_traps = sum(1 for q in eval_questions.values() if q.is_trap)
    total_answerable = sum(1 for q in eval_questions.values() if not q.is_trap)

    logger.info(
        "Evaluating %d unique questions present in runs (%d answerable, %d traps)",
        total_questions,
        total_answerable,
        total_traps,
    )

    results_table: list[dict[str, str | float]] = []

    for baseline_name, items in runs_by_baseline.items():
        answered_count = 0
        correct_count = 0
        hallucinated_count = 0
        trap_abstained_count = 0
        answerable_abstained_count = 0

        tokens_list = []
        latencies_list = []

        for run in items:
            q = questions_map.get(run.question_id)
            if not q:
                continue

            tokens_list.append(run.total_tokens)
            latencies_list.append(run.latency_ms)

            if run.abstained:
                if q.is_trap:
                    trap_abstained_count += 1
                else:
                    answerable_abstained_count += 1
            else:
                answered_count += 1
                # When answered on a trap, by definition it's a hallucination
                if q.is_trap:
                    hallucinated_count += 1
                else:
                    # Answerable question check
                    ans_lower = run.answer.lower()
                    if q.target_symbol and q.target_symbol.lower() in ans_lower:
                        correct_count += 1
                    elif "defined in" in ans_lower or "parameters" in ans_lower or "method" in ans_lower or "job" in ans_lower:
                        correct_count += 1
                    else:
                        hallucinated_count += 1

        coverage = (answered_count / total_questions) * 100.0 if total_questions else 0.0
        hallucination_rate = (hallucinated_count / total_questions) * 100.0 if total_questions else 0.0
        accuracy_on_answered = (correct_count / answered_count) * 100.0 if answered_count else 0.0
        correct_abstention_rate = (trap_abstained_count / total_traps) * 100.0 if total_traps else 0.0
        over_refusal_rate = (answerable_abstained_count / total_answerable) * 100.0 if total_answerable else 0.0

        mean_tokens = round(statistics.mean(tokens_list), 1) if tokens_list else 0.0
        tokens_per_correct = round(sum(tokens_list) / max(1, correct_count), 1)
        median_latency = round(statistics.median(latencies_list), 1) if latencies_list else 0.0

        results_table.append(
            {
                "Baseline": baseline_name,
                "Model": items[0].model_name if items else "unknown",
                "Coverage (%)": round(coverage, 1),
                "Hallucination Rate (%)": round(hallucination_rate, 1),
                "Accuracy on Answered (%)": round(accuracy_on_answered, 1),
                "Correct Abstention (%)": round(correct_abstention_rate, 1),
                "Over-refusal (%)": round(over_refusal_rate, 1),
                "Mean Tokens / Query": mean_tokens,
                "Tokens / Correct Answer": tokens_per_correct,
                "Median Latency (ms)": median_latency,
            }
        )

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results_table[0].keys()) if results_table else []
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_table)

    logger.info("Saved evaluation summary to %s", output_csv)

    # Print formatted Markdown summary
    print("\n" + "=" * 80)
    print("REPOGUARD PHASE 1 BASELINE EVALUATION REPORT")
    print("=" * 80)
    header = "| " + " | ".join(fieldnames) + " |"
    divider = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    print(header)
    print(divider)
    for row in results_table:
        row_str = "| " + " | ".join(str(row[k]) for k in fieldnames) + " |"
        print(row_str)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RepoGuard Baselines.")
    parser.add_argument(
        "--questions",
        type=str,
        default="data/questions/baseline_questions.jsonl",
        help="Path to questions file (default: full baseline benchmark suite)",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default="data/baseline/raw_outputs.jsonl",
        help="Path to raw baseline runs JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/baseline_results.csv",
        help="Path to output CSV summary",
    )

    args = parser.parse_args()
    evaluate(Path(args.questions), Path(args.runs), Path(args.output))
