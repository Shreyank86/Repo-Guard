#!/usr/bin/env python
"""Evaluate retrieval on the benchmark and validate the RL free signals.

    python scripts/evaluate_retrieval.py                       # data/retrieval/retrieval_benchmark_v1.jsonl
    python scripts/evaluate_retrieval.py --out-dir reports/retrieval

Writes:
  <out>/retrieval_eval_<backend>.json   summaries, ablations, signal checks
  <out>/retrieval_eval_<backend>_cases.jsonl   per-case rankings and metrics
  <out>/retrieval_eval_<backend>.md     tables: per repository x case type, ablation, signals

Ablations on the same cases: dense-only (no symbol boosts) and hybrid + exclude_tests.
Boost defaults were chosen on the dev repositories only; `requests` (test) is reported apart.
Signal check: on Member 3's baseline questions, how well each free signal separates
answerable questions from traps (AUROC; 0.5 = no separation).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repoguard.retrieval import Retriever, SearchFilters, SignalExtractor  # noqa: E402
from repoguard.retrieval.metrics import evaluate_cases  # noqa: E402

SIGNALS = [
    "symbol_found", "name_similarity", "retr_top1", "retr_margin",
    "passage_mentions_symbol", "stack_match", "coverage", "vocab_overlap",
]  # fmt: skip


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """P(score_pos > score_neg) with ties counted half; None if a class is empty."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    return head + "".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n" for r in rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "retrieval" / "retrieval_benchmark_v1.jsonl",
    )
    ap.add_argument("--indexes", type=Path, default=ROOT / "data" / "indexes")
    ap.add_argument(
        "--questions", type=Path, default=ROOT / "data" / "questions" / "baseline_questions.jsonl"
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "retrieval")
    args = ap.parse_args(argv)

    cases = [json.loads(line) for line in args.benchmark.read_text().splitlines() if line.strip()]
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_repo[c["repository_id"]].append(c)

    main_cases, dense_cases, ablation_cases, backends, mismatched = [], [], [], set(), []
    retrievers: dict[str, Retriever] = {}
    for repo_id, rcases in sorted(by_repo.items()):
        r = Retriever.from_dir(args.indexes / repo_id)
        if r.index.index_id != rcases[0]["index_id"]:
            mismatched.append(repo_id)  # benchmark built from a different index build
        retrievers[repo_id] = r
        backends.add(r.backend)
        main_cases += evaluate_cases(r, rcases).cases
        dense_cases += evaluate_cases(r, rcases, definition_boost=0.0, mention_boost=0.0).cases
        ablation_cases += evaluate_cases(r, rcases, filters=SearchFilters(exclude_tests=True)).cases
    if mismatched:
        print(
            f"WARNING: benchmark index_id differs from current index for {mismatched}; rebuild the benchmark"
        )

    from repoguard.retrieval.metrics import EvalReport

    main_rep, abl_rep = EvalReport(main_cases), EvalReport(ablation_cases)
    per_repo_type = main_rep.summary(("repository_id", "case_type"))
    per_split = EvalReport(main_cases).summary(("case_type",))
    by_split = EvalReport(main_cases).summary(("split", "case_type"))
    dense_by_split = EvalReport(dense_cases).summary(("split", "case_type"))
    abl = abl_rep.summary(("case_type",))

    # ---- free-signal validation on Member 3's questions
    signal_rows, auc_rows = [], []
    if args.questions.exists():
        qs = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
        by_type: dict[str, list] = defaultdict(list)
        for q in qs:
            repo_id = q["repository"].split("/")[-1]
            if repo_id not in retrievers:
                continue
            sig = SignalExtractor(retrievers[repo_id]).compute(q["question"])
            by_type[q["question_type"]].append((q, sig))
        for t, rows in sorted(by_type.items()):
            row = {"question_type": t, "n": len(rows), "is_trap": rows[0][0]["is_trap"]}
            for s in SIGNALS:
                row[s] = round(statistics.fmean(getattr(sg, s) for _, sg in rows), 3)
            signal_rows.append(row)
        flat = [x for rows in by_type.values() for x in rows]
        answerable = [sg for q, sg in flat if not q["is_trap"]]
        for trap_kind in (
            "trap_nonexistent",
            "trap_near_miss",
            "trap_parameters",
            "trap_off_topic",
        ):
            traps = [sg for q, sg in flat if q["question_type"] == trap_kind]
            row = {"answerable_vs": trap_kind, "n_traps": len(traps)}
            for s in SIGNALS:
                row[s] = auroc([getattr(x, s) for x in answerable], [getattr(x, s) for x in traps])
            auc_rows.append(row)

    backend = "+".join(sorted(backends)).replace(" ", "").replace("(", "-").replace(")", "")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"retrieval_eval_{backend}.json"
    out_md = args.out_dir / f"retrieval_eval_{backend}.md"
    out_cases = args.out_dir / f"retrieval_eval_{backend}_cases.jsonl"
    out_cases.write_text(
        "".join(json.dumps(c.__dict__) + "\n" for c in main_cases), encoding="utf-8"
    )
    out_json.write_text(
        json.dumps(
            {
                "benchmark": str(args.benchmark.relative_to(ROOT))
                if args.benchmark.is_relative_to(ROOT)
                else str(args.benchmark),
                "backend": sorted(backends),
                "index_ids": {k: v.index.index_id for k, v in retrievers.items()},
                "summary_by_repo_type": per_repo_type,
                "summary_by_type": per_split,
                "summary_by_split_type": by_split,
                "dense_only_by_split_type": dense_by_split,
                "ablation_exclude_tests_by_type": abl,
                "signals_by_question_type": signal_rows,
                "signal_auroc": auc_rows,
                "cases_file": out_cases.name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cols = ["repository_id", "case_type", "n", "hit@1", "recall@5", "recall@10", "mrr", "ndcg@10",
            "latency_p50_ms", "latency_p95_ms", "empty_rate", "mean_tokens"]  # fmt: skip
    md = [
        f"# Retrieval evaluation ({', '.join(sorted(backends))})\n",
        f"Benchmark: `{args.benchmark.name}` ({len(cases)} cases). Index ids: "
        + ", ".join(f"`{v.index.index_id}`" for v in retrievers.values())
        + "\n",
        "Relevance unit = symbol (all parts of one function count once). `requests` is the held-out test repository.\n",
        "## By repository and case type\n",
        table(per_repo_type, cols),
        "\n## By case type (all repositories)\n",
        table(per_split, cols[1:]),
        "\n## Hybrid (default) by split\n",
        table(by_split, ["split", *cols[1:]]),
        "\n## Ablation: dense only (definition_boost = mention_boost = 0) by split\n",
        table(dense_by_split, ["split", *cols[1:]]),
        "\n## Ablation: exclude test files (`SearchFilters(exclude_tests=True)`)\n",
        table(abl, cols[1:]),
    ]
    if signal_rows:
        md += [
            "\n## Free signals on the baseline question set (means)\n",
            table(signal_rows, ["question_type", "n", "is_trap", *SIGNALS]),
            "\n## Signal separation: AUROC answerable vs. each trap type (0.5 = none, 1.0 = perfect)\n",
            table(auc_rows, ["answerable_vs", "n_traps", *SIGNALS]),
        ]
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(table(by_split, ["split", *cols[1:7]]))
    print(table(dense_by_split, ["split", *cols[1:7]]))
    if auc_rows:
        print(table(auc_rows, ["answerable_vs", "n_traps", *SIGNALS]))
    print(f"Wrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
