"""Retrieval metrics and an evaluation loop over labelled cases.

Relevance is judged per *unit*: a passage's ``symbol_id`` (all parts of one function
count as one unit) or, for module overviews, its ``passage_id``. Ranked units are taken
in order of first appearance, so a multi-part function is not double-counted.

Binary relevance throughout: Recall@k, Precision@k, Hit@k, MRR, nDCG@k.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from repoguard.retrieval.retriever import Retriever
from repoguard.schemas import Evidence, SearchFilters


def evidence_unit(e: Evidence) -> str:
    """Relevance unit of an evidence item (``<file>::<qualified_name>`` or module passage id)."""
    pid = e.passage_id
    if "@" in pid:  # multi-part passage id  ->  base symbol id
        pid = pid.rsplit("@", 1)[0]
    return pid


def ranked_units(evidence: Iterable[Evidence]) -> list[str]:
    out: list[str] = []
    for e in evidence:
        u = evidence_unit(e)
        if u not in out:
            out.append(u)
    return out


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / k if k else 0.0


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, u in enumerate(ranked, start=1):
        if u in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, u in enumerate(ranked[:k], start=1) if u in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(q / 100 * len(s)) - 1))
    return s[idx]


@dataclass
class CaseResult:
    case_id: str
    case_type: str
    repository_id: str
    relevant: list[str]
    ranked: list[str]
    metrics: dict[str, float]
    latency_ms: float
    retrieved_tokens: int
    duplicates_removed: int
    empty: bool
    top1_score: float
    split: str = ""


@dataclass
class EvalReport:
    cases: list[CaseResult] = field(default_factory=list)

    def summary(
        self, group_by: Sequence[str] = ("repository_id", "case_type")
    ) -> list[dict[str, Any]]:
        groups: dict[tuple, list[CaseResult]] = defaultdict(list)
        for c in self.cases:
            groups[tuple(getattr(c, g) for g in group_by)].append(c)
            groups[tuple("ALL" for _ in group_by)].append(c)
        rows = []
        for key in sorted(groups, key=lambda t: tuple(x == "ALL" for x in t) + t):
            cs = groups[key]
            row: dict[str, Any] = dict(zip(group_by, key, strict=True))
            row["n"] = len(cs)
            for m in cs[0].metrics:
                row[m] = round(statistics.fmean(c.metrics[m] for c in cs), 4)
            lat = [c.latency_ms for c in cs]
            row["latency_p50_ms"] = round(percentile(lat, 50), 3)
            row["latency_p95_ms"] = round(percentile(lat, 95), 3)
            row["empty_rate"] = round(sum(c.empty for c in cs) / len(cs), 4)
            row["dup_rate"] = round(sum(c.duplicates_removed > 0 for c in cs) / len(cs), 4)
            row["mean_tokens"] = round(statistics.fmean(c.retrieved_tokens for c in cs), 1)
            row["mean_top1_score"] = round(statistics.fmean(c.top1_score for c in cs), 4)
            rows.append(row)
        return rows


def evaluate_cases(
    retriever: Retriever,
    cases: Iterable[dict[str, Any]],
    ks: Sequence[int] = (1, 3, 5, 10),
    filters: SearchFilters | None = None,
    **search_kwargs: Any,
) -> EvalReport:
    """Run every case (``query`` + ``relevant_units``) through ``retriever.search``.

    ``search_kwargs`` are passed to :meth:`Retriever.search` (e.g. boosts for ablations).
    """
    report = EvalReport()
    depth = max(ks)
    for case in cases:
        relevant = set(case["relevant_units"])
        res = retriever.search(case["query"], k=depth, filters=filters, **search_kwargs)
        ranked = ranked_units(res.evidence)
        metrics: dict[str, float] = {}
        for k in ks:
            metrics[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        metrics["hit@1"] = hit_at_k(ranked, relevant, 1)
        metrics["mrr"] = reciprocal_rank(ranked, relevant)
        metrics[f"ndcg@{depth}"] = ndcg_at_k(ranked, relevant, depth)
        top5 = res.evidence[:5]
        report.cases.append(
            CaseResult(
                case_id=case["case_id"],
                case_type=case["case_type"],
                repository_id=case["repository_id"],
                relevant=sorted(relevant),
                ranked=ranked,
                metrics=metrics,
                latency_ms=res.diagnostics.latency_ms,
                retrieved_tokens=sum(e.approx_tokens for e in top5),
                duplicates_removed=res.diagnostics.duplicates_removed,
                empty=res.diagnostics.empty,
                top1_score=res.diagnostics.top1_score,
                split=case.get("split_hint", ""),
            )
        )
    return report
