"""Retrieval metric functions and the evaluation loop."""

from __future__ import annotations

import math

import pytest

from repoguard.repository import build_index
from repoguard.retrieval import Retriever
from repoguard.retrieval.metrics import (
    evaluate_cases,
    evidence_unit,
    hit_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    ranked_units,
    recall_at_k,
    reciprocal_rank,
)
from repoguard.schemas import Evidence

RANKED = ["a", "x", "b", "y", "z"]
REL = {"a", "b", "c"}


def test_metric_values():
    assert recall_at_k(RANKED, REL, 1) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, REL, 3) == pytest.approx(2 / 3)
    assert precision_at_k(RANKED, REL, 5) == pytest.approx(2 / 5)
    assert hit_at_k(RANKED, REL, 1) == 1.0 and hit_at_k(["x"], REL, 1) == 0.0
    assert reciprocal_rank(["x", "y", "b"], REL) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], REL) == 0.0
    dcg = 1 + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, REL, 5) == pytest.approx(dcg / idcg)
    assert recall_at_k(RANKED, set(), 5) == 0.0 and ndcg_at_k(RANKED, set(), 5) == 0.0
    assert percentile([5, 1, 3, 2, 4], 50) == 3 and percentile([], 95) == 0.0


def _ev(pid: str) -> Evidence:
    return Evidence(
        evidence_id="ev_x",
        passage_id=pid,
        file_path="f.py",
        start_line=1,
        end_line=1,
        snippet="",
        similarity_score=0.5,
    )


def test_units_collapse_multi_part_passages():
    assert evidence_unit(_ev("f.py::m.big@2")) == "f.py::m.big"
    assert evidence_unit(_ev("f.py::<module>@1")) == "f.py::<module>"
    assert ranked_units([_ev("f.py::m.a@1"), _ev("f.py::m.a@2"), _ev("f.py::m.b")]) == [
        "f.py::m.a",
        "f.py::m.b",
    ]


def test_evaluate_cases_end_to_end(fixture_repo, tmp_path):
    r = Retriever(build_index(fixture_repo, tmp_path / "i"))
    cases = [
        {"case_id": "c1", "case_type": "symbol_lookup", "repository_id": "indexer-repo",
         "query": "Where is `validate` defined?", "relevant_units": ["pkg/config.py::pkg.config.validate"]},
        {"case_id": "c2", "case_type": "callers", "repository_id": "indexer-repo",
         "query": "Which functions call `load_config`?", "relevant_units": ["pkg/app.py::pkg.app.App.initialize_app"]},
    ]  # fmt: skip
    rep = evaluate_cases(r, cases)
    c1 = rep.cases[0]
    assert (
        c1.metrics["hit@1"] == 1.0
        and c1.metrics["mrr"] == 1.0
        and c1.ranked[0] == "pkg/config.py::pkg.config.validate"
    )
    assert rep.cases[1].metrics["recall@10"] == 1.0
    rows = rep.summary(("case_type",))
    assert rows[-1]["case_type"] == "ALL" and rows[-1]["n"] == 2
    dense = evaluate_cases(r, cases, definition_boost=0.0, mention_boost=0.0)
    assert len(dense.cases) == 2
