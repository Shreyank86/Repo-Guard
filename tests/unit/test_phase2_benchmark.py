from collections import Counter
from pathlib import Path

import pytest

from repoguard.evaluation.benchmark import BenchmarkCase, load_benchmark

BENCHMARK = Path(__file__).resolve().parents[2] / "data/questions/phase2_benchmark_v1.jsonl"


def _write(path: Path, cases: list[BenchmarkCase]) -> None:
    path.write_text("".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8")


def test_versioned_benchmark_keeps_the_test_repository_held_out() -> None:
    cases = load_benchmark(BENCHMARK)

    assert Counter(case.split for case in cases) == {"train": 36, "validation": 12, "test": 12}
    assert {case.repository_id for case in cases if case.split == "test"} == {"requests"}
    assert len([case for case in cases if case.question_type == "multi_hop"]) == 5
    assert {case.answerable for case in cases if case.split == "test"} == {True, False}


def test_rejects_repository_split_leakage(tmp_path: Path) -> None:
    cases = load_benchmark(BENCHMARK)
    cases[0] = cases[0].model_copy(update={"split": "test"})
    path = tmp_path / "leaked.jsonl"
    _write(path, cases)

    with pytest.raises(ValueError, match="different snapshots or splits"):
        load_benchmark(path)


def test_rejects_normalized_duplicate_questions(tmp_path: Path) -> None:
    cases = load_benchmark(BENCHMARK)
    cases[1] = cases[1].model_copy(update={"question": "   " + cases[0].question.upper()})
    path = tmp_path / "duplicates.jsonl"
    _write(path, cases)

    with pytest.raises(ValueError, match="Duplicate benchmark question"):
        load_benchmark(path)
