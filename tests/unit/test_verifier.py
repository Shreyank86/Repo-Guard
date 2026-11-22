import json
from pathlib import Path

import pytest

from repoguard.schemas import IndexArtifacts
from repoguard.verification import Claim, VerdictStatus, extract_claims, verify


@pytest.fixture
def mock_index() -> IndexArtifacts:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "mock_index.json"
    return IndexArtifacts.model_validate_json(fixture_path.read_text(encoding="utf-8"))


def test_rejects_baseline_job_claim_in_unindexed_file(mock_index: IndexArtifacts) -> None:
    raw_outputs = Path(__file__).resolve().parents[2] / "data" / "baseline" / "raw_outputs.jsonl"
    with raw_outputs.open(encoding="utf-8") as source:
        job_run = next(
            run
            for run in map(json.loads, source)
            if run["question_id"] == "q_ans_01" and run["baseline"] == "always_answer"
        )
    claims = extract_claims(job_run["answer"])
    result = verify(claims, mock_index)

    assert claims == [Claim(symbol="Job", file="job.py")]
    assert result.verdict == "unsupported"
    assert result.claims[0].status == VerdictStatus.UNVERIFIABLE
    assert result.claims[0].evidence == []
    assert result.unsupported_names == ["job.py"]


def test_verifies_job_in_its_defining_file(mock_index: IndexArtifacts) -> None:
    result = verify([Claim(symbol="schedule.Job", file="schedule/__init__.py")], mock_index)

    assert result.verdict == "verified"
    assert result.claims[0].status == VerdictStatus.VERIFIED
    assert result.claims[0].evidence[0].file == "schedule/__init__.py"
    assert result.claims[0].evidence[0].line == 209
    assert result.unsupported_names == []


def test_reports_mixed_verdicts_when_job_is_claimed_in_two_files(
    mock_index: IndexArtifacts,
) -> None:
    result = verify(
        [
            Claim(symbol="Job", file="schedule/__init__.py"),
            Claim(symbol="Job", file="test_schedule.py"),
        ],
        mock_index,
    )

    assert result.verdict == "partially_verified"
    assert [verdict.status for verdict in result.claims] == [
        VerdictStatus.VERIFIED,
        VerdictStatus.REFUTED,
    ]
    assert result.unsupported_names == ["Job"]


def test_empty_claims_are_unsupported(mock_index: IndexArtifacts) -> None:
    result = verify([], mock_index)

    assert result.verdict == "unsupported"
    assert result.claims == []


@pytest.mark.parametrize(
    ("fields", "status"),
    [
        ({"type": "file_exists", "file": "test_schedule.py"}, VerdictStatus.VERIFIED),
        ({"type": "file_exists", "file": "missing.py"}, VerdictStatus.UNVERIFIABLE),
        ({"type": "file_exists", "file": "README.rst"}, VerdictStatus.UNVERIFIABLE),
        (
            {
                "type": "function_exists",
                "symbol": "schedule.run_pending",
                "file": "schedule/__init__.py",
            },
            VerdictStatus.VERIFIED,
        ),
        (
            {
                "type": "function_exists",
                "symbol": "schedule.run_pending",
                "file": "test_schedule.py",
            },
            VerdictStatus.REFUTED,
        ),
        ({"type": "function_exists", "symbol": "run_pendin"}, VerdictStatus.REFUTED),
        ({"type": "function_exists", "symbol": "job.do"}, VerdictStatus.REFUTED),
        ({"type": "function_exists", "symbol": "Job"}, VerdictStatus.REFUTED),
        (
            {"type": "parameter_exists", "symbol": "Job.do", "parameter": "job_func"},
            VerdictStatus.VERIFIED,
        ),
        (
            {"type": "parameter_exists", "symbol": "Job.do", "parameter": "timeout"},
            VerdictStatus.REFUTED,
        ),
        (
            {"type": "parameter_exists", "symbol": "run_pending", "parameter": "self"},
            VerdictStatus.UNVERIFIABLE,
        ),
        (
            {"type": "calls", "caller": "schedule.run_pending", "callee": "Scheduler.run_pending"},
            VerdictStatus.VERIFIED,
        ),
        (
            {"type": "calls", "caller": "Scheduler.run_pending", "callee": "schedule.run_pending"},
            VerdictStatus.REFUTED,
        ),
        (
            {"type": "calls", "caller": "Scheduler._run_job", "callee": "Job.run"},
            VerdictStatus.UNVERIFIABLE,
        ),
        (
            {"type": "imports", "file": "schedule/__init__.py", "module": "datetime"},
            VerdictStatus.VERIFIED,
        ),
        (
            {
                "type": "imports",
                "file": "schedule/__init__.py",
                "module": "collections.abc.Hashable",
            },
            VerdictStatus.VERIFIED,
        ),
        (
            {"type": "imports", "file": "schedule/__init__.py", "module": "datatime"},
            VerdictStatus.REFUTED,
        ),
    ],
)
def test_checks_structural_facts(
    mock_index: IndexArtifacts, fields: dict[str, str], status: VerdictStatus
) -> None:
    verdict = verify([fields], mock_index).claims[0]

    assert verdict.status == status
    assert bool(verdict.evidence) == (status == VerdictStatus.VERIFIED)


@pytest.mark.parametrize(
    "invalid",
    [
        {"type": "does_not_exist", "symbol": "Job"},
        {"type": "calls", "caller": "Job.do"},
        {"type": "function_exists", "symbol": "  "},
        {"type": "file_exists", "file": "test_schedule.py", "symbol": "Job"},
    ],
)
def test_malformed_claims_do_not_block_valid_claims(
    mock_index: IndexArtifacts, invalid: dict[str, str]
) -> None:
    good = Claim(symbol="Job", file="schedule/__init__.py")
    result = verify([invalid, good], mock_index)

    assert result.verdict == "partially_verified"
    assert result.claims[0].claim is None
    assert result.claims[0].status == VerdictStatus.UNVERIFIABLE
    assert result.claims[1].status == VerdictStatus.VERIFIED


def test_empty_and_failed_indexes_never_verify_an_unavailable_definition(
    mock_index: IndexArtifacts,
) -> None:
    question = Claim(symbol="Job", file="schedule/__init__.py")
    empty = mock_index.model_copy(update={"files": [], "symbols": []})
    assert verify([question], empty).claims[0].status == VerdictStatus.UNVERIFIABLE

    mock_index.files[0].parse_status = "error"
    assert verify([question], mock_index).claims[0].status == VerdictStatus.UNVERIFIABLE
