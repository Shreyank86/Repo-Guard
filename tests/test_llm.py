"""
Unit Tests for Member 3 Frozen LLM Inference and Baselines.
Owner: Member 3 (LLM & Experimentation Engineer)
"""

import pytest
from repoguard.baselines import always_answer, always_use_evidence, threshold_gate
from repoguard.llm import LLMService, MockLLM
from repoguard.repository import load_index
from repoguard.schemas import BaselineType, BenchmarkQuestion, GenerationResult


@pytest.fixture
def mock_llm_service() -> LLMService:
    return LLMService(
        model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        device="cpu",
        mock_mode=True,
    )


@pytest.fixture
def schedule_index():
    return load_index("data/indexes/schedule")


def test_mock_llm_generation(mock_llm_service):
    """Test prompt inference, token counting, and latency metrics."""
    res = mock_llm_service.generate("Where is load_config defined in schedule?", max_new_tokens=100)
    assert isinstance(res, GenerationResult)
    assert res.text != ""
    assert res.prompt_tokens > 0
    assert res.completion_tokens > 0
    assert res.total_tokens == res.prompt_tokens + res.completion_tokens
    assert res.latency_ms >= 0.0
    assert res.is_mock is True


def test_yes_no_prob(mock_llm_service):
    """Test binary probability extraction for self-check signals."""
    # High probability expected for known existing symbol
    p_high = mock_llm_service.yes_no_prob("Does the symbol 'every' exist in schedule?")
    assert 0.0 <= p_high <= 1.0
    assert p_high > 0.8

    # Low probability expected for fake/trap symbol
    p_low = mock_llm_service.yes_no_prob("Does the symbol 'sync_all_users_fake' exist in schedule?")
    assert 0.0 <= p_low <= 1.0
    assert p_low < 0.2


def test_sampling_consensus(mock_llm_service):
    """Test generating k stochastic samples for consensus analysis."""
    samples = mock_llm_service.sample("What is the role of Scheduler in schedule?", k=4)
    assert len(samples) == 4
    assert all(isinstance(s, str) and len(s) > 0 for s in samples)


def test_baseline_always_answer(mock_llm_service):
    """Test always_answer baseline generates ungrounded answers without abstention."""
    output = always_answer(
        question_id="test_q1",
        question="Where is Job defined in schedule?",
        llm=mock_llm_service,
    )
    assert output.baseline == BaselineType.ALWAYS_ANSWER
    assert output.abstained is False
    assert output.total_tokens > 0
    assert len(output.evidence_ids) == 0


def test_baseline_always_use_evidence(mock_llm_service, schedule_index):
    """Test always_use_evidence retrieves code passages and supplies evidence."""
    output = always_use_evidence(
        question_id="test_q2",
        question="Where is the Job class defined?",
        index=schedule_index,
        llm=mock_llm_service,
        top_k=2,
    )
    assert output.baseline == BaselineType.ALWAYS_EVIDENCE
    assert output.abstained is False
    assert len(output.evidence_ids) > 0
    assert output.total_tokens > 0


def test_baseline_threshold_gate_abstention(mock_llm_service, schedule_index):
    """Test threshold gate correctly abstains when similarity is below threshold."""
    output = threshold_gate(
        question_id="test_q3",
        question="Where is the Celery redis broker URL configured?",
        index=schedule_index,
        llm=mock_llm_service,
        threshold=0.99,  # Impossibly high threshold to trigger abstention
    )
    assert output.baseline == BaselineType.THRESHOLD_GATE
    assert output.abstained is True
    assert output.prompt_tokens == 0
    assert output.completion_tokens == 0


def test_baseline_threshold_gate_answers(mock_llm_service, schedule_index):
    """Test threshold gate answers when similarity meets threshold."""
    output = threshold_gate(
        question_id="test_q4",
        question="Where is Job defined?",
        index=schedule_index,
        llm=mock_llm_service,
        threshold=0.01,  # Low threshold to guarantee pass
    )
    assert output.baseline == BaselineType.THRESHOLD_GATE
    assert output.abstained is False
    assert output.total_tokens > 0


def test_benchmark_question_schema():
    """Verify BenchmarkQuestion serialization and data fields."""
    q = BenchmarkQuestion(
        question_id="test_q5",
        repository="dbader/schedule",
        commit="82a43db1b938d8fdf60103bd41f329e06c8d3651",
        question_type="where_defined",
        question="Where is Job defined?",
        expected_action="answer_with_evidence",
        is_trap=False,
        ground_truth_answer="schedule/__init__.py line 28",
        target_symbol="Job",
        target_file="schedule/__init__.py",
        relevant_lines=[28, 160],
    )
    dumped = q.model_dump_json()
    reloaded = BenchmarkQuestion.model_validate_json(dumped)
    assert reloaded.question_id == "test_q5"
    assert reloaded.is_trap is False
    assert reloaded.target_symbol == "Job"
