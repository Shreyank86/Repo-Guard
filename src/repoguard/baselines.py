"""
Comparison Baseline Systems for RepoGuard.
Owner: Member 3 (LLM & Experimentation Engineer)

Implements:
1. always_answer: Direct zero-shot generation without retrieval or verification.
2. always_use_evidence: Standard dense retrieval-augmented generation (RAG).
3. threshold_gate: Static confidence gate that abstains if top retrieval similarity is below threshold.
"""

from __future__ import annotations

import logging
from typing import Any

from repoguard.llm import LLMService
from repoguard.retrieval.search import search
from repoguard.schemas import BaselineRunOutput, BaselineType, IndexArtifacts

logger = logging.getLogger("repoguard.baselines")

ZERO_SHOT_SYSTEM_PROMPT = """You are a code developer assistant answering questions about a Python repository.
Answer the developer question directly based on your knowledge.
Enclose any code identifiers (functions, classes, files, parameters) in backticks."""

EVIDENCE_SYSTEM_PROMPT = """You are a repository-grounded developer assistant.
Answer the question strictly using the provided code repository passages.
If the provided code does not contain sufficient information, state that the evidence is missing.
Enclose any code identifiers (functions, classes, files, parameters) in backticks."""


def build_evidence_context(evidences: list[Any]) -> str:
    """Format retrieved evidence snippets into a structured prompt block."""
    if not evidences:
        return "No relevant repository passages found."

    blocks = []
    for idx, ev in enumerate(evidences, 1):
        blocks.append(
            f"--- Evidence {idx} (File: {ev.file_path}, Lines {ev.start_line}-{ev.end_line}) ---\n"
            f"{ev.snippet}\n"
        )
    return "\n".join(blocks)


def always_answer(
    question_id: str,
    question: str,
    llm: LLMService,
) -> BaselineRunOutput:
    """
    Baseline 1: Always Answer (Plain Zero-Shot LLM).
    Never accesses the repository index and never abstains.
    """
    prompt = f"{ZERO_SHOT_SYSTEM_PROMPT}\n\nQuestion: {question}\n\nAnswer:"
    gen_result = llm.generate(prompt=prompt, max_new_tokens=128, temperature=0.1)

    return BaselineRunOutput(
        question_id=question_id,
        baseline=BaselineType.ALWAYS_ANSWER,
        model_name=llm.model_name,
        question=question,
        answer=gen_result.text,
        abstained=False,
        evidence_ids=[],
        prompt_tokens=gen_result.prompt_tokens,
        completion_tokens=gen_result.completion_tokens,
        total_tokens=gen_result.total_tokens,
        latency_ms=gen_result.latency_ms,
        is_mock=gen_result.is_mock,
    )


def always_use_evidence(
    question_id: str,
    question: str,
    index: IndexArtifacts,
    llm: LLMService,
    top_k: int = 3,
) -> BaselineRunOutput:
    """
    Baseline 2: Always Use Evidence (Dense RAG).
    Retrieves top-K passages and always attempts an answer from code evidence.
    """
    evidences = search(question=question, index=index, k=top_k)
    context_text = build_evidence_context(evidences)

    prompt = (
        f"{EVIDENCE_SYSTEM_PROMPT}\n\n"
        f"Repository Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    gen_result = llm.generate(prompt=prompt, max_new_tokens=128, temperature=0.1)

    return BaselineRunOutput(
        question_id=question_id,
        baseline=BaselineType.ALWAYS_EVIDENCE,
        model_name=llm.model_name,
        question=question,
        answer=gen_result.text,
        abstained=False,
        evidence_ids=[e.evidence_id for e in evidences],
        prompt_tokens=gen_result.prompt_tokens,
        completion_tokens=gen_result.completion_tokens,
        total_tokens=gen_result.total_tokens,
        latency_ms=gen_result.latency_ms,
        is_mock=gen_result.is_mock,
    )


def threshold_gate(
    question_id: str,
    question: str,
    index: IndexArtifacts,
    llm: LLMService,
    threshold: float = 0.45,
    top_k: int = 3,
) -> BaselineRunOutput:
    """
    Baseline 3: Threshold Gate.
    Heuristic rule-based router that abstains if top retrieval similarity is below threshold.
    """
    evidences = search(question=question, index=index, k=top_k)
    max_sim = max([e.similarity_score for e in evidences], default=0.0)

    # Heuristic gate: If best similarity is weak, abstain without consuming LLM tokens
    if not evidences or max_sim < threshold:
        return BaselineRunOutput(
            question_id=question_id,
            baseline=BaselineType.THRESHOLD_GATE,
            model_name=llm.model_name,
            question=question,
            answer=f"I can't find this in the repository (top similarity {max_sim:.2f} < threshold {threshold}). Abstaining.",
            abstained=True,
            evidence_ids=[e.evidence_id for e in evidences],
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=5.0,  # nominal search lookup time
            is_mock=llm.mock_mode,
        )

    # Otherwise, proceed to generate grounded answer with retrieved evidence
    context_text = build_evidence_context(evidences)
    prompt = (
        f"{EVIDENCE_SYSTEM_PROMPT}\n\n"
        f"Repository Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    gen_result = llm.generate(prompt=prompt, max_new_tokens=128, temperature=0.1)

    return BaselineRunOutput(
        question_id=question_id,
        baseline=BaselineType.THRESHOLD_GATE,
        model_name=llm.model_name,
        question=question,
        answer=gen_result.text,
        abstained=False,
        evidence_ids=[e.evidence_id for e in evidences],
        prompt_tokens=gen_result.prompt_tokens,
        completion_tokens=gen_result.completion_tokens,
        total_tokens=gen_result.total_tokens,
        latency_ms=gen_result.latency_ms,
        is_mock=gen_result.is_mock,
    )
