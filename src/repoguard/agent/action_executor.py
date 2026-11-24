"""Action Executor for RepoGuard RL Agent.

Owner: Member 3 (RL Environment, Training & Model Integration)

Executes the 6 canonical RLAction choices:
1. SEARCH: Retrieve repository evidence passages using Retriever
2. INSPECT: Read full symbol definition or file context around best match
3. VERIFY: Generate draft claims and verify against index
4. ANSWER: Direct answer generation from frozen LLM
5. ANSWER_WITH_EVIDENCE: Grounded answer generation using retrieved context
6. ABSTAIN: Honest refusal with closest matches from repository vocabulary
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repoguard.agent.cache import OperationCache
from repoguard.llm import LLMService
from repoguard.retrieval.retriever import Retriever
from repoguard.schemas import Evidence, GenerationResult, IndexArtifacts, RLAction
from repoguard.verification.claim_extractor import extract_claims
from repoguard.verification.models import Claim, VerdictStatus, VerificationResult
from repoguard.verification.verifier import verify

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    action: RLAction
    success: bool
    output: Any
    tokens_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class ActionExecutor:
    """Executes environment actions against repository indexes and frozen LLM."""

    def __init__(
        self,
        repository_id: str,
        index: IndexArtifacts,
        retriever: Retriever,
        repo_path: Path | None = None,
        llm_service: LLMService | None = None,
        cache: OperationCache | None = None,
    ) -> None:
        self.repository_id = repository_id
        self.index = index
        self.retriever = retriever
        self.repo_path = repo_path
        self.commit = index.repository.commit
        self.llm_service = llm_service or LLMService(mock_mode=True)
        self.cache = cache

    def execute_search(self, question: str, k: int = 5) -> ActionResult:
        """Action 1: SEARCH - retrieve relevant passages."""
        cache_inputs = {"question": question, "k": k}
        if self.cache and self.cache.has("search", self.repository_id, self.commit, cache_inputs):
            raw_ev = self.cache.get("search", self.repository_id, self.commit, cache_inputs)
            evidences = [Evidence.model_validate(e) for e in raw_ev]
            return ActionResult(action=RLAction.SEARCH, success=True, output=evidences, tokens_used=0)

        res = self.retriever.search(question, k=k)
        evidences = res.evidence
        if self.cache:
            self.cache.set(
                "search",
                self.repository_id,
                self.commit,
                cache_inputs,
                [e.model_dump() for e in evidences],
            )
        return ActionResult(action=RLAction.SEARCH, success=True, output=evidences, tokens_used=0)

    def execute_inspect(self, evidence: Evidence) -> ActionResult:
        """Action 2: INSPECT - expand symbol or file lines."""
        cache_inputs = {
            "file_path": evidence.file_path,
            "start_line": evidence.start_line,
            "end_line": evidence.end_line,
            "symbol": evidence.associated_symbol,
        }
        if self.cache and self.cache.has("inspect", self.repository_id, self.commit, cache_inputs):
            content = self.cache.get("inspect", self.repository_id, self.commit, cache_inputs)
            tokens = max(1, len(content) // 4)
            return ActionResult(
                action=RLAction.INSPECT,
                success=True,
                output=content,
                tokens_used=tokens,
            )

        snippet = evidence.snippet
        if self.repo_path:
            target_file = self.repo_path / evidence.file_path
            if target_file.is_file():
                try:
                    lines = target_file.read_text(encoding="utf-8", errors="replace").splitlines()
                    # Expand lines with 5 lines of padding
                    s = max(0, evidence.start_line - 5)
                    e = min(len(lines), evidence.end_line + 5)
                    snippet = "\n".join(lines[s:e])
                except Exception as exc:
                    logger.warning("Could not read source file for inspection: %s", exc)

        tokens = max(1, len(snippet) // 4)
        if self.cache:
            self.cache.set("inspect", self.repository_id, self.commit, cache_inputs, snippet)
        return ActionResult(
            action=RLAction.INSPECT,
            success=True,
            output=snippet,
            tokens_used=tokens,
        )

    def execute_verify(self, draft_text: str) -> ActionResult:
        """Action 3: VERIFY - extract claims and verify against index."""
        claims = extract_claims(draft_text)
        claims_json = [c.model_dump() for c in claims]
        cache_inputs = {"claims": claims_json}

        if self.cache and self.cache.has("verify", self.repository_id, self.commit, cache_inputs):
            raw_res = self.cache.get("verify", self.repository_id, self.commit, cache_inputs)
            v_res = VerificationResult.model_validate(raw_res)
            return ActionResult(action=RLAction.VERIFY, success=True, output=v_res, tokens_used=0)

        v_res = verify(claims, self.index)
        if self.cache:
            self.cache.set("verify", self.repository_id, self.commit, cache_inputs, v_res.model_dump())
        return ActionResult(action=RLAction.VERIFY, success=True, output=v_res, tokens_used=0)

    def execute_answer(self, question: str) -> ActionResult:
        """Action 4: ANSWER - generate direct answer from frozen LLM."""
        cache_inputs = {"question": question, "type": "direct"}
        if self.cache and self.cache.has("answer", self.repository_id, self.commit, cache_inputs):
            raw = self.cache.get("answer", self.repository_id, self.commit, cache_inputs)
            res = GenerationResult.model_validate(raw)
            return ActionResult(
                action=RLAction.ANSWER,
                success=True,
                output=res.text,
                tokens_used=res.total_tokens,
                metadata={"entropy": res.entropy, "min_token_prob": res.min_token_prob},
            )

        prompt = f"Answer the developer question directly based on your knowledge:\nQuestion: {question}"
        res = self.llm_service.generate(prompt, max_new_tokens=128, return_token_stats=True)
        if self.cache:
            self.cache.set("answer", self.repository_id, self.commit, cache_inputs, res.model_dump())
        return ActionResult(
            action=RLAction.ANSWER,
            success=True,
            output=res.text,
            tokens_used=res.total_tokens,
            metadata={"entropy": res.entropy, "min_token_prob": res.min_token_prob},
        )

    def execute_answer_with_evidence(self, question: str, evidence: list[Evidence]) -> ActionResult:
        """Action 5: ANSWER_WITH_EVIDENCE - generate grounded answer using code context."""
        context = "\n\n".join(
            f"# {e.file_path}:{e.start_line}-{e.end_line}\n{e.snippet}"
            for e in evidence
        )
        cache_inputs = {"question": question, "context_hash": hashlib.sha256(context.encode()).hexdigest()}
        if self.cache and self.cache.has("answer_with_evidence", self.repository_id, self.commit, cache_inputs):
            raw = self.cache.get("answer_with_evidence", self.repository_id, self.commit, cache_inputs)
            res = GenerationResult.model_validate(raw)
            return ActionResult(
                action=RLAction.ANSWER_WITH_EVIDENCE,
                success=True,
                output=res.text,
                tokens_used=res.total_tokens,
                metadata={"entropy": res.entropy, "min_token_prob": res.min_token_prob},
            )

        prompt = (
            f"You are a repository-grounded assistant. Answer strictly using this code:\n"
            f"{context}\n\nQuestion: {question}"
        )
        res = self.llm_service.generate(prompt, max_new_tokens=128, return_token_stats=True)
        if self.cache:
            self.cache.set(
                "answer_with_evidence",
                self.repository_id,
                self.commit,
                cache_inputs,
                res.model_dump(),
            )
        return ActionResult(
            action=RLAction.ANSWER_WITH_EVIDENCE,
            success=True,
            output=res.text,
            tokens_used=res.total_tokens,
            metadata={"entropy": res.entropy, "min_token_prob": res.min_token_prob},
        )

    def execute_abstain(self, nearest_symbol: str | None = None) -> ActionResult:
        """Action 6: ABSTAIN - honest refusal with closest real matches."""
        if nearest_symbol:
            text = f"I cannot find this symbol in the repository. Closest real match: `{nearest_symbol}`."
        else:
            text = "The repository does not contain sufficient evidence to answer this question."
        return ActionResult(action=RLAction.ABSTAIN, success=True, output=text, tokens_used=0)
