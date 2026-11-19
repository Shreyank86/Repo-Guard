# Team Responsibilities and Task Allocation

This document outlines the clear division of responsibilities for the three engineering team members developing **RepoGuard**.

---

## Member 1 — Repository Intelligence & Retrieval Engineer

### Primary Ownership:
- `src/repoguard/repository/`
  - `parser.py`: Python AST parsing, symbol extraction, parameter detection, line ranges, and path traversal protection.
  - `callgraph.py`: Static call graph generation mapping caller-to-callee relationships.
  - `passage_builder.py`: Splitting repositories into semantically bounded code passages.
  - `indexer.py`: Orchestrating the extraction pipeline.
- `src/repoguard/retrieval/`
  - `embedder.py`: Vector embeddings using Sentence Transformers (with Mock fallback).
  - `search.py`: FAISS vector index creation and similarity search.
- `scripts/build_index.py`: CLI tool for indexing a repository.

### Working Branch:
`feature/indexer`

### Expected Outputs & Deliverables:
1. `symbols.json`: Formatted symbol table containing all classes, functions, parameters, and line spans.
2. `callgraph.json`: Directed call graph mapping caller symbols to callee functions.
3. `passages.jsonl`: Document chunks with precise file paths and line ranges.
4. FAISS index binary file for fast vector lookup.
5. Unit tests in `tests/unit/test_parser.py`, `tests/unit/test_callgraph.py`, and `tests/unit/test_indexer.py`.

### Shared Interface Contracts to Comply With:
- `build_index(repo_path: str, output_dir: str, config: dict) -> IndexArtifacts`
- `search(question: str, index: IndexArtifacts, top_k: int) -> list[Evidence]`
- Models: `RepositoryMetadata`, `SymbolMetadata`, `FunctionMetadata`, `ClassMetadata`, `CallEdge`, `Passage`, `Evidence`, `IndexArtifacts`.

---

## Member 2 — Verification & Quality Engineer

### Primary Ownership:
- `src/repoguard/verification/`
  - `claim_extractor.py`: Extracting factual code claims from model answers (files, functions, classes, parameters, calls).
  - `verifier.py`: Validating extracted claims against `symbols.json` and `callgraph.json`.
- `tests/`
  - `tests/fixtures/sample_repo/`: Standardized deterministic repository fixture.
  - Negative and adversarial test cases (hallucinated parameters, non-existent functions, wrong call edges).
- `src/repoguard/evaluation/metrics.py`: Implementing verification precision, hallucination rate, and abstention accuracy.

### Working Branch:
`feature/verifier`

### Expected Outputs & Deliverables:
1. Deterministic verification module checking:
   - `file_exists`: Verifies file is in the repository.
   - `function_exists`: Verifies function is defined in target file.
   - `class_exists`: Verifies class is defined in target file.
   - `parameter_exists`: Verifies parameter is in function's AST signature.
   - `calls`: Verifies caller-to-callee edge exists in static call graph.
2. Adversarial test suite with known hallucinated claims to test verifier robustness.
3. Unit tests in `tests/unit/test_verifier.py` and `tests/unit/test_claim_extractor.py`.

### Shared Interface Contracts to Comply With:
- `verify(claims: list[Claim], index: IndexArtifacts) -> VerificationResult`
- `extract_claims(text: str) -> list[Claim]`
- Models: `Claim`, `ClaimType`, `ClaimVerdict`, `VerdictStatus`, `VerificationResult`.

---

## Member 3 — Frozen LLM & Reinforcement Learning Engineer

### Primary Ownership:
- `src/repoguard/llm/`
  - `model_loader.py`: Loading frozen code LLM (Qwen2.5-Coder 0.5B/1.5B) with 4-bit quantization and CPU/Mock fallback.
  - `prompt_builder.py`: Designing grounded prompts with repository evidence and backtick claim instructions.
  - `inference.py`: Generation interface with token counting and latency tracking.
- `src/repoguard/agent/`
  - `actions.py`: Discrete action space (`SEARCH`, `INSPECT`, `VERIFY`, `ANSWER`, `ANSWER_WITH_EVIDENCE`, `ABSTAIN`).
  - `state.py`: 18-dimensional state vector representation.
  - `reward.py`: Multi-objective reward function.
  - `environment.py`: Gym-compatible `RepoGuardEnv` environment.
  - `policy.py`: PyTorch Deep Q-Network (DQN) policy network and baseline heuristic controller.
  - `controller.py`: High-level agent orchestration.
- `scripts/`
  - `scripts/run_baseline.py` & `scripts/evaluate_baseline.py`
  - `scripts/train_agent.py` & `scripts/evaluate_agent.py`

### Working Branch:
`feature/llm-baseline`

### Expected Outputs & Deliverables:
1. Frozen LLM inference module with token & latency logging.
2. 18-dimensional state feature vector implementation.
3. Multi-objective reward calculation handling verification outcomes and abstention rewards.
4. Trainable PyTorch DQN policy with replay buffer and epsilon-greedy exploration.
5. Baseline rule-based controller for rigorous comparative benchmarking.
6. Training and evaluation scripts with MLflow logging.

### Shared Interface Contracts to Comply With:
- `generate(prompt: str, config: dict) -> GenerationResult`
- `run_agent(question: str, repository_id: str) -> QueryResponse`
- Models: `AgentState`, `AgentAction`, `AgentTransition`, `ActionStep`, `GenerationResult`, `QueryRequest`, `QueryResponse`.

---

## Shared / Collaborative Tasks

All three team members collaborate on:
- API endpoints in `src/repoguard/api/` (Member 1: Repositories route, Member 2: Health/status, Member 3: Query route).
- End-to-end integration tests in `tests/integration/test_api.py`.
- Docker Compose deployment and verification.
- Writing research reports and final project documentation.

