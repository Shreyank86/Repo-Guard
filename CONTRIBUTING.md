# Contributing to RepoGuard

Welcome to the RepoGuard project! This guide outlines the collaboration protocols, branch strategies, and interface compliance rules for our three-member engineering team.

---

## 1. Team Roles & Ownership

| Member | Workstream | Primary Directory | Core Responsibilities |
| :--- | :--- | :--- | :--- |
| **Member 1** | Repository Intelligence | `src/repoguard/repository/`, `src/repoguard/retrieval/` | AST parsing, symbol indexing, call graph generation, FAISS passage indexing |
| **Member 2** | Verification & Quality | `src/repoguard/verification/`, `tests/` | Claim extraction, structural verification against symbols, adversarial test suite |
| **Member 3** | Frozen LLM & RL Agent | `src/repoguard/llm/`, `src/repoguard/agent/`, `scripts/` | Frozen LLM inference, 18-dim RL state & environment, DQN training, baseline |

---

## 2. Git Branch Strategy

* `main`: Protected, always stable, passes CI tests. Production and demo branch.
* `feature/indexer`: Member 1 working branch for repository intelligence and retrieval.
* `feature/verifier`: Member 2 working branch for structural fact-checking and fixtures.
* `feature/llm-baseline`: Member 3 working branch for LLM inference, RL environment, and policy.
* `fix/<topic>`: Critical bugfixes.

### Commit Conventions
Follow conventional commits:
* `feat(indexer): add method parameter parsing in AST walker`
* `feat(verifier): add check for call relationship existence`
* `feat(agent): implement 18-dim state tensor normalization`
* `test(api): add tests for query endpoint with mock engine`
* `docs(arch): update cloud deployment guide`

---

## 3. The Golden Rule: Shared Interface Immutability

The shared data models and service contracts are located in:
* `src/repoguard/schemas.py`

> [!CAUTION]
> **No team member may independently modify `src/repoguard/schemas.py` on their feature branch.**
> If your module requires schema changes:
> 1. Discuss with all three members during daily sync.
> 2. Agree on backward-compatible additions or deprecations.
> 3. Open a dedicated PR modifying `schemas.py` and updating mock fixtures together.

---

## 4. Pull Request Protocol

1. Ensure code passes local formatting and linting:
   ```bash
   ruff check src/ tests/
   ruff format --check src/ tests/
   ```
2. Ensure test suite passes with mock mode:
   ```bash
   pytest tests/
   ```
3. Open PR against `main`. Provide:
   - Summary of changes.
   - Evidence of passing tests.
   - Confirmation that hardware limits (e.g. CPU/mock testability) are preserved.
4. Requires approval from at least one peer member before merge.

