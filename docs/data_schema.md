# Data Contracts & Schemas Documentation

All three team members must work against the agreed Pydantic models defined in `src/repoguard/schemas.py`.

---

## 1. Core Data Models

### 1.1 Repository & Code Entities
- **`RepositoryMetadata`**: Contains repository identity, root filesystem path, total file counts, total symbol counts, and current indexing status (`pending`, `indexing`, `ready`, `failed`).
- **`FileMetadata`**: Line counts, list of import statements, functions, and classes within a Python file.
- **`FunctionMetadata`**: Subclass of `SymbolMetadata`. Contains parameter list (with annotations & defaults), return annotations, async flag, and parent class name if a method.
- **`ClassMetadata`**: Subclass of `SymbolMetadata`. Contains base classes, method names, and class variable names.
- **`CallEdge`**: Directed invocation edge: `(caller_file, caller_symbol) -> callee_name` at `line_number`.

### 1.2 Retrieval & Evidence
- **`Passage`**: Bounded text excerpt from a file with `passage_id`, `file_path`, `start_line`, `end_line`, and `content`.
- **`Evidence`**: Search result passage returned to the agent with a normalized `similarity_score` (0.0 to 1.0) and associated symbol.
- **`IndexArtifacts`**: Container holding file paths and metadata for `symbols.json`, `callgraph.json`, `passages.jsonl`, and FAISS index.

### 1.3 Claims & Verification
- **`ClaimType`**: Enum with values:
  - `file_exists`
  - `function_exists`
  - `class_exists`
  - `parameter_exists`
  - `calls`
  - `imports`
- **`Claim`**: Factual assertion extracted from text:
  - `claim_type`: One of `ClaimType`.
  - `subject`: Entity name (e.g. `Calculator`).
  - `target`: Target entity or parameter (e.g. `add`).
  - `details`: Contextual dictionary.
- **`ClaimVerdict`**:
  - `status`: `verified`, `refuted`, or `unverifiable`.
  - `explanation`: Reason for the verdict.
  - `ground_truth_reference`: Exact matching symbol or call edge from index.
- **`VerificationResult`**:
  - `is_valid`: Boolean overall verdict.
  - `verdicts`: List of individual `ClaimVerdict` items.
  - `verified_ratio`: Fraction of claims verified.

### 1.4 Reinforcement Learning
- **`AgentAction`**: Enum: `SEARCH`, `INSPECT`, `VERIFY`, `ANSWER`, `ANSWER_WITH_EVIDENCE`, `ABSTAIN`.
- **`AgentState`**: 18-dimensional feature vector:
  1. `query_length_norm`: Normalized question length.
  2. `query_complexity`: Heuristic query complexity.
  3. `retrieval_count_norm`: Retrieved evidence count / maximum.
  4. `max_retrieval_score`: Top similarity score.
  5. `mean_retrieval_score`: Average similarity score.
  6. `evidence_coverage`: Ratio of query entities covered.
  7. `claim_count_norm`: Number of extracted claims / maximum.
  8. `verified_claim_ratio`: Fraction of verified claims.
  9. `refuted_claim_ratio`: Fraction of refuted claims.
  10. `unverifiable_claim_ratio`: Fraction of unverifiable claims.
  11. `uncertainty_score`: Model perplexity or uncertainty estimate.
  12. `risk_tolerance`: User-configured risk preference (0.0 to 1.0).
  13. `step_count_norm`: Current episode step / budget.
  14. `budget_remaining_norm`: Remaining step budget.
  15. `token_usage_norm`: Cumulative tokens consumed / budget.
  16. `has_searched`: 1.0 if SEARCH was executed, else 0.0.
  17. `has_inspected`: 1.0 if INSPECT was executed, else 0.0.
  18. `has_verified`: 1.0 if VERIFY was executed, else 0.0.
- **`AgentTransition`**: `(state, action, reward, next_state, done, info)`.

### 1.5 API Contracts
- **`QueryRequest`**: `{ repository_id: str, question: str, risk_tolerance: float, max_steps: int }`
- **`QueryResponse`**: `{ query_id, repository_id, question, answer, abstained, evidences, verification, action_trace, risk_score, total_tokens, latency_ms }`

