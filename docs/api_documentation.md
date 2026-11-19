# RepoGuard REST API Documentation

The RepoGuard backend is built with FastAPI. Interactive Swagger UI is available at `/docs` and ReDoc at `/redoc`.

---

## 1. System Health

### `GET /health`
Returns the status of the server and individual subsystems.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "app_name": "RepoGuard",
  "version": "0.1.0",
  "environment": "development",
  "mock_mode": true,
  "components": [
    {
      "name": "repository_indexer",
      "status": "operational",
      "available": true,
      "message": "AST parser and call graph ready"
    },
    {
      "name": "retrieval_engine",
      "status": "operational",
      "available": true,
      "message": "Mock embedder active"
    },
    {
      "name": "verification_engine",
      "status": "operational",
      "available": true,
      "message": "Structural verifier ready"
    },
    {
      "name": "llm_inference",
      "status": "operational",
      "available": true,
      "message": "Mock LLM generator active"
    },
    {
      "name": "rl_agent",
      "status": "operational",
      "available": true,
      "message": "DQN policy & baseline controller loaded"
    }
  ]
}
```

---

## 2. Repository Management

### `POST /repositories`
Registers a repository for scanning and indexing. Validates paths to prevent path traversal attacks.

**Request Body:**
```json
{
  "name": "Sample Calculator Project",
  "repo_path": "tests/fixtures/sample_repo",
  "reindex": false
}
```

**Response (201 Created):**
```json
{
  "repository_id": "sample-calculator-project",
  "name": "Sample Calculator Project",
  "repo_path": "tests/fixtures/sample_repo",
  "status": "ready",
  "total_files": 3,
  "total_symbols": 8,
  "created_at": "2026-11-19T12:00:00Z"
}
```

### `GET /repositories`
List all registered repositories.

### `GET /repositories/{repository_id}`
Get details and metadata for a specific repository.

### `GET /repositories/{repository_id}/status`
Get current indexing progress and any error messages.

---

## 3. Query & Question Answering

### `POST /query`
Executes an end-to-end question answering episode through the Risk-Conditioned RL Agent and Verification Engine.

**Request Body:**
```json
{
  "repository_id": "sample-calculator-project",
  "question": "How does the Calculator class compute subtraction?",
  "risk_tolerance": 0.5,
  "max_steps": 6
}
```

**Response (200 OK):**
```json
{
  "query_id": "q-123456",
  "repository_id": "sample-calculator-project",
  "question": "How does the Calculator class compute subtraction?",
  "answer": "The `Calculator` class defines `subtract(a, b)` which returns `a - b`.",
  "abstained": false,
  "abstention_reason": null,
  "evidences": [
    {
      "evidence_id": "ev-1",
      "file_path": "calculator.py",
      "start_line": 15,
      "end_line": 20,
      "snippet": "    def subtract(self, a: float, b: float) -> float:\n        return a - b",
      "similarity_score": 0.88,
      "associated_symbol": "Calculator.subtract"
    }
  ],
  "verification": {
    "is_valid": true,
    "verdicts": [
      {
        "claim": {
          "claim_id": "c-1",
          "claim_type": "class_exists",
          "subject": "Calculator"
        },
        "status": "verified",
        "explanation": "Class Calculator found in calculator.py",
        "ground_truth_reference": "calculator.py:Calculator"
      },
      {
        "claim": {
          "claim_id": "c-2",
          "claim_type": "function_exists",
          "subject": "subtract"
        },
        "status": "verified",
        "explanation": "Method subtract found in class Calculator",
        "ground_truth_reference": "calculator.py:Calculator.subtract"
      }
    ],
    "verified_count": 2,
    "refuted_count": 0,
    "unverifiable_count": 0,
    "verified_ratio": 1.0,
    "summary": "All 2 code claims verified against AST symbols."
  },
  "action_trace": [
    {
      "step_number": 1,
      "action": "SEARCH",
      "reasoning": "Initial evidence retrieval"
    },
    {
      "step_number": 2,
      "action": "VERIFY",
      "reasoning": "Extracted claims verified against AST"
    },
    {
      "step_number": 3,
      "action": "ANSWER_WITH_EVIDENCE",
      "reasoning": "High confidence verified answer produced"
    }
  ],
  "risk_score": 0.12,
  "total_tokens": 142,
  "latency_ms": 235.4,
  "is_mock": true
}
```

