# Research Methodology and Experimental Framework

This document outlines the scientific methodology, benchmarking protocol, and evaluation metrics for RepoGuard.

---

## 1. Research Hypothesis

> **Hypothesis**: A risk-conditioned reinforcement learning agent orchestrating a frozen code language model, static AST verification, and passage retrieval will achieve higher answer precision, lower hallucination rates, and reduced token expenditure compared to direct zero-shot generation and standard static RAG pipelines, primarily by learning to verify claims and abstain when supporting evidence is insufficient.

---

## 2. Baselines for Comparison

1. **Zero-Shot LLM**: Code LLM prompted directly with query and full prompt without retrieval or verification.
2. **Static RAG**: Standard dense retrieval (top-K FAISS passages) concatenated with query, generating answers without verification or abstention.
3. **Rule-Based Heuristic Baseline**: A deterministic state machine that retrieves, generates, verifies, and abstains if verification ratio is below a static threshold ($0.8$).
4. **RepoGuard RL Agent**: Trainable DQN policy operating over the 18-dimensional state vector learning dynamic decision boundaries.

---

## 3. Evaluation Metrics

### 3.1 Verification & Hallucination Metrics
- **Claim Precision**: Fraction of generated code claims that are verified true against repository AST.
- **Hallucination Rate**: Ratio of refuted claims to total extracted claims.
- **Verification Ratio**: $\frac{\text{Verified Claims}}{\text{Total Claims}}$.

### 3.2 Retrieval Metrics
- **Recall@K**: Proportion of relevant code passages retrieved in top-$K$.
- **MRR (Mean Reciprocal Rank)**: Reciprocal rank of the first relevant passage.

### 3.3 Abstention Metrics
- **Abstention Accuracy**: Correct abstentions when answers are impossible or evidence missing.
- **False Abstention Rate**: Abstentions triggered when verifiable answers were available.

### 3.4 Efficiency Metrics
- **Average Token Consumption**: Total prompt + completion tokens per query.
- **Decision Steps per Query**: Number of agent actions taken before termination.
- **End-to-End Latency (ms)**.

---

## 4. Experiment Tracking with MLflow

All training and evaluation runs record:
- Git commit hash.
- Configuration parameters (`model_config.yaml`, `rl_config.yaml`).
- Random seed.
- Epoch-by-epoch reward curves, Q-value loss, and epsilon decay.
- Evaluation metrics across test benchmarks.

