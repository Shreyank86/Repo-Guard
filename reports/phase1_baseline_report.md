# RepoGuard Phase 1: Frozen LLM & Baseline Experimentation Report

**Author / Owner:** Member 3 (Frozen LLM & Reinforcement Learning Engineer)  
**Branch:** `llm`  
**Date:** November 22, 2026  
**Target Repository:** `dbader/schedule` (`git commit: 82a43db1b938d8fdf60103bd41f329e06c8d3651`, MIT License)  
**Model:** `Qwen/Qwen2.5-Coder-3B-Instruct` (NF4 4-bit quantized via `bitsandbytes`)  

---

## 1. Executive Summary

In Phase 1 of **RepoGuard**, Member 3 established the frozen code language model inference infrastructure and executed the 100-question repository-grounded benchmark suite across three baseline systems:
1. **Baseline 1 (`always_answer`)**: Zero-shot ungrounded generation.
2. **Baseline 2 (`always_evidence`)**: Standard dense retrieval-augmented generation (RAG).
3. **Baseline 3 (`threshold_gate`)**: Static confidence-gated RAG that abstains if similarity < 0.45.

Across 300 executed evaluations (100 benchmark questions $\times$ 3 baselines: 50 ground-truth answerable questions and 50 adversarial hallucination traps), the empirical results conclusively demonstrate the **fundamental failure modes of naive static strategies**:
* Zero-shot and standard RAG suffer an intolerable **50.0% Hallucination Rate** on adversarial traps.
* Naive RAG inflates token consumption by **3.22x** without reducing hallucinations on nonexistent symbols.
* Static similarity gating eliminates hallucinations but incurs a catastrophic **74.0% Over-refusal Rate**, collapsing system coverage to a mere **13.0%**.

These empirical findings provide the definitive scientific justification for **Phase 2 Reinforcement Learning**, where an adaptive policy agent dynamically balances retrieval depth, verification rigor, and abstention risk.

---

## 2. Experimental Setup & Hardware Constraints

### 2.1 Hardware Environment
* **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB GDDR6 VRAM, TGP 75W, Compute Capability 8.6).
* **Host Platform:** Windows 11 x64, Driver 581.86, CUDA 13.0.
* **Software Stack:** Python 3.10, PyTorch 2.5.1+cu121, Transformers 4.41.2, BitsAndBytes 0.42.0, Accelerate 1.15.0.

### 2.2 VRAM Optimization & Engineering Resolution
To operate strictly within the **6.0 GB VRAM budget**, the 3-billion-parameter model was loaded in 4-bit NormalFloat4 (NF4) with double quantization and float16 compute:
* **Static Model Footprint:** ~2,150 MB VRAM.
* **Peak Generation VRAM:** ~2,810 MB VRAM.
* **WDDM Paging Fix:** During initial testing, consecutive unbatched generations caused PyTorch's caching allocator to retain KV caches across questions. Once total memory exceeded ~5.2 GB, Windows WDDM paged CUDA allocations into shared system RAM across PCIe, spiking latency from 8s to over 90s per question. This was permanently resolved by:
  1. Adding explicit device buffer pinning (`torch.nn.Module.to(model, "cuda:0")`).
  2. Enforcing `torch.cuda.empty_cache()` immediately after every generation step.
  3. Constraining `max_new_tokens=128`.
  4. Executing questions in 20-question balanced shards.
* **Post-Optimization Latency:** Consistent **~8.1–8.2 seconds** per question (a **10x speedup**).

---

## 3. Quantitative Baseline Results

The final results computed over all 100 questions (50 answerable, 50 traps) and saved to [`reports/baseline_results.csv`](file:///c:/Users/shrey/Desktop/Project/Repo-Guard/reports/baseline_results.csv):

| Metric | Baseline 1: `always_answer` | Baseline 2: `always_evidence` | Baseline 3: `threshold_gate` |
| :--- | :---: | :---: | :---: |
| **Model** | `Qwen2.5-Coder-3B-Instruct` | `Qwen2.5-Coder-3B-Instruct` | `Qwen2.5-Coder-3B-Instruct` |
| **Coverage (%)** | **100.0%** | **100.0%** | **13.0%** |
| **Hallucination Rate (%)** | **50.0%** | **50.0%** | **0.0%** |
| **Accuracy on Answered (%)** | 50.0% | 50.0% | **100.0%** |
| **Correct Abstention (%)** | 0.0% | 0.0% | **100.0%** |
| **Over-refusal Rate (%)** | **0.0%** | **0.0%** | **74.0%** |
| **Mean Tokens / Query** | 181.8 | 586.0 | 67.8 |
| **Tokens / Correct Answer** | 363.6 | 1172.0 | 521.9 |
| **Median Latency (ms)** | 8,108.9 ms | 8,228.0 ms | **5.0 ms** |

---

## 4. In-Depth Failure Mode Analysis

### 4.1 Baseline 1 (`always_answer`): Parametric Fabrications
* **Mechanism:** Plain LLM without access to repository indexing or evidence.
* **Findings:** The model exhibits high fluency but confidently fabricates non-existent parameters, classes, and helper functions. On the 50 hallucination traps, it answered 100% of them, resulting in a **50.0% hallucination rate**.
* **Conclusion:** Zero-shot code LLMs cannot be trusted for repository maintenance without external grounding.

### 4.2 Baseline 2 (`always_evidence`): Retrieval-Induced Confusion
* **Mechanism:** Top-3 dense vector passages injected directly into the prompt context.
* **Findings:** While accuracy on answerable questions is solid, the presence of code evidence **fails to prevent hallucinations on traps**. When presented with a nonexistent method (e.g., `Job.run_async`), the model retrieves partially related passages (`Job.run`), misinterprets the context as permission to hypothesize, and fabricates signatures. Furthermore, prompt context expands average token usage from **181.8 to 586.0 tokens** (~3.2x cost inflation) with zero reduction in hallucination rate.
* **Conclusion:** Standard RAG is not a defense against subtle code hallucinations.

### 4.3 Baseline 3 (`threshold_gate`): The Over-Refusal Trap
* **Mechanism:** Computes the maximum FAISS cosine similarity of retrieved evidence. If `max_sim < 0.45`, the system immediately abstains without calling the LLM.
* **Findings:** This static heuristic successfully stops hallucinations on traps (0% hallucination rate, 100% correct abstention, median latency 5ms). However, semantic vector similarity in codebases is inherently noisy—queries with subtle domain phrasing or complex questions fail the rigid 0.45 threshold. Consequently, it refuses to answer **74.0% of perfectly valid, answerable repository questions**, collapsing total coverage to **13.0%**.
* **Conclusion:** Static thresholding creates an overly timid assistant that is virtually useless in production developer workflows.

---

## 5. Phase 1 Deliverables Summary (Member 3)

| Deliverable | Location | Status | Verification |
| :--- | :--- | :--- | :--- |
| **Inference Engine** | `src/repoguard/llm.py` | Complete | Tested with `tests/test_llm.py` (8/8 passing). Supports 4-bit NF4, CPU, Mock mode. |
| **Baseline Runner** | `src/repoguard/baselines.py`, `scripts/run_baseline.py` | Complete | Tested with `--limit`, `--reset`, real-time streaming JSONL logging. |
| **Evaluation Suite** | `scripts/evaluate_baseline.py` | Complete | Validated across 100 benchmark questions (300 runs). |
| **Benchmark Suite** | `data/questions/baseline_questions.jsonl` | Complete | 100 curated questions (50 answerable, 50 adversarial traps). Sharded into 5 batches. |
| **Raw Trace Data** | `data/baseline/raw_outputs.jsonl` | Complete | 300 complete records with prompt/completion tokens, latency, and abstention status. |
| **Metrics Summary** | `reports/baseline_results.csv` | Complete | Verified CSV with 10 comprehensive evaluation dimensions. |
| **Team Unit Tests** | `tests/` | Complete | 89 passed, 2 skipped across entire repository. |

---

## 6. Scientific Transition: Hand-Off to Phase 2 (RL Policy Agent)

Phase 1 conclusively proves that **neither static ungrounded generation, naive RAG, nor rigid confidence gating can satisfy the dual requirements of high coverage and low hallucination**.

In **Phase 2**, Member 3 will lead the implementation of the **Reinforcement Learning Agent**:
1. **Gym-Compatible Environment (`RepoGuardEnv`):** Discrete action space (`SEARCH`, `INSPECT`, `VERIFY`, `ANSWER`, `ANSWER_WITH_EVIDENCE`, `ABSTAIN`).
2. **18-Dimensional State Representation:** Incorporating query intent, retrieval similarities from Member 1, AST verification confidence from Member 2, and step budget signals.
3. **Multi-Objective Reward Function:** Rewarding correct answers ($+1.0$), rewarding justified abstentions ($+0.4$), heavily penalizing verified hallucinations ($-2.0$), and penalizing token and action step costs.
4. **DQN / Policy Optimization:** Training a lightweight neural policy to dynamically navigate the Pareto frontier between coverage and reliability.
