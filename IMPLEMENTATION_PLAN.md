# RepoGuard — Phase 2: Local Implementation Plan (Member 3)
## Core RL Engineering, Environment, Policy Networks & Colab Handoff

> **Workstream:** Member 3 — RL Agent Development, Environment, Policy Training & Evaluation  
> **Execution Boundary:** **Local Windows Machine Only** (Code, Architecture, Contracts, Unit Tests & Colab Pipeline)  
> **Post-Completion Hand-off:** Cloud Training on Google Colab GPU + Google Drive Checkpoints  
> **Primary Decisions:**
> - **Action Space:** Canonical **6 actions** (`SEARCH`, `INSPECT`, `VERIFY`, `ANSWER`, `ANSWER_WITH_EVIDENCE`, `ABSTAIN`).
> - **State Space:** Canonical **18-dimensional normalized observation vector** (`StateVector`).
> - **Reward Function:** Dynamic risk-conditioned formula $r = R_{\text{final}}(\lambda) - 0.2 \cdot \frac{T}{2000} - 0.01 \cdot n$, where $\lambda \in [0.25, 5.0]$.
> - **Frozen Model:** `Qwen2.5-Coder` + `MockLLM` (allows full local verification without GPU bottlenecks).

---

## 1. Local vs. Cloud Responsibility Matrix

To protect your time and local machine resources while guaranteeing smooth cloud execution, the work is strictly partitioned:

| Work Component | Execution Location | Responsibility / Deliverable |
|---|---|---|
| **Core Architecture & Code** | **Local (Windows)** | All Python modules under `src/repoguard/agent/` (`environment.py`, `dqn.py`, `reinforce.py`, `cache.py`, `policy.py`). |
| **Multi-Repo Indexing** | **Local (Windows)** | Sourcing and indexing the 4 remaining repositories (`python-dotenv`, `cachetools`, `tabulate`, `requests`). |
| **Unit & Integration Tests** | **Local (Windows)** | 100% test coverage using `MockLLM` and deterministic fixtures before any cloud run. |
| **CLI Runners & Scripts** | **Local (Windows)** | Production scripts: `run_episodes.py`, `train_agent.py`, `evaluate_rl.py`. |
| **Colab Orchestration Notebook** | **Local (Windows)** | Complete, tested notebook (`notebooks/01_train_colab.ipynb`) ready to run in one click on Colab. |
| **Heavy / Long Training Runs** | **Cloud (Google Colab)** | Running 2,000+ episode training runs on Colab GPU with Google Drive checkpoint persistence. |
| **Final Benchmark Evaluation** | **Cloud (Google Colab)** | Running live model evaluations on held-out test splits. |

---

## 2. Local Module Architecture (`src/repoguard/agent/`)

All code is developed within the existing package structure so that git pushes to GitHub automatically make it available in Google Colab:

```text
src/repoguard/agent/
├── __init__.py             # Public exports: RepoGuardEnv, RepoGuardPolicy, OperationCache
├── cache.py                # Operation-level disk cache (SEARCH, INSPECT, VERIFY, LLM outputs)
├── action_executor.py      # Executes tool calls and LLM generation for the 6 RLAction variants
├── environment.py          # Gymnasium-compatible RL environment (reset, step, state, reward)
├── dqn.py                  # Q-Network MLP, ReplayBuffer, Target Net, Epsilon-greedy exploration
├── reinforce.py            # Policy Network MLP, trajectory collector, baseline subtraction
└── policy.py               # Phase 3 production serving interface (RepoGuardPolicy)

scripts/
├── run_episodes.py         # Offline episode transition generator (pre-caches environment rollouts)
├── train_agent.py          # Unified CLI training runner (supports local CPU/GPU & Colab GPU)
└── evaluate_rl.py          # Evaluation script: risk curves, coverage, hallucination rate, ablations

notebooks/
└── 01_train_colab.ipynb    # End-to-end Colab notebook for Drive mounting, training & evaluation
```

---

## 3. Local Milestones to Execute

### Milestone 1: Multi-Repo Sourcing & Index Verification — [COMPLETED]
* **Objective:** Ensure all 5 repositories from Member 2's benchmark (`schedule`, `python-dotenv`, `cachetools`, `python-tabulate`, and `requests`) are cloned and indexed with schema `1.1` so that multi-repo episodes execute locally without `IndexNotFoundError`.
* **Deliverables:**
  - Cloned all 4 remaining repositories with pinned commits into `data/repos/`.
  - Built semantic vector indexes (`sentence-transformers/all-MiniLM-L6-v2`, dim 384, schema 1.1) for all 5 repositories.
* **Verification:** Generated 180 baseline predictions across all splits with 0 index errors.

---

### Milestone 2: Operation-Level Persistent Cache (`cache.py`) — [COMPLETED]
* **Objective:** Implement a fine-grained, persistent disk cache for individual operations so that episodes can be replayed or generated without repeatedly calling heavy models or retrievers.
* **Deliverables:**
  - `src/repoguard/agent/cache.py`: `OperationCache` class backed by SQLite at `data/cache/operations.db`.
  - Cache key: `sha256(repo_id + commit + op_type + canonical_inputs)`.
* **Verification:** Unit tests in `tests/unit/test_agent_cache.py`: 4 passed.

---

### Milestone 3: Action Executor & RL Environment (`environment.py`) — [COMPLETED]
* **Objective:** Build the Gymnasium-compatible MDP environment managing state transitions, action execution, and dynamic rewards.
* **Deliverables:**
  - `src/repoguard/agent/action_executor.py`: Dispatches each of the 6 canonical `RLAction` choices.
  - `src/repoguard/agent/environment.py`: `RepoGuardEnv` with `spaces.Box(18,)` and `spaces.Discrete(6)`, step limits, action masking, and dynamic reward formula $r = R_{\text{final}}(\lambda) - 0.2 \cdot \frac{T}{2000} - 0.01 \cdot n$.
* **Verification:** Unit tests in `tests/unit/test_environment.py`: 5 passed.

---

### Milestone 4: Deep Q-Network (DQN) Implementation (`dqn.py`) — [COMPLETED]
* **Objective:** Implement the primary Q-learning agent with experience replay, target networks, and exploration scheduling.
* **Deliverables:**
  - `QNetwork` MLP (18 -> 128 -> 64 -> 6).
  - `ReplayBuffer` with action masks (capacity 20,000).
  - `DQNAgent` with epsilon-greedy masked action selection, Huber loss, target network sync.
* **Verification:** Unit tests in `tests/unit/test_dqn.py`: 5 passed.

---

### Milestone 5: REINFORCE Policy Gradient Implementation (`reinforce.py`) — [COMPLETED]
* **Objective:** Implement the policy-gradient baseline for research comparison.
* **Deliverables:**
  - `PolicyNetwork` MLP with categorical distribution and masked logits.
  - `REINFORCEAgent` with discounted cumulative returns $G_t$ and running-mean baseline subtraction.
* **Verification:** Unit tests in `tests/unit/test_reinforce.py`: 4 passed.

---

### Milestone 6: Offline Episode Collector & Training Runner (`train_agent.py`) — [COMPLETED]
* **Objective:** Create scripts to collect transitions and train agents both locally (fast smoke tests) and on cloud (full training).
* **Deliverables:**
  - `scripts/run_episodes.py`: Rolls out benchmark questions across actions, logging state transitions to `data/episodes/episode_log.jsonl` (verified with 240 transitions).
  - `scripts/train_agent.py`: Unified CLI runner supporting `--algo dqn|reinforce`, device selection, seeds, random $\lambda \sim \text{LogUniform}(0.25, 5.0)$, periodic checkpoints.
* **Verification:** Local CUDA training runs for DQN (30 episodes) and REINFORCE (20 episodes) completed in <15 seconds.

---

### Milestone 7: Evaluation, Risk Curves & Phase 3 Serving Interface — [COMPLETED]
* **Objective:** Provide automated evaluation tools and the production-ready inference class.
* **Deliverables:**
  - `scripts/evaluate_rl.py`: Evaluates trained policy against baselines across $\lambda \in \{0.25, 0.5, 1.0, 2.0, 3.0, 5.0\}$. Generated `reports/rl_evaluation_results.csv`.
  - `src/repoguard/agent/policy.py`: Production-serving `RepoGuardPolicy` class with `select_action` and `get_action_probabilities`.
* **Verification:** Unit tests in `tests/unit/test_policy_serving.py`: 3 passed.

---

### Milestone 8: Google Colab Setup & Training Guide Package — [COMPLETED]
* **Objective:** Package the Colab notebook and step-by-step walkthrough so you can execute the heavy 2,000+ episode training runs on Google Colab with one click.
* **Deliverables:**
  - `notebooks/01_train_colab.ipynb`: Complete 11-step end-to-end Colab notebook for GPU training, Google Drive checkpoint mounting, and evaluation visualizations.
  - `docs/COLAB_TRAINING_GUIDE.md`: Comprehensive walkthrough covering GitHub push, runtime configuration, Drive persistence, training execution, and troubleshooting.
* **Verification:** Validated `.ipynb` JSON formatting and local CLI commands.

---

## 4. Local Execution Roadmap

```
[M1: Multi-Repo Sourcing] ────> [M2: Operation Cache] ────> [M3: RL Environment]
                                                                     │
                                                                     v
[M6: Training Scripts]    <──── [M5: REINFORCE]        <──── [M4: DQN Agent]
         │
         v
[M7: Evaluation & Serving] ───> [M8: Colab Package & Hand-off]
```

Once Milestone 8 is validated locally, the code is committed to GitHub, and training in Colab runs with zero configuration issues.
