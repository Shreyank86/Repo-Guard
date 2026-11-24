# Google Colab Training & Cloud Execution Guide (Member 3)

> **Workstream:** Member 3 — RL Agent Development, Training, Evaluation & Model Integration  
> **Target Environment:** Google Colab (Free T4 GPU or Colab Pro) + Google Drive Persistence  
> **Primary Notebook:** [`notebooks/01_train_colab.ipynb`](../notebooks/01_train_colab.ipynb)  
> **Repository:** `https://github.com/Shreyank86/Repo-Guard`  
> **Core Architecture:**
> - **Action Space:** Canonical 6 actions (`SEARCH`, `INSPECT`, `VERIFY`, `ANSWER`, `ANSWER_WITH_EVIDENCE`, `ABSTAIN`)
> - **Observation Space:** 18-dimensional normalized state vector ($s \in [0, 1]^{18}$)
> - **Dynamic Risk Penalty:** $\lambda \in [0.25, 5.0]$
> - **Reward Formula:** $r = R_{\text{final}}(\lambda) - 0.2 \cdot \frac{T}{2000} - 0.01 \cdot n$
> - **Evaluation Splits:** `train` (36 cases), `validation` (12 cases), `test` (12 cases)

---

## 1. Overview & Cloud Architecture

All core engineering, environments, policies, caches, and test suites are developed and verified locally on Windows. Google Colab is used for:
1. **Accelerated Multi-Episode Training:** 2,000 episodes for Deep Q-Network (DQN) and 1,500 episodes for REINFORCE.
2. **Persistent Checkpoints & Auto-Resume:** All checkpoints are saved directly to Google Drive. If Colab disconnects overnight, re-running the training cell automatically detects the latest checkpoint and resumes seamlessly from where it stopped.
3. **Persistent Indexes & SQLite Cache:** Repository vector indexes and the SQLite operation cache (`operations.db`) are saved to Drive so they are never rebuilt or recomputed twice.
4. **Academic Split Evaluation:** Separate validation evaluations (`val_dqn.csv`) and unpolluted held-out test evaluations (`test_dqn.csv`) across all 6 risk levels ($\lambda \in \{0.25, 0.5, 1.0, 2.0, 3.0, 5.0\}$).

```
+-------------------------------------------------------------------------------------------------+
|                                    Google Colab (CUDA GPU)                                      |
|                                                                                                 |
|   1. Clone Shreyank86/Repo-Guard                                                                |
|   2. Fetch 5 benchmark repos (symlink indexes from Drive if already built)                      |
|   3. Run Pre-Training Validation Gate & Real 4-bit NF4 Qwen Model Validation                   |
|   4. train_agent.py --algo dqn --split train --auto-resume (DQN: 2000 eps, replay 20k)          |
|   5. train_agent.py --algo reinforce --split train --auto-resume (REINFORCE: 1500 eps)          |
|   6. evaluate_rl.py --split validation & evaluate_rl.py --split test (6 risk levels)            |
+-----------------------------------------------+-------------------------------------------------+
                                                |
                                                | (Atomic writes every 50 episodes)
                                                v
+-------------------------------------------------------------------------------------------------+
|                                  Google Drive Persistent Storage                                |
|                                                                                                 |
|   /MyDrive/RepoGuard/                                                                           |
|   ├── data/                                                                                     |
|   │   ├── indexes/ (Dense vector indexes for schedule, dotenv, cachetools, tabulate, requests)  |
|   │   └── cache/operations.db (SQLite cache for SEARCH, INSPECT, VERIFY, LLM outputs)           |
|   ├── checkpoints/                                                                              |
|   │   ├── dqn/ (dqn_policy_latest.pt, dqn_checkpoint_ep*.pt, training_history.json)            |
|   │   └── reinforce/ (reinforce_policy_latest.pt, reinforce_checkpoint_ep*.pt)                 |
|   └── reports/                                                                                  |
|       ├── val_dqn.csv & val_reinforce.csv (Validation split evaluation)                        |
|       ├── test_dqn.csv & test_reinforce.csv (Held-out test split evaluation)                    |
|       ├── test_risk_curves.png (Publication-quality figures)                                    |
|       └── experiment_manifest.json (Reproducibility metadata)                                  |
+-------------------------------------------------------------------------------------------------+
```

---

## 2. Step 1: Push Local Changes to GitHub

From your local Windows terminal in `Repo-Guard`:

```powershell
# 1. Review status
git status

# 2. Stage all updated files
git add .

# 3. Commit
git commit -m "feat(colab): add auto-resume, pre-flight gate, split evaluation, and persistent Drive caching"

# 4. Push to remote
git push origin main
```

---

## 3. Step 2: Open in Google Colab

1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload `notebooks/01_train_colab.ipynb` from your computer (or select the **GitHub** tab, enter `https://github.com/Shreyank86/Repo-Guard`, and open `notebooks/01_train_colab.ipynb`).

---

## 4. Step 3: Configure GPU Runtime

1. In Colab's menu: **Runtime** > **Change runtime type**.
2. Select **T4 GPU** (Free) or **A100 / V100** (Colab Pro).
3. Click **Save**.

---

## 5. Overnight Execution & Browser Keep-Alive

If you plan to let training run overnight without actively clicking the browser tab:

### Browser Keep-Alive Tip
To prevent Google Colab from disconnecting due to tab inactivity:
1. Press `F12` (or `Ctrl+Shift+I`) in your browser to open Developer Tools.
2. Click on the **Console** tab.
3. Paste the following script and press Enter:
   ```javascript
   function ClickConnect(){
       console.log("Colab Keep-Alive ping");
       var colabButton = document.querySelector("colab-connect-button");
       if (colabButton) colabButton.click();
   }
   setInterval(ClickConnect, 60000);
   ```
4. Leave the browser tab open.

### Automatic Resuming (If Colab Still Disconnects)
Because `--auto-resume` is active:
- Checkpoints are saved atomically every 50 episodes to `/content/drive/MyDrive/RepoGuard/checkpoints/dqn/`.
- If Colab drops connection, simply re-open the notebook, run Cells 1–4, and run the training cell again.
- It will automatically detect `dqn_checkpoint_ep*.pt` in Google Drive, restore the exact weights, $\epsilon$-schedule, and optimizer state, and continue from the interrupted episode.

---

## 6. The 16-Stage Notebook Execution Order

The notebook [`notebooks/01_train_colab.ipynb`](../notebooks/01_train_colab.ipynb) is organized into 16 distinct, sequential stages:

| Step | Section Name | What It Does |
|---|---|---|
| **1** | GPU & Runtime Validation | Runs `nvidia-smi`, asserts CUDA availability, records GPU model, compute capability, and total VRAM. |
| **2** | Google Drive Mount & Directories | Mounts `/content/drive` and initializes persistent directories for indexes, cache, episodes, checkpoints, and reports. |
| **3** | Repository Clone & Commit Recording | Clones or pulls `https://github.com/Shreyank86/Repo-Guard.git` and records exact commit hash and summary. |
| **4** | Dependency Installation | Installs `requirements.txt`, `bitsandbytes` (for 4-bit NF4 quantized inference), and `repoguard` in editable mode. |
| **5** | Repository & Index Preparation | Fetches source code; checks Drive for schema 1.1 indexes and symlinks them; rebuilds only missing/invalid indexes. |
| **6** | Benchmark & Split Validation | Validates all 60 cases: Train (36), Validation (12), Held-out Test (12). Enforces strict repository isolation. |
| **7** | Model & Cache Validation | Real frozen LLM smoke test: loads `Qwen/Qwen2.5-Coder-3B-Instruct` in 4-bit NF4 on CUDA, generates test tokens, measures latency/VRAM, and verifies SQLite operation cache on Drive. |
| **8** | RL Environment Validation | Runs pytest suite and verifies 18-dim Box observation vector and 6-dim action masking contracts. |
| **9** | Offline Transition Collection/Validation | Prewarms cache and validates transitions **strictly on train split** (Option C). Asserts zero val/test leakage. |
| **10** | Checkpoint Smoke Tests | Rigorous two-process recovery verification for both DQN (replay buffer + RNG) and REINFORCE (baseline + RNG). |
| **11** | Full DQN Training | 2,000 episodes on CUDA GPU with atomic checkpoints every 50 episodes and auto-resume from Google Drive. |
| **12** | Full REINFORCE Training | 1,500 episodes with running-mean baseline, atomic checkpoints, and auto-resume from Google Drive. |
| **13** | Validation Split Evaluation | Evaluates both models on the isolated `validation` split across $\lambda \in [0.25, 5.0] \to$ `val_dqn.csv`, `val_reinforce.csv`. |
| **14** | Held-Out Test Evaluation | Evaluates both models on the strictly held-out `test` split across $\lambda \in [0.25, 5.0] \to$ `test_dqn.csv`, `test_reinforce.csv`. Zero weight updates. |
| **15** | Risk-Curve Plots | Generates 3-panel publication-quality risk curves (Coverage, Hallucination Rate, Trap Abstention vs. $\lambda$). |
| **16** | Experiment Manifest | Records actual measured values (Git commit, repo hashes, models, split sizes, GPU, VRAM, actual episodes, duration, paths). |

---

## 7. Artifacts Saved in Google Drive

At the end of training and evaluation, your Google Drive folder (`/MyDrive/RepoGuard/`) will contain:

```
MyDrive/
└── RepoGuard/
    ├── data/
    │   ├── indexes/                      <- Reusable dense vector indexes
    │   └── cache/
    │       └── operations.db             <- Reusable SQLite cache (SEARCH, INSPECT, VERIFY, LLM)
    ├── checkpoints/
    │   ├── dqn/
    │   │   ├── dqn_policy_latest.pt      <- Best deployable PyTorch weights
    │   │   ├── dqn_checkpoint_ep500.pt   <- Intermediate checkpoint
    │   │   ├── dqn_checkpoint_ep1000.pt
    │   │   ├── dqn_checkpoint_ep1500.pt
    │   │   ├── dqn_checkpoint_ep2000.pt
    │   │   └── dqn_training_history.json <- Episode rewards and losses
    │   └── reinforce/
    │       ├── reinforce_policy_latest.pt
    │       └── reinforce_training_history.json
    └── reports/
        ├── val_dqn.csv                   <- Validation split evaluation
        ├── val_reinforce.csv
        ├── test_dqn.csv                  <- Held-out test evaluation
        ├── test_reinforce.csv
        ├── test_risk_curves.png          <- Publication-ready figure
        └── experiment_manifest.json      <- Full reproducibility record
```

---

## 8. Phase 3 API Serving Integration

To serve the trained policy in Phase 3 FastAPI:

```python
from repoguard.agent import RepoGuardPolicy, StateVector, RLAction

# Load weights from Google Drive checkpoint
policy = RepoGuardPolicy("data/checkpoints/dqn/dqn_policy_latest.pt", device="cpu")

# Select action given 18-dim observation vector
action: RLAction = policy.select_action(state_vector, valid_mask=[True, True, True, True, True, True])
print(f"Optimal Action: {action.name}")
```
