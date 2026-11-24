# Google Colab Training & Cloud Execution Guide (Member 3)

> **Workstream:** Member 3 — RL Agent Development, Training, Evaluation & Model Integration  
> **Target Environment:** Google Colab (Free T4 GPU or Colab Pro) + Google Drive Checkpoint Persistence  
> **Primary Notebook:** [`notebooks/01_train_colab.ipynb`](../notebooks/01_train_colab.ipynb)  
> **Core Architecture:**
> - **Action Space:** Canonical 6 actions (`SEARCH`, `INSPECT`, `VERIFY`, `ANSWER`, `ANSWER_WITH_EVIDENCE`, `ABSTAIN`)
> - **Observation Space:** 18-dimensional normalized state vector ($s \in [0, 1]^{18}$)
> - **Reward Function:** Dynamic risk-conditioned formula $r = R_{\text{final}}(\lambda) - 0.2 \cdot \frac{T}{2000} - 0.01 \cdot n$ where $\lambda \in [0.25, 5.0]$

---

## 1. Overview & Cloud Architecture

All core engineering, environments, policies, caches, and test suites are built and verified locally on Windows. Google Colab is used exclusively for:
1. **Accelerated Multi-Episode Training:** Running 2,000 episodes for Deep Q-Network (DQN) and 1,500 episodes for REINFORCE.
2. **Persistent Checkpoints:** Writing model weights, optimizer states, replay buffers, and metrics directly to Google Drive so no training progress is lost upon Colab disconnect.
3. **Multi-Risk Evaluation:** Evaluating policies across risk tolerance penalties ($\lambda \in \{0.25, 0.5, 1.0, 2.0, 3.0, 5.0\}$) and generating high-resolution risk curves.

```
+-------------------------------------------------------------+
|                      Google Colab (GPU)                     |
|                                                             |
|   1. Clone RepoGuard from GitHub                            |
|   2. Fetch 5 benchmark repos + build dense vector indexes   |
|   3. Run train_agent.py (DQN: 2000 eps, REINFORCE: 1500 eps)|
|   4. Run evaluate_rl.py across lambda levels                |
+------------------------------+------------------------------+
                               |
                               | (Saves models & logs every 50 episodes)
                               v
+-------------------------------------------------------------+
|                  Google Drive Persistent Storage            |
|                                                             |
|   /MyDrive/RepoGuard/                                       |
|   ├── checkpoints/                                          |
|   │   ├── dqn/dqn_policy_latest.pt                          |
|   │   └── reinforce/reinforce_policy_latest.pt              |
|   └── reports/                                              |
|       ├── eval_dqn.csv                                      |
|       ├── eval_reinforce.csv                                |
|       └── risk_curves_comparison.png                        |
+-------------------------------------------------------------+
```

---

## 2. Step 1: Push Local Work to GitHub

Before launching Google Colab, push all local Member 3 implementation work to GitHub.

Open your local terminal in `Repo-Guard` and run:

```powershell
# 1. Review status
git status

# 2. Stage implementation files (note: large artifacts & caches are auto-ignored by .gitignore)
git add src/repoguard/agent/
git add scripts/train_agent.py scripts/evaluate_rl.py scripts/run_episodes.py
git add tests/unit/test_*.py
git add notebooks/01_train_colab.ipynb
git add docs/COLAB_TRAINING_GUIDE.md
git add requirements.txt IMPLEMENTATION_PLAN.md .gitignore

# 3. Commit
git commit -m "feat(agent): complete Phase 2 RL agent, environment, tests, and Colab pipeline"

# 4. Push to remote
git push origin main
```

---

## 3. Step 2: Open Notebook in Google Colab

You can open the training notebook in Google Colab using either method:

### Method A: Upload File Directly
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select `notebooks/01_train_colab.ipynb` from your local machine.

### Method B: Open from GitHub
1. In [Google Colab](https://colab.research.google.com/), select the **GitHub** tab.
2. Enter your repository URL (`https://github.com/Shreyank86/Repo-Guard`).
3. Select `notebooks/01_train_colab.ipynb`.

---

## 4. Step 3: Configure GPU Runtime

To enable GPU acceleration:
1. In Google Colab's top menu, click **Runtime** > **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (Standard/Free tier) or **A100 / V100** (Colab Pro).
3. Click **Save**.
4. Run the first cell (`!nvidia-smi`) to confirm GPU detection:
   ```bash
   +-----------------------------------------------------------------------------------------+
   | NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2       |
   | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC   |
   |   0  Tesla T4                       Off | 00000000:00:04.0 Off |                    0   |
   +-----------------------------------------------------------------------------------------+
   ```

---

## 5. Step 4: Step-by-Step Notebook Execution

The notebook [`notebooks/01_train_colab.ipynb`](../notebooks/01_train_colab.ipynb) is organized into 11 distinct sections. Run them cell-by-cell:

### Cell 1–2: Mount Google Drive
Mounts Google Drive to `/content/drive` and automatically initializes the required folder structure:
- `/content/drive/MyDrive/RepoGuard/checkpoints/dqn/`
- `/content/drive/MyDrive/RepoGuard/checkpoints/reinforce/`
- `/content/drive/MyDrive/RepoGuard/reports/`

### Cell 3–4: Clone Repository & Install Dependencies
Clones your GitHub repository into `/content/Repo-Guard` and runs:
```bash
pip install -q -r requirements.txt
pip install -q -e .
```

### Cell 5: One-Command Repository Fetching & Indexing
Clones all 5 benchmark repositories (`schedule`, `python-dotenv`, `cachetools`, `python-tabulate`, `requests`) at their pinned snapshot commits and generates dense semantic vector indexes (`sentence-transformers/all-MiniLM-L6-v2`) in ~60 seconds:
```bash
python scripts/fetch_repos.py --index
```

### Cell 6: Smoke-Test Unit Verification
Verifies the RL environment, action executor, Q-network, REINFORCE agent, and serving interface:
```bash
pytest tests/unit/test_environment.py tests/unit/test_dqn.py tests/unit/test_reinforce.py tests/unit/test_policy_serving.py -v
```

### Cell 7: Train Deep Q-Network (DQN) Policy
Trains the primary DQN policy over **2,000 episodes** with:
- Dynamic risk sampling $\lambda \sim \text{LogUniform}(0.25, 5.0)$
- Epsilon-greedy schedule ($\epsilon: 1.0 \to 0.05$)
- Experience replay buffer capacity: 20,000 transitions
- Smooth L1 (Huber) loss and target network sync every 10 episodes
- Atomic checkpointing to Google Drive every 50 episodes:
```bash
python scripts/train_agent.py \
    --algo dqn \
    --episodes 2000 \
    --device cuda \
    --learning-rate 0.0005 \
    --batch-size 32 \
    --checkpoint-dir /content/drive/MyDrive/RepoGuard/checkpoints/dqn \
    --log-interval 50
```

### Cell 8: Train REINFORCE Agent (Ablation Baseline)
Trains the REINFORCE baseline over **1,500 episodes** with masked action probabilities and running-mean baseline subtraction:
```bash
python scripts/train_agent.py \
    --algo reinforce \
    --episodes 1500 \
    --device cuda \
    --learning-rate 0.0005 \
    --checkpoint-dir /content/drive/MyDrive/RepoGuard/checkpoints/reinforce \
    --log-interval 50
```

### Cell 9: Comprehensive Policy Evaluation
Evaluates both policies across 6 risk tolerance levels ($\lambda \in \{0.25, 0.5, 1.0, 2.0, 3.0, 5.0\}$):
```bash
python scripts/evaluate_rl.py \
    --policy /content/drive/MyDrive/RepoGuard/checkpoints/dqn/dqn_policy_latest.pt \
    --split all \
    --output-csv /content/drive/MyDrive/RepoGuard/reports/eval_dqn.csv

python scripts/evaluate_rl.py \
    --policy /content/drive/MyDrive/RepoGuard/checkpoints/reinforce/reinforce_policy_latest.pt \
    --split all \
    --output-csv /content/drive/MyDrive/RepoGuard/reports/eval_reinforce.csv
```

### Cell 10: Comparative Visualization & Risk Curves
Generates high-resolution comparative figures showing:
1. **Coverage vs. $\lambda$:** How selective the agent becomes as risk penalty increases.
2. **Hallucination Rate vs. $\lambda$:** Demonstrating monotonic reduction in unsupported claims.
3. **Trap Abstention vs. $\lambda$:** Demonstrating proactive abstention on unanswerable questions.
Saves figure to `/content/drive/MyDrive/RepoGuard/reports/risk_curves_comparison.png`.

---

## 6. Checkpoint Structure in Google Drive

At the end of training, your Google Drive will contain:

```
MyDrive/
└── RepoGuard/
    ├── checkpoints/
    │   ├── dqn/
    │   │   ├── dqn_policy_latest.pt             <- Deployable PyTorch weights
    │   │   ├── dqn_checkpoint_episode_500.pt    <- Midpoint checkpoint
    │   │   ├── dqn_checkpoint_episode_1000.pt
    │   │   ├── dqn_checkpoint_episode_1500.pt
    │   │   ├── dqn_checkpoint_episode_2000.pt
    │   │   └── training_metrics.json            <- Loss and reward history
    │   │
    │   └── reinforce/
    │       ├── reinforce_policy_latest.pt       <- Deployable PyTorch weights
    │       ├── reinforce_checkpoint_episode_500.pt
    │       ├── reinforce_checkpoint_episode_1000.pt
    │       ├── reinforce_checkpoint_episode_1500.pt
    │       └── training_metrics.json
    │
    └── reports/
        ├── eval_dqn.csv                         <- Full evaluation table
        ├── eval_reinforce.csv                   <- Full evaluation table
        └── risk_curves_comparison.png           <- Publication-ready figure
```

---

## 7. Using Trained Policy in Phase 3 Serving

Once downloaded or synced from Google Drive, loading the trained policy in Python or FastAPI requires just 3 lines of code:

```python
from repoguard.agent import RepoGuardPolicy, StateVector, RLAction

# 1. Load weights
policy = RepoGuardPolicy("data/checkpoints/dqn/dqn_policy_latest.pt", device="cpu")

# 2. Select optimal action given observation vector
action: RLAction = policy.select_action(state_vector, valid_mask=[True, True, True, True, True, True])
print(f"Selected action: {action.name}")

# 3. Inspect policy distribution
action_probs = policy.get_action_probabilities(state_vector)
for act, prob in action_probs.items():
    print(f"  {act.name}: {prob:.3f}")
```

---

## 8. Troubleshooting & FAQ

### Q1: What happens if Google Colab disconnects during training?
> Because the training loop periodically saves checkpoints (e.g. `dqn_checkpoint_episode_500.pt`) directly to Google Drive, no progress is lost. If your session drops, simply reconnect, re-mount Drive, and run `train_agent.py`.

### Q2: Can I run with live LLM (`Qwen2.5-Coder`) instead of `MockLLM`?
> Yes! In Colab, install `bitsandbytes` (`!pip install bitsandbytes`) and pass `--no-mock` to `train_agent.py` and `evaluate_rl.py`. You can also configure your Hugging Face token in Colab Secrets (`userdata.get('HF_TOKEN')`).

### Q3: How long does 2,000 episodes take to train on Colab GPU?
> With operation caching and MockLLM, 2,000 episodes on a Tesla T4 GPU takes approximately **4 to 6 minutes**. With live 4-bit quantized LLM inference, it takes approximately **35 to 50 minutes**.
