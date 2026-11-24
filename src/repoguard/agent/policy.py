"""Production-Ready Serving Interface for Trained RepoGuard Policies.

Owner: Member 3 (RL Environment, Training & Model Integration)

Provides a clean, decoupled policy wrapper for Phase 3 API (FastAPI) and UI (Gradio):
- Loads checkpoints (.pt)
- Receives StateVector (18 dims)
- Returns canonical RLAction (SEARCH, INSPECT, VERIFY, ANSWER, ANSWER_WITH_EVIDENCE, ABSTAIN)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from repoguard.agent.dqn import QNetwork
from repoguard.agent.environment import ACTION_LOOKUP
from repoguard.agent.reinforce import PolicyNetwork
from repoguard.schemas import RLAction, StateVector


class RepoGuardPolicy:
    """Production policy serving wrapper for trained RepoGuard RL controllers."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.algo: str = "dqn"
        self.model: torch.nn.Module | None = None
        self.metadata: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
        self.metadata = checkpoint.get("metadata", {})
        self.algo = self.metadata.get("algo", "dqn")
        state_dim = checkpoint.get("state_dim", 18)
        action_dim = checkpoint.get("action_dim", 6)

        if "policy_net" in checkpoint and self.algo == "reinforce":
            self.model = PolicyNetwork(state_dim, action_dim).to(self.device)
            self.model.load_state_dict(checkpoint["policy_net"])
        else:
            self.model = QNetwork(state_dim, action_dim).to(self.device)
            self.model.load_state_dict(checkpoint["policy_net"])

        self.model.eval()

    def select_action(
        self,
        state: StateVector | np.ndarray | list[float],
        valid_mask: list[bool] | None = None,
    ) -> RLAction:
        """Select optimal action according to learned policy and action mask."""
        if isinstance(state, StateVector):
            state_arr = np.array(state.to_tensor(), dtype=np.float32)
        else:
            state_arr = np.array(state, dtype=np.float32)

        state_t = torch.tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(state_t).squeeze(0)

            if valid_mask is not None:
                for i, valid in enumerate(valid_mask):
                    if not valid:
                        outputs[i] = -1e9

            action_idx = int(torch.argmax(outputs).item())

        return ACTION_LOOKUP[action_idx]

    def get_action_probabilities(
        self,
        state: StateVector | np.ndarray | list[float],
        valid_mask: list[bool] | None = None,
    ) -> dict[RLAction, float]:
        """Return probability distribution over permitted actions."""
        if isinstance(state, StateVector):
            state_arr = np.array(state.to_tensor(), dtype=np.float32)
        else:
            state_arr = np.array(state, dtype=np.float32)

        state_t = torch.tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(state_t).squeeze(0)
            if valid_mask is not None:
                for i, valid in enumerate(valid_mask):
                    if not valid:
                        logits[i] = -1e9
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        return {action: float(round(probs[i], 4)) for i, action in enumerate(ACTION_LOOKUP)}
