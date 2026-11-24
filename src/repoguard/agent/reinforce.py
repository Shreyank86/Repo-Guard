"""REINFORCE Policy Gradient Agent for RepoGuard.

Owner: Member 3 (RL Environment, Training & Model Integration)

Architecture:
- Policy Network: MLP mapping 18 state features to 6 action logits
- Categorical sampling with action masking
- Discounted cumulative returns G_t with baseline subtraction for variance reduction
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Policy Network outputting action logits for categorical sampling."""

    def __init__(self, state_dim: int = 18, action_dim: int = 6, hidden_dims: tuple[int, ...] = (128, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class REINFORCEAgent:
    """REINFORCE (Monte Carlo Policy Gradient) Agent with baseline variance reduction."""

    def __init__(
        self,
        state_dim: int = 18,
        action_dim: int = 6,
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        device: str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device(device)

        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Episode trajectory buffers
        self.saved_log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []

        # Running baseline to reduce gradient variance
        self.baseline: float = 0.0
        self.baseline_alpha: float = 0.05

    def select_action(
        self,
        state: np.ndarray | list[float],
        valid_mask: list[bool] | None = None,
    ) -> int:
        """Select action via categorical sampling over masked action logits."""
        state_t = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy_net(state_t).squeeze(0)

        if valid_mask is not None:
            for i, valid in enumerate(valid_mask):
                if not valid:
                    logits[i] = -1e9

        dist = Categorical(logits=logits)
        action_t = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action_t))
        return int(action_t.item())

    def store_reward(self, reward: float) -> None:
        """Record step or terminal reward."""
        self.rewards.append(reward)

    def update(self) -> float | None:
        """Compute Monte Carlo policy gradient update over the finished episode."""
        if not self.rewards or not self.saved_log_probs:
            self.clear_episode()
            return None

        # 1. Calculate discounted cumulative returns G_t
        returns: list[float] = []
        g = 0.0
        for r in reversed(self.rewards):
            g = r + self.gamma * g
            returns.insert(0, g)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # 2. Update running baseline for variance reduction
        mean_return = float(returns_t.mean().item())
        if self.baseline == 0.0:
            self.baseline = mean_return
        else:
            self.baseline = (1.0 - self.baseline_alpha) * self.baseline + self.baseline_alpha * mean_return

        # 3. Advantage: A_t = G_t - b
        advantages = returns_t - self.baseline
        if len(advantages) > 1 and advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 4. Policy gradient loss: - sum_t (log_prob_t * A_t)
        policy_loss: list[torch.Tensor] = []
        for log_prob, advantage in zip(self.saved_log_probs, advantages):
            policy_loss.append(-log_prob * advantage)

        total_loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = float(total_loss.item())
        self.clear_episode()
        return loss_val

    def clear_episode(self) -> None:
        """Reset trajectory buffer for next episode."""
        self.saved_log_probs.clear()
        self.rewards.clear()

    def save(self, checkpoint_path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """Save policy weights, optimizer state, running baseline, RNG states, and metadata atomically."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

        checkpoint_data = {
            "algo": "reinforce",
            "policy_net": self.policy_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "baseline": self.baseline,
            "rng_state": rng_state,
            "metadata": metadata or {},
        }

        # Atomic write to temporary file first
        tmp_path = path.with_suffix(f"{path.suffix}.tmp.{random.randint(10000, 99999)}")
        torch.save(checkpoint_data, str(tmp_path))
        tmp_path.replace(path)

        # Post-save verification: confirm file exists and decodes properly
        verified = torch.load(str(path), map_location="cpu", weights_only=False)
        assert "policy_net" in verified, f"Corrupted checkpoint saved at {path}"

    def load(self, checkpoint_path: str | Path, restore_rng: bool = True) -> dict[str, Any]:
        """Load policy weights, baseline, and RNG state from checkpoint."""
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.baseline = checkpoint.get("baseline", 0.0)

        if restore_rng and "rng_state" in checkpoint and checkpoint["rng_state"]:
            rng = checkpoint["rng_state"]
            if "torch" in rng and rng["torch"] is not None:
                torch.set_rng_state(rng["torch"])
            if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(rng["cuda"])
                except Exception:
                    pass
            if "numpy" in rng and rng["numpy"] is not None:
                np.random.set_state(rng["numpy"])
            if "python" in rng and rng["python"] is not None:
                random.setstate(rng["python"])

        return checkpoint.get("metadata", {})
