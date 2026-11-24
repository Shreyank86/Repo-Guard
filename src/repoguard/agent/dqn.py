"""Deep Q-Network (DQN) Agent for RepoGuard.

Owner: Member 3 (RL Environment, Training & Model Integration)

Architecture:
- State input: 18 features (StateVector)
- Action output: 6 Q-values (RLAction)
- Experience Replay Buffer + Target Network + Action Masking + Huber Loss
"""

from __future__ import annotations

import collections
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """MLP estimating state-action Q-values Q(s, a)."""

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


class ReplayBuffer:
    """Experience replay buffer with action mask tracking."""

    def __init__(self, capacity: int = 20000) -> None:
        self.capacity = capacity
        self.buffer: collections.deque[tuple[np.ndarray, int, float, np.ndarray, bool, list[bool]]] = (
            collections.deque(maxlen=capacity)
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        valid_mask: list[bool],
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done, valid_mask))

    def sample(
        self, batch_size: int, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, masks = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.array(masks), dtype=torch.bool, device=device),
        )

    def get_state(self) -> list[tuple[np.ndarray, int, float, np.ndarray, bool, list[bool]]]:
        """Return raw list of transitions for serialization."""
        return list(self.buffer)

    def load_state(self, transitions: list[tuple[np.ndarray, int, float, np.ndarray, bool, list[bool]]]) -> None:
        """Restore transitions into the replay buffer."""
        self.buffer = collections.deque(transitions, maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network Agent with valid-action masking and target updates."""

    def __init__(
        self,
        state_dim: int = 18,
        action_dim: int = 6,
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        buffer_capacity: int = 20000,
        device: str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device(device)

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.sync_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def sync_target_network(self) -> None:
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(
        self,
        state: np.ndarray | list[float],
        epsilon: float = 0.0,
        valid_mask: list[bool] | None = None,
    ) -> int:
        """Select action with epsilon-greedy exploration and action masking."""
        if valid_mask is None:
            valid_mask = [True] * self.action_dim

        valid_actions = [i for i, valid in enumerate(valid_mask) if valid]
        if not valid_actions:
            valid_actions = list(range(self.action_dim))

        # Exploration
        if random.random() < epsilon:
            return random.choice(valid_actions)

        # Exploitation with masking
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0)
            # Mask out invalid actions by setting Q(s, a) = -infinity
            for i, valid in enumerate(valid_mask):
                if not valid:
                    q_values[i] = -1e9
            action = int(torch.argmax(q_values).item())

        return action

    def update(self, batch_size: int = 32) -> float | None:
        """Sample from replay buffer and perform one gradient descent update."""
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones, next_masks = self.replay_buffer.sample(
            batch_size, device=self.device
        )

        # Current Q: Q(s, a)
        current_q = self.policy_net(states).gather(1, actions)

        # Target Q: r + gamma * max_a' Q_target(s', a') [with valid action masking]
        with torch.no_grad():
            next_q = self.target_net(next_states)
            # Apply mask to target Q
            next_q[~next_masks] = -1e9
            max_next_q = next_q.max(dim=1, keepdim=True)[0]
            # Replace any all-masked rows with 0.0
            max_next_q[max_next_q < -1e8] = 0.0
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.item())

    def save(self, checkpoint_path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """Save policy, optimizer, replay buffer, RNG states, and metadata atomically."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

        checkpoint_data = {
            "algo": "dqn",
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "replay_buffer": self.replay_buffer.get_state(),
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
        """Load policy weights, replay buffer, and RNG state from checkpoint."""
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        if "target_net" in checkpoint:
            self.target_net.load_state_dict(checkpoint["target_net"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "replay_buffer" in checkpoint and checkpoint["replay_buffer"] is not None:
            self.replay_buffer.load_state(checkpoint["replay_buffer"])

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
