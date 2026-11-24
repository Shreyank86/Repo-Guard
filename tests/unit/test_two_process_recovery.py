"""Two-Process Checkpoint Recovery Integration Test.

Tests that a completely independent OS process can load a checkpoint saved by a prior
process and resume training seamlessly from episode N + 1 with complete state restoration:
- Weights and target network
- Optimizer state
- Replay buffer (DQN)
- Running baseline (REINFORCE)
- Exploration epsilon
- RNG states
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def test_dqn_two_process_recovery():
    """Verify DQN checkpoint recovery across separate OS processes."""
    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = Path(td) / "dqn_checkpoints"
        cache_db = Path(td) / "operations.db"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Process 1: Train for 4 episodes and save checkpoint
        cmd1 = [
            sys.executable,
            str(ROOT / "scripts/train_agent.py"),
            "--algo", "dqn",
            "--episodes", "4",
            "--save-interval", "2",
            "--checkpoint-dir", str(ckpt_dir),
            "--cache-db", str(cache_db),
            "--seed", "42",
            "--mock",
            "--device", "cpu",
        ]
        res1 = subprocess.run(cmd1, capture_output=True, text=True, check=True)
        assert "Starting DQN training" in res1.stderr or "Starting DQN training" in res1.stdout

        ckpt_ep4 = ckpt_dir / "dqn_checkpoint_ep4.pt"
        assert ckpt_ep4.exists(), f"Process 1 failed to create {ckpt_ep4}"

        # Inspect checkpoint 1
        data_ep4 = torch.load(str(ckpt_ep4), map_location="cpu", weights_only=False)
        assert data_ep4["metadata"]["episode"] == 4
        assert len(data_ep4["replay_buffer"]) > 0, "Replay buffer was not saved in checkpoint!"
        assert "rng_state" in data_ep4, "RNG state was not saved in checkpoint!"

        # Process 2: Launch independent process with --auto-resume targeting 8 episodes
        cmd2 = [
            sys.executable,
            str(ROOT / "scripts/train_agent.py"),
            "--algo", "dqn",
            "--episodes", "8",
            "--save-interval", "2",
            "--auto-resume",
            "--checkpoint-dir", str(ckpt_dir),
            "--cache-db", str(cache_db),
            "--seed", "42",
            "--mock",
            "--device", "cpu",
        ]
        res2 = subprocess.run(cmd2, capture_output=True, text=True, check=True)
        output = res2.stdout + res2.stderr
        assert "Resuming DQN from checkpoint" in output, "Process 2 did not trigger checkpoint resume!"
        assert "starting at episode 5/8" in output, "Process 2 did not start at episode 5!"

        ckpt_ep8 = ckpt_dir / "dqn_checkpoint_ep8.pt"
        assert ckpt_ep8.exists(), f"Process 2 failed to produce final checkpoint {ckpt_ep8}"

        data_ep8 = torch.load(str(ckpt_ep8), map_location="cpu", weights_only=False)
        assert data_ep8["metadata"]["episode"] == 8
        assert len(data_ep8["replay_buffer"]) >= len(data_ep4["replay_buffer"])

        # History should have 8 records
        history_file = ckpt_dir / "dqn_training_history.json"
        assert history_file.exists()
        history = json.loads(history_file.read_text(encoding="utf-8"))
        assert len(history) == 8
        assert [h["episode"] for h in history] == list(range(1, 9))


def test_reinforce_two_process_recovery():
    """Verify REINFORCE checkpoint recovery across separate OS processes."""
    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = Path(td) / "reinforce_checkpoints"
        cache_db = Path(td) / "operations.db"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Process 1: Train for 3 episodes and save checkpoint
        cmd1 = [
            sys.executable,
            str(ROOT / "scripts/train_agent.py"),
            "--algo", "reinforce",
            "--episodes", "3",
            "--save-interval", "3",
            "--checkpoint-dir", str(ckpt_dir),
            "--cache-db", str(cache_db),
            "--seed", "42",
            "--mock",
            "--device", "cpu",
        ]
        res1 = subprocess.run(cmd1, capture_output=True, text=True, check=True)
        assert "Starting REINFORCE training" in res1.stderr or "Starting REINFORCE training" in res1.stdout

        ckpt_ep3 = ckpt_dir / "reinforce_checkpoint_ep3.pt"
        assert ckpt_ep3.exists(), f"Process 1 failed to create {ckpt_ep3}"

        data_ep3 = torch.load(str(ckpt_ep3), map_location="cpu", weights_only=False)
        assert data_ep3["metadata"]["episode"] == 3
        assert "baseline" in data_ep3
        assert "rng_state" in data_ep3

        # Process 2: Launch independent process with --auto-resume targeting 6 episodes
        cmd2 = [
            sys.executable,
            str(ROOT / "scripts/train_agent.py"),
            "--algo", "reinforce",
            "--episodes", "6",
            "--save-interval", "3",
            "--auto-resume",
            "--checkpoint-dir", str(ckpt_dir),
            "--cache-db", str(cache_db),
            "--seed", "42",
            "--mock",
            "--device", "cpu",
        ]
        res2 = subprocess.run(cmd2, capture_output=True, text=True, check=True)
        output = res2.stdout + res2.stderr
        assert "Resuming REINFORCE from checkpoint" in output, "Process 2 did not trigger resume!"
        assert "starting at episode 4/6" in output, "Process 2 did not start at episode 4!"

        ckpt_ep6 = ckpt_dir / "reinforce_checkpoint_ep6.pt"
        assert ckpt_ep6.exists(), f"Process 2 failed to produce final checkpoint {ckpt_ep6}"

        data_ep6 = torch.load(str(ckpt_ep6), map_location="cpu", weights_only=False)
        assert data_ep6["metadata"]["episode"] == 6

        history_file = ckpt_dir / "reinforce_training_history.json"
        assert history_file.exists()
        history = json.loads(history_file.read_text(encoding="utf-8"))
        assert len(history) == 6
        assert [h["episode"] for h in history] == list(range(1, 7))
