"""Unit tests for REINFORCEAgent in src/repoguard/agent/reinforce.py."""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch

from repoguard.agent.reinforce import REINFORCEAgent, PolicyNetwork


def test_policy_network_forward_shape():
    net = PolicyNetwork(state_dim=18, action_dim=6)
    dummy_input = torch.randn(4, 18)
    logits = net(dummy_input)
    assert logits.shape == (4, 6)


def test_reinforce_action_masking():
    agent = REINFORCEAgent(state_dim=18, action_dim=6)
    state = np.zeros(18, dtype=np.float32)

    # Allow only action 5 (ABSTAIN)
    strict_mask = [False, False, False, False, False, True]
    for _ in range(20):
        action = agent.select_action(state, valid_mask=strict_mask)
        assert action == 5
    agent.clear_episode()


def test_reinforce_episode_update():
    agent = REINFORCEAgent(state_dim=18, action_dim=6, learning_rate=0.01)
    state = np.random.randn(18).astype(np.float32)

    for _ in range(3):
        agent.select_action(state)
        agent.store_reward(1.0)

    loss = agent.update()
    assert loss is not None
    assert isinstance(loss, float)
    # Episode buffers must be cleared after update
    assert len(agent.saved_log_probs) == 0
    assert len(agent.rewards) == 0


def test_reinforce_save_and_load():
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "reinforce_test.pt"
        agent1 = REINFORCEAgent(state_dim=18, action_dim=6)

        with torch.no_grad():
            for p in agent1.policy_net.parameters():
                p.fill_(0.77)

        agent1.save(ckpt_path, metadata={"episodes": 50})

        agent2 = REINFORCEAgent(state_dim=18, action_dim=6)
        meta = agent2.load(ckpt_path)

        assert meta["episodes"] == 50
        for p in agent2.policy_net.parameters():
            assert torch.allclose(p, torch.tensor(0.77))
