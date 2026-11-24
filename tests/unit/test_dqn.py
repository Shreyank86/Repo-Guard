"""Unit tests for DQNAgent and QNetwork in src/repoguard/agent/dqn.py."""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch

from repoguard.agent.dqn import DQNAgent, QNetwork, ReplayBuffer


def test_q_network_forward_shape():
    net = QNetwork(state_dim=18, action_dim=6)
    dummy_input = torch.randn(4, 18)
    out = net(dummy_input)
    assert out.shape == (4, 6)


def test_replay_buffer_push_and_sample():
    buf = ReplayBuffer(capacity=100)
    s = np.zeros(18, dtype=np.float32)
    s_next = np.ones(18, dtype=np.float32)
    mask = [True, False, False, True, False, True]

    for i in range(10):
        buf.push(s, action=0, reward=1.0, next_state=s_next, done=False, valid_mask=mask)

    assert len(buf) == 10
    states, actions, rewards, next_states, dones, masks = buf.sample(4)
    assert states.shape == (4, 18)
    assert actions.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert next_states.shape == (4, 18)
    assert dones.shape == (4, 1)
    assert masks.shape == (4, 6)


def test_action_masking_strictly_respected():
    agent = DQNAgent(state_dim=18, action_dim=6)
    state = np.zeros(18, dtype=np.float32)

    # Allow only action 5 (ABSTAIN)
    strict_mask = [False, False, False, False, False, True]
    for _ in range(20):
        action = agent.select_action(state, epsilon=0.0, valid_mask=strict_mask)
        assert action == 5

    # Allow only action 0 (SEARCH) or 3 (ANSWER)
    mask_0_3 = [True, False, False, True, False, False]
    for _ in range(20):
        action = agent.select_action(state, epsilon=0.0, valid_mask=mask_0_3)
        assert action in (0, 3)


def test_dqn_update_decreases_loss():
    agent = DQNAgent(state_dim=18, action_dim=6, learning_rate=0.01)
    s = np.random.randn(18).astype(np.float32)
    s_next = np.random.randn(18).astype(np.float32)
    mask = [True] * 6

    # Fill buffer with identical transitions to test overfitting
    for _ in range(64):
        agent.replay_buffer.push(s, 3, 1.0, s_next, True, mask)

    initial_loss = agent.update(batch_size=32)
    assert initial_loss is not None

    for _ in range(20):
        final_loss = agent.update(batch_size=32)

    assert final_loss is not None
    assert final_loss < initial_loss


def test_dqn_checkpoint_save_and_load():
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "dqn_test.pt"
        agent1 = DQNAgent(state_dim=18, action_dim=6)

        # Modify weight
        with torch.no_grad():
            for p in agent1.policy_net.parameters():
                p.fill_(0.42)

        agent1.save(ckpt_path, metadata={"episodes": 100})

        agent2 = DQNAgent(state_dim=18, action_dim=6)
        meta = agent2.load(ckpt_path)

        assert meta["episodes"] == 100
        for p in agent2.policy_net.parameters():
            assert torch.allclose(p, torch.tensor(0.42))
