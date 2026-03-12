"""
Unit Tests — marl package (Layer 3: Cooperative MARL).

Tests cover:
    - RolloutBuffer: add, get_tensors, mini_batches
    - GAEEstimator: advantage shape, normalization, return computation
    - CooperativeRewardShaper: blending, conflict detection, coordination bonus
    - CommunicationProtocol: broadcast, global state, command routing
    - MAPPOTrainer: construction, one-episode smoke test
"""

import pytest
import numpy as np
import torch

N_AGENTS  = 3
OBS_DIM   = 8
N_ACTIONS = 5
T_STEPS   = 50    # short rollout for tests


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------

class TestRolloutBuffer:
    def _make_buffer(self):
        from marl.buffer import RolloutBuffer
        return RolloutBuffer(n_agents=N_AGENTS, obs_dim=OBS_DIM, episode_len=T_STEPS)

    def test_add_and_size(self):
        buf = self._make_buffer()
        for _ in range(10):
            buf.add(
                s_locals=np.random.rand(N_AGENTS, OBS_DIM).astype(np.float32),
                s_global=np.random.rand(N_AGENTS * OBS_DIM).astype(np.float32),
                actions=np.zeros(N_AGENTS, dtype=np.int64),
                log_probs=np.zeros(N_AGENTS, dtype=np.float32),
                rewards=np.ones(N_AGENTS, dtype=np.float32),
                value=0.5,
                done=False,
            )
        assert buf.size == 10

    def test_get_tensors_shapes(self):
        buf = self._make_buffer()
        for _ in range(20):
            buf.add(
                s_locals=np.random.rand(N_AGENTS, OBS_DIM).astype(np.float32),
                s_global=np.random.rand(N_AGENTS * OBS_DIM).astype(np.float32),
                actions=np.zeros(N_AGENTS, dtype=np.int64),
                log_probs=np.zeros(N_AGENTS, dtype=np.float32),
                rewards=np.ones(N_AGENTS, dtype=np.float32),
                value=0.5,
                done=False,
            )
        tensors = buf.get_tensors(torch.device("cpu"))
        assert tensors["s_locals"].shape  == (20, N_AGENTS, OBS_DIM)
        assert tensors["s_globals"].shape == (20, N_AGENTS * OBS_DIM)
        assert tensors["actions"].shape   == (20, N_AGENTS)
        assert tensors["rewards"].shape   == (20, N_AGENTS)
        assert tensors["values"].shape    == (20,)
        assert tensors["dones"].shape     == (20,)

    def test_clear_resets(self):
        buf = self._make_buffer()
        for _ in range(5):
            buf.add(
                s_locals=np.zeros((N_AGENTS, OBS_DIM), dtype=np.float32),
                s_global=np.zeros(N_AGENTS * OBS_DIM, dtype=np.float32),
                actions=np.zeros(N_AGENTS, dtype=np.int64),
                log_probs=np.zeros(N_AGENTS, dtype=np.float32),
                rewards=np.zeros(N_AGENTS, dtype=np.float32),
                value=0.0, done=False,
            )
        buf.clear()
        assert buf.size == 0

    def test_mini_batches_coverage(self):
        buf = self._make_buffer()
        T = 40
        for _ in range(T):
            buf.add(
                s_locals=np.random.rand(N_AGENTS, OBS_DIM).astype(np.float32),
                s_global=np.random.rand(N_AGENTS * OBS_DIM).astype(np.float32),
                actions=np.zeros(N_AGENTS, dtype=np.int64),
                log_probs=np.zeros(N_AGENTS, dtype=np.float32),
                rewards=np.ones(N_AGENTS, dtype=np.float32),
                value=0.5, done=False,
            )
        dev = torch.device("cpu")
        adv = torch.zeros(T, N_AGENTS)
        ret = torch.zeros(T, N_AGENTS)
        n_items = sum(b["s_locals"].shape[0] for b in buf.mini_batches(16, dev, adv, ret))
        assert n_items == T


# ---------------------------------------------------------------------------
# GAEEstimator
# ---------------------------------------------------------------------------

class TestGAEEstimator:
    def test_output_shapes(self):
        from marl.advantage_estimator import GAEEstimator
        gae = GAEEstimator(gamma=0.99, lam=0.95)
        T = 20
        rewards = torch.rand(T, N_AGENTS)
        values  = torch.rand(T)
        dones   = torch.zeros(T)
        advantages, returns = gae.compute(rewards, values, dones, last_value=0.0, n_agents=N_AGENTS)
        assert advantages.shape == (T, N_AGENTS)
        assert returns.shape    == (T, N_AGENTS)

    def test_normalized_advantages_zero_mean(self):
        from marl.advantage_estimator import GAEEstimator
        gae = GAEEstimator()
        T = 100
        rewards = torch.rand(T, N_AGENTS)
        values  = torch.rand(T)
        dones   = torch.zeros(T)
        advantages, _ = gae.compute(rewards, values, dones)
        assert abs(advantages.mean().item()) < 0.1   # approximately zero mean

    def test_done_breaks_gae_chain(self):
        """Done flag should zero out the future value bootstrap for that step."""
        from marl.advantage_estimator import GAEEstimator
        gae = GAEEstimator(gamma=0.99, lam=0.95)
        T = 10
        rewards = torch.ones(T, N_AGENTS)
        values  = torch.ones(T)
        dones   = torch.zeros(T)
        dones[5] = 1.0   # mid-episode done
        advantages, returns = gae.compute(rewards, values, dones)
        # Should not raise; just shape check
        assert advantages.shape == (T, N_AGENTS)

    def test_functional_api(self):
        from marl.advantage_estimator import compute_gae
        T = 30
        rewards = torch.rand(T, N_AGENTS)
        values  = torch.rand(T)
        dones   = torch.zeros(T)
        adv, ret = compute_gae(rewards, values, dones)
        assert adv.shape == (T, N_AGENTS)
        assert ret.shape == (T, N_AGENTS)


# ---------------------------------------------------------------------------
# CooperativeRewardShaper
# ---------------------------------------------------------------------------

class TestCooperativeRewardShaper:
    def _make_shaper(self):
        from marl.cooperative_rewards import CooperativeRewardShaper
        return CooperativeRewardShaper(n_agents=N_AGENTS)

    def test_output_shape(self):
        shaper = self._make_shaper()
        local  = np.array([0.5, 0.3, 0.7])
        acts   = np.array([0, 1, 2])
        obs    = np.random.rand(N_AGENTS, OBS_DIM).astype(np.float32)
        shaped = shaper.shape(local, acts, obs)
        assert shaped.shape == (N_AGENTS,)
        assert shaped.dtype == np.float32

    def test_conflict_penalty_applied(self):
        """All agents relaying should produce a lower mean reward than no conflict."""
        shaper = self._make_shaper()
        local  = np.ones(N_AGENTS) * 0.5
        obs    = np.random.rand(N_AGENTS, OBS_DIM).astype(np.float32)
        # No conflict
        acts_ok = np.array([0, 1, 2])
        shaped_ok = shaper.shape(local, acts_ok, obs)
        # All relay (conflict)
        acts_conflict = np.array([3, 3, 3])
        shaped_conflict = shaper.shape(local, acts_conflict, obs)
        assert shaped_ok.mean() > shaped_conflict.mean()

    def test_functional_api(self):
        from marl.cooperative_rewards import cooperative_reward
        local = np.array([0.5, 0.3, 0.7])
        r = cooperative_reward(local, global_outcome=1.0, conflict_detected=False)
        assert r.shape == (N_AGENTS,)

    def test_conflict_functional(self):
        from marl.cooperative_rewards import cooperative_reward
        local = np.array([0.5, 0.5, 0.5])
        r_ok      = cooperative_reward(local, 1.0, False)
        r_conflict= cooperative_reward(local, 1.0, True)
        assert r_ok.mean() > r_conflict.mean()


# ---------------------------------------------------------------------------
# CommunicationProtocol
# ---------------------------------------------------------------------------

class TestCommunicationProtocol:
    def _make_comm(self):
        from marl.communication_protocol import CommunicationProtocol
        return CommunicationProtocol(n_agents=N_AGENTS, obs_dim=OBS_DIM)

    def test_broadcast_and_receive(self):
        comm = self._make_comm()
        obs  = np.random.rand(OBS_DIM).astype(np.float32)
        comm.broadcast_state(agent_id=0, obs=obs, sim_time=0.0)
        messages = comm.receive_all_states()
        assert messages.shape == (N_AGENTS, OBS_DIM)
        np.testing.assert_array_almost_equal(messages[0], obs)

    def test_global_state_shape(self):
        comm = self._make_comm()
        local_obs = np.random.rand(N_AGENTS, OBS_DIM).astype(np.float32)
        global_s  = comm.get_global_state(local_obs)
        assert global_s.shape == (N_AGENTS * OBS_DIM,)

    def test_send_receive_command(self):
        from marl.communication_protocol import CommandMessage
        comm = self._make_comm()
        cmd  = CommandMessage(target_id=1, task="hibernate", priority=2)
        comm.send_command(cmd)
        received = comm.receive_commands(agent_id=1)
        assert len(received) == 1
        assert received[0].task == "hibernate"

    def test_receive_commands_empty_after_drain(self):
        from marl.communication_protocol import CommandMessage
        comm = self._make_comm()
        comm.send_command(CommandMessage(target_id=0, task="relay_mode"))
        comm.receive_commands(0)
        second = comm.receive_commands(0)
        assert len(second) == 0

    def test_broadcast_to_all(self):
        from marl.communication_protocol import CommunicationProtocol
        comm = CommunicationProtocol(n_agents=N_AGENTS)
        comm.broadcast_command_to_all("charge_priority")
        for i in range(N_AGENTS):
            cmds = comm.receive_commands(i)
            assert len(cmds) == 1
            assert cmds[0].task == "charge_priority"

    def test_compress_top_k(self):
        from marl.communication_protocol import CommunicationProtocol
        comm = CommunicationProtocol(n_agents=N_AGENTS, obs_dim=OBS_DIM, compress_top_k=4)
        obs  = np.random.rand(OBS_DIM).astype(np.float32)
        comm.broadcast_state(0, obs, 0.0)
        messages = comm.receive_all_states()
        assert messages.shape == (N_AGENTS, 4)

    def test_isl_latency(self):
        from marl.communication_protocol import ISLinkModel
        isl = ISLinkModel(base_delay_ms=5.0)
        latency = isl.compute_latency(distance_km=1000.0)
        expected = 5e-3 + 1000.0 / 299792.0
        assert abs(latency - expected) < 1e-6


# ---------------------------------------------------------------------------
# MAPPOTrainer (smoke test)
# ---------------------------------------------------------------------------

class TestMAPPOTrainer:
    def _make_trainer(self, n_agents=2, episode_length=20, batch_size=16):
        from environment.constellation_env import ConstellationEnv
        from marl.mappo_trainer import MAPPOTrainer
        env = ConstellationEnv(n_satellites=n_agents, episode_length=episode_length)
        return MAPPOTrainer.make(
            n_agents=n_agents,
            env=env,
            predict_k=3,
            device="cpu",
            n_epochs=2,
            batch_size=batch_size,
            episode_length=episode_length,
            checkpoint_dir="/tmp/casc_marl_test",
        )

    def test_construction(self):
        trainer = self._make_trainer()
        assert len(trainer.actors) == 2
        assert trainer.critic is not None

    def test_collect_step(self):
        trainer = self._make_trainer()
        obs = np.random.rand(2, OBS_DIM).astype(np.float32)
        actions, log_probs, value = trainer._collect_step(obs)
        assert actions.shape   == (2,)
        assert log_probs.shape == (2,)
        assert isinstance(value, float)

    def test_train_episode_returns_scalars(self):
        trainer = self._make_trainer(n_agents=2, episode_length=20, batch_size=8)
        ep_reward, a_loss, c_loss, entropy = trainer.train_episode()
        assert isinstance(ep_reward, float)
        assert isinstance(a_loss, float)
        assert isinstance(c_loss, float)
        assert isinstance(entropy, float)

    def test_rollout_output(self):
        trainer = self._make_trainer()
        result = trainer.rollout(n_steps=10, deterministic=True)
        assert "rewards" in result
        assert len(result["rewards"]) <= 10

    def test_save_load_checkpoint(self, tmp_path):
        from marl.mappo_trainer import MAPPOTrainer
        from environment.constellation_env import ConstellationEnv
        env = ConstellationEnv(n_satellites=2, episode_length=10)
        trainer = MAPPOTrainer.make(
            n_agents=2, env=env, predict_k=3, device="cpu",
            checkpoint_dir=str(tmp_path),
        )
        trainer.save_checkpoint(suffix="test")
        import os
        assert os.path.exists(str(tmp_path / "mappo_test.pt"))
