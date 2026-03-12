"""
Unit Tests — agents package (Layer 2b).

Tests cover:
    - ActorNetwork forward, distribution, act, evaluate_actions
    - CriticNetwork local and centralized
    - ActionSelector safety override cases
    - SatelliteAgent full pipeline smoke test
"""

import pytest
import numpy as np
import torch

OBS_DIM   = 8
N_ACTIONS = 5
BATCH     = 8
PREDICT_K = 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def s_t():
    return torch.rand(BATCH, OBS_DIM)


@pytest.fixture()
def s_future():
    return torch.rand(BATCH, PREDICT_K, OBS_DIM)


@pytest.fixture()
def obs_safe():
    """Observation with safe SoC (0.8) and safe temperature (0.2 normalized)."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = 0.80  # SoC
    obs[1] = 0.95  # SoH
    obs[2] = 0.20  # T normalized (20°C / 100)
    return obs


@pytest.fixture()
def obs_low_soc():
    """Observation with critically low SoC."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = 0.08  # SoC below critical threshold (0.10)
    obs[2] = 0.25
    return obs


@pytest.fixture()
def obs_warn_soc():
    """Observation with SoC in warning zone [0.10, 0.15)."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = 0.12  # above critical, below min
    obs[2] = 0.25
    return obs


@pytest.fixture()
def obs_high_temp():
    """Observation with overtemperature condition."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = 0.75
    obs[2] = 0.65  # normalized: 65°C / 100 > T_MAX=0.60
    return obs


# ---------------------------------------------------------------------------
# ActorNetwork
# ---------------------------------------------------------------------------

class TestActorNetwork:
    def test_logits_shape(self, s_t, s_future):
        from agents.policy_network import ActorNetwork
        actor = ActorNetwork(state_dim=OBS_DIM, n_actions=N_ACTIONS, predict_k=PREDICT_K,
                             hidden_dims=[64, 64])
        logits = actor(s_t, s_future)
        assert logits.shape == (BATCH, N_ACTIONS)

    def test_act_shapes(self, s_t, s_future):
        from agents.policy_network import ActorNetwork
        actor = ActorNetwork(state_dim=OBS_DIM, n_actions=N_ACTIONS, predict_k=PREDICT_K,
                             hidden_dims=[64, 64])
        action, log_prob, entropy = actor.act(s_t, s_future)
        assert action.shape   == (BATCH,)
        assert log_prob.shape == (BATCH,)
        assert entropy.shape  == (BATCH,)

    def test_deterministic_act(self, s_t, s_future):
        from agents.policy_network import ActorNetwork
        actor = ActorNetwork(state_dim=OBS_DIM, n_actions=N_ACTIONS, predict_k=PREDICT_K,
                             hidden_dims=[64, 64])
        action1, _, _ = actor.act(s_t, s_future, deterministic=True)
        action2, _, _ = actor.act(s_t, s_future, deterministic=True)
        assert torch.equal(action1, action2), "Deterministic actions should be identical"

    def test_action_range(self, s_t, s_future):
        from agents.policy_network import ActorNetwork
        actor = ActorNetwork(state_dim=OBS_DIM, n_actions=N_ACTIONS, predict_k=PREDICT_K,
                             hidden_dims=[64, 64])
        for _ in range(10):
            action, _, _ = actor.act(s_t, s_future)
            assert (action >= 0).all() and (action < N_ACTIONS).all()

    def test_evaluate_actions(self, s_t, s_future):
        from agents.policy_network import ActorNetwork
        actor = ActorNetwork(state_dim=OBS_DIM, n_actions=N_ACTIONS, predict_k=PREDICT_K,
                             hidden_dims=[64, 64])
        actions = torch.randint(0, N_ACTIONS, (BATCH,))
        s_future_flat = s_future.view(BATCH, -1)
        log_prob, entropy = actor.evaluate_actions(s_t, s_future_flat, actions)
        assert log_prob.shape == (BATCH,)
        assert entropy.shape  == (BATCH,)
        assert (entropy >= 0).all()

    def test_get_distribution(self, s_t, s_future):
        from agents.policy_network import ActorNetwork
        from torch.distributions import Categorical
        actor = ActorNetwork(state_dim=OBS_DIM, n_actions=N_ACTIONS, predict_k=PREDICT_K,
                             hidden_dims=[64, 64])
        dist = actor.get_distribution(s_t, s_future)
        assert isinstance(dist, Categorical)
        assert dist.probs.shape == (BATCH, N_ACTIONS)
        # Probabilities should sum to 1
        assert torch.allclose(dist.probs.sum(dim=-1), torch.ones(BATCH), atol=1e-5)


# ---------------------------------------------------------------------------
# CriticNetwork
# ---------------------------------------------------------------------------

class TestCriticNetwork:
    def test_local_critic_shape(self, s_t):
        from agents.critic_network import CriticNetwork
        critic = CriticNetwork(state_dim=OBS_DIM, hidden_dims=[64, 64])
        v = critic(s_t)
        assert v.shape == (BATCH, 1)

    def test_local_critic_value(self, s_t):
        from agents.critic_network import CriticNetwork
        critic = CriticNetwork(state_dim=OBS_DIM, hidden_dims=[64, 64])
        v = critic.value(s_t)
        assert v.shape == (BATCH,)

    def test_1d_state(self):
        from agents.critic_network import CriticNetwork
        critic = CriticNetwork(state_dim=OBS_DIM, hidden_dims=[64, 64])
        s = torch.rand(OBS_DIM)
        v = critic(s)
        assert v.shape == (1, 1)

    def test_centralized_critic_shape(self):
        from agents.critic_network import CentralizedCriticNetwork
        n_sat = 3
        critic = CentralizedCriticNetwork(n_satellites=n_sat, state_dim=OBS_DIM,
                                          hidden_dims=[64, 64])
        global_state = torch.rand(BATCH, n_sat * OBS_DIM)
        v = critic(global_state)
        assert v.shape == (BATCH, 1)

    def test_centralized_critic_2d_input(self):
        """Accept (n_satellites, state_dim) as a single-step global obs."""
        from agents.critic_network import CentralizedCriticNetwork
        n_sat = 3
        critic = CentralizedCriticNetwork(n_satellites=n_sat, state_dim=OBS_DIM,
                                          hidden_dims=[64, 64])
        global_state = torch.rand(n_sat, OBS_DIM)
        v = critic(global_state)
        assert v.shape == (1, 1)


# ---------------------------------------------------------------------------
# ActionSelector
# ---------------------------------------------------------------------------

class TestActionSelector:
    def test_nominal_passthrough(self, obs_safe):
        from agents.action_selector import ActionSelector
        sel = ActionSelector(agent_id=0)
        action, overridden, reason = sel.select(policy_action=0, obs=obs_safe)
        assert action     == 0
        assert overridden == False
        assert reason     == "nominal"

    def test_critical_soc_forces_charge(self, obs_low_soc):
        from agents.action_selector import ActionSelector
        sel = ActionSelector(agent_id=0)
        action, overridden, reason = sel.select(policy_action=0, obs=obs_low_soc)
        assert action     == 4   # ACTION_CHARGE_PRIORITY
        assert overridden == True
        assert "CRITICAL" in reason

    def test_warn_soc_blocks_payload(self, obs_warn_soc):
        from agents.action_selector import ActionSelector
        sel = ActionSelector(agent_id=0)
        # policy wants payload ON but SoC is 0.12 (below min=0.15)
        action, overridden, reason = sel.select(policy_action=0, obs=obs_warn_soc)
        assert action     == 1   # ACTION_PAYLOAD_OFF
        assert overridden == True

    def test_warn_soc_allows_non_payload(self, obs_warn_soc):
        from agents.action_selector import ActionSelector
        sel = ActionSelector(agent_id=0)
        # relay_mode should still be allowed at warn-level SoC
        action, overridden, reason = sel.select(policy_action=3, obs=obs_warn_soc)
        assert overridden == False
        assert action == 3

    def test_high_temp_forces_hibernate(self, obs_high_temp):
        from agents.action_selector import ActionSelector
        sel = ActionSelector(agent_id=0)
        action, overridden, reason = sel.select(policy_action=0, obs=obs_high_temp)
        assert action     == 2   # ACTION_HIBERNATE
        assert overridden == True
        assert "CRITICAL" in reason or "Temperature" in reason

    def test_override_rate(self, obs_low_soc, obs_safe):
        from agents.action_selector import ActionSelector
        sel = ActionSelector(agent_id=0)
        sel.select(0, obs_low_soc)   # override
        sel.select(0, obs_safe)      # nominal
        assert abs(sel.override_rate() - 0.5) < 1e-6

    def test_reset_clears_log(self, obs_low_soc):
        from agents.action_selector import ActionSelector
        sel = ActionSelector(agent_id=0)
        sel.select(0, obs_low_soc)
        sel.reset()
        assert len(sel.override_log) == 0
        assert sel._step == 0


# ---------------------------------------------------------------------------
# SatelliteAgent
# ---------------------------------------------------------------------------

class TestSatelliteAgent:
    def _make_agent(self):
        from agents.satellite_agent import SatelliteAgent
        return SatelliteAgent.make(
            agent_id=0,
            predict_k=PREDICT_K,
            hidden_dims=[64, 64],
            device="cpu",
        )

    def test_act_returns_valid_action(self, obs_safe):
        agent = self._make_agent()
        action, log_prob, value, info = agent.act(obs_safe)
        assert 0 <= action < N_ACTIONS
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert "s_future" in info
        assert "raw_action" in info

    def test_act_safety_override_on_critical(self, obs_low_soc):
        agent = self._make_agent()
        action, _, _, info = agent.act(obs_low_soc)
        # SoC critical -> must be CHARGE_PRIORITY
        assert action == 4
        assert info["was_overridden"] == True

    def test_receive_command(self, obs_safe):
        agent = self._make_agent()
        agent.receive_command("hibernate")
        assert agent._current_task == "hibernate"
        assert agent._task_action_bias == 2

    def test_receive_command_none_clears(self, obs_safe):
        agent = self._make_agent()
        agent.receive_command("relay_mode")
        agent.receive_command(None)
        assert agent._task_action_bias is None

    def test_reset_episode(self, obs_safe):
        agent = self._make_agent()
        agent.act(obs_safe)
        agent.reset_episode()
        assert agent._step == 0
        assert agent._override_count == 0

    def test_state_dict_roundtrip(self, obs_safe):
        agent = self._make_agent()
        sd = agent.state_dict()
        assert "actor" in sd
        assert "critic" in sd
        # Load back
        agent_2 = self._make_agent()
        agent_2.load_state_dict(sd)
        a1, _, _, _ = agent.act(obs_safe, deterministic=True)
        a2, _, _, _ = agent_2.act(obs_safe, deterministic=True)
        assert a1 == a2

    def test_episode_summary(self, obs_safe, obs_low_soc):
        agent = self._make_agent()
        agent.act(obs_safe)
        agent.act(obs_low_soc)
        summary = agent.episode_summary()
        assert summary["agent_id"] == 0
        assert summary["steps"] == 2
        assert summary["safety_overrides"] >= 1
