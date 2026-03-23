"""
Satellite Agent — Layer 2b: Satellite Cognitive Layer (top-level agent).

SatelliteAgent is the complete cognitive loop for one satellite:
    1. Receive current observation from the environment
    2. Query WorldModel to predict k future states
    3. Feed (s_t, s_future) to ActorNetwork to sample an action
    4. Pass action through ActionSelector safety gate
    5. Return final safe action + metadata for the training loop

The agent also accepts high-level task commands from Layer 4 (hierarchical
coordinator), which it stores as a priority hint that biases action scoring.

Architecture diagram:

    Env obs  -->  WorldModel.predict_k_steps()  -->  s_future
                                                         |
    s_t  + s_future  -->  ActorNetwork.act()  -->  raw_action
                                                         |
    raw_action + s_t  -->  ActionSelector.select()  -->  safe_action
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from agents.policy_network    import ActorNetwork
from agents.critic_network    import CriticNetwork
from agents.action_selector   import ActionSelector, SafetyConstraints
from world_model.world_model  import WorldModel

OBS_DIM   = 8
N_ACTIONS = 5

# Action names for verbose logging
ACTION_NAMES = {
    0: "payload_ON",
    1: "payload_OFF",
    2: "hibernate",
    3: "relay_mode",
    4: "charge_priority",
}

# ----- Algorithm 4 reward weights (doc: w1=1.0, w2=0.5, w3=0.3) -----
# Index positions in the 8-dim observation vector (must match constellation_env.py)
_IDX_SOC  = 0   # State of Charge
_IDX_SOH  = 1   # State of Health
_IDX_TEMP = 2   # Temperature (normalized: raw_C / 100)
# w1·SoC  − w2·degradation_rate  − w3·thermal_risk
W1_SOC   = 1.0
W2_DEGRAD = 0.5
W3_THERMAL = 0.3


class SatelliteAgent:
    """
    Cognitive autonomous satellite agent (Layer 2b).

    Encapsulates the actor, critic, world model, and safety gate for a single
    satellite. Designed to be instantiated once per satellite in a constellation.

    Args:
        agent_id       (int)         : satellite index (0-based)
        actor          (ActorNetwork): policy network
        critic         (CriticNetwork): value network
        world_model    (WorldModel)  : dynamics prediction model
        action_selector(ActionSelector): safety constraint gate
        predict_k      (int)         : world model rollout horizon. Default 5.
        device         (str)         : "cpu" or "cuda"

    Typical usage:
        agent = SatelliteAgent.make(agent_id=0, predict_k=5, device="cpu")
        action, lp, val, info = agent.act(obs)
    """

    def __init__(
        self,
        agent_id:        int,
        actor:           ActorNetwork,
        critic:          CriticNetwork,
        world_model:     WorldModel,
        action_selector: ActionSelector,
        predict_k:       int = 5,
        device:          str = "cpu",
    ) -> None:
        self.agent_id        = agent_id
        self.actor           = actor
        self.critic          = critic
        self.world_model     = world_model
        self.action_selector = action_selector
        self.predict_k       = predict_k
        self.device          = torch.device(device)

        # Move networks to device
        self.actor.to(self.device)
        self.critic.to(self.device)

        # Layer 4 task command (priority hint)
        self._current_task: Optional[str] = None
        self._task_action_bias: Optional[int] = None

        # Episode statistics
        self._step = 0
        self._override_count = 0

    # ------------------------------------------------------------------
    # Factory constructor
    # ------------------------------------------------------------------

    @classmethod
    def make(
        cls,
        agent_id:    int = 0,
        predict_k:   int = 5,
        hidden_dims: Optional[List[int]] = None,
        device:      str = "cpu",
        constraints: Optional[SafetyConstraints] = None,
    ) -> "SatelliteAgent":
        """
        Convenience factory: create a SatelliteAgent with default components.

        Args:
            agent_id   : satellite index
            predict_k  : world model rollout horizon
            hidden_dims: actor hidden layer widths
            device     : "cpu" or "cuda"
            constraints: safety constraints (uses defaults if None)

        Returns:
            Fully initialized SatelliteAgent
        """
        actor   = ActorNetwork(state_dim=OBS_DIM, n_actions=N_ACTIONS,
                               predict_k=predict_k, hidden_dims=hidden_dims)
        critic  = CriticNetwork(state_dim=OBS_DIM)
        wm      = WorldModel(device=device)
        sel     = ActionSelector(constraints=constraints, agent_id=agent_id)
        return cls(agent_id, actor, critic, wm, sel, predict_k, device)

    # ------------------------------------------------------------------
    # Core cognitive decision pipeline
    # ------------------------------------------------------------------

    def act(
        self,
        obs:           np.ndarray,
        deterministic: bool = False,
        verbose:       bool = False,
    ) -> Tuple[int, float, float, Dict[str, Any]]:
        """
        Full cognitive decision pipeline.

        Args:
            obs          : observation from env, shape (OBS_DIM,) normalized
            deterministic: use greedy action (eval mode)
            verbose      : log decision details

        Returns:
            safe_action  (int)  : final action to execute in the environment
            log_prob     (float): log probability of the raw policy action
            value        (float): critic's state value estimate
            info         (dict) : metadata {raw_action, was_overridden, reason, s_future}
        """
        self._step += 1
        self.actor.eval()
        self.critic.eval()

        # Step 1: Build world model future predictions
        s_future_list = self.world_model.predict_k_steps(
            s_t=obs,
            actions=[ACTION_NAMES.get(self._task_action_bias, 1)] * self.predict_k
            if self._task_action_bias is not None
            else [1] * self.predict_k,               # default: payload_OFF
            k=self.predict_k,
        )
        # Stack into tensor (predict_k, OBS_DIM)
        s_future_np = np.stack(s_future_list, axis=0).astype(np.float32)

        # Step 2: Prepare tensors
        s_t_tensor      = torch.tensor(obs,         dtype=torch.float32, device=self.device).unsqueeze(0)
        s_future_tensor = torch.tensor(s_future_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        # s_future_tensor shape: (1, predict_k, OBS_DIM)

        with torch.no_grad():
            # Step 3: Actor forward pass
            action_tensor, log_prob_tensor, _ = self.actor.act(
                s_t_tensor, s_future_tensor, deterministic=deterministic
            )
            raw_action = int(action_tensor.item())
            log_prob   = float(log_prob_tensor.item())

            # Step 4: Critic value estimate
            value = float(self.critic.value(s_t_tensor).item())

        # Step 5: Safety gate
        safe_action, was_overridden, reason = self.action_selector.select(
            policy_action=raw_action,
            obs=obs,
            verbose=verbose,
        )

        if was_overridden:
            self._override_count += 1

        if verbose:
            logger.debug(
                f"[SAT-{self.agent_id}] step={self._step} | "
                f"raw={ACTION_NAMES[raw_action]} -> safe={ACTION_NAMES[safe_action]} | "
                f"V={value:.3f} | override={was_overridden} ({reason})"
            )

        info = {
            "raw_action":      raw_action,
            "was_overridden":  was_overridden,
            "override_reason": reason,
            "s_future":        s_future_np,
            "value":           value,
        }
        return safe_action, log_prob, value, info

    # ------------------------------------------------------------------
    # Algorithm 4  —  Explicit Cognitive Decision (Model-Predictive Control)
    # ------------------------------------------------------------------

    def cognitive_decision(
        self,
        obs:           np.ndarray,
        k:             int  = 5,
        verbose:       bool = False,
    ) -> Tuple[int, "Dict[str, Any]"]:
        """
        Implement Algorithm 4 from the project document exactly:

            For each candidate action a in {0..4}:
                s_future = world_model.predict_k_steps(s_t, [a]*k, k)
                score    = w1·SoC − w2·degradation_rate − w3·thermal_risk
            best_action = argmax(score)
            safe_action = safety_gate.filter(best_action, s_t)

        This is the explicit Model-Predictive Control (MPC) loop described
        in Algorithm 4.  Unlike act() which relies on the learned ActorNetwork,
        this path is fully transparent — every score is computed analytically
        from world model rollouts and the documented reward weights.

        Reward weights (project document, Algorithm 4):
            w1 = 1.0  (SoC — maximize battery state)
            w2 = 0.5  (degradation rate — minimize wear)
            w3 = 0.3  (thermal risk — minimize overheating)

        Args:
            obs     : current observation, shape (OBS_DIM,) — normalized
            k       : world model rollout horizon steps (default 5)
            verbose : log per-action scores to logger

        Returns:
            safe_action (int)  : best MPC action after safety filter
            info        (dict) : {scores, best_raw_action, best_score,
                                  was_overridden, override_reason, s_futures}
        """
        self._step += 1

        candidate_actions = list(range(N_ACTIONS))   # [0, 1, 2, 3, 4]
        scores:    Dict[int, float] = {}
        s_futures: Dict[int, list]  = {}

        # ---- Step 1-2: Rollout and score each candidate action --------
        for a_candidate in candidate_actions:
            action_seq    = [a_candidate] * k
            future_states = self.world_model.predict_k_steps(
                s_t=obs, actions=action_seq, k=k
            )
            s_futures[a_candidate] = future_states

            # Score the terminal predicted state (k-th step)
            s_k = future_states[-1]   # shape (OBS_DIM,)
            scores[a_candidate] = self._score_future_state(obs, s_k)

        # ---- Step 3: Select best scoring action -----------------------
        best_action = max(scores, key=lambda a: scores[a])
        best_score  = scores[best_action]

        if verbose:
            score_str = " | ".join(
                f"{ACTION_NAMES[a]}={scores[a]:.4f}" for a in candidate_actions
            )
            logger.debug(
                f"[SAT-{self.agent_id}] MPC scores: {score_str} "
                f"→ best={ACTION_NAMES[best_action]} (score={best_score:.4f})"
            )

        # ---- Step 4: Apply safety filter ------------------------------
        safe_action, was_overridden, reason = self.action_selector.select(
            policy_action=best_action,
            obs=obs,
            verbose=verbose,
        )

        if was_overridden:
            self._override_count += 1

        if verbose:
            logger.debug(
                f"[SAT-{self.agent_id}] MPC final: {ACTION_NAMES[safe_action]} "
                f"(override={was_overridden}, reason={reason})"
            )

        info = {
            "scores":           scores,
            "best_raw_action":  best_action,
            "best_score":       best_score,
            "was_overridden":   was_overridden,
            "override_reason":  reason,
            "s_futures":        s_futures,
        }
        return safe_action, info

    @staticmethod
    def _score_future_state(s_t: np.ndarray, s_future: np.ndarray) -> float:
        """
        Compute the Algorithm 4 scalar score for a predicted future state.

        Score = w1·SoC(future) − w2·ΔSoH − w3·thermal_risk(future)

        Weights (project document, Algorithm 4):
            w1=1.0, w2=0.5, w3=0.3

        Args:
            s_t     : current observation, shape (OBS_DIM,)
            s_future: terminal predicted state, shape (OBS_DIM,)

        Returns:
            float: scalar score (higher = better action)
        """
        soc_future   = float(s_future[_IDX_SOC])
        soh_loss     = float(max(0.0, s_t[_IDX_SOH] - s_future[_IDX_SOH]))  # degradation
        thermal_risk = float(s_future[_IDX_TEMP])   # normalized temp: 0=cold, 1=T_max

        return (
            W1_SOC     * soc_future
            - W2_DEGRAD  * soh_loss
            - W3_THERMAL * thermal_risk
        )

    # ------------------------------------------------------------------
    # Layer 4 interface
    # ------------------------------------------------------------------

    def receive_command(self, task: str) -> None:
        """
        Accept a high-level task command from the Layer 4 coordinator.

        The command is stored and used to bias the world model rollout action
        sequence during planning. Valid tasks:
            "payload_active" -> bias toward action 0 (PAYLOAD_ON)
            "hibernate"      -> bias toward action 2 (HIBERNATE)
            "relay_mode"     -> bias toward action 3 (RELAY_MODE)
            "charge_priority"-> bias toward action 4 (CHARGE_PRIORITY)
            None             -> clear command

        Args:
            task (str | None): task string or None to clear
        """
        task_map = {
            "payload_active":  0,
            "payload_off":     1,
            "hibernate":       2,
            "relay_mode":      3,
            "charge_priority": 4,
        }
        self._current_task = task
        self._task_action_bias = task_map.get(task) if task else None
        logger.info(f"[SAT-{self.agent_id}] Received task command: {task!r} (bias={self._task_action_bias})")

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Reset per-episode counters and safety log."""
        self._step = 0
        self._override_count = 0
        self.action_selector.reset()

    def episode_summary(self) -> Dict[str, Any]:
        """Return episode diagnostics."""
        return {
            "agent_id":        self.agent_id,
            "steps":           self._step,
            "safety_overrides": self._override_count,
            "override_rate":   self.action_selector.override_rate(),
            "current_task":    self._current_task,
        }

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Serialize agent state (actor + critic weights)."""
        return {
            "agent_id":     self.agent_id,
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        """Load actor + critic weights from a state dict."""
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def actor_parameters(self):
        return self.actor.parameters()

    @property
    def critic_parameters(self):
        return self.critic.parameters()

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()
