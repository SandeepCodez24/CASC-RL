"""
Cooperative Reward Shaping — Layer 3: Cooperative MARL.

Blends local per-agent rewards with a global team reward signal to
encourage cooperative rather than purely self-interested behavior.

Reward components:
    1. Local reward     : per-satellite SoC, SoH, thermal safety (from env)
    2. Global reward    : constellation-level mission success rate
    3. Conflict penalty : penalize resource contention (two sats relay simultaneously)
    4. Coordination bonus: reward well-distributed task coverage

Formula:
    r_cooperative = alpha * mean(local_rewards)
                  + beta  * global_outcome
                  - gamma * conflict_penalty
                  + delta * coordination_bonus
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CoopRewardWeights:
    """
    Mixing weights for cooperative reward components.

    Defaults match project document specification:
        alpha: weight for mean local reward
        beta : weight for global mission outcome
        gamma: penalty coefficient for relay conflicts
        delta: weight for coordination diversity bonus
    """
    alpha: float = 0.6    # local reward weight
    beta:  float = 0.3    # global outcome weight
    gamma: float = 0.5    # conflict penalty
    delta: float = 0.1    # coordination bonus


class CooperativeRewardShaper:
    """
    Computes blended cooperative rewards for the entire constellation.

    Args:
        n_agents (int): number of satellite agents
        weights  (CoopRewardWeights): mixing coefficients
    """

    def __init__(
        self,
        n_agents: int,
        weights:  Optional[CoopRewardWeights] = None,
    ) -> None:
        self.n_agents = n_agents
        self.w        = weights or CoopRewardWeights()

    # ------------------------------------------------------------------
    # Main shaping call
    # ------------------------------------------------------------------

    def shape(
        self,
        local_rewards:     np.ndarray,       # (n_agents,) per-agent reward from env
        actions:           np.ndarray,        # (n_agents,) discrete actions taken
        obs:               np.ndarray,        # (n_agents, OBS_DIM) current observations
        mission_complete:  bool = False,      # True if task was completed this step
    ) -> np.ndarray:
        """
        Compute shaped cooperative reward for all agents.

        Args:
            local_rewards   : raw per-agent rewards from `constellaton_env.step()`
            actions         : actions executed this step
            obs             : observation matrix for conflict detection
            mission_complete: True if a satellite completed a payload task

        Returns:
            shaped_rewards: np.ndarray shape (n_agents,)
        """
        global_outcome    = self._global_mission_reward(mission_complete, local_rewards)
        conflict_penalty  = self._conflict_penalty(actions)
        coord_bonus       = self._coordination_bonus(actions)

        shaped = (
            self.w.alpha * local_rewards
            + self.w.beta  * global_outcome
            - self.w.gamma * conflict_penalty
            + self.w.delta * coord_bonus
        )
        return shaped.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal reward components
    # ------------------------------------------------------------------

    def _global_mission_reward(
        self,
        mission_complete: bool,
        local_rewards:    np.ndarray,
    ) -> float:
        """
        Global team reward: mission bonus + mean fleet SoC health proxy.

        Returns a scalar shared across all agents.
        """
        # Mission completion bonus
        completion_bonus = 1.0 if mission_complete else 0.0

        # Fleet health proxy: mean of local rewards (already includes SoC)
        # Adding this separately from the alpha term gives extra credit to
        # episodes where the whole fleet stays healthy
        fleet_health = float(np.clip(np.mean(local_rewards), -1.0, 1.0)) * 0.3

        return completion_bonus + fleet_health

    def _conflict_penalty(self, actions: np.ndarray) -> float:
        """
        Penalize cases where multiple satellites choose RELAY_MODE simultaneously.

        In a real deployment, two satellites cannot relay to the same ground
        station simultaneously without causing interference. We approximate
        conflict as any timestep where > 1 satellite selects relay (action=3).

        Returns:
            penalty (float): 0.0 if no conflict, 1.0 if two or more relay,
                             scaled linearly for larger conflicts.
        """
        ACTION_RELAY_MODE = 3
        n_relaying = np.sum(actions == ACTION_RELAY_MODE)
        if n_relaying <= 1:
            return 0.0
        # Penalty scales with number of excess relayers
        return float(n_relaying - 1) / max(self.n_agents - 1, 1)

    def _coordination_bonus(self, actions: np.ndarray) -> float:
        """
        Reward diversity of task coverage across the constellation.

        A constellation that has sats in different modes (payload, relay,
        charge) is more productive than one where all sats do the same thing.

        Returns:
            bonus (float): 0.0-1.0. Higher when more distinct modes are active.
        """
        n_unique_actions = len(np.unique(actions))
        # Normalize: full diversity = n_actions unique, minimum = 1
        max_diversity = min(self.n_agents, 5)   # 5 action types
        return float(n_unique_actions - 1) / max(max_diversity - 1, 1)


# ---------------------------------------------------------------------------
# Functional API (matches project document pseudocode)
# ---------------------------------------------------------------------------

def cooperative_reward(
    local_rewards:    np.ndarray,
    global_outcome:   float,
    conflict_detected: bool,
    alpha: float = 0.6,
    beta:  float = 0.3,
    gamma: float = 0.5,
) -> np.ndarray:
    """
    Pure-function cooperative reward blend.

    Matches the project document formula:
        r = alpha * mean(local_rewards) + beta * global_outcome - gamma * conflict

    Args:
        local_rewards    : (n_agents,) per-agent rewards
        global_outcome   : scalar team-level mission score
        conflict_detected: True if relay conflict occurred
        alpha, beta, gamma: blend weights

    Returns:
        shaped rewards: (n_agents,) float32
    """
    conflict_val = 1.0 if conflict_detected else 0.0
    blended = (
        alpha * local_rewards
        + beta  * global_outcome
        - gamma * conflict_val
    )
    return blended.astype(np.float32)
