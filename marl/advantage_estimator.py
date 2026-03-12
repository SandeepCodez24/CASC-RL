"""
Advantage Estimator — Layer 3: Cooperative MARL.

Implements Generalized Advantage Estimation (GAE) for MAPPO.

GAE formula:
    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t     = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}

The returns (targets for value function) are computed as:
    G_t = A_t + V(s_t)

GAE smooths the bias-variance tradeoff:
    lambda = 1.0  -> high variance (Monte Carlo estimate)
    lambda = 0.0  -> high bias    (1-step TD estimate)
    lambda = 0.95 -> best empirical performance (Schulman et al.)
"""

from __future__ import annotations

import torch
from typing import Tuple


class GAEEstimator:
    """
    Generalized Advantage Estimation for multi-agent trajectories.

    Used by MAPPOTrainer after each rollout to compute per-agent
    advantage estimates before the PPO update epochs.

    Args:
        gamma  (float): discount factor.         Default 0.99.
        lam    (float): GAE lambda smoothing.    Default 0.95.
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        self.gamma = gamma
        self.lam   = lam

    def compute(
        self,
        rewards:       torch.Tensor,    # (T, n_agents)
        values:        torch.Tensor,    # (T,)          centralized critic per step
        dones:         torch.Tensor,    # (T,)          binary episode-end flags
        last_value:    float = 0.0,     # V(s_{T}) for bootstrap
        n_agents:      int   = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and discounted returns.

        Args:
            rewards    : per-agent rewards,        shape (T, n_agents)
            values     : centralized value per step, shape (T,)
            dones      : done flags per step,      shape (T,)
            last_value : bootstrap value at end of trajectory
            n_agents   : number of agents (for broadcasting)

        Returns:
            advantages : shape (T, n_agents)   -- normalized
            returns    : shape (T, n_agents)   -- used as value targets
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)  # (T, n_agents)

        # Append bootstrap value for terminal step
        values_ext = torch.cat([values, torch.tensor([last_value], device=values.device)])

        gae = 0.0
        for t in reversed(range(T)):
            mask     = 1.0 - dones[t]
            delta    = rewards[t] + self.gamma * values_ext[t + 1] * mask - values_ext[t]
            gae      = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae   # broadcast across agents

        returns = advantages + values.unsqueeze(-1).expand_as(advantages)

        # Normalize advantages per agent (zero-mean, unit-variance)
        advantages = self._normalize(advantages)
        return advantages, returns

    @staticmethod
    def _normalize(adv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize advantage tensor to zero mean, unit variance."""
        mean = adv.mean()
        std  = adv.std() + eps
        return (adv - mean) / std


def compute_gae(
    rewards:    torch.Tensor,
    values:     torch.Tensor,
    dones:      torch.Tensor,
    last_value: float = 0.0,
    gamma:      float = 0.99,
    lam:        float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Functional API for GAE computation.

    Args:
        rewards    : (T, n_agents)
        values     : (T,)
        dones      : (T,)
        last_value : bootstrap V at end
        gamma      : discount
        lam        : smoothing

    Returns:
        advantages : (T, n_agents)
        returns    : (T, n_agents)
    """
    estimator = GAEEstimator(gamma=gamma, lam=lam)
    return estimator.compute(rewards, values, dones, last_value)
