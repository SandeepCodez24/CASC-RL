"""
Policy Network (Actor) — Layer 2b: Satellite Cognitive Layer.

Architecture: ActorNetwork  π(a | s_t, s_future)

The actor takes the current state AND the k-step world model predicted future
states as input, enabling the agent to make forward-looking decisions.

Input:
    s_t      : current observation, shape (OBS_DIM,)       = (8,)
    s_future : k predicted future states flattened,          = (k * OBS_DIM,)
    concat   : (OBS_DIM + k * OBS_DIM,)

Output:
    Categorical probability distribution over N_ACTIONS = 5 actions
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional

OBS_DIM   = 8
N_ACTIONS = 5


def _layernorm_mlp(dims: list[int]) -> nn.Sequential:
    """Build MLP: Linear -> LayerNorm -> ReLU for hidden layers, Linear for output."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:          # no activation on output layer
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    """
    Actor (policy) network for a single satellite agent.

    Concatenates current state s_t with flattened world-model predictions
    (s_future) to produce a discrete action distribution.

    Args:
        state_dim   (int): Observation dimension. Default 8.
        n_actions   (int): Number of discrete actions. Default 5.
        predict_k   (int): Number of future steps from world model. Default 5.
        hidden_dims (list): Hidden layer widths. Default [512, 256, 128].
    """

    def __init__(
        self,
        state_dim:   int       = OBS_DIM,
        n_actions:   int       = N_ACTIONS,
        predict_k:   int       = 5,
        hidden_dims: list[int] = None,
    ) -> None:
        super().__init__()
        self.state_dim  = state_dim
        self.n_actions  = n_actions
        self.predict_k  = predict_k

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        in_dim = state_dim + predict_k * state_dim   # current + k future states
        all_dims = [in_dim] + hidden_dims + [n_actions]
        self.net = _layernorm_mlp(all_dims)

    def forward(
        self,
        s_t:      torch.Tensor,
        s_future: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute raw action logits.

        Args:
            s_t     : current state, shape (batch, state_dim)
            s_future: predicted future states, shape (batch, predict_k, state_dim)
                      OR already flattened (batch, predict_k * state_dim)

        Returns:
            logits: shape (batch, n_actions)
        """
        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)
        if s_future.dim() == 3:
            batch = s_future.shape[0]
            s_future = s_future.view(batch, -1)    # flatten k * state_dim
        elif s_future.dim() == 2 and s_future.shape[-1] == self.state_dim:
            # (k, state_dim) with no batch — unsqueeze as batch=1
            s_future = s_future.unsqueeze(0).view(1, -1)

        x = torch.cat([s_t, s_future], dim=-1)    # (batch, in_dim)
        return self.net(x)

    def get_distribution(
        self,
        s_t:      torch.Tensor,
        s_future: torch.Tensor,
    ) -> Categorical:
        """Return Categorical distribution over actions."""
        logits = self.forward(s_t, s_future)
        return Categorical(logits=logits)

    def act(
        self,
        s_t:          torch.Tensor,
        s_future:     torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or argmax) an action and return log-prob + entropy.

        Args:
            s_t          : current state
            s_future     : k predicted future states
            deterministic: if True, return argmax (greedy) action

        Returns:
            action   : shape (batch,) int64
            log_prob : shape (batch,) float
            entropy  : shape (batch,) float
        """
        dist = self.get_distribution(s_t, s_future)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy

    def evaluate_actions(
        self,
        s_t:      torch.Tensor,
        s_future: torch.Tensor,
        actions:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-probs and entropy for given actions (used in PPO update).

        Args:
            s_t     : current state, shape (batch, state_dim)
            s_future: future states,  shape (batch, predict_k * state_dim)
            actions : actions taken,  shape (batch,) int64

        Returns:
            log_prob : shape (batch,)
            entropy  : shape (batch,)
        """
        dist = self.get_distribution(s_t, s_future)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, entropy
