"""
Critic Network — Layer 2b / Layer 3.

Two critic variants:
    1. CriticNetwork            — local critic V(s_i),   used in single-agent PPO
    2. CentralizedCriticNetwork — global critic V(S_all), used in MAPPO (Layer 3)

The centralized critic is defined here to keep network files co-located,
but it is primarily consumed by the MARL layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

OBS_DIM = 8


def _layernorm_mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:          # no activation on output layer
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class CriticNetwork(nn.Module):
    """
    Local value network V(s_i) for a single satellite agent.

    Used for advantage estimation in PPO training of individual agents.

    Args:
        state_dim   (int): Local observation dimension. Default 8.
        hidden_dims (list): Hidden layer widths. Default [256, 256].
    """

    def __init__(
        self,
        state_dim:   int       = OBS_DIM,
        hidden_dims: list[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        dims = [state_dim] + hidden_dims + [1]
        self.net = _layernorm_mlp(dims)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute state value estimate.

        Args:
            s: local state, shape (batch, state_dim) or (state_dim,)

        Returns:
            value: shape (batch, 1)
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        return self.net(s)   # (batch, 1)

    def value(self, s: torch.Tensor) -> torch.Tensor:
        """Returns scalar value per sample: shape (batch,)."""
        return self.forward(s).squeeze(-1)


class CentralizedCriticNetwork(nn.Module):
    """
    Centralized value network V(S_all) for MAPPO (Layer 3).

    Takes the concatenated global state from all satellites as input.
    Defined here for co-location, but imported and used by mappo_trainer.

    Args:
        n_satellites (int): Number of satellites. Sets global_state_dim.
        state_dim    (int): Per-satellite obs dim. Default 8.
        hidden_dims  (list): Hidden layer widths. Default [256, 256].
    """

    def __init__(
        self,
        n_satellites: int       = 3,
        state_dim:    int       = OBS_DIM,
        hidden_dims:  list[int] = None,
    ) -> None:
        super().__init__()
        self.global_state_dim = n_satellites * state_dim
        if hidden_dims is None:
            hidden_dims = [256, 256]
        dims = [self.global_state_dim] + hidden_dims + [1]
        self.net = _layernorm_mlp(dims)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Compute global value estimate from concatenated satellite states.

        Args:
            global_state: shape (batch, n_satellites * state_dim)
                          or (n_satellites, state_dim) for a single step

        Returns:
            value: shape (batch, 1)
        """
        if global_state.dim() == 2 and global_state.shape[-1] != self.global_state_dim:
            # Input was (n_satellites, state_dim) — flatten
            global_state = global_state.view(1, -1)
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        return self.net(global_state)   # (batch, 1)

    def value(self, global_state: torch.Tensor) -> torch.Tensor:
        """Returns scalar value per sample: shape (batch,)."""
        return self.forward(global_state).squeeze(-1)
