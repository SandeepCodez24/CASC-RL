"""
Dynamics Network — Layer 2a: Predictive World Model.

fψ(s_t, a_t) → s_{t+1}   (residual prediction: s_{t+1} = s_t + Δs)

Architecture:
    - Single DynamicsNetwork: shallow residual MLP
    - EnsembleDynamicsNetwork: N independent networks for uncertainty estimation

Input:
    state  (OBS_DIM = 8): [SoC, SoH, T, P_solar, phase, eclipse, P_consumed, comm_delay]
    action (int 0-4)     : one-hot encoded to N_ACTIONS = 5

Output:
    predicted next state  (OBS_DIM = 8)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# Dimensions matching Layer 1 constellation_env.py
OBS_DIM    = 8
N_ACTIONS  = 5
INPUT_DIM  = OBS_DIM + N_ACTIONS  # 13


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mlp(
    in_dim: int,
    hidden_dims: list[int],
    out_dim: int,
    use_layernorm: bool = True,
) -> nn.Sequential:
    """Construct a multi-layer perceptron with optional LayerNorm."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if use_layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _one_hot(action: torch.Tensor, n_classes: int = N_ACTIONS) -> torch.Tensor:
    """Convert integer action tensor to one-hot float tensor.

    Args:
        action: shape (...,) dtype int64
        n_classes: number of action classes

    Returns:
        shape (..., n_classes) dtype float32
    """
    return F.one_hot(action.long(), num_classes=n_classes).float()


# ---------------------------------------------------------------------------
# Single Dynamics Network
# ---------------------------------------------------------------------------

class DynamicsNetwork(nn.Module):
    """
    Single neural dynamics model fψ.

    Predicts the *delta* (change in state) rather than the absolute next state.
    This residual formulation improves training stability for states like SoC
    and SoH that evolve slowly.

    Args:
        state_dim  (int): Dimensionality of the state vector. Default 8.
        action_dim (int): Number of discrete actions. Default 5.
        hidden_dim (int): Width of hidden layers. Default 256.
        n_layers   (int): Number of hidden layers. Default 3.

    Forward signature:
        forward(s, a) -> s_next_pred
    """

    def __init__(
        self,
        state_dim:  int = OBS_DIM,
        action_dim: int = N_ACTIONS,
        hidden_dim: int = 256,
        n_layers:   int = 3,
    ) -> None:
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim

        hidden_dims = [hidden_dim] * n_layers
        self.net = _build_mlp(
            in_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            out_dim=state_dim,
            use_layernorm=True,
        )

    def forward(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next state via residual: s_next = s + delta.

        Args:
            s: state tensor,  shape (batch, state_dim)  or (state_dim,)
            a: action tensor, shape (batch,) int64       or scalar int64

        Returns:
            s_next: predicted next state, shape (batch, state_dim)
        """
        # Ensure batch dimension
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if a.dim() == 0:
            a = a.unsqueeze(0)

        a_onehot = _one_hot(a, self.action_dim).to(s.device)  # (batch, action_dim)
        x = torch.cat([s, a_onehot], dim=-1)                  # (batch, state_dim+action_dim)
        delta = self.net(x)                                     # (batch, state_dim)
        s_next = s + delta                                      # residual addition
        # Clip SoC and SoH to [0, 1] (indices 0 and 1)
        s_next = torch.cat([
            s_next[..., :2].clamp(0.0, 1.0),
            s_next[..., 2:],
        ], dim=-1)
        return s_next


# ---------------------------------------------------------------------------
# Ensemble Dynamics Network
# ---------------------------------------------------------------------------

class EnsembleDynamicsNetwork(nn.Module):
    """
    Ensemble of N independent DynamicsNetworks.

    Provides **uncertainty estimates** (epistemic) by measuring disagreement
    across ensemble members. Used in the WorldModel for risk-aware planning.

    Args:
        n_ensemble  (int): Number of ensemble members. Default 5.
        state_dim   (int): State dimensionality. Default 8.
        action_dim  (int): Number of discrete actions. Default 5.
        hidden_dim  (int): Width per network. Default 256.
        n_layers    (int): Depth per network. Default 3.
    """

    def __init__(
        self,
        n_ensemble: int = 5,
        state_dim:  int = OBS_DIM,
        action_dim: int = N_ACTIONS,
        hidden_dim: int = 256,
        n_layers:   int = 3,
    ) -> None:
        super().__init__()
        self.n_ensemble = n_ensemble
        self.members = nn.ModuleList([
            DynamicsNetwork(state_dim, action_dim, hidden_dim, n_layers)
            for _ in range(n_ensemble)
        ])

    def forward(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all ensemble members.

        Args:
            s: state tensor,  shape (batch, state_dim)
            a: action tensor, shape (batch,) int64

        Returns:
            mean: mean predicted next state,  shape (batch, state_dim)
            std:  std  predicted next state,  shape (batch, state_dim)
        """
        preds = torch.stack(
            [member(s, a) for member in self.members], dim=0
        )  # (n_ensemble, batch, state_dim)
        mean = preds.mean(dim=0)
        std  = preds.std(dim=0)
        return mean, std

    def forward_single(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        member_idx: int = 0,
    ) -> torch.Tensor:
        """Return prediction from a single ensemble member (for training)."""
        return self.members[member_idx](s, a)

    def parameters_per_member(self, member_idx: int):
        """Yield parameters for a specific member (for separate optimizers)."""
        return self.members[member_idx].parameters()
