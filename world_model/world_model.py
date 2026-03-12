"""
World Model — Layer 2a: Predictive World Model.

Wraps EnsembleDynamicsNetwork with:
- Input/output normalization (running stats fitted on the dataset)
- Single-step and multi-step (k-step) rollout prediction
- Checkpoint save/load
- Uncertainty-aware horizon forecasting

Usage:
    wm = WorldModel(device="cpu")
    wm.fit_normalizer(states, actions)
    s_future = wm.predict_k_steps(s_t, action_seq, k=5)
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from world_model.dynamics_network import (
    EnsembleDynamicsNetwork,
    DynamicsNetwork,
    OBS_DIM,
    N_ACTIONS,
)


# ---------------------------------------------------------------------------
# Running Normalizer
# ---------------------------------------------------------------------------

class Normalizer:
    """
    Online running mean/std normalizer (Welford's algorithm).

    Stores stats as numpy arrays and converts to torch for network use.
    """

    def __init__(self, dim: int) -> None:
        self.dim   = dim
        self.mean  = np.zeros(dim, dtype=np.float32)
        self.var   = np.ones(dim,  dtype=np.float32)
        self.count = 0

    def fit(self, data: np.ndarray) -> None:
        """Fit normalizer on a 2D array (N, dim)."""
        self.mean  = data.mean(axis=0).astype(np.float32)
        self.var   = data.var(axis=0).astype(np.float32) + 1e-8
        self.count = len(data)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor using stored stats."""
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std  = torch.tensor(np.sqrt(self.var), dtype=x.dtype, device=x.device)
        return (x - mean) / (std + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse normalization."""
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std  = torch.tensor(np.sqrt(self.var), dtype=x.dtype, device=x.device)
        return x * std + mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean  = d["mean"]
        self.var   = d["var"]
        self.count = d["count"]


# ---------------------------------------------------------------------------
# World Model
# ---------------------------------------------------------------------------

class WorldModel:
    """
    High-level interface to the satellite dynamics prediction model.

    Combines an EnsembleDynamicsNetwork with input normalization and
    convenience methods for single-step and multi-step rollouts.

    Args:
        state_dim   (int): State vector dimension. Default OBS_DIM (8).
        action_dim  (int): Number of discrete actions. Default N_ACTIONS (5).
        hidden_dim  (int): Hidden dim per network. Default 256.
        n_layers    (int): Depth per network. Default 3.
        n_ensemble  (int): Ensemble size. Default 5.
        device      (str): "cpu" or "cuda".
    """

    def __init__(
        self,
        state_dim:  int = OBS_DIM,
        action_dim: int = N_ACTIONS,
        hidden_dim: int = 256,
        n_layers:   int = 3,
        n_ensemble: int = 5,
        device:     str = "cpu",
    ) -> None:
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device     = torch.device(device)

        self.network = EnsembleDynamicsNetwork(
            n_ensemble=n_ensemble,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        ).to(self.device)

        self.state_normalizer  = Normalizer(state_dim)
        self.is_fitted         = False

    # ------------------------------------------------------------------
    # Normalizer fitting
    # ------------------------------------------------------------------

    def fit_normalizer(
        self,
        states: np.ndarray,
        actions: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit the state normalizer from a dataset of observed states.

        Args:
            states: array of shape (N, state_dim)
            actions: unused, kept for API consistency
        """
        self.state_normalizer.fit(states)
        self.is_fitted = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def predict_one_step(
        self,
        s: np.ndarray | torch.Tensor,
        a: int | np.ndarray | torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict next state for a single (s, a) pair.

        Args:
            s: state array/tensor, shape (state_dim,) or (batch, state_dim)
            a: integer action or array of actions
            return_uncertainty: if True, also return std across ensemble

        Returns:
            s_next_mean: predicted next state, numpy array
            s_next_std : std over ensemble (only if return_uncertainty=True)
        """
        self.network.eval()
        with torch.no_grad():
            if not isinstance(s, torch.Tensor):
                s = self._to_tensor(np.asarray(s, dtype=np.float32))
            if s.dim() == 1:
                s = s.unsqueeze(0)

            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, dtype=torch.long, device=self.device)
            if a.dim() == 0:
                a = a.unsqueeze(0)

            # Normalize state
            if self.is_fitted:
                s_norm = self.state_normalizer.normalize(s)
            else:
                s_norm = s

            mean, std = self.network(s_norm, a)

            # Denormalize
            if self.is_fitted:
                mean = self.state_normalizer.denormalize(mean)

            mean_np = mean.squeeze(0).cpu().numpy()
            std_np  = std.squeeze(0).cpu().numpy()

        if return_uncertainty:
            return mean_np, std_np
        return mean_np, None

    def predict_k_steps(
        self,
        s_t: np.ndarray | torch.Tensor,
        actions: List[int] | np.ndarray,
        k: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Rollout k-step prediction by chaining single-step predictions.

        Args:
            s_t    : initial state, shape (state_dim,)
            actions: sequence of integer actions, length k
            k      : number of steps (defaults to len(actions))

        Returns:
            List of k predicted states, each shape (state_dim,)
        """
        if k is None:
            k = len(actions)

        states: List[np.ndarray] = []
        s_curr = np.asarray(s_t, dtype=np.float32)

        for i in range(k):
            s_next, _ = self.predict_one_step(s_curr, int(actions[i]))
            states.append(s_next)
            s_curr = s_next

        return states

    def predict_horizon(
        self,
        s_t: np.ndarray,
        horizon: int = 10,
        action_fn=None,
    ) -> List[np.ndarray]:
        """
        Forecast over a horizon using a provided action generation function.

        Args:
            s_t       : initial state, shape (state_dim,)
            horizon   : number of future steps to predict
            action_fn : callable(s) -> int. Defaults to action=1 (payload_OFF)

        Returns:
            List of horizon predicted states.
        """
        if action_fn is None:
            action_fn = lambda s: 1   # payload_OFF default

        actions = [action_fn(s_t) for _ in range(horizon)]
        return self.predict_k_steps(s_t, actions, k=horizon)

    def predict_all_actions(
        self,
        s_t: np.ndarray,
        k: int = 5,
    ) -> dict:
        """
        Predict k-step rollouts for every possible action (action branching).

        Args:
            s_t: initial state, shape (state_dim,)
            k  : rollout horizon

        Returns:
            dict mapping action_id -> List[np.ndarray] of k future states
        """
        results = {}
        for a in range(self.action_dim):
            actions = [a] * k
            results[a] = self.predict_k_steps(s_t, actions, k)
        return results

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save network weights and normalizer to a .pt file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "state_normalizer": self.state_normalizer.state_dict(),
            "is_fitted": self.is_fitted,
            "config": {
                "state_dim":  self.state_dim,
                "action_dim": self.action_dim,
            },
        }, path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        """Load network weights and normalizer from a .pt file."""
        ckpt = torch.load(path, map_location=map_location or str(self.device))
        self.network.load_state_dict(ckpt["network"])
        self.state_normalizer.load_state_dict(ckpt["state_normalizer"])
        self.is_fitted = ckpt.get("is_fitted", False)
        self.network.to(self.device)

    def train_mode(self) -> None:
        self.network.train()

    def eval_mode(self) -> None:
        self.network.eval()

    def to(self, device: str) -> "WorldModel":
        self.device = torch.device(device)
        self.network = self.network.to(self.device)
        return self
