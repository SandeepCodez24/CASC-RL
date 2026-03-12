"""
Experience Buffer — Layer 3: Cooperative MARL.

Stores per-agent rollout trajectories for MAPPO training.

Each entry captures both the local observation (used by actor) and the
global state (used by the centralized critic). The buffer is episode-based:
data is filled during a rollout and then consumed during policy updates.

Stored per timestep per agent:
    s_local   : local observation   (OBS_DIM,)
    s_global  : global state        (n_agents * OBS_DIM,)
    action    : discrete action     scalar int
    log_prob  : log π_old(a|s)      scalar float
    reward    : per-agent reward    scalar float
    value     : V(s_global)         scalar float
    done      : episode termination scalar bool
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


OBS_DIM = 8


@dataclass
class AgentTransition:
    """Single timestep transition for one agent."""
    s_local:  np.ndarray   # (OBS_DIM,)
    s_global: np.ndarray   # (n_agents * OBS_DIM,)
    action:   int
    log_prob: float
    reward:   float
    value:    float
    done:     bool


class RolloutBuffer:
    """
    Episode-length rollout buffer for all agents in the constellation.

    Holds one full episode worth of transitions (T steps, N agents).
    Cleared between episodes. Used by MAPPOTrainer to compute GAE
    and perform mini-batch policy updates.

    Args:
        n_agents     (int): Number of satellite agents.
        obs_dim      (int): Per-agent observation dimension.
        episode_len  (int): Maximum steps per episode.
    """

    def __init__(
        self,
        n_agents:    int,
        obs_dim:     int = OBS_DIM,
        episode_len: int = 1000,
    ) -> None:
        self.n_agents    = n_agents
        self.obs_dim     = obs_dim
        self.episode_len = episode_len
        self.global_dim  = n_agents * obs_dim

        self._reset_storage()
        self.ptr = 0           # current insertion pointer
        self.full = False

    # ------------------------------------------------------------------
    # Storage management
    # ------------------------------------------------------------------

    def _reset_storage(self) -> None:
        T, N, D = self.episode_len, self.n_agents, self.obs_dim
        G = self.global_dim
        self.s_locals   = np.zeros((T, N, D),  dtype=np.float32)
        self.s_globals  = np.zeros((T, G),      dtype=np.float32)
        self.actions    = np.zeros((T, N),      dtype=np.int64)
        self.log_probs  = np.zeros((T, N),      dtype=np.float32)
        self.rewards    = np.zeros((T, N),      dtype=np.float32)
        self.values     = np.zeros((T,),        dtype=np.float32)
        self.dones      = np.zeros((T,),        dtype=np.float32)

    def clear(self) -> None:
        """Reset buffer for the next episode."""
        self._reset_storage()
        self.ptr  = 0
        self.full = False

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(
        self,
        s_locals:  np.ndarray,    # (n_agents, obs_dim)
        s_global:  np.ndarray,    # (n_agents * obs_dim,)
        actions:   np.ndarray,    # (n_agents,) int
        log_probs: np.ndarray,    # (n_agents,) float
        rewards:   np.ndarray,    # (n_agents,) float
        value:     float,         # scalar from centralized critic
        done:      bool,
    ) -> None:
        """
        Insert one timestep of transitions for all agents.

        Args:
            s_locals : stacked local observations, shape (n_agents, obs_dim)
            s_global : flattened global state,     shape (n_agents * obs_dim,)
            actions  : action per agent,           shape (n_agents,)
            log_probs: log π_old per agent,        shape (n_agents,)
            rewards  : reward per agent,           shape (n_agents,)
            value    : centralized critic value,   scalar
            done     : episode done flag
        """
        t = self.ptr
        self.s_locals[t]   = s_locals
        self.s_globals[t]  = s_global
        self.actions[t]    = actions
        self.log_probs[t]  = log_probs
        self.rewards[t]    = rewards
        self.values[t]     = value
        self.dones[t]      = float(done)

        self.ptr = t + 1
        if self.ptr >= self.episode_len:
            self.full = True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of valid transitions stored."""
        return self.ptr

    def get_tensors(
        self, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Convert stored numpy arrays to tensors on the given device.

        Only returns the valid portion [0:ptr].

        Returns dict with keys:
            s_locals, s_globals, actions, log_probs, rewards, values, dones
        """
        T = self.ptr
        return {
            "s_locals":  torch.tensor(self.s_locals[:T],  device=device),
            "s_globals": torch.tensor(self.s_globals[:T], device=device),
            "actions":   torch.tensor(self.actions[:T],   device=device, dtype=torch.long),
            "log_probs": torch.tensor(self.log_probs[:T], device=device),
            "rewards":   torch.tensor(self.rewards[:T],   device=device),
            "values":    torch.tensor(self.values[:T],    device=device),
            "dones":     torch.tensor(self.dones[:T],     device=device),
        }

    def mini_batches(
        self,
        batch_size: int,
        device:     torch.device,
        advantages: torch.Tensor,
        returns:    torch.Tensor,
    ):
        """
        Yield random mini-batches for PPO epoch updates.

        Args:
            batch_size : size of each mini-batch
            device     : torch device
            advantages : pre-computed GAE advantages, shape (T, n_agents)
            returns    : pre-computed returns,         shape (T, n_agents)

        Yields:
            dict with mini-batch tensors + advantages + returns
        """
        tensors = self.get_tensors(device)
        T = self.ptr
        indices = torch.randperm(T, device=device)

        for start in range(0, T, batch_size):
            idx = indices[start: start + batch_size]
            yield {
                "s_locals":   tensors["s_locals"][idx],
                "s_globals":  tensors["s_globals"][idx],
                "actions":    tensors["actions"][idx],
                "log_probs":  tensors["log_probs"][idx],
                "advantages": advantages[idx],
                "returns":    returns[idx],
            }
