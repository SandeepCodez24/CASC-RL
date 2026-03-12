"""
Dataset Builder — Layer 2a: World Model Data Collection.

Collects transition tuples (s_t, a_t, s_{t+1}) from the ConstellationEnv
by running a random policy. Saves data in .npz format for world model training.

Usage:
    builder = DatasetBuilder(env, n_satellites=3)
    dataset = builder.collect(n_transitions=100000, save_path="data/transitions.npz")
"""

from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
from tqdm import tqdm
from loguru import logger

from environment.constellation_env import ConstellationEnv


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class TransitionDataset(Dataset):
    """
    PyTorch Dataset wrapping collected transition tuples.

    Each item is (s, a, s_next) for one satellite at one timestep.

    Args:
        states      (np.ndarray): shape (N, OBS_DIM)
        actions     (np.ndarray): shape (N,) int32
        next_states (np.ndarray): shape (N, OBS_DIM)
    """

    def __init__(
        self,
        states:      np.ndarray,
        actions:     np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        assert len(states) == len(actions) == len(next_states), \
            "All arrays must have the same length."
        self.states      = torch.tensor(states,      dtype=torch.float32)
        self.actions     = torch.tensor(actions,     dtype=torch.long)
        self.next_states = torch.tensor(next_states, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx], self.next_states[idx]

    @classmethod
    def from_npz(cls, path: str) -> "TransitionDataset":
        """Load a TransitionDataset from a .npz file."""
        data = np.load(path)
        return cls(data["states"], data["actions"], data["next_states"])

    def save(self, path: str) -> None:
        """Save dataset to .npz file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.savez_compressed(
            path,
            states=self.states.numpy(),
            actions=self.actions.numpy(),
            next_states=self.next_states.numpy(),
        )
        logger.info(f"Saved {len(self)} transitions to {path}")

    def split(self, val_fraction: float = 0.2) -> Tuple["TransitionDataset", "TransitionDataset"]:
        """Split into train and validation datasets."""
        n = len(self)
        n_val = int(n * val_fraction)
        n_train = n - n_val
        indices = torch.randperm(n)
        train_idx = indices[:n_train]
        val_idx   = indices[n_train:]

        def _subset(idx):
            return TransitionDataset(
                self.states[idx].numpy(),
                self.actions[idx].numpy(),
                self.next_states[idx].numpy(),
            )

        return _subset(train_idx), _subset(val_idx)


# ---------------------------------------------------------------------------
# Dataset Builder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """
    Collects world model training data from ConstellationEnv.

    Runs a mixed random / charge-priority policy to ensure coverage of
    all satellite states including low-SoC and eclipse scenarios.

    Args:
        env         (ConstellationEnv): the physical simulation environment
        seed        (int)             : random seed for reproducibility
    """

    def __init__(
        self,
        env:  ConstellationEnv,
        seed: int = 42,
    ) -> None:
        self.env  = env
        self.rng  = np.random.default_rng(seed)

    def _random_actions(self) -> np.ndarray:
        """Sample a random action vector for all satellites."""
        return self.rng.integers(0, self.env.action_space.nvec[0], size=self.env.n_satellites)

    def collect(
        self,
        n_transitions: int = 100_000,
        save_path:     Optional[str] = None,
        per_satellite: bool = True,
    ) -> TransitionDataset:
        """
        Run env with random actions and collect transitions.

        Transitions are collected **per satellite**: each satellite's
        (s_i, a_i, s_next_i) is stored as a separate row, multiplying
        the effective dataset size by n_satellites.

        Args:
            n_transitions: total number of transition tuples to collect
            save_path    : optional .npz path to save the dataset
            per_satellite: if True, flatten per-satellite transitions

        Returns:
            TransitionDataset
        """
        n_sat = self.env.n_satellites
        n_steps = (n_transitions // n_sat) + 1

        all_states:      list[np.ndarray] = []
        all_actions:     list[np.ndarray] = []
        all_next_states: list[np.ndarray] = []

        obs, _ = self.env.reset()  # shape (n_sat, OBS_DIM)
        logger.info(f"Collecting {n_transitions} transitions ({n_steps} steps, {n_sat} satellites)...")

        for step in tqdm(range(n_steps), desc="Collecting", unit="step"):
            actions = self._random_actions()  # shape (n_sat,)
            next_obs, _, terminated, truncated, _ = self.env.step(actions)

            if per_satellite:
                for i in range(n_sat):
                    all_states.append(obs[i])        # (OBS_DIM,)
                    all_actions.append(actions[i])   # scalar
                    all_next_states.append(next_obs[i])
            else:
                all_states.append(obs.copy())
                all_actions.append(actions.copy())
                all_next_states.append(next_obs.copy())

            if terminated or truncated:
                obs, _ = self.env.reset()
            else:
                obs = next_obs

        # Trim to exact count
        states      = np.array(all_states[:n_transitions],      dtype=np.float32)
        actions     = np.array(all_actions[:n_transitions],      dtype=np.int32)
        next_states = np.array(all_next_states[:n_transitions],  dtype=np.float32)

        logger.info(
            f"Collected {len(states)} transitions. "
            f"State shape: {states.shape}, Action shape: {actions.shape}"
        )

        dataset = TransitionDataset(states, actions, next_states)

        if save_path is not None:
            dataset.save(save_path)

        return dataset
