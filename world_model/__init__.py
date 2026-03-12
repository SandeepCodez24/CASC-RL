"""
Layer 2 — world_model package.

Exports:
    DynamicsNetwork         — single MLP residual dynamics model
    EnsembleDynamicsNetwork — N-member ensemble for uncertainty estimation
    WorldModel              — high-level interface with normalization + rollout
    Normalizer              — running mean/std normalizer
    DatasetBuilder          — collects transitions from ConstellationEnv
    TransitionDataset       — PyTorch Dataset of (s, a, s_next) tuples
    WorldModelTrainer       — training loop for the ensemble
"""

from world_model.dynamics_network import DynamicsNetwork, EnsembleDynamicsNetwork
from world_model.world_model import WorldModel, Normalizer
from world_model.dataset_builder import DatasetBuilder, TransitionDataset
from world_model.training import WorldModelTrainer

__all__ = [
    "DynamicsNetwork",
    "EnsembleDynamicsNetwork",
    "WorldModel",
    "Normalizer",
    "DatasetBuilder",
    "TransitionDataset",
    "WorldModelTrainer",
]
