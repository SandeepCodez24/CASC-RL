"""
Layer 3 — marl package (Cooperative Intelligence Layer).

Exports:
    RolloutBuffer             — episode rollout storage for MAPPO
    GAEEstimator              — Generalized Advantage Estimation
    compute_gae               — functional GAE API
    CooperativeRewardShaper   — blended local + global reward
    cooperative_reward        — functional cooperative reward API
    CoopRewardWeights         — reward weight config dataclass
    ISLinkModel               — inter-satellite link latency model
    CommunicationProtocol     — state broadcasts + command routing
    StateMessage              — peer state broadcast data type
    CommandMessage            — mission command data type
    MAPPOTrainer              — full MAPPO training orchestrator
"""

from marl.buffer                 import RolloutBuffer
from marl.advantage_estimator    import GAEEstimator, compute_gae
from marl.cooperative_rewards    import (
    CooperativeRewardShaper,
    cooperative_reward,
    CoopRewardWeights,
)
from marl.communication_protocol import (
    CommunicationProtocol,
    ISLinkModel,
    StateMessage,
    CommandMessage,
)
from marl.mappo_trainer          import MAPPOTrainer

__all__ = [
    "RolloutBuffer",
    "GAEEstimator",
    "compute_gae",
    "CooperativeRewardShaper",
    "cooperative_reward",
    "CoopRewardWeights",
    "ISLinkModel",
    "CommunicationProtocol",
    "StateMessage",
    "CommandMessage",
    "MAPPOTrainer",
]
