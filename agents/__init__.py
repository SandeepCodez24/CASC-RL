"""
Layer 2b — agents package.

Exports:
    ActorNetwork            — policy network π(a | s_t, s_future)
    CriticNetwork           — local value network V(s_i)
    CentralizedCriticNetwork— global value network V(S_all) for MAPPO
    SafetyConstraints       — dataclass of hard safety thresholds
    ActionSelector          — safety-constrained action gate
    SatelliteAgent          — full cognitive agent (world model + policy + safety)
"""

from agents.policy_network  import ActorNetwork
from agents.critic_network  import CriticNetwork, CentralizedCriticNetwork
from agents.action_selector import ActionSelector, SafetyConstraints
from agents.satellite_agent import SatelliteAgent

__all__ = [
    "ActorNetwork",
    "CriticNetwork",
    "CentralizedCriticNetwork",
    "SafetyConstraints",
    "ActionSelector",
    "SatelliteAgent",
]
