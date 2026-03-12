"""
Communication Protocol — Layer 3: Cooperative MARL / Layer 4: Coordination.

Models inter-satellite communication for the constellation:
    - Compressed state broadcasts between agents (Layer 3)
    - Mission command broadcasts from coordinator to agents (Layer 4)

Key design choices:
    - Communication latency is modeled as a function of inter-satellite distance
    - Messages are optionally compressed (top-k feature selection)
    - A MessageBus aggregates all broadcasts for a given timestep

Shared by both Layer 3 (peer-to-peer state sharing) and Layer 4 (top-down
mission commands), hence defined in marl/ and imported by coordination/.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger


OBS_DIM = 8      # state vector size per satellite


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class StateMessage:
    """Compressed state broadcast from one satellite to peers."""
    sender_id:  int
    payload:    np.ndarray    # compressed/full state vector
    timestamp:  float         # simulation time when sent (seconds)
    latency:    float         # computed transmission delay (seconds)


@dataclass
class CommandMessage:
    """Mission command from Layer 4 coordinator to a satellite agent."""
    target_id: int
    task:      str            # e.g. "payload_active", "hibernate", "relay_mode"
    priority:  int = 1        # 1=normal, 2=urgent, 3=emergency
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Inter-satellite link model
# ---------------------------------------------------------------------------

class ISLinkModel:
    """
    Inter-Satellite Link (ISL) delay and bandwidth model.

    Latency is approximated as:
        delay = base_delay + distance_km / speed_of_light_kmps

    where speed_of_light_kmps = 299,792 km/s.

    Args:
        base_delay_ms (float): minimum link setup delay in milliseconds. Default 5ms.
        bandwidth_bps (float): link bandwidth in bits/second. Default 10 Mbps.
    """

    SPEED_OF_LIGHT_KMPS = 299_792.0   # km/s

    def __init__(
        self,
        base_delay_ms: float = 5.0,
        bandwidth_bps: float = 10e6,
    ) -> None:
        self.base_delay_s = base_delay_ms / 1000.0
        self.bandwidth    = bandwidth_bps

    def compute_latency(self, distance_km: float) -> float:
        """
        Compute one-way propagation delay in seconds.

        Args:
            distance_km: Euclidean distance between two satellites in km.

        Returns:
            latency_s: one-way delay in seconds.
        """
        prop_delay = distance_km / self.SPEED_OF_LIGHT_KMPS
        return self.base_delay_s + prop_delay

    def transmission_time(self, payload_bytes: int) -> float:
        """Time to transmit a message of given size over the ISL."""
        bits = payload_bytes * 8
        return bits / self.bandwidth


# ---------------------------------------------------------------------------
# Communication manager
# ---------------------------------------------------------------------------

class CommunicationProtocol:
    """
    Manages inter-satellite state broadcasts and coordinator command routing.

    Used in two contexts:
        1. Layer 3: Each satellite broadcasts its compressed local state to
           all peers each timestep. Agents receive a message matrix.
        2. Layer 4: The cluster coordinator sends mission commands to agents.

    Args:
        n_agents       (int)      : number of satellites.
        obs_dim        (int)      : per-satellite observation dimension.
        compress_top_k (int|None) : if set, broadcast only the top-k state dims.
        isl_model      (ISLinkModel|None): latency model. None = zero delay.
    """

    def __init__(
        self,
        n_agents:       int,
        obs_dim:        int           = OBS_DIM,
        compress_top_k: Optional[int] = None,
        isl_model:      Optional[ISLinkModel] = None,
    ) -> None:
        self.n_agents       = n_agents
        self.obs_dim        = obs_dim
        self.compress_top_k = compress_top_k
        self.isl            = isl_model or ISLinkModel()

        # Latest received message per sender (keyed by sender_id)
        self._inbox: Dict[int, StateMessage] = {}
        # Pending command queue per agent
        self._command_queue: Dict[int, List[CommandMessage]] = {
            i: [] for i in range(n_agents)
        }

    # ------------------------------------------------------------------
    # Layer 3: peer-to-peer state sharing
    # ------------------------------------------------------------------

    def broadcast_state(
        self,
        agent_id: int,
        obs:      np.ndarray,
        sim_time: float,
        positions: Optional[np.ndarray] = None,  # (n_agents, 3) ECI km
    ) -> None:
        """
        Agent broadcasts its current state to all peers.

        Args:
            agent_id  : sender satellite index
            obs       : local observation vector, shape (obs_dim,)
            sim_time  : current simulation time (seconds)
            positions : satellite ECI positions for latency computation
        """
        payload = self._compress(obs)

        # Compute latency to all other satellites
        if positions is not None:
            pos_sender = positions[agent_id]
            # Use max latency to any peer as representative delay
            dists = [
                float(np.linalg.norm(positions[j] - pos_sender))
                for j in range(self.n_agents) if j != agent_id
            ]
            latency = self.isl.compute_latency(max(dists) if dists else 0.0)
        else:
            latency = self.isl.base_delay_s

        msg = StateMessage(
            sender_id=agent_id,
            payload=payload,
            timestamp=sim_time,
            latency=latency,
        )
        self._inbox[agent_id] = msg

    def receive_all_states(self) -> np.ndarray:
        """
        Collect all broadcast state messages into a message matrix.

        Returns:
            messages: shape (n_agents, payload_dim)
                      zero-padded for any agents that haven't broadcast yet.
        """
        payload_dim = self.compress_top_k if self.compress_top_k else self.obs_dim
        messages = np.zeros((self.n_agents, payload_dim), dtype=np.float32)

        for agent_id, msg in self._inbox.items():
            messages[agent_id] = msg.payload

        return messages

    def get_global_state(self, local_obs: np.ndarray) -> np.ndarray:
        """
        Build the global state vector by concatenating all local observations.

        Used by the centralized critic during training.

        Args:
            local_obs: shape (n_agents, obs_dim)

        Returns:
            global_state: shape (n_agents * obs_dim,)
        """
        return local_obs.flatten().astype(np.float32)

    def clear_inbox(self) -> None:
        """Clear received state messages (call at start of each step)."""
        self._inbox.clear()

    # ------------------------------------------------------------------
    # Layer 4: coordinator command routing
    # ------------------------------------------------------------------

    def send_command(self, command: CommandMessage) -> None:
        """
        Queue a mission command for delivery to the target agent.

        Args:
            command: CommandMessage with target_id and task string.
        """
        tid = command.target_id
        if 0 <= tid < self.n_agents:
            self._command_queue[tid].append(command)
            logger.debug(
                f"[COMM] Command queued: SAT-{tid} <- task='{command.task}' "
                f"priority={command.priority}"
            )

    def receive_commands(self, agent_id: int) -> List[CommandMessage]:
        """
        Drain and return all pending commands for a given agent.

        Args:
            agent_id: target satellite index.

        Returns:
            list of CommandMessage objects (may be empty).
        """
        cmds = self._command_queue[agent_id][:]
        self._command_queue[agent_id].clear()
        return cmds

    def broadcast_command_to_all(self, task: str, priority: int = 1, timestamp: float = 0.0) -> None:
        """Send the same task command to every satellite."""
        for i in range(self.n_agents):
            self.send_command(CommandMessage(target_id=i, task=task, priority=priority, timestamp=timestamp))

    # ------------------------------------------------------------------
    # Compression helper
    # ------------------------------------------------------------------

    def _compress(self, obs: np.ndarray) -> np.ndarray:
        """
        Optionally compress the state vector by selecting top-k dimensions
        ranked by absolute magnitude (a simple proxy for informativeness).

        Args:
            obs: full state vector, shape (obs_dim,)

        Returns:
            compressed payload, shape (compress_top_k,) or (obs_dim,)
        """
        if self.compress_top_k is None or self.compress_top_k >= self.obs_dim:
            return obs.copy()
        idx = np.argsort(np.abs(obs))[::-1][:self.compress_top_k]
        return obs[idx].astype(np.float32)
