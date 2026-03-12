"""
Communication Protocol — Layer 4: Hierarchical Coordination.

This module re-exports and extends the marl.communication_protocol for use
at the coordination layer. Layer 4 adds:

    - GroundStationLink: models satellite-to-ground communication windows
    - ConstellationBroadcast: send the same command to all satellites at once
    - CommandRouter: routes priority commands from coordinator to agents

The peer-to-peer state broadcasting defined in marl.communication_protocol
is used directly for inter-satellite state sharing in Layer 3. Layer 4
focuses on the downward command channel: coordinator -> individual agents.

Import pattern:
    from coordination.communication_protocol import GroundStationLink, CommandRouter
    from marl.communication_protocol import CommunicationProtocol  # L3 peer comms
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger

# Re-export L3 protocol for convenience
from marl.communication_protocol import (
    CommunicationProtocol,
    ISLinkModel,
    StateMessage,
    CommandMessage,
)


# ---------------------------------------------------------------------------
# Ground Station Link Model
# ---------------------------------------------------------------------------

@dataclass
class GroundStation:
    """Fixed ground station for satellite downlink."""
    station_id:    int
    name:          str
    latitude_deg:  float
    longitude_deg: float
    elevation_deg: float = 5.0    # minimum elevation angle for contact (degrees)


# Default ground station network (3 stations for global coverage)
DEFAULT_GROUND_STATIONS = [
    GroundStation(0, "Svalbard",      78.2,  15.4),
    GroundStation(1, "McMurdo",      -77.8, 166.7),
    GroundStation(2, "Maspalomas",    27.8, -15.6),
]


class GroundStationLink:
    """
    Models communication windows between satellites and ground stations.

    For each (satellite, ground_station) pair, determines whether
    a comm window is open based on the satellite's orbital phase.

    Uses a simplified geometric model: a satellite is visible from a
    ground station if its orbital phase falls within the station's
    coverage arc.

    Args:
        n_satellites    (int): number of satellite agents
        stations        (list): list of GroundStation objects
        orbital_period_s(float): orbital period in seconds
    """

    def __init__(
        self,
        n_satellites:     int,
        stations:         Optional[List[GroundStation]] = None,
        orbital_period_s: float = 5520.0,
    ) -> None:
        self.n_satellites     = n_satellites
        self.stations         = stations or DEFAULT_GROUND_STATIONS
        self.orbital_period_s = orbital_period_s

    def comm_window_open(
        self,
        sat_id:    int,
        sim_time_s: float,
        phase_offset: float = None,
    ) -> bool:
        """
        Check whether satellite sat_id has a ground comm window open.

        Uses a simplified model: comm windows occur n_passes times per orbit,
        each lasting ~10 minutes. Phase offset staggers satellites naturally.

        Args:
            sat_id      : satellite index
            sim_time_s  : current simulation time
            phase_offset: override for testing

        Returns:
            True if at least one station is reachable
        """
        # Each satellite has a phase offset based on its index
        if phase_offset is None:
            phase_offset = (sat_id / self.n_satellites) * self.orbital_period_s

        # Normalized orbital position [0, 1)
        orbit_pos = ((sim_time_s + phase_offset) % self.orbital_period_s) / self.orbital_period_s

        # Simplified: assume 3 comm windows per orbit (one per ground station)
        # Each window spans ~10 min / 92 min = ~11% of orbit
        window_width = 0.11
        window_centers = [i / len(self.stations) for i in range(len(self.stations))]

        for center in window_centers:
            dist = min(abs(orbit_pos - center), 1.0 - abs(orbit_pos - center))
            if dist < window_width / 2:
                return True
        return False

    def next_window(self, sat_id: int, sim_time_s: float) -> float:
        """
        Estimate seconds until the next ground comm window for a satellite.

        Returns:
            seconds until next window (0.0 if currently in window)
        """
        if self.comm_window_open(sat_id, sim_time_s):
            return 0.0

        # Scan forward in 10-second increments
        for dt in range(10, int(self.orbital_period_s), 10):
            if self.comm_window_open(sat_id, sim_time_s + dt):
                return float(dt)
        return float(self.orbital_period_s)


# ---------------------------------------------------------------------------
# Command Router (Layer 4 downward channel)
# ---------------------------------------------------------------------------

class CommandRouter:
    """
    Routes mission commands from the ClusterCoordinator to satellite agents.

    Adds:
        - Priority ordering (urgent commands delivered first)
        - Ground-station relay filtering (commands held until window opens)
        - Broadcast to all satellites in one call

    Args:
        n_satellites (int): number of satellite agents
        comm (CommunicationProtocol): the shared L3/L4 communication bus
        ground_link (GroundStationLink | None): optional ground link model
    """

    def __init__(
        self,
        n_satellites: int,
        comm:         CommunicationProtocol,
        ground_link:  Optional[GroundStationLink] = None,
    ) -> None:
        self.n_satellites = n_satellites
        self.comm         = comm
        self.ground_link  = ground_link
        self._held_cmds:  Dict[int, List[CommandMessage]] = {i: [] for i in range(n_satellites)}

    def send(
        self,
        sat_id:    int,
        task:      str,
        priority:  int   = 1,
        timestamp: float = 0.0,
        via_isl:   bool  = True,    # True = ISL relay (no window needed)
    ) -> bool:
        """
        Route a mission command to a satellite agent.

        If via_isl=False, holds the command until a ground comm window is open.

        Args:
            sat_id    : target satellite index
            task      : task string (e.g. "hibernate", "relay_mode")
            priority  : 1=normal, 2=urgent, 3=emergency
            timestamp : simulation time of command issue
            via_isl   : if True, send immediately over ISL

        Returns:
            True if command was dispatched immediately, False if held.
        """
        cmd = CommandMessage(target_id=sat_id, task=task, priority=priority, timestamp=timestamp)

        if via_isl or self.ground_link is None:
            self.comm.send_command(cmd)
            logger.debug(f"[Router] ISL dispatch: SAT-{sat_id} <- {task}")
            return True
        else:
            # Hold until ground comm window
            window_open = self.ground_link.comm_window_open(sat_id, timestamp)
            if window_open:
                self.comm.send_command(cmd)
                logger.debug(f"[Router] Ground dispatch: SAT-{sat_id} <- {task}")
                return True
            else:
                self._held_cmds[sat_id].append(cmd)
                logger.debug(f"[Router] Command held (no window): SAT-{sat_id} <- {task}")
                return False

    def dispatch_assignment(
        self,
        assignment: Dict[int, str],
        timestamp:  float = 0.0,
        via_isl:    bool  = True,
    ) -> None:
        """
        Dispatch an entire TaskAllocator assignment to all satellites.

        Args:
            assignment: {sat_id: task_name} dict from TaskAllocator
            timestamp : current simulation time
            via_isl   : route via inter-satellite links (True) or ground (False)
        """
        for sat_id, task in assignment.items():
            self.send(sat_id, task, priority=1, timestamp=timestamp, via_isl=via_isl)

    def flush_held_commands(self, sim_time_s: float) -> int:
        """
        Check held commands and dispatch those whose windows are now open.

        Args:
            sim_time_s: current simulation time

        Returns:
            number of commands dispatched
        """
        dispatched = 0
        for sat_id, cmds in self._held_cmds.items():
            if cmds and self.ground_link and self.ground_link.comm_window_open(sat_id, sim_time_s):
                # Sort by priority (lower = higher priority) and dispatch all
                cmds.sort(key=lambda c: c.priority)
                for cmd in cmds:
                    self.comm.send_command(cmd)
                    dispatched += 1
                    logger.debug(f"[Router] Flushed held cmd: SAT-{sat_id} <- {cmd.task}")
                self._held_cmds[sat_id].clear()
        return dispatched

    def broadcast_emergency(self, task: str, timestamp: float = 0.0) -> None:
        """
        Send an emergency command to all satellites immediately (bypasses hold).

        Args:
            task      : emergency task (typically "hibernate" or "charge_priority")
            timestamp : simulation time
        """
        logger.warning(f"[Router] EMERGENCY BROADCAST: all satellites -> {task}")
        self.comm.broadcast_command_to_all(task, priority=3, timestamp=timestamp)
