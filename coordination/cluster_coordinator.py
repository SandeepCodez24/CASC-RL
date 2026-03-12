"""
Cluster Coordinator — Layer 4: Hierarchical Coordination.

The ClusterCoordinator is the constellation's strategic brain. It:
    1. Collects k-step state forecasts from all SatelliteAgent world models
    2. Aggregates them into a global forecast with resource summaries
    3. Assesses mission viability given predicted battery, thermal, and comm state
    4. Delegates constrained task allocation to TaskAllocator
    5. Broadcasts mission commands back to agents via CommunicationProtocol

Designed to handle constellations up to 12 satellites (per project update).

Algorithm 5 from the project document:
    forecasts    = [agent.world_model.predict(horizon=10) for agent in agents]
    global_fcst  = coordinator.aggregate(forecasts)
    assignment   = coordinator.solve_allocation(global_fcst, power_budget, comm)
    [agent.receive_command(task) for sat_id, task in assignment.items()]
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger


# State vector indices (must match constellation_env.py OBS_DIM=8)
IDX_SOC   = 0
IDX_SOH   = 1
IDX_TEMP  = 2
IDX_PSOL  = 3
IDX_PHASE = 4
IDX_ECL   = 5
IDX_PCON  = 6
IDX_COMM  = 7

# Task names (must match satellite_agent.py receive_command vocabulary)
TASKS = ["payload_active", "relay_mode", "hibernate", "charge_priority", "payload_off"]

# Power draw per task (Watts, unnormalized — mirrors POWER_BY_MODE in constellation_env)
TASK_POWER_W = {
    "payload_active":  40.0,
    "relay_mode":      20.0,
    "hibernate":        5.0,
    "charge_priority":  3.0,
    "payload_off":     10.0,
}

# Mission value score per task (for optimization objective)
TASK_VALUE = {
    "payload_active":  1.0,
    "relay_mode":      0.6,
    "hibernate":       0.0,
    "charge_priority": 0.0,
    "payload_off":     0.1,
}


@dataclass
class SatelliteForecast:
    """Aggregated k-step world model forecast for one satellite."""
    sat_id:      int
    horizon:     int                          # number of predicted steps
    states:      np.ndarray                   # shape (horizon, OBS_DIM)

    # Derived summaries (computed from states)
    mean_soc:    float = 0.0
    min_soc:     float = 0.0
    mean_temp:   float = 0.0
    max_temp:    float = 0.0
    eclipse_frac: float = 0.0
    viable:      bool  = True                 # can this sat take a mission?
    viable_tasks: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.states.ndim == 2 and len(self.states) > 0:
            self.mean_soc     = float(self.states[:, IDX_SOC].mean())
            self.min_soc      = float(self.states[:, IDX_SOC].min())
            self.mean_temp    = float(self.states[:, IDX_TEMP].mean())
            self.max_temp     = float(self.states[:, IDX_TEMP].max())
            self.eclipse_frac = float(self.states[:, IDX_ECL].mean())
            self._assess_viability()

    def _assess_viability(self) -> None:
        """Determine which tasks this satellite can safely perform."""
        self.viable_tasks = []
        # Always possible if not critically low
        if self.min_soc > 0.15:
            self.viable_tasks.append("relay_mode")
            self.viable_tasks.append("payload_off")
        if self.min_soc > 0.30 and self.max_temp < 0.55:
            self.viable_tasks.append("payload_active")
        if self.min_soc <= 0.20:
            self.viable_tasks.append("charge_priority")
        self.viable_tasks.append("hibernate")   # always possible
        self.viable = len(self.viable_tasks) > 1


@dataclass
class GlobalForecast:
    """Aggregated constellation-level forecast from all satellites."""
    n_satellites:    int
    horizon:         int
    forecasts:       List[SatelliteForecast]
    total_solar_w:   float = 0.0    # estimated total solar power (not normalized)
    mean_fleet_soc:  float = 0.0
    n_viable:        int   = 0
    n_eclipsed:      int   = 0

    def __post_init__(self):
        self.mean_fleet_soc = float(np.mean([f.mean_soc for f in self.forecasts]))
        self.n_viable       = sum(1 for f in self.forecasts if f.viable)
        self.n_eclipsed     = sum(1 for f in self.forecasts if f.eclipse_frac > 0.5)


class ClusterCoordinator:
    """
    Constellation-level coordinator (Layer 4).

    Aggregates world model forecasts from all satellite agents, assesses
    resource availability, and delegates to TaskAllocator for the final
    mission assignment. Broadcasts commands via CommunicationProtocol.

    Args:
        n_satellites   (int): number of satellites (supports up to 12)
        forecast_horizon (int): world model rollout horizon. Default 10.
        power_budget_w (float): total constellation power budget in Watts.
        min_relay_sats (int): minimum satellites that must be in relay mode.
    """

    def __init__(
        self,
        n_satellites:     int,
        forecast_horizon: int   = 10,
        power_budget_w:   float = 200.0,
        min_relay_sats:   int   = 1,
    ) -> None:
        self.n_satellites     = n_satellites
        self.forecast_horizon = forecast_horizon
        self.power_budget_w   = power_budget_w
        self.min_relay_sats   = min_relay_sats

        self._last_assignment: Dict[int, str] = {}
        self._coordination_step = 0

    # ------------------------------------------------------------------
    # Step 1-2: Collect and aggregate forecasts
    # ------------------------------------------------------------------

    def aggregate(
        self,
        raw_forecasts: List[List[np.ndarray]],   # list of k state arrays per agent
    ) -> GlobalForecast:
        """
        Convert raw per-agent k-step state lists into a GlobalForecast.

        Args:
            raw_forecasts: list of length n_satellites. Each element is a list
                           of k np.ndarray(OBS_DIM,) predicted states.

        Returns:
            GlobalForecast with per-satellite SatelliteForecast objects.
        """
        sat_forecasts = []
        for sat_id, states_list in enumerate(raw_forecasts):
            if len(states_list) == 0:
                states_arr = np.zeros((1, 8), dtype=np.float32)
            else:
                states_arr = np.stack(states_list, axis=0).astype(np.float32)
            sf = SatelliteForecast(
                sat_id=sat_id,
                horizon=len(states_list),
                states=states_arr,
            )
            sat_forecasts.append(sf)

        gf = GlobalForecast(
            n_satellites=self.n_satellites,
            horizon=self.forecast_horizon,
            forecasts=sat_forecasts,
        )
        logger.debug(
            f"[Coord] Aggregated forecasts: mean_fleet_soc={gf.mean_fleet_soc:.2f} | "
            f"viable={gf.n_viable}/{self.n_satellites} | eclipsed={gf.n_eclipsed}"
        )
        return gf

    # ------------------------------------------------------------------
    # Step 3: Solve constrained allocation (delegates to TaskAllocator)
    # ------------------------------------------------------------------

    def coordinate(
        self,
        agents,                                  # list of SatelliteAgent
        comm,                                    # CommunicationProtocol
        world_model = None,                      # optional shared WorldModel
        current_obs: Optional[np.ndarray] = None,  # (n_sat, OBS_DIM)
    ) -> Dict[int, str]:
        """
        Full coordination cycle implementing Algorithm 5.

        Args:
            agents     : list of SatelliteAgent (one per satellite)
            comm       : CommunicationProtocol for broadcasting commands
            world_model: shared WorldModel (or None to use per-agent models)
            current_obs: current observation matrix from env (n_sat, OBS_DIM)

        Returns:
            assignment dict: {sat_id -> task_string}
        """
        self._coordination_step += 1

        # Step 1: Collect world model forecasts from every agent
        raw_forecasts = []
        for i, agent in enumerate(agents):
            obs_i = current_obs[i] if current_obs is not None else np.zeros(8, dtype=np.float32)
            try:
                wm = world_model if world_model is not None else agent.world_model
                states = wm.predict_k_steps(
                    s_t=obs_i,
                    actions=[1] * self.forecast_horizon,
                    k=self.forecast_horizon,
                )
            except Exception as e:
                logger.warning(f"[Coord] World model forecast failed for SAT-{i}: {e}")
                states = [obs_i] * self.forecast_horizon
            raw_forecasts.append(states)

        # Step 2: Aggregate into global forecast
        global_forecast = self.aggregate(raw_forecasts)

        # Step 3: Solve constrained allocation
        from coordination.task_allocator import TaskAllocator
        allocator = TaskAllocator(
            n_satellites=self.n_satellites,
            power_budget_w=self.power_budget_w,
            min_relay_sats=self.min_relay_sats,
        )
        assignment = allocator.solve(global_forecast)

        # Step 4: Broadcast commands
        for sat_id, task in assignment.items():
            comm.send_command(
                __import__("marl.communication_protocol", fromlist=["CommandMessage"]).CommandMessage(
                    target_id=sat_id,
                    task=task,
                    priority=1,
                    timestamp=float(self._coordination_step),
                )
            )
            agents[sat_id].receive_command(task)

        self._last_assignment = assignment
        logger.info(
            f"[Coord step {self._coordination_step}] Assignment: "
            + " | ".join(f"SAT-{k}={v}" for k, v in sorted(assignment.items()))
        )
        return assignment

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def assess_fleet_status(
        self, global_forecast: GlobalForecast
    ) -> Dict[str, float]:
        """Return a fleet health summary dict for logging/dashboard."""
        return {
            "mean_fleet_soc":    global_forecast.mean_fleet_soc,
            "n_viable":          float(global_forecast.n_viable),
            "n_eclipsed":        float(global_forecast.n_eclipsed),
            "viable_fraction":   global_forecast.n_viable / max(self.n_satellites, 1),
        }

    @property
    def last_assignment(self) -> Dict[int, str]:
        """Most recent task assignment dictionary."""
        return dict(self._last_assignment)
