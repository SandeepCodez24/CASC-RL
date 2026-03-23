"""
Constellation Environment — Layer 1 Physical Simulation.

Top-level OpenAI Gymnasium-compatible environment that integrates all
physical simulation modules (orbital, eclipse, solar, battery, thermal, 
degradation) for a multi-satellite constellation.

Observation Space (per satellite):
    [SoC, SoH, temperature, solar_input, orbital_phase,
     eclipse_flag, power_consumption, comm_delay]

Action Space (per satellite, discrete):
    0: payload_ON
    1: payload_OFF
    2: hibernate
    3: relay_mode
    4: charge_priority
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from environment.orbital_dynamics import OrbitalDynamics
from environment.eclipse_model import EclipseModel
from environment.solar_model import SolarModel
from environment.battery_model import BatteryModel
from environment.degradation_model import DegradationModel
from environment.thermal_model import ThermalModel


# Action definitions
ACTION_PAYLOAD_ON      = 0
ACTION_PAYLOAD_OFF     = 1
ACTION_HIBERNATE       = 2
ACTION_RELAY_MODE      = 3
ACTION_CHARGE_PRIORITY = 4
N_ACTIONS = 5

# Power consumption per mode (Watts)
POWER_BY_MODE = {
    ACTION_PAYLOAD_ON:      40.0,
    ACTION_PAYLOAD_OFF:     10.0,
    ACTION_HIBERNATE:        5.0,
    ACTION_RELAY_MODE:      20.0,
    ACTION_CHARGE_PRIORITY:  3.0,
}

# State dimension
OBS_DIM = 8   # SoC, SoH, T, P_solar, phase, eclipse, P_consumed, comm_delay

# Speed of light — for ISL propagation delay computation
SPEED_OF_LIGHT = 2.998e8   # m/s
# Maximum expected inter-satellite range (~2 × orbit altitude for LEO at 400 km)
MAX_ISL_RANGE_M = 8000e3   # 8000 km — for normalization


class SatelliteState:
    """Container for a single satellite's physical state."""

    def __init__(self, sat_id: int, orbital_params: dict, env_cfg: dict):
        self.sat_id = sat_id

        # Initialize all physical sub-models
        self.orbit = OrbitalDynamics(
            semi_major_axis=orbital_params.get("semi_major_axis", 6778.0),
            eccentricity=orbital_params.get("eccentricity", 0.0),
            inclination=orbital_params.get("inclination", 51.6),
            raan=orbital_params.get("raan", 0.0),
            arg_perigee=orbital_params.get("arg_perigee", 0.0),
            true_anomaly_0=orbital_params.get("true_anomaly_0", sat_id * 120.0),
        )
        self.eclipse = EclipseModel(model="conical")
        self.solar = SolarModel(
            panel_area=env_cfg.get("panel_area", 2.0),
            efficiency=env_cfg.get("solar_efficiency", 0.28),
        )
        self.battery = BatteryModel(
            capacity_wh=env_cfg.get("battery_capacity_wh", 100.0),
            soc_initial=env_cfg.get("soc_initial", 0.8),
            soh_initial=env_cfg.get("soh_initial", 1.0),
            soc_min=env_cfg.get("soc_min", 0.15),
            charge_efficiency=env_cfg.get("charge_efficiency", 0.95),
            discharge_efficiency=env_cfg.get("discharge_efficiency", 0.92),
        )
        self.degradation = DegradationModel(soh_initial=env_cfg.get("soh_initial", 1.0))
        self.thermal = ThermalModel(
            T_initial_c=env_cfg.get("T_initial", 20.0),
            T_max_c=env_cfg.get("T_max", 60.0),
            T_min_c=env_cfg.get("T_min", -20.0),
        )

        # Current mode
        self.mode = ACTION_PAYLOAD_OFF
        self.p_consumed = POWER_BY_MODE[self.mode]

        # Injected anomalies
        self._anomalies: Dict[str, Any] = {}

    def get_obs(self, t: float, sun_vector=None, all_positions=None) -> np.ndarray:
        """
        Build observation vector for this satellite.

        Args:
            t            : elapsed simulation time in seconds
            sun_vector   : ECI unit vector from Earth to Sun (optional)
            all_positions: list of ECI position arrays for all satellites in
                           the constellation (used for comm_delay computation);
                           if None, a placeholder delay is used.

        Returns:
            np.ndarray: shape (OBS_DIM,)
        """
        pos, _ = self.orbit.propagate(t)
        phase = self.orbit.get_orbital_phase(t)

        # Eclipse check
        if sun_vector is None:
            sun_vector = np.array([1.0, 0.0, 0.0])
        eclipse_state = self.eclipse.check_eclipse(pos, sun_vector)
        ef = eclipse_state.eclipse_fraction

        # Solar power — pass satellite ECI position for cos(θ) computation
        p_solar = self.solar.compute_solar_power(
            eclipse_fraction=ef,
            sun_vector_eci=sun_vector,
            satellite_pos_eci=pos,
        )

        # Apply anomalies
        if "solar_degradation" in self._anomalies:
            p_solar *= self._anomalies["solar_degradation"]

        # Battery & thermal (read-only at obs time)
        soc = self.battery.soc
        soh = self.battery.soh
        T = self.thermal.temperature_c
        p_consumed = self.p_consumed

        # ----------------------------------------------------------------
        # Comm delay: physics-based ISL propagation delay.
        # Computed as mean propagation delay to all other satellites:
        #   delay_i = mean_j( ||pos_i - pos_j|| / c )
        # Normalized to [0, 1] using MAX_ISL_RANGE_M as reference.
        # Falls back to positional placeholder if positions unavailable.
        # ----------------------------------------------------------------
        comm_delay = self._compute_comm_delay(pos, all_positions)

        obs = np.array([
            soc,
            soh,
            T / 100.0,                      # Normalize: ÷100°C
            p_solar / 200.0,                 # Normalize: ÷200W
            phase / (2 * np.pi),             # [0,1]
            float(ef > 0.5),                 # Binary eclipse flag
            p_consumed / 80.0,               # Normalize: ÷80W
            comm_delay,                      # [0, 1] normalized ISL delay
        ], dtype=np.float32)

        return obs

    def _compute_comm_delay(
        self,
        own_pos: np.ndarray,
        all_positions,
    ) -> float:
        """
        Compute normalized mean ISL propagation delay to all other satellites.

        Delay model: propagation delay = distance / speed_of_light
        Normalized to [0, 1] using MAX_ISL_RANGE_M as reference scale.

        Args:
            own_pos      : ECI position of this satellite (m), shape (3,)
            all_positions: list of ECI position arrays for all sats,
                           or None if not yet available.

        Returns:
            float: normalized comm delay [0, 1]
        """
        if all_positions is None or len(all_positions) <= 1:
            # Fallback: use sat_id-based placeholder (proportional to spacing)
            return float(np.clip(0.05 * (self.sat_id + 1), 0.0, 1.0))

        delays = []
        for other_pos in all_positions:
            if other_pos is None:
                continue
            dist_m = float(np.linalg.norm(own_pos - np.asarray(other_pos)))
            if dist_m < 1.0:
                continue  # skip self
            # Propagation delay (seconds) = range / c
            delay_s = dist_m / SPEED_OF_LIGHT
            # Normalize: 1.0 = delay at MAX_ISL_RANGE_M
            delay_norm = delay_s / (MAX_ISL_RANGE_M / SPEED_OF_LIGHT)
            delays.append(float(np.clip(delay_norm, 0.0, 1.0)))

        if not delays:
            return 0.0
        return float(np.mean(delays))


class ConstellationEnv(gym.Env):
    """
    Multi-satellite constellation Gymnasium environment.

    Supports vectorized multi-agent observations and discrete per-satellite actions.

    Args:
        n_satellites (int): Number of satellites in the constellation.
        dt (float): Simulation time step in seconds.
        episode_length (int): Maximum steps per episode.
        env_cfg (dict): Environment configuration overrides.
        enable_eclipse (bool): Enable/disable eclipse simulation.
        enable_degradation (bool): Enable/disable SoH degradation.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_satellites: int = 3,
        dt: float = 10.0,
        episode_length: int = 1000,
        env_cfg: Optional[dict] = None,
        enable_eclipse: bool = True,
        enable_degradation: bool = True,
    ):
        super().__init__()

        self.n_satellites = n_satellites
        self.dt = dt
        self.episode_length = episode_length
        self.enable_eclipse = enable_eclipse
        self.enable_degradation = enable_degradation
        self.env_cfg = env_cfg or {}

        # Build satellite states
        self.satellites: List[SatelliteState] = []
        for i in range(n_satellites):
            orbital_params = {
                "true_anomaly_0": i * (360.0 / n_satellites),
                **self.env_cfg.get("orbital", {}),
            }
            self.satellites.append(
                SatelliteState(i, orbital_params, self.env_cfg)
            )

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(n_satellites, OBS_DIM),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([N_ACTIONS] * n_satellites)

        # Time tracking
        self.t = 0.0
        self.step_count = 0
        self._sun_vector = np.array([1.0, 0.0, 0.0])  # simplified static sun

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.t = 0.0
        self.step_count = 0

        for sat in self.satellites:
            sat.battery.reset(
                soc=self.env_cfg.get("soc_initial", 0.8),
                soh=self.env_cfg.get("soh_initial", 1.0),
            )
            sat.thermal.reset(T_initial_c=self.env_cfg.get("T_initial", 20.0))
            sat.mode = ACTION_PAYLOAD_OFF
            sat.p_consumed = POWER_BY_MODE[sat.mode]
            sat._anomalies.clear()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Advance the constellation by one time step.

        Args:
            actions (np.ndarray): Integer action per satellite, shape (n_satellites,).

        Returns:
            obs, reward, terminated, truncated, info
        """
        assert len(actions) == self.n_satellites

        rewards = []
        for i, (sat, action) in enumerate(zip(self.satellites, actions)):
            # Apply action
            sat.mode = int(action)
            sat.p_consumed = POWER_BY_MODE[sat.mode]

            # Compute solar power — pass satellite ECI position for cos(θ)
            pos, _ = sat.orbit.propagate(self.t)
            eclipse_state = sat.eclipse.check_eclipse(pos, self._sun_vector)
            ef = eclipse_state.eclipse_fraction if self.enable_eclipse else 0.0
            p_solar = sat.solar.compute_solar_power(
                eclipse_fraction=ef,
                sun_vector_eci=self._sun_vector,
                satellite_pos_eci=pos,
            )

            # Apply anomalies
            if "solar_degradation" in sat._anomalies:
                p_solar *= sat._anomalies["solar_degradation"]

            soc_prev = sat.battery.soc
            bat_state = sat.battery.step(p_solar, sat.p_consumed, self.dt)

            # SoH degradation
            if self.enable_degradation:
                new_soh = sat.degradation.step(
                    bat_state.soc, soc_prev, sat.thermal.temperature_c, self.dt
                )
                sat.battery.update_soh(new_soh)

            # Thermal update
            sat.thermal.step(ef, sat.p_consumed * 0.3, self.dt)  # 30% waste heat

            # Per-satellite reward
            r = self._compute_reward(sat, bat_state, action)
            rewards.append(r)

        # Advance time
        self.t += self.dt
        self.step_count += 1

        obs = self._get_obs()
        reward = float(np.mean(rewards))
        terminated = any(s.battery.is_critical() for s in self.satellites)
        truncated = self.step_count >= self.episode_length
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Stacked observation array for all satellites."""
        # Collect all current ECI positions for comm_delay computation
        all_positions = []
        for sat in self.satellites:
            pos, _ = sat.orbit.propagate(self.t)
            all_positions.append(pos)

        return np.stack(
            [
                sat.get_obs(self.t, self._sun_vector, all_positions)
                for sat in self.satellites
            ],
            axis=0,
        ).astype(np.float32)

    def _compute_reward(self, sat: SatelliteState, bat_state, action: int) -> float:
        """Compute per-satellite reward."""
        w1, w2, w3, w4 = 1.0, 0.5, 0.3, 0.8

        r_soc = w1 * bat_state.soc
        r_degrade = -w2 * (1.0 - bat_state.soh)
        r_thermal = -w3 * sat.thermal.thermal_risk()
        r_mission = w4 * (1.0 if action == ACTION_PAYLOAD_ON else 0.0)

        # Heavy penalty for critical state
        if bat_state.soc <= sat.battery.soc_min:
            r_soc -= 5.0

        return r_soc + r_degrade + r_thermal + r_mission

    def _get_info(self) -> dict:
        """Diagnostic info dictionary."""
        return {
            "t": self.t,
            "step": self.step_count,
            "soc": [s.battery.soc for s in self.satellites],
            "soh": [s.battery.soh for s in self.satellites],
            "temperature": [s.thermal.temperature_c for s in self.satellites],
        }

    def inject_anomaly(self, anomaly_type: str, sat_id: int = 0, **kwargs) -> None:
        """
        Inject a hardware anomaly into a satellite.

        Args:
            anomaly_type (str): "extended_eclipse" | "solar_degradation" | "battery_failure"
            sat_id (int): Target satellite index.
        """
        sat = self.satellites[sat_id]
        if anomaly_type == "extended_eclipse":
            logger.warning(f"[SAT-{sat_id}] Injecting extended eclipse anomaly")
            # Handled externally via eclipse fraction override
        elif anomaly_type == "solar_degradation":
            factor = kwargs.get("efficiency_factor", 0.4)
            sat._anomalies["solar_degradation"] = factor
            logger.warning(f"[SAT-{sat_id}] Solar degradation: {factor:.0%} efficiency")
        elif anomaly_type == "battery_failure":
            cell_loss = kwargs.get("cell_loss", 0.3)
            new_soh = sat.battery.soh * (1.0 - cell_loss)
            sat.battery.update_soh(new_soh)
            logger.warning(f"[SAT-{sat_id}] Battery cell failure: SoH → {new_soh:.2f}")

    def render(self, mode: str = "human") -> None:
        """Basic text render."""
        print(f"\n--- Step {self.step_count} | t={self.t:.0f}s ---")
        for i, sat in enumerate(self.satellites):
            print(
                f"  SAT-{i}: SoC={sat.battery.soc:.2f} | "
                f"SoH={sat.battery.soh:.2f} | "
                f"T={sat.thermal.temperature_c:.1f}°C | "
                f"Mode={sat.mode}"
            )
