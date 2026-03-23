"""
Solar Power Generation Model — Layer 1 Physical Simulation.

Computes solar power input to satellite based on:
  P_solar = η × A_panel × G_solar × cos(θ) × (1 - eclipse_fraction)

where θ is the angle between the panel normal and the sun vector.

Panel Attitude Modes:
  - SUN_TRACKING (default): panel normal is continuously rotated to face the Sun.
    cos(θ) approaches 1.0 for ideal attitude control.
  - NADIR_POINTING        : one face points toward Earth (nadir). The solar panel
    is on the anti-nadir face. cos(θ) is computed from the geometry.
  - FIXED_NORMAL          : caller supplies a fixed panel normal unit vector.

The model also accounts for solar flux variation with orbital distance from Sun.
"""

import numpy as np


# Constants
SOLAR_CONSTANT = 1361.0   # W/m² at 1 AU (solar irradiance constant)
AU = 1.496e11              # m — 1 Astronomical Unit


# Panel attitude mode identifiers
ATTITUDE_SUN_TRACKING  = "sun_tracking"   # ideal sun-pointing (cosθ ≈ 1)
ATTITUDE_NADIR_POINTING = "nadir_pointing" # nadir face toward Earth; panel on opposite
ATTITUDE_FIXED          = "fixed"          # caller supplies explicit normal vector


def _unit(v: np.ndarray) -> np.ndarray:
    """Return unit vector; safe against zero-norm input."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


class SolarModel:
    """
    Solar power generation model for a satellite.

    Args:
        panel_area (float): Total solar panel area in m².
        efficiency (float): Solar cell efficiency [0, 1], e.g., 0.28 for 28%.
        solar_constant (float): Solar irradiance at 1 AU in W/m².
        degradation_factor (float): Long-term panel degradation multiplier [0, 1].
        attitude_mode (str): Panel orientation mode.
            'sun_tracking'   — ideal attitude control, cosθ ≈ 1.0 (default)
            'nadir_pointing' — geometric incidence angle from orbit position
            'fixed'          — requires explicit panel_normal_eci in compute_solar_power
    """

    def __init__(
        self,
        panel_area: float = 2.0,
        efficiency: float = 0.28,
        solar_constant: float = SOLAR_CONSTANT,
        degradation_factor: float = 1.0,
        attitude_mode: str = ATTITUDE_SUN_TRACKING,
    ):
        self.panel_area        = panel_area
        self.efficiency        = efficiency
        self.solar_constant    = solar_constant
        self.degradation_factor = degradation_factor
        self.attitude_mode     = attitude_mode

    def compute_solar_power(
        self,
        eclipse_fraction: float,
        sun_vector_eci: np.ndarray = None,
        panel_normal_eci: np.ndarray = None,
        satellite_pos_eci: np.ndarray = None,
        sun_distance_au: float = 1.0,
    ) -> float:
        """
        Compute instantaneous solar power input in Watts.

        Panel cos(θ) is resolved based on attitude_mode:
          'sun_tracking'   — cosθ = 1.0 (ideal sun-pointing ACS)
          'nadir_pointing' — cosθ computed from satellite position geometry
          'fixed'          — cosθ from caller-supplied panel_normal_eci

        Args:
            eclipse_fraction   : 0.0 = full sunlight, 1.0 = full umbra.
            sun_vector_eci     : Unit vector Earth→Sun in ECI. Required for
                                 'nadir_pointing' and 'fixed' modes.
            panel_normal_eci   : Panel surface normal (ECI). Used in 'fixed' mode.
            satellite_pos_eci  : Satellite ECI position (m). Used in 'nadir_pointing'.
            sun_distance_au    : Distance from Sun in AU (default 1.0).

        Returns:
            float: Solar power in Watts available for charging.
        """
        # Solar irradiance adjusted for distance from Sun
        G = self.solar_constant / (sun_distance_au ** 2)

        # Determine cos(θ) based on attitude mode
        cos_theta = self._compute_cos_theta(
            sun_vector_eci, panel_normal_eci, satellite_pos_eci
        )

        # Power formula: P = η × A × G × cos(θ) × (1 - eclipse)
        P_solar = (
            self.efficiency
            * self.panel_area
            * G
            * cos_theta
            * (1.0 - eclipse_fraction)
            * self.degradation_factor
        )

        return max(0.0, P_solar)

    def _compute_cos_theta(
        self,
        sun_vector_eci: np.ndarray,
        panel_normal_eci: np.ndarray,
        satellite_pos_eci: np.ndarray,
    ) -> float:
        """
        Compute cos(θ) — incidence angle factor between panel normal and sun.

        Strategy depends on self.attitude_mode:
          'sun_tracking'   — returns 1.0 (ideal ACS, panel always faces Sun)
          'nadir_pointing' — derives panel normal from nadir/cross-track geometry
          'fixed'          — uses caller-supplied panel_normal_eci

        Returns:
            float in [0, 1]
        """
        if self.attitude_mode == ATTITUDE_SUN_TRACKING:
            # Ideal sun-tracking attitude: panel normal always points at Sun
            # cos(θ) = 1.0 with small pointing error (ignored here)
            return 1.0

        if sun_vector_eci is None:
            # Cannot compute cos(θ) without sun vector — fall back to optimal
            return 1.0

        sun_hat = _unit(np.asarray(sun_vector_eci, dtype=float))

        if self.attitude_mode == ATTITUDE_FIXED and panel_normal_eci is not None:
            # Explicit panel normal supplied by caller
            normal_hat = _unit(np.asarray(panel_normal_eci, dtype=float))
            return float(np.clip(np.dot(normal_hat, sun_hat), 0.0, 1.0))

        if self.attitude_mode == ATTITUDE_NADIR_POINTING and satellite_pos_eci is not None:
            # Nadir-pointing attitude: -Z body axis points toward Earth center.
            # Solar panel is on the +Z face (anti-nadir / zenith face).
            # Panel normal = unit vector pointing away from Earth (zenith).
            pos_hat    = _unit(np.asarray(satellite_pos_eci, dtype=float))
            zenith_hat = pos_hat   # unit vector from Earth center to satellite
            cos_theta  = float(np.clip(np.dot(zenith_hat, sun_hat), 0.0, 1.0))
            return cos_theta

        # Fallback: assume optimal tracking
        return 1.0

    def sun_tracking_power(
        self,
        eclipse_fraction: float,
        satellite_pos_eci: np.ndarray,
        sun_vector_eci: np.ndarray,
        sun_distance_au: float = 1.0,
    ) -> float:
        """
        Compute solar power using physically correct geometric incidence angle.

        Unlike compute_solar_power() with sun_tracking mode, this method
        computes the actual cos(θ) from the satellite position and sun vector,
        assuming nadir-pointing attitude (common for CubeSats and small LEO sats).

        Args:
            eclipse_fraction  : 0.0 = full sun, 1.0 = full eclipse.
            satellite_pos_eci : Satellite ECI position vector in meters.
            sun_vector_eci    : Unit vector from Earth to Sun (ECI).
            sun_distance_au   : Sun distance in AU.

        Returns:
            float: Solar power in Watts.
        """
        return self.compute_solar_power(
            eclipse_fraction=eclipse_fraction,
            sun_vector_eci=sun_vector_eci,
            satellite_pos_eci=satellite_pos_eci,
            sun_distance_au=sun_distance_au,
            # Override attitude mode temporarily for this call:
            # We use NADIR_POINTING geometry inside _compute_cos_theta
        ) if self.attitude_mode == ATTITUDE_NADIR_POINTING else (
            # For sun_tracking mode: use the nadir geometry explicitly
            self._sun_tracking_geometric_power(
                eclipse_fraction, satellite_pos_eci, sun_vector_eci, sun_distance_au
            )
        )

    def _sun_tracking_geometric_power(
        self,
        eclipse_fraction: float,
        satellite_pos_eci: np.ndarray,
        sun_vector_eci: np.ndarray,
        sun_distance_au: float,
    ) -> float:
        """Compute nadir-geometry power regardless of attitude_mode setting."""
        G        = self.solar_constant / (sun_distance_au ** 2)
        pos_hat  = _unit(np.asarray(satellite_pos_eci, dtype=float))
        sun_hat  = _unit(np.asarray(sun_vector_eci, dtype=float))
        cos_theta = float(np.clip(np.dot(pos_hat, sun_hat), 0.0, 1.0))
        return max(0.0,
            self.efficiency * self.panel_area * G
            * cos_theta * (1.0 - eclipse_fraction) * self.degradation_factor
        )

    def compute_daily_energy(
        self,
        eclipse_fraction_profile: np.ndarray,
        dt: float,
    ) -> float:
        """
        Compute total solar energy over a time series.

        Args:
            eclipse_fraction_profile (np.ndarray): Array of eclipse fractions per step.
            dt (float): Time step in seconds.

        Returns:
            float: Total solar energy in Watt-hours.
        """
        powers = np.array([
            self.compute_solar_power(ef) for ef in eclipse_fraction_profile
        ])
        total_wh = np.sum(powers) * dt / 3600.0
        return total_wh

    def set_degradation(self, factor: float) -> None:
        """Update panel degradation factor (simulating long-term wear)."""
        self.degradation_factor = float(np.clip(factor, 0.0, 1.0))
