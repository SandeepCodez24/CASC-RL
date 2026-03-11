"""
Solar Power Generation Model — Layer 1 Physical Simulation.

Computes solar power input to satellite based on:
  P_solar = η × A_panel × G_solar × cos(θ) × (1 - eclipse_fraction)

where θ is the angle between the panel normal and the sun vector.

The model also accounts for solar flux variation with orbital distance from Sun.
"""

import numpy as np


# Constants
SOLAR_CONSTANT = 1361.0   # W/m² at 1 AU (solar irradiance constant)
AU = 1.496e11              # m — 1 Astronomical Unit


class SolarModel:
    """
    Solar power generation model for a satellite.

    Args:
        panel_area (float): Total solar panel area in m².
        efficiency (float): Solar cell efficiency [0, 1], e.g., 0.28 for 28%.
        solar_constant (float): Solar irradiance at 1 AU in W/m².
        degradation_factor (float): Long-term panel degradation multiplier [0, 1].
    """

    def __init__(
        self,
        panel_area: float = 2.0,
        efficiency: float = 0.28,
        solar_constant: float = SOLAR_CONSTANT,
        degradation_factor: float = 1.0,
    ):
        self.panel_area = panel_area
        self.efficiency = efficiency
        self.solar_constant = solar_constant
        self.degradation_factor = degradation_factor

    def compute_solar_power(
        self,
        eclipse_fraction: float,
        sun_vector_eci: np.ndarray = None,
        panel_normal_eci: np.ndarray = None,
        sun_distance_au: float = 1.0,
    ) -> float:
        """
        Compute instantaneous solar power input in Watts.

        Args:
            eclipse_fraction (float): 0.0 = full sunlight, 1.0 = full umbra.
            sun_vector_eci (np.ndarray, optional): Unit vector from Earth to Sun.
                If None, assumes optimal panel alignment (cos θ = 1).
            panel_normal_eci (np.ndarray, optional): Panel surface normal unit vector.
                If None, assumes optimal alignment.
            sun_distance_au (float): Distance from Sun in AU (default 1.0).

        Returns:
            float: Solar power in Watts available for charging.
        """
        # Solar irradiance adjusted for distance from Sun
        G = self.solar_constant / (sun_distance_au ** 2)

        # Incidence angle factor
        if sun_vector_eci is not None and panel_normal_eci is not None:
            cos_theta = np.dot(
                panel_normal_eci / (np.linalg.norm(panel_normal_eci) + 1e-10),
                sun_vector_eci / (np.linalg.norm(sun_vector_eci) + 1e-10),
            )
            cos_theta = float(np.clip(cos_theta, 0.0, 1.0))
        else:
            cos_theta = 1.0  # Assume optimal sun-tracking panel alignment

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
