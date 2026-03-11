"""
Thermal Model — Layer 1 Physical Simulation.

Implements a nodal heat balance model:
    m · c · dT/dt = Q_solar + Q_internal - Q_radiated

Where:
    Q_solar    = absorbed solar heat (sunlit phase only)
    Q_internal = waste heat from electronics
    Q_radiated = blackbody radiation from satellite surfaces

The model tracks spacecraft temperature across eclipse and sunlit orbital phases,
providing thermal risk assessment for the safety monitor.
"""

import numpy as np


# Physical constants
STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
SOLAR_CONSTANT = 1361.0             # W/m²


class ThermalModel:
    """
    Lumped-node thermal model for a satellite.

    Assumes a single thermal node representing the satellite bus.
    Models solar heating, internal dissipation, and radiative cooling.

    Args:
        mass_kg (float): Satellite mass in kg.
        specific_heat (float): Specific heat capacity J/(kg·K).
        area_solar_absorb (float): Sun-facing area for absorption (m²).
        area_radiator (float): Radiating surface area (m²).
        absorptivity (float): Solar absorptivity of outer surface [0,1].
        emissivity (float): Thermal emissivity of radiating surface [0,1].
        T_initial_c (float): Initial temperature in Celsius.
        T_min_c (float): Minimum safe temperature (°C).
        T_max_c (float): Maximum safe temperature (°C).
    """

    def __init__(
        self,
        mass_kg: float = 50.0,
        specific_heat: float = 900.0,     # J/(kg·K) — aluminum equivalent
        area_solar_absorb: float = 1.0,   # m²
        area_radiator: float = 0.5,       # m²
        absorptivity: float = 0.3,        # typical spacecraft outer coating
        emissivity: float = 0.85,
        T_initial_c: float = 20.0,
        T_min_c: float = -20.0,
        T_max_c: float = 60.0,
    ):
        self.mass = mass_kg
        self.cp = specific_heat
        self.A_solar = area_solar_absorb
        self.A_rad = area_radiator
        self.alpha = absorptivity
        self.eps = emissivity
        self.T_min_c = T_min_c
        self.T_max_c = T_max_c

        # Internal state (Kelvin)
        self.T_k = T_initial_c + 273.15

    @property
    def temperature_c(self) -> float:
        """Current temperature in Celsius."""
        return self.T_k - 273.15

    def step(
        self,
        eclipse_fraction: float,
        p_dissipated_w: float,
        dt: float,
        solar_irradiance: float = SOLAR_CONSTANT,
    ) -> float:
        """
        Advance thermal model by one time step.

        Args:
            eclipse_fraction (float): [0.0 = full sun, 1.0 = full eclipse].
            p_dissipated_w (float): Internal power dissipation (waste heat in W).
            dt (float): Time step in seconds.
            solar_irradiance (float): Incident solar flux in W/m².

        Returns:
            float: Updated temperature in Celsius.
        """
        # Solar heat input (zero during eclipse)
        Q_solar = (
            self.alpha
            * self.A_solar
            * solar_irradiance
            * (1.0 - eclipse_fraction)
        )

        # Internal heat from electronics
        Q_internal = p_dissipated_w

        # Radiative heat loss (Stefan-Boltzmann law)
        Q_radiated = self.eps * STEFAN_BOLTZMANN * self.A_rad * (self.T_k ** 4)

        # Heat balance: m·c·dT/dt = Q_in - Q_out
        dQ_dt = Q_solar + Q_internal - Q_radiated
        dT_dt = dQ_dt / (self.mass * self.cp)

        # Euler integration
        self.T_k += dT_dt * dt
        # Physical floor at 2.7K (space background radiation limit)
        self.T_k = max(self.T_k, 2.7)

        return self.temperature_c

    def is_overheating(self) -> bool:
        """Return True if temperature exceeds T_max safety limit."""
        return self.temperature_c > self.T_max_c

    def is_too_cold(self) -> bool:
        """Return True if temperature is below T_min safety limit."""
        return self.temperature_c < self.T_min_c

    def thermal_risk(self) -> float:
        """
        Compute normalized thermal risk score [0, 1].

        Returns:
            float: 0.0 = nominal, 1.0 = at maximum safe temperature.
        """
        T = self.temperature_c
        if T <= self.T_min_c:
            return 0.5 * (self.T_min_c - T) / max(1.0, abs(self.T_min_c))
        elif T >= self.T_max_c:
            return 1.0
        else:
            # Normalized distance to T_max
            span = self.T_max_c - self.T_min_c
            return max(0.0, (T - self.T_min_c) / span)

    def reset(self, T_initial_c: float = 20.0) -> None:
        """Reset thermal model to initial temperature."""
        self.T_k = T_initial_c + 273.15
