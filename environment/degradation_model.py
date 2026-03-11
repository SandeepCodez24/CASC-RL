"""
Battery Degradation Model — Layer 1 Physical Simulation.

Implements State of Health (SoH) decay modeling via:
  1. Wöhler-based cycle degradation: SoH -= α × ΔDoD^β
  2. Temperature-accelerated degradation (Arrhenius model)

SoH represents the fraction of original capacity remaining [0, 1].
Battery is considered end-of-life when SoH < 0.70 (70% capacity retention).
"""

import numpy as np


# Arrhenius constants for Li-ion degradation
ACTIVATION_ENERGY = 24500.0    # J/mol — activation energy
GAS_CONSTANT = 8.314           # J/(mol·K)
T_REF_K = 298.15               # K — reference temperature (25°C)


class DegradationModel:
    """
    Battery State of Health (SoH) degradation model.

    Combines:
    - Cycle-based degradation: capacity fade per charge/discharge cycle
    - Calendar aging: time-based degradation (temperature-dependent)
    - Thermal acceleration via Arrhenius law

    Args:
        soh_initial (float): Initial SoH [0, 1].
        alpha (float): Cycle degradation coefficient.
        beta (float): Depth-of-Discharge (DoD) exponent (Wöhler curve slope).
        k_calendar (float): Calendar aging rate constant (per second).
        eol_threshold (float): End-of-life SoH threshold (default 0.70).
    """

    def __init__(
        self,
        soh_initial: float = 1.0,
        alpha: float = 2e-5,       # Per unit DoD cycle degradation
        beta: float = 1.5,         # DoD exponent (Wöhler exponent)
        k_calendar: float = 1e-9,  # Calendar aging rate (per second)
        eol_threshold: float = 0.70,
    ):
        self.soh = float(np.clip(soh_initial, 0.0, 1.0))
        self.alpha = alpha
        self.beta = beta
        self.k_calendar = k_calendar
        self.eol_threshold = eol_threshold

        self._cycle_degradation_cumulative = 0.0
        self._calendar_degradation_cumulative = 0.0

    def step(
        self,
        soc_current: float,
        soc_previous: float,
        temperature_c: float,
        dt: float,
    ) -> float:
        """
        Update SoH for one simulation step.

        Args:
            soc_current (float): Current SoC value [0, 1].
            soc_previous (float): Previous SoC value [0, 1].
            temperature_c (float): Current temperature in Celsius.
            dt (float): Time step in seconds.

        Returns:
            float: Updated SoH value.
        """
        # 1. Cycle degradation — only counts discharge half-cycles
        delta_soc = soc_previous - soc_current   # positive = discharging
        if delta_soc > 0:
            dod = delta_soc                      # Depth of Discharge increment
            cycle_loss = self.alpha * (dod ** self.beta)
            self._cycle_degradation_cumulative += cycle_loss
        else:
            cycle_loss = 0.0

        # 2. Calendar aging with Arrhenius temperature acceleration
        T_k = temperature_c + 273.15
        arrhenius_factor = np.exp(
            -ACTIVATION_ENERGY / GAS_CONSTANT * (1.0 / T_k - 1.0 / T_REF_K)
        )
        calendar_loss = self.k_calendar * arrhenius_factor * dt
        self._calendar_degradation_cumulative += calendar_loss

        # 3. Total SoH update
        total_loss = cycle_loss + calendar_loss
        self.soh = float(np.clip(self.soh - total_loss, 0.0, 1.0))

        return self.soh

    def is_end_of_life(self) -> bool:
        """Return True if battery has reached end-of-life threshold."""
        return self.soh < self.eol_threshold

    @property
    def degradation_rate(self) -> float:
        """Instantaneous degradation rate proxy (total cumulative loss / time)."""
        return self._cycle_degradation_cumulative + self._calendar_degradation_cumulative

    def reset(self, soh: float = 1.0) -> None:
        """Reset degradation model."""
        self.soh = float(np.clip(soh, 0.0, 1.0))
        self._cycle_degradation_cumulative = 0.0
        self._calendar_degradation_cumulative = 0.0
