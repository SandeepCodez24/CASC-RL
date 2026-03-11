"""
Battery Model — Layer 1 Physical Simulation.

Implements State of Charge (SoC) dynamics using Coulomb counting with 
charge/discharge efficiency terms and Peukert's law corrections.

Model:
    SoC(t+1) = SoC(t) + (P_solar × η_c - P_consumed / η_d) × dt / C_bat

where:
    η_c = charge efficiency (typically 0.95)
    η_d = discharge efficiency (typically 0.92)
    C_bat = battery capacity in Watt-seconds (= capacity_wh × 3600)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class BatteryState:
    """Current battery state."""
    soc: float          # State of Charge [0, 1]
    soh: float          # State of Health [0, 1]
    voltage: float      # Terminal voltage (V)
    current: float      # Net current (A), positive = charging
    power_net: float    # Net power (W), positive = charging


class BatteryModel:
    """
    Lithium-ion battery model for satellite applications.

    Uses Coulomb counting for SoC tracking with separate charge/discharge
    efficiency paths. Includes hard SoC clipping to physical limits.

    Args:
        capacity_wh (float): Battery capacity in Watt-hours.
        soc_initial (float): Initial State of Charge [0, 1].
        soh_initial (float): Initial State of Health [0, 1].
        soc_min (float): Minimum allowable SoC (deep discharge protection).
        soc_max (float): Maximum allowable SoC (overcharge protection).
        charge_efficiency (float): Charging path efficiency η_c.
        discharge_efficiency (float): Discharging path efficiency η_d.
        nominal_voltage (float): Nominal battery voltage (V).
    """

    def __init__(
        self,
        capacity_wh: float = 100.0,
        soc_initial: float = 0.8,
        soh_initial: float = 1.0,
        soc_min: float = 0.15,
        soc_max: float = 1.0,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.92,
        nominal_voltage: float = 28.0,    # V (typical spacecraft bus)
    ):
        self.capacity_wh = capacity_wh
        self.soc = float(np.clip(soc_initial, soc_min, soc_max))
        self.soh = float(np.clip(soh_initial, 0.0, 1.0))
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.eta_c = charge_efficiency
        self.eta_d = discharge_efficiency
        self.V_nom = nominal_voltage

        # Capacity in Watt-seconds (effective = SoH × nominal)
        self._capacity_ws = capacity_wh * 3600.0

        # Cycle counting for degradation model integration
        self.cycle_count = 0.0
        self._prev_soc = self.soc

    @property
    def effective_capacity_ws(self) -> float:
        """Effective capacity accounting for State of Health."""
        return self._capacity_ws * self.soh

    def step(
        self,
        p_solar_w: float,
        p_consumed_w: float,
        dt: float,
    ) -> BatteryState:
        """
        Advance battery state by one time step.

        Args:
            p_solar_w (float): Solar power input in Watts.
            p_consumed_w (float): Subsystem power consumption in Watts.
            dt (float): Time step in seconds.

        Returns:
            BatteryState: Updated battery state after this step.
        """
        # Net power (positive = charging surplus, negative = discharging)
        p_net = p_solar_w - p_consumed_w

        if p_net >= 0:
            # Charging: apply charge efficiency
            delta_energy_ws = p_net * self.eta_c * dt
        else:
            # Discharging: penalize by discharge efficiency
            delta_energy_ws = p_net / self.eta_d * dt

        # Update SoC
        capacity = self.effective_capacity_ws
        delta_soc = delta_energy_ws / capacity if capacity > 0 else 0.0
        new_soc = float(np.clip(self.soc + delta_soc, self.soc_min, self.soc_max))

        # Track half-cycles for degradation
        if new_soc < self._prev_soc:
            self.cycle_count += abs(new_soc - self._prev_soc)
        self._prev_soc = new_soc

        self.soc = new_soc

        # Approximate terminal voltage (simple open-circuit model)
        voltage = self.V_nom * (0.8 + 0.2 * self.soc)
        current = p_net / voltage if voltage > 0 else 0.0

        return BatteryState(
            soc=self.soc,
            soh=self.soh,
            voltage=voltage,
            current=current,
            power_net=p_net,
        )

    def update_soh(self, new_soh: float) -> None:
        """
        Update State of Health (called by DegradationModel).

        Args:
            new_soh (float): New SoH value [0, 1].
        """
        self.soh = float(np.clip(new_soh, 0.0, 1.0))

    def is_critical(self) -> bool:
        """Return True if SoC is at or below the minimum threshold."""
        return self.soc <= self.soc_min

    def reset(self, soc: float = None, soh: float = None) -> None:
        """Reset battery to initial state (or provided values)."""
        if soc is not None:
            self.soc = float(np.clip(soc, self.soc_min, self.soc_max))
        if soh is not None:
            self.soh = float(np.clip(soh, 0.0, 1.0))
        self.cycle_count = 0.0
        self._prev_soc = self.soc
