"""
Safety Monitor — Layer 5: Safety & Fault Recovery.

Implements a hierarchical state machine that enforces hard safety constraints
on satellite operation, completely overriding the RL policy when needed.

State Machine:
    NOMINAL   → all constraints satisfied, RL policy runs freely
    WARNING   → one constraint approaching threshold, RL policy constrained
    CRITICAL  → hard constraint violated, safety mode activated
    RECOVERY  → executing pre-programmed recovery sequence
    DEGRADED  → recovered but operating at reduced capacity

Transitions:
    NOMINAL   → WARNING   : SoC < soc_warn OR temp > temp_warn
    WARNING   → NOMINAL   : constraints back in safe range for hold_steps
    WARNING   → CRITICAL  : SoC < soc_critical OR temp > temp_critical
    CRITICAL  → RECOVERY  : automatic — triggers RecoveryPolicy
    RECOVERY  → NOMINAL   : recovery complete (SoC > soc_safe AND temp < temp_safe)
    RECOVERY  → DEGRADED  : after max_recovery_steps without full resolution
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from loguru import logger

from safety.anomaly_detector import AnomalyReport, AnomalyType


# State vector indices
IDX_SOC  = 0
IDX_SOH  = 1
IDX_TEMP = 2
OBS_DIM  = 8

# Action constants (must match constellation_env.py)
ACTION_PAYLOAD_ON      = 0
ACTION_PAYLOAD_OFF     = 1
ACTION_HIBERNATE       = 2
ACTION_RELAY_MODE      = 3
ACTION_CHARGE_PRIORITY = 4


class SafetyState(Enum):
    """Safety state machine states."""
    NOMINAL  = "NOMINAL"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"
    RECOVERY = "RECOVERY"
    DEGRADED = "DEGRADED"


@dataclass
class SafetyThresholds:
    """
    Configuration for all safety thresholds.

    Temperature values are in normalized space (raw_celsius / 100.0).
    SoC values are in [0, 1].

    Hierarchy per parameter:
        warn < critical (for temperature: warn < critical too — higher = worse)
        For SoC: warn > critical (lower SoC = worse)
    """
    # SoC thresholds (normalized [0,1])
    soc_warn:       float = 0.20   # WARNING trigger
    soc_critical:   float = 0.10   # CRITICAL trigger / emergency hibernate
    soc_safe:       float = 0.30   # safe level for exiting recovery

    # Temperature thresholds (normalized: raw_C / 100)
    temp_warn:      float = 0.50   # WARNING trigger (50°C)
    temp_critical:  float = 0.60   # CRITICAL trigger (60°C)
    temp_safe:      float = 0.45   # safe level for exiting recovery (45°C)

    # SoH minimum — below this the satellite enters DEGRADED permanently
    soh_minimum:    float = 0.60

    # Steps warning must persist before transitioning back to NOMINAL
    hold_steps_warn: int  = 5


@dataclass
class SafetyEvent:
    """A state machine transition or constraint violation event."""
    step:       int
    sat_id:     int
    from_state: SafetyState
    to_state:   SafetyState
    trigger:    str
    obs_soc:    float
    obs_temp:   float


class SafetyMonitor:
    """
    Satellite safety state machine and hard-constraint enforcer.

    Used by:
        - Layer 2 (ActionSelector) for lightweight per-step constraint checks
        - Layer 5 (this module) for full multi-state FSM with recovery triggers

    The monitor both enforces constraints AND drives the recovery sequence
    by returning a mandatory safe action when in CRITICAL or RECOVERY states.

    Args:
        sat_id     (int): satellite index
        thresholds (SafetyThresholds): threshold configuration
    """

    def __init__(
        self,
        sat_id:     int = 0,
        thresholds: Optional[SafetyThresholds] = None,
    ) -> None:
        self.sat_id     = sat_id
        self.th         = thresholds or SafetyThresholds()
        self.state      = SafetyState.NOMINAL
        self._step      = 0
        self._warn_consecutive = 0     # steps continuously in WARNING
        self._recovery_steps = 0       # steps spent in RECOVERY
        self.event_log: List[SafetyEvent] = []

    # ------------------------------------------------------------------
    # Main check interface
    # ------------------------------------------------------------------

    def check(
        self,
        obs:           np.ndarray,
        policy_action: int,
        anomalies:     Optional[List[AnomalyReport]] = None,
    ) -> Tuple[int, SafetyState, str]:
        """
        Evaluate constraints and return the final safe action.

        Args:
            obs            : normalized observation, shape (OBS_DIM,)
            policy_action  : RL actor's proposed action (0-4)
            anomalies      : anomaly reports from AnomalyDetector (optional)

        Returns:
            safe_action  (int)       : action to execute (may differ from policy_action)
            state        (SafetyState): current state machine state
            reason       (str)       : human-readable explanation
        """
        self._step += 1
        soc  = float(obs[IDX_SOC])
        temp = float(obs[IDX_TEMP])
        soh  = float(obs[IDX_SOH])

        # Determine next FSM state
        prev_state = self.state
        self._transition(soc, temp, soh, anomalies)

        # Compute mandatory safe action based on current state
        safe_action, reason = self._state_action(soc, temp, policy_action)

        if self.state != prev_state:
            event = SafetyEvent(
                step=self._step, sat_id=self.sat_id,
                from_state=prev_state, to_state=self.state,
                trigger=reason, obs_soc=soc, obs_temp=temp,
            )
            self.event_log.append(event)
            logger.warning(
                f"[SafetyMonitor SAT-{self.sat_id}] "
                f"{prev_state.value} -> {self.state.value} | {reason}"
            )

        return safe_action, self.state, reason

    # ------------------------------------------------------------------
    # State machine transitions
    # ------------------------------------------------------------------

    def _transition(
        self,
        soc:       float,
        temp:      float,
        soh:       float,
        anomalies: Optional[List[AnomalyReport]],
    ) -> None:
        """Update FSM state based on current observations."""
        th = self.th

        # Permanent degradation: SoH too low
        if soh < th.soh_minimum:
            self.state = SafetyState.DEGRADED
            return

        if self.state == SafetyState.NOMINAL:
            if soc < th.soc_critical or temp > th.temp_critical:
                self.state = SafetyState.CRITICAL
                self._recovery_steps = 0
            elif soc < th.soc_warn or temp > th.temp_warn:
                self.state = SafetyState.WARNING
                self._warn_consecutive = 1

        elif self.state == SafetyState.WARNING:
            if soc < th.soc_critical or temp > th.temp_critical:
                self.state = SafetyState.CRITICAL
                self._recovery_steps = 0
            elif soc >= th.soc_warn and temp <= th.temp_warn:
                self._warn_consecutive += 1
                if self._warn_consecutive >= th.hold_steps_warn:
                    self.state = SafetyState.NOMINAL
                    self._warn_consecutive = 0
            else:
                self._warn_consecutive = 0  # still in warning zone

        elif self.state == SafetyState.CRITICAL:
            # Immediately enter recovery
            self.state = SafetyState.RECOVERY
            self._recovery_steps = 0

        elif self.state == SafetyState.RECOVERY:
            self._recovery_steps += 1
            if soc >= th.soc_safe and temp <= th.temp_safe:
                self.state = SafetyState.NOMINAL
                self._recovery_steps = 0
            elif self._recovery_steps > 200:
                # Could not fully recover — settle to DEGRADED
                self.state = SafetyState.DEGRADED

    # ------------------------------------------------------------------
    # State-driven action selection
    # ------------------------------------------------------------------

    def _state_action(
        self,
        soc:           float,
        temp:          float,
        policy_action: int,
    ) -> Tuple[int, str]:
        """Return (safe_action, reason) based on current FSM state."""
        th = self.th

        if self.state == SafetyState.NOMINAL:
            # Pass policy action through — apply only lightweight checks
            if soc < th.soc_warn and policy_action == ACTION_PAYLOAD_ON:
                return ACTION_CHARGE_PRIORITY, "NOMINAL: SoC approaching warn, prefer charging"
            return policy_action, "nominal"

        elif self.state == SafetyState.WARNING:
            # Restrict energy-intensive actions
            if policy_action == ACTION_PAYLOAD_ON:
                return ACTION_RELAY_MODE, "WARNING: payload denied, relay allowed"
            if soc < th.soc_warn:
                return ACTION_CHARGE_PRIORITY, "WARNING: SoC low, charging enforced"
            return policy_action, "warning — action allowed"

        elif self.state in (SafetyState.CRITICAL, SafetyState.RECOVERY):
            # Hard override — must hibernate or charge
            if soc < th.soc_critical:
                return ACTION_CHARGE_PRIORITY, "CRITICAL/RECOVERY: emergency charge"
            if temp > th.temp_critical:
                return ACTION_HIBERNATE, "CRITICAL/RECOVERY: thermal emergency hibernate"
            return ACTION_HIBERNATE, "CRITICAL/RECOVERY: mandatory hibernate"

        elif self.state == SafetyState.DEGRADED:
            # Reduced operation — no payload, relay and charge only
            if policy_action in (ACTION_PAYLOAD_ON, ACTION_RELAY_MODE):
                return ACTION_CHARGE_PRIORITY if soc < 0.5 else ACTION_RELAY_MODE, \
                       "DEGRADED: operation reduced"
            return policy_action, "degraded — limited action"

        return policy_action, "unknown state"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset FSM to NOMINAL (call at episode start)."""
        self.state = SafetyState.NOMINAL
        self._step = 0
        self._warn_consecutive = 0
        self._recovery_steps   = 0
        self.event_log.clear()

    @property
    def is_safe(self) -> bool:
        """True if satellite is in NOMINAL or WARNING state."""
        return self.state in (SafetyState.NOMINAL, SafetyState.WARNING)

    @property
    def needs_recovery(self) -> bool:
        """True if satellite is in CRITICAL or RECOVERY state."""
        return self.state in (SafetyState.CRITICAL, SafetyState.RECOVERY)

    def state_summary(self) -> Dict[str, object]:
        """Return state dict for logging/dashboard."""
        return {
            "sat_id":           self.sat_id,
            "state":            self.state.value,
            "step":             self._step,
            "recovery_steps":   self._recovery_steps,
            "n_events":         len(self.event_log),
            "is_safe":          self.is_safe,
        }
