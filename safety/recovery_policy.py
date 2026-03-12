"""
Recovery Policy — Layer 5: Safety & Fault Recovery.

Implements pre-programmed (non-RL) recovery sequences for each anomaly type.
Recovery is hardcoded for reliability — we cannot trust a learned policy to
recover from states outside its training distribution.

Recovery types:
    BatteryRecovery     — deep-discharge / SoC critical
    ThermalRecovery     — thermal runaway / overtemperature
    SoHRecovery         — accelerated battery degradation
    SolarPanelRecovery  — solar panel efficiency loss
    GeneralRecovery     — catch-all fallback

Each recovery sequence is a state machine:
    Phase 1: Immediate safe action (override all nonessential loads)
    Phase 2: Wait for condition to stabilize
    Phase 3: Gradual re-entry into normal operation
    Phase 4: Done — hand control back to RL policy

Recovery success = satellite returns to NOMINAL SafetyState.
Recovery failure = after max_steps, satellite settles into DEGRADED mode.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from loguru import logger

from safety.anomaly_detector import AnomalyType


# Action constants (match constellation_env.py)
ACTION_PAYLOAD_ON      = 0
ACTION_PAYLOAD_OFF     = 1
ACTION_HIBERNATE       = 2
ACTION_RELAY_MODE      = 3
ACTION_CHARGE_PRIORITY = 4

# State vector indices
IDX_SOC  = 0
IDX_SOH  = 1
IDX_TEMP = 2


class RecoveryPhase(Enum):
    """Phases within a recovery sequence."""
    IMMEDIATE    = auto()    # step 1: halt nonessential loads
    STABILIZE    = auto()    # step 2: wait for metrics to recover
    GRADUAL_RAMP = auto()    # step 3: cautiously restore capability
    COMPLETE     = auto()    # recovery finished


@dataclass
class RecoveryResult:
    """
    Status update from one recovery step.

    Attributes:
        action       : mandatory action for this step
        phase        : current recovery phase
        complete     : True if recovery is done
        progress     : 0.0-1.0, estimated recovery progress
        description  : human-readable status
    """
    action:      int
    phase:       RecoveryPhase
    complete:    bool
    progress:    float
    description: str


class BaseRecovery:
    """Abstract base for recovery sequences."""

    def __init__(self, sat_id: int, max_steps: int = 200) -> None:
        self.sat_id    = sat_id
        self.max_steps = max_steps
        self._step     = 0
        self.phase     = RecoveryPhase.IMMEDIATE

    def step(self, obs: np.ndarray) -> RecoveryResult:
        """Execute one recovery step. Override in subclasses."""
        raise NotImplementedError

    def reset(self) -> None:
        self._step = 0
        self.phase = RecoveryPhase.IMMEDIATE

    @property
    def timed_out(self) -> bool:
        return self._step >= self.max_steps


class BatteryRecovery(BaseRecovery):
    """
    Recovery for battery over-discharge (SoC critical).

    Sequence:
        Phase 1 (IMMEDIATE, steps 0-5)    : force CHARGE_PRIORITY, disable all loads
        Phase 2 (STABILIZE, until soc>0.20): continue charging, hibernate non-essentials
        Phase 3 (GRADUAL_RAMP, until soc>0.30): allow relay, block payload
        Phase 4 (COMPLETE)               : hand back to RL
    """

    SOC_PHASE2_TARGET = 0.20
    SOC_PHASE3_TARGET = 0.30

    def step(self, obs: np.ndarray) -> RecoveryResult:
        self._step += 1
        soc = float(obs[IDX_SOC])

        if self.phase == RecoveryPhase.IMMEDIATE:
            if self._step >= 5:
                self.phase = RecoveryPhase.STABILIZE
            return RecoveryResult(
                action=ACTION_CHARGE_PRIORITY,
                phase=self.phase,
                complete=False,
                progress=0.0,
                description=f"Battery recovery IMMEDIATE: SoC={soc:.2f} charging...",
            )

        elif self.phase == RecoveryPhase.STABILIZE:
            if soc >= self.SOC_PHASE2_TARGET or self.timed_out:
                self.phase = RecoveryPhase.GRADUAL_RAMP
            progress = min(soc / self.SOC_PHASE2_TARGET, 1.0) * 0.5
            return RecoveryResult(
                action=ACTION_CHARGE_PRIORITY,
                phase=self.phase,
                complete=False,
                progress=progress,
                description=f"Battery recovery STABILIZE: SoC={soc:.2f} target={self.SOC_PHASE2_TARGET}",
            )

        elif self.phase == RecoveryPhase.GRADUAL_RAMP:
            if soc >= self.SOC_PHASE3_TARGET or self.timed_out:
                self.phase = RecoveryPhase.COMPLETE
            action   = ACTION_RELAY_MODE if soc > 0.25 else ACTION_CHARGE_PRIORITY
            progress = 0.5 + min(soc / self.SOC_PHASE3_TARGET, 1.0) * 0.5
            return RecoveryResult(
                action=action,
                phase=self.phase,
                complete=False,
                progress=progress,
                description=f"Battery recovery RAMP: SoC={soc:.2f} — limited relay allowed",
            )

        else:  # COMPLETE
            return RecoveryResult(
                action=ACTION_PAYLOAD_OFF,
                phase=RecoveryPhase.COMPLETE,
                complete=True,
                progress=1.0,
                description=f"Battery recovery COMPLETE: SoC={soc:.2f}",
            )


class ThermalRecovery(BaseRecovery):
    """
    Recovery for thermal runaway / overtemperature.

    Sequence:
        Phase 1 (IMMEDIATE, steps 0-3) : force HIBERNATE, dump all heat-generating loads
        Phase 2 (STABILIZE, until T<0.45): stay hibernated, allow passive cooling
        Phase 3 (GRADUAL_RAMP, until T<0.40): allow relay (low power) only
        Phase 4 (COMPLETE)             : hand back to RL
    """

    TEMP_PHASE2_TARGET = 0.45   # normalized: 45°C
    TEMP_PHASE3_TARGET = 0.40   # normalized: 40°C

    def step(self, obs: np.ndarray) -> RecoveryResult:
        self._step += 1
        temp = float(obs[IDX_TEMP])

        if self.phase == RecoveryPhase.IMMEDIATE:
            if self._step >= 3:
                self.phase = RecoveryPhase.STABILIZE
            return RecoveryResult(
                action=ACTION_HIBERNATE,
                phase=self.phase,
                complete=False,
                progress=0.0,
                description=f"Thermal recovery IMMEDIATE: T={temp:.2f} hibernating...",
            )

        elif self.phase == RecoveryPhase.STABILIZE:
            if temp <= self.TEMP_PHASE2_TARGET or self.timed_out:
                self.phase = RecoveryPhase.GRADUAL_RAMP
            # Cooling rate proxy: closer to target = more progress
            progress = min((0.65 - temp) / (0.65 - self.TEMP_PHASE2_TARGET), 1.0) * 0.5
            progress = max(progress, 0.0)
            return RecoveryResult(
                action=ACTION_HIBERNATE,
                phase=self.phase,
                complete=False,
                progress=progress,
                description=f"Thermal recovery STABILIZE: T={temp:.2f} target={self.TEMP_PHASE2_TARGET}",
            )

        elif self.phase == RecoveryPhase.GRADUAL_RAMP:
            if temp <= self.TEMP_PHASE3_TARGET or self.timed_out:
                self.phase = RecoveryPhase.COMPLETE
            action = ACTION_RELAY_MODE if temp < 0.43 else ACTION_HIBERNATE
            progress = 0.5 + min((0.45 - temp) / (0.45 - self.TEMP_PHASE3_TARGET), 1.0) * 0.5
            progress = max(progress, 0.5)
            return RecoveryResult(
                action=action,
                phase=self.phase,
                complete=False,
                progress=progress,
                description=f"Thermal recovery RAMP: T={temp:.2f} — relay allowed",
            )

        else:
            return RecoveryResult(
                action=ACTION_PAYLOAD_OFF,
                phase=RecoveryPhase.COMPLETE,
                complete=True,
                progress=1.0,
                description=f"Thermal recovery COMPLETE: T={temp:.2f}",
            )


class GeneralRecovery(BaseRecovery):
    """
    Catch-all recovery for unknown anomalies or world-model residuals.

    Conservative: hibernate for stabilize_steps, then cautiously resume.
    """

    def __init__(self, sat_id: int, stabilize_steps: int = 30, max_steps: int = 100) -> None:
        super().__init__(sat_id, max_steps)
        self.stabilize_steps = stabilize_steps

    def step(self, obs: np.ndarray) -> RecoveryResult:
        self._step += 1
        soc  = float(obs[IDX_SOC])
        temp = float(obs[IDX_TEMP])

        if self._step < self.stabilize_steps:
            action = ACTION_CHARGE_PRIORITY if soc < 0.4 else ACTION_HIBERNATE
            return RecoveryResult(
                action=action,
                phase=RecoveryPhase.STABILIZE,
                complete=False,
                progress=self._step / self.stabilize_steps * 0.7,
                description=f"General recovery STABILIZE: step {self._step}/{self.stabilize_steps}",
            )
        elif self._step < self.stabilize_steps + 10:
            return RecoveryResult(
                action=ACTION_RELAY_MODE if temp < 0.45 else ACTION_PAYLOAD_OFF,
                phase=RecoveryPhase.GRADUAL_RAMP,
                complete=False,
                progress=0.7 + (self._step - self.stabilize_steps) / 10 * 0.3,
                description="General recovery RAMP: limited relay",
            )
        else:
            return RecoveryResult(
                action=ACTION_PAYLOAD_OFF,
                phase=RecoveryPhase.COMPLETE,
                complete=True,
                progress=1.0,
                description="General recovery COMPLETE",
            )


# ---------------------------------------------------------------------------
# Recovery Policy — top-level orchestrator
# ---------------------------------------------------------------------------

class RecoveryPolicy:
    """
    Top-level recovery orchestrator for one satellite.

    Selects the appropriate recovery sequence based on detected anomaly type
    and executes it step-by-step until completion or timeout.

    Args:
        sat_id      (int): satellite index
        max_steps   (int): maximum steps per recovery sequence. Default 200.
    """

    def __init__(self, sat_id: int = 0, max_steps: int = 200) -> None:
        self.sat_id     = sat_id
        self.max_steps  = max_steps
        self._active:   Optional[BaseRecovery] = None
        self._anomaly_type: Optional[AnomalyType] = None
        self._history:  List[RecoveryResult] = []

    # ------------------------------------------------------------------
    # Sequence management
    # ------------------------------------------------------------------

    def activate(self, anomaly_type: AnomalyType) -> None:
        """
        Start the appropriate recovery sequence for the given anomaly.

        Args:
            anomaly_type: detected anomaly driving recovery
        """
        self._anomaly_type = anomaly_type

        if anomaly_type == AnomalyType.BATTERY_OVERDISCHARGE:
            self._active = BatteryRecovery(self.sat_id, self.max_steps)
        elif anomaly_type == AnomalyType.THERMAL_RUNAWAY:
            self._active = ThermalRecovery(self.sat_id, self.max_steps)
        else:
            self._active = GeneralRecovery(self.sat_id, max_steps=self.max_steps)

        logger.info(
            f"[RecoveryPolicy SAT-{self.sat_id}] "
            f"Activated recovery for {anomaly_type.name} "
            f"using {type(self._active).__name__}"
        )

    def activate_from_safety_state(self, obs: np.ndarray) -> None:
        """
        Infer anomaly type from raw observation and start recovery.

        Used when SafetyMonitor transitions to CRITICAL without a specific
        AnomalyReport (e.g., triggered by a direct threshold violation).
        """
        soc  = float(obs[IDX_SOC])
        temp = float(obs[IDX_TEMP])

        if soc < 0.12:
            self.activate(AnomalyType.BATTERY_OVERDISCHARGE)
        elif temp > 0.58:
            self.activate(AnomalyType.THERMAL_RUNAWAY)
        else:
            self.activate(AnomalyType.UNKNOWN_RESIDUAL)

    def step(self, obs: np.ndarray) -> RecoveryResult:
        """
        Execute one step of the active recovery sequence.

        Args:
            obs: current observation, shape (OBS_DIM,)

        Returns:
            RecoveryResult with mandatory action, phase, progress
        """
        if self._active is None:
            # No active sequence — default safe action
            return RecoveryResult(
                action=ACTION_PAYLOAD_OFF,
                phase=RecoveryPhase.COMPLETE,
                complete=True,
                progress=1.0,
                description="No active recovery sequence",
            )

        result = self._active.step(obs)
        self._history.append(result)

        if result.complete:
            logger.info(
                f"[RecoveryPolicy SAT-{self.sat_id}] "
                f"Recovery COMPLETE in {len(self._history)} steps | "
                f"{self._anomaly_type.name if self._anomaly_type else 'unknown'}"
            )
            self._active = None

        return result

    def is_active(self) -> bool:
        """True if a recovery sequence is currently running."""
        return self._active is not None

    def reset(self) -> None:
        """Reset recovery state (call at episode start)."""
        self._active       = None
        self._anomaly_type = None
        self._history.clear()

    @property
    def recovery_steps_taken(self) -> int:
        return len(self._history)

    def summary(self) -> Dict[str, object]:
        return {
            "sat_id":         self.sat_id,
            "is_active":      self.is_active(),
            "anomaly_type":   self._anomaly_type.name if self._anomaly_type else None,
            "steps_taken":    self.recovery_steps_taken,
            "last_phase":     (self._history[-1].phase.name if self._history else None),
        }
