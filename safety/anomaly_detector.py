"""
Anomaly Detector — Layer 5: Safety & Fault Recovery.

Detects abnormal satellite states using two complementary methods:

1. Statistical Detection (Z-score)
   - Computes running mean/std for each state dimension over a sliding window
   - Flags a reading as anomalous when |z| > z_threshold
   - Used for: battery over-discharge, thermal runaway, sudden SoH drop

2. World-Model Residual Detection
   - Compares the world model's predicted next state with the actual observed state
   - Large prediction error (residual) indicates the physics are behaving unexpectedly
   - Useful for detecting hardware failures that alter the satellite's dynamics

Detected anomaly types:
    BATTERY_OVERDISCHARGE  — SoC drops faster than expected
    THERMAL_RUNAWAY        — Temperature rising faster than expected
    SOH_RAPID_DEGRADATION  — SoH falling faster than expected
    SOLAR_PANEL_FAILURE    — Solar power much lower than world model predicted
    COMMUNICATION_FAULT    — Comm delay out of expected range
    UNKNOWN                — Large residual without specific pattern match
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
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

OBS_DIM = 8


class AnomalyType(Enum):
    NONE                   = auto()
    BATTERY_OVERDISCHARGE  = auto()
    THERMAL_RUNAWAY        = auto()
    SOH_RAPID_DEGRADATION  = auto()
    SOLAR_PANEL_FAILURE    = auto()
    COMMUNICATION_FAULT    = auto()
    UNKNOWN_RESIDUAL       = auto()


@dataclass
class AnomalyReport:
    """
    Record of a detected anomaly event.

    Attributes:
        sat_id       : which satellite triggered the anomaly
        step         : simulation step at detection time
        anomaly_type : AnomalyType enum value
        severity     : 0.0-1.0, scaled magnitude of the anomaly
        z_scores     : per-dimension z-scores at detection time
        residuals    : per-dimension world-model residuals (if available)
        description  : human-readable summary
    """
    sat_id:       int
    step:         int
    anomaly_type: AnomalyType
    severity:     float
    z_scores:     np.ndarray
    residuals:    Optional[np.ndarray] = None
    description:  str = ""

    def __str__(self) -> str:
        return (
            f"[SAT-{self.sat_id} | step {self.step}] "
            f"{self.anomaly_type.name} | severity={self.severity:.2f} | {self.description}"
        )


class StatisticalDetector:
    """
    Sliding-window Z-score anomaly detector for one satellite.

    Maintains a rolling window of past observations and computes
    per-dimension z-scores. When any z-score exceeds the threshold
    for a tracked dimension, an anomaly is reported.

    Args:
        sat_id     (int)  : satellite index for logging
        window_size(int)  : number of past steps in rolling window. Default 50.
        z_threshold(float): z-score trigger level. Default 3.0 (3-sigma rule).
    """

    # Dimensions to monitor and associated anomaly types
    MONITORED_DIMS: Dict[int, Tuple[str, AnomalyType]] = {
        IDX_SOC:  ("SoC",         AnomalyType.BATTERY_OVERDISCHARGE),
        IDX_SOH:  ("SoH",         AnomalyType.SOH_RAPID_DEGRADATION),
        IDX_TEMP: ("Temperature",  AnomalyType.THERMAL_RUNAWAY),
        IDX_PSOL: ("Solar Power",  AnomalyType.SOLAR_PANEL_FAILURE),
        IDX_COMM: ("Comm Delay",   AnomalyType.COMMUNICATION_FAULT),
    }

    def __init__(
        self,
        sat_id:      int   = 0,
        window_size: int   = 50,
        z_threshold: float = 3.0,
    ) -> None:
        self.sat_id      = sat_id
        self.window_size = window_size
        self.z_threshold = z_threshold
        self._window: deque = deque(maxlen=window_size)
        self._step   = 0

    def update(self, obs: np.ndarray) -> Optional[AnomalyReport]:
        """
        Update the rolling window with a new observation and check for anomalies.

        Args:
            obs: normalized observation vector, shape (OBS_DIM,)

        Returns:
            AnomalyReport if anomaly detected, else None
        """
        self._step += 1
        self._window.append(obs.copy())

        # Need at least 10 samples for reliable statistics
        if len(self._window) < 10:
            return None

        history = np.stack(list(self._window), axis=0)   # (T, OBS_DIM)
        mean    = history.mean(axis=0)
        std     = history.std(axis=0) + 1e-8
        z_scores = (obs - mean) / std

        # Check each monitored dimension
        worst_severity = 0.0
        worst_type     = AnomalyType.NONE
        worst_desc     = ""

        for dim, (name, atype) in self.MONITORED_DIMS.items():
            z = abs(z_scores[dim])
            if z > self.z_threshold:
                severity = min(float(z / (self.z_threshold * 2)), 1.0)
                if severity > worst_severity:
                    worst_severity = severity
                    worst_type     = atype
                    worst_desc     = (
                        f"{name} z-score={z_scores[dim]:.2f} "
                        f"(mean={mean[dim]:.3f}, obs={obs[dim]:.3f})"
                    )

        if worst_type != AnomalyType.NONE:
            report = AnomalyReport(
                sat_id=self.sat_id,
                step=self._step,
                anomaly_type=worst_type,
                severity=worst_severity,
                z_scores=z_scores.copy(),
                description=worst_desc,
            )
            logger.warning(f"[AnomalyDetector] {report}")
            return report

        return None

    def reset(self) -> None:
        self._window.clear()
        self._step = 0


class ResidualDetector:
    """
    World-model residual anomaly detector.

    Compares world model predictions with actual observations.
    Large residuals indicate unexpected physics — typically hardware failure.

    Args:
        sat_id              (int)  : satellite index
        residual_threshold  (float): per-dim RMSE threshold. Default 0.1.
        window_size         (int)  : rolling window for baseline residual. Default 30.
    """

    def __init__(
        self,
        sat_id:            int   = 0,
        residual_threshold: float = 0.1,
        window_size:        int   = 30,
    ) -> None:
        self.sat_id             = sat_id
        self.residual_threshold = residual_threshold
        self._window: deque     = deque(maxlen=window_size)
        self._step  = 0
        self._last_pred: Optional[np.ndarray] = None

    def set_prediction(self, pred_state: np.ndarray) -> None:
        """Store the world model's prediction for the next step."""
        self._last_pred = pred_state.copy()

    def update(
        self,
        actual_obs: np.ndarray,
    ) -> Optional[AnomalyReport]:
        """
        Compute residual between predicted and actual state.

        Args:
            actual_obs: the actual observation from the environment

        Returns:
            AnomalyReport if residual exceeds threshold, else None
        """
        self._step += 1

        if self._last_pred is None:
            return None

        residuals = actual_obs - self._last_pred
        rmse      = float(np.sqrt(np.mean(residuals ** 2)))
        self._window.append(rmse)

        # Compute baseline RMSE from window
        if len(self._window) < 5:
            return None

        baseline = np.mean(list(self._window)[:-1])
        current  = rmse

        # Significant spike above baseline
        if current > self.residual_threshold and current > 2.0 * baseline + 1e-6:
            # Identify which dimension has the largest residual
            max_dim = int(np.argmax(np.abs(residuals)))
            dim_names = ["SoC", "SoH", "Temp", "PSolar", "Phase", "Eclipse", "PCons", "Comm"]

            severity = min(float(current / (self.residual_threshold * 3)), 1.0)
            report   = AnomalyReport(
                sat_id=self.sat_id,
                step=self._step,
                anomaly_type=AnomalyType.UNKNOWN_RESIDUAL,
                severity=severity,
                z_scores=np.zeros(OBS_DIM),
                residuals=residuals.copy(),
                description=(
                    f"World model residual RMSE={current:.4f} "
                    f"(baseline={baseline:.4f}) | "
                    f"max residual dim: {dim_names[max_dim]}"
                ),
            )
            logger.warning(f"[ResidualDetector] {report}")
            return report

        return None

    def reset(self) -> None:
        self._window.clear()
        self._last_pred = None
        self._step = 0


class AnomalyDetector:
    """
    Combined anomaly detector for one satellite agent.

    Wraps both the StatisticalDetector (Z-score) and ResidualDetector
    (world model prediction error) and fuses their outputs.

    Args:
        sat_id             (int)  : satellite index
        window_size        (int)  : Z-score rolling window. Default 50.
        z_threshold        (float): Z-score trigger. Default 3.0.
        residual_threshold (float): RMSE trigger. Default 0.1.
        enable_residual    (bool) : use world-model residual detection. Default True.
    """

    def __init__(
        self,
        sat_id:             int   = 0,
        window_size:        int   = 50,
        z_threshold:        float = 3.0,
        residual_threshold: float = 0.1,
        enable_residual:    bool  = True,
    ) -> None:
        self.sat_id          = sat_id
        self.enable_residual = enable_residual
        self.stat_detector   = StatisticalDetector(sat_id, window_size, z_threshold)
        self.res_detector    = ResidualDetector(sat_id, residual_threshold)
        self._anomaly_history: List[AnomalyReport] = []

    def step(
        self,
        obs:      np.ndarray,
        pred_obs: Optional[np.ndarray] = None,
    ) -> List[AnomalyReport]:
        """
        Process one observation and return all detected anomaly reports.

        Args:
            obs     : current observation from env, shape (OBS_DIM,)
            pred_obs: world model prediction for this step (optional)

        Returns:
            list of AnomalyReport (may be empty)
        """
        reports: List[AnomalyReport] = []

        # Statistical (Z-score) detection
        stat_report = self.stat_detector.update(obs)
        if stat_report:
            reports.append(stat_report)

        # World-model residual detection
        if self.enable_residual:
            res_report = self.res_detector.update(obs)
            if res_report:
                reports.append(res_report)
            # Update prediction for next step
            if pred_obs is not None:
                self.res_detector.set_prediction(pred_obs)

        self._anomaly_history.extend(reports)
        return reports

    def reset(self) -> None:
        """Reset detectors (call at episode start)."""
        self.stat_detector.reset()
        self.res_detector.reset()
        self._anomaly_history.clear()

    @property
    def anomaly_count(self) -> int:
        return len(self._anomaly_history)

    @property
    def recent_anomalies(self) -> List[AnomalyReport]:
        """Return last 10 anomaly reports."""
        return self._anomaly_history[-10:]

    def anomaly_rate(self, last_n: int = 100) -> float:
        """Fraction of last_n steps that triggered an anomaly."""
        recent = self._anomaly_history[-last_n:]
        return len(recent) / max(last_n, 1)
