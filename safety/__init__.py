"""
Layer 5 — safety package (Safety & Fault Recovery Layer).

Exports:
    AnomalyType          — enum of detectable anomaly categories
    AnomalyReport        — anomaly event dataclass
    StatisticalDetector  — sliding-window Z-score anomaly detector
    ResidualDetector     — world-model prediction residual detector
    AnomalyDetector      — combined detector (Z-score + residual)
    SafetyThresholds     — configurable constraint thresholds
    SafetyState          — FSM state enum (NOMINAL/WARNING/CRITICAL/RECOVERY/DEGRADED)
    SafetyEvent          — FSM transition event dataclass
    SafetyMonitor        — 5-state FSM safety enforcer + hard action override
    RecoveryPhase        — enum of phases within a recovery sequence
    RecoveryResult       — per-step recovery output dataclass
    BatteryRecovery      — 3-phase battery discharge recovery
    ThermalRecovery      — 3-phase thermal overrun recovery
    GeneralRecovery      — catch-all fallback recovery sequence
    RecoveryPolicy       — top-level recovery orchestrator
"""

from safety.anomaly_detector import (
    AnomalyType,
    AnomalyReport,
    StatisticalDetector,
    ResidualDetector,
    AnomalyDetector,
)
from safety.safety_monitor import (
    SafetyThresholds,
    SafetyState,
    SafetyEvent,
    SafetyMonitor,
)
from safety.recovery_policy import (
    RecoveryPhase,
    RecoveryResult,
    BatteryRecovery,
    ThermalRecovery,
    GeneralRecovery,
    RecoveryPolicy,
)

__all__ = [
    # Anomaly detection
    "AnomalyType",
    "AnomalyReport",
    "StatisticalDetector",
    "ResidualDetector",
    "AnomalyDetector",
    # Safety monitor
    "SafetyThresholds",
    "SafetyState",
    "SafetyEvent",
    "SafetyMonitor",
    # Recovery policy
    "RecoveryPhase",
    "RecoveryResult",
    "BatteryRecovery",
    "ThermalRecovery",
    "GeneralRecovery",
    "RecoveryPolicy",
]
